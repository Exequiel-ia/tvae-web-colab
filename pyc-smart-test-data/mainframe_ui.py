"""Additional Streamlit workspace. Never writes the relational workspace keys."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mainframe_codec import CODECS, FormatError, load_mask, parse_records, raw_input, validate_schema
from mainframe_model import (MAX_RUNS, STATE_KEY, TREATMENTS, default_policy, demo_bundle,
                             export_zip, generate, header_defaults, new_state,
                             replace_source, train_model, validate_policy, view_by_name)

HELP = {
    'Origen': 'Carga la mascara y un extracto fijo de prueba por separado. El Excel aporta COPY/CLUSTER; su hoja DATA no se usa como archivo binario. Primero se valida lectura y reconstruccion byte a byte.',
    'Diccionario y privacidad': 'Selecciona que campos de negocio aprender. Claves, cabecera, rellenos y fechas se reconstruyen por reglas. Confidencial no significa sintetizable: en esta demo los campos confidenciales se generan ficticios, no se entrenan.',
    'Entrenamiento': 'TVAE aprende solo los campos aprobados del detalle. No aprende la cabecera, la clave ni el formato COMP-3. Se conserva un modelo mainframe por sesion, independiente de la demo relacional.',
    'Generacion': 'Genera detalles desde el modelo ya entrenado. Se crean claves nuevas, se reconstruye la cabecera y se codifican campos posicionales. Solo se entrega el resultado si puede volver a leerse con la misma mascara.',
    'Comparacion y exportacion': 'Compara el original con cada corrida de archivos. Las barras muestran porcentajes lado a lado. La validacion fisica no equivale a certificacion de privacidad ni a validacion funcional en COBOL.'
}


def _error(exc):
    # Third-party exceptions can embed source values: do not show/log their text.
    if isinstance(exc, FormatError):
        st.error(str(exc))
    else:
        logging.getLogger(__name__).error('Mainframe operation failed (%s)', type(exc).__name__)
        st.error('No se completo la operacion. No se modifico la demo relacional. Tipo de error: ' + type(exc).__name__)


def _preview(rows, policy=None, original=True):
    table = pd.DataFrame(rows).head(30)
    table = table.drop(columns=[c for c in table if c.startswith('FILLER@')], errors='ignore')
    confidential = {p['Campo'] for p in policy or [] if p['Confidencial']}
    for c in table:
        if original and (policy is None or c in confidential):
            table[c] = '[oculto]'
        else:
            table[c] = table[c].map(lambda v: str(v)[:120])
    return table


def _origin(state):
    st.subheader('1. Mascara y extracto de origen')
    st.warning('Demo publica: solo datos ficticios o previamente autorizados y desensibilizados. No subas cuentas, identificaciones ni movimientos reales. Retira DATA del Excel antes de cargar una mascara que incluya muestras reales.')
    occupied = state['source'] is not None or state['model'] is not None or bool(state['runs'])
    confirmed = (st.checkbox('Confirmo reemplazar SOLO el espacio mainframe (se reinician sus modelos y corridas).', key='mf_replace')
                 if occupied else True)
    with st.expander('Probar sin subir archivos', expanded=state['schema'] is None):
        st.write('Ejemplo inventado de 600 bytes, EBCDIC cp037, cabecera y detalle, con importes COMP-3. No contiene la mascara ni los registros de ningun cliente.')
        if st.button('Preparar ejemplo de archivos', disabled=not confirmed, key='mf_demo'):
            schema, binary, config, _ = demo_bundle()
            source = parse_records(binary, schema, **config)
            st.session_state[STATE_KEY] = replace_source(state, schema, source, config)
            st.rerun()
        schema, binary, _, copy = demo_bundle()
        st.download_button('Copybook ficticio de ejemplo', copy, 'demo_archivos.cpy', key='mf_example_mask')
        st.download_button('Extracto ficticio de ejemplo', binary, 'demo_archivos.bin', key='mf_example_binary')
        st.download_button('Layout JSON de ejemplo', json.dumps(schema, indent=2), 'demo_layout.json', key='mf_example_json')
    mask = st.file_uploader('Mascara: Excel con hoja COPY, copybook UTF-8 o layout JSON', type=['xlsx', 'cpy', 'cob', 'txt', 'json'], key='mf_mask_upload')
    if st.button('Leer mascara', disabled=mask is None or not confirmed, key='mf_read_mask'):
        try:
            schema = load_mask(mask.getvalue(), mask.name)
            if schema['record_length'] != 600:
                raise FormatError('Esta primera version de la interfaz admite registros de 600 bytes.')
            st.session_state[STATE_KEY] = {**new_state(), 'schema': schema, 'revision': state['revision'] + 1}
            st.rerun()
        except Exception as exc:
            _error(exc)
    if state['schema'] is None:
        return
    schema = state['schema']
    views = [v['name'] for v in schema['views']]
    revision = str(state['revision'])
    st.success(f"Mascara interpretada: {schema['record_length']} bytes por registro; {len(views)} vista(s).")
    if schema.get('ignored_redefines'):
        st.info('REDEFINES internos tratados como vistas alternativas, no como columnas independientes: ' + ', '.join(schema['ignored_redefines']))
    st.caption('Las posiciones se calculan desde COPY. La hoja POSICIONES no se reconcilia automaticamente. Condiciones 88 no se convierten automaticamente en dominios de negocio.')
    chosen_view = st.selectbox('Vista de la mascara', views, key='mf_schema_view_' + revision)
    view = view_by_name(schema, chosen_view)
    st.dataframe(pd.DataFrame([{'Campo': f['name'], 'Inicio (1-based)': f['start'] + 1,
                               'Fin': f['start'] + f['length'], 'Bytes': f['length'],
                               'PIC': f['picture'], 'USAGE': f['usage'] or 'DISPLAY'}
                              for f in view['fields']]), hide_index=True, use_container_width=True)
    st.download_button('Guardar mascara normalizada', json.dumps(schema, ensure_ascii=False, indent=2),
                       'layout_mainframe.json', key='mf_save_layout')
    st.markdown('#### Configurar lectura del extracto')
    default_detail = next((i for i, n in enumerate(views) if 'DET' in n.upper()), len(views) - 1)
    detail = st.selectbox('Vista de detalle', views, index=default_detail, key='mf_detail_' + revision)
    headers = ['Sin cabecera'] + [v for v in views if v != detail]
    header = st.selectbox('Vista de cabecera', headers, index=1 if len(headers) > 1 else 0, key='mf_header_' + revision)
    codec = st.selectbox('Codificacion de campos de texto y DISPLAY', CODECS, key='mf_codec_' + revision,
                         help='No se detecta automaticamente. COMP-3 y binarios no se traducen a texto.')
    c1, c2 = st.columns(2)
    dm = c1.text_input('Tipo de detalle (un caracter)', 'D', max_chars=1, key='mf_dm_' + revision)
    hm = c2.text_input('Tipo de cabecera (un caracter)', '0', max_chars=1, key='mf_hm_' + revision)
    k1, k2 = st.columns(2)
    kl = k1.number_input('Longitud de clave (bytes; 0 = sin clave)', 0, 599, int(schema['key_length']), key='mf_kl_' + revision)
    ko = k2.number_input('Desplazamiento de clave (base 0)', 0, 599, int(schema['key_offset']), key='mf_ko_' + revision)
    st.caption('Para KEYS(20 1), la clave ocupa posiciones 2 a 21. El adaptador recibe registros logicos concatenados, no un volcado de control intervals VSAM.')
    consent = st.checkbox('Confirmo que el extracto es de prueba y no contiene informacion real sensible.', key='mf_source_consent')
    extract = st.file_uploader('Extracto binario o HEX puro (sin RDW/BDW, offsets ni saltos de registro)',
                               type=['bin', 'dat', 'raw', 'hex'], disabled=not consent, key='mf_extract_upload')
    is_hex = st.checkbox('Interpretar el extracto como texto hexadecimal', value=False, key='mf_is_hex')
    if st.button('Validar y cargar extracto', disabled=extract is None or not consent or not confirmed, key='mf_load_source'):
        try:
            candidate = {**schema, 'key_length': int(kl), 'key_offset': int(ko)}
            validate_schema(candidate)
            config = dict(codec=codec, detail_name=detail, header_name=None if header == 'Sin cabecera' else header,
                          detail_marker=dm, header_marker=hm)
            source = parse_records(raw_input(extract.getvalue(), is_hex), candidate, **config)
            st.session_state[STATE_KEY] = replace_source(state, candidate, source, config)
            st.rerun()
        except Exception as exc:
            _error(exc)
    if state['source'] is not None:
        source = state['source']
        st.success(f"Origen activo: {len(source['details'])} detalles y {len(source['headers'])} cabecera(s). Reconstruccion sin cambios: byte a byte correcta.")
        st.caption('Los selectores de lectura no alteran el origen activo hasta pulsar Validar y cargar. Los valores se ocultan por defecto.')
        show = st.checkbox('Mostrar la vista decodificada (solo datos ficticios)', key='mf_show_source')
        st.dataframe(_preview(source['details'], original=not show), hide_index=True, use_container_width=True)
        st.info('Siguiente: Diccionario y privacidad. Cargar el origen no entrena ni genera datos sinteticos.')


def _dictionary(state):
    if state['source'] is None:
        st.info('Carga y valida el origen primero.')
        return
    view = view_by_name(state['schema'], state['config']['detail_name'])
    st.subheader('2. Clasificacion del detalle')
    st.caption('Las posiciones, el control, los rellenos, las claves y las fechas reconocidas son estructurales. No se pueden convertir en variables aprendidas. Los textos se proponen ficticios/confidenciales por defecto: revisa los codigos de negocio.')
    rows = state['policy'] or default_policy(state['schema'], view)
    edited = st.data_editor(pd.DataFrame(rows), num_rows='fixed', hide_index=True, use_container_width=True,
                            disabled=['Campo', 'Inicio', 'Bytes', 'Tipo fisico'],
                            column_config={'Tratamiento': st.column_config.SelectboxColumn(options=TREATMENTS),
                                           'Confidencial': st.column_config.CheckboxColumn(),
                                           'Constante': st.column_config.TextColumn()},
                            key='mf_policy_editor_' + str(state['revision']))
    if st.button('Aprobar politica de archivos', key='mf_approve'):
        try:
            policy = edited.to_dict('records')
            validate_policy(state['schema'], view, policy, state['config']['codec'])
            if state['policy'] != policy:
                state['model'] = None
            state['policy'] = policy
            st.success('Politica aprobada. Se conserva el historial; un cambio exige entrenar un modelo nuevo.')
        except Exception as exc:
            _error(exc)
    st.warning('Validez de formato no garantiza reglas del negocio. Cuentas ficticias no incluyen validadores bancarios; fechas se generan dentro de rangos por grupo, sin reglas entre grupos. Confirma la cabecera antes de cargar en z/OS.')


def _training(state):
    if state['policy'] is None:
        st.info('Primero valida el origen y aprueba su politica.')
        return
    st.subheader('3. Entrenar un modelo para archivos')
    epochs = st.slider('Epocas TVAE', 5, 200, 10, 5, key='mf_epochs')
    seed = st.number_input('Semilla del muestreo', 0, 999999, 42, key='mf_train_seed')
    st.caption('CPU; batch 64; embedding 32; capas 64 x 64. Maximo 2000 filas de entrenamiento, 30 campos y 100 categorias por campo. La semilla controla el muestreo; no garantiza entrenamiento TVAE identico.')
    if state['model'] is not None:
        st.info('Entrenar reemplaza solamente el modelo mainframe activo. Sus corridas anteriores y toda la demo relacional se conservan.')
    if st.button('Entrenar modelo de archivos', key='mf_train'):
        try:
            with st.spinner('Entrenando solo los campos de negocio aprobados...'):
                model = train_model(state['schema'], state['source'], state['config'], state['policy'], epochs, seed)
            state['model'] = model
            st.success('Modelo de archivos preparado: ' + model['algorithm'])
        except Exception as exc:
            _error(exc)
    if state['model'] is not None:
        m = state['model']
        st.write({'Motor': m['algorithm'], 'Filas de entrenamiento': m['training_rows'], 'Campos aprendidos': m['learned']})
    st.caption('No se guardan modelos ni datos en disco. No se incluye almacenamiento permanente ni entrenamiento relacional multitabla.')


def _generation(state):
    st.subheader('4. Generar un archivo sintetico')
    if state['model'] is None:
        st.info('Entrena primero el modelo de archivos.')
        return
    quantity = st.number_input('Cantidad de detalles sinteticos', 1, 2000, 200, key='mf_quantity')
    seed = st.number_input('Semilla de reglas e identificadores', 0, 999999, 42, key='mf_generation_seed')
    header_values = None
    approved = True
    if state['config']['header_name']:
        st.markdown('#### Cabecera nueva: revisar antes de generar')
        d = st.date_input('Fecha de proceso para la cabecera', date(2026, 1, 1), key='mf_process_date')
        view = view_by_name(state['schema'], state['config']['header_name'])
        header_values = header_defaults(view, state['config']['header_marker'], d)
        fields = {f['name']: f for f in view['fields']}
        editor = st.data_editor(pd.DataFrame([{'Campo': f['name'], 'Valor': str(header_values[f['name']])}
                                              for f in view['fields'] if not f['filler'] and f['start'] != 0]),
                                disabled=['Campo'], hide_index=True, num_rows='fixed', use_container_width=True,
                                key='mf_header_values_' + str(state['revision']) + str(d))
        for r in editor.to_dict('records'):
            header_values[r['Campo']] = r['Valor']
        approved = st.checkbox('He revisado los valores de la cabecera y las reglas de este archivo de prueba.', key='mf_approve_header')
        st.caption('No se copia la cabecera original. El tipo de registro y los rellenos se reconstruyen automaticamente.')
    room = len(state['runs']) < MAX_RUNS
    if not room:
        st.warning('Limite de tres corridas mainframe en memoria. Exporta y elimina una en Comparacion para continuar.')
    if st.button('Generar y validar archivo', disabled=not approved or not room, key='mf_generate'):
        try:
            with st.spinner('Generando, codificando y validando la salida...'):
                run = generate(state['schema'], state['source'], state['config'], state['model'],
                               int(quantity), int(seed), header_values)
            run['generated_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
            run['name'] = 'Archivo ' + str(state.get('counter', 0) + 1)
            state['counter'] = state.get('counter', 0) + 1
            state['runs'].append(run)
            st.success(f"{run['name']}: {run['quantity']} detalles, {run['total_records']} registros, {run['size_bytes']:,} bytes. Salida releida correctamente.")
        except Exception as exc:
            _error(exc)
    st.caption('Los registros se ordenan por los bytes de la clave declarada. El ZIP contiene un secuencial binario, no un cluster VSAM ni un despliegue en z/OS.')


def _comparison(state):
    st.subheader('5. Comparacion y exportacion de archivos')
    if not state['runs']:
        st.info('Genera una corrida de archivos primero.')
        return
    st.dataframe(pd.DataFrame([{'Corrida': r['name'], 'Detalles': r['quantity'], 'Epocas': r['epochs'],
                               'Motor': r['algorithm'], 'Fecha UTC': r['generated_at']}
                              for r in state['runs']]), hide_index=True, use_container_width=True)
    index = st.selectbox('Corrida mainframe', range(len(state['runs'])),
                         format_func=lambda i: state['runs'][i]['name'], key='mf_run_index')
    run = state['runs'][index]
    a, b, c = st.columns(3)
    a.metric('Detalles sinteticos', run['quantity'])
    b.metric('Bytes por registro', state['schema']['record_length'])
    c.metric('Lectura fisica de salida', 'Correcta' if run['roundtrip'] else 'Error')
    st.caption('Controles: longitud, tipos fisicos, claves unicas (si se declaran), dominios aprendidos y reconstruccion. No son las MET1-MET7 ni una certificacion de privacidad.')
    col = st.selectbox('Variable de negocio a comparar', run['learned'], key='mf_compare_field_' + run['name'])
    view = view_by_name(state['schema'], state['config']['detail_name'])
    f = next(f for f in view['fields'] if f['name'] == col)
    real = pd.Series([r[col] for r in state['source']['details']])
    synth = pd.Series([r[col] for r in run['details']])
    if f['kind'] != 'text':
        original, synthetic = real.astype(float), synth.astype(float)
        bins = np.histogram_bin_edges(np.concatenate([original, synthetic]), bins=10)
        labels = [f'{bins[i]:.2f} - {bins[i + 1]:.2f}' for i in range(len(bins) - 1)]
        counts = [(label, 'Original', n * 100 / len(original)) for label, n in zip(labels, np.histogram(original, bins)[0])]
        counts += [(label, 'Sintetico', n * 100 / len(synthetic)) for label, n in zip(labels, np.histogram(synthetic, bins)[0])]
        st.write({'Promedio original': round(original.mean(), 4), 'Promedio sintetico': round(synthetic.mean(), 4)})
    else:
        counts = []
        for label, series in [('Original', real), ('Sintetico', synth)]:
            counts += [(str(k), label, float(v) * 100) for k, v in series.value_counts(normalize=True).items()]
    chart = pd.DataFrame(counts, columns=['Valor o intervalo', 'Dataset', 'Porcentaje'])
    fig = px.bar(chart, x='Valor o intervalo', y='Porcentaje', color='Dataset', barmode='group',
                 title=f'{col}: original vs sintetico',
                 color_discrete_map={'Original': '#172658', 'Sintetico': '#56b4a9'})
    fig.update_layout(template='plotly_white', legend_title_text='', yaxis_ticksuffix='%')
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Barras separadas y porcentajes por dataset; mismos intervalos para ambos. No hay correspondencia registro a registro ni un tercer grupo.')
    left, right = st.columns(2)
    with left:
        st.markdown('**Original: campos confidenciales ocultos**')
        st.dataframe(_preview(state['source']['details'], run['policy']), use_container_width=True, hide_index=True)
    with right:
        st.markdown('**Sintetico: primeros 30 registros**')
        st.dataframe(_preview(run['details'], run['policy'], original=False), use_container_width=True, hide_index=True)
    st.download_button('Descargar experimento de archivos (ZIP)', export_zip(run, state['schema'], state['config']),
                       'pyc_archivo_sintetico.zip', mime='application/zip', key='mf_export_zip')
    st.warning('Valida la salida en un ambiente mainframe de pruebas antes de usarla en una aplicacion. No se ha certificado carga en z/OS, reglas bancarias, anonimato ni relaciones con un segundo archivo.')
    remove = st.checkbox('Confirmo eliminar esta corrida mainframe de la sesion.', key='mf_delete_confirm_' + run['name'])
    if st.button('Eliminar corrida seleccionada', disabled=not remove, key='mf_delete_run'):
        del state['runs'][index]
        st.session_state.pop('mf_run_index', None)
        st.rerun()


def render():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = new_state()
    state = st.session_state[STATE_KEY]
    st.markdown('''<style>
    .block-container{max-width:1400px;padding-top:1.5rem}
    [data-testid="stSidebar"]{background:#172658}
    [data-testid="stSidebar"] *{color:white!important}
    </style>''', unsafe_allow_html=True)
    st.title('PyC Smart Test Data Lab | Archivos mainframe')
    st.caption('Extension aditiva v1 - fijo de 600 bytes - espacio separado de la demo relacional')
    st.sidebar.page_link('streamlit_app.py', label='Volver a la demo relacional')
    step = st.sidebar.radio('Etapas de archivos', list(HELP), key='mf_step')
    with st.expander('Exe-IA | Guia de esta pantalla', expanded=True):
        st.info(HELP[step])
    st.caption('Solo estado de sesion. No se envian archivos a GitHub ni a un LLM. Cerrar/reiniciar la sesion puede perder modelos y corridas.')
    actions = {'Origen': _origin, 'Diccionario y privacidad': _dictionary,
               'Entrenamiento': _training, 'Generacion': _generation,
               'Comparacion y exportacion': _comparison}
    try:
        actions[step](state)
    except Exception as exc:
        _error(exc)
