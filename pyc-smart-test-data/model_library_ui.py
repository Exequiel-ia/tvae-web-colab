"""Explicit model save/restore/delete UI; mainframe is never touched."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from model_library import (LibraryError, MAX_BYTES, STATE_KEY, ModelLibrary, activate,
                           capture_model, new_code, remove_active, valid_code)


def _setting(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def _error(exc):
    if isinstance(exc, LibraryError):
        st.error(str(exc))
    else:
        # Do not expose exception text, paths, source data, or secrets.
        st.error('La operación no se completó (' + type(exc).__name__ + '). No se eliminó la biblioteca.')


def _notify(message):
    st.session_state['ml_feedback'] = message
    st.rerun()


def render_library():
    st.markdown('''<style>
    [data-testid="stSidebar"]{background:#172658}
    [data-testid="stSidebar"] *{color:white!important}
    .block-container{max-width:1250px;padding-top:2rem}
    .ml-hero{background:linear-gradient(115deg,#172658,#3458a4);color:white;padding:24px;border-radius:18px;margin-bottom:20px}
    .ml-hero h1{color:white;font-size:1.9rem;margin:0;padding:0}
    .ml-hero p{margin:8px 0 0;color:#e3ebff}
    </style><div class="ml-hero"><h1>Biblioteca de modelos</h1>
    <p>Entrena una vez · guarda el modelo · recupéralo para tu demo</p></div>''', unsafe_allow_html=True)
    st.sidebar.page_link('streamlit_app.py', label='Volver a la demo relacional')
    st.sidebar.caption('Modelos de Clientes. No modifica el módulo de archivos mainframe.')
    st.write('Después de entrenar, guarda aquí el modelo completo. Recuperarlo no ejecuta el entrenamiento. '
             'Se conserva también su base original y diccionario; las generaciones se exportan por separado.')
    st.warning('Almacenamiento local de la instancia: sobrevive a cambios de pantalla y sesiones nuevas mientras el servidor conserve sus archivos. '
               'Streamlit Cloud puede perderlos al recrear la instancia. Para una demo importante, descarga el respaldo firmado antes de salir.')
    st.caption('Documentación de Streamlit: Community Cloud no garantiza persistencia de archivos locales. '
               'No es almacenamiento permanente externo ni recuperación de un entrenamiento interrumpido.')
    root = _setting('PYC_MODEL_LIBRARY_DIR') or str(Path.home()/'.pyc-model-library')
    try:
        store = ModelLibrary(Path(root), _setting('PYC_MODEL_SIGNING_KEY'))
    except Exception as exc:
        _error(exc)
        return
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = new_code()
    code = st.session_state[STATE_KEY]
    if st.session_state.get('ml_feedback'):
        st.success(st.session_state.pop('ml_feedback'))
    with st.expander('Mi biblioteca: código de acceso para volver a abrirla', expanded=True):
        st.info('Guarda tu código privado. Al volver en otra sesión, úsalo para abrir esta misma biblioteca. '
                'No es tu contraseña de GitHub y no garantiza la conservación del disco de Streamlit.')
        st.download_button('Descargar mi código de biblioteca', code+'\n', 'codigo_biblioteca_PyC.txt',
                           mime='text/plain', key='ml_download_access')
        if st.checkbox('Mostrar mi código privado', key='ml_show_access'):
            st.code(code, language=None)
        entered = st.text_input('Abrir otra biblioteca: pega su código', type='password', key='ml_enter_code')
        if st.button('Abrir biblioteca', key='ml_open_code'):
            try:
                candidate = valid_code(entered.strip())
                found = store.list(candidate)
                if not found:
                    st.warning('No se encontraron modelos para ese código en esta instancia. '
                               'Revisa el código o importa tu respaldo descargado si el servidor fue recreado.')
                else:
                    st.session_state[STATE_KEY] = candidate
                    st.session_state.pop('ml_restore_upload', None)
                    _notify('Biblioteca abierta. Los modelos no se activan hasta que elijas Recuperar.')
            except Exception as exc:
                _error(exc)
    if store.stable:
        st.success('Respaldo portable habilitado: el servidor tiene una clave estable en Secrets.')
    else:
        st.warning('Puedes guardar y recuperar en esta instancia. Para descargar/importar respaldos que sobrevivan a su recreación, '
                   'el administrador debe configurar una vez PYC_MODEL_SIGNING_KEY en Secrets.')
        with st.expander('Configuración administrativa del respaldo seguro'):
            st.write('Genera una clave aleatoria de 32 bytes en tu equipo, conserva una copia privada y agrégala a los Secrets de Streamlit. '
                     'No uses contraseñas cortas ni publiques esta clave en GitHub. Cambiarla invalida respaldos previos.')
            st.code('python -c "import secrets; print(secrets.token_hex(32))"', language='bash')
            st.code('PYC_MODEL_SIGNING_KEY = "PEGA_AQUI_LA_CLAVE_HEXADECIMAL_DE_64_CARACTERES"', language='toml')
            st.caption('Manage app → Settings → Secrets. No se pega en esta pantalla. No se aceptan modelos PKL arbitrarios.')
    save, saved, restore = st.tabs(['1. Guardar entrenamientos', '2. Recuperar o eliminar', '3. Importar respaldo'])
    with save:
        models = st.session_state.get('models', [])
        if not models:
            st.info('No hay modelos activos. Entrena uno en la demo relacional o recupera uno ya guardado en la segunda pestaña.')
        else:
            rows = [{'Modelo': m.get('name','Sin nombre'), 'Épocas':m.get('params',{}).get('épocas'),
                     'Entrenamiento (s)':m.get('train_sec'), 'Guardado desde esta sesión':bool(m.get('library_id'))}
                    for m in models]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption('Guardar conserva pesos, transformaciones, parámetros, origen y diccionario. '
                       'No vuelve a entrenar ni incluye las corridas sintéticas. No se escribe en GitHub.')
            picked = st.multiselect('Modelos que deseas guardar', list(range(len(models))),
                                    default=list(range(len(models))),
                                    format_func=lambda i:f"{i+1} · {models[i].get('name','Modelo')}", key='ml_pick_save')
            confirm = st.checkbox('Confirmo que la base y el diccionario activos son los usados al entrenar estos modelos, '
                                  'y que contienen solo datos ficticios de la demo.', key='ml_confirm_source')
            st.caption('Los entrenamientos anteriores no registran una huella de su origen. La confirmación se conserva como declaración del usuario, no como verificación histórica.')
            if st.button('Guardar modelos seleccionados', type='primary', disabled=not picked or not confirm, key='ml_save'):
                count, errors = 0, []
                for i in picked:
                    try:
                        payload = capture_model(models[i], st.session_state.get('tables',{}),
                                                st.session_state.get('dictionary'), confirm_legacy=True)
                        identifier = store.save(code, payload)
                        models[i]['library_id'] = identifier
                        count += 1
                    except Exception as exc:
                        errors.append(exc)
                if count:
                    st.success(f'{count} modelo(s) guardado(s). Descarga su respaldo en Recuperar o eliminar.')
                for exc in errors:
                    _error(exc)
    with saved:
        try:
            entries = store.list(code)
        except Exception as exc:
            _error(exc)
            entries = []
        if not entries:
            st.info('Esta biblioteca todavía no tiene modelos guardados.')
        else:
            by_id = {e['id']:e for e in entries}
            selected = st.selectbox('Modelo guardado', list(by_id),
                                    format_func=lambda i: f"{by_id[i]['name']} · {by_id[i].get('epochs','—')} épocas · {by_id[i]['captured_at']}",
                                    key='ml_selected_saved')
            item = by_id[selected]
            st.write({'Modelo':item['name'], 'Guardado (UTC)':item['captured_at'],
                      'Épocas':item.get('epochs'), 'Tamaño (MiB)':round(item['bytes']/1024**2,2)})
            st.caption('Recuperar agrega el modelo cuando el origen y el diccionario coinciden. '
                       'Si son distintos, se requiere confirmar el reemplazo del contexto activo; la biblioteca y mainframe se conservan.')
            replacement = st.checkbox('Si el origen o diccionario son distintos, autorizo restaurar los del respaldo. '
                                      'Se sustituyen los modelos activos y se vacían sus generaciones actuales, pero no la biblioteca.',
                                      key='ml_allow_replace_'+selected)
            if st.button('Recuperar modelo para generar sin entrenar', type='primary', key='ml_activate'):
                try:
                    payload = store.load(code, selected)
                    activate(st.session_state, payload, replace_reference=replacement)
                    _notify('Modelo recuperado sin entrenamiento. Vuelve a la demo relacional → Generación y selecciónalo.')
                except Exception as exc:
                    _error(exc)
            if store.stable:
                if st.button('Preparar respaldo descargable', key='ml_prepare_backup'):
                    try:
                        data = store.backup(code, selected)
                        st.session_state['ml_backup_data'] = (code, selected, data)
                    except Exception as exc:
                        _error(exc)
                ready = st.session_state.get('ml_backup_data')
                if ready and ready[:2] == (code, selected):
                    st.download_button('Descargar respaldo del modelo', ready[2], f'modelo_{selected[:12]}.pycmodel',
                                       mime='application/octet-stream', key='ml_download_model')
                    st.caption('El archivo contiene el modelo y su referencia, no solo parámetros. Está firmado, no cifrado. '
                               'Consérvalo privado. Necesita la misma clave del servidor y versiones de librerías para restaurarlo.')
            with st.expander('Eliminar este modelo'):
                confirm_delete = st.checkbox('Confirmo eliminar este modelo de la biblioteca y de la sesión activa. '
                                             'Sus generaciones ya creadas se conservan.', key='ml_confirm_delete_'+selected)
                if st.button('Eliminar definitivamente de la biblioteca', disabled=not confirm_delete, key='ml_delete'):
                    try:
                        store.delete(code, selected)
                        remove_active(st.session_state, selected)
                        st.session_state.pop('ml_backup_data',None)
                        _notify('Modelo eliminado de la biblioteca y de los modelos activos de esta sesión. '
                                'No se borraron generaciones. Las copias descargadas y otras sesiones no se pueden borrar desde aquí.')
                    except Exception as exc:
                        _error(exc)
    with restore:
        st.write('Carga un respaldo .pycmodel descargado desde esta aplicación. Se verificará la firma y las versiones '
                 'antes de abrirlo. No carga archivos .pkl de terceros.')
        upload = st.file_uploader('Respaldo del modelo', type=['pycmodel'], disabled=not store.stable,
                                  key='ml_restore_upload')
        st.caption('Límite real de esta función: 64 MiB por modelo, aunque el navegador anuncie un límite mayor.')
        if st.button('Importar a mi biblioteca', disabled=upload is None or not store.stable, key='ml_import'):
            try:
                if upload.size > MAX_BYTES:
                    raise LibraryError('El respaldo supera 64 MiB.')
                identifier = store.import_backup(code, upload.getvalue())
                _notify('Respaldo importado. En Recuperar o eliminar, selecciona el modelo y pulsa Recuperar. No se reentrenó.')
            except Exception as exc:
                _error(exc)
    st.caption('Nada se guarda automáticamente solo por abrir esta página. Guarda los modelos antes de cerrar la sesión. '
               'Guardar el diccionario en la pantalla anterior puede limpiar los modelos activos, pero no elimina esta biblioteca.')
