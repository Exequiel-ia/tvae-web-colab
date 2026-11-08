"""Relational A/B screen. Reads runs/tables, never trains or mutates their state."""
from __future__ import annotations

import html
import inspect
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comparison_core import (METRIC_NAMES, METRIC_NOTES, SENSITIVE, business_columns,
    categorical_distribution, compare_metrics, metrics_match, model_relation,
    numeric_distribution, numeric_summary, outlier_evidence, parameter_table,
    provenance, safe_csv, schema_evidence, source_fingerprint)

COLORS = {'Original': '#8893a6', 'A': '#29458e', 'B': '#168b82'}
CSS = '''<style>
[data-testid="stSidebar"]{background:#172658}
[data-testid="stSidebar"] *{color:#fff!important}
.block-container{max-width:1350px;padding-top:2rem}
.cmp-hero{background:linear-gradient(115deg,#172658,#3458a4);padding:26px 30px;border-radius:20px;color:#fff;margin-bottom:24px}
.cmp-hero h1{color:#fff;font-size:2rem;padding:0;margin:4px 0 8px}
.cmp-hero p{color:#e2eaff;margin:0;line-height:1.5}
.cmp-kicker{font-size:.78rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
.cmp-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:16px 0}
.cmp-card{border:1px solid #dfe6f0;border-top:5px solid #29458e;border-radius:15px;background:white;padding:19px 22px;min-width:0;box-shadow:0 5px 20px #17265808}
.cmp-card.b{border-top-color:#168b82}
.cmp-card h3{margin:6px 0;color:#172658;font-size:1.12rem;overflow-wrap:anywhere}
.cmp-card p{margin:6px 0;color:#536176;font-size:.92rem;overflow-wrap:anywhere}
.cmp-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}
.cmp-stat{border:1px solid #dfe6f0;border-radius:14px;padding:14px 18px;background:#f7f9fd}
.cmp-stat b{display:block;color:#172658;font-size:1.7rem}
.cmp-stat span{color:#536176;font-size:.85rem}
.cmp-note{border-left:4px solid #29458e;border-radius:8px;background:#eff4ff;color:#243353;padding:16px 20px;line-height:1.6;margin:16px 0}
.cmp-badge{font-size:.76rem;font-weight:700;color:#29458e;letter-spacing:.06em}
@media(max-width:900px){.cmp-grid{grid-template-columns:1fr}.cmp-stats{grid-template-columns:repeat(2,1fr)}.cmp-hero{padding:20px}.cmp-hero h1{font-size:1.6rem}}
</style>'''


def esc(value):
    return html.escape(str(value), quote=True)


def _df(frame, **kwargs):
    width = inspect.signature(st.dataframe).parameters.get('width')
    sizing = {'width': 'stretch'} if width is not None and isinstance(width.default, str) else {'use_container_width': True}
    st.dataframe(frame, hide_index=True, **sizing, **kwargs)


def _chart(fig, key):
    fig.update_layout(template='plotly_white', font=dict(family='Arial, sans-serif', size=13, color='#243353'),
                      paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=10,r=70,t=55,b=25),
                      legend=dict(orientation='h', y=1.1, x=0, title_text=''),
                      hoverlabel=dict(bgcolor='white'), bargap=.22)
    fig.update_xaxes(gridcolor='#edf1f7', automargin=True)
    fig.update_yaxes(gridcolor='#edf1f7', automargin=True)
    sizing = {'width': 'stretch'} if 'width' in inspect.signature(st.plotly_chart).parameters else {'use_container_width': True}
    st.plotly_chart(fig, key=key, config={'displaylogo': False, 'scrollZoom': False, 'responsive': True}, **sizing)


def _fmt(value):
    return 'No aplica' if pd.isna(value) else f'{value:.2f}'


def _run_label(i, runs):
    r = runs[i]
    return f"G{i+1} · {r.get('nombre', 'Sin nombre')} · {r.get('modelo', 'Modelo no registrado')}"


def _cards(a, b, ia, ib, entity):
    cards = []
    for tag, run, idx, role in [('A',a,ia,'Referencia'), ('B',b,ib,'Candidato')]:
        n = len(run['tables'][entity])
        cards.append(f'''<div class="cmp-card {'b' if tag == 'B' else ''}">
        <span class="cmp-badge">{tag} · {role.upper()} · GENERACIÓN {idx+1}</span>
        <h3>{esc(run.get('nombre','Sin nombre'))}</h3>
        <p><b>Modelo:</b> {esc(run.get('modelo','No registrado'))}</p>
        <p><b>{n:,} registros de {esc(entity)}</b> · {esc(run.get('params',{}).get('épocas','—'))} épocas</p>
        <p>Generado: {esc(run.get('generado_el','Fecha no registrada'))}</p></div>''')
    st.markdown('<div class="cmp-grid">'+''.join(cards)+'</div>', unsafe_allow_html=True)


def _summary(scores, parameters, entity, unverified):
    changed = parameters[(parameters.Grupo == 'Entrenamiento') & (parameters.Estado == 'Cambió')]
    counts = scores.Lectura.value_counts()
    items = [(len(changed),'Parámetros de entrenamiento distintos'), (counts.get('Sube',0),'Puntuaciones que suben'),
             (counts.get('Baja',0),'Puntuaciones que bajan'), (counts.get('No comparable',0),'Métricas no comparables')]
    st.markdown('<div class="cmp-stats">'+''.join(f'<div class="cmp-stat"><b>{n}</b><span>{esc(label)}</span></div>' for n,label in items)+'</div>', unsafe_allow_html=True)
    st.markdown('### Exe-IA · Qué cambió y qué observar')
    changes = '; '.join(f"{r['Parámetro']}: {r['A']} → {r['B']}" for r in changed.to_dict('records'))
    text = ('Cambios registrados: '+changes+'. ') if changes else 'No hay diferencias registradas entre los parámetros de entrenamiento conocidos. '
    comparable = scores.dropna(subset=['Δ B − A (puntos)'])
    for label, sign, word in [('Sube',1,'mayor subida'), ('Baja',-1,'mayor bajada')]:
        part = comparable[comparable.Lectura == label]
        if not part.empty:
            row = part.loc[(part['Δ B − A (puntos)']*sign).idxmax()]
            text += f"La {word} es {row['Dimensión']}: {row['Δ B − A (puntos)']:+.2f} puntos. "
    if comparable.empty:
        text += 'No hay puntuaciones comparables para concluir. '
    elif (comparable.Lectura == 'Sin cambio').all():
        text += 'Las puntuaciones calculables no cambian a dos decimales. '
    text += 'Son diferencias observadas, no prueba de causalidad ni un ganador global.'
    st.markdown('<div class="cmp-note">'+esc(text)+'</div>', unsafe_allow_html=True)
    if unverified:
        st.caption('Lectura exploratoria: el historial no permite certificar qué original se usó al generar A y B.')
    st.caption('Cambios expresados en puntos de la escala 0–100, no en porcentaje de calidad ni en significancia estadística. Se redondea a dos decimales.')
    if entity != 'clientes':
        st.info('Esta entidad se reconstruye con reglas y patrones empíricos, no con un TVAE propio. Sus diferencias no deben atribuirse automáticamente a las épocas de Clientes.')
    st.info('MET1 requiere al menos dos variables numéricas de negocio. MET2 mide cobertura, no frecuencia. MET4 incluye textos ficticios: 100 no certifica privacidad.')


def _metric_chart(scores):
    fig = go.Figure()
    labels = [key+' · '+ '<br>'.join(textwrap.wrap(name, 27)) for key,name in zip(scores['Métrica'],scores['Dimensión'])]
    for label in ['A','B']:
        fig.add_bar(name=label, y=labels, x=scores[label], orientation='h', marker_color=COLORS[label],
                    text=[_fmt(v) if pd.notna(v) else '' for v in scores[label]], textposition='outside', cliponaxis=False,
                    hovertemplate='%{y}<br>'+label+': %{x:.2f}/100<extra></extra>')
    annotations = [dict(x=3,y=label,text='No aplica / no calculable',showarrow=False,xanchor='left',font=dict(color='#6b7280',size=12))
                   for label,row in zip(labels,scores.to_dict('records')) if pd.isna(row['A']) and pd.isna(row['B'])]
    fig.update_layout(barmode='group', height=620, annotations=annotations)
    fig.update_xaxes(range=[0,100], title='Puntuación frente al original (0–100)')
    fig.update_yaxes(autorange='reversed')
    _chart(fig,'ab_metric_chart')


def _data(frames, entity, widget_suffix=''):
    common = [c for c in frames['Original'] if all(c in df for df in frames.values())]
    preferred = [c for c in business_columns(frames['Original']) if c in common]
    numeric = [c for c in preferred if all(pd.api.types.is_numeric_dtype(df[c]) for df in frames.values())]
    fields = numeric + [c for c in preferred if c not in numeric]
    st.markdown('### Original · A · B: evidencia en los datos')
    st.caption('Gris = Original · Azul = A · Turquesa = B. Cada proporción usa el tamaño de su propio conjunto. No se emparejan registros por fila o por ID.')
    st.write('Registros por conjunto:', {label: len(df) for label,df in frames.items()})
    if fields:
        default = fields.index('edad') if 'edad' in fields else 0
        column = st.selectbox('Variable a comparar', fields, index=default, key='ab_variable_'+entity)
        if column in numeric:
            bins = st.slider('Intervalos comunes para los tres conjuntos', 5, 30, 10, key='ab_bins_'+entity)
            dist = numeric_distribution(frames,column,bins)
            catcol = 'Intervalo'
            st.caption('Distribución sobre valores numéricos finitos. Los nulos y valores no evaluados se informan en el resumen. Los intervalos se calculan una sola vez usando Original, A y B.')
        else:
            dist = categorical_distribution(frames,column)
            catcol = 'Categoría'
            st.caption('Frecuencias de categorías: incluye nulos. Si hay más de 15 categorías, el resto se agrupa; consulta el detalle para los conteos.')
        if not dist.empty:
            fig = go.Figure()
            for label in frames:
                part = dist[dist.Conjunto == label]
                fig.add_bar(name=label,x=part[catcol],y=part.Porcentaje,marker_color=COLORS[label],
                            customdata=part[['Cantidad']],hovertemplate='%{x}<br>%{y:.2f}% · %{customdata[0]} registros<extra>'+label+'</extra>')
            fig.update_layout(barmode='group',height=420)
            fig.update_yaxes(title='Porcentaje dentro de cada conjunto', rangemode='tozero')
            fig.update_xaxes(title=column)
            _chart(fig,'ab_distribution_'+entity)
            with st.expander('Ver porcentajes y cantidades exactas'):
                _df(dist)
        else:
            st.info('No hay valores comparables para dibujar esta distribución.')
        if column in numeric:
            _df(numeric_summary(frames,column).round(3))
            with st.expander('Casos extremos: límites, conteos y registros'):
                method = st.radio('Regla de referencia', ['Media ± 3 desviaciones (MET5)', 'IQR (exploración adicional)'], key='ab_extreme_method_'+entity)
                limits, summary, masks = outlier_evidence(frames,column,'sigma' if method.startswith('Media') else 'iqr')
                if limits is None:
                    st.info('No se establece un límite con esta regla: pocos valores o IQR igual a cero. No significa que haya cero casos extremos.')
                else:
                    st.write(f'Límite inferior: {limits[0]:.4g} · superior: {limits[1]:.4g}. Ambos se calculan exclusivamente sobre el original y se aplican a los tres conjuntos.')
                    _df(summary.round(3))
                    label = st.selectbox('Casos extremos de', list(frames), key='ab_extreme_source_'+entity)
                    _df(frames[label].loc[masks[label], preferred].head(100))
                    st.caption('Se muestran hasta 100 casos. Un extremo puede ser válido para QA; el conteo no es una puntuación de calidad.')
    else:
        st.info('No hay variables de negocio comunes. Revisa estructura y registros.')
    with st.expander('Estructura, tipos y nulos por columna'):
        _df(schema_evidence(frames).round(3))
    st.markdown('### Inspeccionar y descargar registros')
    dataset = st.radio('Conjunto a inspeccionar',list(frames),horizontal=True,key='ab_record_source_'+entity)
    available = list(frames[dataset].columns)
    cols = st.multiselect('Columnas visibles y a descargar',available,default=[c for c in available if c not in SENSITIVE],key='ab_columns_'+entity+'_'+dataset)
    text = st.text_input('Filtrar texto en las columnas seleccionadas (no distingue mayúsculas)',key='ab_filter_'+entity+'_'+dataset)
    show = frames[dataset][cols].copy()
    if text and cols:
        keep = show.astype(str).apply(lambda s:s.str.contains(text,case=False,regex=False,na=False)).any(axis=1)
        show = show[keep]
    limit = st.selectbox('Registros visibles', [40,100,250],key='ab_preview_limit_'+entity)
    st.caption(f'{len(show):,} registros cumplen el filtro; vista de hasta {limit}. La descarga incluye todos los filtrados. El filtro no altera métricas ni gráficos.')
    if cols:
        _df(show.head(limit))
        st.download_button('Descargar estos registros (CSV)',safe_csv(show),f'comparador_{entity}_{dataset}.csv','text/csv',key='ab_download_records')
    st.caption('CSV en UTF-8; texto con prefijos de fórmula se neutraliza para abrirlo en una hoja de cálculo. No se descargan modelos ni se modifica el origen.')


def render_comparator():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="cmp-hero"><div class="cmp-kicker">PyC · Laboratorio relacional</div><h1>Comparador A/B</h1><p>Qué cambió · qué subió o bajó · los datos que explican la diferencia</p></div>',unsafe_allow_html=True)
    st.sidebar.markdown('### Comparación relacional')
    st.sidebar.page_link('streamlit_app.py',label='Volver a la demo relacional')
    st.sidebar.caption('Lectura de generaciones existentes. No entrena, no genera ni borra resultados.')
    runs = st.session_state.get('runs',[])
    original = st.session_state.get('tables',{})
    usable = [i for i,r in enumerate(runs) if isinstance(r.get('tables'),dict)]
    if len(usable) < 2 or not original:
        st.info('Necesitas al menos dos generaciones relacionales en esta sesión. Entrenar dos modelos no crea todavía sus datos.')
        st.markdown('En la demo relacional: **Entrenamiento** → guarda uno o más modelos; **Generación** → elige un modelo y genera datos dos veces; luego vuelve a **Comparador A/B**. También puedes comparar dos generaciones del mismo modelo.')
        st.warning('Los resultados viven en la sesión. Un reinicio o una sesión nueva puede dejar este comparador vacío. No se han borrado datos desde esta pantalla.')
        return
    aindex = st.selectbox('A · Generación de referencia',usable,index=0,format_func=lambda i:_run_label(i,runs),key='ab_pick_a')
    choices = [i for i in usable if i != aindex]
    bindex = st.selectbox('B · Generación candidata',choices,index=len(choices)-1,format_func=lambda i:_run_label(i,runs),key='ab_pick_b')
    a,b = runs[aindex],runs[bindex]
    entities = [e for e in original if e in a['tables'] and e in b['tables']]
    if not entities:
        st.error('A, B y el original activo no comparten entidades. No se puede comparar.')
        return
    entity = st.selectbox('Entidad a comparar',entities,format_func=str.capitalize,key='ab_entity')
    _cards(a,b,aindex,bindex,entity)
    st.caption(model_relation(a,b))
    params = parameter_table(a,b)
    changed = params[(params.Grupo=='Entrenamiento') & (params.Estado=='Cambió')]
    if len(changed):
        st.markdown('**Qué cambió:** ' + ' · '.join(f"{r['Parámetro']}: {r['A']} → {r['B']}" for r in changed.to_dict('records')))
    status,message = provenance(a,b,source_fingerprint(original))
    if status == 'blocked':
        st.error(message)
        _df(params)
        return
    if status == 'verified':
        st.success(message)
    else:
        st.warning(message)
    if len(a['tables'][entity]) != len(b['tables'][entity]):
        st.info('A y B tienen tamaños distintos. Las distribuciones se muestran en porcentajes; las puntuaciones aún pueden depender del tamaño de la muestra.')
    if not metrics_match():
        st.error('Las fórmulas del comparador no coinciden con la pantalla original o no se pudieron verificar. Se detiene la evaluación para no mezclar versiones.')
        return
    frames = {'Original':original[entity], 'A':a['tables'][entity], 'B':b['tables'][entity]}
    scores,warnings = compare_metrics(frames['Original'],frames['A'],frames['B'])
    for warning in warnings:
        st.warning(warning)
    summary,parameters,metrics,data = st.tabs(['Resumen','Parámetros','Métricas','Datos'])
    with summary:
        _summary(scores,params,entity,status!='verified')
        st.markdown('### Puntuaciones A/B frente al mismo original activo')
        _metric_chart(scores)
        st.caption('Sin nota global ni ganador automático. No aplica no equivale a cero. El detalle metodológico está en Métricas y la evidencia en Datos.')
    with parameters:
        only = st.checkbox('Mostrar solo diferencias o información no registrada',value=True,key='ab_only_diff')
        shown = params[params.Estado!='Igual'] if only else params
        if shown.empty:
            st.info('No hay diferencias registradas. Desmarca la opción para revisar todos los parámetros.')
        else:
            _df(shown)
        st.caption('Los tiempos son resultados de ejecución, no parámetros. Una semilla no registrada no se infiere desde la semilla del entrenamiento.')
        if len(changed)>1:
            st.warning('Cambió más de un parámetro de entrenamiento: no es posible aislar el efecto de cada uno con este par de resultados.')
        st.info('Incluso cambiando un solo parámetro, conviene repetir generaciones y entrenamientos antes de concluir que produjo la diferencia.')
    with metrics:
        st.markdown('### Métricas y explicación')
        _df(scores.rename(columns={'A':'A /100','B':'B /100'}))
        for row in scores.to_dict('records'):
            delta = row['Δ B − A (puntos)']
            delta_text = 'No comparable' if pd.isna(delta) else f"{delta:+.2f} puntos · {row['Lectura']}"
            with st.expander(f"{row['Métrica']} · {row['Dimensión']} · {delta_text}"):
                st.write(f"A: {_fmt(row['A'])} /100 · B: {_fmt(row['B'])} /100")
                st.write(METRIC_NOTES[row['Métrica']])
                st.caption('Consulta Datos para distribuciones, estructura, nulos, límites y registros. Las fórmulas son las mismas de 7 métricas QE, sin ponderaciones de Exploración.')
        st.download_button('Descargar métricas y diferencias (CSV)',safe_csv(scores),f'comparacion_{entity}_G{aindex+1}_G{bindex+1}.csv','text/csv',key='ab_download_metrics')
        report = {'entity':entity,'reference':'original activo de la sesión','provenance':status,'warning':message,
                  'A':_run_label(aindex,runs),'B':_run_label(bindex,runs),
                  'parameters':params.to_dict('records'), 'metrics':json.loads(scores.to_json(orient='records'))}
        st.download_button('Descargar resumen comparativo (JSON)',json.dumps(report,ensure_ascii=False,indent=2), 'comparacion_AB.json','application/json',key='ab_download_report')
    with data:
        _data(frames,entity)
    st.caption('PoC · Métricas demostrativas. No certifican privacidad ni utilidad funcional. Exe-IA usa reglas locales; no envía información a un LLM.')
