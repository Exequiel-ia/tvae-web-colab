"""PyC Smart Test Data Lab — demonstration with synthetic-only data."""
import io
import sqlite3
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer

st.set_page_config(page_title="PyC Smart Test Data Lab", page_icon="🧪", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1400px}
[data-testid="stSidebar"]{background:#172658}
[data-testid="stSidebar"] *{color:#fff!important}
.hero{background:linear-gradient(125deg,#172658,#2f4f9d);color:white;padding:28px;border-radius:20px;margin-bottom:22px}
.hero h1{color:white;margin:0;font-size:2.2rem}
.hero p{color:#dbe6ff;margin:8px 0 0}
.wordmark{background:white;color:#344b9f;display:inline-block;padding:8px 22px;border-radius:12px;font:italic 700 32px Georgia;margin-bottom:12px}
.section{padding:10px 0;font-weight:700;color:#172658;font-size:1.2rem}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="wordmark">PyC</div><h1>Smart Test Data Lab</h1><p>Calidad de datos · Inteligencia sintética · Comparación de escenarios</p></div>',unsafe_allow_html=True)
st.caption("PoC · Datos 100% ficticios · TVAE sobre clientes; entidades dependientes generadas mediante reglas")

def build_data(n=250, seed=42):
    rng=np.random.default_rng(seed)
    clientes=pd.DataFrame({
        "cliente_id":np.arange(1,n+1),
        "segmento":rng.choice(["Personas","Pyme","Empresa"],n,p=[.62,.28,.10]),
        "region":rng.choice(["Lima","Arequipa","Cusco","Trujillo"],n),
        "nivel_riesgo":rng.choice(["Bajo","Medio","Alto"],n,p=[.65,.25,.10]),
        "edad":rng.integers(18,81,n)})
    ventas=[]
    for cid in clientes.cliente_id:
        for _ in range(int(rng.integers(1,4))):
            ventas.append([len(ventas)+1,int(cid),rng.choice(["Web","Sucursal","App"]),round(float(rng.lognormal(5.5,.8)),2)])
    ventas=pd.DataFrame(ventas,columns=["venta_id","cliente_id","canal","monto_venta"])
    facturas=ventas[["venta_id","cliente_id","monto_venta"]].copy()
    facturas.insert(0,"factura_id",np.arange(1,len(facturas)+1))
    facturas["monto_total"]=(facturas.pop("monto_venta")*1.18).round(2)
    facturas["estado"]=rng.choice(["Pendiente","Pagada","Parcial"],len(facturas),p=[.2,.65,.15])
    pagos=[]
    for f in facturas.itertuples(index=False):
        if f.estado=="Pendiente":continue
        fraccion=1.0 if f.estado=="Pagada" else .5
        pagos.append([len(pagos)+1,int(f.factura_id),int(f.cliente_id),round(float(f.monto_total)*fraccion,2)])
    pagos=pd.DataFrame(pagos,columns=["pago_id","factura_id","cliente_id","monto_pago"])
    return dict(clientes=clientes,ventas=ventas,facturas=facturas,pagos=pagos)

def save_sqlite(tables):
    c=sqlite3.connect(":memory:")
    for name,frame in tables.items():frame.to_sql(name,c,if_exists="replace",index=False)
    return c

def verify(t):
    a=t["ventas"].cliente_id.isin(t["clientes"].cliente_id).all()
    b=t["facturas"].venta_id.isin(t["ventas"].venta_id).all()
    c=t["pagos"].factura_id.isin(t["facturas"].factura_id).all()
    return bool(a and b and c)


METRIC_DESCRIPTIONS={
"MET1":"Fidelidad estadística: similitud de correlaciones entre variables numéricas comparables.",
"MET2":"Diversidad categórica: cobertura de categorías y adherencia a los dominios observados.",
"MET3":"Coherencia estructural: columnas y proporción de nulos comparables.",
"MET4":"Reproducción directa: tasa de registros sintéticos sin copia exacta del original; NO garantiza privacidad.",
"MET5":"Outliers: similitud de tasas fuera de los límites definidos por media ± 3 desviaciones.",
"MET6":"Utilidad potencial QA: cobertura de rangos y respeto de límites; NO demuestra utilidad funcional.",
"MET7":"Compatibilidad técnica: columnas compartidas y coincidencia de tipos físicos."
}
def assess_seven(original,synth):
    """Métricas orientativas, no reproducción exacta del evaluador del Capstone."""
    numeric=[c for c in original.select_dtypes(include="number").columns if c in synth.columns and pd.api.types.is_numeric_dtype(synth[c]) and not c.endswith("_id")]
    cat=[c for c in original.columns if c in synth.columns and c not in numeric and not c.endswith("_id")]
    common=[c for c in original.columns if c in synth.columns]
    if len(numeric)>=2:
        a=original[numeric].corr().fillna(0).to_numpy()
        b=synth[numeric].corr().fillna(0).to_numpy()
        met1=100*max(0,1-float(np.mean(abs(a-b)))/2)
    else:met1=np.nan
    vals=[]
    for c in cat:
        r=set(original[c].dropna().astype(str)); q=set(synth[c].dropna().astype(str))
        if r and q:vals.append((len(r&q)/len(r)+len(r&q)/len(q))/2)
    met2=100*np.mean(vals) if vals else np.nan
    null_sim=np.mean([1-abs(original[c].isna().mean()-synth[c].isna().mean()) for c in common])
    met3=100*(len(common)/len(original.columns)+null_sim)/2
    cols=[c for c in common if not c.endswith("_id")]
    real_rows=set(map(tuple,original[cols].fillna("<NA>").astype(str).to_numpy())) if cols else set()
    copies=sum(tuple(x) in real_rows for x in synth[cols].fillna("<NA>").astype(str).to_numpy()) if cols else 0
    met4=100*(1-copies/max(len(synth),1))
    deviations=[];ranges=[]
    for c in numeric:
        r=pd.to_numeric(original[c],errors="coerce").dropna()
        q=pd.to_numeric(synth[c],errors="coerce").dropna()
        if len(r)<2 or len(q)==0:continue
        mean,sd=r.mean(),r.std()
        deviations.append(abs(((r<mean-3*sd)|(r>mean+3*sd)).mean()-((q<mean-3*sd)|(q>mean+3*sd)).mean()))
        low,high=r.min(),r.max()
        span=max(float(high-low),1e-9)
        coverage=max(0,min(high,q.max())-max(low,q.min()))/span
        adherence=((q>=low)&(q<=high)).mean()
        ranges.append(.7*coverage+.3*adherence)
    met5=100*max(0,1-10*np.mean(deviations)) if deviations else np.nan
    met6=100*np.mean(ranges) if ranges else np.nan
    met7=100*np.mean([str(original[c].dtype)==str(synth[c].dtype) for c in original.columns]) if len(original.columns) else np.nan
    return dict(zip(METRIC_DESCRIPTIONS,[met1,met2,met3,met4,met5,met6,met7]))

if "tables" not in st.session_state:st.session_state.tables=build_data()
if "runs" not in st.session_state:st.session_state.runs=[]
if "models" not in st.session_state:st.session_state.models=[]
page=st.sidebar.radio("Secciones",["Base de datos","Exploración","Entrenamiento","Generación","Comparar experimentos","7 métricas QE"])
t=st.session_state.tables
if page=="Base de datos":
    st.subheader("Modelo: Clientes → Ventas → Facturas → Pagos")
    n=st.number_input("Clientes ficticios",min_value=50,max_value=2000,value=250,step=50)
    seed=st.number_input("Semilla",min_value=0,value=42)
    if st.button("Crear base de datos ficticia"):
        st.session_state.tables=build_data(int(n),int(seed));st.rerun()
    conn=save_sqlite(t)
    table=st.selectbox("Entidad",list(t))
    st.dataframe(pd.read_sql_query("SELECT * FROM "+table+" LIMIT 100",conn),use_container_width=True)
    st.metric("Integridad referencial","Correcta" if verify(t) else "Error")
    buf=io.BytesIO()
    with sqlite3.connect(":memory:") as temp:
        for name,frame in t.items():frame.to_sql(name,temp,index=False)
        import tempfile,os
        with tempfile.NamedTemporaryFile(suffix=".sqlite",delete=False) as f:path=f.name
        target=sqlite3.connect(path);temp.backup(target);target.close()
        with open(path,"rb") as f:buf.write(f.read())
        os.unlink(path)
    st.download_button("Descargar SQLite",buf.getvalue(),"pyc_demo.sqlite","application/x-sqlite3")
    st.info("Los datos de ejemplo son inventados. Puedes seleccionar y leer cada entidad desde SQLite.")
elif page=="Exploración":
    name=st.selectbox("Entidad",list(t))
    df=t[name]
    st.dataframe(df.head(30),use_container_width=True)
    column=st.selectbox("Variable",df.columns.tolist())
    if pd.api.types.is_numeric_dtype(df[column]):
        st.plotly_chart(px.histogram(df,x=column,title="Distribución: "+column),use_container_width=True)
    else:
        st.plotly_chart(px.bar(df[column].value_counts().reset_index(),x=column,y="count"),use_container_width=True)
    st.write("Nulos:",int(df.isna().sum().sum())," · Filas:",len(df)," · Duplicados:",int(df.duplicated().sum()))
elif page=="Entrenamiento":
    st.subheader("01 · Entrenamiento de TVAE")
    st.markdown("El entrenamiento aprende patrones estadísticos del dataset original. **No genera todavía nuevos registros**. Al terminar, el modelo queda disponible para ejecutar diferentes generaciones.")
    st.info("En esta primera PoC se entrena exclusivamente la tabla Clientes, sin cliente_id. Ventas, Facturas y Pagos siguen siendo construidas mediante reglas de ejemplo. No es todavía un modelo generativo relacional.")
    experiment_name=st.text_input("Nombre del modelo",f"TVAE {len(st.session_state.models)+1}")
    c1,c2,c3=st.columns(3)
    with c1:
        epochs=st.slider("Épocas",min_value=10,max_value=400,value=100,step=10,help="Número de pasadas completas por todos los registros. Más épocas dan más oportunidades de aprender, pero tardan más y pueden sobreajustar.")
        st.caption("↑ Mayor aprendizaje potencial y tiempo. ↓ Entrena más rápido; puede perder patrones.")
        batch=st.selectbox("Batch size",[20,50,100,200,500],index=2,help="Registros usados por cada actualización del modelo. Un batch pequeño introduce más variabilidad y más pasos por época; uno grande suele usar más memoria.")
        st.caption("↑ Actualizaciones más estables, más memoria. ↓ Más pasos y variabilidad.")
    with c2:
        embedding=st.selectbox("Dimensión latente (embedding)",[16,32,64,128,256],index=3,help="Capacidad del espacio interno comprimido donde el modelo representa patrones de los datos.")
        st.caption("↑ Más capacidad para relaciones complejas, mayor riesgo de ajustar ruido.")
        layers=st.selectbox("Capas internas encoder/decoder",["64 × 64","128 × 128","256 × 256"],index=1,help="Número de neuronas por capa interna, tanto en el codificador como en el decodificador.")
        st.caption("↑ Mayor complejidad y costo. ↓ Modelo más simple y rápido.")
    with c3:
        l2=st.selectbox("Regularización L2",[0.0,1e-6,1e-5,1e-4,1e-3],index=2,format_func=lambda v:f"{v:g}",help="Penalización que desalienta pesos excesivos; puede reducir sobreajuste, pero si es alta pierde detalles.")
        st.caption("↑ Penaliza más el sobreajuste. ↓ Modelo más flexible.")
        loss=st.selectbox("Peso de reconstrucción (loss factor)",[1,2,3,5,10],index=1,help="Prioridad relativa de reconstruir los datos con fidelidad frente a la regularización variacional.")
        st.caption("↑ Prioriza reconstrucción; conviene validar diversidad y copias.")
    with st.expander("Opciones avanzadas y reproducibilidad",expanded=False):
        seed=st.number_input("Semilla del experimento",0,999999,42,help="Punto de partida del azar. Usar la misma semilla ayuda a comparar configuraciones, aunque no garantiza resultados idénticos en todo entorno.")
        minmax=st.checkbox("Respetar mínimo y máximo numérico observado",value=True,help="Durante el posprocesamiento, limita valores numéricos a los límites observados. Puede recortar extremos.")
        rounding=st.checkbox("Conservar redondeo numérico",value=True,help="Intenta devolver los valores numéricos con una precisión equivalente a la de origen.")
        st.caption("La semilla es reproducibilidad, no calidad. El número de filas sintéticas se define en Generación y NO requiere reentrenar.")
    if st.button("Entrenar y guardar modelo",type="primary"):
        source=t["clientes"].drop(columns=["cliente_id"])
        meta=SingleTableMetadata();meta.detect_from_dataframe(data=source)
        dims=tuple(int(v) for v in layers.split(" × "))
        with st.spinner("Entrenando TVAE (esta etapa puede tomar tiempo)..."):
            try:
                import torch
                torch.manual_seed(int(seed));np.random.seed(int(seed))
                model=TVAESynthesizer(meta,epochs=int(epochs),batch_size=int(batch),embedding_dim=int(embedding),compress_dims=dims,decompress_dims=dims,l2scale=float(l2),loss_factor=int(loss),enforce_min_max_values=minmax,enforce_rounding=rounding,enable_gpu=False)
                start=time.perf_counter();model.fit(source);duration=time.perf_counter()-start
                st.session_state.models.append({"name":experiment_name,"model":model,"train_sec":round(duration,2),"params":{"épocas":epochs,"batch":batch,"embedding":embedding,"capas":layers,"L2":l2,"loss":loss,"semilla":seed,"minmax":minmax,"rounding":rounding}})
                st.success(f"Modelo '{experiment_name}' entrenado y guardado en esta sesión. Tiempo: {duration:.2f} s.")
            except Exception as exc:st.error(f"Error de entrenamiento: {exc}")
    if st.session_state.models:
        st.markdown("**Modelos entrenados disponibles**")
        st.dataframe(pd.DataFrame([{"Modelo":m["name"],"Tiempo (s)":m["train_sec"],**m["params"]} for m in st.session_state.models]),use_container_width=True)
    st.caption("Los modelos se mantienen en memoria durante esta sesión. Streamlit Community Cloud puede reiniciar la sesión y perderlos; persistencia será una fase posterior.")

elif page=="Generación":
    st.subheader("02 · Generación de datos sintéticos")
    st.markdown("Selecciona un modelo **ya entrenado**, define cuántos clientes generar y crea uno o más datasets. Puedes repetir la generación sin entrenar otra vez.")
    if not st.session_state.models:
        st.warning("No hay modelos entrenados. Primero entra a la sección Entrenamiento.")
    else:
        model_idx=st.selectbox("Modelo entrenado",list(range(len(st.session_state.models))),format_func=lambda i:f"{st.session_state.models[i]['name']} · {st.session_state.models[i]['params']['épocas']} épocas")
        qty=st.slider("Cantidad de clientes sintéticos",50,2000,250,step=50,help="Número de filas solicitadas al modelo entrenado. Cambiar este número no afecta los pesos del modelo ni exige reentrenamiento.")
        batch_name=st.text_input("Nombre del dataset generado",f"Generación {len(st.session_state.runs)+1}")
        if st.button("Generar datos con el modelo seleccionado",type="primary"):
            with st.spinner("Generando datos nuevos..."):
                try:
                    selected=st.session_state.models[model_idx]
                    start=time.perf_counter()
                    generated=selected["model"].sample(num_rows=int(qty))
                    generation=time.perf_counter()-start
                    generated["edad"]=pd.to_numeric(generated["edad"],errors="coerce").fillna(40).clip(18,100).astype(int)
                    generated["cliente_id"]=np.arange(1,len(generated)+1)
                    synthetic=build_data(int(qty),int(len(st.session_state.runs)+100))
                    synthetic["clientes"]=generated[["cliente_id","segmento","region","nivel_riesgo","edad"]]
                    score=100*max(0,1-abs(float(t["clientes"].edad.mean())-float(generated.edad.mean()))/max(float(t["clientes"].edad.mean()),1))
                    run={"nombre":batch_name,"modelo":selected["name"],"epochs":selected["params"]["épocas"],"batch":selected["params"]["batch"],"rows":qty,"train_sec":selected["train_sec"],"gen_sec":round(generation,2),"edad_media_similitud":round(score,2),"tables":synthetic}
                    st.session_state.runs.append(run)
                    st.success(f"Dataset '{batch_name}' generado en {generation:.2f}s, sin reentrenar. Integridad básica: {verify(synthetic)}")
                except Exception as exc:st.error(f"No se pudo generar: {exc}")
        if st.session_state.runs:
            idx=st.selectbox("Dataset generado para inspeccionar",list(range(len(st.session_state.runs))),format_func=lambda i:f"{st.session_state.runs[i]['nombre']} · {st.session_state.runs[i]['modelo']}")
            selected_run=st.session_state.runs[idx]
            name=st.selectbox("Tabla sintética",list(selected_run["tables"]))
            st.dataframe(selected_run["tables"][name].head(30),use_container_width=True)
            st.download_button("Descargar CSV sintético",selected_run["tables"][name].to_csv(index=False),name+".csv")

elif page=="Comparar experimentos":
    st.subheader("Historial en la sesión")
    if not st.session_state.runs:st.info("Primero ejecuta al menos una generación.")
    else:
        rows=[{k:v for k,v in x.items() if k!="tables"} for x in st.session_state.runs]
        df=pd.DataFrame(rows);st.dataframe(df,use_container_width=True)
        st.plotly_chart(px.line(df.reset_index(),x="index",y=["edad_media_similitud"],markers=True,title="Indicador orientativo por experimento"),use_container_width=True)
        chosen=st.selectbox("Experimento",range(len(st.session_state.runs)),format_func=lambda i:f"Experimento {i+1}")
        syn=st.session_state.runs[chosen]["tables"]["clientes"]
        orig=t["clientes"]
        joined=pd.concat([orig.assign(origen="Original"),syn.assign(origen="Sintético")])
        st.plotly_chart(px.histogram(joined,x="edad",color="origen",barmode="overlay"),use_container_width=True)
        st.caption("Esta comparación es ilustrativa, no reproduce las siete métricas completas del Capstone ni certifica privacidad.")

elif page=="7 métricas QE":
    st.subheader("Siete mediciones: datos originales frente a sintéticos")
    st.info("Métricas demostrativas de las dimensiones del Capstone. No equivalen a evaluación de privacidad ni a certificación funcional.")
    if not st.session_state.runs:
        st.warning("Entrena primero un modelo en Entrenamiento y después genera un dataset en Generación.")
    else:
        idx=st.selectbox("Experimento",list(range(len(st.session_state.runs))),format_func=lambda i:f"Experimento {i+1} · {st.session_state.runs[i]['epochs']} épocas")
        name=st.selectbox("Entidad",list(t))
        real=t[name];synt=st.session_state.runs[idx]["tables"][name]
        scores=assess_seven(real,synt)
        metric_rows=[{"Métrica":k,"Puntuación (0–100)":round(v,2) if pd.notna(v) else None,"Qué evalúa":METRIC_DESCRIPTIONS[k]} for k,v in scores.items()]
        df=pd.DataFrame(metric_rows)
        st.dataframe(df,use_container_width=True,hide_index=True)
        plot=df.dropna(subset=["Puntuación (0–100)"])
        st.plotly_chart(px.bar(plot,x="Métrica",y="Puntuación (0–100)",title="Indicadores originales vs sintéticos",range_y=[0,100]),use_container_width=True)
        a,b=st.columns(2)
        with a:
            st.markdown("**Datos originales**")
            st.dataframe(real.head(40),use_container_width=True)
        with b:
            st.markdown("**Datos sintéticos**")
            st.dataframe(synt.head(40),use_container_width=True)
        common=[c for c in real.select_dtypes(include="number").columns if c in synt.columns and pd.api.types.is_numeric_dtype(synt[c])]
        if common:
            field=st.selectbox("Variable numérica a comparar",common)
            combined=pd.concat([real[[field]].assign(Dataset="Original"),synt[[field]].assign(Dataset="Sintético")],ignore_index=True)
            st.plotly_chart(px.histogram(combined,x=field,color="Dataset",barmode="overlay",marginal="box",title="Distribución comparada"),use_container_width=True)
        st.caption("Los identificadores de entidades reconstruidas no se comparan como registros equivalentes. MET1, MET5 y MET6 requieren variables numéricas de negocio y pueden no aplicar a todas las entidades.")
