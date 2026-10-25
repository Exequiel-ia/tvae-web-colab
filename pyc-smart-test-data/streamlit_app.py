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

st.set_page_config(page_title="PyC Smart Test Data Lab", layout="wide")
st.title("PyC Smart Test Data Lab")
st.caption("Demostración comercial · Datos ficticios · TVAE real sobre tabla Clientes")

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

if "tables" not in st.session_state:st.session_state.tables=build_data()
if "runs" not in st.session_state:st.session_state.runs=[]
page=st.sidebar.radio("Secciones",["Base de datos","Exploración","Entrenar y generar","Comparar experimentos"])
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
elif page=="Entrenar y generar":
    st.subheader("TVAE — modelo entrenado sobre datos ficticios de clientes")
    epochs=st.slider("Épocas",10,300,30,step=10)
    batch=st.selectbox("Batch size",[32,64,128],index=1)
    qty=st.slider("Clientes sintéticos",50,1000,250,step=50)
    st.info("La demo entrena TVAE con la tabla Clientes (sin identificador), luego genera las otras tres entidades mediante reglas explícitas. Esta aproximación preserva integridad pero no modela todas las relaciones estadísticas entre entidades.")
    if st.button("Entrenar y generar",type="primary"):
        with st.spinner("Entrenando TVAE..."):
            source=t["clientes"].drop(columns=["cliente_id"])
            meta=SingleTableMetadata()
            meta.detect_from_dataframe(data=source)
            model=TVAESynthesizer(meta,epochs=int(epochs),batch_size=int(batch),enforce_min_max_values=True)
            start=time.perf_counter()
            model.fit(source)
            training=time.perf_counter()-start
            start=time.perf_counter()
            generated=model.sample(num_rows=int(qty))
            generation=time.perf_counter()-start
            generated["edad"]=pd.to_numeric(generated["edad"],errors="coerce").fillna(40).clip(18,100).astype(int)
            generated["cliente_id"]=np.arange(1,len(generated)+1)
            synthetic=build_data(int(qty),int(len(st.session_state.runs)+100))
            synthetic["clientes"]=generated[["cliente_id","segmento","region","nivel_riesgo","edad"]]
            score=100*max(0,1-abs(float(source["edad"].mean())-float(generated["edad"].mean()))/max(float(source["edad"].mean()),1))
            run={"epochs":epochs,"batch":batch,"rows":qty,"train_sec":round(training,2),"gen_sec":round(generation,2),"edad_media_similitud":round(score,2),"tables":synthetic}
            st.session_state.runs.append(run)
            st.success(f"Experimento guardado. Entrenamiento: {training:.2f}s · Generación: {generation:.2f}s · Integridad: {verify(synthetic)}")
    if st.session_state.runs:
        last=st.session_state.runs[-1]
        name=st.selectbox("Tabla sintética",list(last["tables"]))
        st.dataframe(last["tables"][name].head(25),use_container_width=True)
        st.download_button("Descargar CSV sintético",last["tables"][name].to_csv(index=False),name+".csv")
elif page=="Comparar experimentos":
    st.subheader("Historial en la sesión")
    if not st.session_state.runs:st.info("Primero ejecuta al menos un entrenamiento.")
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
