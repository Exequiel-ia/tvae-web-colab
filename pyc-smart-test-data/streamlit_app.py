"""PyC Smart Test Data Lab — demonstration with synthetic-only data."""
import io
from datetime import datetime
import json
import zipfile
import sqlite3
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from relational import generate_children, verify_relations

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
st.caption("**Última actualización del código: 30/10/2026 · 18:28 (Chile, UTC−3)** · Rama main")

def build_data(n=250, seed=42):
    rng=np.random.default_rng(seed)
    clientes=pd.DataFrame({
        "nombre_cliente":[f"Cliente Demostración {i+1:05d}" for i in range(n)],
        "direccion":[f"Calle Ficticia {i+1}, Sector Demo" for i in range(n)],
        "email":[f"demo{i+1:05d}@example.invalid" for i in range(n)],
        "telefono":[f"000-{i+1:07d}" for i in range(n)],
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

CONFIDENTIAL_FIELDS={"nombre_cliente","direccion","email","telefono"}
DOMAINS={"segmento":"Personas,Pyme,Empresa","region":"Lima,Arequipa,Cusco,Trujillo","nivel_riesgo":"Bajo,Medio,Alto","canal":"Web,Sucursal,App","estado":"Pendiente,Pagada,Parcial"}

def default_dictionary(tables):
    rows=[]
    for entity,df in tables.items():
        for col in df.columns:
            role="Clave primaria" if col==entity[:-1]+"_id" or (entity=="clientes" and col=="cliente_id") else ("Clave foránea" if col.endswith("_id") else ("Numérica" if pd.api.types.is_numeric_dtype(df[col]) else "Categórica"))
            rows.append({"Entidad":entity,"Columna":col,"Rol":role,"Confidencial":"Sí" if col in CONFIDENTIAL_FIELDS else "No","Tratamiento":"Generar ficticio" if col in CONFIDENTIAL_FIELDS else ("Generar clave primaria" if role=="Clave primaria" else ("Asignar clave foránea" if role=="Clave foránea" else "Sintetizar")),"Dominio permitido":DOMAINS.get(col,""),"Aprobado":False})
    return pd.DataFrame(rows)

def dictionary_ready():
    d=st.session_state.get("dictionary")
    return d is not None and len(d)>0 and bool(d["Aprobado"].all())

def validate_dictionary(tables,d):
    expected={(e,c) for e,frame in tables.items() for c in frame.columns}
    actual=set(zip(d["Entidad"],d["Columna"]))
    errors=[]
    if expected!=actual:errors.append("El diccionario no coincide con las columnas de la base activa.")
    if d.duplicated(["Entidad","Columna"]).any():errors.append("Hay columnas duplicadas.")
    for _,r in d.iterrows():
        intended={"Clave primaria":"Generar clave primaria","Clave foránea":"Asignar clave foránea"}
        if r["Rol"] in intended and r["Tratamiento"]!=intended[r["Rol"]]:errors.append(f"{r['Entidad']}.{r['Columna']}: tratamiento esperado {intended[r['Rol']]}")
    if not bool(d["Aprobado"].all()):errors.append("Faltan columnas por revisar y aprobar.")
    if ((d["Confidencial"]=="Sí") & (d["Tratamiento"]=="Conservar")).any():errors.append("No se puede conservar una columna confidencial en una demo pública.")
    for row in d.itertuples(index=False):
        dom=str(getattr(row,"_6","") or "") if False else str(row[d.columns.get_loc("Dominio permitido")])
        if row[d.columns.get_loc("Rol")]=="Categórica" and dom.strip():
            entity,col=row[0],row[1]
            actual_values=set(tables[entity][col].dropna().astype(str).unique())
            allowed={v.strip() for v in dom.split(",") if v.strip()}
            if not actual_values.issubset(allowed):errors.append(f"{entity}.{col}: hay valores fuera del dominio permitido")
    return errors

def training_policy(tables,dictionary):
    """Create an enforceable training projection for the present customer-only PoC."""
    errors=[]
    customer=dictionary[dictionary["Entidad"]=="clientes"].set_index("Columna")
    required=["segmento","region","nivel_riesgo","edad"]
    generated_identifiers=["nombre_cliente","direccion","email","telefono"]
    for col in required:
        if col not in customer.index:
            errors.append(f"Falta definir clientes.{col}")
            continue
        row=customer.loc[col]
        if row["Confidencial"]=="Sí":
            errors.append(f"clientes.{col} está marcada confidencial: el motor no puede entrenarla en esta PoC pública")
        if row["Tratamiento"]!="Sintetizar":
            errors.append(f"clientes.{col} necesita tratamiento Sintetizar para el motor actual")
        expected_role="Numérica" if col=="edad" else "Categórica"
        if row["Rol"]!=expected_role:
            errors.append(f"clientes.{col} debe clasificarse {expected_role}")
    for col in generated_identifiers:
        if col in customer.index:
            row=customer.loc[col]
            if row["Tratamiento"]!="Generar ficticio":
                errors.append(f"clientes.{col}: solo se admite Generar ficticio o Excluir")
            if row["Confidencial"]=="Sí" and row["Tratamiento"]!="Generar ficticio":
                errors.append(f"clientes.{col}: el esquema actual requiere un valor ficticio")
    if "cliente_id" in customer.index and customer.loc["cliente_id","Tratamiento"]!="Generar clave primaria":
        errors.append("clientes.cliente_id debe reconstruirse, no aprenderse")
    # Only the 4 business features are trained, regardless of other metadata fields.
    return errors, required

def check_synthetic_domains(synthetic,dictionary):
    errors=[]
    for _,row in dictionary.iterrows():
        if row["Rol"]!="Categórica" or not str(row["Dominio permitido"]).strip():
            continue
        ent,col=row["Entidad"],row["Columna"]
        if ent not in synthetic or col not in synthetic[ent].columns:
            continue
        allowed={x.strip() for x in str(row["Dominio permitido"]).split(",") if x.strip()}
        seen=set(synthetic[ent][col].dropna().astype(str))
        unknown=seen-allowed
        if unknown:errors.append(f"{ent}.{col}: valores generados fuera del dominio: {', '.join(sorted(unknown)[:4])}")
    return errors

def validate_counts(client_count,sales_count,invoice_count,payment_count):
    errors=[]
    if sales_count<client_count:errors.append("Para el generador actual cada cliente tiene al menos una venta.")
    if invoice_count>sales_count:errors.append("No puede haber más facturas que ventas en el esquema actual 1:0..1.")
    if payment_count>invoice_count:errors.append("El generador actual admite como máximo un pago por factura.")
    if payment_count<0:errors.append("Los pagos no pueden ser negativos.")
    return errors

if "tables" not in st.session_state:st.session_state.tables=build_data()
if "runs" not in st.session_state:st.session_state.runs=[]
if "models" not in st.session_state:st.session_state.models=[]
if "dictionary" not in st.session_state:st.session_state.dictionary=default_dictionary(st.session_state.tables)
page=st.sidebar.radio("Secciones",["Base de datos","Diccionario y privacidad","Exploración","Entrenamiento","Generación","Comparar experimentos","7 métricas QE"])
t=st.session_state.tables
if page=="Base de datos":
    st.subheader("Modelo: Clientes → Ventas → Facturas → Pagos")
    n=st.number_input("Clientes ficticios",min_value=50,max_value=2000,value=250,step=50)
    seed=st.number_input("Semilla",min_value=0,value=42)
    if st.button("Crear base de datos ficticia"):
        st.session_state.tables=build_data(int(n),int(seed));st.session_state.dictionary=default_dictionary(st.session_state.tables);st.session_state.models=[];st.session_state.runs=[];st.rerun()
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
elif page=="Diccionario y privacidad":
    st.subheader("Configurar datos por tabla")
    st.write("**1. Elige la tabla. 2. Revisa las columnas y sus valores. 3. Clasifica las columnas que quieras proteger o controlar. 4. Guarda la configuración de esa tabla.**")
    st.info("Puedes configurar una tabla a la vez. Nombre, email, dirección y teléfono aparecen sugeridos como confidenciales. No subas datos personales reales a esta demo pública.")
    entity=st.selectbox("Tabla que deseas configurar",list(t),format_func=lambda x:x.capitalize(),key="dictionary_entity")
    df=t[entity]
    st.markdown(f"**Vista previa de {entity}**")
    st.dataframe(df.head(8),use_container_width=True,hide_index=True)
    dic=st.session_state.dictionary.copy()
    scope=dic[dic["Entidad"]==entity].copy()
    scope["Ejemplos"]=[", ".join(df[c].dropna().astype(str).head(3).tolist()) for c in scope["Columna"]]
    st.markdown("**Clasifica las columnas de esta tabla**")
    st.caption("Las claves primarias se generan nuevas y únicas; las foráneas se asignan desde registros padre existentes. Ninguna se elimina del dataset sintético.")
    edited=st.data_editor(scope,hide_index=True,num_rows="fixed",use_container_width=True,disabled=["Entidad","Columna","Ejemplos"],column_config={
        "Rol":st.column_config.SelectboxColumn("Tipo de columna",options=["Clave primaria","Clave foránea","Numérica","Categórica"]),
        "Confidencial":st.column_config.SelectboxColumn("¿Es confidencial?",options=["Sí","No"]),
        "Tratamiento":st.column_config.SelectboxColumn("Qué hacer con el dato",options=["Sintetizar","Generar ficticio","Generar clave primaria","Asignar clave foránea","Excluir del modelo","Conservar"]),
        "Dominio permitido":st.column_config.TextColumn("Valores permitidos (separados por coma)"),
        "Ejemplos":st.column_config.TextColumn("Ejemplos del original"),
        "Aprobado":st.column_config.CheckboxColumn("Clasificación lista")
    },key=f"dictionary_editor_{entity}")
    if st.button(f"Guardar clasificación de {entity}",type="primary"):
        updated=edited.drop(columns=["Ejemplos"])
        candidate=dic[dic["Entidad"]!=entity]
        candidate=pd.concat([candidate,updated],ignore_index=True)
        errors=[]
        if ((updated["Confidencial"]=="Sí") & (updated["Tratamiento"]=="Conservar")).any():errors.append("Una columna confidencial no puede conservarse sin tratamiento.")
        expected={"Clave primaria":"Generar clave primaria","Clave foránea":"Asignar clave foránea"}
        for _,rr in updated.iterrows():
            if rr["Rol"] in expected and rr["Tratamiento"]!=expected[rr["Rol"]]:errors.append(f"{rr['Columna']}: el tratamiento debe ser {expected[rr['Rol']]}")
        for _,row in updated.iterrows():
            if row["Rol"]=="Categórica" and str(row["Dominio permitido"]).strip():
                allowed={z.strip() for z in str(row["Dominio permitido"]).split(",") if z.strip()}
                observed=set(df[row["Columna"]].dropna().astype(str))
                if not observed.issubset(allowed):errors.append(f"{row['Columna']}: hay valores originales fuera del dominio declarado")
        if errors:st.error("; ".join(errors))
        else:
            st.session_state.dictionary=candidate
            st.session_state.models=[]
            st.session_state.runs=[]
            st.success(f"Configuración de {entity} guardada.")
    count=int(st.session_state.dictionary.groupby("Entidad")["Aprobado"].all().sum())
    st.caption(f"Tablas completamente clasificadas: {count} de {len(t)}. La aprobación general se completa cuando todas las tablas quedan configuradas.")
    st.warning("La PoC todavía solo entrena los campos no sensibles de Clientes. Los otros campos y entidades no están aún integrados en el aprendizaje relacional del modelo. La protección de datos reales requiere controles adicionales.")
elif page=="Exploración":
    st.title("Exploración y criterios de preservación")
    st.markdown("**Objetivo:** entender el dataset original y decidir qué características deben conservar los datos sintéticos. Los criterios definidos se guardan en esta sesión para comparar los experimentos.")
    name=st.selectbox("Tabla original a analizar",list(t),key="eda_tabla")
    df=t[name]
    numeric=[c for c in df.select_dtypes(include="number").columns if not c.endswith("_id")]
    cats=[c for c in df.columns if c not in numeric and not c.endswith("_id")]
    a,b,c,d=st.columns(4)
    a.metric("Registros",len(df));b.metric("Columnas",len(df.columns))
    c.metric("Nulos",int(df.isna().sum().sum()));d.metric("Duplicados",int(df.duplicated().sum()))
    st.subheader("1. ¿Qué características queremos conservar?")
    criteria={
      "Distribuciones numéricas":"Mantener rangos, dispersión y comportamiento de montos o edades.",
      "Categorías y dominios":"Respetar códigos permitidos y conservar su variedad y frecuencia.",
      "Correlaciones":"Mantener relaciones entre variables relevantes.",
      "Casos extremos":"Conservar escenarios de borde y valores elevados válidos.",
      "Valores nulos":"Mantener patrones de ausencia de información.",
      "Integridad relacional":"Mantener relaciones Clientes → Ventas → Facturas → Pagos."
    }
    if "eda_config" not in st.session_state:st.session_state.eda_config={}
    if name not in st.session_state.eda_config:
        st.session_state.eda_config[name]={k:{"activo":True,"prioridad":"Media","observacion":""} for k in criteria}
    cfg=st.session_state.eda_config[name]
    st.caption("Activa las características relevantes y asigna prioridad. Estos criterios permiten interpretar las métricas, no modifican por sí solos el algoritmo de entrenamiento.")
    for i,(k,desc) in enumerate(criteria.items()):
        with st.expander(k,expanded=i<2):
            x,y=st.columns([1,1])
            with x:cfg[k]["activo"]=st.checkbox("Preservar",value=cfg[k]["activo"],key=f"pres_{name}_{i}")
            with y:cfg[k]["prioridad"]=st.selectbox("Prioridad",["Alta","Media","Baja"],index=["Alta","Media","Baja"].index(cfg[k]["prioridad"]),key=f"pri_{name}_{i}")
            st.caption(desc)
            cfg[k]["observacion"]=st.text_input("Criterio / comentario",value=cfg[k]["observacion"],key=f"obs_{name}_{i}",placeholder="Ej.: conservar clientes multiproducto y montos extremos")
    st.session_state.eda_config[name]=cfg
    st.subheader("2. Evidencia del dataset original")
    field=st.selectbox("Variable a explorar",numeric+cats if numeric+cats else list(df.columns),key=f"field_{name}")
    c1,c2=st.columns(2)
    with c1:
        if field in numeric:
            fig=px.histogram(df,x=field,nbins=30,marginal="box",title=f"Distribución · {field}")
        else:
            counts=df[field].value_counts(dropna=False).head(20).rename_axis("Valor").reset_index(name="Registros")
            fig=px.bar(counts,x="Valor",y="Registros",title=f"Dominio y frecuencias · {field}")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        if len(numeric)>1:
            corr=df[numeric].corr().fillna(0)
            st.plotly_chart(px.imshow(corr,text_auto=True,title="Correlaciones numéricas"),use_container_width=True)
        else:
            st.info("Esta entidad no tiene dos variables numéricas de negocio para calcular correlaciones.")
    c3,c4=st.columns(2)
    with c3:
        nulls=df.isna().sum().sort_values(ascending=False)
        st.plotly_chart(px.bar(x=nulls.index,y=nulls.values,labels={"x":"Columna","y":"Nulos"},title="Valores faltantes"),use_container_width=True)
    with c4:
        out=[]
        for col in numeric:
            v=pd.to_numeric(df[col],errors="coerce").dropna()
            if len(v)>3:
                q1,q3=v.quantile(.25),v.quantile(.75);iqr=q3-q1
                out.append({"Variable":col,"Casos extremos":int(((v<q1-1.5*iqr)|(v>q3+1.5*iqr)).sum()) if iqr>0 else 0})
        if out:st.plotly_chart(px.bar(pd.DataFrame(out),x="Variable",y="Casos extremos",title="Valores extremos (criterio IQR)"),use_container_width=True)
        else:st.info("No hay variables numéricas de negocio suficientes para detectar valores extremos.")
    st.subheader("3. Integridad entre entidades")
    st.metric("Relaciones básicas","Correctas" if verify(t) else "Revisar")
    rel=[]
    for parent,child,key in [("clientes","ventas","cliente_id"),("ventas","facturas","venta_id"),("facturas","pagos","factura_id")]:
        counts=t[child].groupby(key).size()
        rel.append({"Relación":parent+" → "+child,"Registros padre":len(t[parent]),"Registros hijo":len(t[child]),"Hijos por padre (promedio sobre padres con hijos)":round(float(counts.mean()),2) if len(counts) else 0})
    st.dataframe(pd.DataFrame(rel),use_container_width=True,hide_index=True)
    st.subheader("4. Resumen de criterios definidos")
    chosen=[{"Característica":k,"Prioridad":v["prioridad"],"Observación":v["observacion"] or "—"} for k,v in cfg.items() if v["activo"]]
    st.dataframe(pd.DataFrame(chosen) if chosen else pd.DataFrame(columns=["Característica","Prioridad","Observación"]),use_container_width=True,hide_index=True)
    if st.button("Guardar criterios de exploración",type="primary"):
        st.session_state.eda_config[name]=cfg
        st.success("Criterios guardados en esta sesión. Podrás consultarlos al interpretar las siete métricas.")
    with st.expander("Ver registros originales"):
        st.dataframe(df.head(100),use_container_width=True)
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
        if not dictionary_ready() or validate_dictionary(t,st.session_state.dictionary):
            st.error("Revisa y aprueba el diccionario de columnas en Diccionario y privacidad antes de entrenar.")
            st.stop()
        policy_errors, train_columns=training_policy(t,st.session_state.dictionary)
        if policy_errors:
            st.error("La clasificación seleccionada no es compatible con este motor: "+"; ".join(policy_errors))
            st.stop()
        source=t["clientes"][train_columns].copy()
        meta=SingleTableMetadata();meta.detect_from_dataframe(data=source)
        dims=tuple(int(v) for v in layers.split(" × "))
        with st.spinner("Entrenando TVAE (esta etapa puede tomar tiempo)..."):
            try:
                import torch
                torch.manual_seed(int(seed));np.random.seed(int(seed))
                model=TVAESynthesizer(meta,epochs=int(epochs),batch_size=int(batch),embedding_dim=int(embedding),compress_dims=dims,decompress_dims=dims,l2scale=float(l2),loss_factor=int(loss),enforce_min_max_values=minmax,enforce_rounding=rounding,enable_gpu=False)
                start=time.perf_counter();model.fit(source);duration=time.perf_counter()-start
                st.session_state.models.append({"name":experiment_name,"model":model,"train_sec":round(duration,2),"params":{"épocas":epochs,"batch":batch,"embedding":embedding,"capas":layers,"L2":l2,"loss":loss,"semilla":seed,"minmax":minmax,"rounding":rounding,"diccionario":st.session_state.dictionary.to_json(orient="records")}})
                st.success(f"Modelo '{experiment_name}' entrenado y guardado en esta sesión. Tiempo: {duration:.2f} s.")
            except Exception as exc:st.error(f"Error de entrenamiento: {exc}")
    if st.session_state.models:
        st.markdown("**Modelos entrenados disponibles**")
        st.dataframe(pd.DataFrame([{"Modelo":m["name"],"Tiempo (s)":m["train_sec"],**m["params"]} for m in st.session_state.models]),use_container_width=True)
    st.caption("Persistencia: los modelos están en memoria de sesión, no en almacenamiento externo. Para almacenamiento permanente hace falta configurar un servicio externo privado; esta versión no afirma conservarlos tras reinicios.")

elif page=="Generación":
    st.subheader("02 · Generación de datos sintéticos")
    st.markdown("Selecciona un modelo **ya entrenado**, define cuántos clientes generar y crea uno o más datasets. Puedes repetir la generación sin entrenar otra vez.")
    if not st.session_state.models:
        st.warning("No hay modelos entrenados. Primero entra a la sección Entrenamiento.")
    else:
        model_idx=st.selectbox("Modelo entrenado",list(range(len(st.session_state.models))),format_func=lambda i:f"{st.session_state.models[i]['name']} · {st.session_state.models[i]['params']['épocas']} épocas")
        qty=st.slider("Cantidad de clientes sintéticos",50,2000,250,step=50,help="Número de clientes. La cantidad de entidades dependientes debe respetar relaciones y reglas.")
        st.markdown("**Relaciones basadas en datos:** la cantidad de ventas por cliente, proporción de ventas facturadas y patrones de pagos se estiman desde las tablas originales. Las claves y reglas de montos se reconstruyen y validan.")
        estimated_min,estimated_max=qty,qty*3
        st.caption("Los volúmenes dependientes no son independientes: se determinan a partir de los patrones observados. Esta etapa utiliza muestreo empírico, no TVAE multitabla.")
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
                    generated["nombre_cliente"]=[f"Sintético {i+1:05d}" for i in range(len(generated))]
                    generated["direccion"]=[f"Avenida Inventada {i+1}, Distrito Prueba" for i in range(len(generated))]
                    generated["email"]=[f"sintetico{i+1:05d}@example.invalid" for i in range(len(generated))]
                    generated["telefono"]=[f"999-{i+1:07d}" for i in range(len(generated))]
                    synthetic=generate_children(t,generated[t["clientes"].columns.tolist()],seed=int(len(st.session_state.runs)+100))
                    relation_errors=verify_relations(synthetic)
                    if relation_errors:raise ValueError("Relaciones: "+"; ".join(relation_errors))
                    domain_errors=check_synthetic_domains(synthetic,st.session_state.dictionary)
                    if domain_errors:raise ValueError("Validación de dominios: "+"; ".join(domain_errors))
                    conf_check={c:len(set(t["clientes"][c].astype(str)) & set(synthetic["clientes"][c].astype(str))) for c in CONFIDENTIAL_FIELDS if c in t["clientes"]}
                    if any(conf_check.values()):raise ValueError(f"Coincidencias confidenciales detectadas: {conf_check}")
                    if not verify(synthetic):raise ValueError("Error de integridad referencial entre las cuatro entidades")
                    score=100*max(0,1-abs(float(t["clientes"].edad.mean())-float(generated.edad.mean()))/max(float(t["clientes"].edad.mean()),1))
                    generated_at=datetime.now().astimezone().isoformat(timespec="seconds")
                    run={"generado_el":generated_at,"nombre":batch_name,"modelo":selected["name"],"epochs":selected["params"]["épocas"],"batch":selected["params"]["batch"],"rows":qty,"train_sec":selected["train_sec"],"gen_sec":round(generation,2),"edad_media_similitud":round(score,2),"tables":synthetic,"params":selected["params"].copy(),"dataset_role":"clientes TVAE; entidades relacionadas por patrones empíricos observados y reglas contables"}
                    st.session_state.runs.append(run)
                    st.success(f"Dataset '{batch_name}' generado en {generation:.2f}s, sin reentrenar. Integridad básica: {verify(synthetic)}")
                    st.write("Fecha y hora de generación:",generated_at)
                    st.write("Entidades resultantes:",{k:len(v) for k,v in synthetic.items()})
                    st.write("Coincidencias exactas en columnas confidenciales:",conf_check)
                    st.success("Validaciones superadas: dominios categóricos, relaciones padre-hijo, coherencia factura/pagos y campos confidenciales ficticios.")
                except Exception as exc:st.error(f"No se pudo generar: {exc}")
        if st.session_state.runs:
            idx=st.selectbox("Dataset generado para inspeccionar",list(range(len(st.session_state.runs))),format_func=lambda i:f"{st.session_state.runs[i]['nombre']} · {st.session_state.runs[i]['modelo']}")
            selected_run=st.session_state.runs[idx]
            name=st.selectbox("Tabla sintética",list(selected_run["tables"]))
            st.dataframe(selected_run["tables"][name].head(30),use_container_width=True)
            st.download_button("Descargar CSV sintético",selected_run["tables"][name].to_csv(index=False),name+".csv")
            archive=io.BytesIO()
            with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
                for entity,frame in selected_run["tables"].items():
                    z.writestr(f"data/{entity}.csv",frame.to_csv(index=False))
                metadata={key:value for key,value in selected_run.items() if key not in ("tables",)}
                z.writestr("experiment.json",json.dumps(metadata,ensure_ascii=False,indent=2,default=str))
            st.download_button("Exportar experimento completo (ZIP)",archive.getvalue(),file_name=f"experimento_{idx+1}.zip",mime="application/zip",help="Permite conservar los CSV y parámetros fuera de Streamlit. No incluye los pesos entrenados del modelo.")

elif page=="Comparar experimentos":
    st.subheader("Historial en la sesión")
    if not st.session_state.runs:st.info("Primero ejecuta al menos una generación.")
    else:
        rows=[{k:v for k,v in x.items() if k!="tables"} for x in st.session_state.runs]
        df=pd.DataFrame(rows);st.dataframe(df,use_container_width=True)
        st.markdown("**Parámetros asociados a cada dataset generado**")
        st.dataframe(pd.DataFrame([{"Experimento":x["nombre"],"Modelo":x["modelo"],**x.get("params",{})} for x in st.session_state.runs]).drop(columns=["diccionario"],errors="ignore"),use_container_width=True)
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
        if name=="clientes":
            confidentiality=pd.DataFrame([{"Columna":c,"Coincidencias originales-sintéticas":len(set(real[c].astype(str))&set(synt[c].astype(str)))} for c in CONFIDENTIAL_FIELDS if c in real.columns and c in synt.columns])
            st.markdown("**Control de confidencialidad por columna**")
            st.dataframe(confidentiality,use_container_width=True,hide_index=True)
            st.caption("Un cruce exacto igual a cero es un control inicial; no constituye prueba de anonimato ni de protección frente a inferencia.")
        st.markdown("**Parámetros del modelo que produjo este dataset:**")
        st.json({k:v for k,v in st.session_state.runs[idx].get("params",{}).items() if k!="diccionario"})
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
