# Archivos mainframe — extensión aditiva v1

## Integración y protección de lo existente

Nueva página nativa de Streamlit en `pages/1_Archivos_mainframe.py`. La entrada
continúa siendo `streamlit_app.py`. No se reemplaza la aplicación ni se cambia
el flujo Clientes → Ventas → Facturas → Pagos. No se modifica `relational.py`,
`requirements.txt`, el diccionario relacional ni sus gráficas.

Todo el nuevo espacio se mantiene en `st.session_state['pyc_mainframe_lab_v1']`;
los widgets usan el prefijo `mf_`. No se usan ni reinician `tables`, `dictionary`,
`models` o `runs` de la demo relacional. Es una extensión de la misma aplicación,
no otra versión ni un nuevo despliegue. El menú nativo muestra **Archivos mainframe**;
**Volver a la demo relacional** regresa a la entrada original.

## Uso inicial

1. Abrir **Archivos mainframe → Origen → Preparar ejemplo de archivos**.
2. Revisar **Diccionario y privacidad**. Los textos son ficticios/confidenciales
   por defecto. Para un código de negocio no sensible, marcarlo no confidencial
   y elegir `Sintetizar`. Aprobar la política.
3. Entrenar con 10 épocas inicialmente. Se mantiene un modelo mainframe activo.
4. Generar: revisar la cabecera, confirmar y seleccionar la cantidad de detalles.
5. Comparar distribuciones y descargar el ZIP. Los gráficos nuevos usan barras
   separadas y porcentajes con intervalos comunes, no colores superpuestos.

El ejemplo se crea en ejecución con datos inventados. No incluye nombres de
clientes, datasets, usuarios, registros ni la máscara privada aportada durante
el análisis. También permite descargar un copybook, layout JSON y binario ficticios.

## Máscara y extracto son entradas distintas

- Máscara: Excel `.xlsx` con hoja `COPY` y opcional `CLUSTER`, copybook UTF-8,
  o JSON normalizado descargado desde el módulo.
- Excel: solo COPY y CLUSTER se usan para el layout. DATA no se usa para entrenar
  ni se interpreta como binario. Quitar cualquier hoja con datos reales ANTES de
  subir el libro a una demo pública. Las fórmulas se rechazan, no se evalúan.
- Posiciones: se calculan desde COPY; POSICIONES no se reconcilia automáticamente.
- Extracto: registros lógicos concatenados de **600 bytes**, en transferencia
  binaria, o HEX puro. No acepta un volcado de control intervals VSAM, RDW, BDW,
  direcciones de hexdump ni saltos de línea entre registros binarios.
- Codificación explícita: cp037, cp500, cp1140, latin-1 o ASCII. No se autodetecta.
- Definir vistas de cabecera/detalle y sus marcadores de un byte (por defecto 0/D).
  Para `KEYS(20 1)`, longitud = 20 y desplazamiento base 0 = 1.
- Antes de cargar: validar la estructura, cada campo, claves declaradas y lectura/
  reconstrucción sin cambios. Una fila inválida bloquea la carga completa.

## Subconjunto físico admitido

Un nivel 01, grupos fijos, vistas completas de cabecera y detalle mediante
REDEFINES, PIC X, DISPLAY numérico sin signo, COMP-3 y COMP/BINARY de IBM
big-endian. COMP-3 se procesa con Decimal, dígitos y signo validados. Las
reinterpretaciones internas se enumeran como alternativas y no se aprenden como
columnas independientes. Las condiciones 88 no ocupan bytes y no se transforman
automáticamente en restricciones de negocio.

Fuera de alcance: VB/VBS/RDW/BDW, OCCURS/ODO, SIGN/overpunch, COMP-5, COPY anidados,
REDEFINES condicionados automáticamente, archivos vinculados y conexión a z/OS.
La sintaxis no admitida se rechaza: no se adivinan tipos ni se truncan campos.

## Síntesis y privacidad

TVAE recibe exclusivamente campos aprobados del detalle. Claves y confidenciales
se generan con reglas, no con muestras de sus identificadores originales. Se
rechazan coincidencias exactas en los campos declarados confidenciales generados
por reglas y en la clave de detalle completa cuando se declara KEYS. La cabecera
se crea a partir de valores nuevos revisados por el usuario, no se copia del origen.

Los importes se entrenan en unidades mínimas enteras (por ejemplo centavos) y se
codifican con Decimal; se bloquean magnitudes superiores a la precisión entera
exacta del entrenamiento. Los grupos AA/MM/DD y HH/MM/SS reconocidos se validan
como fechas/horas y se generan dentro de sus rangos observados; NO se aprenden ni
validan relaciones temporales entre grupos. No se asumen ecuaciones contables,
validadores de cuentas o dígitos de control. **Validez física no implica validez
funcional ni anonimización.** Las relaciones con un segundo archivo se dejan para
una ampliación posterior.

Los campos de negocio constantes se reconstruyen como constantes; si todos lo
son, la interfaz indica explícitamente **sin TVAE**. Los resultados se comparan
por variable, no mediante una puntuación global de calidad ni MET1–MET7 reutilizadas
sin adaptación. No hay LLM externo.

## Límites de la demo

Máximo 8 MiB por archivo, 5000 detalles de origen y una cabecera, 2000 filas para
entrenamiento, 30 campos aprendidos, 100 categorías por campo, 200 épocas, 2000
detalles generados y tres corridas en memoria. El usuario debe exportar/eliminar
corridas antes de superar ese límite. Solo un modelo mainframe activo. SDV se
importa al entrenar y se configura sin GPU usando el parámetro soportado por la
versión instalada (`enable_gpu` o `cuda`). No se agregan dependencias.

No hay persistencia en disco ni cargas de modelos pickle. No se envían originales
a GitHub ni a servicios LLM. La aplicación recibe los archivos en su servidor:
no es procesamiento local dentro del navegador. La demo pública no es un entorno
aprobado para datos reales sensibles. Reiniciar/cerrar una sesión puede perder el
trabajo; la configuración de privacidad no es una certificación.

## Salida y carga en mainframe

ZIP con secuencial binario sintético, CSV de inspección, layout normalizado,
política, manifiesto y LEEME. No incluye originales ni pesos de modelos. Cuando
se declara una clave, se ordenan los registros por sus bytes y se valida unicidad
incluyendo cabecera. La asignación de dataset, bloqueo físico y carga a KSDS
son operaciones externas en un ambiente de pruebas. El receptor debe confirmar
RECFM/LRECL, codepage, KEYS, cabecera y reglas del negocio antes de una carga.
El CSV es para lectura, no un reemplazo del archivo físico.

## Pruebas y límites de validación

```sh
python -m unittest discover -s pyc-smart-test-data/tests -v
```

20 pruebas offline con datos inventados: COMP-3 (signos, límites y precisión),
binarios, DISPLAY, bytes/longitudes, errores, HEX, máscaras y REDEFINES, Excel sin
DATA, privacidad de políticas, orden/duplicidad de claves, fechas, exportación y
separación del estado. El adaptador TVAE se prueba con un doble controlado: esas
pruebas NO sustituyen ejecutar el SDV real ni comprobar el despliegue Streamlit.

La máscara de entrada del piloto se verificó localmente: dos vistas de 600 bytes,
clave de 20 bytes desde offset 1 y seis importes COMP-3 de 8 bytes. No se incluyó
esa máscara ni sus datos en este repositorio. Pendiente validación con un extracto
binario íntegro y no sensible, entrenamiento real en el entorno desplegado y
lectura/carga en un mainframe de pruebas.

## Referencias técnicas

- Streamlit, páginas nativas y estado compartido:
  https://docs.streamlit.io/develop/concepts/multipage-apps/pages-directory
- SDV TVAE, parámetros y aprendizaje:
  https://docs.sdv.dev/sdv/modeling/single-table-synthesizers/tvaesynthesizer
- IBM COBOL, computational items:
  https://www.ibm.com/docs/en/cobol-zos/6.5.0?topic=clause-computational-items
