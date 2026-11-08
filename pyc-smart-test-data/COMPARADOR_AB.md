# Comparador A/B — PyC Smart Test Data Lab

Pantalla adicional de **solo lectura** para las generaciones de la demo relacional.
No sustituye `Comparar experimentos` ni `7 métricas QE`, no entrena modelos y no
interviene en el módulo mainframe. No se cambian las dependencias.

## Uso

1. En la demo relacional, entrenar uno o varios modelos.
2. En Generación, producir al menos dos datasets en la misma sesión, con el mismo
   modelo o con modelos diferentes. Entrenar dos modelos no crea sus datasets.
3. Abrir **Comparador AB** en el menú de páginas de Streamlit (el título interior
   es **Comparador A/B**). Elegir A, B y la entidad.
4. Revisar **Resumen**, **Parámetros**, **Métricas** y **Datos**.

A es la referencia de la comparación y B el candidato, no el original. Las dos
puntuaciones se calculan contra el **original activo de la sesión**. Ninguna
acción de esta pantalla borra ni modifica tablas, diccionarios, modelos o corridas.
Si no hay dos generaciones, muestra instrucciones; no inventa resultados de ejemplo.
El historial se mantiene únicamente en la sesión y puede perderse tras reiniciar.

## Qué muestra

- Tarjetas con generación, nombre del modelo, registros, épocas y fecha disponible.
- Diferencias de entrenamiento separadas de tamaño de generación, tiempos y política.
- MET1–MET7 para A y B, diferencia B menos A en puntos y lectura Sube/Baja/Sin cambio.
- Sin promedio global, ganador automático ni atribución causal a un parámetro.
- Gráficos agrupados: Original gris, A azul PyC y B turquesa.
- Distribuciones numéricas con los mismos intervalos y porcentajes por conjunto.
- Distribuciones categóricas con nulos y hasta 15 categorías, más Otras.
- Resúmenes numéricos, estructura/tipos/nulos y casos extremos con límites del original.
- Registros filtrables por conjunto y columnas; CSV y resumen comparativo JSON.
- Exe-IA contextual por reglas locales. No hay llamadas a un LLM ni envíos de datos.

## Precauciones de interpretación

Las corridas actuales no guardan `source_fingerprint` ni `model_id` estables.
Por eso el comparador **no certifica retrospectivamente su procedencia** ni afirma
que dos modelos con igual nombre sean el mismo objeto entrenado. Muestra una
advertencia explícita y califica la lectura como exploratoria. Se utiliza el origen
activo, sin inventar un snapshot histórico ni añadir metadatos a las corridas.
Si en el futuro una corrida lleva `source_fingerprint` y no coincide con el origen
activo, el comparador bloquea el cálculo contra esa referencia.

La semilla de generación no registrada se muestra como tal; no se confunde con
la semilla de entrenamiento. Diferentes tamaños de muestra pueden afectar a
las métricas aunque los histogramas se normalicen. No hay pruebas de significancia.

MET1 exige al menos dos variables numéricas comparables. MET2 compara cobertura,
no frecuencias, e incluye los textos según la fórmula existente. MET3 no es un
validador completo de integridad referencial. MET4 incluye textos ficticios y no
certifica privacidad. MET5 usa media ± 3 desviaciones del original, no IQR.
MET6 puede ser baja en campos constantes por la fórmula existente y no demuestra
utilidad funcional. No aplica/no calculable se presenta sin transformarlo en cero.
Los textos de Exploración no se convierten en pesos de una nota global.

En Datos se puede elegir IQR para inspección adicional; si IQR es cero o faltan
valores, se indica que no hay evaluación, no se inventa un conteo cero. Los gráficos
numéricos excluyen nulos, infinitos y texto no numérico y reportan su cantidad.
Filtrar la tabla de registros no recalcula ni altera métricas o distribuciones.
No hay correspondencia persona-a-persona ni fila-a-fila entre Original, A y B.

## Conservación de las fórmulas y aislamiento

`comparison_core.assess_seven` es una copia literal de la función existente de
`streamlit_app.py` (blob `74b0b447399b7a7827da5a26b9fce01b5641f55c`). La página verifica
el árbol sintáctico de ambas funciones y las descripciones **antes de puntuar**.
Si divergen o no puede leer la fuente, se detiene, en vez de mezclar versiones.
No se ejecuta código del archivo leído ni se importa la app original o SDV.
La duplicación explícita evita editar la app existente en esta entrega aditiva;
una centralización futura requiere migración y pruebas propias.

Todo el estado de widgets empieza por `ab_`. Solo se leen `tables` y `runs`.
La nueva página no escribe archivos de datos en servidor, no usa caché global de
datasets, ni publica datos en GitHub. Las descargas ocurren bajo acción del usuario;
los textos con prefijos de fórmula se neutralizan en CSV, no en memoria de origen.
Nombres introducidos por el usuario se escapan en las tarjetas HTML.

## Validación

```sh
python -m unittest discover -s pyc-smart-test-data/tests -p 'test_comparison.py' -v
```

Las pruebas cubren cálculo y dirección de diferencias, N/A, columnas faltantes,
proveniencia, parámetros, histogramas de tamaños desiguales, nulos, extremos,
CSV y no mutación del estado. La prueba de equivalencia necesita el archivo
original `streamlit_app.py` al lado del módulo.

Las pruebas de recorrido UI usan un **stub de Streamlit**, no AppTest ni una sesión
real. También se revisó el HTML/CSS y Plotly en Chromium con tamaños de escritorio
y tablet; eso no sustituye validar la integración en Streamlit Cloud.
No se ejecutó entrenamiento TVAE nuevo ni se ensayó un despliegue z/OS.
La prueba final en Cloud debe confirmar navegación desde y hacia la demo relacional,
selección de dos generaciones reales, ambas entidades con/sin MET1 aplicable,
filtros, descargas y conservación del espacio mainframe.
