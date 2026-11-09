# Biblioteca de modelos — PyC Smart Test Data Lab

Página adicional para guardar **modelos entrenados completos**, recuperarlos sin
`fit()` y eliminarlos individualmente. No es un registro de parámetros ni una
exportación de CSV. No modifica el entrenamiento, las métricas, las dependencias
ni el módulo mainframe. El usuario debe pulsar **Guardar modelos seleccionados**;
no existe autoguardado del entrenamiento en esta entrega.

## Preparar una demo

1. Configura una vez la clave privada de firma en Streamlit (ver abajo).
2. En la demo relacional, entrena los modelos que quieres mostrar.
3. Abre **Biblioteca modelos** en el menú superior de páginas.
4. Descarga tu código privado de biblioteca. No lo compartas: funciona como una
   llave de acceso a tus archivos dentro de la instancia.
5. En **Guardar entrenamientos**, selecciona los modelos, confirma que la base y
   el diccionario corresponden a esos entrenamientos y pulsa Guardar.
6. En **Recuperar o eliminar**, prepara y descarga el respaldo `.pycmodel` de cada
   modelo. Conserva los respaldos fuera de Streamlit antes de salir.
7. En otra sesión, abre la biblioteca con tu código. Selecciona Recuperar.
   Si la instancia ya no conserva sus archivos, importa el respaldo descargado.
8. Vuelve a **streamlit app → Generación** y elige el modelo recuperado. No se
   reentrena. Para comparar necesitas generar los datasets; no se incluyen las
   generaciones antiguas en este respaldo de entrenamiento.

Guardar en la biblioteca no invalida ni cambia el modelo en memoria. El código
anterior todavía limpia modelos activos al guardar el diccionario; **no elimina
los archivos de esta biblioteca**. Recupera un modelo antes de volver a usarlo.

## Persistencia: lo que está garantizado y lo que no

- Los archivos se guardan por defecto en `~/.pyc-model-library`, fuera de GitHub.
- Persiste a cambios de página y a sesiones nuevas **si el disco de la instancia
  sigue disponible**. No depende de una caché global ni de mantener una pestaña.
- Streamlit Community Cloud NO garantiza conservar ese disco, incluso si existe
  una firma estable. No es almacenamiento externo permanente.
- La copia descargada permite recuperar después de perder el disco, **con la misma
  clave de firma y las mismas versiones de librerías**. Se registra Python
  mayor/menor y versiones de SDV, CTGAN, RDT, copulas, PyTorch, NumPy, pandas,
  scikit-learn y cloudpickle. Se bloquea la carga ante diferencias de versión;
  no se intenta una migración silenciosa.
- No recupera entrenamientos fallidos, interrumpidos o ya perdidos antes de guardar.
- En un despliegue propio se puede configurar `PYC_MODEL_LIBRARY_DIR` a un volumen
  persistente, pero en Community Cloud cambiar una ruta no crea persistencia.
- No se conectó un servicio S3/Supabase ni se contrataron recursos.

## Configuración administrativa de la firma

Generar una clave aleatoria en un equipo de confianza:

```sh
python -c "import secrets; print(secrets.token_hex(32))"
```

Agregar en **Manage app → Settings → Secrets** una entrada de primer nivel:

```toml
PYC_MODEL_SIGNING_KEY = "CLAVE_HEXADECIMAL_ALEATORIA_DE_64_CARACTERES"
```

No guardar la clave en GitHub ni introducirla en un cargador público. Conservar una
copia privada. Debe mantenerse al recrear el servidor para restaurar respaldos.
Cambiar esta clave puede invalidar respaldos antiguos. La clave NO es el código
personal de biblioteca y NO debe distribuirse con los modelos.

Sin esta configuración, la biblioteca local funciona mediante una clave privada
generada en el disco de la instancia, pero exportar/importar respaldos portables
permanece deshabilitado. Una vez configurada la clave estable, se pueden exportar
los modelos locales existentes si aún están presentes. La aplicación los verifica
con la clave local anterior y los vuelve a firmar con la clave estable.

## Seguridad y referencia

SDV serializa modelos con cloudpickle. Este módulo también lo utiliza, pero nunca
abre directamente un PKL cargado por el navegador. Primero valida tamaño, formato,
firma HMAC-SHA256 de todos los bytes y versiones. Solo después deserializa.
La clave de firma se obtiene exclusivamente de configuración del administrador o
del archivo local privado; nunca de un archivo subido o campo de la interfaz.

Los respaldos están **firmados, no cifrados**: contienen el modelo, parámetros,
base original y diccionario. No son anónimos por definición. Solo se admiten en
esta demo datos ficticios y se deben conservar las copias en privado.

Cada biblioteca usa un código aleatorio de 256 bits; se guarda su hash en el nombre
de directorio, no el código. Los archivos tienen permisos 0600; las carpetas 0700.
No hay explorador público de bibliotecas. Límites: 10 modelos por biblioteca,
64 MiB por respaldo y 512 MiB totales por instancia. Guardado con reemplazo atómico.
Esto es una PoC de acceso por posesión de código, no identidad empresarial/RBAC.

Los modelos antiguos no tienen huella del origen registrada al entrenar. Exige una
confirmación del usuario y la marca como tal; no fabrica trazabilidad histórica.
Al restaurar, si base y diccionario coinciden, agrega el modelo sin borrar corridas.
Si difieren, exige confirmación para reemplazar el contexto relacional activo y
vaciar sus corridas/modelos activos. No borra bibliotecas ni el espacio mainframe.

Eliminar requiere confirmación en pantalla y elimina el archivo seleccionado y
su modelo activo en esa sesión. No elimina datasets ya generados, otros modelos,
archivos descargados ni copias activas en otras sesiones.

## Pruebas

```sh
python -m unittest discover -s pyc-smart-test-data/tests -p 'test_model_library*.py' -v
```

45 pruebas locales: 34 de almacenamiento, firmas, aislamiento, activación y pesos
PyTorch reales; 11 de recorrido UI con un doble de Streamlit. No se entrenó TVAE
real en este entorno (SDV y Streamlit no están instalados). No equivale a probar
AppTest ni el servidor Cloud. Validar allí el recorrido entrenar → guardar →
recuperar → generar con un TVAE real antes de la presentación.

Fuentes técnicas:
- https://docs.sdv.dev/sdv/modeling/single-table-synthesizers/tvaesynthesizer
- https://docs.streamlit.io/develop/concepts/connections/connecting-to-data
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- https://docs.python.org/3/library/pickle.html
