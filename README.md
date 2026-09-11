# Aplicación web TVAE para Google Colab

Esta aplicación permite cargar un CSV, configurar TVAE, ejecutar el entrenamiento con todos los registros y descargar los datos sintéticos, el modelo, los metadatos y los tiempos.

## Ejecución rápida en Google Colab

1. Cree un repositorio en GitHub y suba los archivos de este proyecto.
2. Abra `Ejecutar_TVAE_Web_en_Colab.ipynb` en Google Colab.
3. En Colab seleccione **Entorno de ejecución > Cambiar tipo de entorno de ejecución > T4 GPU**.
4. Reemplace el valor de `REPO_URL` por la dirección de su repositorio.
5. Ejecute las celdas en orden.
6. Abra el enlace público temporal que mostrará Gradio.
7. Desde la interfaz cargue el CSV y ejecute TVAE.

## Subir el proyecto a GitHub desde el navegador

1. Ingrese a https://github.com y seleccione **New repository**.
2. Asigne un nombre, por ejemplo `tvae-web-colab`.
3. Para evitar autenticación desde Colab, puede dejarlo público. El repositorio no contiene los datos reales.
4. Abra el repositorio, seleccione **Add file > Upload files** y cargue:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
   - `README.md`
   - `Ejecutar_TVAE_Web_en_Colab.ipynb`
5. Presione **Commit changes**.
6. Copie la URL del repositorio y úsela en el notebook de Colab.

## Parámetros

- **Épocas:** 300 reproduce la configuración principal del proyecto; use 5 para comprobar rápidamente el funcionamiento.
- **Batch size:** 32 corresponde a la configuración utilizada en el Capstone.
- **Embedding:** 128 corresponde a la configuración utilizada en el Capstone.
- **Semilla:** 42 mejora la trazabilidad de la ejecución.
- **Registros sintéticos:** 0 genera la misma cantidad de registros que tiene el dataset real.

## Resultados

La descarga contiene:

- `datos_sinteticos_TVAE.csv`
- `modelo_tvae.pkl`
- `metadata_tvae.json`
- `configuracion_y_tiempos.json`

## Importante

- El CSV no se sube a GitHub; se carga directamente en la aplicación durante la ejecución.
- La URL de Gradio es temporal y deja de funcionar cuando termina la sesión de Colab.
- La pestaña de Colab debe permanecer conectada durante el entrenamiento.
- Las métricas MET1–MET7 no se calculan en esta primera versión; corresponden a la fase de evaluación posterior.
