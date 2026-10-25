# PyC Smart Test Data Lab — Streamlit
Demostración de datos sintéticos para Quality Engineering. Datos 100% ficticios.

## Despliegue
En https://share.streamlit.io, selecciona el repositorio `Exequiel-ia/tvae-web-colab`, rama `main`, archivo principal `pyc-smart-test-data/streamlit_app.py`.
El archivo `pyc-smart-test-data/requirements.txt` contiene las dependencias; en caso de que Community Cloud exija dependencias en raíz, configúralas según la detección del entrypoint.

## Capacidades
- Base SQLite ficticia con clientes, ventas, facturas y pagos.
- Consultas SQL y gráficos de exploración.
- TVAESynthesizer real para datos de clientes.
- Generación de datos sintéticos y reconstrucción relacional con reglas explícitas.
- Comparación original/sintético y métricas orientativas.

## Limitaciones
El entrenamiento sobre clientes no equivale al modelo relacional del Capstone. Los datos descendientes se generan por reglas de negocio, no por cuatro modelos entrenados. Los resultados experimentales se pierden al reiniciar Streamlit Community Cloud (SQLite local efímera). No cargar datos sensibles. Ninguna métrica implica anonimato certificado.
