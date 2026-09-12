from pathlib import Path
import json
import random
import shutil
import tempfile
import time

import gradio as gr
import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import TVAESynthesizer


APP_TITLE = "TVAE · Generador de datos sintéticos"


def _duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inspect_csv(csv_file):
    if not csv_file:
        return "Carga un archivo CSV para comenzar.", None

    try:
        data = pd.read_csv(csv_file, low_memory=False)
        if data.empty:
            return "El archivo no contiene registros.", None
        duplicated = data.columns[data.columns.duplicated()].tolist()
        if duplicated:
            return f"Hay columnas duplicadas: {duplicated}", None

        message = (
            f"Archivo válido · {len(data):,} registros · {len(data.columns)} columnas · "
            f"{int(data.isna().sum().sum()):,} valores ausentes"
        )
        return message, data.head(10)
    except Exception as exc:
        return f"No fue posible leer el CSV: {exc}", None


def train_tvae(
    csv_file,
    epochs,
    batch_size,
    embedding_dim,
    seed,
    progress=gr.Progress(),
):
    if not csv_file:
        raise gr.Error("Primero debes cargar el archivo CSV.")

    epochs = int(epochs)
    batch_size = int(batch_size)
    embedding_dim = int(embedding_dim)
    seed = int(seed)
    if min(epochs, batch_size, embedding_dim) <= 0:
        raise gr.Error("Los parámetros del entrenamiento deben ser números positivos.")

    work_dir = Path(tempfile.mkdtemp(prefix="tvae_run_"))
    progress(0.05, desc="Leyendo el dataset completo")

    try:
        real_data = pd.read_csv(csv_file, low_memory=False)
        if real_data.empty:
            raise ValueError("El dataset está vacío.")
        if real_data.columns.duplicated().any():
            raise ValueError("El dataset contiene nombres de columnas duplicados.")

        # Evita incompatibilidades de serialización con StringDtype.
        for column in real_data.columns:
            if isinstance(real_data[column].dtype, pd.StringDtype):
                real_data[column] = real_data[column].astype(object)

        _seed_everything(seed)

        progress(0.12, desc="Detectando metadatos")
        metadata = Metadata.detect_from_dataframe(
            data=real_data,
            table_name="transacciones",
            infer_keys=None,
        )
        metadata.validate()
        metadata_path = work_dir / "metadata_tvae.json"
        metadata.save_to_json(filepath=metadata_path, mode="overwrite")

        progress(0.20, desc=f"Entrenando TVAE con {len(real_data):,} registros")
        model = TVAESynthesizer(
            metadata=metadata,
            batch_size=batch_size,
            epochs=epochs,
            embedding_dim=embedding_dim,
            enable_gpu=torch.cuda.is_available(),
            verbose=True,
        )

        training_start = time.perf_counter()
        model.fit(real_data)
        training_seconds = time.perf_counter() - training_start

        model_path = work_dir / "modelo_tvae.pkl"
        model.save(filepath=model_path)

        config = {
            "modelo": "TVAE",
            "filas_entrenamiento": len(real_data),
            "columnas": len(real_data.columns),
            "epochs": epochs,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "seed": seed,
            "gpu_disponible": torch.cuda.is_available(),
            "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "segundos_entrenamiento": round(training_seconds, 3),
        }
        config_path = work_dir / "configuracion_entrenamiento.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        status = (
            "Entrenamiento finalizado\n\n"
            f"- Entrenamiento: {_duration(training_seconds)}\n"
            f"- Registros usados: {len(real_data):,}\n"
            f"- Columnas: {len(real_data.columns):,}\n"
            f"- Dispositivo: {config['dispositivo']}\n\n"
            "El modelo quedó disponible para generar datos sin volver a entrenar."
        )
        progress(1.0, desc="Listo")
        return status, str(model_path), str(model_path), str(config_path)
    except Exception as exc:
        raise gr.Error(f"El entrenamiento se detuvo: {exc}") from exc


def generate_synthetic_data(
    trained_model_path,
    uploaded_model,
    synthetic_rows,
    progress=gr.Progress(),
):
    """Genera datos desde un modelo ya entrenado, sin repetir el entrenamiento."""
    synthetic_rows = int(synthetic_rows)
    if synthetic_rows <= 0:
        raise gr.Error("La cantidad de registros a generar debe ser mayor que cero.")

    model_path = uploaded_model or trained_model_path
    if not model_path:
        raise gr.Error(
            "Primero entrena un modelo en el paso 1 o carga un modelo TVAE guardado."
        )

    work_dir = Path(tempfile.mkdtemp(prefix="tvae_generation_"))
    try:
        progress(0.10, desc="Cargando el modelo TVAE")
        model = TVAESynthesizer.load(filepath=model_path)

        progress(0.25, desc=f"Generando {synthetic_rows:,} registros")
        generation_start = time.perf_counter()
        synthetic_data = model.sample(num_rows=synthetic_rows)
        generation_seconds = time.perf_counter() - generation_start

        progress(0.85, desc="Guardando los resultados")
        synthetic_path = work_dir / f"datos_sinteticos_TVAE_{synthetic_rows}.csv"
        synthetic_data.to_csv(synthetic_path, index=False, encoding="utf-8-sig")

        generation_config = {
            "modelo": "TVAE",
            "modelo_utilizado": Path(model_path).name,
            "filas_generadas": len(synthetic_data),
            "columnas": len(synthetic_data.columns),
            "segundos_generacion": round(generation_seconds, 3),
            "dispositivo": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            ),
        }
        (work_dir / "configuracion_generacion.json").write_text(
            json.dumps(generation_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        archive = shutil.make_archive(
            str(work_dir.parent / f"{work_dir.name}_resultados"), "zip", work_dir
        )

        status = (
            "Generación finalizada\n\n"
            f"- Registros generados: {len(synthetic_data):,}\n"
            f"- Columnas: {len(synthetic_data.columns):,}\n"
            f"- Tiempo de generación: {_duration(generation_seconds)}\n\n"
            "Puedes cambiar la cantidad y volver a generar sin reentrenar."
        )
        progress(1.0, desc="Generación lista")
        return status, synthetic_data.head(20), str(synthetic_path), archive
    except Exception as exc:
        raise gr.Error(f"La generación se detuvo: {exc}") from exc


CSS = """
:root { --accent: #16b8d4; --ink: #13283a; }
.gradio-container { max-width: 1180px !important; margin: auto !important; }
.app-title { color: var(--ink); letter-spacing: -0.02em; }
.status-box { min-height: 180px; }
.primary-btn { background: linear-gradient(90deg, #087ca7, #16b8d4) !important; }
"""


with gr.Blocks(title=APP_TITLE, css=CSS, theme=gr.themes.Soft(primary_hue="cyan")) as demo:
    gr.Markdown(
        "# TVAE · Generador de datos sintéticos\n"
        "Entrena el modelo una vez y luego genera diferentes cantidades de datos "
        "sintéticos sin repetir el entrenamiento."
    )

    trained_model_state = gr.State(value=None)

    with gr.Tab("1 · Entrenar modelo"):
        gr.Markdown(
            "### Entrenamiento\nCarga el dataset real, define los parámetros y entrena TVAE. "
            "Esta etapa no genera datos sintéticos."
        )
        with gr.Row():
            with gr.Column(scale=5):
                csv_input = gr.File(
                    label="Dataset real (.csv)", file_types=[".csv"], type="filepath"
                )
                file_status = gr.Markdown("Carga un archivo CSV para comenzar.")
                real_preview = gr.Dataframe(label="Vista previa del dataset", interactive=False)

            with gr.Column(scale=4):
                gr.Markdown("### Parámetros de entrenamiento")
                epochs = gr.Number(label="Épocas", value=300, minimum=1, precision=0)
                batch_size = gr.Dropdown(
                    label="Batch size", choices=[16, 32, 64, 128], value=32
                )
                embedding_dim = gr.Dropdown(
                    label="Embedding", choices=[64, 128, 256], value=128
                )
                seed = gr.Number(label="Semilla", value=42, minimum=0, precision=0)
                train_button = gr.Button(
                    "Entrenar modelo", variant="primary", elem_classes="primary-btn"
                )

        training_status = gr.Markdown(
            "Aún no se ha iniciado el entrenamiento.", elem_classes="status-box"
        )
        with gr.Row():
            trained_model_download = gr.File(label="Descargar modelo entrenado")
            training_config_download = gr.File(label="Descargar configuración del entrenamiento")

    with gr.Tab("2 · Generar datos"):
        gr.Markdown(
            "### Generación\nUtiliza el modelo recién entrenado o carga un archivo `.pkl`. "
            "Puedes repetir esta etapa con distintas cantidades sin reentrenar."
        )
        with gr.Row():
            with gr.Column(scale=4):
                uploaded_model = gr.File(
                    label="Modelo TVAE guardado (.pkl) · opcional",
                    file_types=[".pkl"],
                    type="filepath",
                )
                synthetic_rows = gr.Number(
                    label="Cantidad de registros sintéticos a generar",
                    value=14359,
                    minimum=1,
                    precision=0,
                )
                generate_button = gr.Button(
                    "Generar datos sintéticos",
                    variant="primary",
                    elem_classes="primary-btn",
                )
            with gr.Column(scale=5):
                generation_status = gr.Markdown(
                    "Primero entrena un modelo o carga un modelo guardado.",
                    elem_classes="status-box",
                )
                synthetic_preview = gr.Dataframe(
                    label="Vista previa de los datos sintéticos", interactive=False
                )

        with gr.Row():
            synthetic_csv_download = gr.File(label="Descargar CSV sintético")
            generation_archive_download = gr.File(
                label="Descargar resultados y configuración (.zip)"
            )

    csv_input.change(inspect_csv, inputs=csv_input, outputs=[file_status, real_preview])
    train_button.click(
        train_tvae,
        inputs=[csv_input, epochs, batch_size, embedding_dim, seed],
        outputs=[
            training_status,
            trained_model_download,
            trained_model_state,
            training_config_download,
        ],
    )
    generate_button.click(
        generate_synthetic_data,
        inputs=[trained_model_state, uploaded_model, synthetic_rows],
        outputs=[
            generation_status,
            synthetic_preview,
            synthetic_csv_download,
            generation_archive_download,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True, favicon_path="favicon.svg")
