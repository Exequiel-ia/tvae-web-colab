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


def generate_tvae(
    csv_file,
    epochs,
    batch_size,
    embedding_dim,
    seed,
    synthetic_rows,
    progress=gr.Progress(),
):
    if not csv_file:
        raise gr.Error("Primero debes cargar el archivo CSV.")

    epochs = int(epochs)
    batch_size = int(batch_size)
    embedding_dim = int(embedding_dim)
    seed = int(seed)
    synthetic_rows = int(synthetic_rows)

    if min(epochs, batch_size, embedding_dim) <= 0 or synthetic_rows < 0:
        raise gr.Error("Los parámetros deben ser números positivos; filas sintéticas puede ser 0.")

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

        rows_to_generate = len(real_data) if synthetic_rows == 0 else synthetic_rows
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

        progress(0.82, desc=f"Generando {rows_to_generate:,} registros sintéticos")
        generation_start = time.perf_counter()
        synthetic_data = model.sample(num_rows=rows_to_generate)
        generation_seconds = time.perf_counter() - generation_start
        synthetic_data = synthetic_data.reindex(columns=real_data.columns)

        synthetic_path = work_dir / "datos_sinteticos_TVAE.csv"
        synthetic_data.to_csv(synthetic_path, index=False, encoding="utf-8-sig")

        config = {
            "modelo": "TVAE",
            "filas_entrenamiento": len(real_data),
            "filas_generadas": len(synthetic_data),
            "columnas": len(real_data.columns),
            "epochs": epochs,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "seed": seed,
            "gpu_disponible": torch.cuda.is_available(),
            "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "segundos_entrenamiento": round(training_seconds, 3),
            "segundos_generacion": round(generation_seconds, 3),
        }
        (work_dir / "configuracion_y_tiempos.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        progress(0.94, desc="Preparando descarga")
        archive = shutil.make_archive(
            str(work_dir.parent / f"{work_dir.name}_resultados_TVAE"), "zip", work_dir
        )

        status = (
            "Ejecución finalizada\n\n"
            f"- Entrenamiento: {_duration(training_seconds)}\n"
            f"- Generación: {_duration(generation_seconds)}\n"
            f"- Registros usados: {len(real_data):,}\n"
            f"- Registros generados: {len(synthetic_data):,}\n"
            f"- Dispositivo: {config['dispositivo']}"
        )
        progress(1.0, desc="Listo")
        return status, synthetic_data.head(20), archive
    except Exception as exc:
        raise gr.Error(f"La ejecución se detuvo: {exc}") from exc


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
        "Carga el dataset, configura la ejecución y descarga los resultados. "
        "El entrenamiento utiliza siempre el archivo completo."
    )

    with gr.Row():
        with gr.Column(scale=5):
            csv_input = gr.File(label="Dataset real (.csv)", file_types=[".csv"], type="filepath")
            file_status = gr.Markdown("Carga un archivo CSV para comenzar.")
            real_preview = gr.Dataframe(label="Vista previa del dataset", interactive=False)

        with gr.Column(scale=4):
            gr.Markdown("### Parámetros del modelo")
            epochs = gr.Number(label="Épocas", value=300, minimum=1, precision=0)
            batch_size = gr.Dropdown(label="Batch size", choices=[16, 32, 64, 128], value=32)
            embedding_dim = gr.Dropdown(label="Embedding", choices=[64, 128, 256], value=128)
            seed = gr.Number(label="Semilla", value=42, minimum=0, precision=0)
            synthetic_rows = gr.Number(
                label="Registros sintéticos (0 = misma cantidad del dataset)",
                value=0,
                minimum=0,
                precision=0,
            )
            run_button = gr.Button("Entrenar y generar", variant="primary", elem_classes="primary-btn")

    gr.Markdown("### Ejecución y resultados")
    with gr.Row():
        status = gr.Markdown("Aún no se ha iniciado una ejecución.", elem_classes="status-box")
        synthetic_preview = gr.Dataframe(label="Vista previa sintética", interactive=False)
    result_file = gr.File(label="Descargar resultados completos")

    csv_input.change(inspect_csv, inputs=csv_input, outputs=[file_status, real_preview])
    run_button.click(
        generate_tvae,
        inputs=[csv_input, epochs, batch_size, embedding_dim, seed, synthetic_rows],
        outputs=[status, synthetic_preview, result_file],
    )


if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True, favicon_path="favicon.svg")
