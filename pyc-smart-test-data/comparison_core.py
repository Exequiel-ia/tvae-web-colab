"""Read-only A/B comparison for the relational demo; no training or persistence.

assess_seven is a verbatim mirror of the legacy function (blob 74b0b447).
The UI verifies its AST against streamlit_app.py before using it. This keeps
existing screens unchanged and prevents silent divergence of metric formulas.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

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

METRIC_NAMES = {
    'MET1': 'Fidelidad estadística', 'MET2': 'Diversidad categórica',
    'MET3': 'Coherencia estructural', 'MET4': 'Ausencia de copia exacta',
    'MET5': 'Comportamiento de valores extremos', 'MET6': 'Cobertura potencial para QA',
    'MET7': 'Compatibilidad técnica',
}
METRIC_NOTES = {
    'MET1': 'Compara matrices de correlación numérica, no toda la distribución. Con menos de dos variables numéricas comparables no aplica.',
    'MET2': 'Compara conjuntos de categorías: cobertura y adherencia, NO sus frecuencias. La fórmula existente también incluye textos ficticios; se conserva sin cambios.',
    'MET3': 'Combina proporción de columnas comunes y similitud de nulos. No verifica por sí sola claves foráneas ni reglas contables.',
    'MET4': 'Cuenta registros sin coincidencia exacta, excluyendo claves *_id. Incluye los textos ficticios; un 100 no demuestra anonimato ni ausencia de memorización de otros campos.',
    'MET5': 'Compara tasas fuera de media ± 3 desviaciones estándar del ORIGINAL. No es la regla IQR de Exploración. Más extremos no equivale a mejor resultado.',
    'MET6': 'Combina solapamiento de rangos y valores dentro del rango original. No prueba utilidad funcional. Una variable constante puede tener una puntuación baja por la fórmula existente.',
    'MET7': 'Compara tipos físicos de columnas. No certifica compatibilidad con un sistema externo.',
}
PARAMETERS = {
    'épocas': 'Épocas', 'batch': 'Tamaño de lote', 'embedding': 'Dimensión latente',
    'capas': 'Capas encoder/decoder', 'L2': 'Regularización L2',
    'loss': 'Peso de reconstrucción', 'semilla': 'Semilla de entrenamiento',
    'minmax': 'Respetar mínimo/máximo', 'rounding': 'Conservar redondeo',
}
SENSITIVE = {'nombre_cliente', 'direccion', 'email', 'telefono'}


def metric_signature(source: str) -> str:
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'assess_seven')
    return ast.dump(fn, include_attributes=False)


def metrics_match(app_path: Path | None = None) -> bool:
    """Inspect trusted local source; never execute it, import SDV, or run the app."""
    path = app_path or Path(__file__).with_name('streamlit_app.py')
    try:
        text = path.read_text(encoding='utf-8')
        if metric_signature(text) != metric_signature(inspect.getsource(assess_seven)):
            return False
        node = next(n.value for n in ast.parse(text).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'METRIC_DESCRIPTIONS' for t in n.targets))
        return ast.literal_eval(node) == METRIC_DESCRIPTIONS
    except (OSError, SyntaxError, StopIteration, ValueError):
        return False


def source_fingerprint(tables: Mapping[str, pd.DataFrame]) -> str:
    """Fingerprint the ACTIVE reference only. Never retrofit provenance to runs."""
    digest = hashlib.sha256()
    for name in sorted(tables):
        frame = tables[name]
        digest.update(json.dumps([name, list(frame.columns), [str(x) for x in frame.dtypes]],
                                 ensure_ascii=False).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def canonical(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (list, dict)):
                value = parsed
        except (ValueError, TypeError):
            pass
    if isinstance(value, list) and all(isinstance(x, dict) and 'Columna' in x for x in value):
        value = sorted(value, key=lambda x: (str(x.get('Entidad')), str(x.get('Columna'))))
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def display_value(value):
    if value is None:
        return 'No registrado'
    if isinstance(value, bool):
        return 'Sí' if value else 'No'
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def parameter_table(a: dict, b: dict) -> pd.DataFrame:
    rows = []
    pa, pb = a.get('params', {}), b.get('params', {})
    keys = list(PARAMETERS) + sorted(set(pa).union(pb) - set(PARAMETERS) - {'diccionario'})
    for key in keys:
        va, vb = pa.get(key), pb.get(key)
        state = 'No registrado' if va is None or vb is None else ('Igual' if canonical(va) == canonical(vb) else 'Cambió')
        rows.append({'Grupo': 'Entrenamiento', 'Parámetro': PARAMETERS.get(key, key),
                     'A': display_value(va), 'B': display_value(vb), 'Estado': state})
    for group, label, key in [('Generación', 'Clientes solicitados', 'rows'),
                               ('Generación', 'Semilla de generación', 'generation_seed'),
                               ('Ejecución', 'Entrenamiento (s)', 'train_sec'),
                               ('Ejecución', 'Generación (s)', 'gen_sec')]:
        va, vb = a.get(key), b.get(key)
        rows.append({'Grupo': group, 'Parámetro': label, 'A': display_value(va), 'B': display_value(vb),
                     'Estado': 'No registrado' if va is None or vb is None else ('Igual' if canonical(va) == canonical(vb) else 'Cambió')})
    da, db = pa.get('diccionario'), pb.get('diccionario')
    state = 'No registrado' if da is None or db is None else ('Igual' if canonical(da) == canonical(db) else 'Cambió')
    rows.append({'Grupo': 'Política', 'Parámetro': 'Diccionario de entrenamiento',
                 'A': 'Registrado' if da is not None else 'No registrado',
                 'B': 'Registrado' if db is not None else 'No registrado', 'Estado': state})
    return pd.DataFrame(rows)


def provenance(a: dict, b: dict, current: str) -> tuple[str, str]:
    fa, fb = a.get('source_fingerprint'), b.get('source_fingerprint')
    if any(f is not None and f != current for f in (fa, fb)):
        return 'blocked', 'La procedencia registrada no coincide con el original activo. No se calcularán puntuaciones ni diferencias contra esa referencia.'
    if not fa or not fb:
        return 'unverified', 'Procedencia histórica no verificada: estas corridas no guardan una huella de su origen. La comparación es exploratoria y recalcula ambas contra el original activo de esta sesión.'
    return 'verified', 'Ambas generaciones registran la misma huella del original activo.'


def model_relation(a: dict, b: dict) -> str:
    ia, ib = a.get('model_id'), b.get('model_id')
    if ia and ib:
        return 'Dos generaciones del mismo modelo' if ia == ib else 'Generaciones de modelos distintos'
    if a.get('modelo') != b.get('modelo'):
        return 'Generaciones con nombres de modelo distintos'
    return 'Mismo nombre de modelo; identidad no verificada (sin ID registrado)'


def safe_scores(original: pd.DataFrame, synthetic: pd.DataFrame,
                evaluator: Callable = assess_seven) -> tuple[dict, str]:
    if original.empty or synthetic.empty:
        return {k: np.nan for k in METRIC_NAMES}, 'Sin registros suficientes.'
    if not original.columns.is_unique or not synthetic.columns.is_unique:
        return {k: np.nan for k in METRIC_NAMES}, 'Columnas duplicadas: no se evalúa.'
    if any(c not in synthetic for c in original):
        return {k: np.nan for k in METRIC_NAMES}, 'Faltan columnas originales; la fórmula actual requiere todas. Revisa estructura.'
    try:
        scores = evaluator(original, synthetic)
        scores = {k: float(v) if v is not None and np.isfinite(v) else np.nan for k, v in scores.items()}
        return scores, ''
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return {k: np.nan for k in METRIC_NAMES}, 'No se pudo calcular con la fórmula existente; revisa tipos y estructura.'


def compare_metrics(original, a, b) -> tuple[pd.DataFrame, list[str]]:
    sa, wa = safe_scores(original, a)
    sb, wb = safe_scores(original, b)
    rows = []
    for key, label in METRIC_NAMES.items():
        va, vb = sa[key], sb[key]
        delta = round(vb - va, 2) if np.isfinite(va) and np.isfinite(vb) else np.nan
        state = 'No comparable' if pd.isna(delta) else ('Sube' if delta > 0 else 'Baja' if delta < 0 else 'Sin cambio')
        rows.append({'Métrica': key, 'Dimensión': label, 'A': round(va, 2), 'B': round(vb, 2),
                     'Δ B − A (puntos)': delta, 'Lectura': state})
    return pd.DataFrame(rows), [f'{label}: {warning}' for label, warning in [('A', wa), ('B', wb)] if warning]


def numeric_values(series):
    values = pd.to_numeric(series, errors='coerce')
    return values[np.isfinite(values)].astype(float)


def numeric_distribution(frames, column, bins=10):
    if not 2 <= int(bins) <= 40:
        raise ValueError('Intervalos fuera de rango.')
    valid = {label: numeric_values(df[column]) for label, df in frames.items()}
    pool = pd.concat(list(valid.values()), ignore_index=True)
    if pool.empty:
        return pd.DataFrame(columns=['Conjunto', 'Intervalo', 'Cantidad', 'Porcentaje'])
    edges = np.histogram_bin_edges(pool, bins=int(bins))
    rows = []
    for label, values in valid.items():
        counts, _ = np.histogram(values, bins=edges)
        for i, count in enumerate(counts):
            interval = f'{i+1:02d} · [{edges[i]:.4g}, {edges[i+1]:.4g}' + (']' if i == len(counts)-1 else ')')
            rows.append({'Conjunto': label, 'Intervalo': interval, 'Cantidad': int(count),
                         'Porcentaje': 100 * int(count) / len(values) if len(values) else 0.0})
    return pd.DataFrame(rows)


def categorical_distribution(frames, column, limit=15):
    def token(value):
        return '(Nulo)' if pd.isna(value) else 'Valor: ' + str(value)
    counts = {label: df[column].map(token).value_counts(dropna=False) for label, df in frames.items()}
    # Rank by proportions, not counts, so a bigger dataset cannot dominate the selection.
    mass = pd.concat([s / max(s.sum(), 1) for s in counts.values()], axis=1).fillna(0).sum(axis=1)
    categories = sorted(mass.index, key=lambda x: (-mass[x], x))
    top = categories[:limit]
    rows = []
    for label, series in counts.items():
        n = int(series.sum())
        for cat in top:
            c = int(series.get(cat, 0))
            rows.append({'Conjunto': label, 'Categoría': cat, 'Cantidad': c, 'Porcentaje': 100*c/n if n else 0.0})
        if len(categories) > limit:
            c = int(series.drop(top, errors='ignore').sum())
            rows.append({'Conjunto': label, 'Categoría': 'Otras categorías (agrupadas)', 'Cantidad': c, 'Porcentaje': 100*c/n if n else 0.0})
    return pd.DataFrame(rows, columns=['Conjunto', 'Categoría', 'Cantidad', 'Porcentaje'])


def numeric_summary(frames, column):
    rows = []
    for label, df in frames.items():
        values = numeric_values(df[column])
        rows.append({'Conjunto': label, 'Registros': len(df), 'Válidos': len(values),
                     'Nulos': int(df[column].isna().sum()), 'No numéricos / infinitos': int(len(df)-len(values)-df[column].isna().sum()),
                     'Promedio': values.mean(), 'Mediana': values.median(), 'Desv. estándar': values.std(),
                     'Mínimo': values.min(), 'Máximo': values.max()})
    return pd.DataFrame(rows)


def outlier_evidence(frames, column, method='sigma'):
    original = numeric_values(frames['Original'][column])
    if len(original) < 2:
        return None, pd.DataFrame(), {}
    if method == 'iqr':
        q1, q3 = original.quantile([.25, .75])
        span = q3-q1
        if span == 0:
            return None, pd.DataFrame(), {}
        low, high = q1-1.5*span, q3+1.5*span
    elif method == 'sigma':
        mean, sd = original.mean(), original.std()
        low, high = mean-3*sd, mean+3*sd
    else:
        raise ValueError('Método desconocido.')
    rows, indices = [], {}
    for label, df in frames.items():
        values = pd.to_numeric(df[column], errors='coerce')
        valid = np.isfinite(values)
        mask = valid & ((values < low) | (values > high))
        indices[label] = mask
        n, c = int(valid.sum()), int(mask.sum())
        rows.append({'Conjunto': label, 'Evaluados': n, 'Casos extremos': c, '% extremos': 100*c/n if n else np.nan,
                     'No evaluados': len(df)-n})
    return (float(low), float(high)), pd.DataFrame(rows), indices


def schema_evidence(frames):
    columns = list(dict.fromkeys(c for df in frames.values() for c in df.columns))
    rows = []
    for c in columns:
        row = {'Columna': c}
        for label, frame in frames.items():
            row[f'{label} · tipo'] = str(frame[c].dtype) if c in frame else 'Ausente'
            row[f'{label} · nulos %'] = 100*frame[c].isna().mean() if c in frame and len(frame) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def business_columns(frame):
    return [c for c in frame if not c.endswith('_id') and c not in SENSITIVE]


def safe_csv(frame):
    """Neutralize spreadsheet-formula text; do not change numeric values."""
    clean = frame.copy(deep=True)
    def safe(v):
        if isinstance(v, str) and v.lstrip(' ').startswith(('=', '+', '-', '@', '\t', '\r', '\n')):
            return "'" + v
        return v
    for c in clean:
        if not pd.api.types.is_numeric_dtype(clean[c]):
            clean[c] = clean[c].map(safe)
    return clean.to_csv(index=False).encode('utf-8-sig')
