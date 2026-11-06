"""Session-only synthetic generation for the fixed-record mainframe page.

The relational demo is not imported or mutated. Identifiers, fillers, markers,
dates and headers are reconstructed, not learned. SDV is imported only on fit.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import random
import zipfile
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_EVEN

import pandas as pd

from mainframe_codec import (FormatError, decode_record, encode_field,
                             encode_record, parse_copybook, parse_records)

STATE_KEY = 'pyc_mainframe_lab_v1'
MAX_RUNS = 3
TREATMENTS = ('Sintetizar', 'Ficticio', 'Constante', 'Clave', 'Fecha/Hora', 'Control', 'Relleno')


def new_state() -> dict:
    return dict(schema=None, source=None, config=None, policy=None, model=None, runs=[], revision=0)


def replace_source(state: dict, schema: dict, source: dict, config: dict) -> dict:
    """Only replace the new workspace; caller must confirm before discarding it."""
    return {**new_state(), 'schema': schema, 'source': source, 'config': config,
            'revision': state.get('revision', 0) + 1}


def view_by_name(schema: dict, name: str) -> dict:
    return next(v for v in schema['views'] if v['name'] == name)


def temporal_groups(view: dict) -> list[dict]:
    groups = {}
    for f in view['fields']:
        if f['kind'] == 'display' and not f['scale'] and f.get('groups'):
            groups.setdefault(tuple(f['groups']), []).append(f)
    result = []
    for group, fields in groups.items():
        endings = {f['name'].rsplit('-', 1)[-1]: f for f in fields}
        for parts, kind in ((('AA', 'MM', 'DD'), 'date'), (('HH', 'MM', 'SS'), 'time')):
            if len(fields) == 3 and set(endings) == set(parts):
                expected = (4, 2, 2) if kind == 'date' else (2, 2, 2)
                if tuple(endings[p]['length'] for p in parts) == expected:
                    result.append(dict(name=group[-1], kind=kind,
                                       fields=[endings[p]['name'] for p in parts]))
    return result


def key_fields(schema: dict, view: dict) -> set[str]:
    a, b = schema['key_offset'], schema['key_offset'] + schema['key_length']
    result = set()
    for f in view['fields']:
        x, y = f['start'], f['start'] + f['length']
        if schema['key_length'] and max(a, x) < min(b, y):
            if x < a or y > b or f['filler']:
                raise FormatError('La clave debe coincidir con campos completos no FILLER en esta version.')
            result.add(f['name'])
    return result


def default_policy(schema: dict, view: dict) -> list[dict]:
    keys = key_fields(schema, view)
    temporal = {name for g in temporal_groups(view) for name in g['fields']}
    rows = []
    for f in view['fields']:
        if f['start'] == 0:
            treatment = 'Control'
        elif f['filler']:
            treatment = 'Relleno'
        elif f['name'] in keys:
            treatment = 'Clave'
        elif f['name'] in temporal:
            treatment = 'Fecha/Hora'
        else:
            treatment = 'Ficticio' if f['kind'] == 'text' else 'Sintetizar'
        rows.append({'Campo': f['name'], 'Inicio': f['start'] + 1, 'Bytes': f['length'],
                     'Tipo fisico': f['usage'] or f['kind'], 'Tratamiento': treatment,
                     'Confidencial': treatment in ('Ficticio', 'Clave'), 'Constante': ''})
    return rows


def validate_policy(schema: dict, view: dict, policy: list[dict], codec: str):
    defaults = {p['Campo']: p for p in default_policy(schema, view)}
    if len(policy) != len(defaults) or {p['Campo'] for p in policy} != set(defaults):
        raise FormatError('La politica no coincide con los campos del layout.')
    fields = {f['name']: f for f in view['fields']}
    for p in policy:
        name, treatment = p['Campo'], p['Tratamiento']
        fixed = defaults[name]['Tratamiento']
        if treatment not in TREATMENTS:
            raise FormatError('Tratamiento no soportado.')
        if fixed in ('Clave', 'Fecha/Hora', 'Control', 'Relleno'):
            if treatment != fixed:
                raise FormatError(f'{name}: su tratamiento estructural debe ser {fixed}.')
        elif treatment not in ('Sintetizar', 'Ficticio', 'Constante'):
            raise FormatError(f'{name}: tratamiento incompatible.')
        if p['Confidencial'] and treatment in ('Sintetizar', 'Constante', 'Fecha/Hora'):
            raise FormatError(f'{name}: un campo confidencial no puede aprenderse ni conservarse en esta demo.')
        if treatment == 'Constante':
            encode_field(str(p['Constante']), fields[name], codec)


def _temporal_ranges(rows: list[dict], view: dict) -> list[dict]:
    result = []
    for group in temporal_groups(view):
        values = []
        for index, row in enumerate(rows, 1):
            parts = [int(row[c]) for c in group['fields']]
            try:
                if group['kind'] == 'date':
                    value = date(*parts).toordinal()
                else:
                    h, m, s = parts
                    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
                        raise ValueError
                    value = h * 3600 + m * 60 + s
                values.append(value)
            except (ValueError, OverflowError) as exc:
                raise FormatError(f"Registro {index}: fecha/hora no valida en {group['name']}.") from exc
        result.append({**group, 'min': min(values), 'max': max(values)})
    return result


def _training_frame(rows, view, policy):
    fields = {f['name']: f for f in view['fields']}
    columns, domains = {}, {}
    for p in policy:
        if p['Tratamiento'] != 'Sintetizar':
            continue
        f = fields[p['Campo']]
        if f['kind'] == 'text':
            values = [r[f['name']] for r in rows]
            if len(set(values)) > 100:
                raise FormatError(f"{f['name']}: mas de 100 categorias; revise si es un identificador.")
            domains[f['name']] = sorted(set(values))
        else:
            # Integer minor units preserve cents in the logical/binary boundary.
            values = [int(Decimal(str(r[f['name']])).scaleb(f['scale'])) for r in rows]
            if max(map(abs, values), default=0) > 2**53 - 1:
                raise FormatError(f"{f['name']}: precision superior al limite de entrenamiento; use reglas.")
        columns[f['name']] = values
    if len(columns) > 30:
        raise FormatError('Maximo 30 campos aprendidos por modelo en esta demo.')
    return pd.DataFrame(columns), domains


def train_model(schema: dict, source: dict, config: dict, policy: list[dict], epochs=10, seed=42) -> dict:
    view = view_by_name(schema, config['detail_name'])
    validate_policy(schema, view, policy, config['codec'])
    rows = source['details']
    if len(rows) < 20:
        raise FormatError('Se necesitan al menos 20 detalles para entrenar esta demo.')
    if not 1 <= int(epochs) <= 200:
        raise FormatError('Epocas fuera de rango.')
    temporal = _temporal_ranges(rows, view)
    frame, domains = _training_frame(rows, view, policy)
    if frame.empty:
        raise FormatError('Seleccione al menos un campo de negocio para Sintetizar.')
    constant = {c: frame[c].iloc[0] for c in frame if frame[c].nunique(dropna=False) == 1}
    variable = frame.drop(columns=list(constant))
    # Whole source is validated; explicit cap is only applied to model training.
    sample = variable.sample(n=min(len(variable), 2000), random_state=int(seed))
    model = None
    if len(sample.columns):
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import TVAESynthesizer
        metadata = SingleTableMetadata()
        for c in sample:
            metadata.add_column(column_name=c, sdtype='categorical' if c in domains else 'numerical')
        options = dict(epochs=int(epochs), batch_size=64, embedding_dim=32,
                       compress_dims=(64, 64), decompress_dims=(64, 64),
                       enforce_min_max_values=True, enforce_rounding=True)
        # SDV >=1.18 uses cuda in older versions, enable_gpu in newer ones.
        params = inspect.signature(TVAESynthesizer.__init__).parameters
        options['enable_gpu' if 'enable_gpu' in params else 'cuda'] = False
        model = TVAESynthesizer(metadata, **options)
        model.fit(sample)
    return dict(engine=model, constant=constant, domains=domains, learned=list(frame.columns),
                policy=[dict(p) for p in policy], temporal=temporal,
                epochs=int(epochs), seed=int(seed), training_rows=len(sample),
                algorithm='TVAE + reglas' if model else 'Constantes de negocio + reglas (sin TVAE)',
                schema_fingerprint=hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest())


def _token(n: int, width: int) -> str:
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    text = ''
    while n:
        n, d = divmod(n, 36)
        text = alphabet[d] + text
    return ('T' + (text or '0').zfill(max(0, width - 1)))[-width:]


def _fake(f, n):
    if f['kind'] == 'text':
        return _token(n, f['length'])
    return Decimal(n % 10**min(f['digits'], 12)).scaleb(-f['scale'])


def _apply_time(row, group, number):
    if group['kind'] == 'date':
        d = date.fromordinal(number)
        values = (d.year, d.month, d.day)
    else:
        values = (number // 3600, number // 60 % 60, number % 60)
    for name, value in zip(group['fields'], values):
        row[name] = Decimal(value)


def header_defaults(view: dict, marker: str, process_date: date) -> dict:
    row = {f['name']: (' ' * f['length'] if f['kind'] == 'text' else Decimal(0))
           for f in view['fields']}
    row[view['fields'][0]['name']] = marker
    for group in temporal_groups(view):
        _apply_time(row, group, process_date.toordinal() if group['kind'] == 'date' else 0)
    return row


def generate(schema: dict, source: dict, config: dict, trained: dict, quantity: int,
             seed: int = 42, header_values: dict | None = None) -> dict:
    if not 1 <= int(quantity) <= 2000:
        raise FormatError('Genere de 1 a 2000 detalles por corrida.')
    current = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    if current != trained['schema_fingerprint']:
        raise FormatError('El modelo no pertenece a esta mascara.')
    view = view_by_name(schema, config['detail_name'])
    policy = trained['policy']
    validate_policy(schema, view, policy, config['codec'])
    fields = {f['name']: f for f in view['fields']}
    rng = random.Random(int(seed))
    sampled = (trained['engine'].sample(num_rows=int(quantity)) if trained['engine'] is not None
               else pd.DataFrame(index=range(int(quantity))))
    if len(sampled) != quantity:
        raise FormatError('El motor no devolvio la cantidad solicitada.')
    sampled = sampled.reset_index(drop=True)
    for name, value in trained['constant'].items():
        sampled[name] = value
    forbidden = {p['Campo']: {str(r[p['Campo']]) for r in source['details']}
                 for p in policy if p['Confidencial'] and p['Tratamiento'] in ('Ficticio', 'Clave')}
    synthetic = []
    start = rng.randrange(100_000, 9_000_000)
    for i in range(quantity):
        row = {}
        for p in policy:
            name, treatment = p['Campo'], p['Tratamiento']
            f = fields[name]
            if treatment == 'Control':
                value = config['detail_marker']
            elif treatment == 'Relleno':
                value = ' ' * f['length'] if f['kind'] == 'text' else Decimal(0)
            elif treatment == 'Fecha/Hora':
                continue
            elif treatment == 'Constante':
                value = str(p['Constante'])
            elif treatment in ('Ficticio', 'Clave'):
                value = _fake(f, start + i)
                # Compare decoded forms, including padding. No source identifier reuse.
                for attempt in range(100):
                    normalized = (value.ljust(f['length']) if f['kind'] == 'text' else str(value))
                    if name not in forbidden or normalized not in forbidden[name]:
                        break
                    value = _fake(f, start + i + (attempt + 1) * (quantity + 1))
                else:
                    raise FormatError(f'{name}: no se encontro un valor ficticio sin coincidencia exacta.')
            elif f['kind'] == 'text':
                value = sampled.at[i, name]
                if value not in trained['domains'][name]:
                    raise FormatError(f'{name}: categoria generada fuera del dominio observado.')
            else:
                try:
                    units = Decimal(str(sampled.at[i, name]))
                    if not units.is_finite():
                        raise ValueError
                    value = units.quantize(Decimal(1), rounding=ROUND_HALF_EVEN).scaleb(-f['scale'])
                except Exception as exc:
                    raise FormatError(f'{name}: resultado numerico no valido.') from exc
            row[name] = value
        for group in trained['temporal']:
            _apply_time(row, group, rng.randint(group['min'], group['max']))
        synthetic.append(row)
    blobs = [encode_record(r, view, config['codec']) for r in synthetic]
    if config.get('header_name'):
        if header_values is None:
            raise FormatError('Revise los valores de la cabecera antes de generar.')
        hv = view_by_name(schema, config['header_name'])
        if header_values[hv['fields'][0]['name']] != config['header_marker']:
            raise FormatError('Tipo de cabecera no coincide con el seleccionado.')
        # Header is created from rules/user values, never from the original record.
        _temporal_ranges([header_values], hv)
        blobs.append(encode_record(header_values, hv, config['codec']))
    if schema['key_length']:
        a, b = schema['key_offset'], schema['key_offset'] + schema['key_length']
        blobs.sort(key=lambda raw: raw[a:b])
        source_keys = {encode_record(r, view, config['codec'])[a:b] for r in source['details']}
        synthetic_keys = {raw[a:b] for raw in blobs if raw[:1] == config['detail_marker'].encode(config['codec'])}
        if source_keys & synthetic_keys:
            raise FormatError('Se detectaron claves de detalle coincidentes con el origen; cambie la semilla.')
    output = b''.join(blobs)
    checked = parse_records(output, schema, **config)
    return dict(binary=output, details=checked['details'], quantity=quantity,
                total_records=checked['record_count'], size_bytes=len(output),
                sha256=hashlib.sha256(output).hexdigest(), seed=seed,
                algorithm=trained['algorithm'], learned=trained['learned'],
                policy=[dict(p) for p in policy], epochs=trained['epochs'],
                training_rows=trained['training_rows'], codec=config['codec'],
                roundtrip=checked['roundtrip'], sorted_keys=checked['source_sorted'])


def export_zip(run: dict, schema: dict, config: dict) -> bytes:
    """Only synthetic data and configuration. No source rows/model weights."""
    stream = io.BytesIO()
    table = pd.DataFrame(run['details'])
    # CSV is for reading, not byte-accurate re-import. Prevent spreadsheet formulas.
    table = table.map(lambda x: "'" + x if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@')) else x)
    manifest = {k: v for k, v in run.items() if k not in ('binary', 'details', 'policy')}
    manifest.update(record_length=schema['record_length'], key_offset=schema['key_offset'],
                    key_length=schema['key_length'], transfer='binary; no RDW/BDW/newlines',
                    privacy_certified=False, business_validated=False)
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('mainframe_sintetico.bin', run['binary'])
        z.writestr('detalle_sintetico.csv', table.to_csv(index=False))
        z.writestr('layout.json', json.dumps(schema, ensure_ascii=False, indent=2))
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        z.writestr('politica.json', json.dumps(run['policy'], ensure_ascii=False, indent=2))
        z.writestr('LEEME.txt', (
            'PyC Smart Test Data Lab - salida de laboratorio\n'
            'Registros logicos fijos concatenados, sin RDW, BDW ni separadores.\n'
            f"Longitud: {schema['record_length']} bytes. Codificacion: {config['codec']}.\n"
            'Transferir en modo BINARIO a un entorno privado de pruebas.\n'
            'No es un cluster VSAM. La asignacion del dataset y la carga KSDS son externas.\n'
            'Verificar RECFM/LRECL, codificacion, KEYS, cabecera y reglas con el responsable z/OS.\n'
            'La validez fisica NO certifica privacidad ni aceptacion por aplicaciones COBOL.\n'
            'Identificadores ficticios no incluyen validadores bancarios ni digitos de control.\n'
            'Fechas y horas son coherentes individualmente; no se validan reglas entre campos.\n'
            'CSV es una vista de lectura: los importes del binario se codifican con Decimal.\n'
            'No se incluyen originales ni pesos del modelo.\n'))
    return stream.getvalue()


def demo_bundle():
    """Invented public fixture: shares length, not any client's layout or data."""
    text = '''01 DEMO-RECORD.
 03 DEMO-HEADER.
  05 HEADER-TYPE PIC X(1).
  05 HEADER-KEY PIC X(20).
  05 PROCESS-DATE.
   07 PROCESS-AA PIC 9(4).
   07 PROCESS-MM PIC 9(2).
   07 PROCESS-DD PIC 9(2).
  05 FILLER PIC X(571).
 03 DEMO-DETAIL REDEFINES DEMO-HEADER.
  05 RECORD-TYPE PIC X(1).
  05 CLIENT-KEY PIC X(12).
  05 SEQUENCE-NO PIC 9(8).
  05 AMOUNT PIC S9(13)V99 COMP-3.
  05 FEE PIC S9(13)V99 COMP-3.
  05 STATUS PIC X(2).
  05 EVENT-DATE.
   07 EVENT-AA PIC 9(4).
   07 EVENT-MM PIC 9(2).
   07 EVENT-DD PIC 9(2).
  05 DESCRIPTION PIC X(30).
  05 FILLER PIC X(523).
'''
    schema = parse_copybook(text, (20, 1))
    config = dict(codec='cp037', detail_name='DEMO-DETAIL', header_name='DEMO-HEADER',
                  detail_marker='D', header_marker='0')
    view = view_by_name(schema, 'DEMO-DETAIL')
    rng = random.Random(42)
    rows = []
    for i in range(200):
        d = date(2026, 1, 1) + timedelta(days=rng.randrange(90))
        rows.append({'RECORD-TYPE': 'D', 'CLIENT-KEY': f'SRC{i + 1:09d}',
                     'SEQUENCE-NO': i + 1, 'AMOUNT': Decimal(rng.randrange(100, 100000)).scaleb(-2),
                     'FEE': Decimal(rng.randrange(0, 1000)).scaleb(-2),
                     'STATUS': rng.choice(['OK', 'NO']), 'EVENT-AA': d.year,
                     'EVENT-MM': d.month, 'EVENT-DD': d.day,
                     'DESCRIPTION': 'EJEMPLO INVENTADO', 'FILLER@78': ' ' * 523})
    hv = view_by_name(schema, 'DEMO-HEADER')
    header = header_defaults(hv, '0', date(2026, 4, 1))
    payload = encode_record(header, hv, 'cp037') + b''.join(encode_record(r, view, 'cp037') for r in rows)
    return schema, payload, config, text
