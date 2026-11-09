"""Private model library and authenticated SDV checkpoints.

Only server-created objects are serialized. Never load a bare pickle: authenticate
all bytes with a server-only key and verify runtime versions before deserialization.
Local files are NOT durable on Community Cloud; downloaded backups require a stable
PYC_MODEL_SIGNING_KEY configured by the administrator in Secrets before export.
"""
from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import os
import re
import secrets
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MAGIC = b'PYC_MODEL_V1\x00'
MAX_BYTES = 64 * 1024 * 1024
MAX_HEADER = 16 * 1024
MAX_MODELS = 10
MAX_TOTAL_BYTES = 512 * 1024 * 1024
STATE_KEY = 'ml_access_code'
MODEL_ID = re.compile(r'^[a-f0-9]{32}$')
ACCESS_CODE = re.compile(r'^[A-Za-z0-9_-]{43}$')
_LOCK = threading.RLock()
PACKAGES = ('sdv', 'ctgan', 'rdt', 'copulas', 'torch', 'numpy', 'pandas',
            'scikit-learn', 'cloudpickle')


class LibraryError(ValueError):
    """Safe user-facing errors; no source data or credentials in the message."""


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def versions():
    result = {'python': '.'.join(map(str, sys.version_info[:2]))}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not-installed'
    return result


def new_code():
    return secrets.token_urlsafe(32)


def valid_code(code):
    if not isinstance(code, str) or not ACCESS_CODE.fullmatch(code):
        raise LibraryError('El código de biblioteca debe ser el código completo de 43 caracteres.')
    return code


def fingerprint(tables):
    # Same convention as comparison_core.source_fingerprint.
    digest = hashlib.sha256()
    for name in sorted(tables):
        frame = tables[name]
        digest.update(json.dumps([name, list(frame.columns), [str(x) for x in frame.dtypes]],
                                 ensure_ascii=False).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def policy_signature(dictionary):
    if not isinstance(dictionary, pd.DataFrame):
        raise LibraryError('Falta el diccionario de la base original.')
    columns = sorted(dictionary.columns)
    sort = [c for c in ('Entidad', 'Columna') if c in columns]
    normalized = dictionary[columns].sort_values(sort).reset_index(drop=True) if sort else dictionary[columns]
    return normalized.to_json(orient='records', force_ascii=False)


def _validated_payload(payload):
    if not isinstance(payload, dict) or set(payload) != {'model', 'tables', 'dictionary', 'captured_at'}:
        raise LibraryError('Contenido de modelo no compatible con esta versión.')
    record, tables, dictionary = payload['model'], payload['tables'], payload['dictionary']
    if not isinstance(record, dict) or not callable(getattr(record.get('model'), 'sample', None)):
        raise LibraryError('El respaldo no contiene un modelo utilizable para generar.')
    if not isinstance(record.get('params'), dict):
        raise LibraryError('El modelo no tiene parámetros registrados.')
    if not isinstance(tables, dict) or set(tables) != {'clientes', 'ventas', 'facturas', 'pagos'}:
        raise LibraryError('El respaldo no pertenece a la demo relacional de cuatro entidades.')
    if any(not isinstance(v, pd.DataFrame) or not v.columns.is_unique or len(v) > 10000
           for v in tables.values()):
        raise LibraryError('Tablas de referencia fuera del formato o límite permitido.')
    policy_signature(dictionary)
    return payload


def capture_model(record, tables, dictionary, *, confirm_legacy=False):
    """Copy the reference/policy. Never mutate the live model or source state."""
    import io
    if not isinstance(record, dict):
        raise LibraryError('Selecciona un modelo entrenado.')
    saved_policy = record.get('params', {}).get('diccionario')
    if not isinstance(saved_policy, str):
        raise LibraryError('El modelo no registra su diccionario de entrenamiento.')
    try:
        training_dictionary = pd.read_json(io.StringIO(saved_policy), orient='records')
    except (ValueError, TypeError):
        raise LibraryError('No se puede leer el diccionario del modelo.') from None
    if policy_signature(training_dictionary) != policy_signature(dictionary):
        raise LibraryError('El diccionario activo no coincide con el del entrenamiento. No se guardó el modelo con otra política.')
    source_fp = fingerprint(tables)
    known = record.get('source_fingerprint')
    if known and known != source_fp:
        raise LibraryError('La base activa no coincide con la referencia registrada del modelo.')
    if not known and not confirm_legacy:
        raise LibraryError('Confirma que la base activa es la usada para este entrenamiento sin huella histórica.')
    copied = dict(record)
    copied['params'] = dict(record['params'])
    copied['library_id'] = record.get('library_id') or secrets.token_hex(16)
    if not MODEL_ID.fullmatch(copied['library_id']):
        raise LibraryError('Identificador de biblioteca no válido.')
    # Do NOT invent a training fingerprint or model_id for legacy records.
    copied['library_reference'] = source_fp
    copied['library_provenance'] = ('registered_at_training' if known else 'confirmed_by_user_at_save')
    payload = {'model': copied, 'tables': {k:v.copy(deep=True) for k,v in tables.items()},
               'dictionary': dictionary.copy(deep=True), 'captured_at': now()}
    return _validated_payload(payload)


def _json_object(raw):
    def no_duplicates(pairs):
        obj = {}
        for key, val in pairs:
            if key in obj:
                raise ValueError('duplicate')
            obj[key] = val
        return obj
    try:
        obj = json.loads(raw, object_pairs_hook=no_duplicates)
        if not isinstance(obj, dict):
            raise ValueError('not an object')
        return obj
    except (ValueError, UnicodeError, RecursionError):
        raise LibraryError('Cabecera de respaldo inválida.') from None


class ModelLibrary:
    def __init__(self, root: Path, signing_secret: str | None = None):
        self.root = Path(root).expanduser()
        if self.root.is_symlink():
            raise LibraryError('Ruta de biblioteca no permitida.')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.stable = bool(signing_secret)
        self.keys = []
        if signing_secret:
            # Configuration comes only from server environment/Secrets, NEVER uploads.
            if not isinstance(signing_secret, str) or not re.fullmatch(r'[a-fA-F0-9]{64,128}', signing_secret):
                raise LibraryError('PYC_MODEL_SIGNING_KEY debe contener de 64 a 128 caracteres hexadecimales aleatorios.')
            self.keys.append(bytes.fromhex(signing_secret))
        keyfile = self.root / '.local_signing_key'
        with _LOCK:
            if keyfile.is_symlink():
                raise LibraryError('Archivo de clave no permitido.')
            if not keyfile.exists():
                try:
                    fd = os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(secrets.token_bytes(32))
                        stream.flush()
                        os.fsync(stream.fileno())
                except FileExistsError:
                    pass
            local_key = keyfile.read_bytes()
        if len(local_key) != 32:
            raise LibraryError('No se pudo abrir la clave local de la biblioteca.')
        self.keys.append(local_key)
        self.key = self.keys[0]

    def _folder(self, code, create=False):
        token = valid_code(code)
        folder = self.root / hashlib.sha256(token.encode()).hexdigest()
        if folder.is_symlink():
            raise LibraryError('Ruta de biblioteca no permitida.')
        if create:
            folder.mkdir(mode=0o700, exist_ok=True)
        return folder

    def _path(self, code, identifier):
        if not isinstance(identifier, str) or not MODEL_ID.fullmatch(identifier):
            raise LibraryError('Identificador de modelo no válido.')
        path = self._folder(code) / (identifier + '.pycmodel')
        if path.is_symlink():
            raise LibraryError('Archivo de biblioteca no permitido.')
        return path

    def seal(self, payload):
        import cloudpickle
        _validated_payload(payload)
        model = payload['model']
        params = model['params']
        header = {'format': 1, 'id': model['library_id'], 'name': str(model.get('name','Modelo'))[:120],
                  'captured_at': payload['captured_at'], 'epochs': params.get('épocas'),
                  'train_sec': model.get('train_sec'), 'reference': fingerprint(payload['tables']),
                  'provenance': model['library_provenance'], 'versions': versions()}
        meta = json.dumps(header, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(meta) > MAX_HEADER:
            raise LibraryError('Los metadatos superan el límite del respaldo.')
        data = cloudpickle.dumps(payload, protocol=5)
        signed = MAGIC + len(meta).to_bytes(4,'big') + meta + data
        if len(signed) + 32 > MAX_BYTES:
            raise LibraryError('Modelo y referencia superan el límite de 64 MiB.')
        return signed + hmac.digest(self.key, signed, 'sha256')

    def inspect(self, data):
        if not isinstance(data, bytes) or not len(MAGIC)+4+32 < len(data) <= MAX_BYTES:
            raise LibraryError('Respaldo vacío, demasiado grande o de formato no admitido.')
        if not data.startswith(MAGIC):
            raise LibraryError('Solo se admiten respaldos .pycmodel firmados por esta aplicación, no archivos PKL.')
        signed, tag = data[:-32], data[-32:]
        if not any(hmac.compare_digest(hmac.digest(k, signed, 'sha256'), tag) for k in self.keys):
            raise LibraryError('Firma no válida. El archivo fue modificado o pertenece a otra clave de servidor. No se cargó.')
        start = len(MAGIC)+4
        length = int.from_bytes(data[len(MAGIC):start], 'big')
        if not 0 < length <= MAX_HEADER or start + length >= len(signed):
            raise LibraryError('Longitud de cabecera inválida.')
        header = _json_object(signed[start:start+length])
        if header.get('format') != 1 or not MODEL_ID.fullmatch(str(header.get('id',''))):
            raise LibraryError('Versión o identificador de respaldo no compatible.')
        if not isinstance(header.get('versions'), dict):
            raise LibraryError('El respaldo no registra versiones de las librerías.')
        return header, signed[start+length:]

    def decode(self, data):
        header, payload_bytes = self.inspect(data)
        runtime = versions()
        changed = [k for k,v in runtime.items() if header['versions'].get(k) != v]
        if changed:
            raise LibraryError('Versiones diferentes: '+', '.join(changed)+'. Usa el mismo entorno del respaldo; no se deserializó el modelo.')
        import cloudpickle
        # Authentication and version checks above are mandatory before this call.
        try:
            payload = _validated_payload(cloudpickle.loads(payload_bytes))
        except LibraryError:
            raise
        except Exception:
            raise LibraryError('No se pudo recuperar el modelo. La sesión activa no se modificó.') from None
        if payload['model'].get('library_id') != header['id'] or fingerprint(payload['tables']) != header.get('reference'):
            raise LibraryError('El respaldo no coincide con su referencia declarada.')
        return payload

    def save(self, code, payload):
        valid_code(code)
        data = self.seal(payload)
        identifier = payload['model']['library_id']
        self._folder(code, create=True)
        path = self._path(code, identifier)
        with _LOCK:
            existing = list(path.parent.glob('*.pycmodel'))
            if not path.exists() and len(existing) >= MAX_MODELS:
                raise LibraryError('Límite de 10 modelos guardados por biblioteca. Respalda y elimina uno antes de continuar.')
            size = sum(p.stat().st_size for p in self.root.glob('*/*.pycmodel') if p.is_file())
            old_size = path.stat().st_size if path.exists() else 0
            if size-old_size+len(data) > MAX_TOTAL_BYTES:
                raise LibraryError('Se alcanzó el límite de almacenamiento local. No se borró ningún modelo.')
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.write_')
            try:
                with os.fdopen(fd,'wb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        return identifier

    def list(self, code):
        folder = self._folder(code)
        result = []
        if folder.exists():
            for path in sorted(folder.glob('*.pycmodel')):
                if path.is_symlink() or path.stat().st_size > MAX_BYTES:
                    raise LibraryError('Archivo de biblioteca fuera de límite.')
                header, _ = self.inspect(path.read_bytes())
                if path.stem != header['id']:
                    raise LibraryError('Identificador de archivo inconsistente.')
                result.append({**header, 'bytes': path.stat().st_size})
        return result

    def read_bytes(self, code, identifier):
        path = self._path(code, identifier)
        if not path.exists() or path.stat().st_size > MAX_BYTES:
            raise LibraryError('El modelo no está disponible en esta biblioteca local.')
        data = path.read_bytes()
        self.inspect(data)
        return data

    def load(self, code, identifier):
        return self.decode(self.read_bytes(code, identifier))

    def backup(self, code, identifier):
        if not self.stable:
            raise LibraryError('Antes de descargar un respaldo, configura PYC_MODEL_SIGNING_KEY en Secrets. Así podrá verificarse tras recrear el servidor.')
        # Re-sign local checkpoints using the stable key, after trusted validation.
        return self.seal(self.load(code, identifier))

    def import_backup(self, code, data):
        if not self.stable:
            raise LibraryError('Configura la misma PYC_MODEL_SIGNING_KEY usada al exportar.')
        return self.save(code, self.decode(data))

    def delete(self, code, identifier):
        with _LOCK:
            path = self._path(code, identifier)
            if not path.exists():
                raise LibraryError('El modelo ya no está en la biblioteca.')
            path.unlink()


def activate(state, payload, *, replace_reference=False):
    """No .fit(). Change only relational keys, after all checks are complete."""
    _validated_payload(payload)
    tables, dictionary, record = payload['tables'], payload['dictionary'], payload['model']
    matches = (bool(state.get('tables')) and fingerprint(state['tables']) == fingerprint(tables)
               and 'dictionary' in state and policy_signature(state['dictionary']) == policy_signature(dictionary))
    if not matches and not replace_reference:
        raise LibraryError('La referencia o el diccionario activo son distintos. Confirma restaurar el contexto guardado.')
    models = list(state.get('models', [])) if matches else []
    if not any(m.get('library_id') == record['library_id'] for m in models):
        models.append(record)
    if not matches:
        # Call only from the separate library page, before relational widgets render.
        for key in list(state):
            if str(key).startswith(('dictionary_editor_', 'pres_', 'pri_', 'obs_', 'field_')):
                del state[key]
        state['tables'] = {k:v.copy(deep=True) for k,v in tables.items()}
        state['dictionary'] = dictionary.copy(deep=True)
        state['runs'] = []
        state.pop('eda_config', None)
    state['models'] = models
    return matches


def remove_active(state, identifier):
    # Generation datasets stay available to the comparator.
    state['models'] = [m for m in state.get('models',[]) if m.get('library_id') != identifier]
