"""Fixed-record adapter. No network, code execution, or source-data persistence.

Supported subset: one COBOL 01, fixed groups/REDEFINES, PIC X/9/S9,
DISPLAY (unsigned), COMP-3 and IBM big-endian COMP/BINARY. Unsupported
constructs fail closed; nested REDEFINES remain alternate views, not columns.
"""
from __future__ import annotations

import io
import json
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from xml.etree import ElementTree as ET

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ROWS = 5000
CODECS = ('cp037', 'cp500', 'cp1140', 'latin-1', 'ascii')


class FormatError(ValueError):
    """A safe, bounded message; never includes source record contents."""


def _xml(data: bytes):
    # ElementTree never resolves external references; disallow DTDs as well.
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise FormatError('XML con DTD/entidades no permitido.')
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise FormatError('XML de la mascara no valido.') from exc


def xlsx_mask(data: bytes) -> tuple[str, tuple[int, int] | None]:
    """Read COPY and optional CLUSTER only. DATA is never used as a dataset.

    Read sharedStrings only for cell indexes used by those two sheets.
    Formula cells are rejected. ZIP members are never extracted to disk.
    """
    if len(data) > MAX_FILE_BYTES:
        raise FormatError('La mascara supera el limite de 8 MiB.')
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rel_ns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
            if (len(infos) > 400 or len({i.filename for i in infos}) != len(infos)
                    or sum(i.file_size for i in infos) > 40 * 1024 * 1024):
                raise FormatError('Excel demasiado grande o estructura ZIP ambigua.')
            if any(i.flag_bits & 1 for i in infos):
                raise FormatError('Excel cifrado no admitido.')
            book = _xml(z.read('xl/workbook.xml'))
            relations = _xml(z.read('xl/_rels/workbook.xml.rels'))
            targets = {}
            for item in relations:
                if item.attrib.get('TargetMode') == 'External':
                    continue
                target = item.attrib.get('Target', '')
                path = posixpath.normpath(target.lstrip('/') if target.startswith('/') else 'xl/' + target)
                if path.startswith('xl/worksheets/'):
                    targets[item.attrib['Id']] = path
            cells = {}
            wanted = set()
            for sheet in book.findall('m:sheets/m:sheet', ns):
                name = sheet.attrib.get('name', '').upper()
                if name not in ('COPY', 'CLUSTER'):
                    continue
                rid = sheet.attrib.get('{' + rel_ns + '}id')
                if rid not in targets:
                    raise FormatError('La hoja de mascara no tiene una ruta local valida.')
                root = _xml(z.read(targets[rid]))
                entries = []
                for row in root.findall('m:sheetData/m:row', ns):
                    for c in row.findall('m:c', ns):
                        if c.find('m:f', ns) is not None:
                            raise FormatError('No se admiten formulas en COPY/CLUSTER.')
                        kind = c.attrib.get('t')
                        if kind == 's':
                            v = c.findtext('m:v', '', ns)
                            idx = int(v)
                            wanted.add(idx)
                            entries.append(('s', idx))
                        elif kind == 'inlineStr':
                            entries.append(('t', ''.join(t.text or '' for t in c.findall('.//m:t', ns))))
                        elif kind == 'str':
                            entries.append(('t', c.findtext('m:v', '', ns)))
                cells[name] = entries
            if 'COPY' not in cells:
                raise FormatError('El Excel debe contener una hoja COPY con el copybook.')
            shared = {}
            if wanted:
                payload = z.read('xl/sharedStrings.xml')
                # DTD rejection before streaming; unreferenced strings not retained.
                if b'<!DOCTYPE' in payload.upper() or b'<!ENTITY' in payload.upper():
                    raise FormatError('XML con DTD/entidades no permitido.')
                idx = 0
                for _, elem in ET.iterparse(io.BytesIO(payload), events=('end',)):
                    if elem.tag == '{' + ns['m'] + '}si':
                        if idx in wanted:
                            shared[idx] = ''.join(t.text or '' for t in elem.iter('{' + ns['m'] + '}t'))
                        idx += 1
                        elem.clear()
                if not wanted.issubset(shared):
                    raise FormatError('Referencias de texto incompletas en el Excel.')
            texts = {name: '\n'.join(shared[v] if k == 's' else v for k, v in rows)
                     for name, rows in cells.items()}
            match = re.search(r'KEYS\s*\(\s*(\d+)\s*[, ]\s*(\d+)\s*\)', texts.get('CLUSTER', ''), re.I)
            key = (int(match[1]), int(match[2])) if match else None
            return texts['COPY'], key
    except FormatError:
        raise
    except (zipfile.BadZipFile, KeyError, ValueError, ET.ParseError) as exc:
        raise FormatError('No se pudo interpretar la mascara Excel.') from exc


@dataclass
class _Node:
    level: int
    name: str
    picture: str = ''
    usage: str = ''
    redefines: str = ''
    children: list = field(default_factory=list)
    offset: int = 0
    length: int = 0


def picture_info(picture: str, usage: str = '') -> dict:
    picture = picture.upper()
    usage = usage.upper()
    if len(picture) > 100:
        raise FormatError('PICTURE demasiado extenso.')
    def expand(match):
        count = int(match[2])
        if not 1 <= count <= 32760:
            raise FormatError('Repeticion PICTURE fuera de limite.')
        return match[1] * count
    expanded = re.sub(r'([X9])\((\d+)\)', expand, picture)
    if re.fullmatch('X+', expanded) and not usage:
        return dict(kind='text', length=len(expanded), digits=0, scale=0, signed=False)
    if not re.fullmatch(r'S?9+(V9+)?', expanded):
        raise FormatError('PICTURE no soportado: use X, 9 o S9 con V opcional.')
    signed = expanded.startswith('S')
    digits = expanded.count('9')
    scale = len(expanded.split('V')[1]) if 'V' in expanded else 0
    if digits > 18:
        raise FormatError('Esta version admite hasta 18 digitos numericos.')
    if usage in ('COMP-3', 'COMPUTATIONAL-3', 'PACKED-DECIMAL'):
        kind, length = 'packed', (digits + 2) // 2
    elif usage in ('COMP', 'COMPUTATIONAL', 'BINARY'):
        kind, length = 'binary', (2 if digits <= 4 else 4 if digits <= 9 else 8)
    elif not usage or usage == 'DISPLAY':
        if signed:
            raise FormatError('DISPLAY con signo/overpunch no soportado en esta version.')
        kind, length = 'display', digits
    else:
        raise FormatError('USAGE no soportado en esta version.')
    return dict(kind=kind, length=length, digits=digits, scale=scale, signed=signed)


def parse_copybook(text: str, key: tuple[int, int] | None = None) -> dict:
    """Restricted parser: unsupported syntax is an error, never a guess."""
    if len(text) > 200_000:
        raise FormatError('Copybook demasiado grande.')
    lines = []
    for raw in text.splitlines():
        # Classic COBOL area A/B, with sequence area and identification columns.
        if len(raw) >= 7 and (raw[:6].isspace() or raw[:6].isdigit()):
            if raw[6] in '*/':
                continue
            if raw[6] == '-':
                raise FormatError('Normaliza las continuaciones COBOL antes de importar.')
            if raw[6] == ' ':
                raw = raw[7:72]
        raw = raw.split('*>', 1)[0].strip()
        if raw and not raw.startswith('*'):
            lines.append(raw)
    source = ' '.join(lines)
    if re.search(r'\b(OCCURS|DEPENDING|SYNC|SYNCHRONIZED|SIGN|JUSTIFIED|RENAMES|COPY)\b', source, re.I):
        raise FormatError('OCCURS/ODO, COPY, SIGN, SYNC y otras clausulas avanzadas no soportadas.')
    statements = [s.strip() for s in source.split('.') if s.strip()]
    roots, stack = [], []
    for statement in statements:
        m = re.fullmatch(r'(\d{2})\s+([\w-]+)(.*)', statement, re.I)
        if not m:
            raise FormatError('Sentencia COBOL no reconocida; revise el copybook.')
        level, name, rest = int(m[1]), m[2], m[3].strip()
        if level == 88:
            continue  # Conditions do not occupy storage; domains require review.
        if not 1 <= level <= 49:
            raise FormatError('Solo se admiten niveles COBOL 01 a 49 y condiciones 88.')
        redef = ''
        rm = re.match(r'REDEFINES\s+([\w-]+)\s*', rest, re.I)
        if rm:
            redef, rest = rm[1], rest[rm.end():]
        pic, usage = '', ''
        if rest:
            pm = re.fullmatch(r'(?:PIC|PICTURE)\s+(?:IS\s+)?([XS9V()0-9]+)'
                              r'(?:\s+(?:USAGE\s+(?:IS\s+)?)?([A-Z0-9-]+))?', rest, re.I)
            if not pm:
                raise FormatError(f'Clausula no soportada en {name}.')
            pic, usage = pm[1].upper(), (pm[2] or '').upper()
        node = _Node(level, name, pic, usage, redef)
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            if stack[-1].picture:
                raise FormatError('Un campo elemental no puede contener subcampos.')
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    if len(roots) != 1 or roots[0].level != 1:
        raise FormatError('Se requiere exactamente un registro de nivel 01.')
    ignored = []
    def place(node, base):
        node.offset = base
        if node.picture:
            node.length = picture_info(node.picture, node.usage)['length']
            return
        cursor, siblings = base, {}
        for child in node.children:
            if child.redefines:
                if child.redefines not in siblings:
                    raise FormatError('REDEFINES debe referir a un hermano previo.')
                previous = siblings[child.redefines]
                place(child, previous.offset)
            else:
                place(child, cursor)
            cursor = max(cursor, child.offset + child.length)
            siblings[child.name] = child
        node.length = cursor - base
    root = roots[0]
    place(root, 0)
    if not 1 <= root.length <= 32760:
        raise FormatError('Longitud de registro fuera de rango.')
    variants = ([n for n in root.children if n.offset == 0 and n.length == root.length]
                if any(n.redefines for n in root.children) else [root])
    if not variants or len(variants) > 10:
        raise FormatError('No se pudieron determinar las vistas completas del registro.')
    views = []
    for variant in variants:
        fields = []
        def flatten(node, groups):
            if node.picture:
                info = picture_info(node.picture, node.usage)
                filler = node.name.upper() == 'FILLER'
                fields.append(dict(name=f'FILLER@{node.offset + 1}' if filler else node.name,
                                   start=node.offset, picture=node.picture, usage=node.usage,
                                   filler=filler, groups=list(groups), **info))
            else:
                for child in node.children:
                    if child.redefines:
                        ignored.append(child.name)
                        continue
                    flatten(child, groups + (node.name,))
        flatten(variant, ())
        views.append(dict(name=variant.name, fields=fields))
    schema = dict(version=1, record_length=root.length, views=views,
                  key_length=key[0] if key else 0, key_offset=key[1] if key else 0,
                  ignored_redefines=sorted(set(ignored)))
    validate_schema(schema)
    return schema


def validate_schema(schema: dict):
    try:
        length = schema['record_length']
        if type(length) is not int or not 1 <= length <= 32760:
            raise FormatError('Longitud de registro no valida.')
        if not 1 <= len(schema['views']) <= 10:
            raise FormatError('Cantidad de vistas no valida.')
        view_names = set()
        for view in schema['views']:
            if not isinstance(view['name'], str) or view['name'] in view_names:
                raise FormatError('Nombre de vista repetido o no valido.')
            view_names.add(view['name'])
            if not 1 <= len(view['fields']) <= 300:
                raise FormatError('Cantidad de campos fuera de limite.')
            cursor, names = 0, set()
            for f in view['fields']:
                expected = picture_info(f['picture'], f.get('usage', ''))
                if any(f.get(k) != v for k, v in expected.items()):
                    raise FormatError('Tipo fisico incompatible con PICTURE.')
                if f['start'] != cursor or f['name'] in names or not isinstance(f['name'], str):
                    raise FormatError('Campos duplicados, huecos o posiciones superpuestas.')
                cursor += f['length']
                names.add(f['name'])
            if cursor != length:
                raise FormatError('La vista no cubre la longitud completa del registro.')
        kl, ko = schema.get('key_length', 0), schema.get('key_offset', 0)
        if type(kl) is not int or type(ko) is not int or min(kl, ko) < 0 or kl + ko > length:
            raise FormatError('Definicion de clave fuera del registro.')
    except (KeyError, TypeError) as exc:
        raise FormatError('Esquema incompleto o no valido.') from exc


def load_mask(data: bytes, filename: str) -> dict:
    if len(data) > MAX_FILE_BYTES:
        raise FormatError('Mascara demasiado grande.')
    if filename.lower().endswith('.xlsx'):
        text, key = xlsx_mask(data)
        return parse_copybook(text, key)
    if filename.lower().endswith('.json'):
        try:
            schema = json.loads(data.decode('utf-8'))
            validate_schema(schema)
            return schema
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise FormatError('Esquema JSON no valido.') from exc
    try:
        return parse_copybook(data.decode('utf-8-sig'))
    except UnicodeError as exc:
        raise FormatError('El copybook debe estar en UTF-8; el extracto se carga por separado.') from exc


def _codec(codec):
    if codec not in CODECS:
        raise FormatError('Seleccione una codificacion de un byte soportada.')


def _scaled(value, f):
    try:
        d = Decimal(str(value))
        scaled = d.scaleb(f['scale'])
        if not scaled.is_finite() or scaled != scaled.to_integral_value():
            raise FormatError('Valor numerico no finito o con demasiados decimales.')
        n = int(scaled)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise FormatError('Valor numerico no valido.') from exc
    if abs(n) >= 10 ** f['digits'] or (n < 0 and not f['signed']):
        raise FormatError('Valor fuera de la capacidad del PICTURE.')
    return n


def decode_field(raw: bytes, f: dict, codec: str):
    _codec(codec)
    if len(raw) != f['length']:
        raise FormatError('Longitud fisica de campo incorrecta.')
    kind = f['kind']
    if kind == 'text':
        return raw.decode(codec, errors='strict')
    if kind == 'display':
        text = raw.decode(codec, errors='strict')
        if not re.fullmatch('[0-9]+', text):
            raise FormatError('DISPLAY contiene bytes no numericos.')
        n = int(text)
    elif kind == 'packed':
        text = raw.hex().upper()
        sign, digits = text[-1], text[:-1]
        if sign not in 'ABCDEF' or not digits.isdigit():
            raise FormatError('COMP-3 contiene digitos o signo invalidos.')
        if len(digits) > f['digits'] and any(x != '0' for x in digits[:-f['digits']]):
            raise FormatError('COMP-3 tiene nibble de relleno no nulo.')
        n = int(digits) * (-1 if sign in 'BD' else 1)
        if sign in 'BD' and not f['signed']:
            raise FormatError('Signo negativo en campo no firmado.')
    elif kind == 'binary':
        n = int.from_bytes(raw, 'big', signed=f['signed'])
    else:
        raise FormatError('Tipo fisico no soportado.')
    if abs(n) >= 10 ** f['digits']:
        raise FormatError('Valor numerico excede el PICTURE.')
    return Decimal(n).scaleb(-f['scale'])


def encode_field(value, f: dict, codec: str) -> bytes:
    _codec(codec)
    if f['kind'] == 'text':
        if not isinstance(value, str):
            raise FormatError('El campo alfanumerico requiere texto.')
        try:
            raw = value.encode(codec, errors='strict')
        except UnicodeError as exc:
            raise FormatError('Texto no representable en la codificacion elegida.') from exc
        if len(raw) > f['length']:
            raise FormatError('El texto excede la longitud; no se trunca silenciosamente.')
        return raw.ljust(f['length'], ' '.encode(codec))
    n = _scaled(value, f)
    if f['kind'] == 'display':
        return str(n).zfill(f['digits']).encode(codec)
    if f['kind'] == 'packed':
        sign = 'D' if n < 0 else 'C' if f['signed'] else 'F'
        text = str(abs(n)).zfill(f['digits']) + sign
        return bytes.fromhex(text.zfill(f['length'] * 2))
    try:
        return n.to_bytes(f['length'], 'big', signed=f['signed'])
    except OverflowError as exc:
        raise FormatError('Valor fuera del rango binario.') from exc


def decode_record(raw: bytes, view: dict, codec: str) -> dict:
    expected = sum(f['length'] for f in view['fields'])
    if len(raw) != expected:
        raise FormatError('Longitud de registro incorrecta.')
    result = {}
    for f in view['fields']:
        try:
            result[f['name']] = decode_field(raw[f['start']:f['start'] + f['length']], f, codec)
        except (FormatError, UnicodeError) as exc:
            raise FormatError(f"Campo {f['name']}, posicion {f['start'] + 1}: formato invalido.") from exc
    return result


def encode_record(values: dict, view: dict, codec: str, original: bytes | None = None) -> bytes:
    """original is only for the no-change test, never for synthetic generation.

    Preserve alternative valid sign nibbles and negative zero when unchanged.
    All fields are decoded/validated, including filler bytes.
    """
    chunks = []
    for f in view['fields']:
        if f['name'] not in values:
            raise FormatError(f"Falta el campo {f['name']}.")
        value = values[f['name']]
        encoded = encode_field(value, f, codec)
        if original is not None:
            raw = original[f['start']:f['start'] + f['length']]
            if decode_field(raw, f, codec) == value:
                encoded = raw
        chunks.append(encoded)
    return b''.join(chunks)


def raw_input(data: bytes, is_hex: bool = False) -> bytes:
    if len(data) > MAX_FILE_BYTES:
        raise FormatError('El extracto supera 8 MiB.')
    if not is_hex:
        return data
    try:
        text = data.decode('ascii')
        if not re.fullmatch(r'[0-9A-Fa-f\s]*', text):
            raise FormatError('HEX admite solo pares hexadecimales y espacios; sin direcciones ni columna ASCII.')
        return bytes.fromhex(text)
    except (UnicodeError, ValueError) as exc:
        raise FormatError('Extracto hexadecimal invalido.') from exc


def parse_records(data: bytes, schema: dict, codec: str, detail_name: str,
                  header_name: str | None = None, detail_marker='D', header_marker='0') -> dict:
    validate_schema(schema)
    _codec(codec)
    length = schema['record_length']
    if not data or len(data) % length or len(data) > MAX_FILE_BYTES:
        raise FormatError('El extracto no es una secuencia completa de registros fijos. No agregue RDW, BDW ni saltos de linea.')
    count = len(data) // length
    if count > MAX_ROWS + 1:
        raise FormatError('Limite de 5000 detalles mas una cabecera.')
    views = {v['name']: v for v in schema['views']}
    if detail_name not in views or header_name and header_name not in views:
        raise FormatError('Vista seleccionada no existe.')
    if header_name == detail_name:
        raise FormatError('Cabecera y detalle deben usar vistas distintas.')
    markers = {}
    for name, marker in ((detail_name, detail_marker), (header_name, header_marker)):
        if not name:
            continue
        first = views[name]['fields'][0]
        if first['start'] != 0 or first['length'] != 1 or first['kind'] != 'text':
            raise FormatError('Esta version requiere tipo de registro alfanumerico de un byte en posicion 1.')
        try:
            b = marker.encode(codec)
        except UnicodeError as exc:
            raise FormatError('Tipo de registro no representable.') from exc
        if len(b) != 1 or b in markers:
            raise FormatError('Tipos de registro deben ser distintos y de un byte.')
        markers[b] = name
    details, headers, keys = [], [], []
    for idx in range(count):
        raw = data[idx * length:(idx + 1) * length]
        name = markers.get(raw[:1])
        if name is None:
            raise FormatError(f'Registro {idx + 1}: tipo desconocido; no se omitio ninguna fila.')
        try:
            row = decode_record(raw, views[name], codec)
            if encode_record(row, views[name], codec, original=raw) != raw:
                raise FormatError('Fallo de reconstruccion byte a byte.')
        except FormatError as exc:
            raise FormatError(f'Registro {idx + 1}: {exc}') from exc
        (details if name == detail_name else headers).append(row)
        if schema['key_length']:
            a, b = schema['key_offset'], schema['key_offset'] + schema['key_length']
            keys.append(raw[a:b])
    if not details or len(details) > MAX_ROWS or len(headers) > 1:
        raise FormatError('Se requieren de 1 a 5000 detalles y como maximo una cabecera.')
    if keys and len(set(keys)) != len(keys):
        raise FormatError('La clave declarada tiene duplicados, incluida cabecera; confirme KEYS.')
    return dict(details=details, headers=headers, roundtrip=True,
                source_sorted=keys == sorted(keys), record_count=count)
