"""Offline tests. All fixtures are invented; no uploaded client rows/layouts.

Run: python -m unittest discover -s pyc-smart-test-data/tests -v
TVAE adapter tests use a test double when SDV is not installed. They do NOT
claim to validate TVAE statistical quality or a live Streamlit deployment.
"""
import ast
import copy
import io
import json
from pathlib import Path
import sys
import types
import unittest
from datetime import date
from decimal import Decimal
from unittest.mock import patch
import zipfile
from xml.sax.saxutils import escape

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mainframe_codec import *
from mainframe_model import *


def numeric(pic='S9(13)V99', usage='COMP-3'):
    return dict(name='AMOUNT', start=0, filler=False, picture=pic, usage=usage,
                groups=[], **picture_info(pic, usage))


class FakeMetadata:
    def __init__(self):
        self.columns = {}
    def add_column(self, column_name, sdtype):
        self.columns[column_name] = sdtype


class FakeTVAE:
    """Not a generative model: test double verifies the interface contract."""
    instances = []
    def __init__(self, metadata, enable_gpu=True, **kwargs):
        self.metadata, self.options, self.enable_gpu = metadata, kwargs, enable_gpu
        self.instances.append(self)
    def fit(self, data):
        self.data = data.copy()
    def sample(self, num_rows):
        return self.data.sample(n=num_rows, replace=True, random_state=7).reset_index(drop=True)


def sdv_double():
    modules = {n: types.ModuleType(n) for n in ('sdv', 'sdv.metadata', 'sdv.single_table')}
    modules['sdv.metadata'].SingleTableMetadata = FakeMetadata
    modules['sdv.single_table'].TVAESynthesizer = FakeTVAE
    return patch.dict(sys.modules, modules)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.schema, self.binary, self.config, self.copybook = demo_bundle()
        self.source = parse_records(self.binary, self.schema, **self.config)
        self.view = view_by_name(self.schema, self.config['detail_name'])
        self.policy = default_policy(self.schema, self.view)

    def trained(self):
        with sdv_double():
            return train_model(self.schema, self.source, self.config, self.policy)

    def header(self):
        return header_defaults(view_by_name(self.schema, self.config['header_name']), '0', date(2026, 4, 1))

    def test_fixed600_layout_views_and_overlay(self):
        self.assertEqual(self.schema['record_length'], 600)
        self.assertEqual((self.schema['key_length'], self.schema['key_offset']), (20, 1))
        self.assertEqual(len(self.schema['views']), 2)
        extra = self.copybook.replace('  05 STATUS PIC X(2).',
            '  05 STATUS PIC X(2).\n  05 STATUS-ALT REDEFINES STATUS.\n   07 STATUS-A PIC X.\n   07 STATUS-B PIC X.')
        schema = parse_copybook(extra, (20, 1))
        self.assertEqual(schema['record_length'], 600)
        names = [f['name'] for f in schema['views'][1]['fields']]
        self.assertNotIn('STATUS-A', names)
        self.assertEqual(schema['ignored_redefines'], ['STATUS-ALT'])

    def test_byte_roundtrip_every_record(self):
        for start in range(0, len(self.binary), 600):
            raw = self.binary[start:start + 600]
            view = self.schema['views'][0 if start == 0 else 1]
            self.assertEqual(encode_record(decode_record(raw, view, 'cp037'), view, 'cp037'), raw)
        self.assertTrue(self.source['roundtrip'])
        self.assertEqual(len(self.source['details']), 200)

    def test_comp3_known_vectors(self):
        f = numeric('S9(3)V99')
        for value, raw in [('123.45', '12345C'), ('-123.45', '12345D'), ('0.00', '00000C')]:
            self.assertEqual(encode_field(Decimal(value), f, 'cp037'), bytes.fromhex(raw))
            self.assertEqual(decode_field(bytes.fromhex(raw), f, 'cp037'), Decimal(value))
        f = numeric('S9(4)')
        self.assertEqual(encode_field(1234, f, 'cp037').hex().upper(), '01234C')

    def test_comp3_limits_precision_and_bad_nibbles(self):
        f = numeric()
        value = Decimal('9999999999999.99')
        self.assertEqual(decode_field(encode_field(value, f, 'cp037'), f, 'cp037'), value)
        for value in ('10000000000000.00', '1.001', 'NaN', 'Infinity'):
            with self.assertRaises(FormatError): encode_field(value, f, 'cp037')
        for raw in ('FA0000000000000C', '0000000000000000', '0000000000000009'):
            with self.assertRaises(FormatError): decode_field(bytes.fromhex(raw), f, 'cp037')
        with self.assertRaises(FormatError): decode_field(bytes.fromhex('11234C'), numeric('S9(4)'), 'cp037')

    def test_preserve_alternative_sign_and_negative_zero(self):
        f = numeric('S9(3)V99')
        view = dict(fields=[f])
        for h in ('12345F', '12345A', '00000D', '12345B'):
            raw = bytes.fromhex(h)
            values = decode_record(raw, view, 'cp037')
            self.assertEqual(encode_record(values, view, 'cp037', raw), raw)

    def test_comp_and_display(self):
        f = numeric('S9(4)', 'COMP')
        self.assertEqual(encode_field(-123, f, 'cp037'), b'\xff\x85')
        self.assertEqual(decode_field(b'\xff\x85', f, 'cp037'), -123)
        f = numeric('9(8)', '')
        self.assertEqual(encode_field(123, f, 'cp037'), '00000123'.encode('cp037'))
        with self.assertRaises(FormatError): decode_field(b'ABCDEFGH', f, 'cp037')
        with self.assertRaises(FormatError): encode_field(-1, f, 'cp037')

    def test_text_is_never_truncated(self):
        f = dict(name='T', **picture_info('X(4)'))
        self.assertEqual(encode_field('A', f, 'cp037'), 'A   '.encode('cp037'))
        with self.assertRaises(FormatError): encode_field('ABCDE', f, 'cp037')
        with self.assertRaises(FormatError): encode_field('😀', f, 'cp037')
        with self.assertRaises(FormatError): encode_field('A', f, 'utf-8')

    def test_reject_partial_records_wrong_codepage_and_duplicate_keys(self):
        for data in (b'', self.binary[:-1], self.binary + b'\n', b'\0\0\0\0' + self.binary):
            with self.assertRaises(FormatError): parse_records(data, self.schema, **self.config)
        with self.assertRaises(FormatError):
            parse_records(self.binary, self.schema, **{**self.config, 'codec': 'ascii'})
        with self.assertRaises(FormatError):
            parse_records(self.binary + self.binary[600:1200], self.schema, **self.config)
        with self.assertRaises(FormatError):
            parse_records(self.binary + self.binary[:600], self.schema, **self.config)

    def test_hex_roundtrip_and_invalid_input(self):
        text = '\n'.join(self.binary[i:i+16].hex() for i in range(0, len(self.binary), 16))
        self.assertEqual(raw_input(text.encode(), True), self.binary)
        for value in (b'0', b'zz', b'0000: AABB', b'0xAB'):
            with self.assertRaises(FormatError): raw_input(value, True)

    def test_fail_closed_unsupported_copybook(self):
        for clause in ('OCCURS 2 TIMES', 'COMP-5', 'SIGN LEADING', 'VALUE 0', 'SYNC'):
            text = self.copybook.replace('PIC 9(8).', 'PIC 9(8) ' + clause + '.')
            with self.assertRaises(FormatError): parse_copybook(text)
        with self.assertRaises(FormatError): picture_info('S9(4)', '')
        with self.assertRaises(FormatError): parse_copybook(self.copybook + '\n01 SECOND PIC X.')
        with self.assertRaises(FormatError): parse_copybook('01 R. 05 A PIC X(999999999).')

    def test_schema_json_validation(self):
        self.assertEqual(load_mask(json.dumps(self.schema).encode(), 'x.json'), self.schema)
        schema = copy.deepcopy(self.schema)
        schema['views'][1]['fields'][2]['start'] += 1
        with self.assertRaises(FormatError): validate_schema(schema)
        schema = copy.deepcopy(self.schema)
        schema['views'][1]['fields'][3]['length'] = 9
        with self.assertRaises(FormatError): validate_schema(schema)

    def test_privacy_and_structural_policy(self):
        for target, treatment in [('CLIENT-KEY', 'Sintetizar'), ('EVENT-MM', 'Sintetizar')]:
            policy = copy.deepcopy(self.policy)
            next(p for p in policy if p['Campo'] == target)['Tratamiento'] = treatment
            with self.assertRaises(FormatError): validate_policy(self.schema, self.view, policy, 'cp037')
        p = copy.deepcopy(self.policy)
        next(v for v in p if v['Campo'] == 'AMOUNT')['Confidencial'] = True
        with self.assertRaises(FormatError): validate_policy(self.schema, self.view, p, 'cp037')

    def test_only_approved_business_columns_reach_tvae(self):
        trained = self.trained()
        self.assertEqual(trained['learned'], ['AMOUNT', 'FEE'])
        self.assertEqual(list(trained['engine'].data.columns), ['AMOUNT', 'FEE'])
        self.assertFalse(trained['engine'].enable_gpu)
        self.assertEqual(trained['engine'].data['AMOUNT'].dtype.kind, 'i')

    def test_generation_and_export(self):
        run = generate(self.schema, self.source, self.config, self.trained(), 80, header_values=self.header())
        self.assertEqual(run['size_bytes'], 600 * 81)
        self.assertTrue(run['sorted_keys'])
        self.assertTrue(run['roundtrip'])
        for name in ('CLIENT-KEY', 'SEQUENCE-NO', 'DESCRIPTION'):
            self.assertFalse({str(r[name]) for r in self.source['details']} & {str(r[name]) for r in run['details']})
        parsed = parse_records(run['binary'], self.schema, **self.config)
        self.assertEqual(len(parsed['details']), 80)
        payload = export_zip(run, self.schema, self.config)
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            self.assertEqual(z.read('mainframe_sintetico.bin'), run['binary'])
            for name in z.namelist():
                self.assertNotIn(b'EJEMPLO INVENTADO', z.read(name))
                self.assertNotIn(b'SRC000000001', z.read(name))
            self.assertFalse(json.loads(z.read('manifest.json'))['privacy_certified'])

    def test_temporal_ranges_and_header_validation(self):
        source = copy.deepcopy(self.source)
        source['details'][0]['EVENT-MM'] = Decimal(13)
        with sdv_double(), self.assertRaises(FormatError):
            train_model(self.schema, source, self.config, self.policy)
        h = self.header()
        h['PROCESS-DD'] = 99
        with self.assertRaises(FormatError):
            generate(self.schema, self.source, self.config, self.trained(), 20, header_values=h)

    def test_constant_only_model_needs_no_sdv(self):
        source = copy.deepcopy(self.source)
        for r in source['details']:
            r['AMOUNT'], r['FEE'] = Decimal('10.00'), Decimal('0.00')
        trained = train_model(self.schema, source, self.config, self.policy)
        self.assertIsNone(trained['engine'])
        self.assertIn('sin TVAE', trained['algorithm'])
        run = generate(self.schema, source, self.config, trained, 30, header_values=self.header())
        self.assertTrue(all(r['AMOUNT'] == Decimal('10.00') for r in run['details']))

    def test_bounds_and_incompatible_model(self):
        trained = self.trained()
        for q in (0, 2001):
            with self.assertRaises(FormatError): generate(self.schema, self.source, self.config, trained, q)
        changed = copy.deepcopy(self.schema)
        changed['key_offset'] = 2
        with self.assertRaises(FormatError): generate(changed, self.source, self.config, trained, 20)
        with self.assertRaises(FormatError): generate(self.schema, self.source, self.config, trained, 20)

    def test_isolated_state_and_source_replacement(self):
        relational = {'tables': {'sentinel': 1}, 'models': [object()], 'runs': [{'old': True}], 'dictionary': 'kept'}
        original = {k: id(v) for k, v in relational.items()}
        state = new_state()
        state['runs'] = [{'old': 'mainframe'}]
        relational[STATE_KEY] = replace_source(state, self.schema, self.source, self.config)
        self.assertEqual(original, {k: id(relational[k]) for k in original})
        self.assertEqual(relational[STATE_KEY]['runs'], [])
        self.assertNotEqual(STATE_KEY, 'tables')

    def test_new_modules_compile_and_no_relational_import(self):
        for name in ('mainframe_codec.py', 'mainframe_model.py', 'mainframe_ui.py', 'pages/1_Archivos_mainframe.py'):
            text = (ROOT / name).read_text()
            ast.parse(text)
            self.assertNotIn('import relational', text)
            self.assertNotIn('from relational', text)
            self.assertNotIn('st.session_state.tables', text)
            self.assertNotIn('st.session_state.models', text)
            self.assertNotIn('st.session_state.runs', text)
            self.assertNotIn('st.session_state.dictionary', text)

    def test_xlsx_copy_only_and_security(self):
        # Minimal OOXML with inline strings. Deliberately malformed DATA must never be read.
        def make_xlsx(formula=False, dtd=False):
            out = io.BytesIO()
            rows = ''.join(f'<row r="{i}"><c r="B{i}" t="inlineStr">'
                           + ('<f>1+1</f>' if formula and i == 1 else '')
                           + '<is><t>' + escape(line) + '</t></is></c></row>'
                           for i, line in enumerate(self.copybook.splitlines(), 1))
            with zipfile.ZipFile(out, 'w') as z:
                z.writestr('xl/workbook.xml', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="COPY" r:id="r1"/><sheet name="DATA" r:id="r2"/></sheets></workbook>')
                z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/><Relationship Id="r2" Target="worksheets/sheet2.xml"/></Relationships>')
                z.writestr('xl/worksheets/sheet1.xml', ('<!DOCTYPE x>' if dtd else '') + '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + rows + '</sheetData></worksheet>')
                z.writestr('xl/worksheets/sheet2.xml', 'SOURCE_PRIVATE_DO_NOT_READ')
            return out.getvalue()
        schema = load_mask(make_xlsx(), 'test.xlsx')
        self.assertEqual(schema['record_length'], 600)
        self.assertNotIn('SOURCE_PRIVATE', json.dumps(schema))
        for data in (make_xlsx(formula=True), make_xlsx(dtd=True), b'not zip'):
            with self.assertRaises(FormatError): load_mask(data, 'bad.xlsx')


if __name__ == '__main__':
    unittest.main()
