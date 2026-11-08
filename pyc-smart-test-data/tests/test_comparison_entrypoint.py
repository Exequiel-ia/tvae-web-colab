"""Regression for the former age-only menu. No Streamlit/TVAE execution.

The routing call uses a controlled spy; numerical checks execute the actual
comparison_core formulas. This does not replace a live Streamlit Cloud test.
"""
import ast
import copy
import hashlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comparison_core import assess_seven, compare_metrics, metrics_match

APP_TEXT = (ROOT / 'streamlit_app.py').read_text(encoding='utf-8')
TREE = ast.parse(APP_TEXT)


def page_branch(name):
    for node in ast.walk(TREE):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == 'page'
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == name):
            return node
    raise AssertionError('Missing menu section: ' + name)


class ComparisonEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.original = pd.DataFrame({
            'cliente_id': [1, 2, 3, 4, 5, 6],
            'edad': [20, 30, 40, 50, 60, 70],
            'saldo': [10., 20., 15., 30., 50., 60.],
            'segmento': ['Persona', 'Pyme', 'Empresa', 'Persona', 'Pyme', 'Empresa'],
        })
        self.a = self.original.copy(deep=True)
        self.a['saldo'] += 1.
        self.b = self.original.copy(deep=True)
        self.b['segmento'] = 'Persona'
        self.b['saldo'] = [5., 60., 10., 50., 20., 30.]

    def test_main_menu_calls_the_full_comparator(self):
        body = page_branch('Comparar experimentos').body
        self.assertEqual(len(body), 2)
        self.assertIsInstance(body[0], ast.ImportFrom)
        self.assertEqual(body[0].module, 'comparison_ui')
        self.assertEqual(body[0].names[0].name, 'render_comparator')
        self.assertIsInstance(body[1], ast.Expr)
        self.assertEqual(body[1].value.func.id, 'render_comparator')

    def test_no_age_only_chart_or_state_reset_in_route(self):
        body = page_branch('Comparar experimentos').body
        text = '\n'.join(ast.get_source_segment(APP_TEXT, n) for n in body)
        for old in ('edad_media_similitud', 'histogram', 'px.line', 'switch_page',
                    'session_state.models', 'session_state.runs', 'st.rerun'):
            self.assertNotIn(old, text)

    def test_route_delegates_without_mutating_session(self):
        branch = ast.Module(body=page_branch('Comparar experimentos').body, type_ignores=[])
        for n in (0, 1, 2):
            with self.subTest(generations=n):
                state = {'models': [{'name': 'Inventado', 'params': {'épocas': 10}}],
                         'runs': [{'nombre': str(i)} for i in range(n)],
                         'tables': {'clientes': self.original.copy(deep=True)},
                         'dictionary': {'test': True}, 'mainframe': {'untouched': True}}
                old = copy.deepcopy(state)
                objects = {k: id(v) for k, v in state.items()}
                mod = types.ModuleType('comparison_ui')
                mod.render_comparator = Mock()
                with patch.dict(sys.modules, {'comparison_ui': mod}):
                    exec(compile(branch, 'comparison_menu', 'exec'),
                         {'st': types.SimpleNamespace(session_state=state)})
                mod.render_comparator.assert_called_once_with()
                self.assertEqual(objects, {k: id(v) for k, v in state.items()})
                for key in ('models', 'runs', 'dictionary', 'mainframe'):
                    self.assertEqual(old[key], state[key])
                pd.testing.assert_frame_equal(old['tables']['clientes'], state['tables']['clientes'])

    def test_all_seven_metrics_compare_each_run_to_original(self):
        frame, warnings = compare_metrics(self.original, self.a, self.b)
        self.assertFalse(warnings)
        self.assertEqual(frame['Métrica'].tolist(), ['MET'+str(i) for i in range(1, 8)])
        sa, sb = assess_seven(self.original, self.a), assess_seven(self.original, self.b)
        for row in frame.to_dict('records'):
            key = row['Métrica']
            self.assertAlmostEqual(row['A'], round(sa[key], 2))
            self.assertAlmostEqual(row['B'], round(sb[key], 2))
            self.assertAlmostEqual(row['Δ B − A (puntos)'], round(sb[key]-sa[key], 2))

    def test_same_ages_do_not_hide_other_metric_changes(self):
        pd.testing.assert_series_equal(self.a.edad, self.b.edad)
        scores, _ = compare_metrics(self.original, self.a, self.b)
        rows = scores.set_index('Métrica')
        self.assertEqual(rows.loc['MET2', 'Lectura'], 'Baja')
        self.assertLess(rows.loc['MET2', 'B'], rows.loc['MET2', 'A'])
        self.assertNotEqual(rows.loc['MET1', 'A'], rows.loc['MET1', 'B'])

    def test_not_applicable_remains_visible_not_zero(self):
        cols = ['cliente_id', 'edad', 'segmento']
        scores, _ = compare_metrics(self.original[cols], self.a[cols], self.b[cols])
        self.assertEqual(len(scores), 7)
        met1 = scores.set_index('Métrica').loc['MET1']
        self.assertTrue(pd.isna(met1['A']))
        self.assertTrue(pd.isna(met1['B']))
        self.assertTrue(pd.isna(met1['Δ B − A (puntos)']))
        self.assertEqual(met1['Lectura'], 'No comparable')

    def test_source_and_generations_are_unchanged_by_scoring(self):
        frames = (self.original, self.a, self.b)
        before = [f.copy(deep=True) for f in frames]
        compare_metrics(*frames)
        for old, current in zip(before, frames):
            pd.testing.assert_frame_equal(old, current)

    def test_existing_metric_formulas_still_match(self):
        self.assertTrue(metrics_match())

    def test_other_sections_preserved(self):
        # Baseline: 71e4fea4, streamlit_app.py blob 74b0b447 (text of each body).
        expected = {'Base de datos': '38aff2aa873204668aeaf97a06c8956a8e4495a94a493e376d81c02cc473d8fd', 'Diccionario y privacidad': '58b30b6604be90d8900211f6431768884eef27655d55c4e212856c54b2ce2a81', 'Exploración': '42c4fbab152655cc7a6862389f2478bbb5f687ad1da0c9b4e5f9cb5a95ef7ff9', 'Entrenamiento': '715600c12b2fa352ac52f2fb31a8bbe5b376ff26995b685665acc18ce254a4dc', 'Generación': 'f83a79f23ed2b6897a95833b9a7a93dbb8dab70727e8e49dc30d1b68bca472c3', '7 métricas QE': 'acc5acd2470cde655a632081bdbe89a56ed58fcd3e2304bf66db3a706b2434b1'}
        for name, digest in expected.items():
            with self.subTest(section=name):
                text = '\n'.join(ast.get_source_segment(APP_TEXT, n) for n in page_branch(name).body)
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), digest)

    def test_help_explains_seven_metrics_against_original(self):
        config = next(n for n in TREE.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'EXEIA_HELP' for t in n.targets))
        help_text = ast.literal_eval(config.value)['Comparar experimentos']
        self.assertIn('MET1–MET7', help_text['explicacion'])
        self.assertIn('datos originales', help_text['explicacion'])
        self.assertIn('No aplica', help_text['tip'])


if __name__ == '__main__':
    unittest.main()
