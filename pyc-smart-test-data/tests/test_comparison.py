"""Unit tests use fictional data. UI traversal uses a stub, not Streamlit AppTest."""
import copy
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import comparison_core as core


def example(n=40):
    return pd.DataFrame({'cliente_id':range(1,n+1), 'edad':np.arange(n,dtype='int64')+20,
                         'segmento':['Personas','Pyme']*(n//2),
                         'nombre_cliente':['Cliente ficticio '+str(i) for i in range(n)]})


def runs_fixture():
    original = example()
    a = original.copy(deep=True)
    b = original.copy(deep=True)
    a['edad'] = a['edad']//2+10
    b['edad'] = b['edad']+1
    a['nombre_cliente'] = 'Ficticio A'
    b['nombre_cliente'] = 'Ficticio B'
    params = {'épocas':100,'batch':100,'embedding':128,'capas':'128 × 128','L2':1e-5,
              'loss':2,'semilla':42,'minmax':True,'rounding':True,'diccionario':'[]'}
    runs = [dict(nombre='Generación A',modelo='TVAE 1',params=params,tables={'clientes':a},rows=40,train_sec=3.,gen_sec=.1),
            dict(nombre='Generación B',modelo='TVAE 2',params={**params,'épocas':300},tables={'clientes':b},rows=40,train_sec=6.,gen_sec=.1)]
    return {'tables':{'clientes':original},'runs':runs,'models':[object()], 'mainframe_workspace':{'unchanged':True}}


class CoreTests(unittest.TestCase):
    def test_exact_copy_metric_low(self):
        scores = core.assess_seven(example(),example())
        self.assertEqual(scores['MET4'],0)
        self.assertEqual(scores['MET2'],100)
        self.assertEqual(scores['MET3'],100)
        self.assertEqual(scores['MET5'],100)
        self.assertEqual(scores['MET6'],100)
        self.assertEqual(scores['MET7'],100)
        self.assertTrue(np.isnan(scores['MET1']))

    def test_mirror_matches_repository_source(self):
        path = ROOT/'streamlit_app.py'
        if not path.exists():
            self.skipTest('Original app source not present in this standalone test bundle')
        self.assertTrue(core.metrics_match(path))

    def test_formula_drift_blocks(self):
        text = 'METRIC_DESCRIPTIONS='+repr(core.METRIC_DESCRIPTIONS)+'\n'+inspect.getsource(core.assess_seven)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'app.py'
            path.write_text(text,encoding='utf-8')
            self.assertTrue(core.metrics_match(path))
            path.write_text(text.replace('1-10*np.mean(deviations)','1-5*np.mean(deviations)'),encoding='utf-8')
            self.assertFalse(core.metrics_match(path))

    def test_missing_file_blocks(self):
        self.assertFalse(core.metrics_match(Path('/not_a_file/absent.py')))

    def test_missing_column_does_not_invent_zero(self):
        scores,warning = core.safe_scores(example(),example().drop(columns=['edad']))
        self.assertTrue(warning)
        self.assertTrue(all(np.isnan(x) for x in scores.values()))

    def test_empty_dataset(self):
        scores,warning = core.safe_scores(example(),example().head(0))
        self.assertTrue(warning)
        self.assertTrue(all(np.isnan(x) for x in scores.values()))

    def test_duplicate_columns(self):
        frame = example(); frame.columns = ['x','x','y','z']
        self.assertTrue(core.safe_scores(frame,frame)[1])

    def test_delta_direction_and_na(self):
        state = runs_fixture()
        frame,warnings = core.compare_metrics(state['tables']['clientes'],state['runs'][0]['tables']['clientes'],state['runs'][1]['tables']['clientes'])
        self.assertFalse(warnings)
        self.assertTrue(pd.isna(frame.iloc[0]['Δ B − A (puntos)']))
        for row in frame.dropna(subset=['Δ B − A (puntos)']).to_dict('records'):
            self.assertAlmostEqual(row['B']-row['A'],row['Δ B − A (puntos)'],places=1)

    def test_parameter_difference(self):
        runs = runs_fixture()['runs']
        frame = core.parameter_table(*runs)
        diff = frame[(frame.Grupo=='Entrenamiento') & (frame.Estado=='Cambió')]
        self.assertEqual(list(diff.Parámetro),['Épocas'])
        seed = frame[frame.Parámetro=='Semilla de generación'].iloc[0]
        self.assertEqual(seed.Estado,'No registrado')

    def test_false_and_zero_are_registered(self):
        frame = core.parameter_table({'params':{'minmax':False,'L2':0}},{'params':{'minmax':True,'L2':0}})
        self.assertEqual(frame.loc[frame.Parámetro=='Respetar mínimo/máximo','Estado'].item(),'Cambió')
        self.assertEqual(frame.loc[frame.Parámetro=='Regularización L2','Estado'].item(),'Igual')

    def test_dictionary_row_order(self):
        data = [{'Entidad':'clientes','Columna':'edad'},{'Entidad':'clientes','Columna':'region'}]
        a,b = {'params':{'diccionario':json.dumps(data)}},{'params':{'diccionario':json.dumps(data[::-1])}}
        self.assertEqual(core.parameter_table(a,b).iloc[-1].Estado,'Igual')

    def test_provenance_not_retrofitted(self):
        self.assertEqual(core.provenance({}, {}, 'active')[0],'unverified')
        self.assertEqual(core.provenance({'source_fingerprint':'active'}, {}, 'active')[0],'unverified')
        self.assertEqual(core.provenance({'source_fingerprint':'old'}, {}, 'active')[0],'blocked')
        self.assertEqual(core.provenance({'source_fingerprint':'active'}, {'source_fingerprint':'active'}, 'active')[0],'verified')

    def test_model_names_not_identity(self):
        self.assertIn('no verificada',core.model_relation({'modelo':'M'},{'modelo':'M'}))
        self.assertEqual(core.model_relation({'model_id':'id'},{'model_id':'id'}),'Dos generaciones del mismo modelo')

    def test_fingerprint(self):
        frame = example()
        before = core.source_fingerprint({'clientes':frame})
        self.assertEqual(before,core.source_fingerprint({'clientes':frame.copy()}))
        frame.loc[0,'edad'] += 1
        self.assertNotEqual(before,core.source_fingerprint({'clientes':frame}))

    def test_numeric_bins_and_different_sizes(self):
        a = example()
        frames = {'Original':a,'A':pd.concat([a,a],ignore_index=True),'B':a}
        dist = core.numeric_distribution(frames,'edad',5)
        self.assertEqual(dist.groupby('Conjunto').Cantidad.sum().to_dict(),{'Original':40,'A':80,'B':40})
        self.assertTrue(np.allclose(dist.groupby('Conjunto').Porcentaje.sum(),100))
        self.assertEqual(dist[dist.Conjunto=='A'].Intervalo.tolist(),dist[dist.Conjunto=='Original'].Intervalo.tolist())
        self.assertTrue(np.allclose(dist[dist.Conjunto=='A'].Porcentaje,dist[dist.Conjunto=='Original'].Porcentaje))

    def test_numeric_missing_and_inf(self):
        frame = pd.DataFrame({'x':[1,None,np.inf,'bad',3]})
        dist = core.numeric_distribution({'Original':frame,'A':frame,'B':frame},'x')
        self.assertTrue(np.allclose(dist.groupby('Conjunto').Cantidad.sum(),2))
        summary = core.numeric_summary({'Original':frame},'x').iloc[0]
        self.assertEqual(summary['Nulos'],1)
        self.assertEqual(summary['No numéricos / infinitos'],2)

    def test_numeric_empty_and_constant(self):
        empty = pd.DataFrame({'x':[np.nan,np.nan]})
        self.assertTrue(core.numeric_distribution({'Original':empty},'x').empty)
        constant = pd.DataFrame({'x':[5]*10})
        self.assertEqual(core.numeric_distribution({'Original':constant},'x').Cantidad.sum(),10)

    def test_categorical_totals_other_and_null(self):
        frame = pd.DataFrame({'x':[str(i) for i in range(20)]+[None]})
        dist = core.categorical_distribution({'Original':frame,'A':frame,'B':frame},'x',3)
        self.assertTrue(np.allclose(dist.groupby('Conjunto').Porcentaje.sum(),100))
        self.assertIn('Otras categorías (agrupadas)',dist.Categoría.tolist())
        self.assertEqual(dist.groupby('Conjunto').Cantidad.sum().tolist(),[21,21,21])

    def test_null_category_not_confused_with_literal(self):
        frame = pd.DataFrame({'x':[None,'(Nulo)']})
        dist = core.categorical_distribution({'Original':frame},'x')
        self.assertEqual(len(dist),2)

    def test_outliers_original_limits(self):
        original = pd.DataFrame({'x':[10,11,12,13,14]*10})
        synth = pd.DataFrame({'x':[10,100]})
        limits,summary,masks = core.outlier_evidence({'Original':original,'B':synth},'x')
        self.assertEqual(int(summary.loc[summary.Conjunto=='B','Casos extremos'].item()),1)
        self.assertEqual(masks['B'].tolist(),[False,True])
        self.assertLess(limits[1],100)

    def test_iqr_degenerate_not_zero_cases(self):
        frame = pd.DataFrame({'x':[1,1,1,1,100]})
        limits,summary,_ = core.outlier_evidence({'Original':frame},'x','iqr')
        self.assertIsNone(limits)
        self.assertTrue(summary.empty)

    def test_schema_missing_column(self):
        frame = core.schema_evidence({'Original':example(),'A':example().drop(columns=['edad'])})
        self.assertEqual(frame.loc[frame.Columna=='edad','A · tipo'].item(),'Ausente')

    def test_safe_csv_formula_and_negative_number(self):
        frame = pd.DataFrame({'text':['=SUM(A1:A2)','+X','@X','normal'],'number':[-1,2,3,4]})
        output = core.safe_csv(frame).decode('utf-8-sig')
        self.assertIn("'=SUM",output)
        self.assertIn("'+X",output)
        self.assertIn(',\u002d1',output)
        self.assertEqual(frame.loc[0,'text'],'=SUM(A1:A2)')

    def test_no_mutation(self):
        state = runs_fixture(); before = copy.deepcopy(state['runs'])
        original_before = state['tables']['clientes'].copy(deep=True)
        core.parameter_table(*state['runs'])
        core.compare_metrics(state['tables']['clientes'],*(r['tables']['clientes'] for r in state['runs']))
        core.numeric_distribution({'Original':state['tables']['clientes']},'edad')
        core.safe_csv(state['tables']['clientes'])
        pd.testing.assert_frame_equal(original_before,state['tables']['clientes'])
        for left,right in zip(before,state['runs']):
            pd.testing.assert_frame_equal(left['tables']['clientes'],right['tables']['clientes'])
            self.assertEqual(left['params'],right['params'])


class Block:
    def __enter__(self): return self
    def __exit__(self,*args): return False


class FakeStreamlit(types.ModuleType):
    """Captures rendering and exercises branches. Not a browser/Streamlit test."""
    def __init__(self,state,options=None):
        super().__init__('streamlit')
        self.session_state = state
        self.options = options or {}
        self.sidebar = self
        self.messages=[]; self.frames=[]; self.charts=[]; self.downloads=[]
    def markdown(self,text,**kwargs): self.messages.append(str(text))
    def info(self,text,**kwargs): self.messages.append(str(text))
    warning=error=success=caption=info
    def write(self,*args,**kwargs): self.messages.append(' '.join(map(str,args)))
    def page_link(self,*args,**kwargs): pass
    def selectbox(self,label,options,index=0,key=None,**kwargs): return self.options.get(key,list(options)[index])
    def radio(self,label,options,index=0,key=None,**kwargs): return self.options.get(key,list(options)[index])
    def slider(self,label,*args,key=None,**kwargs): return self.options.get(key,args[2] if len(args)>2 else kwargs.get('value'))
    def checkbox(self,label,value=False,key=None,**kwargs): return self.options.get(key,value)
    def multiselect(self,label,options,default=None,key=None,**kwargs): return self.options.get(key,default or [])
    def text_input(self,label,value='',key=None,**kwargs): return self.options.get(key,value)
    def tabs(self,labels): return [Block() for _ in labels]
    def expander(self,*args,**kwargs): return Block()
    def dataframe(self,frame,**kwargs): self.frames.append(frame)
    def plotly_chart(self,fig,**kwargs): self.charts.append(fig)
    def download_button(self,label,data,*args,**kwargs): self.downloads.append((label,data))


class UITests(unittest.TestCase):
    def render(self,state,options=None,match=True):
        stub = FakeStreamlit(state,options)
        with patch.dict(sys.modules,{'streamlit':stub}):
            import comparison_ui
            importlib.reload(comparison_ui)
            with patch.object(comparison_ui,'metrics_match',return_value=match):
                comparison_ui.render_comparator()
        return stub

    def test_empty_guidance(self):
        stub = self.render({})
        self.assertTrue(any('dos generaciones' in m for m in stub.messages))
        self.assertFalse(stub.charts)

    def test_one_run_guidance(self):
        state=runs_fixture(); state['runs']=state['runs'][:1]
        self.assertFalse(self.render(state).charts)

    def test_all_tabs_and_state_unchanged(self):
        state=runs_fixture(); snapshots=copy.deepcopy(state['runs']); identities={k:id(v) for k,v in state.items()}
        stub = self.render(state)
        self.assertEqual(len(stub.charts),2)
        self.assertGreaterEqual(len(stub.downloads),3)
        self.assertEqual(stub.charts[0].layout.xaxis.range,(0,100))
        self.assertEqual(stub.charts[0].layout.barmode,'group')
        self.assertTrue(any('no verificada' in m.lower() for m in stub.messages))
        self.assertEqual(identities,{k:id(v) for k,v in state.items()})
        for before,after in zip(snapshots,state['runs']):
            pd.testing.assert_frame_equal(before['tables']['clientes'],after['tables']['clientes'])

    def test_categorical_branch(self):
        stub=self.render(runs_fixture(),{'ab_variable_clientes':'segmento'})
        self.assertEqual(len(stub.charts),2)
        self.assertTrue(any('frecuencias' in m.lower() for m in stub.messages))

    def test_filter_empty_columns(self):
        stub=self.render(runs_fixture(),{'ab_columns_clientes_Original':[],'ab_filter_clientes_Original':'nobody'})
        self.assertEqual(len(stub.downloads),2)

    def test_blocked_reference(self):
        state=runs_fixture(); state['runs'][0]['source_fingerprint']='other'
        self.assertFalse(self.render(state).charts)

    def test_formula_drift_screen(self):
        stub=self.render(runs_fixture(),match=False)
        self.assertFalse(stub.charts)
        self.assertTrue(any('fórmulas' in m for m in stub.messages))

    def test_no_common_entities(self):
        state=runs_fixture(); state['runs'][1]['tables']={'other':example()}
        self.assertFalse(self.render(state).charts)

    def test_html_escaped(self):
        state=runs_fixture();state['runs'][0]['nombre']='<script>alert(1)</script>'
        stub=self.render(state)
        cards=next(m for m in stub.messages if 'cmp-card' in m and '<div class="cmp-grid">' in m)
        self.assertIn('&lt;script&gt;',cards)
        self.assertNotIn('<script>',cards)


if __name__=='__main__':
    unittest.main()
