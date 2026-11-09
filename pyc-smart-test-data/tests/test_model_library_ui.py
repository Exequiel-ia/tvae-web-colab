"""UI control-flow tests with an explicit Streamlit double (not browser/AppTest)."""
import importlib
import os
import secrets
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_model_library import fixture
from model_library import ModelLibrary, STATE_KEY, capture_model, new_code

class Rerun(BaseException):pass

class Context:
    def __enter__(self):return self
    def __exit__(self,*args):return False

class StreamlitDouble(types.ModuleType):
    def __init__(self):
        super().__init__('streamlit')
        self.session_state={}
        self.secrets={}
        self.sidebar=self
        self.clicked=set()
        self.checked=set()
        self.errors=[]
        self.messages=[]
        self.upload=None
    def markdown(self,*a,**k):pass
    def page_link(self,*a,**k):pass
    def caption(self,*a,**k):pass
    def code(self,*a,**k):pass
    def dataframe(self,*a,**k):pass
    def warning(self,text,**k):self.messages.append(text)
    def info(self,text,**k):self.messages.append(text)
    def write(self,*a,**k):pass
    def success(self,text,**k):self.messages.append(text)
    def error(self,text,**k):self.errors.append(text)
    def download_button(self,*a,**k):return False
    def expander(self,*a,**k):return Context()
    def tabs(self,names):return [Context() for _ in names]
    def checkbox(self,label,**kwargs):return kwargs.get('key') in self.checked
    def text_input(self,*a,**k):return ''
    def button(self,label,**kwargs):return kwargs.get('key') in self.clicked and not kwargs.get('disabled',False)
    def multiselect(self,label,options,**kwargs):return kwargs.get('default',[])
    def selectbox(self,label,options,**kwargs):return list(options)[0]
    def file_uploader(self,*a,**k):return self.upload
    def rerun(self):raise Rerun()

class UITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.secret=secrets.token_hex(32)
        self.env=mock.patch.dict(os.environ,{'PYC_MODEL_SIGNING_KEY':self.secret,'PYC_MODEL_LIBRARY_DIR':self.tmp.name})
        self.env.start()
        self.st=StreamlitDouble()
        self.modules=mock.patch.dict(sys.modules,{'streamlit':self.st});self.modules.start()
        sys.modules.pop('model_library_ui',None)
        self.ui=importlib.import_module('model_library_ui')
        self.st.session_state[STATE_KEY]=new_code()
        self.record,self.tables,self.dictionary=fixture()
    def tearDown(self):
        sys.modules.pop('model_library_ui',None)
        self.modules.stop();self.env.stop();self.tmp.cleanup()
    def prepare_active(self):
        self.st.session_state.update(models=[self.record],tables=self.tables,dictionary=self.dictionary,runs=[{'nombre':'existing'}])
    def save_one(self):
        self.prepare_active()
        self.st.clicked={'ml_save'};self.st.checked={'ml_confirm_source'}
        self.ui.render_library()
        self.st.clicked=set();self.st.checked=set()
        return self.record['library_id']
    def test_empty_page_does_not_train_or_fail(self):
        self.ui.render_library();self.assertEqual(self.st.errors,[])
    def test_save_active_model(self):
        self.save_one();self.assertEqual(self.st.errors,[])
        store=ModelLibrary(Path(self.tmp.name),self.secret)
        self.assertEqual(len(store.list(self.st.session_state[STATE_KEY])),1)
    def test_recover_model_no_training(self):
        self.save_one();self.st.session_state['models']=[];self.st.clicked={'ml_activate'}
        with self.assertRaises(Rerun):self.ui.render_library()
        self.assertEqual(len(self.st.session_state['models']),1)
        self.assertEqual(self.st.session_state['runs'],[{'nombre':'existing'}])
    def test_delete_requires_confirmation(self):
        self.save_one();self.st.clicked={'ml_delete'}
        self.ui.render_library()
        self.assertEqual(len(self.st.session_state['models']),1)
    def test_delete_with_confirmation(self):
        ident=self.save_one();self.st.clicked={'ml_delete'};self.st.checked={'ml_confirm_delete_'+ident}
        with self.assertRaises(Rerun):self.ui.render_library()
        self.assertEqual(self.st.session_state['models'],[])
        self.assertEqual(self.st.session_state['runs'],[{'nombre':'existing'}])
    def test_prepare_backup(self):
        self.save_one();self.st.clicked={'ml_prepare_backup'};self.ui.render_library()
        self.assertIn('ml_backup_data',self.st.session_state)
    def test_wrong_context_not_replaced_without_checkbox(self):
        self.save_one();self.st.session_state['dictionary']=self.dictionary.assign(Tratamiento='Conservar')
        self.st.clicked={'ml_activate'};self.ui.render_library()
        self.assertTrue(self.st.errors)
        self.assertEqual(self.st.session_state['dictionary'].iloc[0]['Tratamiento'],'Conservar')
    def test_new_session_restore_with_context_confirmation(self):
        ident=self.save_one();code=self.st.session_state[STATE_KEY]
        self.st.session_state={STATE_KEY:code,'mf_state':{'original':'keep'}}
        self.st.clicked={'ml_activate'};self.st.checked={'ml_allow_replace_'+ident}
        with self.assertRaises(Rerun):self.ui.render_library()
        self.assertEqual(self.st.session_state['mf_state'],{'original':'keep'})
        self.assertEqual(len(self.st.session_state['models']),1)
    def test_legacy_no_confirmation_does_not_save(self):
        self.prepare_active();self.st.clicked={'ml_save'}
        self.ui.render_library()
        self.assertNotIn('library_id',self.record)
    def test_no_secret_local_mode_opens(self):
        with mock.patch.dict(os.environ,{'PYC_MODEL_SIGNING_KEY':''}):
            self.ui.render_library()
        self.assertEqual(self.st.errors,[])
    def test_import_signed_backup(self):
        ident=self.save_one();data=ModelLibrary(Path(self.tmp.name),self.secret).backup(self.st.session_state[STATE_KEY],ident)
        self.st.session_state[STATE_KEY]=new_code()
        self.st.upload=types.SimpleNamespace(size=len(data),getvalue=lambda:data)
        self.st.clicked={'ml_import'}
        with self.assertRaises(Rerun):self.ui.render_library()
        self.assertEqual(len(ModelLibrary(Path(self.tmp.name),self.secret).list(self.st.session_state[STATE_KEY])),1)

if __name__=='__main__':unittest.main()
