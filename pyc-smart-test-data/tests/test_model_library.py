"""Offline persistence/security tests; no private data, network or SDV training."""
import copy
import hmac
import json
import os
import secrets
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cloudpickle
import pandas as pd
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import model_library as ml


class FittedExample:
    def __init__(self):
        self.weights = np.array([2., 3., 5.])
    def fit(self, data):
        raise AssertionError('Restoring must never train')
    def sample(self, num_rows):
        return pd.DataFrame({'edad': np.resize(self.weights, num_rows)})


def fixture():
    tables = {
        'clientes':pd.DataFrame({'cliente_id':[1,2], 'edad':[20,50]}),
        'ventas':pd.DataFrame({'venta_id':[1,2], 'cliente_id':[1,2], 'monto_venta':[10.,20.]}),
        'facturas':pd.DataFrame({'factura_id':[1,2], 'venta_id':[1,2], 'monto_total':[11.8,23.6]}),
        'pagos':pd.DataFrame({'pago_id':[1,2], 'factura_id':[1,2], 'monto_pago':[11.8,23.6]})}
    dictionary = pd.DataFrame([{'Entidad':'clientes','Columna':'edad','Rol':'Numérica',
                               'Confidencial':'No','Tratamiento':'Sintetizar','Dominio permitido':'','Aprobado':True}])
    record = {'name':'Demo ficticia', 'model':FittedExample(), 'train_sec':12.5,
              'params':{'épocas':50,'batch':100,'diccionario':dictionary.to_json(orient='records')}}
    return record,tables,dictionary


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.secret = secrets.token_hex(32)
        self.store = ml.ModelLibrary(self.root/'one',self.secret)
        self.code = ml.new_code()
        self.record,self.tables,self.dictionary = fixture()
        self.payload = ml.capture_model(self.record,self.tables,self.dictionary,confirm_legacy=True)
    def tearDown(self):
        self.tmp.cleanup()
    def test_full_model_is_retained(self):
        identifier=self.store.save(self.code,self.payload)
        loaded=self.store.load(self.code,identifier)
        np.testing.assert_array_equal(loaded['model']['model'].weights, self.record['model'].weights)
        self.assertEqual(len(loaded['model']['model'].sample(9)),9)
    def test_no_training_is_called(self):
        loaded=self.store.decode(self.store.seal(self.payload))
        loaded['model']['model'].sample(3)  # .fit raises if invoked
    def test_new_server_instance_same_root(self):
        identifier=self.store.save(self.code,self.payload)
        next_store=ml.ModelLibrary(self.root/'one',self.secret)
        self.assertEqual(len(next_store.list(self.code)),1)
        self.assertEqual(next_store.load(self.code,identifier)['model']['name'],'Demo ficticia')
    def test_new_root_from_download_with_stable_key(self):
        identifier=self.store.save(self.code,self.payload)
        data=self.store.backup(self.code,identifier)
        elsewhere=ml.ModelLibrary(self.root/'new-instance',self.secret)
        newcode=ml.new_code()
        elsewhere.import_backup(newcode,data)
        restored=elsewhere.load(newcode,identifier)
        self.assertEqual(restored['model']['params']['épocas'],50)
    def test_namespace_isolation(self):
        identifier=self.store.save(self.code,self.payload)
        other=ml.new_code()
        self.assertEqual(self.store.list(other),[])
        with self.assertRaises(ml.LibraryError):self.store.load(other,identifier)
    def test_changed_payload_rejected_before_unpickle(self):
        data=bytearray(self.store.seal(self.payload));data[-33]^=1
        with mock.patch('cloudpickle.loads') as loader:
            with self.assertRaises(ml.LibraryError):self.store.decode(bytes(data))
            loader.assert_not_called()
    def test_unauthenticated_pickle_rejected(self):
        data=cloudpickle.dumps({'bad':'not a signed model'})
        with mock.patch('cloudpickle.loads') as loader:
            with self.assertRaises(ml.LibraryError):self.store.decode(data)
            loader.assert_not_called()
    def test_foreign_signer_rejected(self):
        elsewhere=ml.ModelLibrary(self.root/'other',secrets.token_hex(32))
        with self.assertRaises(ml.LibraryError):elsewhere.decode(self.store.seal(self.payload))
    def test_versions_checked_before_unpickle(self):
        data=self.store.seal(self.payload)
        versions=ml.versions();versions['torch']='incompatible'
        with mock.patch.object(ml,'versions',return_value=versions),mock.patch('cloudpickle.loads') as loader:
            with self.assertRaises(ml.LibraryError):self.store.decode(data)
            loader.assert_not_called()
    def test_local_mode_refuses_portable_backup(self):
        local=ml.ModelLibrary(self.root/'local')
        identifier=local.save(self.code,self.payload)
        with self.assertRaises(ml.LibraryError):local.backup(self.code,identifier)
        with self.assertRaises(ml.LibraryError):local.import_backup(self.code,b'anything')
    def test_local_signing_key_survives_new_object(self):
        local=ml.ModelLibrary(self.root/'local')
        identifier=local.save(self.code,self.payload)
        again=ml.ModelLibrary(self.root/'local')
        self.assertEqual(again.load(self.code,identifier)['model']['name'],'Demo ficticia')
    def test_adding_stable_key_can_resign_existing_local_checkpoint(self):
        local=ml.ModelLibrary(self.root/'local'); identifier=local.save(self.code,self.payload)
        promoted=ml.ModelLibrary(self.root/'local',self.secret)
        backup=promoted.backup(self.code,identifier)
        self.assertEqual(self.store.decode(backup)['model']['library_id'],identifier)
    def test_capture_does_not_modify_live_objects(self):
        self.assertNotIn('library_id',self.record)
        self.payload['tables']['clientes'].loc[0,'edad']=999
        self.assertEqual(self.tables['clientes'].loc[0,'edad'],20)
    def test_legacy_requires_confirmation(self):
        with self.assertRaises(ml.LibraryError):ml.capture_model(self.record,self.tables,self.dictionary)
    def test_no_false_provenance(self):
        self.assertNotIn('source_fingerprint',self.payload['model'])
        self.assertEqual(self.payload['model']['library_provenance'],'confirmed_by_user_at_save')
    def test_known_fingerprint_mismatch_rejected(self):
        self.record['source_fingerprint']='wrong'
        with self.assertRaises(ml.LibraryError):ml.capture_model(self.record,self.tables,self.dictionary,confirm_legacy=True)
    def test_changed_dictionary_not_silently_paired(self):
        other=self.dictionary.copy();other.loc[0,'Tratamiento']='Conservar'
        with self.assertRaises(ml.LibraryError):ml.capture_model(self.record,self.tables,other,confirm_legacy=True)
    def test_dictionary_reordering_does_not_change_signature(self):
        other=self.dictionary[list(reversed(self.dictionary.columns))]
        self.assertEqual(ml.policy_signature(other),ml.policy_signature(self.dictionary))
    def test_reference_preserves_values_and_types(self):
        loaded=self.store.decode(self.store.seal(self.payload))
        for name,frame in self.tables.items():pd.testing.assert_frame_equal(frame,loaded['tables'][name])
        pd.testing.assert_frame_equal(self.dictionary,loaded['dictionary'])
    def test_activate_matching_adds_without_deleting_runs(self):
        runs=[{'nombre':'preservar'}];mf={'private':'mainframe'}
        state={'tables':self.tables,'dictionary':self.dictionary,'models':[], 'runs':runs,'mf_workspace':mf}
        self.assertTrue(ml.activate(state,self.payload))
        self.assertIs(state['runs'],runs);self.assertIs(state['mf_workspace'],mf)
        self.assertEqual(len(state['models']),1)
    def test_activate_twice_no_duplicate(self):
        state={'tables':self.tables,'dictionary':self.dictionary,'models':[],'runs':[]}
        ml.activate(state,self.payload);ml.activate(state,self.payload)
        self.assertEqual(len(state['models']),1)
    def test_restore_different_base_needs_confirmation_and_is_atomic(self):
        state={'tables':{'other':pd.DataFrame({'x':[1]})},'models':['old'],'runs':['old'],'mf_state':{'x':1}}
        before=copy.deepcopy(state)
        with self.assertRaises(ml.LibraryError):ml.activate(state,self.payload)
        self.assertEqual(state['models'],before['models']);self.assertEqual(state['runs'],before['runs'])
        self.assertIn('other',state['tables'])
    def test_restore_new_session_without_fit(self):
        mf={'x':1};state={'mf_state':mf,'dictionary_editor_clientes':{'old':1}}
        self.assertFalse(ml.activate(state,self.payload,replace_reference=True))
        self.assertIs(state['mf_state'],mf);self.assertEqual(state['runs'],[])
        self.assertNotIn('dictionary_editor_clientes',state)
        self.assertEqual(len(state['models'][0]['model'].sample(4)),4)
    def test_individual_delete_keeps_other_saved_models(self):
        i1=self.store.save(self.code,self.payload)
        other=ml.capture_model(self.record,self.tables,self.dictionary,confirm_legacy=True)
        i2=self.store.save(self.code,other)
        self.store.delete(self.code,i1)
        self.assertEqual([r['id'] for r in self.store.list(self.code)],[i2])
    def test_delete_active_keeps_generation_history(self):
        identifier=self.payload['model']['library_id'];runs=[{'modelo':'Demo ficticia'}]
        state={'models':[self.payload['model'],{'library_id':'different'}],'runs':runs}
        ml.remove_active(state,identifier)
        self.assertEqual(len(state['models']),1);self.assertIs(state['runs'],runs)
    def test_file_and_directory_permissions_private(self):
        identifier=self.store.save(self.code,self.payload)
        path=self.store._path(self.code,identifier)
        self.assertEqual(path.stat().st_mode & 0o777,0o600)
        self.assertEqual(path.parent.stat().st_mode & 0o777,0o700)
    def test_path_traversal_rejected(self):
        with self.assertRaises(ml.LibraryError):self.store.read_bytes('../bad','x')
        with self.assertRaises(ml.LibraryError):self.store.read_bytes(self.code,'../key')
    def test_file_symlink_rejected(self):
        identifier=self.store.save(self.code,self.payload);p=self.store._path(self.code,identifier)
        p.unlink();p.symlink_to(self.root/'missing')
        with self.assertRaises(ml.LibraryError):self.store.read_bytes(self.code,identifier)
    def test_quota_does_not_delete_previous(self):
        identifier=self.store.save(self.code,self.payload)
        other=ml.capture_model(self.record,self.tables,self.dictionary,confirm_legacy=True)
        with mock.patch.object(ml,'MAX_MODELS',1):
            with self.assertRaises(ml.LibraryError):self.store.save(self.code,other)
        self.assertEqual(len(self.store.list(self.code)),1)
        self.store.load(self.code,identifier)
    def test_idempotent_save_overwrites_only_same_id(self):
        identifier=self.store.save(self.code,self.payload);self.store.save(self.code,self.payload)
        self.assertEqual(len(self.store.list(self.code)),1)
        self.assertEqual(self.store.list(self.code)[0]['id'],identifier)
    def test_secret_not_in_backup(self):
        data=self.store.seal(self.payload)
        self.assertNotIn(self.secret.encode(),data);self.assertNotIn(self.code.encode(),data)
    def test_listing_does_not_deserialize_models(self):
        self.store.save(self.code,self.payload)
        with mock.patch('cloudpickle.loads') as loader:
            self.assertEqual(len(self.store.list(self.code)),1)
            loader.assert_not_called()
    def test_small_malformed_input_rejected(self):
        for data in (b'',b'notamodel',b'PYC_MODEL_V1\x00'):
            with self.assertRaises(ml.LibraryError):self.store.decode(data)
    def test_torch_weights_roundtrip(self):
        import torch
        class TorchExample:
            def __init__(self):self.layer=torch.nn.Linear(3,2)
            def sample(self,num_rows):return self.layer(torch.zeros(num_rows,3)).detach().numpy()
        self.record['model']=TorchExample()
        payload=ml.capture_model(self.record,self.tables,self.dictionary,confirm_legacy=True)
        loaded=self.store.decode(self.store.seal(payload))
        torch.testing.assert_close(loaded['model']['model'].layer.weight,self.record['model'].layer.weight)
        self.assertEqual(loaded['model']['model'].sample(5).shape,(5,2))

if __name__=='__main__':unittest.main()
