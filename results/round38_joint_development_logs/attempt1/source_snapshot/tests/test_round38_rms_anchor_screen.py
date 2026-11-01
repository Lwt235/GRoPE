import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments import round38_rms_anchor_contract as C
from experiments import round38_rms_anchor_screen as P


class ScreenTests(unittest.TestCase):
    def fixture(self,root):
        root.mkdir();(root/'prepared').mkdir();specs=[];rows=[]
        for index in range(3):
            spec=dict(id='toy%d'%index,pair_id='toy',scene='toy',pair=[0,1],pair_name='toy',query_size=24,
                query_index=index,query_bounds=[index*24,0,index*24+24,24]);specs.append(spec)
            arrays=dict(input__mu=np.array([0.,0.,1.]),input__factor=np.eye(3),
                input__bounds=np.array([[0.,0.,1.,1.]]),input__content_bias=np.zeros(1))
            row=dict(spec,**P.save_archive(root,root/'prepared'/('%d.npz'%index),arrays));rows.append(row)
        C.write(root/'prepared/records.json',rows);C.write(root/'query_specs.json',specs)
        C.write(root/'prepared_complete.json',{});C.write(root/'lock.json',{})
        return rows

    def run_forward(self,root,audit,model):
        audit.mkdir();C.write(audit/'lock.json',{});C.write(audit/'input_verification.json',{})
        with patch.object(P,'require_inputs',return_value=dict(policy=C.POLICY,gates=C.GATES)),patch.object(P.MODEL,'integrate',side_effect=model):
            P.forward(root,audit)
        return C.read(root/'forward/records.json'),C.read(root/'forward_complete.json')

    def test_unexpected_model_failure_retains_unknown_work_and_suffix(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';self.fixture(root)
            calls=[]
            def fail(*args,**kwargs):calls.append(1);raise RuntimeError('synthetic unexpected failure')
            rows,seal=self.run_forward(root,Path(temp)/'audit',fail)
            self.assertEqual(len(calls),1);self.assertEqual(len(rows),3)
            self.assertTrue(rows[0]['work_unavailable']);self.assertFalse(seal['phase_node_budget_passed'])
            for row in rows[1:]:
                self.assertFalse(row['attempted']);self.assertEqual(row['array_sha256'],{})
                self.assertIsNone(row['started_unix']);self.assertEqual(row['charged_cell_node_evaluations'],0)

    def test_input_delivery_failure_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';rows=self.fixture(root);(root/rows[0]['archive_path']).write_bytes(b'corrupted')
            calls=[]
            rows,seal=self.run_forward(root,Path(temp)/'audit',lambda *a,**k:calls.append(1))
            self.assertEqual(calls,[]);self.assertEqual(rows[0]['failure_stage'],'input_delivery')
            self.assertFalse(rows[0]['attempted']);self.assertFalse(seal['all_production_completed'])

    def test_deadline_before_read_retains_every_position(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';self.fixture(root)
            with patch.dict(C.GATES,production_max_phase_wall_seconds=0.):
                rows,seal=self.run_forward(root,Path(temp)/'audit',lambda *a,**k:None)
            self.assertEqual(len(rows),3);self.assertEqual(rows[0]['failure_stage'],'precall')
            self.assertFalse(any(r['attempted'] for r in rows));self.assertFalse(seal['all_production_completed'])

    def test_archive_never_overwrites_existing_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'a.npz';P.save_archive(root,path,{})
            with self.assertRaises(ValueError):P.save_archive(root,path,{})


if __name__=='__main__':unittest.main()
