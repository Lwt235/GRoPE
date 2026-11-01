"""Independent moment checks and rejection of corrupted synthetic evidence."""
import ast
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from experiments import round38_rms_anchor_inputs as I
from experiments import round38_rms_anchor_audit as A
from experiments import round38_rms_anchor_arithmetic as AR
from test_round38_rms_anchor_inputs import observations,cache


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.o=observations();self.c=cache();self.b=I.prepare_inputs(self.o,self.c)
        self.guard=patch.object(np,'load',side_effect=AssertionError('no archives in geometry audit tests'))
        self.guard.start();self.addCleanup(self.guard.stop)

    def test_independent_direct_moments_agree(self):
        checked=A.pair_check(self.o,self.c,self.b)
        self.assertTrue(checked['passed']);self.assertLess(checked['maximum_normalized_error'],1e-12)

    def test_extra_prepared_fields_and_factor_rows_rejected(self):
        b=dict(self.b);b['unreviewed_extra']=np.ones(1)
        with self.assertRaisesRegex(ValueError,'pair fields'):A.pair_check(self.o,self.c,b)
        for name in ('source_factor','target_factor','factor_homogeneous','anchored_factor'):
            b=dict(self.b);b[name]=np.concatenate((b[name],b[name][:1]))
            with self.assertRaisesRegex(ValueError,'factor population'):A.pair_check(self.o,self.c,b)

    def test_center_ray_depth_differs_from_patch_average(self):
        self.o['q_predicted_depth'][11:13,11:13]=3.
        b=I.prepare_inputs(self.o,self.c);checked=A.pair_check(self.o,self.c,b)
        self.assertTrue(checked['passed']);self.assertNotEqual(b['anchor_depth'][0],b['source_mean'][0,2])
        self.assertGreater(np.linalg.norm(b['anchor_delta'][0]),0)

    def test_anchor_and_covariance_tampering_rejected(self):
        for name,index in (('anchor_homogeneous',(0,0)),('anchor_covariance',(0,0,0)),('anchored_factor',(0,0,0)),('source_factor',(0,0,0))):
            b={k:v.copy() for k,v in self.b.items()};b[name][index]+=.01
            self.assertFalse(A.pair_check(self.o,self.c,b)['passed'],name)

    def test_missing_within_pixel_covariance_rejected(self):
        self.o['q_pixel_cov3']*=0
        b=I.prepare_inputs(self.o,self.c)
        self.assertFalse(A.pair_check(self.o,self.c,b)['passed'])

    def test_bias_tampering_rejected(self):
        self.b['bias32'][0,0]=np.nextafter(self.b['bias32'][0,0],np.float32(np.inf))
        self.assertFalse(A.pair_check(self.o,self.c,self.b)['passed'])

    def test_negative_depth_not_filtered(self):
        self.o['q_to_k_transform'][2,3]=-3
        b=I.prepare_inputs(self.o,self.c)
        self.assertTrue(A.pair_check(self.o,self.c,b)['passed'])
        self.assertFalse(b['anchor_strict_front'].any())

    def test_exact_query_copy_and_corruption(self):
        q={'input__'+k:v for k,v in I.model_inputs(self.b,3).items()}
        self.assertTrue(A.query_check(self.b,q,3)['passed'])
        q['input__content_bias'][0]+=1e-15
        if np.array_equal(q['input__content_bias'],self.b['content_bias64'][3]):q['input__content_bias'][0]+=1e-6
        self.assertFalse(A.query_check(self.b,q,3)['passed'])

    def test_log_arithmetic_source_is_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        old=ast.parse((root/'experiments/round30_log_rms_applicability_audit.py').read_text())
        new=ast.parse((root/'experiments/round38_rms_anchor_arithmetic.py').read_text())
        old={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
        for node in new.body:
            if isinstance(node,ast.FunctionDef):self.assertEqual(ast.dump(node,include_attributes=False),old[node.name])

    def test_zero_law_with_nonzero_content_is_not_uniform(self):
        bounds=np.array([[0.,0.,1.,1.],[1.,0.,2.,1.]])
        p=AR.probabilities(np.full(2,-np.inf),bounds,np.array([0.,1.]),1e-30)['probability']
        np.testing.assert_allclose(p,np.array([1.,np.e])/(1+np.e),rtol=2e-14)

    def test_tiny_positive_log_mass_not_mistaken_for_zero(self):
        raw=np.array([-900.,-901.])
        self.assertTrue(np.isfinite(AR.validate_logs(raw,False)).all())
        with self.assertRaises(ValueError):AR.validate_logs(np.array([-np.inf,-900.]),False)


class PhaseEvidenceTests(unittest.TestCase):
    def evidence(self,wall=1.):
        rows=[dict(attempted=True,passed=True,work_unavailable=False) for _ in range(2304)]
        forward=dict(wall_seconds=wall,charged_cell_node_evaluations=0,stop_trigger=None,
            phase_wall_passed=wall<=7200,phase_node_budget_passed=True,work_unavailable=False,
            all_production_completed=True,complete_case_inventory=True)
        seal=dict(phase_wall_seconds=wall,charged_cell_node_evaluations=0,stop_trigger=None,
            phase_wall_passed=wall<=7200,phase_node_budget_passed=True,all_production_completed=True,
            case_count=2304,actual_calls=2304,completed_calls=2304)
        return rows,seal,forward

    def test_complete_synthetic_population_resource_evidence(self):
        self.assertEqual(A.phase_check(*self.evidence(),0,None),
            dict(ledger_passed=True,all_production_completed=True))

    def test_matching_failed_wall_flags_cannot_approve_complete_claim(self):
        rows,seal,forward=self.evidence(wall=7201.)
        checked=A.phase_check(rows,seal,forward,0,None)
        self.assertFalse(checked['ledger_passed']);self.assertFalse(checked['all_production_completed'])
        seal['all_production_completed']=forward['all_production_completed']=False
        checked=A.phase_check(rows,seal,forward,0,None)
        self.assertTrue(checked['ledger_passed']);self.assertFalse(checked['all_production_completed'])

    def test_unknown_work_and_forged_call_count_rejected(self):
        rows,seal,forward=self.evidence();rows[0]['work_unavailable']=True
        self.assertFalse(A.phase_check(rows,seal,forward,0,None)['ledger_passed'])
        rows,seal,forward=self.evidence();seal['actual_calls']-=1
        self.assertFalse(A.phase_check(rows,seal,forward,0,None)['ledger_passed'])

    def test_chronology_rejects_production_before_input_audit_and_nonfinite(self):
        A.check_chronology(1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.)
        # Input-audit completion at 8, followed by falsely declared production at 7.
        with self.assertRaisesRegex(ValueError,'chronology'):A.check_chronology(1.,2.,3.,4.,5.,6.,7.,8.,7.,10.,11.,12.)
        for invalid in (float('nan'),float('inf'),None,True):
            with self.assertRaisesRegex(ValueError,'chronology'):A.check_chronology(1.,invalid,3.)


if __name__=='__main__':unittest.main()
