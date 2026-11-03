import unittest
import numpy as np
import torch
from models.round62_rgb_relations import make_model, common_frame, FLOOR, POINT_VARIANCE
from models.round63_interventions import InterventionMatcher, INTERVENTIONS
from experiments.round63_diagnostic import covariance_distance, covariance_range, native_coverage
from tests.test_round62_rgb_relations import fixture


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_original_branches_and_analysis_field_exclusion(self):
        data = fixture()
        for role in ('predicted_full','predicted_point','observed_full'):
            torch.manual_seed(11);original = make_model(role).double().eval()
            wrapped = InterventionMatcher(role,capture=role!='observed_full').double().eval()
            wrapped.load_state_dict(original.state_dict())
            with torch.no_grad():
                expected = original(data);actual = wrapped(data)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                self.assertEqual(len(wrapped.captured),0 if role=='observed_full' else 4)
                poisoned = dict(data,gt_valid=torch.zeros(1,7,dtype=torch.bool),
                    qnative_valid=torch.ones(1,7,dtype=torch.bool),kpatch_count=torch.zeros(1,7))
                if role!='observed_full':
                    poisoned.update({k:torch.full_like(v,float('nan')) for k,v in data.items() if 'observed' in k})
                torch.testing.assert_close(wrapped(poisoned),expected,atol=0,rtol=0)
            self.assertEqual(set(original.state_dict()),set(wrapped.state_dict()))

    def test_declared_geometry_replacements_and_zero_relation(self):
        data=fixture();torch.manual_seed(22)
        original=make_model('predicted_full').double().eval()
        with torch.no_grad():original.field['predictor'].weight.normal_(std=.02)
        features=torch.randn(1,7,96,dtype=torch.float64)
        for case in INTERVENTIONS:
            model=InterventionMatcher('predicted_full',case).double().eval()
            model.load_state_dict(original.state_dict())
            with torch.no_grad():
                encoded=model.encode(data)
                if case=='zero_relation':
                    expected=original.encode(data,zero_relation=True)
                    for a,b in zip(encoded,expected):torch.testing.assert_close(a,b,atol=0,rtol=0)
                for prefix in ('q','k'):
                    mean,cov=original.geometry(features,data,prefix)
                    om,oc=common_frame(data[prefix+'observed_mean'],data[prefix+'observed_cov'],data[prefix+'geom'])
                    expected_mean=om if case in ('observed_center','observed_both') else mean
                    expected_cov=oc if case in ('observed_shape','observed_both') else cov
                    if case in ('initial_shape','sphere'):
                        diagonal=torch.tensor([.025**2+FLOOR**2,.030**2+FLOOR**2,.010**2+FLOOR**2],dtype=torch.float64)
                        if case=='sphere':diagonal.fill_(POINT_VARIANCE)
                        r=data[prefix+'geom'][:,0,:9].reshape(-1,3,3)
                        expected_cov=(r@torch.diag(diagonal)@r.transpose(-1,-2))[:,None].expand_as(cov)
                    actual_mean,actual_cov=model.geometry(features,data,prefix)
                    torch.testing.assert_close(actual_mean,expected_mean,atol=1e-14,rtol=1e-14)
                    torch.testing.assert_close(actual_cov,expected_cov,atol=1e-14,rtol=1e-14)

    def test_native_missing_depth_strata(self):
        depth=np.full((144,192),3.)
        valid,count=native_coverage(depth)
        self.assertTrue(valid.all());self.assertTrue((count==64).all())
        depth[:4,:8]=0
        valid,count=native_coverage(depth)
        self.assertEqual(count[0],32);self.assertTrue(valid[0]);self.assertTrue((count[1:]==64).all())
        valid,count=native_coverage(np.zeros_like(depth))
        self.assertFalse(valid.any());self.assertFalse(count.any())

    def test_shape_distance_and_range_have_analytic_limits(self):
        covariance=np.diag(np.array([.025,.030,.010])**2)+FLOOR**2*np.eye(3)
        self.assertAlmostEqual(float(covariance_distance(4*covariance,covariance)),np.sqrt(3)*np.log(4),places=11)
        result=covariance_range(covariance)
        self.assertFalse(result['any_coordinate_outside'])
        self.assertLess(float(result['relative_frobenius']),1e-7)
        large=covariance.copy();large[0,0]=1+FLOOR**2
        result=covariance_range(large)
        self.assertTrue(result['diagonal_above'][0]);self.assertTrue(result['any_coordinate_outside'])
        self.assertGreater(float(result['relative_frobenius']),.9)


if __name__=='__main__':unittest.main()
