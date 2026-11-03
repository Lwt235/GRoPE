"""Causal intervention and pairing checks for the retraining ablation."""
import unittest
import numpy as np  # Host runtime ordering.
import torch
from models.round65_actual_gs import NativeGSClassifier
from models.round65_content_route import ContentRouteClassifier, CONTENT_COLUMNS, ROLES
from experiments.round65_content_route import contrasts
from experiments.round65_common import SEEDS, VARIANTS


class ContentRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def inputs(self):
        g=torch.Generator().manual_seed(65004)
        centers=torch.randn(2,9,3,generator=g)
        factors=torch.randn(2,9,3,3,generator=g,dtype=torch.float64)*.03
        covariance=factors@factors.transpose(-1,-2)+.004*torch.eye(3,dtype=torch.float64)
        appearance=torch.randn(2,9,4,generator=g)
        return centers,covariance,appearance

    def test_original_initialization_and_rng_preserved(self):
        for role in ROLES:
            for seed in SEEDS:
                torch.manual_seed(seed);original=NativeGSClassifier(role);rng=torch.random.get_rng_state()
                torch.manual_seed(seed);reduced=ContentRouteClassifier(role)
                self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
                for name,tensor in original.state_dict().items():
                    expected=tensor[:,CONTENT_COLUMNS] if name=='stem.0.weight' else tensor
                    self.assertTrue(torch.equal(expected,reduced.state_dict()[name]),(role,seed,name))
                self.assertEqual(sum(x.numel() for x in original.parameters())-
                                 sum(x.numel() for x in reduced.parameters()),576)

    def test_exact_intervention_matches_masked_original(self):
        args=self.inputs()
        def mask(module,inputs):
            x=inputs[0].clone();x[...,3:9]=0
            return (x,)
        for role in ROLES:
            torch.manual_seed(11);original=NativeGSClassifier(role).eval()
            torch.manual_seed(11);reduced=ContentRouteClassifier(role).eval()
            handle=original.stem.register_forward_pre_hook(mask)
            with torch.no_grad():
                torch.testing.assert_close(reduced(*args),original(*args),atol=2e-6,rtol=2e-6)
            handle.remove()

    def test_covariance_removed_only_from_content(self):
        c,s,a=self.inputs()
        for role in ROLES:
            torch.manual_seed(11);model=ContentRouteClassifier(role).eval()
            with torch.no_grad():
                x=model(c,s,a);y=model(c,s*3,a)
            if role=='distance_mlp':
                self.assertTrue(torch.equal(x,y))
            else:
                self.assertGreater(float((x-y).abs().max()),1e-5)
            model.zero_grad();model(c,s,a).square().mean().backward()
            for name,param in model.named_parameters():
                self.assertIsNotNone(param.grad,name)
                self.assertTrue(torch.isfinite(param.grad).all(),name)
                self.assertGreater(float(param.grad.norm()),0,name)

    def synthetic_metrics(self):
        # Full-content Gaussian trails distance; removal reverses that gap.
        rates={'full':(.6,.7,.8),'removed':(.55,.5,.4)}
        return {(content,role,seed,variant):{'class_accuracy':[rate]*2+[None]*58}
                for content,values in rates.items() for role,rate in zip(ROLES,values)
                for seed in SEEDS for variant in VARIANTS}

    def test_factorial_contrast_signs_and_complete_coverage(self):
        metrics=self.synthetic_metrics();rows=contrasts(metrics,[0,1])
        def gain(kind,label):
            return next(r['mean_gain_pp'] for r in rows if r['variant']=='base' and r['kind']==kind and r['label']==label)
        self.assertAlmostEqual(gain('content_full_minus_removed','distance_mlp'),40)
        self.assertAlmostEqual(gain('gaussian_minus_control_full','distance_mlp'),-20)
        self.assertAlmostEqual(gain('gaussian_minus_control_removed','distance_mlp'),15)
        self.assertAlmostEqual(gain('gaussian_gap_removed_minus_full','distance_mlp'),35)
        metrics.pop(('removed','distance_mlp',33,'tilt'))
        with self.assertRaisesRegex(ValueError,'incomplete108'):
            contrasts(metrics,[0,1])


if __name__=='__main__':
    unittest.main()
