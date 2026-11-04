import unittest
import numpy as np
import torch
from experiments.round65_run import synthetic_inputs,shared_digest
from experiments.round65_audit import independent_relation,independent_projection
from models.round65_actual_gs import NativeGSClassifier
from models.round69_compact_relation import descriptors
from models.round73_native_relative5 import CompactNativeClassifier


class NativeCompactTests(unittest.TestCase):
    def test_shared_initialization_and_no_relation_match(self):
        c,s,a=synthetic_inputs(2,12,'cpu')
        for seed in (11,22,33):
            torch.manual_seed(seed);old=NativeGSClassifier('nope').eval()
            torch.manual_seed(seed);new=CompactNativeClassifier('relative5').eval()
            self.assertEqual(shared_digest(old),shared_digest(new))
            torch.manual_seed(seed);distance=NativeGSClassifier('distance_mlp')
            self.assertEqual(shared_digest(distance),shared_digest(new))
            with torch.inference_mode():
                torch.testing.assert_close(old(c,s,a),new(c,s,a,disable_relation=True),atol=0,rtol=0)

    def test_independent_five_inputs_and_projection(self):
        c,s,_=synthetic_inputs(1,9,'cpu');torch.manual_seed(11);model=CompactNativeClassifier('relative5')
        expected=independent_relation(c[0].numpy(),s[0].numpy())[...,[0,7,8,9,10]]
        np.testing.assert_allclose(descriptors('relative5',c,s)[0].numpy(),expected,atol=2e-5,rtol=2e-5)
        reference=independent_projection(expected,model.position)
        with torch.inference_mode():actual=model.position(c,s)[0].numpy()
        np.testing.assert_allclose(actual,reference,atol=2e-5,rtol=2e-5)

    def test_shape_channels_receive_gradient(self):
        c,s,a=synthetic_inputs(2,13,'cpu');torch.manual_seed(11);model=CompactNativeClassifier('relative5')
        loss=torch.nn.functional.cross_entropy(model(c,s,a),torch.tensor([1,59]));loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            if parameter.grad is not None:self.assertTrue(torch.isfinite(parameter.grad).all())
        norms=model.position.mlp[0].weight.grad.norm(dim=0)
        self.assertTrue((norms>0).all())
        self.assertEqual(sum(p.numel() for p in model.position.parameters()),9792)


if __name__=='__main__':unittest.main()
