import unittest
import numpy as np
import torch
from models.native_shape_attribution import NativeFixedIsotropic
from models.round73_native_relative5 import CompactNativeClassifier
from experiments.round65_run import synthetic_inputs


class NativeShapeAttributionTests(unittest.TestCase):
    def test_initialization_and_full_forward_match_explicit_fixed_PE(self):
        for seed in (11, 22, 33):
            torch.manual_seed(seed)
            reference = CompactNativeClassifier('relative5').eval()
            torch.manual_seed(seed)
            actual = NativeFixedIsotropic().eval()
            self.assertEqual(set(reference.state_dict()), set(actual.state_dict()))
            for name, value in reference.state_dict().items():
                self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)
            self.assertEqual(sum(p.numel() for p in actual.parameters()), 352220)
            self.assertEqual(sum(p.numel() for p in actual.position.parameters()), 9792)

            def explicit_PE(module, args):
                center, covariance = args
                fixed = (.04 * torch.eye(3, dtype=covariance.dtype, device=covariance.device)).expand_as(covariance)
                return center, fixed

            reference.position.register_forward_pre_hook(explicit_PE)
            inputs = synthetic_inputs(2, 9, 'cpu')
            with torch.inference_mode():
                torch.testing.assert_close(actual(*inputs), reference(*inputs), atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
