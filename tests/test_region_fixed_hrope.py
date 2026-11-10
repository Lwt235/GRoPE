"""The added control removes only the learned homography predictor."""
import numpy as np
import torch
import unittest

from experiments.round58_run import shared_digest
from models.region_hrope_control import CameraRegionMatcher, homographies
from models.region_fixed_hrope import FixedHRegionMatcher


class FixedHControlTest(unittest.TestCase):
    def test_initial_forward_common_gradients_and_fixed_geometry(self):
        torch.set_num_threads(4)
        torch.manual_seed(19)
        full = CameraRegionMatcher('predicted_hrope').double()
        torch.manual_seed(19)
        fixed = FixedHRegionMatcher().double()
        self.assertEqual(shared_digest(full), shared_digest(fixed))
        self.assertEqual(sum(p.numel() for p in fixed.position.parameters()), 0)
        self.assertEqual(sum(p.numel() for p in full.parameters()) -
                         sum(p.numel() for p in fixed.parameters()), 1261)
        b, v, n = 2, 2, 7
        features = torch.randn(b, v, n, 11, dtype=torch.float64)
        centers = torch.randn(b, v, n, 3, dtype=torch.float64)
        shape = torch.randn(b, v, n, 3, 3, dtype=torch.float64)
        covariance = shape @ shape.transpose(-1, -2) + torch.eye(3)
        xy = torch.randn(b, v, n, 2, dtype=torch.float64)
        k = torch.tensor([[1.2, .1, -.2], [0., .9, .15], [0., 0., 1.]],
                         dtype=torch.float64).expand(b, v, 3, 3).clone()
        k[:, 1, 0, 0] = .8
        permutation = torch.arange(n).expand(b, v, n)
        args = (features, centers, covariance, permutation, k, xy)
        a, c = full(*args), fixed(*args)
        torch.testing.assert_close(a, c, rtol=0, atol=1e-10)
        a.square().mean().backward()
        c.square().mean().backward()
        left = dict(full.named_parameters())
        for name, parameter in fixed.named_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            torch.testing.assert_close(parameter.grad, left[name].grad, rtol=1e-8, atol=1e-9)
        raw = fixed.position(torch.randn(b, v, n, 96, dtype=torch.float64))
        h, hi = homographies(raw, k.unsqueeze(-3))
        torch.testing.assert_close(h, k.unsqueeze(-3).expand_as(h), rtol=0, atol=0)
        torch.testing.assert_close(hi, torch.linalg.inv(k).unsqueeze(-3).expand_as(hi), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
