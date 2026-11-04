import unittest
import numpy as np
import torch
from models.region_shape_attribution import FixedIsotropicMatcher, fixed_inputs
from models.round69_compact_relation import CompactMatcher, descriptors


class RegionShapeAttributionTests(unittest.TestCase):
    def test_full_initial_state_and_capacity_match(self):
        for seed in (11, 22, 33):
            torch.manual_seed(seed)
            original = CompactMatcher('relative5')
            torch.manual_seed(seed)
            control = FixedIsotropicMatcher()
            self.assertEqual(set(original.state_dict()), set(control.state_dict()))
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, control.state_dict()[name]), name)
            self.assertEqual(sum(p.numel() for p in control.parameters()), 325105)
            self.assertEqual(sum(p.numel() for p in control.position.parameters()), 9792)

    def test_matches_explicit_isotropic_covariance(self):
        torch.manual_seed(76001)
        center = torch.randn(2, 2, 9, 3)
        matrix = torch.randn(2, 2, 9, 3, 3)
        covariance = matrix @ matrix.transpose(-1, -2) + .01 * torch.eye(3)
        fixed = (.04 * torch.eye(3)).expand_as(covariance)
        actual = fixed_inputs(descriptors('relative5', center, covariance))
        expected = descriptors('relative5', center, fixed)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        control = FixedIsotropicMatcher()
        original = CompactMatcher('relative5')
        original.load_state_dict(control.state_dict())
        torch.testing.assert_close(control.position(center, covariance),
                                   original.position(center, fixed), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(control.position(center, covariance * 17),
                                   control.position(center, covariance), atol=0, rtol=0)

    def test_only_shape_inputs_are_removed_and_nonself_center_gradient_remains(self):
        torch.manual_seed(76002)
        center = torch.randn(7, 3, requires_grad=True)
        other = torch.randn(7, 3)
        distance = torch.linalg.vector_norm(center - other, dim=-1) / .2
        shape = torch.randn(7, 4, requires_grad=True)
        inputs = torch.cat((distance[:, None], shape), -1)
        fixed_inputs(inputs).square().mean().backward()
        self.assertTrue(torch.isfinite(center.grad).all())
        self.assertGreater(float(center.grad.norm()), 0)
        self.assertEqual(float(shape.grad.abs().max()), 0)


if __name__ == '__main__':
    unittest.main()
