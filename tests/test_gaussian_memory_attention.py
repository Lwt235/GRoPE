"""Geometry and scene-state invariants relevant to interpreting the reader experiment."""
import io
import unittest

import numpy as np
import torch

from models.gaussian_memory_attention import GaussianAttentionReader, geometry_inputs


class AttentionReaderTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(31)
        self.memory = {'means': torch.tensor([[0., 0., 2.], [.3, -.2, 3.]]),
                       'covars': torch.diag_embed(torch.tensor([[.04, .01, .09], [.02, .05, .01]])),
                       'opacities': torch.tensor([.7, .8]), 'features': torch.randn(2, 32)}
        self.camera = torch.eye(4)[None]
        self.K = torch.tensor([[[10., 0., 10.], [0., 10., 7.5], [0., 0., 1.]]])

    def test_analytic_projection(self):
        data = geometry_inputs(self.memory, self.camera, self.K)
        # On-axis z=2, fx=10: projected variance is 25 * xy variance + 0.1.
        self.assertAlmostEqual(data['projected_variance'][0, 0].item(), (1.1 + .35) / 2, places=6)
        # Pixel (10.5,7.5) is half a pixel right of the first center.
        self.assertAlmostEqual(data['mahalanobis'][0, 7 * 20 + 10, 0].item(), .25 / 1.1, places=6)

    def test_rigid_frame_and_memory_order(self):
        model = GaussianAttentionReader('gaussian', 1.).eval()
        base = model.read_memory(self.memory, self.camera, self.K)
        T = torch.eye(4)
        T[:3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        T[:3, 3] = torch.tensor([2., -3., 1.])
        m = dict(self.memory)
        m['means'] = self.memory['means'] @ T[:3, :3].T + T[:3, 3]
        m['covars'] = T[:3, :3] @ self.memory['covars'] @ T[:3, :3].T
        torch.testing.assert_close(base, model.read_memory(m, T[None] @ self.camera, self.K), rtol=2e-5, atol=2e-6)
        permuted = {k: v.flip(0) for k, v in self.memory.items()}
        torch.testing.assert_close(base, model.read_memory(permuted, self.camera, self.K), rtol=2e-5, atol=2e-6)

    def test_empty_memory_and_serialization(self):
        model = GaussianAttentionReader('gaussian', 1.).eval()
        m = dict(self.memory, opacities=torch.zeros(2))
        self.assertEqual(model.read_memory(m, self.camera, self.K).abs().max().item(), 0.)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer); buffer.seek(0)
        other = GaussianAttentionReader('gaussian', 1.).eval()
        other.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(model.read_memory(self.memory, self.camera, self.K),
                                   other.read_memory(self.memory, self.camera, self.K), rtol=0, atol=0)

    def test_paired_initialization_and_equal_geometry_limit(self):
        torch.manual_seed(70)
        center = GaussianAttentionReader('center', 2.)
        torch.manual_seed(70)
        gaussian = GaussianAttentionReader('gaussian', 2.)
        for key, value in center.state_dict().items():
            torch.testing.assert_close(value, gaussian.state_dict()[key], atol=0, rtol=0)
        d = geometry_inputs(self.memory, self.camera, self.K)
        d['mahalanobis'] = d['distance2'] / 2
        torch.testing.assert_close(center(d), gaussian(d), atol=0, rtol=0)
        loss = gaussian(d).square().mean()
        loss.backward()
        for name, parameter in gaussian.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(gaussian.position_scale.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
