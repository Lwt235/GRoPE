import unittest

import numpy as np
import torch

from models.gaussian_memory_attention import geometry_inputs
from models.gaussian_memory_view_token import ViewTokenAdapter, apply_rgb_mask, inject_last_layer


class ViewTokenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71); torch.set_num_threads(2)
        self.memory = {'means': torch.randn(9, 3), 'covars': torch.eye(3).repeat(9, 1, 1) * .1,
                       'features': torch.randn(9, 32), 'opacities': torch.rand(9)}
        self.memory['means'][:, 2] += 5
        self.pose = torch.eye(4)[None]
        self.K = torch.tensor([[[20., 0, 10], [0, 20., 7.5], [0, 0, 1.]]])
        self.geometry = geometry_inputs(self.memory, self.pose, self.K)
        self.x = torch.randn(1, 300, 2048)

    def test_initialization_and_actual_list_injection(self):
        for variant in ('local', 'center', 'gaussian'):
            net = ViewTokenAdapter(variant)
            self.assertTrue(torch.equal(net(self.x, self.geometry), self.x))
            tokens = [None] * 23 + [torch.cat([torch.randn(1, 5, 2048), self.x], 1)[:, None]]
            injected = inject_last_layer(tokens, 5, net, self.geometry)
            self.assertTrue(torch.equal(injected[-1], tokens[-1]))
            with torch.no_grad(): net.output.weight.normal_(0, .01)
            self.assertFalse(torch.equal(net(self.x, self.geometry), self.x))

    def test_order_invariance_empty_and_view_dependency(self):
        net = ViewTokenAdapter('center')
        with torch.no_grad(): net.output.weight.normal_(0, .01)
        expected = net(self.x, self.geometry)
        order = torch.randperm(9)
        permuted = geometry_inputs({k: v[order] for k, v in self.memory.items()}, self.pose, self.K)
        torch.testing.assert_close(net(self.x, permuted), expected)
        empty = dict(self.geometry, log_weight=torch.full_like(self.geometry['log_weight'], -torch.inf))
        torch.testing.assert_close(net(self.x, empty), net(self.x, self.geometry, 'removed'))
        self.assertTrue(torch.isfinite(net(self.x, empty)).all())
        self.assertGreater((net(self.x + torch.randn_like(self.x), self.geometry) - expected).abs().max().item(), .1)

    def test_mask_prevents_hidden_rgb_leakage(self):
        images = torch.rand(2, 3, 210, 280)
        masks = torch.zeros(2, 15, 20, dtype=torch.bool); masks[:, 3:10, 5:15] = True
        pixels = masks.repeat_interleave(14, 1).repeat_interleave(14, 2)[:, None]
        altered = torch.where(pixels, torch.rand_like(images), images)
        self.assertTrue(torch.equal(apply_rgb_mask(images, masks), apply_rgb_mask(altered, masks)))
        self.assertEqual(int(masks[0].sum()), 70)

    def test_common_rigid_coordinates(self):
        net = ViewTokenAdapter('center')
        with torch.no_grad(): net.output.weight.normal_(0, .01)
        R = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1.]])
        t = torch.tensor([.2, -.8, 1.3])
        moved = dict(self.memory, means=self.memory['means'] @ R.T + t,
                     covars=R @ self.memory['covars'] @ R.T)
        pose = self.pose.clone(); pose[0, :3, :3] = R; pose[0, :3, 3] = t
        g = geometry_inputs(moved, pose, self.K)
        torch.testing.assert_close(net(self.x, g), net(self.x, self.geometry), atol=2e-6, rtol=2e-5)


if __name__ == '__main__': unittest.main()
