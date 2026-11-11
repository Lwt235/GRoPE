"""Analytic plane checks for the reference-only VGGT/camera scale interface."""
import sys
import unittest

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT
from experiments.gaussian_memory_data import HEIGHT, WIDTH, PATCH, REF, source_geometry

sys.path.insert(0, str(ROOT / 'third_party/vggt'))


class ReferenceGeometryTest(unittest.TestCase):
    def test_known_plane_and_scale(self):
        torch.set_num_threads(4)
        # Predicted coordinates use half a metric unit; known baseline restores 2x.
        c2w = torch.eye(4)[None].repeat(13, 1, 1)
        c2w[:, 0, 3] = torch.arange(13) * .05
        K = torch.tensor([[140., 0., 140.], [0., 140., 105.], [0., 0., 1.]])[None].repeat(13, 1, 1)
        pose = torch.zeros(1, 4, 9)
        pose[0, :, 0] = -c2w[REF, 0, 3] / 2
        pose[0, :, 6] = 1  # VGGT is XYZW; scalar part is last.
        pose[0, :, 7:] = 1
        depth = torch.full((1, 4, HEIGHT, WIDTH, 1), 5.)
        conf = torch.ones(1, 4, HEIGHT, WIDTH)
        mean, cov, valid, camera, meta = source_geometry(depth, conf, pose, K, c2w)
        self.assertAlmostEqual(meta['metric_scale'], 2., places=5)
        self.assertAlmostEqual(meta['unit_meters'], 10., places=5)
        torch.testing.assert_close(camera[:, 0, 3], c2w[:, 0, 3] / 10)
        expected_first = torch.tensor([(PATCH / 2 - 140) / 140, (PATCH / 2 - 105) / 140, 1.])
        torch.testing.assert_close(mean[0], expected_first, atol=1e-6, rtol=1e-5)
        per_view = (HEIGHT // PATCH) * (WIDTH // PATCH)
        torch.testing.assert_close(mean[per_view] - mean[0], torch.tensor([.02, 0., 0.]), atol=1e-6, rtol=1e-5)
        variance = (PATCH**2 - 1) / 12 / 140**2
        torch.testing.assert_close(cov[0].diagonal(), torch.tensor([variance + .005**2, variance + .005**2, .005**2]),
                                   atol=1e-7, rtol=1e-4)
        self.assertTrue(valid.all().item())
        # Query camera changes may alter read coordinates, but cannot affect
        # written reference geometry, the reference scale or its covariance.
        changed = c2w.clone()
        changed[[1, 2, 3, 5, 6, 7, 9, 10, 11], :3, 3] += 100
        other = source_geometry(depth, conf, pose, K, changed)
        torch.testing.assert_close(mean, other[0], atol=0, rtol=0)
        torch.testing.assert_close(cov, other[1], atol=0, rtol=0)
        self.assertEqual(meta['unit_meters'], other[-1]['unit_meters'])


if __name__ == '__main__': unittest.main()
