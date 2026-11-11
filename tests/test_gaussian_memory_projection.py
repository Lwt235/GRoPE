"""The renderer-interface correction affects off-axis support, not common inputs."""
import unittest

import numpy as np
import torch

from models.gaussian_memory_attention import geometry_inputs
from models.gaussian_memory_projection import projected_inputs


class ProjectionInterfaceTest(unittest.TestCase):
    def test_off_axis_and_shared_inputs(self):
        torch.set_num_threads(4)
        memory = {'means': torch.tensor([[0., 0., 2.], [5., 0., .5]]),
                  'covars': torch.eye(3)[None].repeat(2, 1, 1) * .01,
                  'opacities': torch.ones(2) * .8, 'features': torch.ones(2, 32)}
        cam = torch.eye(4)[None]
        K = torch.tensor([[[10., 0., 10.], [0., 10., 7.5], [0., 0., 1.]]])
        base = geometry_inputs(memory, cam, K)
        fixed = projected_inputs(memory, cam, K)
        for key in ('query', 'key', 'value', 'distance2', 'valid', 'log_weight'):
            torch.testing.assert_close(base[key], fixed[key], atol=0, rtol=0)
        torch.testing.assert_close(base['mahalanobis'][..., 0], fixed['mahalanobis'][..., 0])
        # The old Jacobian spreads an off-screen center across the image.
        self.assertTrue((fixed['mahalanobis'][..., 1] > base['mahalanobis'][..., 1] * 10).all())
        self.assertTrue((fixed['projected_variance'][..., 1] < base['projected_variance'][..., 1] / 10).all())


if __name__ == '__main__':
    unittest.main()
