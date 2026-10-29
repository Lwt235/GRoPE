import unittest
import torch
from models.geometry import covariance, covariance_moments, expected_phase, logcov_descriptor, project
from experiments.synthetic import make_dataset


class GeometryTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_covariance_distance_rotation_invariance(self):
        torch.manual_seed(10)
        r = torch.linalg.qr(torch.randn(4, 3, 3, dtype=torch.float64))[0]
        c = covariance(r, torch.rand(4, 3, dtype=torch.float64)+.1)
        g = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
        desc = logcov_descriptor(c)
        transformed = logcov_descriptor(g @ c @ g.T)
        torch.testing.assert_close(torch.cdist(desc, desc), torch.cdist(transformed, transformed), atol=1e-7, rtol=1e-7)

    def test_expected_phase_zero_cov_is_deterministic(self):
        m = torch.tensor([[.2, .4, -.5]])
        omega = torch.eye(3)
        c, s = expected_phase(m, torch.zeros(1, 3, 3), omega)
        torch.testing.assert_close(c, m.cos())
        torch.testing.assert_close(s, m.sin())

    def test_scenario_controls_hold_appearance_and_centers_fixed(self):
        baseline = make_dataset(123, 4)
        for scenario in ['shape_shuffled', 'shape_noise', 'independent_support', 'shared_scale', 'shared_normal', 'view_support']:
            changed = make_dataset(123, 4, scenario=scenario)
            for key in ['q', 'k', 'mq', 'mk', 'labels', 'center']:
                torch.testing.assert_close(baseline[key], changed[key], atol=0, rtol=0)
            self.assertFalse(torch.equal(baseline['ck'], changed['ck']))

    def test_moments_smooth_at_repeated_eigenvalues(self):
        c = torch.eye(3, dtype=torch.float64, requires_grad=True)
        covariance_moments(c).sum().backward()
        self.assertTrue(bool(torch.isfinite(c.grad).all()))
        torch.manual_seed(12)
        cov = torch.randn(4, 3, 3, dtype=torch.float64)
        cov = cov @ cov.transpose(-1, -2) + torch.eye(3)*.01
        g = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
        desc1 = covariance_moments(cov)
        desc2 = covariance_moments(7.3 * g @ cov @ g.T)
        torch.testing.assert_close(torch.cdist(desc1, desc1), torch.cdist(desc2, desc2), atol=1e-7, rtol=1e-7)

    def test_extreme_shape_noise_descriptors_are_finite(self):
        # Regression: seed 300022 produces a near-singular float32 covariance.
        data = make_dataset(300022, 192, scenario='shape_noise')
        for key in ['normal', 'scale', 'shape', 'projected', 'moments']:
            self.assertTrue(bool(torch.isfinite(data[key]).all()), key)

    def test_permutation_labels_and_ray_error(self):
        # Perfect centers recover permuted correspondences, independent of index.
        from experiments import synthetic
        synthetic.SCENARIOS['test_zero'] = {'depth_noise': 0.}
        try:
            data = make_dataset(321, 4, scenario='test_zero')
        finally:
            del synthetic.SCENARIOS['test_zero']
        self.assertTrue(torch.equal(data['center'].argmin(-1), data['labels']))
        noisy = make_dataset(321, 4)
        eye = torch.eye(3).repeat(4, 1, 1)
        uv0, _, _ = project(data['mq'], eye, torch.zeros(4, 3))
        uv1, _, _ = project(noisy['mq'], eye, torch.zeros(4, 3))
        torch.testing.assert_close(uv0, uv1, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
