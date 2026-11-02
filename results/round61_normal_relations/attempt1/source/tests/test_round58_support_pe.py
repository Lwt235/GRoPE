"""Bounded scientific checks: formulas, data semantics, actual model behavior."""
import unittest
import numpy as np
import torch
from models.round58_support_pe import SupportMatcher, scalar_geometry, ROLES
from experiments.round58_data import draw_pair, tokenize, loss_function, random_rotation
from experiments.round58_audit import direct_gaussian_distance, metrics_numpy


class SupportTests(unittest.TestCase):
    def test_divergence_and_isotropic_limit(self):
        rng = np.random.default_rng(5801)
        c = rng.normal(size=(7, 3))
        l = rng.normal(size=(7, 3, 3))
        s = l@l.swapaxes(-1, -2)+np.eye(3)*.05
        _, metric, shape = scalar_geometry(torch.from_numpy(c), torch.from_numpy(s))
        np.testing.assert_allclose((.5*(metric+shape)).sqrt().numpy(), direct_gaussian_distance(c, s), atol=2e-12)
        sigma = .2
        equal = np.broadcast_to(np.eye(3)*sigma**2, s.shape).copy()
        d2, metric, shape = scalar_geometry(torch.from_numpy(c), torch.from_numpy(equal))
        np.testing.assert_allclose((.5*(metric+shape)).sqrt().numpy(), np.sqrt(d2.numpy())/sigma, atol=2e-12)

    def test_sampling_and_overlap(self):
        rng = np.random.default_rng(5802)
        points = rng.normal(size=(2048, 3)).astype(np.float32)*.25
        a = draw_pair(points, 'synthetic', 0)
        b = draw_pair(points, 'synthetic', 0)
        self.assertEqual(np.intersect1d(*a['source_ids']).size, 0)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        sparse = draw_pair(points, 'synthetic', 0, 'sparse256')
        np.testing.assert_array_equal(sparse['points'], a['points'][:, :256])
        batch = {k: torch.from_numpy(v[None]) for k, v in a.items()}
        token = tokenize(batch, tokens=8, neighbors=16)
        ids = token['neighbors'][0].numpy()
        for i in range(8):
            for j in range(8):
                x, y = a['clean'][0, ids[0, i]].astype(float), a['clean'][1, ids[1, j]].astype(float)
                mask = np.square(x[:, None]-y[None]).sum(-1) < .05**2
                expected = .5*(mask.any(0).mean()+mask.any(1).mean())
                self.assertAlmostEqual(float(token['overlap'][0, i, j]), expected)

    def test_model_invariance_and_gradients(self):
        rng = np.random.default_rng(5803)
        c = torch.from_numpy(rng.normal(size=(2, 2, 9, 3)))
        l = torch.from_numpy(rng.normal(size=(2, 2, 9, 3, 3)))
        covariance = l@l.transpose(-1, -2)+torch.eye(3)*.1
        features = torch.from_numpy(rng.normal(size=(2, 2, 9, 11)))
        rotation = torch.from_numpy(np.stack([random_rotation(rng) for _ in range(2)]))
        moved = c@rotation.transpose(-1, -2)+torch.tensor([1., -.2, .3])
        changed_cov = rotation[:, None]@covariance@rotation[:, None].transpose(-1, -2)
        overlap = torch.from_numpy(rng.uniform(size=(2, 9, 9)))
        shared = None
        for role in ROLES:
            torch.manual_seed(11)
            model = SupportMatcher(role).double()
            parameters = {k: p.detach().clone() for k, p in model.named_parameters() if not k.startswith('position.')}
            if shared is None:
                shared = parameters
            for key in shared:
                torch.testing.assert_close(parameters[key], shared[key], atol=0, rtol=0)
            output = model(features, c, covariance)
            torch.testing.assert_close(output, model(features, moved, changed_cov), atol=2e-10, rtol=2e-10)
            permuted = model(features.flip(-2), c.flip(-2), covariance.flip(-3)).flip(-2, -1)
            torch.testing.assert_close(output, permuted, atol=2e-10, rtol=2e-10)
            loss_function(output, overlap).backward()
            for parameter in model.position.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.norm()), 0.)

    def test_independent_metrics_and_empty_rows(self):
        from experiments.round58_run import metric_torch
        rng = np.random.default_rng(5804)
        score = rng.normal(size=(3, 7, 7))
        overlap = rng.uniform(size=(3, 7, 7))
        overlap[0] = 0
        world = rng.normal(size=(3, 2, 7, 3))
        a = metric_torch(score, overlap, world)
        b, _ = metrics_numpy(score, overlap, world)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=2e-12)
        empty = loss_function(torch.tensor(score), torch.zeros_like(torch.tensor(score)))
        self.assertEqual(float(empty), 0.)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
