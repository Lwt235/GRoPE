"""Checks for attribution, matched content, native geometry and report direction."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from experiments.round65_run import configure, shared_digest, synthetic_inputs
from experiments.round65_audit import independent_relation, independent_projection
from experiments.native_region_pe_report import paired_summary, contrasts, PAIRS
from experiments.round65_common import SEEDS, VARIANTS
from models.native_region_pe import NativeRegionClassifier, ROLES, pool_regions, FLOOR
from models.round65_actual_gs import common_features
from models.round69_compact_relation import descriptors
from third_party.region_geotransformer_reference.embedding import GeometricStructureEmbedding


class NativeRegionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
        cls.args = synthetic_inputs(2, 128, 'cpu')
        cls.region = pool_regions(*cls.args)

    def test_independent_raw_second_moments_and_feature_mean(self):
        c, s, a = [x.numpy() for x in self.args]
        for b in range(2):
            for g in (0, 11, 63):
                ids = self.region['indices'][b, g].numpy()
                pts = c[b, ids].astype(np.float64)
                mean = pts.mean(0)
                second = (s[b, ids] + pts[:, :, None] * pts[:, None, :]).mean(0)
                expected = second - mean[:, None] * mean[None, :]
                np.testing.assert_allclose(self.region['covariance'][b, g], expected, atol=3e-14, rtol=3e-14)
                np.testing.assert_allclose(self.region['centers'][b, g], mean, atol=1e-7)
                np.testing.assert_allclose(self.region['appearance'][b, g], a[b, ids].mean(0), atol=1e-7)

    def test_shared_content_initialization_and_no_relation_path(self):
        shared, contents, outputs, f_states = [], [], [], []
        for role in ROLES:
            torch.manual_seed(11)
            model = NativeRegionClassifier(role).eval()
            shared.append(shared_digest(model))
            seen = []
            hook = model.stem.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach()))
            with torch.no_grad():
                outputs.append(model.forward_pooled(self.region, disable_relation=True))
            hook.remove()
            contents.append(seen[0])
            if role in ROLES[:2]:
                f_states.append(model.state_dict())
        self.assertEqual(len(set(shared)), 1)
        for content, out in zip(contents, outputs):
            self.assertTrue(torch.equal(content, contents[0]))
            self.assertTrue(torch.equal(out, outputs[0]))
        for name in f_states[0]:
            self.assertTrue(torch.equal(f_states[0][name], f_states[1][name]), name)

    def test_only_physical_pe_term_changes_and_floor_once(self):
        c, s, a = self.args
        zero_physical = torch.eye(3, dtype=s.dtype).expand_as(s).clone() * FLOOR
        z = pool_regions(c, zero_physical, a)
        torch.testing.assert_close(z['covariance'], z['centers_covariance'], atol=1e-15, rtol=1e-14)
        self.assertTrue(torch.equal(z['indices'], self.region['indices']))
        self.assertTrue(torch.equal(z['centers_covariance'], self.region['centers_covariance']))
        self.assertTrue(torch.equal(z['centers'], self.region['centers']))
        self.assertFalse(torch.allclose(z['covariance'], self.region['covariance']))
        point_a = descriptors('relative5', self.region['centers'], self.region['centers_covariance'])
        point_b = descriptors('relative5', z['centers'], z['centers_covariance'])
        self.assertTrue(torch.equal(point_a, point_b))
        full = descriptors('relative5', self.region['centers'], self.region['covariance'])
        self.assertTrue(torch.equal(full[..., 0], point_a[..., 0]))
        self.assertGreater(float((full[..., 1:] - point_a[..., 1:]).abs().max()), 1e-3)
        self.assertFalse(torch.equal(common_features(z['centers'], z['covariance'], z['appearance']),
                                     common_features(self.region['centers'], self.region['covariance'], self.region['appearance'])))

    def test_independent_five_descriptor_and_projection(self):
        for key in ('covariance', 'centers_covariance'):
            c, s = self.region['centers'][:1, :8], self.region[key][:1, :8]
            expected = independent_relation(c[0], s[0])[..., [0, 7, 8, 9, 10]]
            np.testing.assert_allclose(descriptors('relative5', c, s)[0], expected, atol=2e-6, rtol=2e-6)
            torch.manual_seed(11)
            model = NativeRegionClassifier('relative5')
            with torch.no_grad():
                actual = model.position(c, s)[0].numpy()
            np.testing.assert_allclose(actual, independent_projection(expected, model.position), atol=2e-6, rtol=2e-5)

    def test_rigid_geometry_equivariance(self):
        c, s, a = self.args
        c = c.double()
        angle = .47
        r = torch.tensor([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=torch.float64)
        shift = torch.tensor([.2, -.3, .1], dtype=torch.float64)
        x = pool_regions(c, s, a)
        y = pool_regions(c @ r.T + shift, r @ s @ r.T, a)
        self.assertTrue(torch.equal(x['indices'], y['indices']))
        torch.testing.assert_close(y['centers'], x['centers'] @ r.T + shift, atol=1e-12, rtol=1e-12)
        for key in ('covariance', 'centers_covariance'):
            torch.testing.assert_close(y[key], r @ x[key] @ r.T, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(descriptors('relative5', x['centers'], x[key]),
                                       descriptors('relative5', y['centers'], y[key]), atol=1e-10, rtol=1e-10)

    def test_postpool_permutation_for_all_roles(self):
        reverse = {k: v.flip(1) for k, v in self.region.items()}
        for role in ROLES:
            torch.manual_seed(11)
            model = NativeRegionClassifier(role).eval()
            with torch.no_grad():
                left, right = model.forward_pooled(self.region), model.forward_pooled(reverse)
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)

    def test_geo_matches_vendor_and_uses_four_projections(self):
        torch.manual_seed(11)
        model = NativeRegionClassifier('geotransformer_rpe')
        self.assertEqual(len(model.position.projections), 4)
        reference = GeometricStructureEmbedding(96, .2, 15., 3, reduction_a='max')
        reference.load_state_dict(model.position.embedding.state_dict())
        centers = self.region['centers']
        self.assertTrue(torch.equal(reference(centers), model.position(centers)))
        F.cross_entropy(model(*self.args), torch.tensor([2, 7])).backward()
        for i, projection in enumerate(model.position.projections):
            self.assertIsNotNone(projection.weight.grad)
            self.assertGreater(float(projection.weight.grad.norm()), 0., str(i))
        for name, p in model.position.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_paired_summary_and_contrast_sign(self):
        delta = np.ones((3, 60)) * 2
        summary = paired_summary(delta, 1, 200)
        self.assertEqual(summary['mean_gain_pp'], 2)
        self.assertEqual(summary['crossed_seed_class_bootstrap_95_pp'], [2, 2])
        scores = dict(relative5=.8, centers_only=.77, distance_mlp=.75, geotransformer_rpe=.74)
        rows = [dict(role=role, seed=seed, variant=variant,
                     metrics=dict(class_accuracy=[score] * 60, oa=score, ce=1-score))
                for role, score in scores.items() for seed in SEEDS for variant in VARIANTS]
        differences = contrasts(rows, 100)
        self.assertEqual(len(differences), len(PAIRS) * len(VARIANTS))
        for row in differences:
            expected = 100 * (scores[row['left']] - scores[row['right']])
            self.assertAlmostEqual(row['mean_gain_pp'], expected)
            self.assertTrue(all(v < 0 for v in row['paired_ce_difference']))


if __name__ == '__main__':
    unittest.main()
