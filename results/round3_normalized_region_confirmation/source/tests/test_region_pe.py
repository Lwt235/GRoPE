"""Independent contract checks for regional supervision and PE operators."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from experiments.region_pe_probe import soft_loss, evaluate, real_dataset
from models.gaussian_pe import GaussianPEMatcher, REGION_METHODS, spectral_rule, density_features


class SuppliedLogits(nn.Module):
    def forward(self, data):
        return data['test_logits']


def supervision_fixture():
    # Masked candidate logits and the unused dustbin are deliberately enormous.
    logits = torch.tensor([[[math.log(2), 999., 0., 1e6], [7., 4., -9., -1e6]],
                           [[-6., 3., 8., 1e6], [0., -999., math.log(3), 1e6]]],
                          dtype=torch.float64)
    return {'q': torch.zeros(2, 2, 108),
            'test_logits': logits,
            'soft_labels': torch.tensor([[[.75, 0., .25], [0., 1., 0.]],
                                         [[0., 1., 0.], [.1, 0., .9]]], dtype=torch.float64),
            'label_valid': torch.tensor([[True, False], [False, True]]),
            'key_valid': torch.tensor([[True, False, True], [True, False, True]])}


def positional_fixture(count=4):
    generator = torch.Generator().manual_seed(719)
    rgb = torch.rand(1, count, 108, generator=generator)
    mean = torch.randn(1, count, 2, generator=generator)
    a = torch.randn(1, count, 2, 2, generator=generator)
    covariance = a @ a.transpose(-1, -2) * .1 + torch.eye(2) * .02
    world = torch.randn(1, count, 3, generator=generator)
    a3 = torch.randn(1, count, 3, 3, generator=generator)
    world_covariance = a3 @ a3.transpose(-1, -2)*.02 + torch.eye(3)*.01
    return {'q': rgb, 'k': rgb.clone(), 'pixq': mean, 'k_to_q': mean.clone(),
            'mq': world, 'mk': world.clone(), 'projected_qcov': covariance,
            'projected_kcov': covariance.clone(),
            'cq': world_covariance, 'ck': world_covariance.clone(),
            'soft_labels': torch.eye(count)[None],
            'label_valid': torch.ones(1, count, dtype=torch.bool),
            'key_valid': torch.ones(1, count, dtype=torch.bool),
            'retained_label_mass': torch.ones(1, count)}


class RegionalSupervisionTests(unittest.TestCase):
    def test_soft_loss_masks_candidates_rows_and_dustbin(self):
        data = supervision_fixture()
        logits = data['test_logits'].clone().requires_grad_()
        loss = soft_loss(logits, data)
        # Two independent categorical distributions: (2/3,1/3), (1/4,3/4).
        expected = (.75*math.log(1.5)+.25*math.log(3)
                    + .1*math.log(4)+.9*math.log(4/3))/2
        self.assertAlmostEqual(float(loss), expected, places=12)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[..., 1], torch.zeros(2, 2, dtype=torch.float64))
        torch.testing.assert_close(logits.grad[..., -1], torch.zeros(2, 2, dtype=torch.float64))
        torch.testing.assert_close(logits.grad[~data['label_valid']], torch.zeros(2, 4, dtype=torch.float64))

    def test_evaluator_matches_loss_and_ignores_invalid_rows(self):
        data = supervision_fixture()
        metrics, per = evaluate(SuppliedLogits(), data, batch=1)
        self.assertAlmostEqual(metrics['cross_entropy'], float(soft_loss(data['test_logits'], data)), places=12)
        self.assertEqual(metrics['valid_query_rows'], 2)
        self.assertAlmostEqual(metrics['top1_target_mass'], (.75+.9)/2, places=12)
        self.assertEqual(metrics['dominant_region_accuracy'], 1.)
        np.testing.assert_array_equal(per['valid_rows'], [1, 1])
        original_metrics = metrics
        changed = dict(data)
        changed['soft_labels'] = data['soft_labels'].clone()
        changed['test_logits'] = data['test_logits'].clone()
        changed['soft_labels'][~data['label_valid']] = torch.tensor([.9, 0., .1], dtype=torch.float64)
        changed['test_logits'][~data['label_valid']] = torch.tensor([1e4, -1e4, -1e4, 1e4], dtype=torch.float64)
        changed['test_logits'][..., -1] *= -1
        changed_metrics, _ = evaluate(SuppliedLogits(), changed, batch=2)
        self.assertEqual(original_metrics, changed_metrics)

    def test_perfect_distribution_has_zero_divergence(self):
        data = supervision_fixture()
        data['test_logits'] = torch.cat([data['soft_labels'].clamp_min(1e-30).log(),
                                         torch.full((2, 2, 1), 1e4)], -1)
        metrics, per = evaluate(SuppliedLogits(), data)
        self.assertAlmostEqual(metrics['kl'], 0., places=12)
        self.assertAlmostEqual(metrics['js'], 0., places=12)
        np.testing.assert_allclose(per['kl'], 0, atol=1e-12)

    def test_empty_scene_retains_explicit_zero_valid_count(self):
        data = supervision_fixture()
        data['label_valid'][1] = False
        _, per = evaluate(SuppliedLogits(), data)
        # Consumers must filter valid_rows>0. NaN prevents an empty scene from
        # being mistaken for zero KL or a valid statistical replicate.
        np.testing.assert_array_equal(per['valid_rows'], [1, 0])
        self.assertEqual(int((per['valid_rows'] > 0).sum()), 1)
        for name in ['kl', 'js', 'top1_target_mass', 'dominant_region_accuracy']:
            self.assertTrue(np.isfinite(per[name][0]), name)
            self.assertTrue(np.isnan(per[name][1]), name)

    def test_evaluator_rejects_globally_empty_supervision(self):
        data = supervision_fixture()
        data['label_valid'][:] = False
        with self.assertRaisesRegex(ValueError, 'no valid query rows'):
            evaluate(SuppliedLogits(), data)

    def test_real_loader_conditions_on_geometry_without_label_input(self):
        count = 3
        arrays = {'soft_labels': np.array([[.4, .04, .56], [.5, .06, .44], [.2, 0., .8]], np.float32),
                  'label_valid': np.ones(count, bool),
                  'q_valid': np.array([True, True, False]),
                  'k_valid': np.array([True, False, True]),
                  'q_features': np.zeros((count, 108), np.float32),
                  'k_features': np.ones((count, 108), np.float32),
                  'q_mean': np.zeros((count, 3), np.float32),
                  'k_mean': np.ones((count, 3), np.float32),
                  'q_cov': np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))*.01,
                  'k_cov': np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))*.02,
                  'q_pixel_mean': np.arange(count*2, dtype=np.float32).reshape(count, 2),
                  'k_projected_mean': np.arange(count*2, dtype=np.float32).reshape(count, 2),
                  'q_pixel_cov': np.tile(np.eye(2, dtype=np.float32), (count, 1, 1))*48,
                  'k_projected_cov': np.tile(np.eye(2, dtype=np.float32), (count, 1, 1))*24}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'fixture'
            folder.mkdir()
            np.savez(folder/'pair.npz', **arrays)
            (folder/'manifest.json').write_text(json.dumps({'pairs': [
                {'split': 'test', 'file': 'pair.npz', 'q_id': 11, 'k_id': 13, 'npz_sha256': 'fixture'}]}))
            data, metadata = real_dataset(temporary, 'fixture', 'test')
        # Row 0 keeps 96% mass; row 1 has only 94%; row 2 lacks query geometry.
        torch.testing.assert_close(data['label_valid'], torch.tensor([[True, False, False]]))
        torch.testing.assert_close(data['soft_labels'][0, 0], torch.tensor([.4/.96, 0., .56/.96]))
        torch.testing.assert_close(data['soft_labels'].sum(-1), torch.ones(1, count))
        self.assertEqual(metadata['valid_query_rows'], 1)
        torch.testing.assert_close(data['cq'], torch.tensor(arrays['q_cov'][None]))
        torch.testing.assert_close(data['ck'], torch.tensor(arrays['k_cov'][None]))
        torch.testing.assert_close(data['pixq'], torch.tensor(arrays['q_pixel_mean'][None]/24))
        torch.testing.assert_close(data['projected_qcov'], torch.tensor(arrays['q_pixel_cov'][None]/576))
        torch.testing.assert_close(data['projected_kcov'],
                                   torch.tensor(arrays['k_projected_cov'][None]/576)+torch.eye(2)*1e-6)


class RegionalPEOperatorTests(unittest.TestCase):
    def test_density_shape_gradient_survives_spectral_underflow(self):
        omega, proposal = spectral_rule(256)
        for scale in [1., 10., 100.]:
            covariance = (torch.eye(2)[None]*scale).requires_grad_()
            mean = torch.tensor([[.2, .7]], requires_grad=True)
            feature = density_features(mean, covariance, omega, proposal)
            objective = (feature*torch.linspace(-1, 1, feature.numel()).reshape_as(feature)).sum()
            objective.backward()
            self.assertTrue(torch.isfinite(feature).all())
            self.assertTrue(torch.isfinite(covariance.grad).all(), f'covariance scale {scale}')
            self.assertTrue(torch.isfinite(mean.grad).all())

    def test_model_does_not_read_supervision_or_validity_fields(self):
        data = positional_fixture()
        changed = dict(data)
        changed.update({'soft_labels': torch.randn_like(data['soft_labels'])*1e6,
                        'label_valid': ~data['label_valid'],
                        'key_valid': ~data['key_valid'],
                        'retained_label_mass': torch.randn_like(data['retained_label_mass']),
                        'track_counts': torch.randint(0, 100, (1, 4, 4)),
                        'visibility': torch.zeros(1, 4)})
        for method in REGION_METHODS:
            model = GaussianPEMatcher(method, count=16)
            torch.testing.assert_close(model(data), model(changed), rtol=0, atol=0, msg=method)

    def test_isometric_self_scores_preserve_content_norm(self):
        data = positional_fixture()
        for method in ['isometric_fixed', 'isometric_shape']:
            current = dict(data)
            if method.endswith('fixed'):
                current['projected_qcov'] = torch.eye(2).expand(1, 4, 2, 2)/12
            for resolution in [0., .3]:
                model = GaussianPEMatcher(method, count=16, resolution=resolution)
                score = model(current)[..., :-1]/model.logtemp.exp().clamp(max=60)
                # Same RGB and Gaussian on each diagonal: the Gram diagonal
                # directly measures the encoded feature's squared norm.
                torch.testing.assert_close(score.diagonal(dim1=-2, dim2=-1), torch.ones(1, 4), atol=3e-6, rtol=0)

    def test_isometric_point_limit_equals_projected_rope(self):
        data = positional_fixture()
        data['projected_qcov'] = torch.zeros_like(data['projected_qcov'])
        data['projected_kcov'] = torch.zeros_like(data['projected_kcov'])
        data['k_to_q'] = data['k_to_q']+torch.tensor([.4, -.3])
        point = GaussianPEMatcher('projected_rope', count=16)
        isometric = GaussianPEMatcher('isometric_shape', count=16)
        isometric.load_state_dict(point.state_dict())
        torch.testing.assert_close(point(data), isometric(data), atol=2e-6, rtol=1e-6)

    def test_trace_matched_isotropic_control_uses_same_center(self):
        data = positional_fixture()
        model = GaussianPEMatcher('density_isotropic', count=16)
        mq, cq, mk, ck = model.positional_inputs(data)
        torch.testing.assert_close(mq, data['pixq'])
        torch.testing.assert_close(mk, data['k_to_q'])
        torch.testing.assert_close(cq, data['projected_qcov'])
        torch.testing.assert_close(ck.diagonal(dim1=-2, dim2=-1).sum(-1),
                                   data['projected_kcov'].diagonal(dim1=-2, dim2=-1).sum(-1))
        torch.testing.assert_close(ck[..., 0, 1], torch.zeros(1, 4))
        torch.testing.assert_close(ck[..., 0, 0], ck[..., 1, 1])


if __name__ == '__main__':
    unittest.main()
