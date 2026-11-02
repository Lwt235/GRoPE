"""Integration checks for the new task, with no real point-cloud data."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from models.round48_point_transformer import make_model, ROLES, GAUSSIAN, normalize_points
from experiments.round48_train import shared_digest, score_sample
from experiments.round48_audit import check_sample, object_metrics
from experiments.round48_data import split_indices, canonical_indices, evaluation_points


class NativePointPE(unittest.TestCase):
    def test_paired_initialization_and_functions(self):
        torch.manual_seed(481)
        x = torch.randn(2, 13, 3)
        shared, predictions = [], {}
        for role in ROLES:
            torch.manual_seed(17)
            model = make_model(role).eval()
            shared.append(shared_digest(model))
            with torch.no_grad():
                predictions[role] = model(x)
        self.assertEqual(len(set(shared)), 1)
        for role in GAUSSIAN:
            torch.testing.assert_close(predictions[role], predictions['fixed'], atol=0, rtol=0)
        for role in ('fourier', 'mlp'):
            torch.testing.assert_close(predictions[role], predictions['rope'], atol=0, rtol=0)

    def test_nonzero_positions_direct_formula(self):
        torch.manual_seed(482)
        x = torch.randn(1, 19, 3)
        with tempfile.TemporaryDirectory() as folder:
            for role in ROLES:
                model = make_model(role).eval()
                with torch.no_grad():
                    for p in model.position_field.parameters():
                        p.add_(torch.randn_like(p)*.07)
                path = Path(folder)/(role+'.npz')
                score_sample(model, x, path)
                numeric = check_sample(role, model.state_dict(), path)
                self.assertLess(numeric['p_max_error'], 5e-5, role)
                self.assertEqual(numeric['winner_changes'], 0, role)

    def test_geometry_and_permutation(self):
        torch.manual_seed(483)
        x = torch.randn(2, 21, 3)
        perm = torch.randperm(21)
        for role in ROLES:
            model = make_model(role).eval()
            with torch.no_grad():
                for p in model.position_field.parameters():
                    p.add_(torch.randn_like(p)*.03)
                base = model(x)
                moved = model(x*2.3+torch.tensor([3., -7., 2.]))
                shuffled = model(x[:, perm])
            torch.testing.assert_close(base, moved, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(base, shuffled, atol=2e-5, rtol=2e-5)
        centered = normalize_points(x)
        torch.testing.assert_close(centered.mean(1), torch.zeros(2, 3), atol=1e-6, rtol=0)
        torch.testing.assert_close(centered.square().sum(-1).mean(-1), torch.ones(2), atol=1e-6, rtol=0)

    def test_content_and_positional_gradient_paths(self):
        torch.manual_seed(484)
        model = make_model('conditional')
        x = torch.randn(3, 17, 3)
        torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 9, 31])).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(float(model.position_field.predictor.weight.grad.norm()), 0)
        for layer in model.layers:
            self.assertGreater(float(layer.qkv.weight.grad[192:].norm()), 0)

    def test_train_only_split_and_canonical_sampling(self):
        labels = np.repeat(np.arange(40), 23)
        ids = [f'official_train:synthetic:{i}' for i in range(len(labels))]
        parts = split_indices(labels, ids)
        self.assertEqual(len(np.unique(np.concatenate(list(parts.values())))), len(labels))
        self.assertEqual(sum(map(len, parts.values())), len(labels))
        for name, index in parts.items():
            self.assertEqual(len(set(labels[index])), 40)
            np.testing.assert_array_equal(index, split_indices(labels, ids)[name])
        a = canonical_indices(ids[0])
        np.testing.assert_array_equal(a, canonical_indices(ids[0]))
        self.assertEqual(len(set(a)), 2048)
        bundle = dict(ids=np.asarray(ids[:2]), canonical=np.stack([a, a]),
                      points=np.arange(2*2048*3, dtype=np.float32).reshape(2, 2048, 3))
        np.testing.assert_array_equal(evaluation_points(bundle, 'base')[:, :256],
                                      evaluation_points(bundle, 'points256'))
        np.testing.assert_array_equal(evaluation_points(bundle, 'base')[:, ::-1],
                                      evaluation_points(bundle, 'permuted'))

    def test_independent_metric_replay(self):
        labels = np.arange(40)
        logits = np.zeros((40, 40))
        logits[np.arange(40), labels] = 1.
        logits[0, 1] = 2.
        result = object_metrics(logits, labels)
        self.assertEqual(result['correct'], 39)
        self.assertAlmostEqual(result['oa'], .975)
        self.assertAlmostEqual(result['mca'], .975)
        ce = torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels))
        self.assertAlmostEqual(result['ce'], float(ce), places=12)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
