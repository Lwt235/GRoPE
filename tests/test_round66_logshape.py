import unittest
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from models.round65_actual_gs import NativeGSClassifier
from models.round66_logshape import ROLES, LogShapeClassifier, shape_distance
from experiments.round65_run import synthetic_inputs, shared_digest
from experiments.round65_common import SEEDS, VARIANTS
from experiments.round66_logshape import ALL_ROLES, independent_shape, contrasts


class LogShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_known_relative_orientation_and_spectrum_control(self):
        # Same eigenvalues, ninety-degree x/y axis exchange: squared log distance8.
        diagonal = .04*torch.exp(2*torch.tensor([1., -1., .25], dtype=torch.float64))
        a = torch.diag(diagonal)
        b = torch.diag(diagonal[[1, 0, 2]])
        covariance = torch.stack((a, b))[None]
        actual = shape_distance(covariance)[0]
        torch.testing.assert_close(actual, torch.tensor([[0., np.log(9.)], [np.log(9.), 0.]],
                                                       dtype=torch.float64), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(shape_distance(covariance, True), torch.zeros((1, 2, 2), dtype=torch.float64),
                                   atol=1e-12, rtol=0)

    def test_independent_matrix_log_and_invariance(self):
        c, s, _ = synthetic_inputs(2, 9, 'cpu')
        r = torch.tensor(Rotation.from_rotvec([.4, -.7, .2]).as_matrix())
        for role in ROLES:
            model = LogShapeClassifier(role)
            actual = model.position.descriptors(c, s)
            for batch in range(2):
                np.testing.assert_allclose(actual[batch, :, :, 1].numpy(),
                    independent_shape(s[batch].numpy(), role == 'spectrum_distance'), atol=2e-6, rtol=2e-6)
            cc = (c.double()@r.T+torch.tensor([.2, -.3, .7])).float()
            torch.testing.assert_close(actual, model.position.descriptors(cc, r@s@r.T), atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(actual[..., 1], model.position.descriptors(c, 9*s)[..., 1],
                                       atol=2e-6, rtol=2e-6)

    def test_distance_function_initialization_rng_and_capacity(self):
        c, s, a = synthetic_inputs(2, 9, 'cpu')
        for seed in SEEDS:
            torch.manual_seed(seed)
            original = NativeGSClassifier('distance_mlp').eval()
            original_rng = torch.get_rng_state()
            for role in ROLES:
                torch.manual_seed(seed)
                model = LogShapeClassifier(role).eval()
                torch.testing.assert_close(torch.get_rng_state(), original_rng, atol=0, rtol=0)
                self.assertEqual(shared_digest(model), shared_digest(original))
                for name, tensor in original.position.state_dict().items():
                    new = model.position.state_dict()[name]
                    torch.testing.assert_close(new[:, :1] if name == 'mlp.0.weight' else new, tensor, atol=0, rtol=0)
                self.assertEqual(int(torch.count_nonzero(model.position.mlp[0].weight[:, 1])), 0)
                self.assertEqual(sum(p.numel() for p in model.position.parameters()), 9504)
                with torch.no_grad():
                    torch.testing.assert_close(model(c, s, a), original(c, s, a), atol=2e-6, rtol=2e-5)

    def test_new_shape_input_learns_and_permutation_preserves_prediction(self):
        c, s, a = synthetic_inputs(2, 9, 'cpu')
        permutation = torch.tensor([8, 0, 4, 1, 7, 2, 6, 3, 5])
        for role in ROLES:
            torch.manual_seed(11)
            model = LogShapeClassifier(role)
            loss = F.cross_entropy(model(c, s, a), torch.tensor([2, 59]))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(model.position.mlp[0].weight.grad[:, 1].norm()), 0.)
            with torch.no_grad():
                model.position.mlp[0].weight[:, 1].copy_(torch.linspace(-.1, .1, 96))
                model.eval()
                x = model(c, s, a)
                y = model(c[:, permutation], s[:, permutation], a[:, permutation])
                torch.testing.assert_close(x, y, atol=3e-6, rtol=2e-5)
                np.testing.assert_array_equal(x.argmax(1).numpy(), y.argmax(1).numpy())

    def test_contrasts_preserve_seed_signs_and_reject_missing_or_nan(self):
        lookup = {(role, seed, variant): dict(metrics=dict(class_accuracy=[.5]*60))
                  for role in ALL_ROLES for seed in SEEDS for variant in VARIANTS}
        for seed, value in zip(SEEDS, [.53, .49, .51]):
            for variant in VARIANTS:
                lookup[ROLES[0], seed, variant]['metrics']['class_accuracy'] = [value]*60
        result = contrasts(lookup)
        self.assertEqual(len(result), 48)
        row = next(r for r in result if r['candidate'] == ROLES[0]
                   and r['control'] == 'distance_mlp' and r['variant'] == 'base')
        np.testing.assert_allclose(row['paired_seed_gains_pp'], [3., -1., 1.], atol=1e-12)
        self.assertAlmostEqual(row['mean_gain_pp'], 1.)
        self.assertFalse(row['development_gain_criterion'])
        key = ROLES[0], SEEDS[0], 'base'
        entry = lookup.pop(key)
        with self.assertRaisesRegex(ValueError, 'incomplete126'):
            contrasts(lookup)
        lookup[key] = entry
        entry['metrics']['class_accuracy'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'invalid paired'):
            contrasts(lookup)


if __name__ == '__main__':
    unittest.main()
