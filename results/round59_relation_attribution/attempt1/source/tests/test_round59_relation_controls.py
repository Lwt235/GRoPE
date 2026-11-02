"""Bounded independent formulas and matched learned network controls."""
import math
import unittest
import numpy as np
from scipy.special import erf
import torch
from models.round58_support_pe import SupportMatcher
from models.round59_relation_controls import make_model, NEW_ROLES, REFERENCE_ROLES, pair_inputs
from experiments.round58_run import shared_digest
from experiments.round58_data import loss_function, random_rotation


def independent_inputs(role, f, c, covariance):
    n = len(c)
    result = []
    matrix_features = []
    for matrix in covariance:
        eigen, vectors = np.linalg.eigh(matrix)
        logm = (vectors*np.log(eigen/.04)[None, :])@vectors.T
        matrix_features.append(np.r_[np.diag(logm), np.sqrt(2)*logm[[1, 2, 2], [0, 0, 1]]])
    for i in range(n):
        row = []
        neighbor_distance = np.linalg.norm(c-c[i], axis=-1)
        neighbor_distance[i] = np.inf
        neighbors = np.argsort(neighbor_distance)[:3]
        for j in range(n):
            delta = c[i]-c[j]
            distance = np.linalg.norm(delta)/.2
            if role == 'distance_mlp':
                value = np.array([distance])
            elif role == 'matrix_mlp':
                value = np.r_[delta/.2, matrix_features[i], matrix_features[j]]
            else:
                value = np.r_[distance, f[i], f[j]]
                if role == 'angle_descriptor_mlp':
                    angles = []
                    for k in neighbors:
                        a, b = c[k]-c[i], c[j]-c[i]
                        angle = np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))/(np.pi/12)
                        angles.append(np.r_[value, angle])
                    value = np.stack(angles)
            row.append(value)
        result.append(np.stack(row))
    return np.stack(result)


class RelationTests(unittest.TestCase):
    def inputs(self):
        rng = np.random.default_rng(5901)
        f = torch.from_numpy(rng.normal(size=(1, 2, 7, 11)))
        c = torch.from_numpy(rng.normal(size=(1, 2, 7, 3)))
        l = torch.from_numpy(rng.normal(size=(1, 2, 7, 3, 3)))
        s = l@l.transpose(-1, -2)+torch.eye(3)*.1
        return f, c, s

    def test_independent_inputs_and_mlp(self):
        f, c, s = self.inputs()
        for role in NEW_ROLES:
            expected = np.stack([independent_inputs(role, f[0, v].numpy(), c[0, v].numpy(), s[0, v].numpy()) for v in range(2)])[None]
            actual = pair_inputs(role, f, c, s)
            np.testing.assert_allclose(actual.numpy(), expected, atol=3e-12, rtol=3e-12)
            torch.manual_seed(11)
            model = make_model(role).double()
            state = model.position.state_dict()
            hidden = expected@state['mlp.0.weight'].numpy().T+state['mlp.0.bias'].numpy()
            hidden = .5*hidden*(1+erf(hidden/np.sqrt(2)))
            value = hidden@state['mlp.2.weight'].numpy().T
            if role == 'angle_descriptor_mlp':
                value = value.max(-2)
            np.testing.assert_allclose(model.position(f, c, s).detach().numpy(), value, atol=4e-12, rtol=4e-12)

    def test_reference_functions_and_common_initialization(self):
        data = self.inputs()
        for seed in (11, 22, 33):
            torch.manual_seed(seed)
            expected = shared_digest(SupportMatcher('invariant_mlp'))
            for role in NEW_ROLES+tuple(REFERENCE_ROLES):
                torch.manual_seed(seed)
                model = make_model(role)
                self.assertEqual(shared_digest(model), expected)
            for role in REFERENCE_ROLES:
                torch.manual_seed(seed); a = make_model(role).double().eval()
                torch.manual_seed(seed); b = SupportMatcher(role).double().eval()
                self.assertIs(type(a), type(b))
                torch.testing.assert_close(a(*data), b(*data), atol=0, rtol=0)

    def test_geometry_behavior_gradients_and_orientation_isolation(self):
        f, c, s = self.inputs()
        rng = np.random.default_rng(5902)
        r = torch.from_numpy(np.stack([random_rotation(rng) for _ in range(2)]))
        moved = c@r.transpose(-1, -2)+torch.tensor([.7, -.3, .2])
        rotated = r[:, None]@s@r[:, None].transpose(-1, -2)
        truth = torch.from_numpy(rng.uniform(size=(1, 7, 7)))
        for role in NEW_ROLES:
            torch.manual_seed(11); model = make_model(role).double()
            output = model(f, c, s)
            if role != 'matrix_mlp':
                torch.testing.assert_close(output, model(f, moved, rotated), atol=2e-10, rtol=2e-10)
                # These controls intentionally do not read tensor orientation.
                torch.testing.assert_close(output, model(f, c, rotated), atol=0, rtol=0)
            else:
                self.assertGreater(float((output-model(f, moved, rotated)).abs().max()), 1e-6)
            permuted = model(f.flip(-2), c.flip(-2), s.flip(-3)).flip(-2, -1)
            torch.testing.assert_close(output, permuted, atol=2e-10, rtol=2e-10)
            loss_function(output, truth).backward()
            for parameter in model.position.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.norm()), 0)

    def test_fitting_adapter_restores_original_scope(self):
        from experiments import round58_run as original
        from experiments.round59_run import fitting_adapter
        from experiments.round59_common import OUT, CONFIG
        names = ('OUT', 'ROLES', 'SupportMatcher', 'checked', 'load_cache')
        before = {name: getattr(original, name) for name in names}
        fit_function, optimizer_config = original.train, dict(original.CONFIG)
        try:
            with fitting_adapter():
                self.assertEqual(original.OUT, OUT)
                self.assertEqual(original.ROLES, NEW_ROLES)
                self.assertIs(original.SupportMatcher, make_model)
                self.assertIs(original.train, fit_function)
                self.assertEqual(original.CONFIG, CONFIG)
                raise RuntimeError('exercise exception restoration')
        except RuntimeError:
            pass
        for name in names:
            self.assertIs(getattr(original, name), before[name])
        self.assertEqual(original.CONFIG, optimizer_config)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
