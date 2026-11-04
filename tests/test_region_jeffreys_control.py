import unittest
import numpy as np
import torch
from torch.nn import functional as F
from models.region_jeffreys_control import JeffreysMatcher, jeffreys_inputs
from models.round69_compact_relation import CompactMatcher, descriptors


class JeffreysControlTests(unittest.TestCase):
    def test_same_full_initialization_and_registered_capacity(self):
        for seed in (11, 22, 33):
            torch.manual_seed(seed)
            original = CompactMatcher('relative5')
            torch.manual_seed(seed)
            control = JeffreysMatcher()
            self.assertEqual(original.state_dict().keys(), control.state_dict().keys())
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, control.state_dict()[name]), name)
            self.assertEqual(sum(p.numel() for p in control.position.parameters()), 9792)

    def test_independent_closed_KL_and_rigid_motion(self):
        rng = np.random.default_rng(78001)
        c = rng.normal(size=(7, 3))
        a = rng.normal(size=(7, 3, 3))
        s = a @ a.transpose(0, 2, 1) + .04 * np.eye(3)
        actual = jeffreys_inputs(torch.tensor(c), torch.tensor(s)).numpy()
        for i in range(7):
            for j in range(7):
                delta = c[i] - c[j]
                def kl(p, q):
                    return .5 * (np.trace(np.linalg.solve(s[q], s[p]))
                        + delta @ np.linalg.solve(s[q], delta) - 3
                        + np.linalg.slogdet(s[q])[1] - np.linalg.slogdet(s[p])[1])
                expected = max(0., kl(i, j) + kl(j, i))
                self.assertAlmostEqual(actual[i, j, 1], np.log1p(expected), places=11)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        moved = jeffreys_inputs(torch.tensor(c @ rotation.T + .7),
                                torch.tensor(rotation @ s @ rotation.T))
        torch.testing.assert_close(moved, torch.tensor(actual), atol=2e-11, rtol=2e-11)

    def test_two_scalar_information_and_nonidentical_relations(self):
        c = torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
        a = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1) * (8 / 37)
        b = torch.stack((2 * torch.eye(3, dtype=torch.float64), .5 * torch.eye(3, dtype=torch.float64)))
        torch.testing.assert_close(jeffreys_inputs(c, a)[0, 1], jeffreys_inputs(c, b)[0, 1])
        self.assertFalse(torch.allclose(descriptors('relative5', c, a)[0, 1],
                                        descriptors('relative5', c, b)[0, 1]))
        model = JeffreysMatcher().double()
        x = jeffreys_inputs(c, b)
        first = model.position.mlp[0]
        weight = torch.stack((first.weight[:, 0], first.weight[:, 1:].sum(-1)), -1)
        hidden = F.gelu(F.linear(x[..., :2], weight, first.bias))
        collapsed = model.position.mlp[2](hidden)
        torch.testing.assert_close(collapsed, model.position(c, b), atol=1e-14, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
