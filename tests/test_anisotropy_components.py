import unittest
import numpy as np
import torch

from models.anisotropy_components import ROLES, ComponentMatcher, ComponentNative, component_descriptors
from models.round69_compact_relation import CompactMatcher, descriptors
from models.round73_native_relative5 import CompactNativeClassifier


class ComponentTests(unittest.TestCase):
    @staticmethod
    def geometry():
        rng = np.random.default_rng(82201)
        c = rng.normal(size=(2, 7, 3)) * .2
        a = rng.normal(size=(2, 7, 3, 3)) * .12
        return torch.from_numpy(c), torch.from_numpy(a @ a.swapaxes(-1, -2) + .002 * np.eye(3))

    def test_independent_scalar_formulas_and_spherical_limit(self):
        c, s = self.geometry()
        centers, cov = c.numpy(), s.numpy()
        expected = {role: np.empty((2, 7, 7, 5)) for role in ROLES}
        for batch in range(2):
            for i in range(7):
                for j in range(7):
                    delta = centers[batch, i] - centers[batch, j]
                    si, sj = cov[batch, i], cov[batch, j]
                    pi, pj = np.linalg.inv(si), np.linalg.inv(sj)
                    d2, ri, rj = delta @ delta, np.trace(si) / 3, np.trace(sj) / 3
                    metric = [delta @ pi @ delta, delta @ pj @ delta, 3*rj/ri, 3*ri/rj]
                    shape = [d2/ri, d2/rj, np.trace(pi @ sj), np.trace(pj @ si)]
                    for role, terms in zip(ROLES, (metric, shape)):
                        expected[role][batch, i, j] = np.r_[np.sqrt(d2)/.2, np.log1p(terms)]
        radii = s.diagonal(dim1=-2, dim2=-1).mean(-1)
        sphere = radii[..., None, None] * torch.eye(3, dtype=s.dtype)
        for role in ROLES:
            np.testing.assert_allclose(component_descriptors(role, c, s), expected[role], atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(component_descriptors(role, c, sphere), descriptors('relative5', c, sphere))

    def test_rigid_invariance_permutation_and_targeted_sensitivity(self):
        c, s = self.geometry()
        q = torch.linalg.qr(torch.tensor([[1.,2.,3.],[2.,-1.,4.],[3.,1.,-1.]], dtype=torch.float64)).Q
        order = torch.tensor([3, 0, 6, 1, 5, 2, 4])
        for role in ROLES:
            original = component_descriptors(role, c, s)
            torch.testing.assert_close(original, component_descriptors(role, c@q.T+.7, q@s@q.T), rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(original[:, order][:, :, order], component_descriptors(role, c[:, order], s[:, order]))
        # Rotate all ellipsoids relative to fixed displacements. Covariance-pair
        # compatibility and trace are unchanged, but directional metrics change.
        shape_before = component_descriptors(ROLES[1], c, s)
        shape_after = component_descriptors(ROLES[1], c, q@s@q.T)
        torch.testing.assert_close(shape_before, shape_after, rtol=1e-10, atol=1e-10)
        metric_before = component_descriptors(ROLES[0], c, s)
        metric_after = component_descriptors(ROLES[0], c, q@s@q.T)
        self.assertGreater(float((metric_before-metric_after).abs().max()), .01)

    def test_complete_initial_state_rng_and_parameter_count(self):
        for original, new in ((CompactMatcher, ComponentMatcher), (CompactNativeClassifier, ComponentNative)):
            for role in ROLES:
                for seed in (11, 22, 33):
                    torch.manual_seed(seed); a = original('relative5'); state = torch.get_rng_state()
                    torch.manual_seed(seed); b = new(role)
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))
                    self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
                    for key, value in a.state_dict().items():
                        self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
                    self.assertEqual(sum(p.numel() for p in b.position.parameters()), 9792)

    def test_common_content_unchanged_and_finite_gradients(self):
        c, s = self.geometry(); c = c.float(); appearance = torch.zeros(2, 7, 4)
        frozen = s.clone()
        for role in ROLES:
            torch.manual_seed(11); reference = CompactNativeClassifier('relative5')
            torch.manual_seed(11); model = ComponentNative(role)
            seen = []
            hooks = [m.stem.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
                     for m in (reference, model)]
            reference(c, s, appearance); output = model(c, s, appearance)
            torch.testing.assert_close(seen[0], seen[1], atol=0, rtol=0)
            torch.testing.assert_close(frozen, s, atol=0, rtol=0)
            output.square().mean().backward()
            for param in model.position.parameters():
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(float(param.grad.norm()), 0)
            for hook in hooks:
                hook.remove()


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
