import itertools
import unittest
import numpy as np
import torch

from models.mean_precision_relation import (
    MeanPrecisionMatcher, MeanPrecisionNative, mean_precision_descriptors)
from models.round69_compact_relation import CompactMatcher, descriptors
from models.round73_native_relative5 import CompactNativeClassifier


class MeanPrecisionTests(unittest.TestCase):
    @staticmethod
    def geometry():
        rng = np.random.default_rng(83001)
        centers = rng.normal(size=(2, 7, 3)) * .2
        matrix = rng.normal(size=(2, 7, 3, 3)) * .12
        covariance = matrix @ matrix.swapaxes(-1, -2) + .002 * np.eye(3)
        return torch.from_numpy(centers), torch.from_numpy(covariance)

    def test_independent_scalar_formula_sphere_and_self(self):
        c, s = self.geometry()
        expected = np.empty((2, 7, 7, 5))
        for batch in range(2):
            for i in range(7):
                for j in range(7):
                    a, b = s[batch, i].numpy(), s[batch, j].numpy()
                    # Eigenvalues give an independent path to tr(S^-1).
                    ha, hb = (np.mean(1 / np.linalg.eigvalsh(x)) for x in (a, b))
                    ra, rb = np.trace(a) / 3, np.trace(b) / 3
                    delta = (c[batch, i] - c[batch, j]).numpy()
                    d2 = delta @ delta
                    terms = [ha*d2, hb*d2, 3*ha*rb if i != j else 3., 3*hb*ra if i != j else 3.]
                    expected[batch, i, j] = np.r_[np.sqrt(d2)/.2, np.log1p(terms)]
        value = mean_precision_descriptors(c, s)
        np.testing.assert_allclose(value, expected, atol=1e-11, rtol=1e-11)
        full = descriptors('relative5', c, s)
        torch.testing.assert_close(value.diagonal(dim1=-3, dim2=-2),
                                   full.diagonal(dim1=-3, dim2=-2), atol=1e-11, rtol=1e-11)
        radius = s.diagonal(dim1=-2, dim2=-1).mean(-1)
        sphere = radius[..., None, None] * torch.eye(3, dtype=s.dtype)
        torch.testing.assert_close(mean_precision_descriptors(c, sphere), descriptors('relative5', c, sphere))

    def test_exact_finite_rotation_average_before_nonlinearity(self):
        rotations = []
        for permutation in itertools.permutations(range(3)):
            for signs in itertools.product((-1., 1.), repeat=3):
                matrix = np.eye(3)[list(permutation)] * np.asarray(signs)[:, None]
                if np.linalg.det(matrix) > .5:
                    rotations.append(matrix)
        self.assertEqual(len(rotations), 24)
        c, s = self.geometry()
        q = mean_precision_descriptors(c, s).numpy()
        for i, j in ((0, 1), (2, 4), (5, 6)):
            a, b = s[0, i].numpy(), s[0, j].numpy()
            delta = (c[0, i] - c[0, j]).numpy()
            precision = np.linalg.inv(a)
            metric = np.mean([delta @ r @ precision @ r.T @ delta for r in rotations])
            shape = np.mean([np.trace(precision @ r @ b @ r.T) for r in rotations])
            np.testing.assert_allclose(np.expm1(q[0, i, j, [1, 3]]), [metric, shape], rtol=1e-11, atol=1e-11)

    def test_rigid_permutation_and_independent_rotation_behavior(self):
        c, s = self.geometry()
        q = torch.linalg.qr(torch.tensor([[1., 2., 3.], [2., -1., 4.], [3., 1., -1.]], dtype=s.dtype)).Q
        value = mean_precision_descriptors(c, s)
        torch.testing.assert_close(value, mean_precision_descriptors(c @ q.T + .7, q @ s @ q.T), rtol=1e-10, atol=1e-10)
        order = torch.tensor([3, 0, 6, 1, 5, 2, 4])
        torch.testing.assert_close(value[:, order][:, :, order], mean_precision_descriptors(c[:, order], s[:, order]))
        rng = np.random.default_rng(83002)
        independent = torch.from_numpy(np.linalg.qr(rng.normal(size=(2, 7, 3, 3)))[0])
        rotated = independent @ s @ independent.transpose(-1, -2)
        torch.testing.assert_close(value, mean_precision_descriptors(c, rotated), rtol=1e-10, atol=1e-10)
        self.assertGreater(float((descriptors('relative5', c, s) - descriptors('relative5', c, rotated)).abs().max()), .01)
        sphere = s.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(3, dtype=s.dtype)
        self.assertGreater(float((value - mean_precision_descriptors(c, sphere)).abs().max()), .01)

    def test_full_initial_state_rng_and_parameter_count(self):
        for original, new in ((CompactMatcher, MeanPrecisionMatcher), (CompactNativeClassifier, MeanPrecisionNative)):
            for seed in (11, 22, 33):
                torch.manual_seed(seed)
                reference = original('relative5')
                rng = torch.get_rng_state()
                torch.manual_seed(seed)
                model = new('mean_precision5')
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(reference.state_dict().keys(), model.state_dict().keys())
                for key, value in reference.state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                self.assertEqual(sum(p.numel() for p in model.position.parameters()), 9792)

    def test_common_content_and_finite_position_gradients(self):
        c, s = self.geometry()
        c = c.float()
        frozen = s.clone()
        native_args = (c, s, torch.zeros(2, 7, 4))
        region_args = (torch.zeros(1, 2, 7, 11), c[None], s[None], torch.arange(7)[None, None].expand(1, 2, 7))
        for original, new, args in ((CompactNativeClassifier, MeanPrecisionNative, native_args),
                                    (CompactMatcher, MeanPrecisionMatcher, region_args)):
            torch.manual_seed(11)
            reference = original('relative5')
            torch.manual_seed(11)
            model = new('mean_precision5')
            seen = []
            hooks = [m.stem.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach().clone()))
                     for m in (reference, model)]
            reference(*args)
            result = model(*args)
            torch.testing.assert_close(seen[0], seen[1], atol=0, rtol=0)
            torch.testing.assert_close(frozen, s, atol=0, rtol=0)
            result.square().mean().backward()
            for parameter in model.position.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.norm()), 0.)
            for hook in hooks:
                hook.remove()


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
