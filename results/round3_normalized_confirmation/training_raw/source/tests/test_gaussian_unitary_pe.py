"""Mathematical contracts for a fixed-width Gaussian orthogonal completion."""
import unittest

import torch
import torch.nn.functional as F

from models.gaussian_unitary_pe import METHODS, GaussianUnitaryMatcher, encode, frequencies


def fixture():
    generator = torch.Generator().manual_seed(824)
    data = {}
    for side, count in [('q', 4), ('k', 7)]:
        data[side] = torch.randn(2, count, 108, generator=generator, dtype=torch.float64)
        mean_key = 'pixq' if side == 'q' else 'k_to_q'
        covariance_key = 'projected_qcov' if side == 'q' else 'projected_kcov'
        data[mean_key] = torch.randn(2, count, 2, generator=generator, dtype=torch.float64)
        a = torch.randn(2, count, 2, 2, generator=generator, dtype=torch.float64)*.2
        data[covariance_key] = a @ a.transpose(-1, -2)+torch.eye(2)*.03
    return data


class GaussianUnitaryPETest(unittest.TestCase):
    def test_encode_preserves_width_norm_and_all_same_gaussian_inner_products(self):
        generator = torch.Generator().manual_seed(15)
        q = torch.randn(2, 4, 48, generator=generator, dtype=torch.float64)
        k = torch.randn(2, 4, 48, generator=generator, dtype=torch.float64)
        data = fixture()
        frequency = frequencies().double()
        eq = encode(q, data['pixq'], data['projected_qcov'], frequency)
        ek = encode(k, data['pixq'], data['projected_qcov'], frequency)
        self.assertEqual(eq.shape, q.shape)
        torch.testing.assert_close(eq.norm(dim=-1), q.norm(dim=-1), atol=2e-14, rtol=2e-14)
        torch.testing.assert_close((eq*ek).sum(-1), (q*k).sum(-1), atol=2e-14, rtol=2e-14)
        # Test the complete 4x4 operator, including the previously unused v
        # channels, rather than only a 2->4 embedding with v=0.
        basis = torch.eye(4, dtype=torch.float64)
        matrix = encode(basis, torch.tensor([.2, -.3], dtype=torch.float64),
                        torch.tensor([[.3, .04], [.04, .1]], dtype=torch.float64),
                        torch.tensor([[1., .7]], dtype=torch.float64))
        torch.testing.assert_close(matrix @ matrix.T, torch.eye(4, dtype=torch.float64))

    def test_common_translation_preserves_every_method_score(self):
        data = fixture()
        shift = torch.tensor([2.3, -4.7], dtype=torch.float64)
        moved = dict(data, pixq=data['pixq']+shift, k_to_q=data['k_to_q']+shift)
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(14)
                model = GaussianUnitaryMatcher(method).double()
                torch.testing.assert_close(model(data), model(moved), atol=2e-13, rtol=2e-13)

    def test_covariance_gauge_and_positive_isotropic_gradients(self):
        angle = torch.tensor(.34, dtype=torch.float64)
        rotation = torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()]).reshape(2, 2)
        scale = torch.diag(torch.tensor([.03, .2], dtype=torch.float64))
        sign = torch.diag(torch.tensor([-1., 1.], dtype=torch.float64))
        c1 = rotation @ scale @ rotation.T
        c2 = (rotation @ sign) @ scale @ (rotation @ sign).T
        content = torch.linspace(-1, 1, 48, dtype=torch.float64)
        mean = torch.tensor([.2, -.3], dtype=torch.float64)
        frequency = frequencies().double()
        torch.testing.assert_close(encode(content, mean, c1, frequency),
                                   encode(content, mean, c2, frequency), atol=0, rtol=0)
        cov = (torch.eye(2, dtype=torch.float64)*.2).requires_grad_()
        location = mean.clone().requires_grad_()
        output = encode(content, location, cov, frequency)
        (output*torch.linspace(.4, 2., 48, dtype=torch.float64)).sum().backward()
        self.assertTrue(torch.isfinite(cov.grad).all())
        self.assertTrue(torch.isfinite(location.grad).all())
        self.assertGreater(float(cov.grad.norm()), 0)

    def test_zero_variance_is_half_rope_and_other_half_identity(self):
        data = fixture()
        frequency = frequencies().double()
        content = torch.arange(48, dtype=torch.float64).expand(2, 4, 48)/48
        covariance = torch.zeros(2, 4, 2, 2, dtype=torch.float64)
        output = encode(content, data['pixq'], covariance, frequency).reshape(2, 4, 12, 4)
        original = content.reshape(2, 4, 12, 4)
        phase = data['pixq'] @ frequency.T
        x, y = original[..., 0], original[..., 1]
        expected = torch.stack([x*phase.cos()-y*phase.sin(), x*phase.sin()+y*phase.cos()], -1)
        torch.testing.assert_close(output[..., :2], expected)
        torch.testing.assert_close(output[..., 2:], original[..., 2:])
        # A zero-covariance forward limit is supported; finite covariance
        # gradients at this square-root endpoint are deliberately not claimed.
        tiny = encode(content, data['pixq'], torch.eye(2, dtype=torch.float64)*1e-16, frequency)
        torch.testing.assert_close(tiny, output.flatten(-2), atol=1e-7, rtol=1e-7)

    def test_wide_support_swaps_position_channel_instead_of_removing_it(self):
        frequency = frequencies().double()
        mean = torch.tensor([.8, -.2], dtype=torch.float64)
        content = torch.linspace(-1, 1, 48, dtype=torch.float64)
        output = encode(content, mean, torch.eye(2, dtype=torch.float64)*1e9, frequency).reshape(12, 4)
        block = content.reshape(12, 4)
        phase = mean @ frequency.T
        x, y = -block[:, 2], -block[:, 3]
        expected = torch.stack([x*phase.cos()-y*phase.sin(), x*phase.sin()+y*phase.cos()], -1)
        torch.testing.assert_close(output[:, :2], expected)
        torch.testing.assert_close(output[:, 2:], block[:, :2])

    def test_support_controls_share_query_blur_and_only_change_key_covariance(self):
        data = fixture()
        blur = torch.eye(2, dtype=torch.float64)*.25**2
        for method in METHODS:
            model = GaussianUnitaryMatcher(method).double()
            mq, cq, mk, ck = model.positional_inputs(data)
            torch.testing.assert_close(mq, data['pixq'])
            torch.testing.assert_close(mk, data['k_to_q'])
            if method == 'unitary_point':
                torch.testing.assert_close(cq, torch.zeros_like(cq))
                torch.testing.assert_close(ck, torch.zeros_like(ck))
                continue
            torch.testing.assert_close(cq, data['projected_qcov']+blur)
            original_key = data['projected_kcov']
            if method == 'unitary_shape':
                torch.testing.assert_close(ck, original_key+blur)
            elif method == 'unitary_fixed':
                torch.testing.assert_close(ck, (torch.eye(2, dtype=torch.float64)/12+blur).expand_as(ck))
            elif method == 'unitary_trace':
                torch.testing.assert_close((ck-blur).diagonal(dim1=-2, dim2=-1).sum(-1),
                                           original_key.diagonal(dim1=-2, dim2=-1).sum(-1))
            elif method == 'unitary_area':
                torch.testing.assert_close(torch.linalg.det(ck-blur), torch.linalg.det(original_key))
            elif method == 'unitary_shuffled':
                torch.testing.assert_close(ck, original_key.roll(13, dims=-3)+blur)

    def test_models_share_frequencies_capacity_and_ignore_labels(self):
        data = fixture()
        contaminated = dict(data, labels=torch.zeros(2, 4, dtype=torch.long),
                            soft_labels=torch.randn(2, 4, 7), label_valid=torch.zeros(2, 4),
                            track_counts=torch.randint(0, 100, (2, 4, 7)),
                            visibility=torch.randn(2, 4, 7), key_valid=torch.zeros(2, 7))
        parameter_counts = []
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(18)
                model = GaussianUnitaryMatcher(method).double()
                parameter_counts.append(sum(p.numel() for p in model.parameters()))
                torch.testing.assert_close(model.freq2, frequencies().double())
                output = model(data)
                self.assertEqual(output.shape, (2, 4, 8))
                torch.testing.assert_close(output, model(contaminated), atol=0, rtol=0)
                loss = -output.log_softmax(-1)[..., 0].mean()
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                q = F.normalize(model.encoder(data['q']), dim=-1)
                k = F.normalize(model.encoder(data['k']), dim=-1)
                mq, cq, mk, ck = model.positional_inputs(data)
                expected = encode(q, mq, cq, model.freq2) @ encode(k, mk, ck, model.freq2).transpose(-1, -2)
                self.assertLessEqual(float(expected.abs().max()), 1.+1e-12)
        self.assertEqual(len(set(parameter_counts)), 1)

    def test_shape_is_observable_beyond_diagonal_variances(self):
        content = torch.linspace(-1, 1, 48, dtype=torch.float64)
        mean = torch.zeros(2, dtype=torch.float64)
        c1 = torch.tensor([[.3, .1], [.1, .3]], dtype=torch.float64)
        c2 = torch.tensor([[.3, -.1], [-.1, .3]], dtype=torch.float64)
        self.assertGreater(float((encode(content, mean, c1, frequencies().double())
                                  -encode(content, mean, c2, frequencies().double())).abs().max()), 1e-4)


if __name__ == '__main__':
    unittest.main()
