"""Independent calculus checks for resolvent-normalized spectral filtering.

No matcher/model implementation is imported. These tests verify the proposed
mathematical scale-sensitivity bound and an explicit Gaussian counterexample,
not empirical accuracy, calibration, or arbitrary geometry-error robustness.
"""
import math
import unittest

import torch


DTYPE = torch.float64


def theoretical_map(content, logscale, variance, floor_variance, family):
    total = logscale.exp()*variance+floor_variance
    amplitude = (1+total/2).reciprocal() if family == 'resolvent' else (-total/2).exp()
    filtered = content.reshape(-1, 2)*amplitude[:, None]
    # This calculus fixture uses a zero mean: any fixed mean would only left
    # multiply by shared orthogonal rotations and preserve derivative norms.
    return content.norm()*filtered.flatten()/filtered.norm()


class ResolventSensitivityTest(unittest.TestCase):
    def test_covariance_floor_logscale_derivative_equals_weighted_variance(self):
        generator = torch.Generator().manual_seed(4611)
        frequency = torch.randn(24, 2, generator=generator, dtype=DTYPE)
        frequency /= frequency.norm(dim=-1, keepdim=True)
        frequency *= torch.logspace(-1, math.log10(4), 24, dtype=DTYPE)[:, None]
        factor = torch.tensor([[.8, -.2], [.3, .5]], dtype=DTYPE)
        covariance = factor @ factor.T
        variance = torch.einsum('fi,ij,fj->f', frequency, covariance, frequency)
        floor_variance = .25**2*frequency.square().sum(-1)
        for sparse in [False, True]:
            content = torch.randn(48, generator=generator, dtype=DTYPE)
            if sparse:
                content[::3] = 0
            for scale in [1e-6, .01, 1., 100., 1e6]:
                with self.subTest(sparse=sparse, scale=scale):
                    logscale = torch.tensor(math.log(scale), dtype=DTYPE, requires_grad=True)
                    function = lambda value: theoretical_map(content, value, variance, floor_variance, 'resolvent')
                    output = function(logscale)
                    derivative = torch.autograd.functional.jacobian(function, logscale)
                    g = -scale*variance/(2+scale*variance+floor_variance)
                    weights = output.reshape(-1, 2).square().sum(-1)/content.square().sum()
                    mean = (weights*g).sum()
                    expected = ((weights*(g-mean).square()).sum()).sqrt()*content.norm()
                    torch.testing.assert_close(derivative.norm(), expected, atol=2e-13, rtol=2e-11)
                    self.assertLessEqual(float(derivative.norm()/content.norm()), .5+1e-12)
                    torch.testing.assert_close(output.norm(), content.norm())

    def test_one_half_bound_is_tight_even_with_positive_isotropic_floor(self):
        variance = torch.tensor([0., 1e8], dtype=DTYPE)
        floor_variance = torch.full((2,), .25**2, dtype=DTYPE)
        amplitude = (1+(variance+floor_variance)/2).reciprocal()
        content = torch.tensor([1/amplitude[0], 0., 1/amplitude[1], 0.], dtype=DTYPE)
        content /= content.norm()
        point = torch.tensor(0., dtype=DTYPE, requires_grad=True)
        function = lambda value: theoretical_map(content, value, variance, floor_variance, 'resolvent')
        derivative = torch.autograd.functional.jacobian(function, point)
        self.assertGreater(float(derivative.norm()), .4999999)
        self.assertLess(float(derivative.norm()), .5)

    def test_gaussian_normalized_filter_has_no_uniform_one_half_bound(self):
        for contrast in [4., 20., 40.]:
            variance = torch.tensor([0., contrast], dtype=DTYPE)
            floor_variance = torch.full((2,), .25**2, dtype=DTYPE)
            content = torch.tensor([math.exp(-contrast/2), 0., 1., 0.], dtype=DTYPE)
            content /= content.norm()
            point = torch.tensor(0., dtype=DTYPE, requires_grad=True)
            function = lambda value: theoretical_map(content, value, variance, floor_variance, 'gaussian')
            derivative = torch.autograd.functional.jacobian(function, point)
            # At this point the two filtered block energies are exactly equal.
            # Gaussian g=(0,-contrast/2), so the derivative norm is contrast/4.
            torch.testing.assert_close(derivative.norm(), torch.tensor(contrast/4, dtype=DTYPE))
            self.assertGreater(float(derivative.norm()), .5)

    def test_each_side_pair_score_logscale_derivative_is_at_most_one_half(self):
        generator = torch.Generator().manual_seed(777)
        content_q = torch.randn(48, generator=generator, dtype=DTYPE)
        content_k = torch.randn(48, generator=generator, dtype=DTYPE)
        content_q /= content_q.norm()
        content_k /= content_k.norm()
        variance_q = torch.logspace(-3, 3, 24, dtype=DTYPE)
        variance_k = torch.logspace(2, -2, 24, dtype=DTYPE)
        floor = .25**2*torch.logspace(-2, 1.2, 24, dtype=DTYPE)
        scale_q = torch.tensor(.7, dtype=DTYPE, requires_grad=True)
        scale_k = torch.tensor(-.8, dtype=DTYPE, requires_grad=True)
        q = theoretical_map(content_q, scale_q, variance_q, floor, 'resolvent')
        k = theoretical_map(content_k, scale_k, variance_k, floor, 'resolvent')
        dq, dk = torch.autograd.grad(q@k, (scale_q, scale_k))
        self.assertLessEqual(float(dq.abs()), .5+1e-12)
        self.assertLessEqual(float(dk.abs()), .5+1e-12)

    def test_total_support_mixture_is_not_fixed_resolution_blur_convolution(self):
        variance = torch.tensor([0., 1., 10.], dtype=DTYPE)
        floor_variance = torch.tensor([2., .5, 1.], dtype=DTYPE)
        mixed_total = (1+(variance+floor_variance)/2).reciprocal()
        fixed_blur = (-floor_variance/2).exp()/(1+variance/2)
        self.assertTrue((mixed_total > fixed_blur).all())
        self.assertGreater(float((mixed_total-fixed_blur).abs().max()), .1)


if __name__ == '__main__':
    unittest.main()
