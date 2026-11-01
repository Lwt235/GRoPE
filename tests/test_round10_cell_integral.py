"""Independent cell-integral numerics; no benchmark labels or scene data."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import _mvn
import torch

from models.round10_cell_integral import gaussian_cell_mass, normalized_cell_probabilities


DTYPE = torch.float64


def tensor(value):
    return torch.tensor(value, dtype=DTYPE)


def reference(mean, covariance, bounds, bandwidth):
    covariance = np.asarray(covariance) + bandwidth**2 * np.eye(2)
    value, status = _mvn.mvnun(np.asarray(bounds[:2]), np.asarray(bounds[2:]),
        np.asarray(mean), covariance, 100000, 1.e-13)
    assert status == 0
    return value


def test_independent_scipy_random_correlations_and_scales():
    rng = np.random.default_rng(7721)
    means = rng.uniform(-3, 3, (12, 2))
    a = rng.normal(size=(12, 2, 2))
    covs = a @ a.transpose(0, 2, 1) * np.geomspace(.01, 50, 12)[:, None, None]
    low = rng.uniform(-4, 1, (7, 2))
    high = low + rng.uniform(.02, 8, (7, 2))
    bounds = np.concatenate([low, high], axis=-1)
    actual = gaussian_cell_mass(tensor(means), tensor(covs), tensor(bounds), bandwidth=.25)
    expected = np.array([[reference(m, c, b, .25) for b in bounds] for m, c in zip(means, covs)])
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2.e-8, atol=3.e-12)


@pytest.mark.parametrize('rho', [-.99999999, -.999, -.3, 0., .3, .999, .99999999])
def test_nearly_singular_and_opposite_correlations(rho):
    means = tensor([[0., 0.], [1.3, -.8]])
    covariance = tensor([[1., rho], [rho, 1.]]).expand(2, 2, 2)
    bounds = tensor([[-2., -.1, .5, .1], [.001, .001, .002, .002],
                     [-8., -8., 8., 8.], [2.1, 2.2, 3., 3.1]])
    actual = gaussian_cell_mass(means, covariance, bounds, bandwidth=1.e-4)
    expected = np.array([[reference(m, c, b, 1.e-4) for b in bounds.numpy()]
                         for m, c in zip(means.numpy(), covariance.numpy())])
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2.e-6, atol=2.e-11)


def test_isotropic_separable_exact_reference():
    means = tensor([[0., 0.], [3., -2.]])
    cov = tensor([[[.2, 0.], [0., 3.]], [[4., 0.], [0., .04]]])
    bounds = tensor([[-4., -3., 3., 4.], [.3, -.2, .6, .2], [-20., 1., 20., 3.]])
    sigma = np.sqrt(cov.numpy().diagonal(axis1=-2, axis2=-1) + .3**2)
    a = (bounds.numpy()[None, :, :2] - means.numpy()[:, None]) / sigma[:, None]
    b = (bounds.numpy()[None, :, 2:] - means.numpy()[:, None]) / sigma[:, None]
    expected = (ndtr(b) - ndtr(a)).prod(-1)
    actual = gaussian_cell_mass(means, cov, bounds, bandwidth=.3)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2.e-11, atol=5.e-14)


def test_extreme_positive_tail_avoids_cdf_cancellation():
    bounds = tensor([[8., -.5, 8.2, .5], [9., -.5, 9.1, .5], [-9.1, -.5, -9., .5]])
    mean = tensor([[0., 0.]])
    cov = tensor([[[1., 0.], [0., 1.]]])
    actual = gaussian_cell_mass(mean, cov, bounds, bandwidth=.01)[0].numpy()
    sigma = math.sqrt(1.0001)
    expected = []
    for a, b, c, d in bounds.numpy():
        xm = ndtr(-a / sigma) - ndtr(-c / sigma) if a >= 0 else ndtr(c / sigma) - ndtr(a / sigma)
        expected.append(xm * (ndtr(d / sigma) - ndtr(b / sigma)))
    np.testing.assert_allclose(actual, expected, rtol=2.e-12, atol=0)
    assert (actual > 0).all()


def test_correlated_tail_against_adaptive_scalar_quad():
    rho = .8
    cov = tensor([[[1., rho], [rho, 1.]]])
    boxes = tensor([[7., 5., 7.3, 5.4]])
    width = .2
    actual = gaussian_cell_mass(tensor([[0., 0.]]), cov, boxes, bandwidth=width).item()
    sigma = math.sqrt(1. + width**2)
    r = rho / (1. + width**2)
    residual = math.sqrt(1. - r*r)
    def integrand(x):
        a = (5. / sigma - r*x) / residual
        b = (5.4 / sigma - r*x) / residual
        mass = ndtr(-a) - ndtr(-b) if a >= 0 else ndtr(b) - ndtr(a)
        return math.exp(-x*x/2) / math.sqrt(2*math.pi) * mass
    expected, err = quad(integrand, 7. / sigma, 7.3 / sigma, epsabs=1.e-25, epsrel=1.e-12)
    assert err < 1.e-23
    assert actual == pytest.approx(expected, rel=2.e-12, abs=1.e-25)


def test_true_cell_subdivision_raw_mass_and_normalized_consistency():
    mean = tensor([[.3, -.1], [-1., 2.], [2., -1.]])
    cov = tensor([[[2., 1.1], [1.1, 1.]], [[.02, 0.], [0., .03]], [[100., -99.], [-99., 100.]]])
    parent = tensor([[-4., -4., 0., 4.], [0., -4., 4., 4.]])
    child = tensor([[-4., -4., 0., 4.], [0., -4., 1., 0.], [0., 0., 1., 4.],
                    [1., -4., 4., 1.], [1., 1., 4., 4.]])
    a = gaussian_cell_mass(mean, cov, parent, bandwidth=.25)
    b = gaussian_cell_mass(mean, cov, child, bandwidth=.25)
    combined = torch.stack([b[:, 0], b[:, 1:].sum(-1)], -1)
    torch.testing.assert_close(a, combined, rtol=2.e-10, atol=2.e-12)
    torch.testing.assert_close(normalized_cell_probabilities(a),
        torch.stack([normalized_cell_probabilities(b)[:, 0], normalized_cell_probabilities(b)[:, 1:].sum(-1)], -1),
        rtol=2.e-10, atol=2.e-12)


def test_shared_translation_and_unit_scale():
    m = tensor([[.4, -.5], [3., 2.]])
    c = tensor([[[1., .4], [.4, .8]], [[.2, -.05], [-.05, .3]]])
    b = tensor([[-2., -3., 1., 2.], [1., -3., 4., 2.]])
    scale, shift = 9.3, tensor([87., -42.])
    expected = gaussian_cell_mass(m, c, b, bandwidth=.3)
    actual = gaussian_cell_mass(m*scale + shift, c*scale**2,
        b*scale + shift.repeat(2), bandwidth=.3*scale)
    torch.testing.assert_close(actual, expected, rtol=2.e-12, atol=2.e-13)


def test_batch_chunking_and_dtype_contract():
    m = tensor([[[0., 0.], [1., 1.]], [[-1., 2.], [.2, -.4]]])
    c = torch.eye(2, dtype=DTYPE).expand(2, 2, 2, 2) * .7
    b = tensor([[-4., -4., 0., 4.], [0., -4., 4., 4.]])
    a = gaussian_cell_mass(m, c, b, query_chunk_size=1, key_chunk_size=1)
    z = gaussian_cell_mass(m, c, b, query_chunk_size=99, key_chunk_size=99)
    torch.testing.assert_close(a, z, rtol=0., atol=0.)
    fp32 = gaussian_cell_mass(m.float(), c.float(), b.float())
    assert fp32.dtype == torch.float64
    torch.testing.assert_close(fp32, a, rtol=2.e-7, atol=2.e-9)


def test_autograd_through_mean_covariance_bandwidth_and_bounds():
    m = tensor([[.23, -.41]]).requires_grad_()
    factor = tensor([[[1.1, .0], [.35, .7]]]).requires_grad_()
    b = tensor([[-2., -1., .9, 2.2], [.9, -1., 3., 2.2]]).requires_grad_()
    width = tensor(.31).requires_grad_()
    def f(q, a, boxes, h):
        covariance = a @ a.transpose(-1, -2)
        return gaussian_cell_mass(q, covariance, boxes, bandwidth=h, quadrature_order=32)
    assert torch.autograd.gradcheck(f, (m, factor, b, width), eps=1.e-5, atol=2.e-6, rtol=2.e-5)


def test_zero_support_covariance_is_valid_with_positive_bandwidth():
    masses = gaussian_cell_mass(tensor([[0., 0.]]), torch.zeros(1, 2, 2, dtype=DTYPE),
        tensor([[-.5, -.5, .5, .5]]), bandwidth=.25)
    assert masses.item() == pytest.approx((ndtr(2.) - ndtr(-2.))**2, rel=1.e-12)


@pytest.mark.parametrize('reason', ['negative', 'nonsymmetric', 'nonfinite', 'box', 'bandwidth'])
def test_invalid_inputs_rejected(reason):
    m = tensor([[0., 0.]])
    c = torch.eye(2, dtype=DTYPE)[None]
    b = tensor([[-1., -1., 1., 1.]])
    kw = {}
    if reason == 'negative': c[0, 0, 0] = -1.
    if reason == 'nonsymmetric': c[0, 0, 1] = .1
    if reason == 'nonfinite': m[0, 0] = float('nan')
    if reason == 'box': b[0, 2] = -2.
    if reason == 'bandwidth': kw['bandwidth'] = 0.
    with pytest.raises(ValueError): gaussian_cell_mass(m, c, b, **kw)


def test_zero_fov_mass_requires_explicit_fallback():
    with pytest.raises(ValueError, match='zero FOV'):
        normalized_cell_probabilities(torch.zeros(2, 3, dtype=DTYPE))
    fixed = normalized_cell_probabilities(torch.zeros(2, 3, dtype=DTYPE), floor=1.e-30)
    torch.testing.assert_close(fixed, torch.full_like(fixed, 1/3))
