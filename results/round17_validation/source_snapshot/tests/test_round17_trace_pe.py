"""Synthetic math/numeric contracts only; no actual prediction or target reads."""
import math

import pytest
import torch

from models import round17_trace_pe as M
from models.round10_cell_integral_fast import gaussian_cell_mass_fast


def covariance(values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_closed_forms_zero_isotropic_rank_one_and_dtype(dtype):
    c = covariance([[[0, 0], [0, 0]], [[2, 0], [0, 2]], [[2, 2], [2, 2]]], dtype)
    tau = M.trace_temperature(c, .5, 5.)
    expected = covariance([5., 5.*.25/4.25, 5.*.25/4.25], dtype)
    torch.testing.assert_close(tau, expected, atol=0, rtol=2.e-7 if dtype == torch.float32 else 2.e-16)
    assert tau.dtype == dtype and tau.shape == (3,)
    assert bool(((tau >= 0) & (tau <= 5.)).all())


def test_rotation_and_coordinate_unit_scaling():
    c = covariance([[[2, .4], [.4, .7]], [[.3, -.2], [-.2, 1.4]]])
    theta = .731
    r = covariance([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    baseline = M.trace_temperature(c, .5, 5.)
    torch.testing.assert_close(M.trace_temperature(r @ c @ r.T, .5, 5.), baseline, rtol=5.e-16, atol=1.e-16)
    for scale in (1.e-6, 8., 1.e6):
        torch.testing.assert_close(M.trace_temperature(c * scale**2, .5*scale, 5.), baseline, rtol=5.e-16, atol=1.e-16)


def test_continuous_gaussian_response_bound_and_rank_one_equality():
    generator = torch.Generator().manual_seed(1701)
    l = torch.randn((13, 2, 2), generator=generator, dtype=torch.float64)
    c = l @ l.transpose(-1, -2)
    h, t = .5, 5.
    tau = M.trace_temperature(c, h, t)
    largest = torch.linalg.eigvalsh(c + h*h*torch.eye(2, dtype=torch.float64))[:, -1]
    assert bool((tau * largest <= t*h*h + 1.e-15).all())
    rank = covariance([[[4, 0], [0, 0]]])
    assert M.trace_temperature(rank, h, t).item() * 4.25 == t*h*h


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_point_bias_is_bitwise_old_float32_scalar_path(dtype):
    content = torch.tensor([[.12345679, -.9876543, .8], [1., -.3, .22222222]], dtype=torch.float32)
    c = torch.zeros((2, 2, 2), dtype=dtype)
    for h in (.25, .5, 1., 2., .37):
        for t in (0., 5., 10., 1.3):
            assert torch.equal(M.content_bias(c, h, t, content), t * content)


def test_float32_bias_multiply_occurs_before_double_addition():
    c = covariance([[[.2, .03], [.03, .1]]], torch.float32)
    content = torch.tensor([[.12345679, .9876543, -.2]], dtype=torch.float32)
    tau32 = M.trace_temperature(c, .5, 5.)
    actual = M.content_bias(c, .5, 5., content)
    assert torch.equal(actual, tau32[:, None] * content)
    assert not torch.equal(actual.double(), tau32.double()[:, None] * content.double())
    geometry = torch.tensor([[-3., -2., -1.]], dtype=torch.float64)
    expected = (geometry + actual.double()).softmax(-1)
    assert torch.equal(M.probabilities(geometry, content, c, .5, 5.), expected)


def test_t0_and_point_probabilities_exact_with_unchanged_gaussian_geometry():
    mean = covariance([[.2, .4], [1.1, -.2]])
    c = covariance([[[.4, .1], [.1, .6]], [[.2, -.03], [-.03, .3]]], torch.float32)
    boxes = covariance([[-.5, -.5, .5, .5], [.5, -.5, 1.5, .5], [-.5, .5, .5, 1.5]])
    content = torch.tensor([[.3, -.7, .1], [.12345679, .9876543, -.9]], dtype=torch.float32)
    for actual_cov in (c, torch.zeros_like(c)):
        mass = gaussian_cell_mass_fast(mean, actual_cov, boxes, bandwidth=.5, quadrature_order=32)
        geometry = (mass + 1.e-30).log()
        assert torch.equal(M.probabilities(geometry, content, actual_cov, .5, 0.), geometry.softmax(-1))
        if not bool(actual_cov.any()):
            for t in (5., 10.):
                assert torch.equal(M.probabilities(geometry, content, actual_cov, .5, t),
                                   (geometry + t*content).softmax(-1))


def test_tau_gradcheck_and_t0_preserves_temperature_gradient():
    l = covariance([[[1.2, .1], [.3, .7]], [[.6, -.2], [.1, 1.3]]]).requires_grad_()
    h = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(5., dtype=torch.float64, requires_grad=True)
    function = lambda factor, width, temperature: M.trace_temperature(factor @ factor.transpose(-1, -2), width, temperature)
    assert torch.autograd.gradcheck(function, (l, h, t), eps=1.e-6, atol=1.e-6, rtol=1.e-5)
    zero_t = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    tau = M.trace_temperature(l @ l.transpose(-1, -2), h, zero_t)
    derivative, = torch.autograd.grad(tau.sum(), zero_t)
    assert derivative > 0 and torch.equal(tau, torch.zeros_like(tau))


def test_probability_backpropagates_through_all_inputs():
    l = covariance([[[1., .2], [.1, .7]]], torch.float32).requires_grad_()
    c = l @ l.transpose(-1, -2)
    geometry = covariance([[-1., -2., -.5]]).requires_grad_()
    content = torch.tensor([[.1, .7, -.3]], dtype=torch.float32, requires_grad=True)
    t = torch.tensor(5., dtype=torch.float64, requires_grad=True)
    h = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    p = M.probabilities(geometry, content, c, h, t)
    p[0, 0].backward()
    for value in (l, geometry, content, t, h):
        assert value.grad is not None and bool(torch.isfinite(value.grad).all())
        assert bool((value.grad != 0).any())


def test_extreme_finite_covariance_does_not_overflow_trace():
    c32 = torch.diag(torch.tensor([3.e38, 3.e38], dtype=torch.float32))[None]
    tau32 = M.trace_temperature(c32, 1., 5.)
    assert tau32.item() > 0 and torch.isfinite(tau32).all()
    torch.testing.assert_close(tau32.double(), torch.tensor([5./(2.*float(c32[0, 0, 0]))], dtype=torch.float64), atol=1.e-45, rtol=1.e-6)
    c64 = torch.diag(torch.tensor([1.e308, 1.e308], dtype=torch.float64))[None]
    tau64 = M.trace_temperature(c64, 1., 5.)
    torch.testing.assert_close(tau64, torch.tensor([2.5e-308], dtype=torch.float64), rtol=1.e-14, atol=0)
    tiny = covariance([[[1.e-300, 0], [0, 1.e-300]]])
    assert M.trace_temperature(tiny, .5, 5.).item() == 5.


@pytest.mark.parametrize('c', [
    [[[1, 2], [2, 1]]], [[[1, 1], [0, 1]]], [[[-1, 0], [0, 2]]],
    [[[float('nan'), 0], [0, 1]]], [[[float('inf'), 0], [0, 1]]],
])
def test_rejects_invalid_covariance(c):
    with pytest.raises(ValueError):
        M.trace_temperature(covariance(c), .5, 5.)


@pytest.mark.parametrize('h,t', [(0, 5), (-1, 5), (float('nan'), 5), (float('inf'), 5),
                               (.5, -1), (.5, float('nan')), (.5, float('inf'))])
def test_rejects_invalid_scalar_parameters(h, t):
    with pytest.raises(ValueError):
        M.trace_temperature(torch.eye(2), h, t)


def test_rejects_shape_and_dtype_contract_violations():
    with pytest.raises(ValueError):
        M.trace_temperature(torch.ones(3, 3), .5, 5.)
    with pytest.raises(ValueError):
        M.trace_temperature(torch.eye(2, dtype=torch.int64), .5, 5.)
    with pytest.raises(ValueError):
        M.trace_temperature(torch.eye(2), [.25, .5], 5.)
    with pytest.raises(ValueError):
        M.content_bias(torch.eye(2)[None], .5, 5., torch.ones(2, 3))
    with pytest.raises(ValueError):
        M.content_bias(torch.eye(2)[None], .5, 5., torch.ones(1, 3, dtype=torch.float64))
    with pytest.raises(ValueError):
        M.probabilities(torch.ones(1, 3), torch.ones(1, 3), torch.eye(2)[None], .5, 5.)


def test_native_cell_curvature_is_not_continuous_point_curvature():
    # Independent analytic normal-CDF rectangle example. This deliberately
    # refutes interpreting the point response bound as an exact native bound.
    x = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    h, halfwidth = .25, .5
    normal_cdf = lambda value: .5 * (1. + torch.erf(value / math.sqrt(2.)))
    mass = normal_cdf((x+halfwidth)/h) - normal_cdf((x-halfwidth)/h)
    first, = torch.autograd.grad(mass.log(), x, create_graph=True)
    second, = torch.autograd.grad(first, x)
    native_curvature = -second.item()
    assert 3.6 < native_curvature < 3.7
    assert native_curvature < .25 / h**2
    assert 1./native_curvature > 4.*h**2
