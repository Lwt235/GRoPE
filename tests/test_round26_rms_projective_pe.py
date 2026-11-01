"""Independent small synthetic checks, without formal R26 fixture imports."""
import math
import time

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from models import round26_rms_projective_pe as P

POLICY = dict(h=.25, floor_density=1e-30, quadrature_order=24,
    maximum_tile_side_in_h=2., max_cell_node_evaluations=4000000,
    negative_tail_threshold=10., negative_tail_series_terms=20,
    max_negative_tail_relative_remainder=1e-10,
    projection_regularization='rms_depth_xy_gaussian')
BOUNDS = np.asarray([[-.4, -.6, .1, .9], [.1, -.6, .7, .9]], np.float64)
MEAN = np.asarray([.26, -.14, 2.], np.float64)
FACTOR = np.asarray([[.23, .07, .11], [-.03, .31, .17], [.05, -.02, .37]], np.float64)


def diagnostics():
    return dict(constant_depth_nodes=0, positive_branch_nodes=0, negative_branch_nodes=0,
        negative_erfcx_nodes=0, negative_series_nodes=0, maximum_density=0.,
        maximum_log_negative_tail_relative_remainder=None,
        minimum_scaled_negative_tail_series_lower_bound=None,
        maximum_scaled_negative_tail_series_upper_bound=None)


def run(mean=MEAN, factor=FACTOR, bounds=BOUNDS, bias=None, policy=POLICY, deadline=None):
    return P.integrate(mean, factor, bounds, bias, policy=policy, deadline=deadline)


def test_point_limit_is_same_gaussian_cell_mass():
    result = run(factor=np.zeros((3, 3), np.float64))
    assert result['passed'] and result['structural_branch'] == 'positive_constant_depth_gaussian'
    center = MEAN[:2]/MEAN[2]
    expected = np.prod(ndtr((BOUNDS[:, 2:]-center)/POLICY['h'])-
                       ndtr((BOUNDS[:, :2]-center)/POLICY['h']), axis=1)
    np.testing.assert_allclose(result['raw_mass'], expected, rtol=2e-13, atol=1e-15)
    assert result['front_mass'] == 1. and result['zero_raw_cells'] == 0


@pytest.mark.parametrize('z', [0., -1.3])
def test_exact_nonpositive_constant_depth_is_structural_zero(z):
    mean = MEAN.copy(); mean[2] = z
    factor = FACTOR.copy(); factor[2] = 0
    result = run(mean, factor)
    assert result['passed'] and result['structural_zero']
    assert result['front_mass'] == 0. and result['work']['cell_node_evaluations'] == 0
    np.testing.assert_array_equal(result['raw_mass'], np.zeros(2))
    area = np.prod(BOUNDS[:, 2:]-BOUNDS[:, :2], axis=1)
    np.testing.assert_allclose(result['probability'], area/area.sum(), rtol=2e-14)
    assert result['analytic_truncation_bound']['raw_native_l1_absolute'] == 0.


def test_radial_zero_mean_is_half_mass_cauchy_density():
    radial = np.asarray([.19, -.27, 1.])
    factor = np.zeros((3, 3)); factor[:, 0] = 2.3*radial
    p = P.prepare(np.zeros(3), factor, BOUNDS, None, POLICY)
    points = np.asarray([[-.13, .04], [.43, -.12], [1.1, -.7]])
    actual = P.density(points, p, POLICY, diagnostics())
    radius = np.sum((points-radial[:2])**2, axis=1)/POLICY['h']**2
    expected = (1.+radius)**-1.5/(4.*math.pi*POLICY['h']**2)
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    assert p['front_mass'] == .5 and not p['structural_zero']


@pytest.mark.parametrize('depth', [.8, -.7])
def test_density_matches_independent_three_dimensional_ray_integral(depth):
    mean = np.asarray([-.3, .2, depth])
    factor = np.asarray([[.4, .1, .3], [-.2, .5, .1], [0., 0., .8]])
    p = P.prepare(mean, factor, BOUNDS, None, POLICY)
    covariance = factor@factor.T
    covariance[:2, :2] += POLICY['h']**2*(mean[2]**2+np.dot(factor[2], factor[2]))*np.eye(2)
    precision = np.linalg.inv(covariance)
    scale = (2.*math.pi)**-1.5/np.sqrt(np.linalg.det(covariance))
    points = np.asarray([[-.7, .2], [.1, .9], [.8, -.4]])
    expected = []
    for point in points:
        ray = np.r_[point, 1.]
        def integrand(z):
            displacement = z*ray-mean
            return scale*z*z*math.exp(-.5*float(displacement@precision@displacement))
        expected.append(quad(integrand, 0., np.inf, epsabs=1e-13, epsrel=1e-12)[0])
    np.testing.assert_allclose(P.density(points, p, POLICY, diagnostics()), expected, rtol=2e-10, atol=2e-13)


@pytest.mark.parametrize('q', [0., 2.5, 9.99, 10., 17.3])
def test_negative_k_branches_against_positive_scalar_integral(q):
    actual, report = P.negative_log_k(np.asarray([q]), POLICY)
    if q == 0:
        expected = math.sqrt(math.pi/2.)
    else:
        expected = quad(lambda v:v*v*math.exp(-v-v*v/(2.*q*q)), 0., np.inf,
                        epsabs=1e-13, epsrel=1e-13)[0]/q**3
    np.testing.assert_allclose(np.exp(actual), [expected], rtol=8e-12, atol=0)
    assert report['series_nodes'] == int(q >= 10.)
    if q >= 10.:
        assert report['maximum_log_relative_remainder'] <= math.log(1e-10)
        assert report['minimum_scaled_series_lower_bound'] > 0
        assert report['maximum_scaled_series_upper_bound'] >= report['minimum_scaled_series_lower_bound']


def test_conditional_factor_matches_direct_covariance_without_subtraction_in_model():
    p = P.prepare(MEAN, FACTOR, BOUNDS, None, POLICY)
    normalized = FACTOR/p['rms_depth']
    covariance = normalized@normalized.T
    expected = covariance[:2, :2]-np.outer(covariance[:2, 2], covariance[:2, 2])/covariance[2, 2]+POLICY['h']**2*np.eye(2)
    np.testing.assert_allclose(p['conditional_factor']@p['conditional_factor'].T, expected, rtol=2e-14, atol=1e-16)
    assert np.all(np.diag(p['conditional_factor']) > 0)


def test_depth_variation_tending_to_zero_keeps_full_xy_marginal():
    tiny = FACTOR.copy(); tiny[2] *= 1e-10
    exact = tiny.copy(); exact[2] = 0
    a, b = run(factor=tiny), run(factor=exact)
    assert a['passed'] and b['passed'] and a['structural_branch'] == 'variable_depth'
    np.testing.assert_allclose(a['raw_mass'], b['raw_mass'], rtol=2e-9, atol=2e-13)


def test_world_scale_and_factor_basis_leave_native_mass_unchanged():
    q, _ = np.linalg.qr(np.asarray([[.2, .8, -.3], [.7, -.1, .4], [.1, .5, .9]]))
    original = run()
    scaled = run(MEAN*3.7, FACTOR*3.7)
    basis = run(factor=FACTOR@q)
    assert all(r['passed'] for r in (original, scaled, basis))
    np.testing.assert_allclose(scaled['raw_mass'], original['raw_mass'], rtol=3e-13, atol=1e-15)
    np.testing.assert_allclose(basis['raw_mass'], original['raw_mass'], rtol=3e-13, atol=1e-15)


def test_screen_translation_and_uniform_units():
    offset = np.asarray([.14, -.23])
    shifted_mean = MEAN.copy(); shifted_mean[:2] += offset*MEAN[2]
    shifted_factor = FACTOR.copy(); shifted_factor[:2] += offset[:, None]*FACTOR[2]
    shifted_bounds = BOUNDS+np.r_[offset, offset]
    original, shifted = run(), run(shifted_mean, shifted_factor, shifted_bounds)
    unit = 1.7
    mean, factor = MEAN.copy(), FACTOR.copy()
    mean[:2] *= unit; factor[:2] *= unit
    scaled = run(mean, factor, BOUNDS*unit, policy=dict(POLICY, h=POLICY['h']*unit,
        floor_density=POLICY['floor_density']/unit**2))
    assert original['passed'] and shifted['passed'] and scaled['passed']
    np.testing.assert_allclose(shifted['raw_mass'], original['raw_mass'], rtol=3e-13)
    np.testing.assert_allclose(scaled['raw_mass'], original['raw_mass'], rtol=3e-13)
    np.testing.assert_allclose(scaled['probability'], original['probability'], rtol=3e-13)


def test_content_follows_one_floor_without_changing_raw_mass():
    original, biased = run(), run(bias=np.asarray([.31, -.19]))
    np.testing.assert_array_equal(original['raw_mass'], biased['raw_mass'])
    expected = original['floored_mass']*np.exp(np.asarray([.31, -.19]))
    expected /= expected.sum()
    np.testing.assert_allclose(biased['probability'], expected, rtol=1e-14)
    assert abs(biased['raw_mass_sum']-biased['raw_fov_mass']) <= 1e-12


def test_exact_arithmetic_bound_and_all_nodes_are_reported():
    result = run()
    area = np.prod(BOUNDS[:, 2:]-BOUNDS[:, :2], axis=1).sum()
    assert result['analytic_truncation_bound']['raw_native_l1_absolute'] == area/(math.pi*POLICY['h']**2*math.comb(48, 24))
    assert result['analytic_truncation_bound']['exact_arithmetic_only']
    assert not result['quadrature_error_bound_available']
    work, d = result['work'], result['density_diagnostics']
    assert work['cell_node_evaluations'] == (result['native_tile_count']+result['fov_tile_count'])*576
    assert d['positive_branch_nodes']+d['negative_branch_nodes']+d['constant_depth_nodes'] == work['cell_node_evaluations']
    assert d['maximum_density'] <= 1./(2.*math.pi*POLICY['h']**2)*(1+1e-13)


def test_budget_and_deadline_failures_keep_partial_mass():
    low = run(policy=dict(POLICY, max_cell_node_evaluations=576))
    expired = run(deadline=time.perf_counter()-1)
    for result in (low, expired):
        assert not result['passed'] and result['status'] == 'failed'
        assert 'partial_raw_mass' in result and 'partial_raw_fov_mass' in result
        assert 'probability' not in result
    assert expired['work']['cell_node_evaluations'] == 0


def test_positive_density_underflow_is_failure_not_structural_zero():
    result = run(mean=np.asarray([1e6, 1e6, 1.]), factor=np.zeros((3, 3)))
    assert not result['passed'] and not result['structural_zero']
    assert 'not a structural zero' in result['failure_reason']
    assert result['work']['cell_node_evaluations'] > 0 and 'probability' not in result


def test_nonzero_depth_variation_cannot_become_exact_zero_by_underflow():
    mean = np.asarray([0., 0., 1e308]); factor = np.zeros((3, 3))
    factor[2, 0] = np.nextafter(0., 1.)
    result = run(mean, factor)
    assert not result['passed'] and 'nonzero depth variation lost' in result['failure_reason']
    assert 'probability' not in result


@pytest.mark.parametrize('h', [1e153, 1e-160])
def test_unrepresentable_analytic_bound_fails_without_nonfinite_metadata(h):
    result = run(factor=np.zeros((3, 3)), bounds=np.asarray([[-100., -100., 100., 100.]]),
                 policy=dict(POLICY, h=h))
    assert not result['passed'] and 'quadrature bound is not representable' in result['failure_reason']
    assert 'analytic_truncation_bound' not in result and 'probability' not in result


@pytest.mark.parametrize('defect', ['mu_dtype', 'factor_shape', 'nonfinite', 'overlap', 'order', 'tile', 'tail', 'h_square'])
def test_declared_input_contract_rejects_invalid_values(defect):
    mean, factor, bounds, policy = MEAN.copy(), FACTOR.copy(), BOUNDS.copy(), dict(POLICY)
    if defect == 'mu_dtype': mean = mean.astype(np.float32)
    elif defect == 'factor_shape': factor = factor[:, :2]
    elif defect == 'nonfinite': factor[0, 0] = np.nan
    elif defect == 'overlap': bounds[1, 0] = -.1
    elif defect == 'order': policy['quadrature_order'] = 32
    elif defect == 'tile': policy['maximum_tile_side_in_h'] = 4.
    elif defect == 'tail': policy['negative_tail_series_terms'] = 24
    elif defect == 'h_square': policy['h'] = 1e200
    with pytest.raises(ValueError): run(mean, factor, bounds, policy=policy)
