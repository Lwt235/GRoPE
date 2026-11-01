"""Pure toy identities, stability and failed-work tests for R27 reference."""
import ast
import math
from pathlib import Path
import time

import numpy as np
import pytest
from scipy.special import ndtr

from experiments import round27_rms_contract as C
from experiments import round27_rms_reference as R


def toy_bounds():
    return np.asarray([[-.8, -.6, .1, .2], [.1, -.6, .7, .2],
                       [-.8, .2, .1, .9], [.1, .2, .7, .9]], np.float64)


def policy():
    return dict(C.POLICY, h=.31)


def run(mu, factor, bounds=None, bias=None, order=32, **kwargs):
    return R.integrate(np.asarray(mu, np.float64), np.asarray(factor, np.float64),
        toy_bounds() if bounds is None else bounds, bias,
        policy=policy(), reference_policy=C.REFERENCE_POLICY, order=order, **kwargs)


def test_reference_has_no_producer_or_fixture_import():
    tree = ast.parse(Path(R.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert all(not (n or '').startswith(('models', 'experiments')) for n in imports)


@pytest.mark.parametrize('mid,half', [(0., 1e-20), (9., 1e-8), (-9., 1e-8), (.2, .1)])
def test_positive_narrow_normal_interval(mid, half):
    actual = float(R.normal_interval(mid, half, reference_policy=C.REFERENCE_POLICY))
    nodes, weights = np.polynomial.legendre.leggauss(64)
    expected = half*np.sum(weights*np.exp(-.5*(mid+half*nodes)**2))/math.sqrt(2*math.pi)
    assert actual > 0 and math.isclose(actual, expected, rel_tol=1e-13, abs_tol=0)


@pytest.mark.parametrize('mid,half', [(5., .8), (-5., .8), (0., 2.3), (.4, 1.2)])
def test_broad_reflected_normal_intervals(mid, half):
    actual = float(R.normal_interval(mid, half, reference_policy=C.REFERENCE_POLICY))
    expected = ndtr(-mid+half)-ndtr(-mid-half) if mid > 0 else ndtr(mid+half)-ndtr(mid-half)
    assert math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-16)


def test_structural_zero_uses_exact_area_and_bias_with_no_nodes():
    bounds = toy_bounds(); bias = np.asarray([.2, -.7, .4, -.1])
    result = run([.4, -.2, 0.], np.diag([.7, .2, 0.]), bias=bias)
    area = np.prod(bounds[:, 2:]-bounds[:, :2], axis=1)
    expected = area*np.exp(bias); expected /= expected.sum()
    assert result['passed'] and result['structural_branch'] == 'zero_depth'
    assert np.array_equal(result['raw_mass'], np.zeros(4))
    assert result['raw_fov_mass'] == 0. and result['work']['cell_node_evaluations'] == 0
    assert result['truncation_bounds']['raw_l1_over_reference_fov'] is None
    assert np.allclose(result['probability'], expected, rtol=1e-14, atol=0)


def test_negative_constant_depth_is_structural_zero():
    result = run([0., 0., -1.7], np.diag([.7, .2, 0.]))
    assert result['passed'] and result['structural_branch'] == 'nonpositive_constant_depth'
    assert result['front_mass'] == 0 and result['work']['cell_node_evaluations'] == 0


def test_negative_constant_depth_does_not_normalize_irrelevant_huge_xy():
    result = run([1e308, -1e308, -.01], np.diag([1e308, 1e308, 0.]))
    assert result['passed'] and result['structural_zero']
    assert np.array_equal(result['raw_mass'], np.zeros(4))


def test_positive_constant_depth_matches_analytic_product_cdf():
    mu = np.asarray([.3, -.2, 1.7]); factor = np.diag([.6, .4, 0.])
    result = run(mu, factor)
    bounds = toy_bounds(); sd = np.sqrt(np.asarray([.6, .4])**2/1.7**2+.31**2)
    expected = np.prod(ndtr((bounds[:, 2:]-mu[:2]/1.7)/sd)-ndtr((bounds[:, :2]-mu[:2]/1.7)/sd), axis=1)
    assert result['passed'] and result['front_mass'] == 1
    assert np.allclose(result['raw_mass'], expected, rtol=2e-12, atol=1e-15)


def cauchy_rectangle(box, h):
    def corner(x, y):
        return math.atan2(x*y, h*math.sqrt(h*h+x*x+y*y))/(4*math.pi)
    x0, y0, x1, y1 = box
    return corner(x1,y1)-corner(x0,y1)-corner(x1,y0)+corner(x0,y0)


def test_radial_zero_mean_matches_analytic_half_mass_cauchy():
    result = run([0., 0., 0.], np.diag([0., 0., .9]))
    expected = np.asarray([cauchy_rectangle(b, .31) for b in toy_bounds()])
    assert result['passed'] and result['front_mass'] == .5
    assert np.allclose(result['raw_mass'], expected, rtol=2e-11, atol=1e-13)
    assert result['work']['depth_evaluations'] > 0
    assert result['work']['outer_geometry_breakpoints'] > 0


def test_svd_conditioning_preserves_h_noise_and_world_scale():
    mu = np.asarray([.2, -.3, .9]); factor = np.asarray([[.7, .2, .1], [.5, .6, -.2], [.3, -.1, .4]])
    one, two = R.prepare(mu, factor, .31), R.prepare(8*mu, 8*factor, .31)
    for key in ['xy_mean', 'beta', 'conditional_factor']:
        assert np.array_equal(one[key], two[key])
    assert one['conditional_y_residual_std'] >= .31
    fz = factor[2]/math.hypot(mu[2], *factor[2])
    fxy = factor[:2]/math.hypot(mu[2], *factor[2])
    expected = fxy@fxy.T - np.outer(one['beta'], one['beta'])+.31**2*np.eye(2)
    actual = one['conditional_factor']@one['conditional_factor'].T
    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-15)


def test_near_constant_remains_stochastic_and_matches_constant_limit():
    factor = np.diag([.5, .3, 2.**-40])
    result = run([.2, -.1, 1.3], factor)
    constant = run([.2, -.1, 1.3], np.diag([.5, .3, 0.]))
    assert result['passed'] and result['structural_branch'] == 'stochastic_depth'
    assert result['normalized_depth_std'] > 0 and result['work']['depth_evaluations'] > 0
    assert np.max(np.abs(result['raw_mass']-constant['raw_mass'])) < 1e-11


def test_reference_dual_orders_on_correlated_toy():
    mu = [.1, -.2, .8]; factor = [[.5, .1, .1], [.2, .6, -.1], [.2, -.1, .4]]
    low = run(mu, factor, order=32)
    high = run(mu, factor, order=48)
    assert low['passed'] and high['passed']
    assert np.max(np.abs(low['probability']-high['probability'])) < 1e-8
    assert np.sum(np.abs(low['raw_mass']-high['raw_mass']))/sum(high['raw_mass']) < 1e-8


def test_ragged_batch_matches_scalar_rectangle_with_identical_nodes():
    prepared = R.prepare(np.asarray([.2, -.1, .9]), np.asarray([[.6,.1,.2],[.4,.5,-.2],[.1,-.2,.3]]), .31)
    boxes = np.vstack((toy_bounds(), [-.8,-.6,.7,.9]))
    one = R._Ledger(C.REFERENCE_POLICY, None, R.NodeBudget(1000000))
    two = R._Ledger(C.REFERENCE_POLICY, None, R.NodeBudget(1000000))
    location = prepared['xy_mean']+.7*prepared['beta']
    z = prepared['normalized_depth_mean']+.7*prepared['normalized_depth_std']
    batch = R._rectangles(location, z, boxes, prepared, 32, C.REFERENCE_POLICY, one)
    scalar = np.asarray([R._rectangle(location,z,b,prepared,32,C.REFERENCE_POLICY,two) for b in boxes])
    assert np.allclose(batch, scalar, rtol=1e-14, atol=0)
    assert one.work['cell_node_evaluations'] == two.work['cell_node_evaluations']
    assert one.work['inner_panels'] == two.work['inner_panels']


def test_shared_budget_accounts_for_x_and_all_narrow_y_nodes():
    budget = R.NodeBudget(200)
    bounds = np.asarray([[-.1, -.01, .1, .01]], np.float64)
    with pytest.raises(R.ReferenceFailure) as failure:
        run([0., 0., 1.], np.zeros((3,3)), bounds=bounds, budget=budget)
    # Native and independent FOV each have two panels after the fixed grid adds0.
    assert budget.used == 128
    assert failure.value.reference_work['cell_node_evaluations'] == budget.used
    assert failure.value.reference_work['inner_x_nodes'] == 128
    assert failure.value.reference_work['narrow_y_pdf_nodes'] == 0


def test_expired_deadline_preserves_zero_work():
    budget = R.NodeBudget(100000)
    with pytest.raises(R.ReferenceFailure) as failure:
        run([0., 0., 1.], np.zeros((3,3)), budget=budget, deadline=time.perf_counter()-1)
    assert budget.used == failure.value.reference_work['cell_node_evaluations'] == 0


def test_positive_depth_outside_cutoff_is_failure_not_floor_pass():
    with pytest.raises(R.ReferenceFailure, match='beyond the fixed reference cutoff') as failure:
        run([0., 0., -20.], np.diag([0., 0., 1.]))
    assert failure.value.reference_metadata['structural_zero'] is False
    assert failure.value.reference_metadata['front_mass'] > 0
    assert np.array_equal(failure.value.partial_raw_mass, np.zeros(4))


def test_tail_gate_failure_retains_positive_raw_probability_and_work():
    # This toy interval is so deep in a constant-depth Gaussian tail that the
    # deliberately global +/-32 bound is too loose relative to its raw mass.
    bounds = np.asarray([[31.5*.31, -.1, 32.*.31, .1]], np.float64)
    result = run([0., 0., 1.], np.zeros((3,3)), bounds=bounds)
    assert result['passed'] is False and result['status'] == 'failed'
    assert result['raw_mass'][0] > 0 and result['probability'][0] == 1.
    assert result['truncation_bounds']['raw_l1_over_reference_fov'] > 1e-10
    assert result['work']['cell_node_evaluations'] > 0


@pytest.mark.parametrize('bad', ['factor_dtype','bounds_overlap','mu_nonfinite','order'])
def test_invalid_input_is_explicit_failure_with_work(bad):
    mu, factor, bounds, order = np.zeros(3), np.eye(3), toy_bounds(), 32
    if bad == 'factor_dtype': factor = factor.astype(np.float32)
    elif bad == 'bounds_overlap': bounds[1] = bounds[0]
    elif bad == 'mu_nonfinite': mu[0] = np.inf
    else: order = 64
    with pytest.raises(R.ReferenceFailure) as failure:
        R.integrate(mu, factor, bounds, policy=policy(), reference_policy=C.REFERENCE_POLICY, order=order)
    assert failure.value.reference_work['cell_node_evaluations'] == 0


def _analytic_wide_gaussian(xcuts):
    """Independent constant-depth Gaussian CDF oracle, without producer code."""
    h = .29
    mu = np.asarray([.13, -.22, 1.4], np.float64)
    factor = np.diag([.8, .5, 0.])
    mean = mu[:2]/mu[2]
    sd = np.sqrt((np.diag(factor)[:2]/mu[2])**2+h*h)
    ycuts = np.asarray([-1.25, .4, 1.25])
    standardized = np.asarray([(x0,y0,x1,y1) for y0,y1 in zip(ycuts,ycuts[1:])
                               for x0,x1 in zip(xcuts,xcuts[1:])], np.float64)
    bounds = standardized*np.tile(sd,2)+np.tile(mean,2)
    expected = np.prod(ndtr(standardized[:,2:])-ndtr(standardized[:,:2]), axis=1)
    direct = float((ndtr(xcuts[-1])-ndtr(xcuts[0]))*(ndtr(ycuts[-1])-ndtr(ycuts[0])))
    grid = R.standard_x_grid(C.REFERENCE_POLICY)
    expected_panels = 0
    for lo,hi in list(zip(xcuts,xcuts[1:]))*2+[(xcuts[0],xcuts[-1])]:
        cuts = np.unique(np.r_[lo,grid[(lo<grid)&(grid<hi)],hi])
        assert np.max(np.diff(cuts)) <= 2.
        expected_panels += len(cuts)-1
    for order in (32,48):
        result = R.integrate(mu,factor,bounds,policy=dict(C.POLICY,h=h),
                             reference_policy=C.REFERENCE_POLICY,order=order)
        assert result['passed']
        assert np.max(np.abs(result['raw_mass']-expected)) <= 1e-12
        assert abs(result['raw_fov_mass']-direct) <= 1e-12
        assert abs(math.fsum(map(float,result['raw_mass']))-result['raw_fov_mass']) <= 1e-12
        # Physical-to-standardized roundtrip can place a nominal grid endpoint
        # one ulp beyond it; exact duplicate removal then retains a tiny panel.
        assert result['work']['inner_panels'] >= expected_panels
        assert result['work']['inner_x_nodes'] == result['work']['inner_panels']*order


def test_analytic_wide_independent_gaussian_and_unequal_native_additivity():
    _analytic_wide_gaussian(np.asarray([-24.,-.35,.8,24.]))


def test_analytic_asymmetric_wide_independent_gaussian_covers_marginal_peak():
    _analytic_wide_gaussian(np.asarray([-31.7,-.35,.8,30.2]))
