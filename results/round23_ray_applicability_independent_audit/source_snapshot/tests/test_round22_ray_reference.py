"""Separate toy inputs only: no formal fixture forward or old data files."""
from copy import deepcopy
import math
from pathlib import Path
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import ndtr
from experiments import round22_ray_contract as C
from experiments import round22_ray_reference as R
from experiments import round22_ray_audit as A


TOY_BOUNDS = np.array([[-.5, -.25, 0., .75], [0., -.25, .5, .75]], np.float64)


def reference(kind, mu, factor, order=64, bounds=TOY_BOUNDS, **kwargs):
    policy = dict(C.POLICY, h=.35)
    return R.integrate(np.asarray(mu, np.float64), np.asarray(factor, np.float64), bounds,
                       np.eye(2), np.zeros(2), policy, kind, order, C.REFERENCE_POLICY, **kwargs)


def test_normal_interval_mirrored_tails_and_infinite_endpoints():
    lo = np.array([-np.inf, 10., -11., np.inf, -np.inf]); hi = np.array([np.inf, 11., -10., np.inf, -np.inf])
    value = R.normal_interval(lo, hi)
    assert value[0] == 1. and value[1] > 0 and value[1] == value[2] and value[3] == value[4] == 0.
    with pytest.raises(ValueError): R.normal_interval(1., -1.)


def test_point_and_radial_analytic_raw_mass_keeps_positive_depth_weight():
    mu = np.array([-.25, .125, 3.]); result = reference('point', mu, np.empty((3, 0)))
    expected = R.point_rectangle(mu[:2] / mu[2], TOY_BOUNDS, .35)
    np.testing.assert_array_equal(result['raw_mass'], expected)
    direction = np.array([.25, .5, 1.]); m = -7.
    radial = reference('radial', direction * m, direction[:, None])
    np.testing.assert_allclose(radial['raw_mass'], ndtr(m) * R.point_rectangle(direction[:2], TOY_BOUNDS, .35), rtol=1e-14)
    assert 0 < radial['front_mass'] < 1e-10
    for z in [-3., 0.]:
        zero = reference('point', [0., 0., z], np.empty((3, 0)))
        assert zero['front_mass'] == 0. and not zero['raw_mass'].any()


def test_finite_fullrank_point_limit_is_not_mislabelled_exact_reference():
    with pytest.raises(ValueError, match='not an exact point'):
        reference('point', [0., 0., 1.], np.eye(3) * 1e-5)


def test_constant_depth_reference_matches_diagonal_analytic_gaussian():
    mu = [.5, -.25, 1.5]; factor = [[.75, 0.], [0., .6], [0., 0.]]
    result = reference('constant_depth', mu, factor)
    sd = np.sqrt(np.array([.75, .6]) ** 2 / 1.5 ** 2 + .35 ** 2)
    expected = R.normal_interval((TOY_BOUNDS[:, :2] - np.array(mu[:2]) / 1.5) / sd,
                                 (TOY_BOUNDS[:, 2:] - np.array(mu[:2]) / 1.5) / sd).prod(1)
    np.testing.assert_allclose(result['raw_mass'], expected, atol=2e-16, rtol=0)


def test_latent_sphere_triangle_matches_isotropic_closed_solid_angle():
    bounds = np.array([[-.7, -.4, .2, .6], [-3., -.8, 5., 2.]])
    np.testing.assert_allclose(R._latent_solid_angle_rectangle(bounds, np.eye(3)), R._solid_angle_rectangle(bounds), atol=2e-16, rtol=0)
    large = R._latent_solid_angle_rectangle(np.array([[-1e6, -1e6, 1e6, 1e6]]), np.eye(3))[0]
    assert .499 < large < .5  # Raw front mass is 1/2, not 1.


def test_correlated_solid_angle_matches_independent_ray_density_integral():
    factor = np.array([[.8, .1, 0.], [.2, 1.2, .1], [.1, .3, .9]])
    bounds = np.array([[-.4, -.8, .7, .3]])
    nodes, weights = leggauss(96)
    x = (bounds[0, 0] + bounds[0, 2]) / 2 + (bounds[0, 2] - bounds[0, 0]) / 2 * nodes
    y = (bounds[0, 1] + bounds[0, 3]) / 2 + (bounds[0, 3] - bounds[0, 1]) / 2 * nodes
    xx, yy = np.meshgrid(x, y, indexing='ij'); rays = np.stack((xx, yy, np.ones_like(xx)), -1)
    precision = np.linalg.inv(factor @ factor.T)
    quadratic = np.einsum('...i,ij,...j->...', rays, precision, rays)
    density = 1 / (4 * np.pi * abs(np.linalg.det(factor)) * quadratic ** 1.5)
    expected = np.sum(density * np.outer(weights, weights)) * np.prod(bounds[0, 2:] - bounds[0, :2]) / 4
    np.testing.assert_allclose(R._latent_solid_angle_rectangle(bounds, factor)[0], expected, atol=3e-16, rtol=0)
    np.testing.assert_allclose(R._latent_solid_angle_rectangle(bounds, -factor), expected, atol=3e-16, rtol=0)


@pytest.mark.parametrize('kind,factor', [('origin_isotropic', np.eye(3) * 2.),
    ('origin_correlated', np.array([[.8, .1, 0.], [.2, 1.2, .1], [.1, .3, .9]])),
    ('origin_plane', np.array([[1., 0.], [0., 0.], [0., 1.]]))])
def test_blurred_origin_reference_orders_on_distinct_toy_cells(kind, factor):
    low = reference(kind, [0., 0., 0.], factor, 64)
    high = reference(kind, [0., 0., 0.], factor, 96)
    assert low['front_mass'] == high['front_mass'] == .5
    assert np.max(abs(low['raw_mass'] - high['raw_mass'])) < 1e-12
    assert low['analytic_tail']['raw_fov_upper'] == 0.
    assert low['error_estimates']['rigorous_total_error_bound'] is False


def test_offset_projected_coordinate_reference_matches_distinct_latent_toy():
    beta, alpha, mean, sd = 5., -.8, 1.5, .7
    mu = [beta * mean + alpha, 0., mean]; factor = [[beta * sd], [0.], [sd]]
    actual = reference('offset_rank1', mu, factor)['raw_mass']
    expected = []
    for box in TOY_BOUNDS:
        def latent(t):
            z = mean + sd * t
            if z <= 0: return 0.
            point = np.array([beta + alpha / z, 0.])
            return float(R.point_rectangle(point, box[None], .35)[0]) * np.exp(-t * t / 2) / np.sqrt(2 * np.pi)
        expected.append(quad(latent, -mean / sd, 10., epsabs=1e-13, epsrel=1e-11)[0])
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


def test_tilted_projected_coordinate_reference_matches_distinct_plane_toy():
    factor = [[1., 0.], [0., .5], [.5, 0.]]
    actual = reference('tilted_plane', [0., 0., 1.], factor)['raw_mass']; expected = []
    for box in TOY_BOUNDS:
        def latent(s):
            z = 1 + .5 * s
            if z <= 0: return 0.
            wx = float(R.normal_interval((box[0] - s / z) / .35, (box[2] - s / z) / .35))
            sy = math.sqrt((.5 / z) ** 2 + .35 ** 2)
            wy = float(R.normal_interval(box[1] / sy, box[3] / sy))
            return wx * wy * np.exp(-s * s / 2) / np.sqrt(2 * np.pi)
        expected.append(quad(latent, -2., 10., epsabs=1e-13, epsrel=1e-11)[0])
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


def test_reference_budget_and_input_rejection_are_explicit():
    with pytest.raises(ValueError, match='panel budget'):
        R._panels(TOY_BOUNDS, .35, 14., .5, 5., -1, 1)
    with pytest.raises(ValueError, match='float64'):
        R.integrate(np.zeros(3, np.float32), np.empty((3, 0)), TOY_BOUNDS, np.eye(2), np.zeros(2), C.POLICY, 'point', 64, C.REFERENCE_POLICY)
    with pytest.raises(ValueError, match='kind/order'): reference('point', [0., 0., 1.], np.empty((3, 0)), 128)
    bad = TOY_BOUNDS.copy(); bad[1, 0] = -.1
    with pytest.raises(ValueError, match='disjoint'): reference('point', [0., 0., 1.], np.empty((3, 0)), bounds=bad)


def test_raw_tail_relative_checks_cannot_be_hidden_by_floor():
    reference_mass = np.array([1e-35, 1e-35]); actual = 2 * reference_mass
    result = A.compare_raw(actual, reference_mass, TOY_BOUNDS, 1e-30, C.GATES, 'production')
    assert result['maximum_probability_error'] == 0. and not result['passed']
    assert result['raw_fov_relative'] == result['raw_l1_over_reference_fov'] == 1.
    assert result['reference_raw_below_floor']
    zero = A.compare_raw(np.array([0., 1e-320]), np.zeros(2), TOY_BOUNDS, 1e-30, C.GATES, 'production')
    assert not zero['passed'] and not zero['exact_zero_actual']


def test_probability_gate_uses_archived_probability_exactly():
    raw = np.array([.50009999999975, .49990000000025]); ref = np.array([.5, .5])
    gates = dict(C.GATES, production_raw_fov_relative=1., production_raw_l1_over_reference_fov=1.)
    archived = np.array([.50010000000025, .49989999999975])
    assert A.compare_raw(raw, ref, TOY_BOUNDS, 1e-30, gates, 'production')['passed']
    result = A.compare_raw(raw, ref, TOY_BOUNDS, 1e-30, gates, 'production', actual_probability=archived)
    assert not result['passed'] and result['maximum_probability_error'] > gates['production_probability_absolute']


def production_metadata():
    raw = np.array([1e-35, 2e-35]); p, floor, floored = A.probabilities(raw, TOY_BOUNDS, C.POLICY['floor_density'])
    bank = dict(raw_mass=raw, probability=p, floored_mass=floored, quadrature_error_estimate=np.zeros(2),
                inner_error_estimate_mass=np.zeros(2), input__mu=np.array([.1, -.2, 1.]),
                input__factor=np.empty((3, 0)), input__bounds=TOY_BOUNDS)
    row = dict(passed=True, status='complete', raw_fov_mass=float(raw.sum()), raw_mass_sum=float(raw.sum()),
        total_area_floor=float(floor.sum()), normalizer=float(floored.sum()), depth_tail_bound=0., certified_tail_amplification=0.,
        quadrature_error_is_estimate=True, tail_bound_excludes_quadrature_and_roundoff=True,
        positive_mass=1., log_positive_mass=0., exact_zero_positive_mass=False, positive_mass_underflow=False,
        outer_error_estimate=0., fov_quadrature_error_estimate=0., fov_inner_error_estimate=0.,
        zero_raw_cells=0, area_prior_used=False, raw_mass_below_total_floor=True,
        argmax_index=int(p.argmax()), argmax_exact_ties=int(np.count_nonzero(p == p.max())),
        argmax_near_ties_at_1e_12=int(np.count_nonzero(p.max() - p <= 1e-12)),
        work=dict(depth_evaluations=0, inner_evaluations=0, outer_intervals=0), depth_breakpoints=[])
    return bank, row


@pytest.mark.parametrize('kind', ['valid', 'tiny_metadata', 'front', 'log_front', 'outer_budget', 'direct_fov'])
def test_independent_metadata_checks_respect_tiny_mass_and_budget(kind):
    bank, row = production_metadata()
    if kind == 'tiny_metadata': row['raw_mass_sum'] *= 2
    elif kind == 'front': row['positive_mass'] = .9
    elif kind == 'log_front': row['log_positive_mass'] = .01
    elif kind == 'outer_budget': row['work']['outer_intervals'] = C.POLICY['outer_limit'] + 1
    elif kind == 'direct_fov': row['raw_fov_mass'] += 2e-8
    result = A.production_check({}, row, bank, C.GATES, C.POLICY)
    assert result['passed'] is (kind == 'valid')


def test_identity_permutation_merge_and_fixed_limit_are_saved_array_checks():
    parent = dict(raw_mass=np.array([.1, .2, .3, .4]), probability=np.array([.1, .2, .3, .4]))
    permutation = dict(id='toy_p', identity=dict(parent='toy', relation='permutation', permutation=[3, 2, 1, 0]))
    actual = {n: v[::-1] for n, v in parent.items()}
    assert A.identity_check(permutation, actual, parent, C.GATES)['passed']
    merge = dict(id='toy_m', identity=dict(parent='toy', relation='merge', groups=[[0], [1, 2, 3]]))
    actual = {n: np.array([v[0], sum(v[1:])]) for n, v in parent.items()}
    assert A.identity_check(merge, actual, parent, C.GATES)['passed']
    limit = dict(id='toy_l', identity=dict(parent='toy', relation='point_limit'))
    result = A.identity_check(limit, parent, parent, C.GATES)
    assert result['passed'] and result['finite_point_limit_is_independent_reference'] is False


def test_joint_freeze_rejects_existing_forward_without_array_access(tmp_path, monkeypatch):
    run = tmp_path / 'producer'; run.mkdir(); (run / 'started.json').write_text('{}')
    monkeypatch.setattr(A.np, 'load', lambda *args, **kwargs: pytest.fail('array opened'))
    with pytest.raises(ValueError, match='precede every formal case'): A.freeze(run, tmp_path / 'audit')


def test_audit_failure_retains_attempt_and_does_not_open_arrays(tmp_path, monkeypatch):
    out = tmp_path / 'audit'; out.mkdir()
    def reject(*args): raise ValueError('seal rejected')
    monkeypatch.setattr(A, 'verify_seal', reject)
    monkeypatch.setattr(A.np, 'load', lambda *args, **kwargs: pytest.fail('array opened'))
    with pytest.raises(ValueError, match='seal rejected'): A.audit(tmp_path / 'producer', out)
    original = (out / 'failure.json').read_bytes()
    assert A.verify_inventory(out)['delivery_integrity_passed'] is False
    with pytest.raises(ValueError, match='preserve every previous'): A.audit(tmp_path / 'producer', out)
    assert (out / 'failure.json').read_bytes() == original


def test_metadata_helpers_match_producer_without_running_any_fixture():
    from experiments import round22_ray_preflight as P
    assert A.source_names() == P.source_names()
    assert A.runtime() == P.runtime()
    assert A.array_sha(np.array([1., 2.])) == P.array_sha(np.array([1., 2.]))
    assert len(C.fixtures()) == 15 and len(C.identity_fixtures()) == 8
