"""Development toys only: never execute the formal R22 inputs here."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import pytest
from scipy.special import ndtr

from experiments import round22_ray_contract as C
from experiments import round22_ray_preflight as RUN
from models import round22_ray_pushforward as M


BOXES = np.asarray([[-1., -1., 0., 1.], [0., -1., 2., 1.]])
TOY_MU = np.asarray([.25, -.4, 2.])
EMPTY = np.empty((3, 0), np.float64)


def run(mu=TOY_MU, factor=EMPTY, bounds=BOXES, **kwargs):
    return M.integrate(mu, factor, bounds, policy=deepcopy(C.POLICY), **kwargs)


def point_mass(center, bounds=BOXES, h=.25):
    return (ndtr((bounds[:, 2:]-center)/h)-ndtr((bounds[:, :2]-center)/h)).prod(axis=1)


def test_point_raw_before_floor_and_fov_partition():
    value = run()
    assert value['passed']
    expected = point_mass(TOY_MU[:2]/TOY_MU[2])
    np.testing.assert_allclose(value['raw_mass'], expected, rtol=1e-13, atol=1e-15)
    assert abs(value['raw_fov_mass']-sum(expected)) < 1e-14
    assert value['positive_mass'] == 1
    assert value['depth_tail_bound'] == 0
    assert abs(value['probability'].sum()-1) < 2e-15


@pytest.mark.parametrize('z', [-3., 0.])
def test_nonpositive_constant_depth_keeps_area_prior(z):
    value = run(mu=np.asarray([.7, -.1, z]))
    assert value['passed'] and value['exact_zero_positive_mass']
    np.testing.assert_array_equal(value['raw_mass'], [0., 0.])
    np.testing.assert_allclose(value['probability'], [1/3, 2/3], rtol=0, atol=1e-16)


def test_correlated_plane_has_known_quadrant_mass():
    bounds = np.asarray([[-10., -10., 0., 0.], [0., -10., 10., 0.],
                         [-10., 0., 0., 10.], [0., 0., 10., 10.]])
    factor = np.asarray([[1., .2], [.3, .8], [0., 0.]])
    value = run(mu=np.asarray([0., 0., 1.5]), factor=factor, bounds=bounds)
    covariance = factor[:2] @ factor[:2].T / 1.5**2 + .25**2*np.eye(2)
    rho = covariance[0, 1]/math.sqrt(covariance[0, 0]*covariance[1, 1])
    same = .25+math.asin(rho)/(2*math.pi)
    assert value['passed'], value
    np.testing.assert_allclose(value['raw_mass'], [same, .5-same, .5-same, same], rtol=0, atol=1e-11)
    assert value['work']['inner_evaluations'] > 0


def test_factor_rotation_preserves_correlated_plane():
    factor = np.asarray([[.7, .2], [-.1, .4], [0., 0.]])
    rotation = np.asarray([[0., 1.], [-1., 0.]])
    a, b = run(factor=factor), run(factor=factor @ rotation)
    assert a['passed'] and b['passed']
    np.testing.assert_allclose(a['raw_mass'], b['raw_mass'], rtol=0, atol=1e-12)


def test_tiny_radial_mass_is_not_conditioned_before_floor():
    direction = np.asarray([-.4, .7, 1.])
    m, sd = -9., .75
    value = run(mu=direction*m, factor=(direction*sd)[:, None])
    front = float(ndtr(m/sd))
    expected = front*point_mass(direction[:2])
    assert value['passed'], value
    np.testing.assert_allclose(value['raw_mass'], expected, rtol=1e-8, atol=0)
    area = np.prod(BOXES[:, 2:]-BOXES[:, :2], axis=1)
    expected_p = (expected+area*1e-30)/sum(expected+area*1e-30)
    np.testing.assert_allclose(value['probability'], expected_p, rtol=0, atol=1e-12)
    assert value['positive_mass'] < value['total_area_floor']
    assert value['certified_tail_amplification'] == 2*value['depth_tail_bound']/value['total_area_floor']


def test_geometry_breakpoints_follow_narrow_mean_trajectory():
    # This is a distinct toy, not either formal beta32/beta512 case.
    factor = np.asarray([[7.], [0.], [1.]])
    mu = np.asarray([7*(2.-.9)+.1*.9, 0., 2.])
    prepared = M.prepare(mu, factor, BOXES, None, None, C.POLICY)
    lower, upper = max(-14., prepared['a']), 14.
    points = M.geometry_breakpoints(prepared, C.POLICY, lower, upper)
    z = prepared['slope'][0]/(0.-prepared['base'][0])
    expected = z/prepared['depth_sd']+prepared['a']
    assert expected in points
    assert points == sorted(set(points))
    assert all(lower < p < upper for p in points)


def test_world_and_screen_scale_are_distinct_invariances():
    factor = np.asarray([[.3], [-.2], [.6]])
    mu = np.asarray([.1, .5, 1.8])
    base = run(mu=mu, factor=factor)
    world = run(mu=mu*2.5, factor=factor*2.5)
    policy = deepcopy(C.POLICY)
    policy.update(h=.5, floor_density=1e-30/4)
    screen = M.integrate(mu, factor, BOXES*2, H=np.eye(2)*2, policy=policy)
    assert base['passed'] and world['passed'] and screen['passed']
    for result in (world, screen):
        np.testing.assert_allclose(base['raw_mass'], result['raw_mass'], rtol=0, atol=1e-11)
        np.testing.assert_allclose(base['probability'], result['probability'], rtol=0, atol=1e-11)


def test_nonzero_depth_point_limit_uses_varying_depth_path():
    value = run(factor=np.eye(3)*1e-7)
    assert value['passed'], value
    assert value['work']['depth_evaluations'] > 0
    np.testing.assert_allclose(value['raw_mass'], run()['raw_mass'], rtol=0, atol=1e-10)


def test_nested_correlated_path_with_distinct_far_front_toy():
    factor = np.asarray([[.2, 0., 0.], [.05, .3, 0.], [.07, .02, .1]])
    mu = np.asarray([.3, .1, 2.])
    value = run(mu=mu, factor=factor)
    assert value['passed'], value
    assert value['work']['depth_evaluations'] > 0 and value['work']['inner_evaluations'] > 0
    assert abs(value['raw_fov_mass']-value['raw_mass'].sum()) < 1e-10
    assert np.all(value['inner_error_estimate_mass'] >= 0)


def test_budget_failure_is_not_retried_or_given_probability():
    policy = deepcopy(C.POLICY)
    policy['max_depth_evaluations'] = 1
    value = M.integrate(TOY_MU, np.eye(3)*.1, BOXES, policy=policy)
    assert not value['passed'] and value['status'] == 'failed'
    assert value['work']['depth_evaluations'] == 1
    assert 'probability' not in value
    assert 'budget exceeded' in value['failure_reason']


def test_phase_deadline_failure_retained():
    value = M.integrate(TOY_MU, EMPTY, BOXES, policy=C.POLICY, deadline=time.perf_counter()-1)
    assert not value['passed'] and 'wall budget' in value['failure_reason']


@pytest.mark.parametrize('factor', [np.asarray([[0.], [0.], [1e-200]]),
                                    np.asarray([[0.], [0.], [1e200]])])
def test_unrepresentable_depth_norm_rejected_without_rank_repair(factor):
    with pytest.raises(ValueError):
        run(factor=factor)


@pytest.mark.parametrize('kwargs', [dict(mu=np.asarray([0., np.nan, 2.])),
                                  dict(factor=np.zeros((2, 2))),
                                  dict(factor=np.zeros((3, 1), dtype=np.float32)),
                                  dict(H=np.zeros((2, 2))),
                                  dict(bounds=np.asarray([[0., 0., 1., 1.], [.5, .5, 2., 2.]])),
                                  dict(bounds=np.asarray([[0., 0., 1., 1.], [2., 0., 3., 1.]]))])
def test_invalid_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)


def test_mirrored_normal_tail_and_empty_interval():
    a = M.normal_interval(np.asarray([-13., 12., 1., -np.inf]),
                          np.asarray([-12., 13., 1., -np.inf]))
    assert a[0] == a[1] and a[0] > 0
    np.testing.assert_array_equal(a[2:], [0., 0.])


def test_array_hash_binds_shape_and_dtype():
    a = np.asarray([1., 2.])
    assert RUN.array_sha(a) != RUN.array_sha(a.reshape(1, 2))
    assert RUN.array_sha(a) != RUN.array_sha(a.astype(np.float32))


def test_runner_seals_failure_and_refuses_overwrite(tmp_path, monkeypatch):
    out = tmp_path/'run'
    out.mkdir()
    RUN.write(out/'lock.json', {})
    fixture = dict(id='distinct_unit_toy', kind='point', mu=TOY_MU.tolist(), factor=EMPTY.tolist(),
                   bounds=BOXES.tolist(), H=np.eye(2).tolist(), offset=[0., 0.])
    RUN.write(out/'fixtures.json', [fixture])
    lock = dict(policy=C.POLICY, fixtures_sha256=RUN.sha(out/'fixtures.json'),
                base_ids=[fixture['id']], identity_ids=[])
    monkeypatch.setattr(RUN, 'verify', lambda _:lock)
    monkeypatch.setattr(RUN, 'require_audit_freeze', lambda *_:'independent_digest')
    monkeypatch.setattr(RUN.MODEL, 'integrate', lambda **_:dict(status='failed', passed=False,
                        failure_reason='injected unit failure', work={}))
    RUN.run(out)
    seal = RUN.read(out/'forward_complete.json')
    assert seal['all_case_attempts_retained'] and not seal['all_production_calls_completed']
    completion = RUN.read(out/'completion.json')
    RUN.verify_inventory(out, completion['files'], exclude=('completion.json',))
    with pytest.raises(ValueError, match='overwrite'):
        RUN.run(out)


def test_runner_archives_post_start_engineering_failure(tmp_path, monkeypatch):
    out = tmp_path/'run'
    out.mkdir()

    def fail_after_start(directory):
        RUN.write(directory/'started.json', dict(started_unix=time.time()))
        raise RuntimeError('injected post-start source/write failure')

    monkeypatch.setattr(RUN, '_run_started', fail_after_start)
    with pytest.raises(RuntimeError, match='injected'):
        RUN.run(out)
    completion = RUN.read(out/'completion.json')
    assert not completion['complete_case_inventory']
    assert not completion['all_production_calls_completed']
    assert RUN.read(out/'failure.json')['error_type'] == 'RuntimeError'
    RUN.verify_inventory(out, completion['files'], exclude=('completion.json',))


def test_formal_case_names_are_not_used_by_development_toys():
    # Specification inspection only; this test performs no formal forward.
    assert 'distinct_unit_toy' not in {r['id'] for r in C.fixtures()+C.identity_fixtures()}
