"""Small independent development toys; never forward the formal R24 fixtures."""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
import pytest
from scipy.integrate import dblquad
from scipy.stats import multivariate_t

from experiments import round24_student_contract as C
from experiments import round24_student_preflight as RUN
from models import round24_student_t_pe as M


MU = np.asarray([.17, -.29])
COV = np.asarray([[.2, .04], [.04, .3]])
BOXES = np.asarray([[-.75, -.5, .25, .6], [.25, -.5, .9, .6]])


def run(mu=MU, covariance=COV, bounds=BOXES, content_bias=None, **kwargs):
    return M.integrate(mu, covariance, bounds, content_bias, policy=deepcopy(C.POLICY), **kwargs)


def test_normalized_covariance_law_matches_independent_scipy_density_integration():
    result = run()
    assert result['passed'], result
    reference = []
    distribution = multivariate_t(loc=MU, shape=(COV+.25**2*np.eye(2))/2, df=4)
    for left, low, right, high in BOXES:
        value, _ = dblquad(lambda y, x: distribution.pdf([x, y]), left, right,
                          lambda _:low, lambda _:high, epsabs=1e-12, epsrel=1e-12)
        reference.append(value)
    np.testing.assert_allclose(result['raw_mass'], reference, atol=2e-12, rtol=0)
    assert abs(sum(reference)-result['raw_fov_mass']) < 2e-12


def test_floor_precedes_content_and_is_not_added_to_normalized_raw():
    policy = deepcopy(C.POLICY)
    policy['floor_density'] = .2  # Distinct toy chosen to expose the order.
    bias = np.asarray([-.3, .4])
    result = M.integrate(MU, COV, BOXES, bias, policy=policy)
    assert result['passed']
    area = np.prod(BOXES[:, 2:]-BOXES[:, :2], axis=1)
    target = (result['raw_mass']+.2*area)*np.exp(bias)
    target /= target.sum()
    np.testing.assert_allclose(result['probability'], target, atol=2e-16, rtol=0)
    wrong = (result['raw_mass']/result['raw_mass'].sum()+.2*area)*np.exp(bias)
    wrong /= wrong.sum()
    assert abs(wrong-target).max() > 1e-3


def test_point_covariance_is_student_point_and_native_mass_is_additive():
    covariance = np.zeros((2, 2))
    separate = run(covariance=covariance)
    merged = run(covariance=covariance, bounds=np.asarray([[-.75, -.5, .9, .6]]))
    assert separate['passed'] and merged['passed']
    assert abs(separate['raw_mass'].sum()-merged['raw_mass'][0]) < 1e-10
    assert separate['work']['cell_node_evaluations'] == 576*(separate['native_tile_count']+separate['fov_tile_count'])
    assert not separate['tail_truncation_used'] and not separate['quadrature_error_bound_available']


def test_units_translation_and_key_permutation_preserve_probabilities():
    baseline = run()
    shift = np.asarray([.7, -.3])
    translated = run(mu=MU+shift, bounds=BOXES+np.tile(shift, 2))
    permuted = run(bounds=BOXES[::-1].copy())
    policy = deepcopy(C.POLICY)
    unit_scale = 2.3
    policy.update(h=.25*unit_scale, floor_density=1e-30/unit_scale**2)
    scaled = M.integrate(MU*unit_scale, COV*unit_scale**2, BOXES*unit_scale, policy=policy)
    for value in (translated, scaled):
        assert value['passed']
        np.testing.assert_allclose(value['raw_mass'], baseline['raw_mass'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(value['probability'], baseline['probability'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(permuted['probability'][::-1], baseline['probability'], atol=1e-15, rtol=0)


def test_far_tiny_raw_is_retained_before_floor():
    result = run(mu=np.asarray([2e8, -1e8]))
    assert result['passed'] and result['raw_mass_below_total_floor']
    assert (result['raw_mass'] > 0).all() and not result['all_raw_mass_zero']
    assert result['zero_raw_cells'] == 0


def test_nominal_psd_residual_is_recorded_without_changing_covariance():
    covariance = np.diag([-5e-13, .2])
    prepared = M.prepare(MU, covariance, BOXES, None, C.POLICY)
    np.testing.assert_array_equal(prepared['covariance'], covariance)
    np.testing.assert_array_equal(prepared['total_covariance'], covariance+.25**2*np.eye(2))
    assert prepared['admitted_covariance_min_eigenvalue'] == -5e-13


@pytest.mark.parametrize('kwargs', [dict(mu=MU.astype(np.float32)), dict(mu=np.asarray([np.nan, 0.])),
    dict(covariance=np.asarray([[.2, .04+1e-15], [.04, .3]])),
    dict(covariance=np.diag([-.01, .2])), dict(covariance=np.diag([-1., 1e14])),
    dict(bounds=np.asarray([[0., 0., 1., 1.], [.5, 0., 2., 1.]])),
    dict(bounds=np.asarray([[0., 0., 1., 1.], [2., 0., 3., 1.]])),
    dict(content_bias=np.asarray([0., np.inf]))])
def test_invalid_inputs_have_no_silent_repair(kwargs):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        run(**kwargs)


def test_partial_work_budget_failure_does_not_return_a_probability():
    boxes = np.asarray([[float(x), 0., x+1., 1.] for x in range(40)])
    policy = deepcopy(C.POLICY)
    # The native mass fits; the independent FOV pass must still be charged.
    policy['max_cell_node_evaluations'] = 40*576
    value = M.integrate(MU, COV, boxes, policy=policy)
    assert not value['passed'] and 'budget' in value['failure_reason']
    assert value['work']['native_tiles_completed'] == 40 and value['work']['fov_tiles_completed'] == 0
    assert value['partial_raw_mass'].sum() > 0 and 'probability' not in value
    assert value['work']['cell_node_evaluations'] == 40*576


def test_expired_deadline_keeps_explicit_failure():
    result = run(deadline=time.perf_counter()-1.)
    assert not result['passed'] and 'wall budget' in result['failure_reason']
    assert result['work']['cell_node_evaluations'] == 0
    assert 'probability' not in result


def toy_run(tmp_path, monkeypatch):
    out = tmp_path/'run'
    out.mkdir()
    RUN.write(out/'lock.json', {})
    rows = [dict(id='unit_case_'+str(i), kind='base', mu=MU.tolist(), covariance=COV.tolist(),
                 bounds=BOXES.tolist(), content_bias=[0., 0.], content_temperature=0., policy_overrides={})
            for i in range(2)]
    RUN.write(out/'fixtures.json', rows)
    lock = dict(policy=C.POLICY, gates=dict(production_max_phase_wall_seconds=10.),
        fixtures_sha256=RUN.sha(out/'fixtures.json'), base_ids=[r['id'] for r in rows], identity_ids=[])
    monkeypatch.setattr(RUN, 'verify', lambda _:lock)
    monkeypatch.setattr(RUN, 'require_audit_freeze', lambda *_:'unit_audit')
    return out, rows


def test_runner_retains_prepare_exceptions_and_complete_population(tmp_path, monkeypatch):
    out, rows = toy_run(tmp_path, monkeypatch)
    def failure(**_):
        raise ValueError('injected preparation exception')
    monkeypatch.setattr(RUN.MODEL, 'integrate', failure)
    RUN.run(out)
    records = RUN.read(out/'records.json')
    assert [r['id'] for r in records] == [r['id'] for r in rows]
    assert all(not r['passed'] and 'injected preparation' in r['failure_reason'] for r in records)
    assert all(set(r['array_sha256']) == {'input__mu', 'input__covariance', 'input__bounds', 'input__content_bias'} for r in records)
    seal, completion = RUN.read(out/'forward_complete.json'), RUN.read(out/'completion.json')
    assert not seal['all_production_calls_completed'] and not seal['scientific_gate_passed']
    assert not seal['real_inputs_read'] and not seal['goal_complete']
    RUN.verify_inventory(out, completion['files'], exclude=('completion.json',))
    with pytest.raises(ValueError, match='overwrite'):
        RUN.run(out)


def test_runner_seals_post_start_engineering_failure(tmp_path, monkeypatch):
    out, _ = toy_run(tmp_path, monkeypatch)
    def failure(_out):
        RUN.write(_out/'started.json', {'unit':True})
        raise OSError('injected stage exception')
    monkeypatch.setattr(RUN, '_run_started', failure)
    with pytest.raises(OSError, match='injected'):
        RUN.run(out)
    completion = RUN.read(out/'completion.json')
    assert not completion['complete_case_inventory'] and not completion['all_production_calls_completed']
    RUN.verify_inventory(out, completion['files'], exclude=('completion.json',))


def test_pre_freeze_metadata_code_does_not_execute_formal_fixture(tmp_path, monkeypatch):
    out = tmp_path/'freeze'
    rows = [dict(id='unit_base', kind='base'), dict(id='unit_identity', kind='identity')]
    monkeypatch.setattr(RUN.C, 'fixtures', lambda:rows)
    monkeypatch.setattr(RUN, 'source_names', lambda:[])
    monkeypatch.setattr(RUN, 'inherited_bindings', lambda:{})
    monkeypatch.setattr(RUN, 'check_runtime', lambda:{'toy':True})
    monkeypatch.setattr(RUN, 'verify', lambda _:None)
    monkeypatch.setattr(RUN.MODEL, 'integrate', lambda **_:pytest.fail('freeze performed numerical forward'))
    monkeypatch.setattr(RUN.np, 'load', lambda *_a, **_k:pytest.fail('freeze decoded array'))
    RUN.freeze(out, tmp_path/'audit')
    assert RUN.read(out/'fixtures.json') == rows
    assert RUN.read(out/'lock.json')['base_ids'] == ['unit_base']
