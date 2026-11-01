"""Synthetic-only R25 audit regression tests; no inherited arrays or labels."""
import hashlib
import math
import time

import numpy as np
import pytest
import torch

from experiments import round25_student_audit as A
from experiments import round25_student_contract as C


def cache():
    rng = np.random.default_rng(771)
    cov = np.broadcast_to(np.asarray([[.8, .17], [.17, .5]], np.float64), (48, 2, 2)).copy()
    cov[:, 1, 0] += 1e-14
    q = rng.normal(size=(48, 128)).astype(np.float32)
    k = rng.normal(size=(432, 128)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    return {'g24__a': rng.normal(size=(48, 2)), 'g24__anchor_cov': cov,
            'g24__tau32': A.trace_temperature(cov), 'g24__q_features': q,
            'g8__k_features': k,
            'g8__bounds_px': np.asarray([(x, y, x+8, y+8) for y in range(0, 144, 8) for x in range(0, 192, 8)], np.int64)}


def test_upper_copy_preserves_archive_and_uses_upper_not_average():
    old = np.asarray([[[1., .3], [.3+1e-14, 2.]]], np.float64)
    saved = old.copy()
    actual = A.upper_covariance(old)
    assert np.array_equal(old, saved)
    assert actual[0, 0, 1] == actual[0, 1, 0] == old[0, 0, 1]
    assert actual[0, 1, 0] != .5*(old[0, 1, 0]+old[0, 0, 1])


def test_trace_independent_formula_matches_torch_original_dtype_order():
    cov = np.asarray([[[1e200, 0.], [0., 1e199]], [[0., 0.], [0., 0.]], [[.11, .03], [.03, .7]]])
    d = torch.from_numpy(cov).diagonal(dim1=-2, dim2=-1)
    scale = torch.maximum(d.amax(-1), torch.tensor(.25**2, dtype=torch.float64))
    r = .25**2/scale
    expected = (10.*(r/(r+(d/scale[:, None]).sum(-1)))).float().numpy()
    assert np.array_equal(A.trace_temperature(cov), expected)
    assert A.trace_temperature(cov)[1] == np.float32(10.)


def test_complete_content_and_bias_match_full_torch32_gemm():
    values = cache()
    q, k, tau = [values[n] for n in ('g24__q_features', 'g8__k_features', 'g24__tau32')]
    c, b = A.content_arithmetic(q, k, tau)
    expected = torch.from_numpy(q) @ torch.from_numpy(k).T
    assert c.shape == (48, 432) and c.dtype == np.float32
    assert np.array_equal(c, expected.numpy())
    assert np.array_equal(b, (torch.from_numpy(tau)[:, None]*expected).numpy())


def test_six_cache_fields_to_exact_execution_inputs():
    values = cache()
    arrays = A.input_arrays(values, policy=C.POLICY)
    assert len(arrays) == 13
    assert arrays['a64'] is values['g24__a']
    assert arrays['covariance_archive64'] is values['g24__anchor_cov']
    assert np.array_equal(arrays['covariance64'], arrays['covariance64'].transpose(0, 2, 1))
    assert np.array_equal(arrays['bounds64'], (values['g8__bounds_px']-.5)/8)
    assert np.array_equal(arrays['content_bias64'], arrays['bias32'].astype(np.float64))
    assert np.all(arrays['covariance_change_max_absolute'] > 0)


@pytest.mark.parametrize('mutation', ['extra', 'tau', 'bounds', 'dtype', 'asymmetry'])
def test_invalid_cache_identity_or_semantics_rejected(mutation):
    values = cache()
    if mutation == 'extra': values['forbidden_camera'] = np.eye(3)
    elif mutation == 'tau': values['g24__tau32'][0] += .001
    elif mutation == 'bounds': values['g8__bounds_px'][0, 0] += 1
    elif mutation == 'dtype': values['g24__a'] = values['g24__a'].astype(np.float32)
    else: values['g24__anchor_cov'][0, 1, 0] += 1e-5
    with pytest.raises(ValueError): A.input_arrays(values, policy=C.POLICY)


def test_whitelist_does_not_decode_extra_object_array(tmp_path):
    values = cache()
    path = tmp_path/'cached.npz'
    np.savez(path, **values, forbidden=np.asarray([object()], dtype=object))
    entry = dict(cache_path=str(path), cache_sha256=C.sha(path),
                 cache_array_sha256={n:C.array_sha(v) for n,v in values.items()})
    actual = A.load_cached(entry)
    assert set(actual) == set(C.FIELDS)
    assert all(np.array_equal(actual[n], values[n]) for n in C.FIELDS)
    entry['cache_array_sha256'][C.FIELDS[0]] = '0'*64
    with pytest.raises(ValueError, match='array identity'): A.load_cached(entry)


def toy():
    bounds = np.asarray([[0., 0., .5, 1.], [.5, 0., 2., 1.]])
    bias = np.asarray([.7, -.8], np.float64)
    return bounds, bias


def test_single_area_floor_precedes_native_bias():
    bounds, bias = toy()
    raw = np.zeros(2, np.float64)
    p, floored = A.probabilities(raw, bounds, bias, 1e-30)
    expected = np.asarray([.5, 1.5])*np.exp(bias)
    assert np.allclose(p, expected/expected.sum(), rtol=1e-14, atol=0)
    assert np.allclose(floored, np.asarray([.5e-30, 1.5e-30]), rtol=1e-15, atol=0)


def test_subfloor_raw_relative_failure_not_hidden_by_probability():
    bounds, bias = toy()
    reference = np.asarray([1e-35, 3e-35])
    actual = 1.01*reference
    detail, _ = A.compare_raw(actual, reference, bounds, bias, gates=C.GATES, production=True)
    assert detail['maximum_probability_error'] < 1e-6
    assert detail['raw_fov_relative'] > .009
    assert detail['passed'] is False


def test_zero_reference_is_unresolved_not_skipped():
    bounds, bias = toy()
    with pytest.raises(ValueError, match='unresolved'):
        A.compare_raw(np.zeros(2), np.zeros(2), bounds, bias, gates=C.GATES)


def test_single_zero_reference_cell_rejected_even_with_positive_fov():
    bounds, bias = toy()
    with pytest.raises(ValueError, match='unresolved'):
        A.compare_raw(np.asarray([.1, .2]), np.asarray([0., .3]), bounds, bias, gates=C.GATES)


def test_archived_probability_is_used_at_production_gate():
    bounds, bias = toy()
    raw = np.asarray([.2, .3])
    p, _ = A.probabilities(raw, bounds, bias, 1e-30)
    changed = p+np.asarray([1.001e-4, -1.001e-4])
    result, _ = A.compare_raw(raw, raw, bounds, bias, gates=C.GATES,
                             production=True, actual_probability=changed)
    assert result['passed'] is False
    assert result['maximum_probability_error'] > 1e-4


def test_reference_work_includes_direct_fov_panels():
    bounds, _ = toy()
    assert A.reference_work(bounds, C.POLICY, C.REFERENCE_POLICY, 64) == (1+2+2)*64


def test_freeze_blocks_before_reading_actual_arrays(tmp_path, monkeypatch):
    run, out = tmp_path/'run', tmp_path/'audit'
    run.mkdir(); (run/'started.json').write_text('{}')
    monkeypatch.setattr(C, 'freeze', lambda *a, **k: pytest.fail('must not freeze after start'))
    with pytest.raises(ValueError, match='precede'):
        A.freeze(run, out)


def reference_inputs():
    bounds, bias = toy()
    return dict(a64=np.asarray([[.1, .2]]), covariance64=np.asarray([[[.7, .1], [.1, .3]]]),
                bounds64=bounds, content_bias64=bias[None])


def test_reference_failure_charges_its_attempted_batches_and_keeps_partial(monkeypatch):
    inputs = reference_inputs()
    def fail(*args, **kwargs):
        raise A.REF.ReferenceFailure(ValueError('toy failure'), 64, [.03])
    monkeypatch.setattr(A.REF, 'integrate', fail)
    record, arrays, work = A.reference_attempt(inputs, 0, 64, policy=C.POLICY,
        reference_policy=C.REFERENCE_POLICY, gates=C.GATES, phase=time.perf_counter(), cumulative=17)
    assert record['passed'] is False and record['attempted'] is True
    assert record['work'] == {'cell_node_evaluations':64} and work == 81
    assert record['work_accounting'] == 'reference_ledger'
    assert np.array_equal(arrays['partial_raw_mass'], [.03])


def test_reference_budget_exhaustion_records_failure_without_call(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('budget exhausted must not integrate')
    monkeypatch.setattr(A.REF, 'integrate', forbidden)
    limit = dict(C.REFERENCE_POLICY, max_total_cell_node_evaluations=1)
    record, arrays, work = A.reference_attempt(reference_inputs(), 0, 64, policy=C.POLICY,
        reference_policy=limit, gates=C.GATES, phase=time.perf_counter(), cumulative=0)
    assert record['attempted'] is False and record['passed'] is False and work == 0 and arrays == {}


def test_zero_returned_reference_keeps_raw_and_charged_work(monkeypatch):
    inputs = reference_inputs()
    expected = A.reference_work(inputs['bounds64'], C.POLICY, C.REFERENCE_POLICY, 64)
    def fail(*args, **kwargs):
        return dict(raw_mass=np.asarray([0., .1]), raw_fov_mass=.1,
                    work=dict(cell_node_evaluations=expected), order=64)
    monkeypatch.setattr(A.REF, 'integrate', fail)
    record, arrays, work = A.reference_attempt(inputs, 0, 64, policy=C.POLICY,
        reference_policy=C.REFERENCE_POLICY, gates=C.GATES, phase=time.perf_counter(), cumulative=0)
    assert record['passed'] is False and work == expected
    assert np.array_equal(arrays['raw_mass'], [0., .1])
    assert record['work_accounting'] == 'reference_ledger'


def test_keyboard_interrupt_seals_failed_verification_and_journal(tmp_path, monkeypatch):
    out = tmp_path/'audit'; out.mkdir()
    def interrupted(*args): raise KeyboardInterrupt('toy interrupt')
    monkeypatch.setattr(A, 'verify_seal', interrupted)
    with pytest.raises(KeyboardInterrupt): A.audit(tmp_path/'run', out)
    assert C.read(out/'forward_verification.json')['passed'] is False
    complete = C.read(out/'completion.json')
    assert complete['passed'] is False
    C.verify_inventory(out, complete['files'], exclude=('completion.json',))
