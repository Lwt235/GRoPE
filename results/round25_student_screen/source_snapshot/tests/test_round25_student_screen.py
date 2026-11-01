"""Synthetic I/O and arithmetic contracts; never load real caches or labels."""
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from experiments import round25_student_screen as P


def cache():
    rng = np.random.default_rng(2517)
    covariance = np.repeat(np.asarray([[[.7, .11], [.11, .4]]]), P.N_QUERY, axis=0)
    diagonal = torch.from_numpy(covariance).diagonal(dim1=-2, dim2=-1)
    scale = torch.maximum(diagonal.amax(-1), torch.tensor(.25**2))
    h = .25**2/scale
    tau = (10.*(h/(h+(diagonal/scale[:, None]).sum(-1)))).float().numpy()
    return {'g24__a':rng.normal(size=(P.N_QUERY, 2)),
        'g24__anchor_cov':covariance, 'g24__tau32':tau,
        'g24__q_features':rng.normal(size=(P.N_QUERY, P.CHANNELS)).astype(np.float32),
        'g8__k_features':rng.normal(size=(P.N_KEY, P.CHANNELS)).astype(np.float32),
        'g8__bounds_px':np.asarray([[x, y, x+8, y+8]
             for y in range(0, 144, 8) for x in range(0, 192, 8)], np.int64)}


def entry(path='never_open_original_inputs.npz'):
    return dict(scene='synthetic/example', pair=[2, 7], pair_name='toy_pair',
        cache_path=path, cache_sha256='cache_digest',
        cache_array_sha256={f:'digest' for f in P.C.FIELDS},
        input_path='DO_NOT_OPEN_ORIGINAL.npz', label_path='DO_NOT_OPEN_LABEL.npz')


def test_upper_copy_preserves_archive_and_full_matrix_arithmetic():
    values = cache()
    values['g24__anchor_cov'][:, 1, 0] += 2e-13
    original = values['g24__anchor_cov'].copy()
    result = P.prepare_inputs(values)
    assert set(result) == set(P.INPUT_FIELDS)
    np.testing.assert_array_equal(result['covariance_archive64'], original)
    np.testing.assert_array_equal(result['covariance64'][:, 1, 0], original[:, 0, 1])
    assert np.all(result['covariance_change_max_absolute'] > 0)
    np.testing.assert_array_equal(result['content32'],
        (torch.from_numpy(values['g24__q_features']) @ torch.from_numpy(values['g8__k_features']).T).numpy())
    np.testing.assert_array_equal(result['bias32'], values['g24__tau32'][:, None]*result['content32'])
    np.testing.assert_array_equal(result['content_bias64'], result['bias32'].astype(np.float64))
    np.testing.assert_array_equal(result['a64'], values['g24__a'])
    np.testing.assert_array_equal(result['bounds64'], (values['g8__bounds_px']-.5)/8)


@pytest.mark.parametrize('defect', ['asymmetry', 'tau', 'dtype', 'bounds_order', 'extra_field', 'nonfinite'])
def test_cache_contract_rejects_changed_inputs(defect):
    values = cache()
    if defect == 'asymmetry': values['g24__anchor_cov'][0, 1, 0] += 1e-6
    elif defect == 'tau': values['g24__tau32'][0] += np.float32(.01)
    elif defect == 'dtype': values['g24__a'] = values['g24__a'].astype(np.float32)
    elif defect == 'bounds_order': values['g8__bounds_px'] = values['g8__bounds_px'][::-1].copy()
    elif defect == 'extra_field': values['gt_xy'] = np.zeros((1, 2))
    elif defect == 'nonfinite': values['g24__q_features'][0, 0] = np.inf
    with pytest.raises(ValueError): P.prepare_inputs(values)


def test_cache_loader_decodes_only_six_fields(monkeypatch):
    values, accessed = cache(), []
    e = entry()
    e['cache_array_sha256'] = {k:P.C.array_sha(v) for k, v in values.items()}
    class Archive:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def __getitem__(self, name):
            accessed.append(name)
            assert name in P.C.FIELDS
            return values[name]
    def load(path, *, allow_pickle):
        assert str(path) == e['cache_path'] and allow_pickle is False
        return Archive()
    monkeypatch.setattr(P.C, 'sha', lambda path:'cache_digest')
    monkeypatch.setattr(P.np, 'load', load)
    P.load_cache(e)
    assert accessed == list(P.C.FIELDS)


def test_cache_hash_failure_precedes_np_load(monkeypatch):
    monkeypatch.setattr(P.C, 'sha', lambda _: 'changed')
    monkeypatch.setattr(P.np, 'load', lambda *_a, **_k:pytest.fail('unverified archive decoded'))
    with pytest.raises(ValueError, match='archive hash'): P.load_cache(entry())


def test_freeze_delegates_metadata_only(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(P.C, 'freeze', lambda *a, **k:calls.append((a, k)))
    monkeypatch.setattr(P.np, 'load', lambda *_a, **_k:pytest.fail('freeze decoded cache'))
    monkeypatch.setattr(P.MODEL, 'integrate', lambda *_a, **_k:pytest.fail('freeze ran model'))
    P.freeze(tmp_path/'producer', tmp_path/'audit')
    assert calls == [((tmp_path/'producer',), dict(kind='producer', audit_root=tmp_path/'audit'))]


def test_expired_pair_does_not_read_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'load_cache', lambda _:pytest.fail('cache read after deadline'))
    directory, inputs, record = P._pair_inputs(tmp_path, entry(), time.perf_counter()-1)
    assert inputs is None and record['input_path'] is None
    assert record['cache_read_attempted'] is False and record['status'] == 'failed'
    assert (directory/'inputs.json').is_file()


def test_preparation_failure_retains_available_cache(tmp_path, monkeypatch):
    values = cache()
    values['g24__tau32'][0] += 1
    monkeypatch.setattr(P, 'load_cache', lambda _:values)
    _, inputs, record = P._pair_inputs(tmp_path, entry(), time.perf_counter()+30)
    assert inputs is None and record['input_valid'] is False
    with np.load(tmp_path/record['available_cache']['archive_path']) as saved:
        assert set(saved.files) == set(P.C.FIELDS)
        np.testing.assert_array_equal(saved['g24__tau32'], values['g24__tau32'])


def query_inputs():
    # Distinct tiny synthetic law, unrelated to formal fixtures or real caches.
    return dict(a64=np.asarray([[.13, -.21]]), covariance64=np.asarray([[[.2, .02], [.02, .3]]]),
        bounds64=np.asarray([[-1., -1., 0., 1.], [0., -1., 1., 1.]]),
        content_bias64=np.asarray([[.1, -.2]]))


def test_query_success_seals_exact_model_arrays(tmp_path):
    inputs = query_inputs()
    row = P._query(tmp_path, tmp_path/'forward', entry(), inputs,
        dict(input_path='synthetic_inputs.npz', input_sha256='toy'), 0, time.perf_counter()+30)
    assert row['status'] == 'ok' and row['attempted'] is True
    with np.load(tmp_path/row['archive_path']) as result:
        assert set(result.files) == {'raw_mass', 'floored_mass', 'probability', 'argmax'}
        assert result['argmax'].dtype == np.int32 and result['argmax'].shape == ()
        assert result['argmax'] == np.argmax(result['probability'])
        assert abs(result['probability'].sum()-1) <= 2e-12


def test_model_validation_exception_retains_failure_without_probability(tmp_path):
    inputs = query_inputs(); inputs['covariance64'][0, 1, 0] += .1
    row = P._query(tmp_path, tmp_path/'forward', entry(), inputs,
        dict(input_path='synthetic_inputs.npz', input_sha256='toy'), 0, time.perf_counter()+30)
    assert row['status'] == 'failed' and row['attempted']
    assert 'explicitly symmetric' in row['failure_reason']
    with np.load(tmp_path/row['archive_path']) as result: assert not result.files


def test_model_partial_failure_retained(tmp_path, monkeypatch):
    partial = np.asarray([.12, .03])
    monkeypatch.setattr(P.MODEL, 'integrate', lambda *_a, **_k:dict(status='failed', passed=False,
        work=dict(cell_node_evaluations=100), failure_reason='synthetic work failure',
        partial_raw_mass=partial, partial_raw_fov_mass=np.asarray([.15])))
    row = P._query(tmp_path, tmp_path/'forward', entry(), query_inputs(),
        dict(input_path='synthetic_inputs.npz', input_sha256='toy'), 0, time.perf_counter()+30)
    with np.load(tmp_path/row['archive_path']) as result:
        assert set(result.files) == {'partial_raw_mass', 'partial_raw_fov_mass'}
        np.testing.assert_array_equal(result['partial_raw_mass'], partial)
    assert row['work']['cell_node_evaluations'] == 100


def test_expired_query_not_attempted(tmp_path, monkeypatch):
    monkeypatch.setattr(P.MODEL, 'integrate', lambda *_a, **_k:pytest.fail('postdeadline model call'))
    row = P._query(tmp_path, tmp_path/'forward', entry(), query_inputs(),
        dict(input_path='synthetic_inputs.npz', input_sha256='toy'), 0, time.perf_counter()-1)
    assert not row['attempted'] and row['status'] == 'failed'
    with np.load(tmp_path/row['archive_path']) as result: assert not result.files


def fake_forward_lock(tmp_path, monkeypatch, seconds):
    lock = dict(kind='producer', created_unix=1., audit_root=str(tmp_path/'audit'),
        gates=dict(production_max_phase_wall_seconds=seconds), entries=[entry()])
    P.C.write(tmp_path/'lock.json', lock)
    monkeypatch.setattr(P.C, 'verify', lambda _:lock)
    monkeypatch.setattr(P, 'require_audit_freeze', lambda *_:'auditdigest')
    monkeypatch.setattr(P, 'N_QUERY', 2)
    return lock


def test_full_failed_population_and_inventory_are_sealed(tmp_path, monkeypatch):
    fake_forward_lock(tmp_path, monkeypatch, -1.)
    monkeypatch.setattr(P, 'load_cache', lambda _:pytest.fail('expired forward read cache'))
    seal = P.forward(tmp_path)
    rows = P.C.read(tmp_path/'forward/records.json')
    assert len(rows) == 2 and [r['query_index'] for r in rows] == [0, 1]
    assert seal['failed_calls'] == 2 and seal['attempted_calls'] == 0
    assert seal['complete_case_inventory'] and not seal['phase_wall_passed']
    P.C.verify_inventory(tmp_path, P.C.read(tmp_path/'completion.json')['files'], exclude=('completion.json',))
    P.C.verify_inventory(tmp_path/'forward', P.C.read(tmp_path/'forward/completion.json')['files'], exclude=('completion.json',))
    with pytest.raises(ValueError, match='overwrite'): P.forward(tmp_path)


def test_outer_output_failure_archives_partial_inventory(tmp_path, monkeypatch):
    fake_forward_lock(tmp_path, monkeypatch, 10.)
    monkeypatch.setattr(P, '_pair_inputs', lambda *_:(_ for _ in ()).throw(OSError('synthetic write failure')))
    with pytest.raises(OSError, match='synthetic write'): P.forward(tmp_path)
    completion = P.C.read(tmp_path/'completion.json')
    assert not completion['complete_case_inventory']
    assert P.C.read(tmp_path/'failure.json')['error_type'] == 'OSError'
    P.C.verify_inventory(tmp_path, completion['files'], exclude=('completion.json',))


def test_joint_freeze_rejects_source_mismatch_before_arrays(tmp_path, monkeypatch):
    lock = dict(kind='producer', audit_root=str(tmp_path/'audit'), created_unix=1.,
        source_hashes={'p':'one'}, runtime={}, entries=[], bindings={})
    audit = dict(lock, kind='audit', producer_root=str(tmp_path), created_unix=2., source_hashes={'p':'two'})
    monkeypatch.setattr(P.C, 'verify', lambda _:audit)
    monkeypatch.setattr(P.np, 'load', lambda *_a, **_k:pytest.fail('unmatched source decoded arrays'))
    with pytest.raises(ValueError, match='joint freeze'): P.require_audit_freeze(tmp_path, lock)
