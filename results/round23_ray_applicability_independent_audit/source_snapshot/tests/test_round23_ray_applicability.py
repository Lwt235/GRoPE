"""Distinct small toys only; no real R23 archive or formal model input is read."""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np
import pytest

from experiments import round23_ray_applicability as RUN


def toy_inputs(constant=False):
    height, width = 4, 6
    yy, xx = np.mgrid[:height, :width]
    depth = np.full((height, width), 1.7) if constant else 1.2+.03*xx+.07*yy
    intrinsic = np.asarray([[5., .1, 2.], [0., 6., 1.], [0., 0., 1.]])
    inverse = np.linalg.inv(intrinsic)
    rays = np.column_stack((xx.ravel(), yy.ravel(), np.ones(height*width))) @ inverse.T
    points = rays*depth.ravel()[:, None]
    footprint = inverse[:, :2] @ inverse[:, :2].T
    covariance = depth.ravel()[:, None, None]**2*footprint/12.
    theta = .3
    rotation = np.asarray([[math.cos(theta), 0., math.sin(theta)], [0., 1., 0.],
                           [-math.sin(theta), 0., math.cos(theta)]])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, [.1, -.2, -.4]
    target_intrinsic = np.asarray([[7., .2, 2.5], [0., 8., 1.5], [0., 0., 1.]])
    return dict(q_points3=points, q_pixel_cov3=covariance, q_K=intrinsic, k_K=target_intrinsic,
                q_to_k_transform=transform, q_predicted_depth=depth)


def toy_spec(index=0):
    return dict(id='unit_geometry_'+str(index), scene='unit_scene', pair=[2, 7],
                pair_name='unit_pair', query_size=2, query_index=index, query_bounds=[1, 1, 5, 4])


TOY_BOUNDS = np.asarray([[-1., -1., 0., 1.], [0., -1., 1., 1.]])


@pytest.mark.parametrize('constant', [False, True])
def test_qr_factor_matches_direct_archived_within_plus_between(constant):
    data, spec = toy_inputs(constant), toy_spec()
    RUN.validate_inputs(data, (4, 6))
    arrays = RUN.prepare_query(data, spec, image_hw=(4, 6), key_bounds=TOY_BOUNDS)
    points = data['q_points3'].reshape(4, 6, 3)[1:4, 1:5].reshape(-1, 3)
    stored = data['q_pixel_cov3'].reshape(4, 6, 3, 3)[1:4, 1:5].reshape(-1, 3, 3)
    mean = points.mean(0)
    delta = points-mean
    direct = stored.mean(0)+delta.T @ delta/len(points)
    rotation, shift = data['q_to_k_transform'][:3, :3], data['q_to_k_transform'][:3, 3]
    np.testing.assert_allclose(arrays['source__covariance'], direct, rtol=0, atol=1e-16)
    np.testing.assert_allclose(arrays['source__factor'] @ arrays['source__factor'].T, direct, rtol=0, atol=1e-16)
    np.testing.assert_allclose(arrays['input__factor'] @ arrays['input__factor'].T, rotation @ direct @ rotation.T, rtol=0, atol=1e-16)
    np.testing.assert_allclose(arrays['input__mu'], rotation @ mean+shift, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(arrays['input__H'], data['k_K'][:2, :2]/8.)
    np.testing.assert_array_equal(arrays['input__offset'], data['k_K'][:2, 2]/8.)
    np.testing.assert_array_equal(arrays['input__bounds'], TOY_BOUNDS)
    assert set(arrays) == RUN.PREPARED_FIELDS


def test_exact_planar_source_receives_no_normal_thickness():
    data = toy_inputs(True)
    data['q_points3'][:, 2] = 2.
    data['q_predicted_depth'][:] = 2.
    arrays = RUN.prepare_query(data, toy_spec(), image_hw=(4, 6), key_bounds=TOY_BOUNDS)
    np.testing.assert_array_equal(arrays['source__factor'][2], np.zeros(3))
    assert arrays['source__covariance'][2, 2] == 0.


def test_given_camera_residual_is_reported_and_never_repaired():
    data = toy_inputs()
    data['q_to_k_transform'][0, 0] += 1e-5
    supplied = data['q_to_k_transform'].copy()
    diagnostics = RUN.validate_inputs(data, (4, 6))
    assert diagnostics['rotation_orthogonality_max_absolute'] > 1e-10
    arrays = RUN.prepare_query(data, toy_spec(), image_hw=(4, 6), key_bounds=TOY_BOUNDS)
    np.testing.assert_array_equal(data['q_to_k_transform'], supplied)
    np.testing.assert_array_equal(arrays['input__factor'], supplied[:3, :3] @ arrays['source__factor'])


def test_raw_source_depth_sanitation_cases_are_left_for_independent_input_audit():
    data = toy_inputs()
    data['q_predicted_depth'][0, :3] = [np.nan, np.inf, -np.inf]
    RUN.validate_inputs(data, (4, 6))


@pytest.mark.parametrize('kind', ['extra_field', 'float32_camera', 'nonfinite_points', 'bad_shape', 'bad_bottom'])
def test_invalid_allowlist_or_input_storage_is_rejected(kind):
    data = toy_inputs()
    if kind == 'extra_field':
        data['q_sensor_depth'] = np.ones((4, 6))
    elif kind == 'float32_camera':
        data['q_K'] = data['q_K'].astype(np.float32)
    elif kind == 'nonfinite_points':
        data['q_points3'][0, 0] = np.inf
    elif kind == 'bad_shape':
        data['q_pixel_cov3'] = data['q_pixel_cov3'][:1]
    else:
        data['q_to_k_transform'][3, 0] = 1.
    with pytest.raises(ValueError):
        RUN.validate_inputs(data, (4, 6))


def test_metadata_freeze_never_decodes_arrays(tmp_path, monkeypatch):
    out = tmp_path/'freeze'
    monkeypatch.setattr(RUN.C, 'entries', lambda:[])
    monkeypatch.setattr(RUN.C, 'query_specs', lambda:[toy_spec()])
    monkeypatch.setattr(RUN.C, 'key_bounds', lambda:TOY_BOUNDS)
    monkeypatch.setattr(RUN, 'inherited_bindings', lambda _: {})
    monkeypatch.setattr(RUN, 'source_names', lambda:[])
    monkeypatch.setattr(RUN, 'check_runtime', lambda:{'unit':True})
    monkeypatch.setattr(RUN, 'verify', lambda _:None)
    monkeypatch.setattr(RUN.np, 'load', lambda *_a, **_k:pytest.fail('metadata freeze opened array'))
    RUN.freeze(out, tmp_path/'audit')
    lock = RUN.read(out/'lock.json')
    assert lock['case_count'] == 1 and not lock['general_numerical_accuracy_established']
    assert RUN.read(out/'query_specs.json') == [toy_spec()]


def test_missing_input_audit_blocks_model_and_array_read(tmp_path, monkeypatch):
    out = tmp_path/'run'
    out.mkdir()
    monkeypatch.setattr(RUN, 'verify', lambda _:dict(audit_root=str(tmp_path/'missing')))
    monkeypatch.setattr(RUN, 'require_audit_freeze', lambda *_:'unit_audit')
    monkeypatch.setattr(RUN.MODEL, 'integrate', lambda **_:pytest.fail('model before input audit'))
    monkeypatch.setattr(RUN.np, 'load', lambda *_a, **_k:pytest.fail('array before input audit'))
    with pytest.raises(FileNotFoundError):
        RUN.run(out)
    assert not (out/'forward').exists()


@pytest.mark.parametrize('tamper', [None, 'future_audit', 'wrong_source'])
def test_input_audit_guard_binds_source_and_precedes_execution(tmp_path, monkeypatch, tamper):
    out, audit = tmp_path/'run', tmp_path/'audit'
    (out/'prepared').mkdir(parents=True)
    (audit/'input_audit').mkdir(parents=True)
    RUN.write(out/'lock.json', {})
    RUN.write(audit/'lock.json', dict(created_unix=1.))
    RUN.write(out/'query_specs.json', [toy_spec()])
    RUN.write(out/'prepared/records.json', [])
    RUN.write(out/'prepared/completion.json', dict(files={}))
    spec_sha = RUN.sha(out/'query_specs.json')
    RUN.write(out/'prepared_complete.json', dict(producer_lock_sha256=RUN.sha(out/'lock.json'),
        audit_lock_sha256='unit_audit', query_specs_sha256=spec_sha,
        prepared_completion_sha256=RUN.sha(out/'prepared/completion.json'),
        prepared_records_sha256=RUN.sha(out/'prepared/records.json'), case_ids=[toy_spec()['id']],
        case_count=1, all_inputs_prepared=True, started_unix=2., finished_unix=3.))
    lock = dict(audit_root=str(audit), query_specs_sha256=spec_sha, case_count=1,
                entries=[dict(input_path='/unit/archive', input_sha256='unit_source')])
    verification = dict(passed=True, producer_lock_sha256=RUN.sha(out/'lock.json'), audit_lock_sha256='unit_audit',
        producer_prepared_complete_sha256=RUN.sha(out/'prepared_complete.json'),
        producer_prepared_completion_sha256=RUN.sha(out/'prepared/completion.json'),
        query_specs_sha256=spec_sha, checked_queries=1, source_input_hashes={'/unit/archive':'unit_source'},
        labels_read=False, independent_probability_reference=False, general_numerical_accuracy_established=False,
        scientific_gate_passed=False, started_unix=4., finished_unix=5.)
    if tamper == 'future_audit':
        verification['finished_unix'] = time.time()+3600.
    elif tamper == 'wrong_source':
        verification['source_input_hashes']['/unit/archive'] = 'different'
    RUN.write(audit/'input_verification.json', verification)
    RUN.write(audit/'input_audit/completion.json', dict(passed=True, files={},
        verification_sha256=RUN.sha(audit/'input_verification.json')))
    monkeypatch.setattr(RUN, 'require_audit_freeze', lambda *_:'unit_audit')
    monkeypatch.setattr(RUN, 'verify_inventory', lambda *_args, **_kwargs:None)
    if tamper:
        with pytest.raises(ValueError):
            RUN.require_input_audit(out, lock)
    else:
        assert RUN.require_input_audit(out, lock)[1] == verification


def test_prepare_retains_all_fixed_rows_when_archive_read_fails(tmp_path, monkeypatch):
    out = tmp_path/'prepare'
    out.mkdir()
    input_path = tmp_path/'unit_input.npz'
    input_path.write_bytes(b'not a real archive')
    specs = [toy_spec(i) for i in range(3)]
    RUN.write(out/'query_specs.json', specs)
    RUN.write(out/'lock.json', {})
    lock = dict(entries=[dict(scene='unit_scene', input_path=str(input_path), input_sha256=RUN.sha(input_path))],
                input_keys=list(RUN.C.INPUT_KEYS), query_specs_sha256=RUN.sha(out/'query_specs.json'))
    monkeypatch.setattr(RUN, 'verify', lambda _:lock)
    monkeypatch.setattr(RUN, 'require_audit_freeze', lambda *_:'unit_audit')
    def fail_load(*_args, **_kwargs):
        raise ValueError('injected archive failure')
    monkeypatch.setattr(RUN.np, 'load', fail_load)
    RUN.prepare(out)
    records = RUN.read(out/'prepared/records.json')
    assert [r['id'] for r in records] == [s['id'] for s in specs]
    assert all(not r['passed'] and 'injected archive failure' in r['failure_reason'] for r in records)
    seal = RUN.read(out/'prepared_complete.json')
    assert not seal['all_inputs_prepared'] and not seal['predictions_executed']
    completion = RUN.read(out/'prepared/completion.json')
    RUN.verify_inventory(out/'prepared', completion['files'], exclude=('completion.json',))


def make_toy_run(tmp_path, monkeypatch, budget=10.):
    out = tmp_path/'run'
    (out/'prepared/arrays').mkdir(parents=True)
    RUN.write(out/'lock.json', {})
    specs, records = [toy_spec(i) for i in range(3)], []
    RUN.write(out/'query_specs.json', specs)
    RUN.write(out/'prepared_complete.json', {'unit':True})
    for spec in specs:
        arrays = RUN.prepare_query(toy_inputs(), spec, image_hw=(4, 6), key_bounds=TOY_BOUNDS)
        path = out/'prepared/arrays'/(spec['id']+'.npz')
        np.savez_compressed(path, **arrays)
        records.append(dict(spec, archive_path=str(path.relative_to(out)), archive_sha256=RUN.sha(path),
                            array_sha256={k:RUN.array_sha(v) for k, v in arrays.items()}))
    RUN.write(out/'prepared/records.json', records)
    lock = dict(policy=deepcopy(RUN.C.POLICY), gates=dict(production_max_phase_wall_seconds=budget),
                query_specs_sha256=RUN.sha(out/'query_specs.json'))
    monkeypatch.setattr(RUN, 'verify', lambda _:lock)
    monkeypatch.setattr(RUN, 'require_audit_freeze', lambda *_:'unit_audit')
    monkeypatch.setattr(RUN, 'require_input_audit', lambda *_:('unit_input_audit', {}))
    return out, specs


def test_common_deadline_failed_rows_and_partial_evidence_are_all_retained(tmp_path, monkeypatch):
    out, specs = make_toy_run(tmp_path, monkeypatch, budget=-1.)
    deadlines = []
    def failure(**kwargs):
        deadlines.append(kwargs['deadline'])
        assert kwargs['deadline'] < time.perf_counter()
        return dict(passed=False, status='failed', failure_reason='unit expired deadline', work={},
                    partial_raw_mass=np.asarray([.2, .3]))
    monkeypatch.setattr(RUN.MODEL, 'integrate', failure)
    RUN.run(out)
    rows, seal = RUN.read(out/'forward/records.json'), RUN.read(out/'forward_complete.json')
    assert len(deadlines) == len(specs) and len(set(deadlines)) == 1
    assert [r['id'] for r in rows] == [s['id'] for s in specs]
    assert all(set(r['array_sha256']) == {'partial_raw_mass'} for r in rows)
    assert not seal['applicability_calls_completed'] and not seal['phase_wall_passed']
    assert not seal['general_numerical_accuracy_established'] and not seal['scientific_gate_passed']
    completion = RUN.read(out/'completion.json')
    RUN.verify_inventory(out, completion['files'], exclude=('completion.json',))
    with pytest.raises(ValueError, match='preserve'):
        RUN.run(out)


def test_post_start_output_failure_is_archived_and_reraised(tmp_path, monkeypatch):
    out, _ = make_toy_run(tmp_path, monkeypatch)
    monkeypatch.setattr(RUN.MODEL, 'integrate', lambda **_:dict(passed=False, status='failed', work={}))
    def fail_write(*_args, **_kwargs):
        raise OSError('injected output failure')
    monkeypatch.setattr(RUN.np, 'savez_compressed', fail_write)
    with pytest.raises(OSError, match='injected'):
        RUN.run(out)
    failure, completion = RUN.read(out/'forward/failure.json'), RUN.read(out/'forward/completion.json')
    assert failure['error_type'] == 'OSError' and not completion['complete_case_inventory']
    assert not (out/'forward_complete.json').exists()
    RUN.verify_inventory(out/'forward', completion['files'], exclude=('completion.json',))
