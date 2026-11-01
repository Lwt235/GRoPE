"""R23 independent audit on invented geometry/metadata only, no old NPZ."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from experiments import round23_ray_applicability_audit as A
from experiments import round23_ray_contract as C


def toy_inputs():
    yy, xx = np.mgrid[:144, :192]
    depth = (1. + .001 * xx + .002 * yy).astype(np.float32)
    depth[0, :3] = [np.nan, np.inf, -np.inf]
    z = np.nan_to_num(depth.astype(np.float64), nan=10., posinf=20., neginf=.1).clip(.1, 20.).ravel()
    qk = np.array([[100., .5, 96.], [0., 105., 72.], [0., 0., 1.]])
    kk = np.array([[80., -.25, 91.], [0., 90., 68.], [0., 0., 1.]])
    hom = np.column_stack((xx.ravel(), yy.ravel(), np.ones(len(z))))
    inverse = np.linalg.inv(qk); points = (hom @ inverse.T) * z[:, None]
    within = z[:, None, None] ** 2 * (inverse[:, :2] @ inverse[:, :2].T)[None] / 12
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transform = np.eye(4); transform[:3, :3] = rotation; transform[:3, 3] = [.3, -.1, .2]
    return dict(q_points3=points, q_pixel_cov3=within, q_K=qk, k_K=kk,
                q_to_k_transform=transform, q_predicted_depth=depth)


def toy_bank():
    dense = A.dense_reference(toy_inputs())
    bounds = A.pixel_bounds(16)[20]
    spec = dict(query_size=16, query_index=20, query_bounds=bounds.tolist())
    ref = A.patch_reference(dense, bounds)
    eig, vectors = np.linalg.eigh(ref['source_covariance']); assert eig.min() > 0
    factor = vectors * np.sqrt(eig)[None]
    bank = dict(input__mu=ref['target_mean'], input__factor=dense['transform'][:3, :3] @ factor,
        input__bounds=ref['bounds'], input__H=ref['H'], input__offset=ref['offset'],
        source__mu=ref['source_mean'], source__factor=factor, source__covariance=ref['source_covariance'],
        target__covariance=ref['target_covariance'], query_bounds_px=bounds, query_index=np.asarray(20, np.int64))
    return dense, spec, bank, ref


def test_sanitized_depth_backprojection_and_within_covariance_are_independent():
    inputs = toy_inputs(); result = A.dense_reference(inputs)
    assert max(result['errors'].values()) < 1e-14
    np.testing.assert_array_equal(result['H'], inputs['k_K'][:2, :2] / 8)
    np.testing.assert_array_equal(result['offset'], inputs['k_K'][:2, 2] / 8)
    assert result['points'][0, 0, 2] == 10. and result['points'][0, 1, 2] == 20. and result['points'][0, 2, 2] == .1


def test_direct_moments_factor_covariance_and_camera_transform_agree():
    dense, spec, bank, ref = toy_bank()
    check, actual = A.geometry_check(spec, bank, dense, 1e-10)
    assert check['passed'] and max(check['normalized_errors'].values()) < 1e-14
    assert check['general_numerical_accuracy_established'] is False
    assert check['independent_probability_reference'] is False
    assert actual['bounds'].shape == (432, 4)
    np.testing.assert_array_equal(actual['bounds'][0], [-.0625, -.0625, .9375, .9375])
    source = dense['points'][16:32, 128:144].reshape(-1, 3)
    delta = source - source.mean(0)
    expected = dense['covariance'][16:32, 128:144].mean((0, 1)) + delta.T @ delta / 256
    np.testing.assert_allclose(ref['source_covariance'], expected, atol=1e-16, rtol=0)


@pytest.mark.parametrize('field', ['q_points3', 'q_pixel_cov3'])
def test_changed_dense_geometry_cannot_pass_independent_input_check(field):
    inputs = toy_inputs(); inputs[field] = inputs[field].copy()
    if field == 'q_points3': inputs[field][100, 0] += .01
    elif field == 'q_pixel_cov3': inputs[field][100, 0, 0] += .01
    assert max(A.dense_reference(inputs)['errors'].values()) > 1e-10


def test_supplied_camera_is_preserved_without_new_orthogonality_quality_gate():
    inputs = toy_inputs(); inputs['q_to_k_transform'][0, 0] += 1e-5
    dense = A.dense_reference(inputs)
    assert max(dense['errors'].values()) < 1e-14
    assert dense['camera_diagnostics']['rotation_orthogonality_max_absolute'] > 1e-6
    np.testing.assert_array_equal(dense['transform'], inputs['q_to_k_transform'])


@pytest.mark.parametrize('field', ['source__mu', 'source__factor', 'source__covariance', 'input__mu',
    'input__factor', 'target__covariance', 'input__H', 'input__offset', 'input__bounds'])
def test_each_prepared_statistical_or_projection_tamper_fails(field):
    dense, spec, bank, _ = toy_bank(); bank[field] = bank[field].copy(); bank[field].flat[0] += .001
    check, _ = A.geometry_check(spec, bank, dense, 1e-10)
    assert not check['passed']


def test_factor_orientation_is_not_inferred_by_replaying_qr():
    dense, spec, bank, _ = toy_bank()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    bank['source__factor'] = bank['source__factor'] @ rotation
    bank['input__factor'] = bank['input__factor'] @ rotation
    assert A.geometry_check(spec, bank, dense, 1e-10)[0]['passed']


@pytest.mark.parametrize('kind', ['extra_label', 'nonfinite', 'dtype', 'query_index', 'key_cell'])
def test_illegal_geometry_schema_and_native_identity_rejected(kind):
    if kind == 'extra_label':
        inputs = dict(toy_inputs(), gt_xy=np.zeros((1, 2)))
        with pytest.raises(ValueError, match='six legal'): A.dense_reference(inputs)
        return
    dense, spec, bank, _ = toy_bank()
    if kind == 'nonfinite': bank['source__factor'][0, 0] = np.nan
    elif kind == 'dtype': bank['source__factor'] = bank['source__factor'].astype(np.float32)
    elif kind == 'query_index': bank['query_index'] = np.asarray(21, np.int64)
    else: bank['input__bounds'][0, 0] += .5
    if kind == 'key_cell': assert not A.geometry_check(spec, bank, dense, 1e-10)[0]['passed']
    else:
        with pytest.raises(ValueError): A.geometry_check(spec, bank, dense, 1e-10)


def test_population_rebuilt_from_toy_metadata_without_opening_archives(tmp_path, monkeypatch):
    entries = []
    for scene in A.SCENES:
        for i in range(16):
            name = '000_0000030_0000045' if i == 0 else f'toy_{i}'
            path = A.ROOT / 'results/round10_partition_data/train' / scene.replace('/', '__') / name
            entries.append(dict(scene=scene, pair=[30, 45] if i == 0 else [i, i + 1], pair_name=name,
                input_path=str(path / 'inputs.npz'), input_sha256='invented-input', manifest_path=str(path / 'manifest.json'), manifest_sha256='invented-manifest'))
    metadata = tmp_path / 'toy_metadata.json'; A.write(metadata, dict(entries=entries))
    monkeypatch.setattr(C, 'METADATA_PATH', metadata)
    monkeypatch.setattr(A.np, 'load', lambda *args, **kwargs: pytest.fail('actual archive decoded'))
    legal, specs = A.metadata_population()
    assert legal == C.entries() and specs == C.query_specs()
    assert len(specs) == 72 and len({s['id'] for s in specs}) == 72
    assert all(len([s for s in specs if s['scene'] == scene]) == 24 for scene in A.SCENES)


def test_freeze_precedes_any_preparation(tmp_path, monkeypatch):
    run = tmp_path / 'producer'; (run / 'prepared').mkdir(parents=True)
    monkeypatch.setattr(A.np, 'load', lambda *args, **kwargs: pytest.fail('array decoded'))
    with pytest.raises(ValueError, match='precede geometry'): A.freeze(run, tmp_path / 'audit')


def test_input_guard_failure_opens_no_arrays_and_preserves_attempt(tmp_path, monkeypatch):
    out = tmp_path / 'audit'; out.mkdir()
    def reject(*args): raise ValueError('preparation seal rejected')
    monkeypatch.setattr(A, 'verify_prepared', reject)
    monkeypatch.setattr(A.np, 'load', lambda *args, **kwargs: pytest.fail('array decoded'))
    with pytest.raises(ValueError, match='seal rejected'): A.input_audit(tmp_path / 'producer', out)
    original = (out / 'input_audit/failure.json').read_bytes()
    assert not A.verify_inventory(out / 'input_audit')['passed']
    with pytest.raises(ValueError, match='preserve previous'): A.input_audit(tmp_path / 'producer', out)
    assert (out / 'input_audit/failure.json').read_bytes() == original


def test_all_72_input_failures_retained_and_model_never_called(tmp_path, monkeypatch):
    out = tmp_path / 'audit'; out.mkdir(); run = tmp_path / 'producer'
    specs = [dict(id=f'toy_{s}_{i}', scene=scene, query_size=(8, 16, 24)[i // 8], query_index=i)
             for s, scene in enumerate(A.SCENES) for i in range(24)]
    lock = dict(entries=[dict(scene=s, input_path=f'invented_{i}.npz', input_sha256='invented') for i, s in enumerate(A.SCENES)],
                gates=C.GATES, query_specs_sha256='toy')
    rows = [dict(s, passed=True) for s in specs]
    monkeypatch.setattr(A, 'verify_prepared', lambda *args: (lock, dict(finished_unix=0), rows, specs))
    monkeypatch.setattr(A, 'verify_joint_lock', lambda *args: (lock, {}))
    monkeypatch.setattr(A, 'sha', lambda path: 'invented-digest')
    def reject(*args, **kwargs): raise ValueError('invented invalid input')
    monkeypatch.setattr(A.np, 'load', reject)
    monkeypatch.setattr(A, 'load_prepared', lambda *args: pytest.fail('source failure must skip prepared decode'))
    with pytest.raises(RuntimeError, match='no model calls'): A.input_audit(run, out)
    checks = A.read(out / 'input_audit/checks.json'); assert len(checks) == 72 and not any(c['passed'] for c in checks)
    result = A.read(out / 'input_verification.json')
    assert result['checked_queries'] == 72 and not result['passed'] and not result['general_numerical_accuracy_established']


def test_final_guard_failure_opens_no_output_arrays(tmp_path, monkeypatch):
    out = tmp_path / 'audit'; out.mkdir()
    def reject(*args): raise ValueError('input audit failed')
    monkeypatch.setattr(A, 'verify_input_audit', reject)
    monkeypatch.setattr(A.np, 'load', lambda *args, **kwargs: pytest.fail('output decoded'))
    with pytest.raises(ValueError, match='input audit failed'): A.audit(tmp_path / 'producer', out)
    assert A.verify_inventory(out / 'final_audit')['passed'] is False


@pytest.mark.parametrize('kind', ['valid_empty', 'valid_partial', 'reason', 'status', 'work', 'time', 'shape', 'success_arrays'])
def test_failed_calls_need_explicit_complete_retained_metadata(kind):
    row = dict(passed=False, status='failed', failure_reason='fixed budget exhausted', work=dict(depth_evaluations=10),
               call_and_input_load_seconds=.01, peak_rss_kib_process=1000)
    outputs = {}
    if kind == 'valid_partial': outputs['partial_raw_mass'] = np.full(432, np.nan, np.float64)
    elif kind == 'reason': row.pop('failure_reason')
    elif kind == 'status': row['status'] = 'complete'
    elif kind == 'work': row['work']['depth_evaluations'] = -1
    elif kind == 'time': row['call_and_input_load_seconds'] = np.inf
    elif kind == 'shape': outputs['partial_raw_mass'] = np.zeros(431)
    elif kind == 'success_arrays': outputs['probability'] = np.zeros(432)
    if kind.startswith('valid'): assert A.validate_forward_row(row, outputs)
    else:
        with pytest.raises(ValueError): A.validate_forward_row(row, outputs)


def test_source_closure_and_runtime_match_without_model_execution():
    from experiments import round23_ray_applicability as P
    assert A.source_names() == P.source_names()
    assert A.runtime() == P.check_runtime()
    assert C.INPUT_KEYS == A.INPUT_KEYS
