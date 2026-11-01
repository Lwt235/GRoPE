"""Synthetic integration contracts for the R17 regional outer audit."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments import round17_regional_audit as A


def record_fixture():
    entry = dict(scene=A.SCENES[0], pair=[0, 5], input_sha256='input', label_sha256='label')
    condition = dict(scenario='partition', query='g8', key='g8', condition='g8_g8')
    forward = dict(scene=entry['scene'], pair=entry['pair'], family='trace_full', grid_index=4,
        config=dict(h=.5, temperature=5.), **condition, shape=[432, 432], dtype='float64',
        array_sha256='prob', input_sha256='input')
    measured = dict(forward, label_sha256='label', replayed_array_sha256='prob', n_query=432, n_key=432)
    return entry, condition, forward, measured


def test_record_preserves_every_forward_field_and_actual_label_binding():
    entry, condition, forward, measured = record_fixture()
    A.verify_record(forward, measured, entry, condition)
    for name, value in [('array_sha256', 'other'), ('label_sha256', 'other'),
                        ('replayed_array_sha256', 'other'), ('n_query', 431), ('dtype', 'float32')]:
        bad = deepcopy(measured); bad[name] = value
        with pytest.raises(ValueError):
            A.verify_record(forward, bad, entry, condition)
    bad = deepcopy(forward); bad['numerical'] = {'raw_zero_cells': 1}
    with pytest.raises(ValueError, match='committed'):
        A.verify_record(bad, measured, entry, condition)


def overlap_fixture():
    entries = [dict(scene=A.SCENES[0], pair=[0, 5])]
    records, previous = [], []
    for index in range(12):
        for condition in A.CONDITIONS:
            q, k = condition['query'], condition['key']
            row = dict(scene=A.SCENES[0], pair=[0, 5], family='trace_full', grid_index=index,
                config=A.GRID[index], **condition, shape=[A.M.NATIVE_SIZES[q], A.M.NATIVE_SIZES[k]],
                dtype='float64', input_sha256='input', array_sha256=f'{index}/{q}/{k}')
            if k == 'g8' and q in ('g8', 'g16', 'g24'):
                row['stage_a_array_sha256'] = row['array_sha256']
                previous.append(dict(row))
            records.append(row)
    return entries, records, previous


def test_full_14_condition_index_and_all_three_scale_overlap():
    entries, rows, old = overlap_fixture()
    assert len(A.index_rows(rows, entries)) == 168
    assert A.verify_stage_a_overlap(rows, old, entries) == 36
    with pytest.raises(ValueError, match='population'):
        A.index_rows(rows[:-1], entries)
    with pytest.raises(ValueError, match='population'):
        A.index_rows(rows[:-1] + [rows[0]], entries)
    bad = deepcopy(rows); bad[0]['grid_index'] = True
    with pytest.raises(ValueError):
        A.index_rows(bad, entries)


@pytest.mark.parametrize('change', ['missing_old', 'missing_new', 'new_hash', 'input', 'undeclared'])
def test_stage_a_overlap_cannot_be_skipped_or_changed(change):
    entries, rows, old = overlap_fixture()
    if change == 'missing_old':
        old.pop()
    elif change == 'missing_new':
        rows = [r for r in rows if not (r['grid_index'] == 11 and r['condition'] == 'g24_g8')]
    elif change == 'new_hash':
        rows[0]['array_sha256'] = 'changed'
    elif change == 'input':
        rows[0]['input_sha256'] = 'changed'
    else:
        rows[1]['stage_a_array_sha256'] = rows[1]['array_sha256']
    with pytest.raises(ValueError):
        A.verify_stage_a_overlap(rows, old, entries)


def chronology_fixture():
    runtime = dict(device='cpu', cpu_threads=4)
    lock = dict(created_unix=1., runtime=runtime)
    forward = dict(stage='forward', scene=A.SCENES[0], lock_sha256='lock', record_count=2688,
        pair_count=16, stage_a_exact_overlap_count=576, regional_labels_decoded=False,
        all_probabilities_retained=False, all_probability_hashes_retained=True,
        runtime=runtime, started_unix=2., finished_unix=3.)
    evaluation = dict(stage='evaluation', scene=A.SCENES[0], lock_sha256='lock', record_count=2688,
        new_regional_labels_decoded=True, all_replay_hashes_exact=True,
        forward_complete_sha256='seal', runtime=runtime, started_unix=5., finished_unix=6.)
    seal = dict(sealed_unix=4., forward_chain={A.SCENES[0]:dict(completion_sha256='forward', finished_unix=3.)})
    return lock, seal, forward, evaluation


@pytest.mark.parametrize('change', [None, 'early_label', 'early_forward', 'seal', 'runtime', 'count', 'retention'])
def test_complete_global_forward_precedes_every_label_stage(change):
    lock, seal, forward, evaluation = deepcopy(chronology_fixture())
    if change == 'early_label':
        evaluation['started_unix'] = 3.5
    elif change == 'early_forward':
        forward['started_unix'] = .5
    elif change == 'seal':
        seal['forward_chain'][A.SCENES[0]]['completion_sha256'] = 'other'
    elif change == 'runtime':
        evaluation['runtime'] = dict(device='cpu', cpu_threads=8)
    elif change == 'count':
        forward['stage_a_exact_overlap_count'] = 575
    elif change == 'retention':
        forward['all_probabilities_retained'] = True
    kwargs = dict(lock_sha='lock', seal_sha='seal', forward_sha='forward')
    if change is None:
        A.verify_chronology(lock, seal, forward, evaluation, A.SCENES[0], **kwargs)
    else:
        with pytest.raises(ValueError, match='chronology'):
            A.verify_chronology(lock, seal, forward, evaluation, A.SCENES[0], **kwargs)


def support_fixture():
    n = 48
    mean = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    covariance = np.broadcast_to(np.eye(2, dtype=np.float32), (n, 2, 2)).copy()
    content = np.zeros((n, 4), dtype=np.float32)
    bounds = np.arange(16, dtype=np.float64).reshape(4, 4)
    indices = np.linspace(0, n - 1, 8, dtype=np.int64)
    saved = dict(indices=indices, actual_mean=mean[indices], actual_cov=covariance[indices],
                 actual_content=content[indices], bounds=bounds.copy())
    return saved, mean, covariance, content, bounds, n


@pytest.mark.parametrize('change', [None, 'indices', 'mean', 'dtype', 'bounds', 'content'])
def test_reused_reference_requires_identical_prepared_support_and_native_bounds(change):
    saved, mean, covariance, content, bounds, n = support_fixture()
    if change == 'indices':
        saved['indices'][-1] -= 1
    elif change == 'mean':
        saved['actual_mean'][0, 0] += np.float32(1e-10)
    elif change == 'dtype':
        saved['actual_cov'] = saved['actual_cov'].astype(np.float64)
    elif change == 'bounds':
        saved['bounds'][0, 0] += 1e-10
    elif change == 'content':
        saved['actual_content'][0, 0] += np.float32(1e-10)
    if change is None:
        np.testing.assert_array_equal(A.compare_support(saved, mean, covariance, content, bounds, n), saved['indices'])
    else:
        with pytest.raises(ValueError):
            A.compare_support(saved, mean, covariance, content, bounds, n)


def test_numeric_reference_reuse_reports_actual_reference_and_integration_separately():
    p = np.broadcast_to(np.array([.4, .3, .2, .1]), (8, 4)).copy()
    retained = {f'{name}_T5': p.copy() for name in ('actual', 'reference64', 'reference96', 'hybrid')}
    ok = A.reference_check(p, retained, 4)
    assert ok['passed'] and ok['query_count'] == 8
    assert ok['same_prepared_actual_error'] == ok['max_probability_error'] == 0
    retained['actual_T5'][0, :2] += [2e-12, -2e-12]
    bad = A.reference_check(p, retained, 4)
    assert not bad['passed'] and bad['max_probability_error'] == 0
    retained['actual_T5'] = p.copy()
    for name in ('reference64', 'reference96', 'hybrid'):
        retained[name + '_T5'][0, :2] += [2e-4, -2e-4]
    bad = A.reference_check(p, retained, 4)
    assert not bad['passed'] and bad['reference_convergence_error'] == 0
    retained['reference64_T5'][0, :2] += [2e-8, -2e-8]
    assert A.reference_check(p, retained, 4)['reference_convergence_error'] > 1e-8


def test_numerical_reuse_rejects_unormalized_and_dtype_changes():
    p = np.full((8, 4), .25)
    retained = {f'{name}_T0': p.copy() for name in ('actual', 'reference64', 'reference96', 'hybrid')}
    retained['hybrid_T0'] = retained['hybrid_T0'].astype(np.float32)
    with pytest.raises(ValueError, match='probability'):
        A.reference_check(p, retained, 0)
    retained['hybrid_T0'] = p.copy(); retained['hybrid_T0'][0, 0] = .26
    with pytest.raises(ValueError, match='probability'):
        A.reference_check(p, retained, 0)


def test_tree_numeric_tolerance_never_permits_changed_gate_or_selected_index():
    expected = dict(passed=False, primary=dict(grid_index=6, J=.5), eligible=[6, 9])
    assert A.compare_tree(dict(expected, primary=dict(grid_index=6, J=.5 + 1e-11)), expected) < 1e-10
    with pytest.raises(ValueError, match='identity'):
        A.compare_tree(dict(expected, passed=True), expected)
    with pytest.raises(ValueError, match='identity'):
        A.compare_tree(dict(expected, primary=dict(grid_index=9, J=.5)), expected)
    with pytest.raises(ValueError, match='keys'):
        A.compare_tree(dict(expected, extra=True), expected)
    with pytest.raises(ValueError, match='identity'):
        A.compare_tree(dict(expected, primary=dict(grid_index=6., J=.5)), expected)


def test_legacy_flat_snapshots_are_checked_using_original_convention(tmp_path, monkeypatch):
    project = tmp_path / 'project'; directory = tmp_path / 'legacy'
    (project / 'models').mkdir(parents=True); (directory / 'source_snapshot').mkdir(parents=True)
    (project / 'models' / 'old.py').write_text('old\n')
    (directory / 'source_snapshot' / 'old.py').write_text('old\n')
    configurations = {role: dict(method=method, h=h, temperature=t) for role, method, h, t in (
        ('point', 'point', 1., 5.), ('point_pck', 'point', .5, 5.), ('fixed', 'fixed', .5, 5.),
        ('full1', 'full', .25, 0.), ('full4', 'full4', .5, 5.), ('quad4', 'quad4', .5, 0.),
        ('quad16', 'quad16', .5, 5.), ('iso4', 'iso4', .5, 5.))}
    prior = dict(source_hashes={'models/old.py': A.sha(project / 'models' / 'old.py')},
                 inherited=dict(files={}), configurations=configurations)
    (directory / 'lock.json').write_text(json.dumps(prior))
    digest = A.sha(directory / 'lock.json'); (directory / 'lock.sha256').write_text(digest + '\n')
    names = dict(point='gaussian_point', fixed='gaussian_fixed', full='gaussian_full1',
                 full4='gaussian_full4', quad4='gaussian_quad4', quad16='gaussian_quad16')
    selected = {}
    for role, value in configurations.items():
        if role == 'iso4':
            continue
        cfg = {n: value[n] for n in ('h', 'temperature')}
        selected[role] = dict(family=names[value['method']], grid_index=A.GRID.index(cfg), config=cfg)
    lock = dict(legacy=dict(root=str(directory), lock_sha256=digest, upstream_files={},
                            configurations=selected, uncalibrated_legacy_iso4=configurations['iso4']))
    old = dict(metrics={family: [{'example': i} for i in range(12)] for family in names.values()})
    monkeypatch.setattr(A, 'ROOT', project)
    assert A.verify_legacy(lock, old)['point']['summary'] == dict(example=7)
    (directory / 'source_snapshot' / 'old.py').write_text('changed\n')
    with pytest.raises(ValueError, match='snapshot'):
        A.verify_legacy(lock, old)


def test_failed_audit_retains_failure_artifact_and_refuses_overwrite(tmp_path, monkeypatch):
    run, out = tmp_path / 'run', tmp_path / 'audit'
    run.mkdir()
    def fail(run, out, started):
        out.mkdir(); A.write(out / 'progress.json', dict(started=started))
        raise ValueError('synthetic source mismatch')
    monkeypatch.setattr(A, '_audit', fail)
    with pytest.raises(ValueError, match='synthetic'):
        A.audit(run, out)
    assert json.loads((out / 'failure.json').read_text())['passed'] is False
    complete = json.loads((out / 'completion.json').read_text())
    assert complete['passed'] is complete['scientific_gate_passed'] is False
    assert set(complete['files']) == {'progress.json', 'failure.json'}
    with pytest.raises(ValueError, match='overwrite'):
        A.audit(run, out)
    with pytest.raises(ValueError, match='mutation'):
        A.audit(run, run / 'inside')
