from copy import deepcopy
from pathlib import Path

import numpy as np

import pytest

from experiments.round17_feasibility_audit import (
    SCENES, QUERIES, FAMILIES, GRID, independent_center_selection, verify_constraint_table,
)
from experiments import round17_feasibility_audit as A


def table():
    return [dict(family=f, h=h, temperature=t, scene=s, query=q, pair=[1, 2],
                 correct=900, valid=1000)
            for f in FAMILIES for h, t in GRID for s in SCENES for q in QUERIES]


def test_global_point_choice_and_ordered_ties():
    rows = table()
    for r in rows:
        if r['family'] == 'quartic_point' and (r['h'], r['temperature']) == (1., 5.):
            r['correct'] = 910 if r['query'] == 'g24' else 900
    result = independent_center_selection(rows, expected_pairs=1)
    assert result['point_configurations'] == {'gaussian_point': (.25, 0.), 'quartic_point': (1., 5.)}
    assert result['stop_candidate'] and not result['regional_primary_selected']


def test_legacy_scale_protection_survives_a_new_point_selection():
    rows = table()
    for r in rows:
        if r['family'] == 'gaussian_point' and (r['h'], r['temperature']) == (.5, 5.):
            r['correct'] = 920 if r['query'] == 'g8' else 880
    result = independent_center_selection(rows, expected_pairs=1)
    assert result['point_configurations']['gaussian_point'] == (.25, 0.)
    assert result['stop_candidate']
    assert any(c['reference'] == 'legacy_point_pck' and not c['passed'] for c in result['checks'])


def test_equal_scene_not_pooled_and_strict_population_checks():
    rows = table()
    for r in rows:
        if r['scene'] == SCENES[0]:
            r['valid'], r['correct'] = 100, 50
    result = independent_center_selection(rows, expected_pairs=1)
    assert result['scores']['trace_full', (.25, 0.)]['by_query']['g8'] == pytest.approx((.5+.9+.9)/3)
    assert len(result['eligible']) == 12
    with pytest.raises(ValueError, match='duplicate'):
        independent_center_selection(rows+[rows[0]], expected_pairs=1)
    rows[0]['valid'] = 101
    with pytest.raises(ValueError, match='population'):
        independent_center_selection(rows, expected_pairs=1)


def test_no_tolerance_beyond_the_declared_half_point_guard():
    rows = table()
    for r in rows:
        if r['family'] == 'trace_full':
            r['correct'] = 894
    result = independent_center_selection(rows, expected_pairs=1)
    assert result['eligible'] == [] and result['stop_candidate']
    rows[0]['correct'] = 1001
    with pytest.raises(ValueError, match='counts'):
        independent_center_selection(rows, expected_pairs=1)


def test_exact_half_point_boundary_uses_integer_ratios():
    rows = table()
    for row in rows:
        if row['family'] == 'trace_full':
            row['correct'] = 895
    result = independent_center_selection(rows, expected_pairs=1)
    assert len(result['eligible']) == 12
    assert all(c['passed'] and c['difference_pp'] == -.5 for c in result['checks'])


def test_constraint_audit_rejects_missing_and_duplicated_failed_guards():
    rows = table()
    for row in rows:
        if row['family'] == 'trace_full':
            row['correct'] = 890
    result = independent_center_selection(rows, expected_pairs=1)
    references = {'selected_gaussian_point': 'gaussian_point',
                  'selected_quartic_point': 'quartic_point', 'legacy_point_pck': 'legacy_point_pck'}
    saved = []
    for i, (h, temperature) in enumerate(GRID):
        guards = [dict(query=c['query'], reference=references[c['reference']],
                       passed=c['passed'], difference_pp=c['difference_pp'])
                  for c in result['checks'] if (c['h'], c['temperature']) == (h, temperature)]
        saved.append(dict(grid_index=i, config=dict(h=h, temperature=temperature),
                          feasible=False, checks=guards))
    verify_constraint_table(saved, result['checks'], [])
    with pytest.raises(ValueError, match='configuration guards'):
        verify_constraint_table(saved[:-1], result['checks'], [])
    duplicate = deepcopy(saved)
    duplicate[0]['checks'][1] = duplicate[0]['checks'][0]
    with pytest.raises(ValueError, match='duplicate'):
        verify_constraint_table(duplicate, result['checks'], [])
    omitted = deepcopy(saved)
    omitted[0]['checks'].pop()
    with pytest.raises(ValueError, match='configuration guards'):
        verify_constraint_table(omitted, result['checks'], [])


def synthetic_summary(rows):
    result = A.independent_center_selection(rows, expected_pairs=1)
    metrics = {}
    for family in FAMILIES:
        metrics[family] = []
        for i, cfg in enumerate(GRID):
            score = result['scores'][family, cfg]
            by_scene = {}
            for scene in SCENES:
                by_scene[scene] = {}
                for query in QUERIES:
                    subset = [r for r in rows if (r['family'], r['h'], r['temperature'], r['scene'], r['query']) == (family, *cfg, scene, query)]
                    by_scene[scene][query] = dict(pck8=float(score['by_scene'][scene][query]),
                        n_point=sum(r['valid'] for r in subset), n_query=A.COUNTS[query])
            metrics[family].append(dict(grid_index=i, config=dict(h=cfg[0], temperature=cfg[1]),
                by_scene_query=by_scene, by_query={q: float(v) for q, v in score['by_query'].items()},
                equal_scene_equal_scale_pck8=float(score['macro'])))
    eligible = [GRID.index((v['h'], v['temperature'])) for v in result['eligible']]
    refs = {'selected_gaussian_point': ('gaussian_point', result['point_configurations']['gaussian_point']),
            'selected_quartic_point': ('quartic_point', result['point_configurations']['quartic_point']),
            'legacy_point_pck': ('gaussian_point', (.5, 5.))}
    names = {'selected_gaussian_point': 'gaussian_point', 'selected_quartic_point': 'quartic_point',
             'legacy_point_pck': 'legacy_point_pck'}
    checks = []
    for i, cfg in enumerate(GRID):
        local = []
        for c in result['checks']:
            if (c['h'], c['temperature']) != cfg:
                continue
            family, ref = refs[c['reference']]
            q = c['query']
            local.append(dict(query=q, reference=names[c['reference']], reference_family=family,
                reference_grid_index=GRID.index(ref),
                candidate_pck8=float(result['scores']['trace_full', cfg]['by_query'][q]),
                reference_pck8=float(result['scores'][family, ref]['by_query'][q]),
                difference_pp=c['difference_pp'], passed=c['passed']))
        checks.append(dict(grid_index=i, config=dict(h=cfg[0], temperature=cfg[1]), feasible=i in eligible, checks=local))
    return dict(metrics=metrics, feasibility=checks, feasible_grid_indices=eligible, feasible=bool(eligible),
        point_localization_selections={f:dict(grid_index=GRID.index(cfg),config=dict(h=cfg[0],temperature=cfg[1]))
                                     for f,cfg in result['point_configurations'].items()},
        legacy_point=dict(grid_index=4, config=dict(h=.5, temperature=5.)),
        decision_pending_independent_audit='complete_full_regional_calibration' if eligible else 'stop_unchanged_candidate',
        final_primary_selected=False, regional_calibration_done=False, scientific_gate_passed=False, fresh_test_read=False)


def test_full_summary_checks_integer_populations_guard_identity_and_scope():
    rows = table(); saved = synthetic_summary(rows)
    answer = A.verify_summary(saved, rows, expected_pairs=1)
    assert answer['feasible_grid_indices'] == list(range(12)) and answer['maximum_summary_error'] == 0
    changed = deepcopy(saved)
    changed['metrics']['gaussian_point'][0]['by_scene_query'][SCENES[0]]['g8']['n_point'] += 1
    with pytest.raises(ValueError, match='integer population'):
        A.verify_summary(changed, rows, expected_pairs=1)
    changed = deepcopy(saved)
    changed['feasibility'][0]['checks'][0]['reference_grid_index'] = 11
    with pytest.raises(ValueError, match='guard reference'):
        A.verify_summary(changed, rows, expected_pairs=1)
    changed = deepcopy(saved); changed['scientific_gate_passed'] = True
    with pytest.raises(ValueError, match='scope'):
        A.verify_summary(changed, rows, expected_pairs=1)
    changed = deepcopy(saved); changed['feasibility'][0]['checks'][0]['candidate_pck8'] = float('nan')
    with pytest.raises(ValueError, match='guard reference'):
        A.verify_summary(changed, rows, expected_pairs=1)
    changed = deepcopy(saved); changed['feasibility'][0]['checks'][0]['passed'] = 1
    with pytest.raises(ValueError, match='constraint differs'):
        A.verify_summary(changed, rows, expected_pairs=1)


def test_counts_are_integers_not_boolean_or_float_surrogates():
    for bad in (True, 900.):
        rows = table(); rows[0]['correct'] = bad
        with pytest.raises(ValueError, match='integer counts'):
            A.independent_center_selection(rows, expected_pairs=1)


def test_native_centers_first_tie_eight_pixel_boundary_invalid_not_wrong():
    p = np.zeros((3, 432), dtype=np.float64)
    p[0, 0] = 1
    p[1, :2] = .5
    p[2, -1] = 1
    target = np.array([[11.5, 3.5], [19.5, 3.5], [np.nan, np.inf]])
    result = A.center_counts(p, target, np.array([True, True, False]))
    assert result['correct'] == 1 and result['valid'] == 2 and result['n_query'] == 3
    assert result['argmax_sha256'] == A.array_sha(np.array([0, 0, 431], dtype=np.int64))
    with pytest.raises(ValueError):
        A.center_counts(p.astype(np.float32), target, np.array([True, True, False]))


def test_relative_input_paths_resolve_against_workspace_not_filesystem_root(tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'ROOT', tmp_path)
    p = tmp_path/'results'/'example.bin'; p.parent.mkdir(); p.write_bytes(b'input')
    assert A.absolute('results/example.bin') == p
    assert A.absolute(p) == p
    A.verify_files(A.ROOT, {'results/example.bin': A.sha(p)})


def test_completion_rejects_changed_and_unlisted_artifacts(tmp_path):
    p = tmp_path/'records.json'; A.write(p, [])
    A.write(tmp_path/'completion.json', dict(files={'records.json': A.sha(p)}, started_unix=1., finished_unix=2.,
        stage='forward', scene=SCENES[0], lock_sha256='lock'))
    A.checked_completion(tmp_path, expected_stage='forward', scene=SCENES[0], lock_sha='lock')
    extra = tmp_path/'extra.bin'; extra.write_bytes(b'unbound')
    with pytest.raises(ValueError, match='extra'):
        A.checked_completion(tmp_path)
    extra.unlink(); p.write_text('[1]')
    with pytest.raises(ValueError, match='changed artifact'):
        A.checked_completion(tmp_path)


def chronology_fixture():
    scene = SCENES[0]
    runtime = dict(device='cpu', device_name='cpu', cpu_threads=4, deterministic_algorithms=True,
                   tf32=False, numpy_version='np', torch_version='torch')
    lock = dict(created_unix=1., numpy_version='np', torch_version='torch')
    forward = dict(stage='forward', scene=scene, lock_sha256='lock', record_count=576, pair_count=16,
        labels_decoded=False, all_probabilities_retained=True, started_unix=2., finished_unix=3., runtime=runtime)
    evaluation = dict(stage='evaluation', scene=scene, lock_sha256='lock', record_count=576,
        center_labels_decoded=True, regional_labels_decoded=False, forward_complete_sha256='seal',
        started_unix=5., finished_unix=6.)
    seal = dict(lock_sha256='lock', array_count=1728, labels_decoded=False, all_probabilities_retained=True,
                sealed_unix=4., forward_chain={scene: dict(completion_sha256='forward', finished_unix=3.)})
    return lock, seal, forward, evaluation


def test_chronology_requires_forward_finish_then_global_seal_then_target_and_cpu4():
    args = chronology_fixture()
    kwargs = dict(scene=SCENES[0], lock_sha='lock', seal_sha='seal', forward_sha='forward')
    A.verify_chronology(*args, **kwargs)
    changed = deepcopy(args); changed[3]['started_unix'] = 3.5
    with pytest.raises(ValueError, match='chronology'):
        A.verify_chronology(*changed, **kwargs)
    changed = deepcopy(args); changed[2]['runtime']['cpu_threads'] = 8
    with pytest.raises(ValueError, match='runtime'):
        A.verify_chronology(*changed, **kwargs)
    changed = deepcopy(args); changed[2]['labels_decoded'] = True
    with pytest.raises(ValueError, match='chronology'):
        A.verify_chronology(*changed, **kwargs)


def test_joint_source_freeze_uses_producer_design_time_not_later_audit_time():
    args = chronology_fixture()
    kwargs = dict(scene=SCENES[0], lock_sha='lock', seal_sha='seal', forward_sha='forward')
    A.verify_chronology(*args, **kwargs)
    args[0]['created_unix'] = 2.1
    with pytest.raises(ValueError, match='chronology'):
        A.verify_chronology(*args, **kwargs)


def test_record_index_requires_all_new_or_reused_identities():
    entries = [dict(scene=SCENES[0], pair=[1, 2])]
    rows = [dict(scene=SCENES[0], pair=[1, 2], family=f, grid_index=i, query=q)
            for f in ('quartic_point', 'gaussian_point') for i in range(12) for q in QUERIES]
    assert len(A.indexed_rows(rows, entries, ('quartic_point', 'gaussian_point'))) == 72
    for bad in (rows[:-1], rows + rows[:1]):
        with pytest.raises(ValueError, match='population'):
            A.indexed_rows(bad, entries, ('quartic_point', 'gaussian_point'))
    rows[0]['scene'] = SCENES[1]
    with pytest.raises(ValueError, match='scene'):
        A.indexed_rows(rows, entries, ('quartic_point', 'gaussian_point'))


def test_source_sha_in_joint_snapshot_detects_replaced_auditor(tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'ROOT', tmp_path/'workspace')
    name = 'experiments/audit.py'; source = A.ROOT/name; source.parent.mkdir(parents=True)
    source.write_text('original')
    snapshot = tmp_path/'run'/'source_snapshot'/name; snapshot.parent.mkdir(parents=True)
    snapshot.write_text('original'); lock = dict(source_hashes={name:A.sha(source)})
    A.verify_sources(tmp_path/'run', lock)
    source.write_text('changed')
    with pytest.raises(ValueError, match='changed artifact'):
        A.verify_sources(tmp_path/'run', lock)


def test_all_forward_chains_are_checked_before_any_target_decode(tmp_path, monkeypatch):
    run, out = tmp_path/'run', tmp_path/'audit'; run.mkdir()
    for name in ('lock.json', 'partitions.npz', 'summary.json'):
        (run/name).write_bytes(b'fixed metadata only')
    A.write(run/'forward_complete.json', dict(forward_chain={s:{} for s in SCENES}))
    A.write(run/'completion.json', dict(files={p.name:A.sha(p) for p in run.iterdir()},
        lock_sha256=A.sha(run/'lock.json'), summary_sha256=A.sha(run/'summary.json'),
        forward_complete_sha256=A.sha(run/'forward_complete.json'), center_labels_decoded=True,
        regional_labels_decoded=False, fresh_test_read=False, independent_audit_required=True, finished_unix=1.))
    monkeypatch.setattr(A, 'checked_lock', lambda path: dict(partitions_sha256=A.sha(run/'partitions.npz')))
    monkeypatch.setattr(A, 'verify_policy', lambda value: None)
    monkeypatch.setattr(A, 'verify_previous', lambda value: None)
    monkeypatch.setattr(A, 'verify_numerical', lambda value: {})
    def forbidden(*args, **kwargs):
        raise AssertionError('probability/target decode happened before all source/forward metadata')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(FileNotFoundError):
        A.audit(run, out)
    assert not out.exists()


def arithmetic_rows_fixture():
    cohort = [dict(scene=s, pair=[10, 20]) for s in SCENES]
    counts = dict(g8=432, g16=108, mixed=216, g24=48)
    rows, files = [], {}
    hash_names = ('actual_raw', 'actual_cov', 'actual_content', 'bounds', 'saved_probability',
                  'same_mass_reference_probability', 'actual_tau32_recomputed', 'reference_tau32',
                  'actual_bias32_recomputed', 'reference_bias32')
    for scene in SCENES:
        prefix = 'arrays/'+scene.replace('/', '__')+'/'
        for q, k in A.NUMERICAL_CONDITIONS:
            support = prefix+f'{q}_{k}_support.npz'; files[support] = 'a'*64
            for h, t in GRID:
                geometry = prefix+f'{q}_{k}_h{h:g}.npz'; files[geometry] = 'b'*64
                rows.append(dict(scene=scene, pair=[10, 20], query=q, key=k, h=h, temperature=t,
                    query_indices=np.linspace(0, counts[q]-1, 8, dtype=np.int64).tolist(),
                    support_archive=support, support_archive_sha256='a'*64,
                    geometry_archive=geometry, geometry_archive_sha256='b'*64,
                    queries=8, keys=counts[k], passed=True, tau32_exact=True, bias32_exact=True,
                    max_same_mass_probability_error=0., per_query_probability_error=[0.]*8,
                    actual_tau32=[.25]*8, reference_tau32=[.25]*8, max_tau32_error=0., max_bias32_error=0.,
                    tau_mismatch_rows=[], bias_mismatch_rows=[], argmax_mismatch_rows=[],
                    raw_zero_fov_rows=0, raw_tiny_fov_rows=1, array_hashes={n:'c'*64 for n in hash_names}))
    return rows, cohort, files, counts


def test_arithmetic_all_504_unique_matrices_and_210_archives_required():
    rows, cohort, files, counts = arithmetic_rows_fixture()
    result = A.verify_arithmetic_rows(rows, cohort, files, counts)
    assert result['probability_matrices'] == 504 and result['probability_rows'] == 4032
    assert result['support_archives'] == 42 and result['geometry_archives'] == 168
    for bad in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError, match='population'):
            A.verify_arithmetic_rows(bad, cohort, files, counts)
    files['unclaimed.npz'] = 'f'*64
    with pytest.raises(ValueError, match='population'):
        A.verify_arithmetic_rows(rows, cohort, files, counts)


@pytest.mark.parametrize('field,value', [
    ('bias32_exact', False), ('max_bias32_error', 1.e-30),
    ('per_query_probability_error', [float('nan')]*8),
    ('query_indices', list(range(8))), ('geometry_archive_sha256', 'd'*64),
])
def test_arithmetic_metadata_cannot_hide_failed_exactness_or_changed_inputs(field, value):
    rows, cohort, files, counts = arithmetic_rows_fixture()
    rows[0][field] = value
    with pytest.raises(ValueError):
        A.verify_arithmetic_rows(rows, cohort, files, counts)


def test_arithmetic_argmax_is_diagnostic_but_probability_bound_is_mandatory():
    rows, cohort, files, counts = arithmetic_rows_fixture()
    rows[0]['argmax_mismatch_rows'] = [0, 7]
    assert A.verify_arithmetic_rows(rows, cohort, files, counts)['argmax_mismatch_rows'] == 2
    rows[0]['per_query_probability_error'][0] = 1.01e-12
    rows[0]['max_same_mass_probability_error'] = 1.01e-12
    with pytest.raises(ValueError, match='exactness/probability'):
        A.verify_arithmetic_rows(rows, cohort, files, counts)


def arithmetic_tree(tmp_path, monkeypatch, *, frozen_time=2., completion_time=7., changed_source=False):
    monkeypatch.setattr(A, 'ROOT', tmp_path/'workspace')
    directory, root = tmp_path/'arithmetic', tmp_path/'numerical'
    directory.mkdir(); (root/'train').mkdir(parents=True)
    rows, cohort, train_files, counts = arithmetic_rows_fixture()
    sources = {}
    for name in A.ARITHMETIC_SOURCES:
        source, snapshot = A.ROOT/name, directory/'source_snapshot'/name
        source.parent.mkdir(parents=True, exist_ok=True); snapshot.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('fixed audit source'); snapshot.write_text('fixed audit source')
        sources[name] = A.sha(source)
    base = dict(created_unix=1., source_hashes={'models/base.py':'d'*64}, cohort=cohort)
    A.write(root/'lock.json', base)
    np.savez(root/'partitions.npz', **{q+'__bounds':np.zeros((n, 4)) for q, n in counts.items()})
    A.write(root/'train'/'summary.json', dict(passed=True))
    A.write(root/'train'/'completion.json', dict(started_unix=3., finished_unix=4., files=train_files))
    frozen = dict(created_unix=frozen_time, run_root=str(root), base_numeric_lock_sha256=A.sha(root/'lock.json'),
        base_source_hashes=base['source_hashes'], new_source_hashes=sources, train_absent_at_freeze=True,
        support_archives=42, geometry_archives=168, probability_matrices=504, probability_rows=4032,
        same_mass_tolerance=1.e-12, original_observation_inputs_read=False, labels_read=False)
    A.write(directory/'lock.json', frozen); digest=A.sha(directory/'lock.json')
    (directory/'lock.sha256').write_text(digest+'\n')
    A.write(directory/'audit_started.json', dict(started_unix=5., arithmetic_lock_sha256=digest))
    A.write(directory/'matrix_rows.json', rows)
    totals = A.verify_arithmetic_rows(rows, cohort, train_files, counts)
    report = dict(passed=True, arithmetic_lock_sha256=digest,
        base_numeric_lock_sha256=A.sha(root/'lock.json'), train_summary_sha256=A.sha(root/'train'/'summary.json'),
        train_completion_sha256=A.sha(root/'train'/'completion.json'), arithmetic_frozen_unix=frozen_time,
        train_started_unix=3., source_hashes=sources, base_source_hashes=base['source_hashes'], **totals,
        tau32_exact=True, bias32_exact=True, same_mass_tolerance=1.e-12, failed_matrix_indices=[],
        actual_tau_bias_recomputed_from_locked_function=True, original_tau_bias_arrays_were_not_saved=True,
        original_observation_inputs_read=False, labels_read=False, validation_or_test_read=False,
        new_real_predictions=False, scientific_gate_passed=False, started_unix=5., finished_unix=6.)
    A.write(directory/'verification.json', report)
    A.write(directory/'completion.json', dict(passed=True, arithmetic_lock_sha256=digest,
        base_numeric_lock_sha256=A.sha(root/'lock.json'), finished_unix=completion_time,
        files={str(p.relative_to(directory)):A.sha(p) for p in directory.rglob('*') if p.is_file()}))
    contract = dict(root=str(directory), files={n:A.sha(directory/n) for n in ('lock.json', 'verification.json', 'completion.json')})
    lock = dict(source_hashes={**base['source_hashes'], **sources}, created_unix=8.)
    if changed_source:
        (A.ROOT/A.ARITHMETIC_SOURCES[0]).write_text('changed source after freeze')
    return lock, base, root, contract


def test_arithmetic_contract_binds_independent_sources_and_honest_execution_time(tmp_path, monkeypatch):
    args = arithmetic_tree(tmp_path, monkeypatch)
    result = A.verify_arithmetic(*args)
    assert result['independent_arithmetic_audit_passed'] and result['checks_reaggregated']
    assert not result['saved_arrays_recomputed_here']


@pytest.mark.parametrize('kwargs', [dict(frozen_time=3.), dict(completion_time=8.1), dict(changed_source=True)])
def test_arithmetic_contract_rejects_late_freeze_late_result_or_source_change(tmp_path, monkeypatch, kwargs):
    args = arithmetic_tree(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(ValueError, match='chronology|changed artifact'):
        A.verify_arithmetic(*args)


def first_pair_fixture():
    probability = np.full((108, 432), 1/432, dtype=np.float64)
    indices = np.linspace(0, 107, 8, dtype=np.int64)
    reference = probability[indices].copy()
    return probability, indices, reference, reference.copy(), reference.copy(), reference.copy()


def test_first_pair_replay_checks_all_eight_saved_rows_and_distinct_numeric_bounds():
    args = first_pair_fixture()
    result = A.first_pair_probability_check(*args)
    assert result['probability_rows'] == 8 and result['same_prepared_probability_error'] == 0
    assert result['actual_array_sha256'] == result['preflight_actual_array_sha256']
    bad = list(first_pair_fixture()); bad[0][bad[1][-1], :2] += [2.e-12, -2.e-12]
    with pytest.raises(ValueError, match='bound exceeded'):
        A.first_pair_probability_check(*bad)
    bad = list(first_pair_fixture()); bad[3][0, :2] += [2.e-8, -2.e-8]
    with pytest.raises(ValueError, match='bound exceeded'):
        A.first_pair_probability_check(*bad)


def test_first_pair_replay_rejects_wrong_subset_and_requires_both_independent_references():
    bad = list(first_pair_fixture()); bad[1] = np.arange(8, dtype=np.int64)
    with pytest.raises(ValueError, match='indices'):
        A.first_pair_probability_check(*bad)
    for position in (4, 5):
        bad = list(first_pair_fixture()); bad[position][0, :2] += [2.e-4, -2.e-4]
        with pytest.raises(ValueError, match='bound exceeded'):
            A.first_pair_probability_check(*bad)
