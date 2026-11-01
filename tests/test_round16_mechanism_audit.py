"""Synthetic integrity tests only; no old TRAIN or VAL arrays are opened."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import round16_mechanism_audit as audit


def test_auditor_imports_only_frozen_preparation_and_independent_metrics():
    tree = ast.parse(inspect.getsource(audit))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == 'experiments' for alias in node.names}
    assert imported == {'round15_feasibility', 'round16_mechanism_audit_metrics'}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden = {'pair_predictions', 'condition_predictions', 'quartic_cell_mass',
                 'tiled_cell_mass', 'forward', 'freeze'}
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in forbidden for node in calls)


def test_relative_and_absolute_commitments_cannot_escape_roots(tmp_path):
    source = tmp_path / 'source.py'; source.write_text('known source')
    assert audit.relative_path(tmp_path, 'source.py') == source
    assert audit.absolute_path(str(source)) == source
    for name in ('../source.py', str(source), './source.py'):
        with pytest.raises(ValueError):
            audit.relative_path(tmp_path, name)
    with pytest.raises(ValueError):
        audit.absolute_path('source.py')
    with pytest.raises(ValueError):
        audit.absolute_path(str(tmp_path / '..' / 'source.py'))
    outside = tmp_path.parent / (tmp_path.name + '_outside')
    outside.write_text('outside')
    (tmp_path / 'alias').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        audit.relative_path(tmp_path, 'alias')


def test_file_commitments_check_cached_digest_and_mutation(tmp_path):
    source = tmp_path / 'source.py'; source.write_text('first source')
    commitment = {'source.py': audit.sha(source)}
    cache = {}
    audit.verify_files(tmp_path, commitment, cache)
    with pytest.raises(ValueError, match='changed'):
        audit.verify_files(tmp_path, {'source.py': '0' * 64}, cache)
    source.write_text('mutated')
    with pytest.raises(ValueError, match='changed'):
        audit.verify_files(tmp_path, commitment)
    with pytest.raises(ValueError, match='empty'):
        audit.verify_files(tmp_path, {})


@pytest.mark.parametrize('relative_entry', (True, False))
@pytest.mark.parametrize('relative_commitment', (True, False))
def test_stage_a_repository_relative_entries_resolve_to_absolute_upstream_keys(
        tmp_path, monkeypatch, relative_entry, relative_commitment):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    relative = 'results/round10_partition_data/train/harvard_c3__hv_c3_1/000010_000020'
    absolute = tmp_path / relative
    absolute.mkdir(parents=True)
    source = absolute / 'inputs.npz'; source.write_bytes(b'four allowed forward fields')
    entry = {'path': relative if relative_entry else str(absolute)}
    old_key = relative + '/inputs.npz' if relative_commitment else str(source)
    prior = {'inputs': {old_key: audit.sha(source)}}
    upstream = {str(source): audit.sha(source)}
    resolved = audit.repository_path(entry['path']) / 'inputs.npz'
    assert resolved == source
    assert audit.original_input_digest(prior, resolved) == upstream[str(resolved)]
    # No eagerly evaluated relative_to(ROOT) is allowed for a valid external
    # absolute input, whose original key was also absolute.
    outside = tmp_path.parent / (tmp_path.name + '_external.npz')
    outside.write_bytes(b'external shared dependency')
    assert audit.original_input_digest({'inputs': {str(outside): audit.sha(outside)}}, outside) == audit.sha(outside)
    with pytest.raises(ValueError, match='conflicting'):
        audit.original_input_digest({'inputs': {str(source): '1' * 64,
                                               relative + '/inputs.npz': '2' * 64}}, source)


def test_complete_manifest_rejects_uncommitted_artifact_but_old_stage_a_allows_later_decision(tmp_path):
    (tmp_path / 'rows.json').write_text('[]')
    complete = {'files': {'rows.json': audit.sha(tmp_path / 'rows.json')}}
    audit.write(tmp_path / 'completion.json', complete)
    audit.verify_manifest(tmp_path, complete)
    (tmp_path / 'later_decision.json').write_text('{}')
    with pytest.raises(ValueError, match='unbound'):
        audit.verify_manifest(tmp_path, complete)
    audit.verify_manifest(tmp_path, complete, exact=False)


def test_array_comparison_preserves_nan_mask_dtype_and_exact_discrete_values():
    expected = np.array([1., np.nan, 3.], dtype=np.float64)
    actual = expected.copy(); actual[0] += 5.e-11
    assert audit.compare_array(actual, expected, 'floating') < 1.e-10
    actual[0] += 1.e-9
    with pytest.raises(ValueError, match='floating'):
        audit.compare_array(actual, expected, 'floating')
    with pytest.raises(ValueError, match='dtype'):
        audit.compare_array(expected.astype(np.float32), expected, 'dtype')
    with pytest.raises(ValueError, match='floating'):
        audit.compare_array(np.array([1., 0., 3.]), expected, 'nan mask')
    with pytest.raises(ValueError, match='discrete'):
        audit.compare_array(np.array([1, 2], np.int64), np.array([1, 3], np.int64), 'winner')
    with pytest.raises(ValueError, match='dtype'):
        audit.compare_array(np.array([0, 1], np.int8), np.array([False, True]), 'valid')


def test_count_and_derived_tolerances_do_not_inherit_additive_tolerance():
    expected = {'additive': {'n': 2, 'distance_sum': 1.e5}, 'derived': {'mean': 5.e4}}
    actual = deepcopy(expected); actual['additive']['distance_sum'] += 5.e-8
    assert audit.compare_tree(actual, expected) < 1.e-7
    actual['derived']['mean'] += 1.e-8
    with pytest.raises(ValueError, match='floating scalar'):
        audit.compare_tree(actual, expected)
    for wrong in (2., True, 3):
        actual = deepcopy(expected); actual['additive']['n'] = wrong
        with pytest.raises(ValueError, match='discrete'):
            audit.compare_tree(actual, expected)
    with pytest.raises(ValueError, match='fields'):
        audit.compare_tree({'additive': expected['additive']}, expected)


def test_json_null_and_nonfinite_scalar_are_distinguished():
    assert audit.compare_tree({'value': None}, {'value': None}) == 0.
    with pytest.raises(ValueError, match='discrete'):
        audit.compare_tree({'value': float('nan')}, {'value': None})
    with pytest.raises(ValueError, match='invalid floating'):
        audit.compare_tree(float('inf'), float('inf'))


def synthetic_entries(root):
    return [dict(scene=scene, pair=[i, i + 100],
                 path=str(root / scene.replace('/', '__') / f'{i:03d}'),
                 target_path=str(root / 'targets' / scene.replace('/', '__') / f'{i:03d}.npz'),
                 target_sha256='1' * 64)
            for scene in audit.SCENES for i in range(16)]


def synthetic_rows(entries, families):
    return [dict(scene=e['scene'], pair=e['pair'], family=family, grid_index=i,
                 query=q, config=dict(cfg))
            for e in entries for family in families for i, cfg in enumerate(audit.GRID) for q in audit.QUERIES]


def test_full_and_paired_populations_include_every_unselected_configuration(tmp_path):
    entries = synthetic_entries(tmp_path)
    audit.check_cohort(entries)
    rows = synthetic_rows(entries, audit.FAMILIES)
    assert len(audit.population(rows, entries, audit.FAMILIES)) == 5184
    paired = synthetic_rows(entries, audit.PAIRED_FAMILIES)
    assert len(audit.population(paired, entries, audit.PAIRED_FAMILIES)) == 3456
    assert sum(audit.COUNTS[r['query']] for r in rows) == 1016064
    for changed in (rows[:-1], rows + rows[:1], [r for r in rows if r['grid_index'] in (6, 9)]):
        with pytest.raises(ValueError):
            audit.population(changed, entries, audit.FAMILIES)
    wrong = deepcopy(rows); wrong[0]['config']['h'] = 1.
    with pytest.raises(ValueError, match='configuration'):
        audit.population(wrong, entries, audit.FAMILIES)


def test_cohort_rejects_new_scene_repeated_pair_or_colliding_archive_name(tmp_path):
    entries = synthetic_entries(tmp_path)
    for changed in (entries[:-1], entries + entries[:1]):
        with pytest.raises(ValueError):
            audit.check_cohort(changed)
    changed = deepcopy(entries); changed[0]['scene'] = 'new/scene'
    with pytest.raises(ValueError):
        audit.check_cohort(changed)
    changed = deepcopy(entries); changed[1]['pair'] = changed[0]['pair']
    with pytest.raises(ValueError):
        audit.check_cohort(changed)
    changed = deepcopy(entries); changed[1]['path'] = changed[0]['path']
    with pytest.raises(ValueError, match='colliding'):
        audit.check_cohort(changed)


def test_chronology_requires_joint_freeze_before_first_analysis_and_summary_after_all():
    lock = {'created_unix': 1.}
    stage = dict(started_unix=2., finished_unix=3., lock_sha256='digest')
    started = dict(created_unix=2., lock_sha256='digest')
    summary = {'created_unix': 4.}; complete = {'finished_unix': 5.}
    audit.verify_chronology(lock, stage, started, summary, complete, 6., 'digest')
    for bad_lock, bad_stage, bad_started, bad_summary, bad_complete in (
            ({'created_unix': 2.1}, stage, started, summary, complete),
            (lock, dict(stage, finished_unix=4.1), started, summary, complete),
            (lock, stage, dict(started, created_unix=2.1), summary, complete),
            (lock, stage, dict(started, lock_sha256='other'), summary, complete),
            (lock, stage, started, summary, {'finished_unix': float('nan')})):
        with pytest.raises(ValueError):
            audit.verify_chronology(bad_lock, bad_stage, bad_started, bad_summary, bad_complete, 6., 'digest')


def test_center_reference_includes_exact_eight_pixel_boundary_and_ignores_invalid_centers():
    yy, xx = np.meshgrid(np.arange(18) * 8 + 3.5, np.arange(24) * 8 + 3.5, indexing='ij')
    keys = np.column_stack((xx.ravel(), yy.ravel()))
    p = np.full((3, 432), 1.e-6, np.float64)
    p[:, 0] = 1. - 431.e-6
    target = np.array([[11.5, 3.5], [11.50001, 3.5], [np.nan, np.nan]])
    valid = np.array([True, True, False])
    winners, result = audit.reference_center(p, keys, target, valid)
    assert winners.tolist() == [0, 0, 0]
    assert result['pck_count'] == 1 and result['n_point'] == 2 and result['n_query'] == 3
    assert result['argmax_sha256'] == audit.array_sha(winners)
    p[0, 1] = p[0, 0]; p[0] /= p[0].sum()
    assert audit.reference_center(p, keys, target, valid)[0][0] == 0
    with pytest.raises(ValueError, match='probability'):
        audit.reference_center(p.astype(np.float32), keys, target, valid)


def test_temperature_identity_rounds_float32_multiplication_before_promotion():
    p0 = np.array([[.2, .3, .5]], dtype=np.float64)
    content = np.array([[.12345671, .65432173, -.33333331]], dtype=np.float32)
    logits = np.log(p0) + (np.float32(5.) * content).astype(np.float64)
    p = np.exp(logits - logits.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
    assert audit.temperature_identity(p0, p, content, 5.) == 0.
    logits64 = np.log(p0) + 5. * content.astype(np.float64)
    wrong = np.exp(logits64 - logits64.max(axis=1, keepdims=True)); wrong /= wrong.sum(axis=1, keepdims=True)
    with pytest.raises(ValueError, match='temperature'):
        audit.temperature_identity(p0, wrong, content, 5.)


def test_contract_refuses_missing_joint_auditor_source_before_data_access(tmp_path, monkeypatch):
    run = tmp_path / 'run'; run.mkdir()
    lock = dict(joint_producer_and_independent_audit_freeze=True, diagnostic_only=True,
        new_predictions=False, new_data=False, scientific_gate_passed=False,
        expected_rows=5184, expected_paired_rows=3456, expected_query_states=1016064,
        expected_static_queries=28224, fields=list(audit.A.FIELDS), runtime={'synthetic': True}, source_hashes={})
    audit.write(run / 'lock.json', lock)
    (run / 'lock.sha256').write_text(audit.sha(run / 'lock.json') + '\n')
    audit.write(run / 'summary.json', {})
    audit.write(run / 'completion.json', {})
    monkeypatch.setattr(audit.A, 'configure_runtime', lambda device: {'synthetic': True})
    with pytest.raises(ValueError, match='omitted required'):
        audit.verify_contract(run, 6.)


def test_exclusive_writes_and_existing_audit_cannot_overwrite_evidence(tmp_path):
    path = tmp_path / 'evidence.json'
    audit.write(path, {'passed': False})
    with pytest.raises(FileExistsError):
        audit.write(path, {'passed': True})
    assert json.loads(path.read_text()) == {'passed': False}
    # This guard precedes opening any real input or diagnostic file.
    with pytest.raises(ValueError, match='refusing audit overwrite'):
        audit.audit(tmp_path / 'uncreated_run', tmp_path)
