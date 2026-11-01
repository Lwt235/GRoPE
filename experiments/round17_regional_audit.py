"""R17 regional provenance, complete replay, independent metrics and references.

The producer's joint lock freezes this auditor before new regional forward.
Replay shares the production preparation/operator and is NOT an independent
forward implementation. Counts/common projection and all aggregation/selection
are independent. Numerical checks reuse the separately implemented, fully bound
preflight GL64/96 references; this audit performs no new numerical integration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from experiments import round15_regional_audit_metrics as C
from experiments import round17_feasibility_audit as H
from experiments import round17_regional_audit_metrics as M


ROOT, SCENES = H.ROOT, M.SCENES
sha, array_sha, read, write, absolute = H.sha, H.array_sha, H.read, H.write, H.absolute
GRID = tuple(dict(h=h, temperature=t) for h, t in M.GRID)
CONDITIONS = tuple(dict(scenario='partition' if (q, k) in M.PARTITION_CONDITIONS else 'scale',
                        query=q, key=k, condition=q + '_' + k) for q, k in M.CONDITIONS)
OWN_SOURCES = ('experiments/round17_regional_audit.py', 'tests/test_round17_regional_audit.py',
    'experiments/round17_regional_audit_metrics.py', 'tests/test_round17_regional_audit_metrics.py',
    'experiments/round15_regional_audit_metrics.py', 'tests/test_round15_regional_audit_metrics.py')
FIELDS = ('q_to_k_xy', 'q_to_k_pixel_cov', 'q_features', 'k_features')
METRIC_TOLERANCE, SUMMARY_TOLERANCE = 1e-8, 1e-10


def compare_tree(actual, expected, tolerance=SUMMARY_TOLERANCE, path='root'):
    """Compare metadata exactly and finite floating values within fixed atol."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('tree keys differ: ' + path)
        return max((compare_tree(actual[k], v, tolerance, path + '.' + k)
                    for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('tree list differs: ' + path)
        return max((compare_tree(a, b, tolerance, path + f'[{i}]')
                    for i, (a, b) in enumerate(zip(actual, expected))), default=0.)
    if isinstance(expected, (float, np.floating)):
        if isinstance(actual, (bool, str)) or not np.isfinite(actual) or abs(actual - expected) > tolerance:
            raise ValueError('tree float differs: ' + path)
        return float(abs(actual - expected))
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError('tree identity differs: ' + path)
    return 0.


def probability(value, shape):
    if (value.dtype != np.float64 or value.shape != tuple(shape) or not np.isfinite(value).all()
            or np.any(value < 0) or np.max(abs(value.sum(-1) - 1)) > 1e-12):
        raise ValueError('invalid probability shape/dtype/normalization')


def identity(row):
    return row['scene'], tuple(row['pair']), row['family'], row['grid_index'], row['condition']


def index_rows(rows, entries):
    expected = {(e['scene'], tuple(e['pair']), 'trace_full', i, c['condition'])
                for e in entries for i in range(12) for c in CONDITIONS}
    found = {identity(row): row for row in rows}
    if len(found) != len(rows) or set(found) != expected:
        raise ValueError('incomplete, duplicated or unauthorized regional population')
    if any(type(row['grid_index']) is not int for row in rows):
        raise ValueError('configuration indices must be integers')
    return found


def verify_record(forward, measured, entry, condition):
    if any(measured.get(k) != v for k, v in forward.items()):
        raise ValueError('evaluation changed a committed forward record')
    shape = [M.NATIVE_SIZES[condition['query']], M.NATIVE_SIZES[condition['key']]]
    if (forward['config'] != GRID[forward['grid_index']]
            or any(forward[k] != v for k, v in condition.items())
            or forward['shape'] != shape or forward['dtype'] != 'float64'
            or forward['input_sha256'] != entry['input_sha256']
            or measured['label_sha256'] != entry['label_sha256']
            or measured['replayed_array_sha256'] != forward['array_sha256']
            or measured['n_query'] != shape[0] or measured['n_key'] != shape[1]):
        raise ValueError('record configuration, replay or provenance mismatch')


def verify_stage_a_overlap(records, old_records, entries):
    expected = {(tuple(e['pair']), i, q) for e in entries for i in range(12) for q in ('g8', 'g16', 'g24')}
    old = {(tuple(row['pair']), row['grid_index'], row['query']): row for row in old_records}
    if len(old) != len(old_records) or set(old) != expected or any(r['family'] != 'trace_full' for r in old_records):
        raise ValueError('incomplete original Stage A commitments')
    seen = set()
    for row in records:
        key = tuple(row['pair']), row['grid_index'], row['query']
        overlap = row['key'] == 'g8' and row['query'] in ('g8', 'g16', 'g24')
        if overlap:
            if key not in old or key in seen:
                raise ValueError('changed Stage A overlap population')
            before = old[key]
            if (any(row[k] != before[k] for k in ('array_sha256', 'input_sha256', 'shape', 'dtype', 'config'))
                    or row.get('stage_a_array_sha256') != before['array_sha256']):
                raise ValueError('regional predictions differ from eligible Stage A operator')
            seen.add(key)
        elif 'stage_a_array_sha256' in row:
            raise ValueError('undeclared Stage A overlap')
    if seen != expected:
        raise ValueError('missing Stage A exact probability overlap')
    return len(seen)


def verify_chronology(lock, seal, forward, evaluation, scene, *, lock_sha, seal_sha, forward_sha):
    if (forward['stage'] != 'forward' or evaluation['stage'] != 'evaluation'
            or forward['scene'] != scene or evaluation['scene'] != scene
            or forward['lock_sha256'] != lock_sha or evaluation['lock_sha256'] != lock_sha
            or forward['record_count'] != 2688 or evaluation['record_count'] != 2688
            or forward['pair_count'] != 16 or forward['stage_a_exact_overlap_count'] != 576
            or forward['regional_labels_decoded'] or forward['all_probabilities_retained']
            or forward['all_probability_hashes_retained'] is not True
            or evaluation['new_regional_labels_decoded'] is not True
            or evaluation['all_replay_hashes_exact'] is not True
            or evaluation['forward_complete_sha256'] != seal_sha
            or forward['runtime'] != lock['runtime'] or evaluation['runtime'] != lock['runtime']
            or seal['forward_chain'][scene] != dict(completion_sha256=forward_sha, finished_unix=forward['finished_unix'])
            or not lock['created_unix'] <= forward['started_unix'] <= forward['finished_unix']
                <= seal['sealed_unix'] <= evaluation['started_unix'] <= evaluation['finished_unix']):
        raise ValueError('regional forward-before-label/runtime chronology mismatch')


def verify_policy(run, lock):
    for name in OWN_SOURCES + ('experiments/round17_regional.py', 'models/round17_trace_pe.py'):
        if lock['source_hashes'].get(name) != sha(ROOT / name):
            raise ValueError('audit/reference/operator missing from joint source freeze')
    expected_policy = dict(gaussian_quadrature_order=32, probability_dtype='float64',
        preparation_dtype='float32', content_bias_dtype='float32', temperature_formula_dtype='float64', floor=1e-30)
    if (lock['families'] != ['trace_full'] or lock['grid'] != list(GRID)
            or lock['conditions'] != list(CONDITIONS) or lock['numerical_policy'] != expected_policy
            or lock['field_allowlist'] != list(FIELDS) or lock['regional_label_allowlist'] != ['gt_xy', 'gt_visible']
            or lock['objective_weights'] != dict(common9=.5, direct5=.5) or lock['records'] != 8064
            or lock['full_raw_probability_bytes'] != 3074609664 or lock['fixed_sample_raw_bytes'] != 192163104
            or lock['stage_a_probabilities_reused'] or lock['exact_replay_required'] is not True
            or lock['new_regional_labels_decoded'] or lock['fresh_test_read'] or lock['validation_authorized']
            or sha(run / 'partitions.npz') != lock['partitions_sha256']):
        raise ValueError('regional frozen policy changed')
    # This is a pure metadata check, before regional labels or prediction arrays.
    M._cohort(lock['entries'])


def verify_previous(lock):
    previous = lock['stage_a']; stage_a = absolute(previous['root'])
    prior = H.checked_lock(stage_a)
    H.verify_policy(prior)
    H.verify_previous(prior)
    numerical = H.verify_numerical(prior)
    if any(lock['source_hashes'].get(n) != h for n, h in prior['source_hashes'].items()):
        raise ValueError('regional sources differ from accepted Stage A')
    for name in ('lock', 'forward_complete', 'summary', 'completion'):
        if sha(stage_a / (name + '.json')) != previous[f'feasibility_{name}_sha256']:
            raise ValueError('Stage A hash chain changed')
    completed = read(stage_a / 'completion.json')
    H.verify_files(stage_a, completed['files'])
    accepted, summary = read(stage_a / 'decision.json'), read(stage_a / 'summary.json')
    audit = read(absolute(previous['audit_path']))
    eligible = previous['eligible_indices']
    if (sha(stage_a / 'decision.json') != previous['decision_sha256']
            or sha(absolute(previous['audit_path'])) != previous['audit_sha256']
            or accepted['audit_sha256'] != previous['audit_sha256']
            or absolute(accepted['audit_path']) != absolute(previous['audit_path'])
            or audit['passed'] is not True or audit['scientific_gate_passed']
            or not eligible or eligible != sorted(set(eligible)) or any(type(i) is not int or not 0 <= i < 12 for i in eligible)
            or accepted['action'] != 'complete_full_regional_calibration'
            or accepted['primary_selected'] or accepted['scientific_gate_passed'] or accepted['validation_authorized']
            or accepted['regional_design'] != dict(new_families=1, reused_families=8, configurations_per_family=12, conditions=14, unchanged_J=True)
            or any(obj['feasible_grid_indices'] != eligible for obj in (accepted, audit, summary))
            or not completed['finished_unix'] <= audit['started_unix'] <= audit['finished_unix'] <= accepted['created_unix'] <= lock['created_unix']):
        raise ValueError('regional Stage A acceptance/eligibility chronology mismatch')
    for name in ('lock', 'forward_complete', 'summary', 'completion'):
        field = f'feasibility_{name}_sha256'
        if any(obj[field] != previous[field] for obj in (accepted, audit)):
            raise ValueError('Stage A independent acceptance binding differs')
    for name in ('inputs', 'manifests', 'training_manifest_records', 'data_root', 'data_completion_sha256',
                 'target_contract', 'numerical_contract', 'baseline_bindings'):
        if previous[name] != prior[name]:
            raise ValueError('regional changed inherited ' + name)
    if (previous['point_localization_selections'] != summary['point_localization_selections']
            or previous['legacy_point'] != summary['legacy_point']):
        raise ValueError('inherited center reference changed')
    inherited = {(v['scene'], tuple(v['pair'])): v for v in prior['training_manifest_records']}
    for entry in lock['entries']:
        path = absolute(entry['path']); row = inherited[entry['scene'], tuple(entry['pair'])]
        meta = read(path / 'manifest.json')
        if (absolute(row['path']) != path / 'manifest.json' or row['sha256'] != entry['manifest_sha256']
                or sha(path / 'manifest.json') != entry['manifest_sha256']
                or sha(path / 'inputs.npz') != entry['input_sha256']
                or meta['scene'] != entry['scene'] or meta['pair'] != entry['pair'] or meta['split'] != 'train'
                or meta['forward_uses_sensor'] or meta['input_sha256'] != entry['input_sha256']
                or meta['label_sha256'] != entry['label_sha256']):
            raise ValueError('regional observation/cohort differs from inherited metadata')
    old_summary = read(H.OLD_B / 'summary.json')
    M.validate_historical_summary(old_summary)
    return stage_a, numerical, old_summary


def verify_legacy(lock, old_summary):
    legacy = lock['legacy']; directory = absolute(legacy['root'])
    prior = read(directory / 'lock.json')
    if sha(directory / 'lock.json') != (directory / 'lock.sha256').read_text().strip():
        raise ValueError('legacy lock checksum differs')
    H.verify_files(ROOT, prior['source_hashes'])
    # R12 used flat snapshot filenames; later rounds preserve relative paths.
    for name, digest in prior['source_hashes'].items():
        if sha(directory / 'source_snapshot' / Path(name).name) != digest:
            raise ValueError('legacy source snapshot changed')
    if sha(directory / 'lock.json') != legacy['lock_sha256'] or legacy['upstream_files'] != prior['inherited']['files']:
        raise ValueError('legacy dependency chain changed')
    H.verify_files(ROOT, legacy['upstream_files'])
    family = dict(point='gaussian_point', fixed='gaussian_fixed', full='gaussian_full1',
                  full4='gaussian_full4', quad4='gaussian_quad4', quad16='gaussian_quad16')
    expected = {}
    for role in ('point', 'point_pck', 'fixed', 'full1', 'full4', 'quad4', 'quad16'):
        original = prior['configurations'][role]
        config = {n: original[n] for n in ('h', 'temperature')}
        expected[role] = dict(family=family[original['method']], grid_index=GRID.index(config), config=config)
    if (legacy['configurations'] != expected
            or legacy['uncalibrated_legacy_iso4'] != prior['configurations']['iso4']
            or expected['point']['config'] != dict(h=1., temperature=5.)
            or expected['quad16']['config'] != dict(h=.5, temperature=5.)):
        raise ValueError('legacy reference configurations changed')
    return {role: dict(**spec, summary=old_summary['metrics'][spec['family']][spec['grid_index']])
            for role, spec in expected.items()}


def literal_parts(run):
    parts = C.literal_parts()
    with np.load(run / 'partitions.npz', allow_pickle=False) as archive:
        for name, part in parts.items():
            for field in ('bounds', 'owner', 'atoms', 'mass', 'parent', 'mean', 'cov'):
                if not np.array_equal(archive[name + '__' + field], part[field]):
                    raise ValueError('saved partition disagrees with independent literal bounds')
    return parts


def compare_support(saved, actual_mean, actual_cov, actual_content, bounds, count):
    """Require the reused references to describe the identical prepared input."""
    indices = np.linspace(0, count - 1, 8, dtype=np.int64)
    if saved['indices'].dtype != np.int64 or not np.array_equal(saved['indices'], indices):
        raise ValueError('changed numerical query indices')
    for name, expected in (('actual_mean', actual_mean[indices]), ('actual_cov', actual_cov[indices]),
                           ('actual_content', actual_content[indices]), ('bounds', bounds)):
        value = saved[name]
        if value.dtype != expected.dtype or value.shape != expected.shape or not np.array_equal(value, expected):
            raise ValueError('preflight reference preparation/bounds no longer identical: ' + name)
    return indices


def reference_check(prob, retained, index):
    """Reuse frozen independent references; keep every row, tiny tails included."""
    h, t = M.GRID[index]
    arrays = [retained[f'{name}_T{t:g}'] for name in ('actual', 'reference64', 'reference96', 'hybrid')]
    for value in (prob, *arrays):
        probability(value, prob.shape)
    actual, ref64, ref96, hybrid = arrays
    same = float(np.max(abs(prob - actual)))
    error = float(np.max(abs(prob - ref96)))
    integration = float(np.max(abs(prob - hybrid)))
    convergence = float(np.max(abs(ref64 - ref96)))
    return dict(grid_index=index, config=dict(h=h, temperature=t), query_count=len(prob),
        same_prepared_actual_error=same, max_probability_error=error,
        integration_only_error=integration, reference_convergence_error=convergence,
        argmax_changes=int(np.count_nonzero(prob.argmax(-1) != ref96.argmax(-1))),
        passed=same <= 1e-12 and error <= 1e-4 and integration <= 1e-4 and convergence <= 1e-8)


def _audit(run, out, started):
    lock = H.checked_lock(run)
    verify_policy(run, lock)
    stage_a, numerical, old_summary = verify_previous(lock)
    legacy = verify_legacy(lock, old_summary)
    complete, seal, result = (read(run / name) for name in ('completion.json', 'forward_complete.json', 'summary.json'))
    H.verify_files(run, complete['files'])
    actual_files = {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()}
    if actual_files - {'completion.json', 'selection.json', 'selection.sha256'} != set(complete['files']):
        raise ValueError('changed regional completed file population')
    lock_sha, seal_sha = sha(run / 'lock.json'), sha(run / 'forward_complete.json')
    if (complete['lock_sha256'] != lock_sha or complete['summary_sha256'] != sha(run / 'summary.json')
            or complete['forward_complete_sha256'] != seal_sha or complete['all_replay_hashes_exact'] is not True
            or complete['independent_audit_required'] is not True or complete['scientific_gate_passed']
            or complete['validation_run'] or complete['finished_unix'] > started
            or seal['lock_sha256'] != lock_sha or seal['record_count'] != 8064 or seal['raw_sample_count'] != 504
            or seal['raw_sample_bytes'] != 192163104 or seal['regional_labels_decoded'] or seal['all_probabilities_retained']
            or set(seal['forward_chain']) != set(SCENES)):
        raise ValueError('regional complete/seal provenance or scope differs')
    parts = literal_parts(run)
    stages, all_saved_rows, overlap = {}, [], 0
    # All stages and hashes are checked before any regional label is decoded.
    for scene in SCENES:
        slug = scene.replace('/', '__'); forward = run / 'forward' / slug; evaluation = run / 'evaluation' / slug
        fc = H.checked_completion(forward, expected_stage='forward', scene=scene, lock_sha=lock_sha)
        ec = H.checked_completion(evaluation, expected_stage='evaluation', scene=scene, lock_sha=lock_sha)
        verify_chronology(lock, seal, fc, ec, scene, lock_sha=lock_sha, seal_sha=seal_sha,
                          forward_sha=sha(forward / 'completion.json'))
        fstart, estart = read(forward / 'started.json'), read(evaluation / 'started.json')
        if (fstart != dict(created_unix=fc['started_unix'], runtime=lock['runtime'], lock_sha256=lock_sha)
                or estart != dict(created_unix=ec['started_unix'], runtime=lock['runtime'], forward_complete_sha256=seal_sha)
                or ec['finished_unix'] > result['created_unix']):
            raise ValueError('stage start or summary chronology differs')
        entries = sorted((e for e in lock['entries'] if e['scene'] == scene), key=lambda e: Path(e['path']).name)
        if (len(entries) != 16 or lock['sample_pairs'][scene] != entries[0]['pair']
                or fc['saved_sample_pair'] != entries[0]['pair']
                or ec['label_sources'] != {str(absolute(e['path']) / 'labels.npz'): e['label_sha256'] for e in entries}):
            raise ValueError('sample selection or regional label allowlist changed')
        committed, measured = read(forward / 'records.json'), read(evaluation / 'rows.json')
        fm, em = index_rows(committed, entries), index_rows(measured, entries)
        old_records = read(stage_a / 'forward' / slug / 'records.json')
        overlap += verify_stage_a_overlap(committed, old_records, entries)
        repeat = read(forward / 'determinism.json')
        expected_repeat = {(r['condition'], r['family'], r['grid_index']): r['array_sha256']
                           for r in committed if r['pair'] == entries[0]['pair']}
        seen_repeat = {(r['condition'], r['family'], r['grid_index']): r['array_sha256'] for r in repeat['records']}
        if (repeat['passed'] is not True or repeat['labels_decoded'] or repeat['pair'] != entries[0]['pair']
                or repeat['probability_count'] != 168 or len(repeat['records']) != 168 or seen_repeat != expected_repeat):
            raise ValueError('first-pair deterministic replay changed')
        stages[scene] = (forward, entries, fm, em)
        all_saved_rows.extend(measured)
    M.validate_population(all_saved_rows, lock['entries'])
    if not result['created_unix'] <= complete['finished_unix']:
        raise ValueError('summary completed before it was created')
    out.mkdir(parents=True)
    write(out / 'lock.json', dict(created_unix=started, original_design_created_unix=lock['created_unix'],
        joint_source_lock_sha256=lock_sha, run=str(run), source_hashes={n: lock['source_hashes'][n] for n in OWN_SOURCES},
        pre_forward_design=True, shared_production_operator_replay=True, independent_forward=False,
        numerical_reference_reused=True, new_numerical_integration=False,
        metric_sum_tolerance=METRIC_TOLERANCE, summary_tolerance=SUMMARY_TOLERANCE,
        same_prepared_probability_tolerance=1e-12, probability_tolerance=1e-4, reference_tolerance=1e-8))
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')
    # Deliberately share only preparation/forward. Never call producer targets,
    # metrics, aggregation, selection, acceptance, or its source-verification API.
    from experiments import round17_regional as P
    import torch
    if P.runtime() != lock['runtime']:
        raise ValueError('audit replay runtime differs from frozen production')
    with np.load(run / 'partitions.npz', allow_pickle=False) as archive:
        saved_parts = {n: {f.split('__', 1)[1]: archive[f] for f in archive.files if f.startswith(n + '__')}
                       for n in parts}
    tensors = P.S.M.partition_tensors(saved_parts, 'cpu')
    numeric_root = absolute(numerical['root'])
    numeric_lock = H.checked_lock(numeric_root)
    if (numeric_lock['conditions'] != [list(c) for c in M.CONDITIONS]
            or numeric_lock['partitions_sha256'] != lock['partitions_sha256']):
        raise ValueError('reused numerical bounds/conditions changed')
    recomputed, checks, samples, replay_count, count_tables, max_metric = [], [], 0, 0, 0, 0.
    for scene, (forward, entries, fm, em) in stages.items():
        slug = scene.replace('/', '__')
        numeric_entry = next(e for e in numeric_lock['cohort'] if e['scene'] == scene)
        if (numeric_entry['pair'] != entries[0]['pair'] or numeric_entry['input_sha256'] != entries[0]['input_sha256']
                or absolute(numeric_entry['input_path']) != absolute(entries[0]['path']) / 'inputs.npz'):
            raise ValueError('reused numerical sample input differs')
        for pair_index, entry in enumerate(entries):
            path = absolute(entry['path'])
            with np.load(path / 'inputs.npz', allow_pickle=False) as archive:
                inputs = {field: archive[field] for field in FIELDS}
            with torch.no_grad():
                queries, keys = P.prepare_pair(inputs, tensors)
            if sha(path / 'labels.npz') != entry['label_sha256']:
                raise ValueError('regional label archive changed')
            with np.load(path / 'labels.npz', allow_pickle=False) as archive:
                labels = {field: archive[field] for field in ('gt_xy', 'gt_visible')}
            for condition in CONDITIONS:
                q, k, name = condition['query'], condition['key'], condition['condition']
                direct, common = C.independent_counts(labels, parts, q, k); count_tables += 1
                if int(direct.sum()) != int(common.sum()):
                    raise ValueError('native and common visible populations differ')
                sample_path = forward / 'samples' / path.name / (name + '.npz')
                sample = np.load(sample_path, allow_pickle=False) if pair_index == 0 else None
                try:
                    if sample is not None:
                        if set(sample.files) != {f'trace_full__c{i:02d}' for i in range(12)}:
                            raise ValueError('incomplete first-pair probability archive')
                        support_path = numeric_root / 'train' / 'arrays' / slug / (name + '_support.npz')
                        with np.load(support_path, allow_pickle=False) as support:
                            with torch.no_grad():
                                content = (queries[q]['features'] @ keys[k]['features'].T).numpy()
                            indices = compare_support(support, queries[q]['mean'].numpy(), queries[q]['cov'].numpy(),
                                content, (parts[k]['bounds'].astype(np.float64) - .5) / 8, M.NATIVE_SIZES[q])
                    with torch.no_grad():
                        for family, index, prob, diagnostics in P.condition_predictions(queries[q], keys[k]):
                            key = scene, tuple(entry['pair']), family, index, name
                            fr, er = fm[key], em[key]
                            verify_record(fr, er, entry, condition)
                            probability(prob, fr['shape'])
                            digest = array_sha(prob)
                            if digest != fr['array_sha256'] or diagnostics != fr['numerical']:
                                raise ValueError('shared operator replay is not bitwise identical')
                            replay_count += 1
                            if sample is not None:
                                field = f'trace_full__c{index:02d}'
                                retained = sample[field]
                                probability(retained, fr['shape'])
                                if (array_sha(retained) != digest or fr.get('sample_field') != field
                                        or fr.get('sample_path') != str(sample_path.relative_to(run))):
                                    raise ValueError('retained probability differs from replay/commitment')
                                h = M.GRID[index][0]
                                with np.load(numeric_root / 'train' / 'arrays' / slug / f'{name}_h{h:g}.npz', allow_pickle=False) as ref:
                                    checks.append(dict(scene=scene, pair=entry['pair'], query=q, key=k,
                                        **reference_check(retained[indices], ref, index)))
                                samples += 1
                            elif 'sample_path' in fr or 'sample_field' in fr:
                                raise ValueError('undeclared raw sample')
                            independent = C.measure(prob, parts, q, k, condition['scenario'], direct, common)
                            for metric in M.METRIC_FIELDS:
                                error = abs(independent[metric] - er[metric]); max_metric = max(max_metric, error)
                                if error > METRIC_TOLERANCE:
                                    raise ValueError('independent regional metric differs: ' + str((key, metric, error)))
                            if any(independent[n] != er[n] for n in ('mass', 'n_query', 'n_key')):
                                raise ValueError('independent native/count population differs')
                            recomputed.append(dict(er, **independent))
                finally:
                    if sample is not None:
                        sample.close()
            print('R17_REGIONAL_AUDIT', scene, pair_index + 1, '/16', flush=True)
    # Aggregate original and independently measured row sums separately: this
    # catches tiny permitted per-row error affecting selection or a closed gate.
    audited = M.summarize(all_saved_rows, old_summary, lock['stage_a']['eligible_indices'], entries=lock['entries'])
    rebuilt = M.summarize(recomputed, old_summary, lock['stage_a']['eligible_indices'], entries=lock['entries'])
    summary_error = compare_tree(result['metrics'], audited['metrics'])
    independent_metric_error = compare_tree(audited['metrics'], rebuilt['metrics'])
    compare_tree(result['selection'], audited['selection'])
    compare_tree(result['train_screen'], audited['train_screen'])
    compare_tree(audited['selection'], rebuilt['selection'])
    compare_tree(audited['train_screen'], rebuilt['train_screen'])
    compare_tree(result['legacy_references'], legacy)
    if (replay_count != 8064 or samples != 504 or count_tables != 672 or len(checks) != 504 or overlap != 1728
            or result['row_count'] != 8064 or result['pair_count'] != 48
            or result['old_baseline_summary_sha256'] != sha(H.OLD_B / 'summary.json')
            or result['lock_sha256'] != lock_sha or result['forward_complete_sha256'] != seal_sha
            or result['selection_pending_independent_audit'] is not True or result['scientific_gate_passed']
            or result['validation_run'] or result['fresh_confirmation'] or result['all_replay_hashes_exact'] is not True
            or result['stage_a_point_localization_selections'] != lock['stage_a']['point_localization_selections']
            or result['legacy_point_localization'] != lock['stage_a']['legacy_point']):
        raise ValueError('incomplete audit population or overclaimed result')
    numerical_ok = all(c['passed'] for c in checks)
    write(out / 'independent_summary.json', rebuilt)
    write(out / 'numerical_reuse_checks.json', checks)
    verification = dict(passed=numerical_ok, scientific_gate_passed=False,
        **{f'regional_{name}_sha256': sha(run / (name + '.json')) for name in ('lock', 'forward_complete', 'summary', 'completion')},
        independent_audit_lock_sha256=sha(out / 'lock.json'),
        selection=result['selection'], train_screen=result['train_screen'], train_screen_passed=audited['train_screen']['passed'],
        old_baseline_summary_sha256=sha(H.OLD_B / 'summary.json'), all_metric_rows=8064,
        saved_probability_arrays=504, shared_operator_replayed_arrays=replay_count, stage_a_exact_overlap_count=overlap,
        independently_reconstructed_count_tables=count_tables, numerical_matrices=len(checks), numerical_probability_rows=8 * len(checks),
        maximum_metric_sum_error=max_metric, maximum_summary_error=summary_error,
        maximum_recomputed_aggregate_error=independent_metric_error,
        maximum_same_prepared_actual_error=max(c['same_prepared_actual_error'] for c in checks),
        maximum_probability_error=max(c['max_probability_error'] for c in checks),
        numerical_failure_count=sum(not c['passed'] for c in checks),
        shared_production_operator_replay=True, independent_forward=False,
        independent_all_metric_recomputation=True, numerical_reference_reused=True, new_numerical_integration=False,
        numerical_contract=numerical, center_labels_decoded=False, validation_or_fresh_test_read=False,
        original_design_created_unix=lock['created_unix'], started_unix=started, finished_unix=time.time())
    write(out / 'verification.json', verification)
    write(out / 'completion.json', dict(files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()},
        passed=numerical_ok, scientific_gate_passed=False, finished_unix=time.time()))
    return verification


def audit(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists() or run == out or run in out.parents:
        raise ValueError('refusing audit output overwrite or mutation of regional artifacts')
    started = time.time()
    try:
        return _audit(run, out, started)
    except Exception as error:
        if out.exists() and not (out / 'completion.json').exists():
            write(out / 'failure.json', dict(passed=False, scientific_gate_passed=False,
                error_type=type(error).__name__, error=str(error), started_unix=started, failed_unix=time.time()))
            write(out / 'completion.json', dict(passed=False, scientific_gate_passed=False,
                files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}, finished_unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'results/round17_regional')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/round17_regional_independent_audit')
    args = parser.parse_args()
    result = audit(args.run, args.out)
    print('R17_REGIONAL_INDEPENDENT', 'PASS' if result['passed'] else 'FAIL', flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
