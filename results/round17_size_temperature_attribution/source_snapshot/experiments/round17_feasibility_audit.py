"""Independent R17 center-feasibility audit. No producer/model/selector imports.

The canonical audit table uses integer correct/valid counts for every pair,
family, global h/T configuration, scene and query scale. The joint producer source lock freezes this file and its tests before new
forward; the audit run records its honest later execution time.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
QUERIES = ('g8', 'g16', 'g24')
FAMILIES = ('trace_full', 'quartic_point', 'gaussian_point')
GRID = tuple((h, t) for h in (.25, .5, 1., 2.) for t in (0., 5., 10.))
ROOT = Path(__file__).resolve().parents[1]


def independent_center_selection(rows, *, expected_pairs=16):
    """Equal-scene PCK, one global point configuration, all three protections.

    No KL or regional winner selection occurs at this stage. A legacy point
    configuration is included even if either newly selected point is weaker
    at a particular scale. Comparison tolerance is the declared .005 only.
    """
    groups = defaultdict(list)
    populations = {}
    identities = set()
    for row in rows:
        family, cfg = row['family'], (float(row['h']), float(row['temperature']))
        scene, query, pair = row['scene'], row['query'], tuple(row['pair'])
        if family not in FAMILIES or cfg not in GRID or scene not in SCENES or query not in QUERIES:
            raise ValueError('unexpected family/configuration/cohort')
        good, total = row['correct'], row['valid']
        if (not isinstance(good, int) or not isinstance(total, int)
                or isinstance(good, bool) or isinstance(total, bool) or not 0 <= good <= total):
            raise ValueError('invalid integer counts')
        identity = (family, cfg, scene, query, pair)
        if identity in identities:
            raise ValueError('duplicate metric row')
        identities.add(identity)
        population_key = (scene, query, pair)
        if population_key in populations and populations[population_key] != total:
            raise ValueError('changed center population')
        populations[population_key] = total
        groups[family, cfg, scene, query].append((pair, good, total))

    scores = {}
    for family in FAMILIES:
        for cfg in GRID:
            by_query = {}
            by_scene = {s: {} for s in SCENES}
            for query in QUERIES:
                for scene in SCENES:
                    group = groups[family, cfg, scene, query]
                    if len(group) != expected_pairs:
                        raise ValueError('missing pair population')
                    reference_pairs = {p for s, q, p in populations if s == scene and q == query}
                    if {p for p, _, _ in group} != reference_pairs:
                        raise ValueError('pair identity mismatch')
                    denominator = sum(v for _, _, v in group)
                    if denominator <= 0:
                        raise ValueError('empty center population')
                    by_scene[scene][query] = Fraction(sum(g for _, g, _ in group), denominator)
                by_query[query] = sum(by_scene[s][query] for s in SCENES) / len(SCENES)
            scores[family, cfg] = dict(by_query=by_query, by_scene=by_scene,
                                       macro=sum(by_query.values()) / len(QUERIES))

    points = {}
    for family in ('gaussian_point', 'quartic_point'):
        # Python max preserves the declared GRID order for exact ties.
        cfg = max(GRID, key=lambda c: scores[family, c]['macro'])
        points[family] = cfg
    references = (
        ('selected_gaussian_point', 'gaussian_point', points['gaussian_point']),
        ('selected_quartic_point', 'quartic_point', points['quartic_point']),
        ('legacy_point_pck', 'gaussian_point', (.5, 5.)),
    )
    eligible, checks = [], []
    for cfg in GRID:
        all_pass = True
        for query in QUERIES:
            candidate = scores['trace_full', cfg]['by_query'][query]
            for name, family, reference_cfg in references:
                reference = scores[family, reference_cfg]['by_query'][query]
                difference = candidate - reference
                passed = difference >= -Fraction(1, 200)
                checks.append(dict(h=cfg[0], temperature=cfg[1], query=query,
                                   reference=name, difference_pp=float(100*difference), passed=passed))
                all_pass = all_pass and passed
        if all_pass:
            eligible.append(dict(h=cfg[0], temperature=cfg[1]))
    return dict(scores=scores, point_configurations=points, checks=checks,
                eligible=eligible, stop_candidate=len(eligible) == 0,
                regional_primary_selected=False, scientific_gate_passed=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def verify_files(directory, files):
    for name, digest in files.items():
        if sha(Path(directory)/name) != digest:
            raise ValueError('changed artifact: '+name)


def verify_constraint_table(saved, expected, eligible):
    """Require all 108 distinct guards, including failed configurations."""
    reference_names = {'gaussian_point': 'selected_gaussian_point',
                       'quartic_point': 'selected_quartic_point',
                       'legacy_point_pck': 'legacy_point_pck'}
    expected_checks = {(GRID.index((c['h'], c['temperature'])), c['query'], c['reference']): c
                       for c in expected}
    if len(saved) != 12 or {v['grid_index'] for v in saved} != set(range(12)):
        raise ValueError('missing or duplicate configuration guards')
    seen = set()
    for item in saved:
        i = item['grid_index']; h, temperature = GRID[i]
        if (item['config'] != dict(h=h, temperature=temperature)
                or item['feasible'] is not (i in eligible) or len(item['checks']) != 9):
            raise ValueError('invalid configuration guards')
        for c in item['checks']:
            identity = (i, c['query'], reference_names[c['reference']])
            if identity in seen or identity not in expected_checks:
                raise ValueError('duplicate or unexpected constraint')
            seen.add(identity)
            target = expected_checks[identity]
            if (c['passed'] is not target['passed'] or not np.isfinite(c['difference_pp'])
                    or abs(c['difference_pp']-target['difference_pp']) > 1e-10):
                raise ValueError('center constraint differs')
    if seen != set(expected_checks) or len(seen) != 108:
        raise ValueError('incomplete constraint coverage')


SOURCE_NAMES = ('experiments/round17_feasibility_audit.py', 'tests/test_round17_feasibility_audit.py')
MODEL = 'models/round17_trace_pe.py'
COUNTS = dict(g8=432, g16=108, g24=48)
FIELDS = ('q_to_k_xy', 'q_to_k_pixel_cov', 'q_features', 'k_features')
OLD_A = ROOT/'results/round15_feasibility'
OLD_B = ROOT/'results/round15_regional'
ARITHMETIC_SOURCES = ('docs/round17_arithmetic_addendum.md',
                      'experiments/round17_arithmetic_audit.py',
                      'tests/test_round17_arithmetic_audit.py')
NUMERICAL_CONDITIONS = tuple((q, k) for q in ('g8', 'g16', 'mixed')
                            for k in ('g8', 'g16', 'mixed')) + (
    ('g8', 'g24'), ('g24', 'g8'), ('g16', 'g24'), ('g24', 'g16'), ('g24', 'g24'))


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def verify_sources(directory, lock):
    verify_files(ROOT, lock['source_hashes'])
    verify_files(Path(directory)/'source_snapshot', lock['source_hashes'])


def checked_lock(directory):
    directory = Path(directory)
    lock = read(directory/'lock.json')
    if sha(directory/'lock.json') != (directory/'lock.sha256').read_text().strip():
        raise ValueError('lock checksum mismatch')
    verify_sources(directory, lock)
    return lock


def checked_completion(directory, *, expected_stage=None, scene=None, lock_sha=None):
    directory = Path(directory)
    value = read(directory/'completion.json')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p.name != 'completion.json'}
    if set(value['files']) != actual:
        raise ValueError('missing or extra completion artifact')
    verify_files(directory, value['files'])
    if (expected_stage is not None and value['stage'] != expected_stage
            or scene is not None and value['scene'] != scene
            or lock_sha is not None and value['lock_sha256'] != lock_sha):
        raise ValueError('completion identity mismatch')
    if value['started_unix'] > value['finished_unix']:
        raise ValueError('completion time reversed')
    return value


def validate_cohort(entries, *, pairs_per_scene=16):
    identities = [(e['scene'], tuple(e['pair'])) for e in entries]
    if (len(identities) != len(set(identities)) or len(entries) != 3*pairs_per_scene
            or {e['scene'] for e in entries} != set(SCENES)
            or any(sum(e['scene'] == s for e in entries) != pairs_per_scene for s in SCENES)):
        raise ValueError('changed complete old TRAIN cohort')
    if len({e['path'] for e in entries}) != len(entries) or len({e['target_path'] for e in entries}) != len(entries):
        raise ValueError('duplicated observation or target archive')


def verify_policy(lock):
    if (lock['grid'] != [dict(h=h, temperature=t) for h, t in GRID]
            or lock['families'] != ['trace_full']
            or lock['reused_point_families'] != ['quartic_point', 'gaussian_point']
            or lock['queries'] != list(QUERIES) or lock['probability_arrays'] != 1728
            or lock['probability_rows'] != 48*12*sum(COUNTS.values())
            or lock['raw_probability_bytes'] != 48*12*sum(COUNTS.values())*432*8
            or lock['pck_tolerance_exact'] != [1, 200] or lock['legacy_point_index'] != 4
            or lock['field_allowlist'] != list(FIELDS)
            or lock['all_final_probabilities_retained'] is not True
            or lock['all_forward_before_target_decode'] is not True
            or lock['center_labels_decoded'] or lock['regional_labels_decoded'] or lock['fresh_test_read']):
        raise ValueError('changed declared Stage A policy')
    for source in (*SOURCE_NAMES, MODEL, 'experiments/round17_feasibility.py'):
        if lock['source_hashes'].get(source) != sha(ROOT/source):
            raise ValueError('independent auditor/model absent from joint design freeze')
    validate_cohort(lock['target_contract']['entries'])


def verify_arithmetic_rows(rows, cohort, train_files, counts):
    """Audit all saved arithmetic checks without claiming an integration rerun."""
    pairs = {e['scene']: tuple(e['pair']) for e in cohort}
    expected = {(s, q, k, h, t) for s in SCENES for q, k in NUMERICAL_CONDITIONS for h, t in GRID}
    seen, support_archives, geometry_archives = set(), set(), set()
    shared_hashes, maximum, argmax_changes = {}, 0., 0
    hash_names = {'actual_raw', 'actual_cov', 'actual_content', 'bounds', 'saved_probability',
                  'same_mass_reference_probability', 'actual_tau32_recomputed', 'reference_tau32',
                  'actual_bias32_recomputed', 'reference_bias32'}
    for row in rows:
        s, q, k, h, t = (row[x] for x in ('scene', 'query', 'key', 'h', 'temperature'))
        identity = (s, q, k, h, t)
        if identity not in expected or identity in seen or tuple(row['pair']) != pairs[s]:
            raise ValueError('arithmetic matrix identity/population differs')
        seen.add(identity)
        prefix = 'arrays/'+s.replace('/', '__')+'/'
        support = prefix+f'{q}_{k}_support.npz'
        geometry = prefix+f'{q}_{k}_h{h:g}.npz'
        if (row['support_archive'] != support or row['geometry_archive'] != geometry
                or row['support_archive_sha256'] != train_files.get(support)
                or row['geometry_archive_sha256'] != train_files.get(geometry)
                or row['queries'] != 8 or row['keys'] != counts[k]
                or row['query_indices'] != np.linspace(0, counts[q]-1, 8, dtype=np.int64).tolist()):
            raise ValueError('arithmetic saved input/archive binding differs')
        support_archives.add(support); geometry_archives.add(geometry)
        errors = np.asarray(row['per_query_probability_error'], dtype=np.float64)
        tau = np.asarray(row['actual_tau32'], dtype=np.float64)
        reference_tau = np.asarray(row['reference_tau32'], dtype=np.float64)
        error = row['max_same_mass_probability_error']
        if (row['passed'] is not True or row['tau32_exact'] is not True or row['bias32_exact'] is not True
                or errors.shape != (8,) or not np.isfinite(errors).all() or np.any(errors < 0)
                or not np.isfinite(error) or error != float(errors.max()) or error > 1.e-12
                or tau.shape != (8,) or reference_tau.shape != (8,)
                or not np.isfinite(tau).all() or not np.array_equal(tau, reference_tau)
                or row['max_tau32_error'] != 0 or row['max_bias32_error'] != 0
                or row['tau_mismatch_rows'] or row['bias_mismatch_rows']):
            raise ValueError('arithmetic exactness/probability check failed')
        changed = row['argmax_mismatch_rows']
        if (len(changed) != len(set(changed))
                or any(type(i) is not int or not 0 <= i < 8 for i in changed)
                or type(row['raw_zero_fov_rows']) is not int or type(row['raw_tiny_fov_rows']) is not int
                or not 0 <= row['raw_zero_fov_rows'] <= row['raw_tiny_fov_rows'] <= 8):
            raise ValueError('arithmetic diagnostic population differs')
        hashes = row['array_hashes']
        if (set(hashes) != hash_names
                or any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                       for v in hashes.values())
                or hashes['actual_tau32_recomputed'] != hashes['reference_tau32']
                or hashes['actual_bias32_recomputed'] != hashes['reference_bias32']):
            raise ValueError('arithmetic exact array hash differs')
        for name in ('actual_cov', 'actual_content', 'bounds', 'actual_raw'):
            key = (geometry if name == 'actual_raw' else support, name)
            if key in shared_hashes and shared_hashes[key] != hashes[name]:
                raise ValueError('arithmetic shared saved input changed across configurations')
            shared_hashes[key] = hashes[name]
        maximum = max(maximum, error); argmax_changes += len(changed)
    if (seen != expected or len(rows) != 504 or len(support_archives) != 42 or len(geometry_archives) != 168
            or {n for n in train_files if n.endswith('.npz')} != support_archives | geometry_archives):
        raise ValueError('incomplete arithmetic matrix/archive population')
    return dict(support_archives=42, geometry_archives=168, probability_matrices=504,
                probability_rows=4032, max_same_mass_probability_error=maximum,
                argmax_mismatch_rows=argmax_changes)


def verify_arithmetic(lock, numerical, root, contract):
    """Bind the independently frozen same-prepared arithmetic extension."""
    directory = absolute(contract['root'])
    if set(contract['files']) != {'lock.json', 'verification.json', 'completion.json'}:
        raise ValueError('incomplete mandatory arithmetic contract')
    verify_files(directory, contract['files'])
    frozen, report, completion = (read(directory/name) for name in ('lock.json', 'verification.json', 'completion.json'))
    digest, base_digest = sha(directory/'lock.json'), sha(root/'lock.json')
    if (directory/'lock.sha256').read_text().strip() != digest:
        raise ValueError('arithmetic lock checksum differs')
    actual_files = {str(p.relative_to(directory)) for p in directory.rglob('*')
                    if p.is_file() and p != directory/'completion.json'}
    if set(completion['files']) != actual_files:
        raise ValueError('incomplete arithmetic completion artifacts')
    verify_files(directory, completion['files'])
    sources = frozen['new_source_hashes']
    if (set(sources) != set(ARITHMETIC_SOURCES)
            or any(lock['source_hashes'].get(n) != h for n, h in sources.items())
            or frozen['base_source_hashes'] != numerical['source_hashes']
            or absolute(frozen['run_root']) != root
            or frozen['base_numeric_lock_sha256'] != base_digest
            or frozen['train_absent_at_freeze'] is not True
            or frozen['same_mass_tolerance'] != 1.e-12
            or frozen['original_observation_inputs_read'] or frozen['labels_read']):
        raise ValueError('arithmetic source/pre-freeze contract differs')
    verify_files(ROOT, sources); verify_files(directory/'source_snapshot', sources)
    train = read(root/'train'/'completion.json')
    started = read(directory/'audit_started.json')
    if (report['passed'] is not True or completion['passed'] is not True
            or report['arithmetic_lock_sha256'] != digest or completion['arithmetic_lock_sha256'] != digest
            or report['base_numeric_lock_sha256'] != base_digest or completion['base_numeric_lock_sha256'] != base_digest
            or report['train_summary_sha256'] != sha(root/'train'/'summary.json')
            or report['train_completion_sha256'] != sha(root/'train'/'completion.json')
            or report['source_hashes'] != sources or report['base_source_hashes'] != numerical['source_hashes']
            or report['arithmetic_frozen_unix'] != frozen['created_unix']
            or report['train_started_unix'] != train['started_unix']
            or started['arithmetic_lock_sha256'] != digest or started['started_unix'] != report['started_unix']
            or not numerical['created_unix'] <= frozen['created_unix'] < train['started_unix']
                    <= train['finished_unix'] <= report['started_unix'] <= report['finished_unix']
                    <= completion['finished_unix'] <= lock['created_unix']):
        raise ValueError('arithmetic source/result chronology differs')
    if (report['tau32_exact'] is not True or report['bias32_exact'] is not True
            or report['same_mass_tolerance'] != 1.e-12 or report['failed_matrix_indices']
            or report['actual_tau_bias_recomputed_from_locked_function'] is not True
            or report['original_tau_bias_arrays_were_not_saved'] is not True
            or any(report[name] for name in ('original_observation_inputs_read', 'labels_read',
                                            'validation_or_test_read', 'new_real_predictions', 'scientific_gate_passed'))):
        raise ValueError('arithmetic exactness or scope declaration differs')
    with np.load(root/'partitions.npz', allow_pickle=False) as archive:
        counts = {q: len(archive[q+'__bounds']) for q in ('g8', 'g16', 'mixed', 'g24')}
    totals = verify_arithmetic_rows(read(directory/'matrix_rows.json'), numerical['cohort'], train['files'], counts)
    if any(report.get(k) != v for k, v in totals.items()) or any(frozen.get(k) != totals[k]
            for k in ('support_archives', 'geometry_archives', 'probability_matrices', 'probability_rows')):
        raise ValueError('arithmetic aggregate differs from all saved matrix checks')
    return dict(root=str(directory), files=contract['files'], **totals,
                source_identical=True, independent_arithmetic_audit_passed=True,
                checks_reaggregated=True, saved_arrays_recomputed_here=False)


def verify_numerical(lock):
    contract = lock['numerical_contract']
    root = absolute(contract['root'])
    expected = {'lock.json', 'synthetic/summary.json', 'synthetic/completion.json',
                'train/summary.json', 'train/completion.json'}
    if set(contract['files']) != expected:
        raise ValueError('incomplete numerical contract')
    verify_files(root, contract['files'])
    numerical = checked_lock(root)
    if any(lock['source_hashes'].get(name) != digest for name, digest in numerical['source_hashes'].items()):
        raise ValueError('Stage A source differs from passing numerical source')
    if (numerical['reference_orders'] != [64, 96] or numerical['production_order'] != 32
            or numerical['probability_tolerance'] != 1.e-4 or numerical['reference_tolerance'] != 1.e-8
            or numerical['floor'] != 1.e-30 or numerical['fields'] != list(FIELDS)
            or numerical['grid'] != lock['grid'] or numerical['probability_rows'] != 4032
            or numerical['geometry_rows'] != 1344 or numerical['cpu_threads'] != 4
            or numerical['labels_read'] or numerical['validation_or_test_read']):
        raise ValueError('numerical policy changed')
    if sha(root/'partitions.npz') != numerical['partitions_sha256']:
        raise ValueError('numerical partitions changed')
    verify_files(ROOT, numerical['evidence'])
    previous_finish = numerical['created_unix']
    for stage in ('synthetic', 'train'):
        directory = root/stage
        complete = checked_completion(directory, lock_sha=sha(root/'lock.json'))
        result = read(directory/'summary.json')
        started = read(directory/'started.json')
        if (complete['passed'] is not True or result['passed'] is not True
                or result['lock_sha256'] != sha(root/'lock.json')
                or result['labels_read'] or result['validation_or_test_read']
                or result['scientific_gate_passed']
                or started['lock_sha256'] != sha(root/'lock.json')
                or started['started_unix'] != complete['started_unix']
                or not previous_finish <= complete['started_unix'] <= complete['finished_unix'] <= lock['created_unix']):
            raise ValueError('numerical passing/source chronology mismatch')
        if (result['max_probability_error'] > 1.e-4
                or result['max_reference_convergence_error'] > 1.e-8
                or stage == 'synthetic' and result['max_same_mass_probability_error'] > 1.e-12
                or stage == 'train' and result['max_integration_only_error'] > 1.e-4):
            raise ValueError('passing numerical stage exceeds declared bounds')
        previous_finish = complete['finished_unix']
        if stage == 'train' and (result['probability_rows'] != 4032 or result['geometry_rows'] != 1344
                                or result['support_sets'] != 42 or result['failures']):
            raise ValueError('incomplete real TRAIN numerical population')
    entries = lock['target_contract']['entries']
    if len(numerical['cohort']) != 3 or {e['scene'] for e in numerical['cohort']} != set(SCENES):
        raise ValueError('numerical cohort is not three first TRAIN pairs')
    for item in numerical['cohort']:
        first = min((e for e in entries if e['scene'] == item['scene']), key=lambda e: e['path'])
        if (item['pair'] != first['pair'] or absolute(item['input_path']) != absolute(first['path'])/'inputs.npz'
                or item['input_sha256'] != lock['inputs'][str(Path(first['path'])/'inputs.npz')]
                or sha(absolute(item['manifest_path'])) != item['manifest_sha256']):
            raise ValueError('numerical first-pair identity differs')
    arithmetic = verify_arithmetic(lock, numerical, root, contract['arithmetic'])
    return dict(root=str(root), files=contract['files'], arithmetic=arithmetic, slow_reference_rerun=False,
                source_identical=True, synthetic_passed=True, train_passed=True)


def first_pair_probability_check(probability, indices, actual, reference64, reference96, hybrid):
    """Verify the Stage A driver on the already integrated numerical rows.

    These are numerical tolerances only; the center feasibility selector still
    uses exact integer fractions and its unchanged one-half percentage point.
    """
    p = np.asarray(probability)
    idx = np.asarray(indices)
    if (p.dtype != np.float64 or p.ndim != 2 or p.shape[1] != 432
            or idx.dtype != np.int64 or not np.array_equal(idx, np.linspace(0, len(p)-1, 8, dtype=np.int64))):
        raise ValueError('first-pair numerical query/key indices differ')
    candidate = p[idx]
    arrays = [np.asarray(v) for v in (candidate, actual, reference64, reference96, hybrid)]
    if any(v.dtype != np.float64 or v.shape != (8, 432) or not np.isfinite(v).all()
           or np.any(v < 0) or np.max(abs(v.sum(1)-1)) > 1.e-12 for v in arrays):
        raise ValueError('first-pair numerical probability schema differs')
    values = dict(same_prepared_probability_error=float(np.max(abs(candidate-actual))),
                  reference_probability_error=float(np.max(abs(candidate-reference96))),
                  hybrid_probability_error=float(np.max(abs(candidate-hybrid))),
                  reference_convergence_error=float(np.max(abs(reference64-reference96))))
    if (values['same_prepared_probability_error'] > 1.e-12
            or values['reference_probability_error'] > 1.e-4
            or values['hybrid_probability_error'] > 1.e-4
            or values['reference_convergence_error'] > 1.e-8):
        raise ValueError('first-pair numerical probability bound exceeded')
    return dict(**values, query_indices=idx.tolist(), probability_rows=8,
                same_prepared_argmax_changes=int(np.count_nonzero(candidate.argmax(1) != actual.argmax(1))),
                reference_argmax_changes=int(np.count_nonzero(candidate.argmax(1) != reference96.argmax(1))),
                actual_array_sha256=array_sha(candidate), preflight_actual_array_sha256=array_sha(actual))


def verify_previous(lock):
    """Only old accepted metadata/hashes; do not decode label arrays here."""
    verify_files(ROOT, lock['baseline_bindings'])
    old = checked_lock(OLD_A)
    for name in ('target_contract', 'inputs', 'manifests', 'training_manifest_records',
                 'data_root', 'data_completion_sha256', 'partitions_sha256'):
        if lock[name] != old[name]:
            raise ValueError('new/old cohort, input, label or partition contract differs')
    verify_files(ROOT, lock['inputs'])
    verify_files(ROOT, lock['manifests'])
    if sha(absolute(lock['data_root'])/'completion.json') != lock['data_completion_sha256']:
        raise ValueError('observation data completion changed')
    for record in lock['training_manifest_records']:
        if record['split'] != 'train' or record['scene'] not in SCENES:
            raise ValueError('forbidden observation split')
        meta = read(absolute(record['path']))
        if (sha(absolute(record['path'])) != record['sha256'] or meta['split'] != 'train'
                or meta['forward_uses_sensor'] or meta['scene'] != record['scene'] or meta['pair'] != record['pair']):
            raise ValueError('invalid observation metadata')
    for base, prefix, acceptance in ((OLD_A, 'feasibility', 'decision'), (OLD_B, 'regional', 'selection')):
        checked_lock(base)
        completion = read(base/'completion.json')
        for name, digest in completion['files'].items():
            if name.endswith('.json'):
                relative = str((base/name).relative_to(ROOT))
                if lock['baseline_bindings'].get(relative) != digest or sha(base/name) != digest:
                    raise ValueError('old table not bound to completed accepted baseline')
        for name in ('lock', 'summary', 'forward_complete'):
            if completion[name+'_sha256'] != sha(base/(name+'.json')):
                raise ValueError('old completion binding changed')
        accepted = read(base/(acceptance+'.json'))
        audit_path = absolute(accepted['audit_path'])
        if sha(audit_path) != accepted['audit_sha256']:
            raise ValueError('old accepted audit hash changed')
        verification = read(audit_path)
        for name in ('lock', 'summary', 'completion', 'forward_complete'):
            field = prefix+'_'+name+'_sha256'
            if accepted[field] != sha(base/(name+'.json')) or verification[field] != accepted[field]:
                raise ValueError('old independent audit chain changed')
        if verification['passed'] is not True:
            raise ValueError('old controls not independently accepted')
    target = lock['target_contract']; root = absolute(target['root'])
    if (sha(root/'lock.json') != target['lock_sha256']
            or sha(root/'completion.json') != target['completion_sha256']
            or sha(absolute(target['audit_path'])) != target['audit_sha256']):
        raise ValueError('target metadata chain changed')
    target_audit = read(absolute(target['audit_path']))
    if (target_audit['passed'] is not True or target_audit['target_lock_sha256'] != target['lock_sha256']
            or target_audit['target_completion_sha256'] != target['completion_sha256']):
        raise ValueError('independent target audit binding differs')


def verify_chronology(lock, seal, forward, evaluation, scene, *, lock_sha, seal_sha, forward_sha):
    if (forward['stage'] != 'forward' or evaluation['stage'] != 'evaluation'
            or forward['scene'] != scene or evaluation['scene'] != scene
            or forward['lock_sha256'] != lock_sha or evaluation['lock_sha256'] != lock_sha
            or forward['record_count'] != 576 or forward['pair_count'] != 16
            or forward['labels_decoded'] or forward['all_probabilities_retained'] is not True
            or evaluation['record_count'] != 576 or evaluation['center_labels_decoded'] is not True
            or evaluation['regional_labels_decoded'] or evaluation['forward_complete_sha256'] != seal_sha
            or seal['lock_sha256'] != lock_sha or seal['array_count'] != 1728 or seal['labels_decoded']
            or seal['all_probabilities_retained'] is not True
            or seal['forward_chain'][scene] != dict(completion_sha256=forward_sha, finished_unix=forward['finished_unix'])
            or not lock['created_unix'] <= forward['started_unix'] <= forward['finished_unix']
                <= seal['sealed_unix'] <= evaluation['started_unix'] <= evaluation['finished_unix']):
        raise ValueError('forward-before-target chronology/provenance mismatch')
    runtime = forward['runtime']
    if (runtime['device'] != 'cpu' or runtime['device_name'] != 'cpu' or runtime['cpu_threads'] != 4
            or runtime['deterministic_algorithms'] is not True or runtime['tf32'] is not False
            or runtime['numpy_version'] != lock['numpy_version'] or runtime['torch_version'] != lock['torch_version']):
        raise ValueError('changed execution precision/runtime')


def indexed_rows(rows, entries, families):
    expected = {(tuple(e['pair']), f, i, q) for e in entries for f in families for i in range(12) for q in QUERIES}
    index = {(tuple(r['pair']), r['family'], r['grid_index'], r['query']): r for r in rows}
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError('missing, duplicated or changed record population')
    scene = entries[0]['scene']
    if any(row['scene'] != scene for row in rows):
        raise ValueError('record scene mismatch')
    if any(type(row['grid_index']) is not int for row in rows):
        raise ValueError('record configuration index must be integer')
    return index


def center_counts(probability, point, valid):
    """Independent fixed native-g8 readout; never use producer metric helpers."""
    if (probability.ndim != 2 or probability.shape[1] != 432 or probability.dtype != np.float64
            or not np.isfinite(probability).all() or np.any(probability < 0)):
        raise ValueError('invalid native probabilities')
    error = float(np.max(np.abs(probability.sum(-1)-1.)))
    n = len(probability)
    if (error > 1.e-12 or point.shape != (n, 2) or valid.shape != (n,)
            or valid.dtype != np.bool_ or not np.isfinite(point[valid]).all()):
        raise ValueError('invalid probability normalization/center labels')
    winner = np.argmax(probability, axis=1)
    centers = np.column_stack(((winner % 24)*8+3.5, (winner // 24)*8+3.5))
    delta = centers[valid] - point[valid]
    correct = int(np.count_nonzero(delta[:, 0]**2 + delta[:, 1]**2 <= 64.))
    return dict(correct=correct, valid=int(valid.sum()), argmax_sha256=array_sha(winner),
                n_query=n, normalization_error=error)


def verify_record(record, forward, entry, probability, counts, archive, digest, input_sha):
    if any(record.get(k) != v for k, v in forward.items()):
        raise ValueError('evaluation changed a forward commitment')
    index, family, query = record['grid_index'], record['family'], record['query']
    h, t = GRID[index]
    expected = dict(scene=entry['scene'], pair=entry['pair'], config=dict(h=h, temperature=t),
        archive=archive, archive_sha256=digest, field=f'{family}__c{index:02d}__{query}',
        shape=[COUNTS[query], 432], dtype='float64', array_sha256=array_sha(probability),
        input_sha256=input_sha, target_sha256=entry['target_sha256'],
        pck_count=counts['correct'], n_point=counts['valid'], n_query=COUNTS[query],
        argmax_sha256=counts['argmax_sha256'])
    if (any(record.get(k) != v for k, v in expected.items()) or probability.shape != (COUNTS[query], 432)
            or any(type(record[k]) is not int for k in ('pck_count', 'n_point', 'n_query'))):
        raise ValueError('native probability/center/provenance mismatch')


def verify_summary(result, rows, *, expected_pairs=16):
    independent = independent_center_selection(rows, expected_pairs=expected_pairs)
    errors = []
    if set(result['metrics']) != set(FAMILIES):
        raise ValueError('summary family population changed')
    for family in FAMILIES:
        values = result['metrics'][family]
        if len(values) != 12 or [v['grid_index'] for v in values] != list(range(12)):
            raise ValueError('summary configuration population changed')
        for i, cfg in enumerate(GRID):
            saved, score = values[i], independent['scores'][family, cfg]
            if saved['config'] != dict(h=cfg[0], temperature=cfg[1]):
                raise ValueError('summary configuration identity mismatch')
            errors.append(abs(saved['equal_scene_equal_scale_pck8'] - float(score['macro'])))
            if set(saved['by_query']) != set(QUERIES) or set(saved['by_scene_query']) != set(SCENES):
                raise ValueError('summary scene/query population changed')
            for q in QUERIES:
                errors.append(abs(saved['by_query'][q] - float(score['by_query'][q])))
                for scene in SCENES:
                    item = saved['by_scene_query'][scene][q]
                    subset = [r for r in rows if (r['family'], r['h'], r['temperature'], r['scene'], r['query']) == (family, *cfg, scene, q)]
                    if item['n_point'] != sum(r['valid'] for r in subset) or item['n_query'] != expected_pairs*COUNTS[q]:
                        raise ValueError('summary integer population differs')
                    errors.append(abs(item['pck8']-float(score['by_scene'][scene][q])))
    if not all(np.isfinite(v) and v <= 1.e-12 for v in errors):
        raise ValueError('summary differs from exact independent ratios')
    expected_points = {f: dict(grid_index=GRID.index(cfg), config=dict(h=cfg[0], temperature=cfg[1]))
                       for f, cfg in independent['point_configurations'].items()}
    eligible = [GRID.index((v['h'], v['temperature'])) for v in independent['eligible']]
    if (result['point_localization_selections'] != expected_points
            or result['legacy_point'] != dict(grid_index=4, config=dict(h=.5, temperature=5.))
            or result['feasible_grid_indices'] != eligible or result['feasible'] is not bool(eligible)
            or result['decision_pending_independent_audit'] != (
                'complete_full_regional_calibration' if eligible else 'stop_unchanged_candidate')
            or any(result[k] is not False for k in ('final_primary_selected', 'regional_calibration_done',
                                                   'scientific_gate_passed', 'fresh_test_read'))):
        raise ValueError('summary selection or scope changed')
    verify_constraint_table(result['feasibility'], independent['checks'], eligible)
    references = dict(gaussian_point=('gaussian_point', independent['point_configurations']['gaussian_point']),
                      quartic_point=('quartic_point', independent['point_configurations']['quartic_point']),
                      legacy_point_pck=('gaussian_point', (.5, 5.)))
    for cfg in result['feasibility']:
        for check in cfg['checks']:
            family, rcfg = references[check['reference']]
            candidate = independent['scores']['trace_full', GRID[cfg['grid_index']]]['by_query'][check['query']]
            reference = independent['scores'][family, rcfg]['by_query'][check['query']]
            if (check['reference_family'] != family or check['reference_grid_index'] != GRID.index(rcfg)
                    or not np.isfinite(check['candidate_pck8']) or not np.isfinite(check['reference_pck8'])
                    or abs(check['candidate_pck8']-float(candidate)) > 1.e-12
                    or abs(check['reference_pck8']-float(reference)) > 1.e-12):
                raise ValueError('guard reference/value identity changed')
    return dict(feasible_grid_indices=eligible, maximum_summary_error=max(errors))


def audit(run, out):
    """The joint source lock is the pre-forward design freeze, not this run time."""
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists() or run == out or run in out.parents:
        raise ValueError('refusing audit output overwrite')
    started = time.time()
    lock = checked_lock(run)
    verify_policy(lock)
    verify_previous(lock)
    numerical = verify_numerical(lock)
    lock_sha, seal_sha = sha(run/'lock.json'), sha(run/'forward_complete.json')
    if sha(run/'partitions.npz') != lock['partitions_sha256']:
        raise ValueError('native partitions changed')
    complete = read(run/'completion.json')
    files = {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()}
    if files - {'completion.json', 'decision.json'} != set(complete['files']):
        raise ValueError('top-level completed artifact population differs')
    verify_files(run, complete['files'])
    if (complete['lock_sha256'] != lock_sha or complete['summary_sha256'] != sha(run/'summary.json')
            or complete['forward_complete_sha256'] != seal_sha or complete['center_labels_decoded'] is not True
            or complete['regional_labels_decoded'] or complete['fresh_test_read']
            or complete['independent_audit_required'] is not True or complete['finished_unix'] > started):
        raise ValueError('new completed experiment binding mismatch')
    seal = read(run/'forward_complete.json')
    if set(seal['forward_chain']) != set(SCENES):
        raise ValueError('global seal missing a training scene')
    stages = {}
    # Verify all three complete forward/evaluation chains BEFORE decoding even
    # the first old target. Only archive names/hashes and metadata are read here.
    for scene in SCENES:
        slug = scene.replace('/', '__'); f = run/'forward'/slug; e = run/'evaluation'/slug
        fc = checked_completion(f, expected_stage='forward', scene=scene, lock_sha=lock_sha)
        ec = checked_completion(e, expected_stage='evaluation', scene=scene, lock_sha=lock_sha)
        verify_chronology(lock, seal, fc, ec, scene, lock_sha=lock_sha,
                          seal_sha=seal_sha, forward_sha=sha(f/'completion.json'))
        beginning = read(f/'started.json')
        if (beginning['created_unix'] != fc['started_unix'] or beginning['lock_sha256'] != lock_sha
                or any(beginning.get(k) != v for k, v in fc['runtime'].items())
                or ec['finished_unix'] > complete['finished_unix']):
            raise ValueError('forward runtime/start or final completion time changed')
        entries = sorted((v for v in lock['target_contract']['entries'] if v['scene'] == scene), key=lambda v: Path(v['path']).name)
        if ec['label_sources'] != {v['target_path']: v['target_sha256'] for v in entries}:
            raise ValueError('evaluation center-label allowlist changed')
        stages[scene] = (f, e, entries)
    out.mkdir(parents=True)
    write(out/'lock.json', dict(created_unix=started, original_design_created_unix=lock['created_unix'],
        joint_source_lock_sha256=lock_sha, run=str(run), source_hashes={n: lock['source_hashes'][n] for n in SOURCE_NAMES},
        chronological_design_source='producer joint source lock and source_snapshot',
        pre_forward_design=True, independent_new_forward=False))
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    rows, max_normalization, t0_matches, numerical_checks = [], 0., 0, []
    for scene, (forward, evaluation, entries) in stages.items():
        slug = scene.replace('/', '__')
        new_forward = indexed_rows(read(forward/'records.json'), entries, ('trace_full',))
        new_evaluation = indexed_rows(read(evaluation/'rows.json'), entries, ('trace_full',))
        point_families = ('quartic_point', 'gaussian_point')
        old_forward = indexed_rows([r for r in read(OLD_A/'forward'/slug/'records.json') if r['family'] in point_families], entries, point_families)
        old_evaluation = indexed_rows([r for r in read(OLD_A/'evaluation'/slug/'rows.json') if r['family'] in point_families], entries, point_families)
        t0_records = [r for r in read(OLD_B/'forward'/slug/'records.json') if r['family'] == 'gaussian_full1'
                      and r['grid_index'] % 3 == 0 and r['query'] in QUERIES and r['key'] == 'g8']
        t0 = {(tuple(r['pair']), r['grid_index'], r['query']): r['array_sha256'] for r in t0_records}
        expected_t0 = {(tuple(v['pair']), i, q) for v in entries for i in (0, 3, 6, 9) for q in QUERIES}
        if len(t0_records) != len(t0) or set(t0) != expected_t0:
            raise ValueError('old Gaussian T0 commitment population differs')
        deterministic = read(forward/'determinism.json')
        first = {r['field']: r['array_sha256'] for r in new_forward.values() if r['pair'] == entries[0]['pair']}
        if (deterministic['passed'] is not True or deterministic['scene'] != scene
                or deterministic['pair'] != entries[0]['pair'] or deterministic['array_hashes'] != first
                or deterministic['repeated_array_count'] != 36 or deterministic['labels_decoded']):
            raise ValueError('determinism evidence differs from actual first pair')
        for entry in entries:
            target = absolute(entry['target_path'])
            if sha(target) != entry['target_sha256']:
                raise ValueError('old center target archive changed')
            with np.load(target, allow_pickle=False) as archive:
                targets = {q: (archive[q+'_point_xy'], archive[q+'_point_visible']) for q in QUERIES}
            input_sha = lock['inputs'][str(Path(entry['path'])/'inputs.npz')]
            for base, families, fr, er in ((forward, ('trace_full',), new_forward, new_evaluation),
                                          (OLD_A/'forward'/slug, point_families, old_forward, old_evaluation)):
                path = base/(Path(entry['path']).name+'.npz'); digest = sha(path)
                expected_families = ('trace_full',) if families == ('trace_full',) else ('quartic_full', *point_families)
                expected_fields = {f'{f}__c{i:02d}__{q}' for f in expected_families for i in range(12) for q in QUERIES}
                with np.load(path, allow_pickle=False) as archive:
                    if set(archive.files) != expected_fields:
                        raise ValueError('saved archive field population differs')
                    for family in families:
                        for index, (h, t) in enumerate(GRID):
                            for q in QUERIES:
                                identity = (tuple(entry['pair']), family, index, q)
                                p = archive[f'{family}__c{index:02d}__{q}']
                                metric = center_counts(p, *targets[q])
                                verify_record(er[identity], fr[identity], entry, p, metric, path.name, digest, input_sha)
                                if family == 'trace_full' and entry['pair'] == entries[0]['pair']:
                                    numeric_root = absolute(numerical['root'])/'train'
                                    support = numeric_root/'arrays'/slug/f'{q}_g8_support.npz'
                                    geometry = numeric_root/'arrays'/slug/f'{q}_g8_h{h:g}.npz'
                                    with np.load(support, allow_pickle=False) as saved_support, np.load(geometry, allow_pickle=False) as saved:
                                        check = first_pair_probability_check(p, saved_support['indices'],
                                            saved[f'actual_T{t:g}'], saved[f'reference64_T{t:g}'],
                                            saved[f'reference96_T{t:g}'], saved[f'hybrid_T{t:g}'])
                                    numerical_checks.append(dict(scene=scene, pair=entry['pair'], query=q, key='g8',
                                        grid_index=index, h=h, temperature=t, **check,
                                        support_archive=str(support), support_archive_sha256=sha(support),
                                        geometry_archive=str(geometry), geometry_archive_sha256=sha(geometry)))
                                if family == 'trace_full' and t == 0:
                                    if array_sha(p) != t0[tuple(entry['pair']), index, q]:
                                        raise ValueError('new T0 differs from accepted old full1 probability')
                                    t0_matches += 1
                                rows.append(dict(scene=scene, pair=entry['pair'], family=family, h=h,
                                                 temperature=t, query=q, correct=metric['correct'], valid=metric['valid']))
                                max_normalization = max(max_normalization, metric['normalization_error'])
        print('AUDITED', scene, 'all new and reused point arrays', flush=True)
    result = read(run/'summary.json')
    if (len(rows) != 5184 or t0_matches != 576 or len(numerical_checks) != 108
            or sum(r['probability_rows'] for r in numerical_checks) != 864
            or result['row_count'] != 5184 or result['pair_count'] != 48
            or result['new_row_count'] != 1728 or result['reused_point_row_count'] != 3456
            or result['lock_sha256'] != lock_sha or result['forward_complete_sha256'] != seal_sha
            or not max(read(e/'completion.json')['finished_unix'] for _, e, _ in stages.values())
                <= result['created_unix'] <= complete['finished_unix']):
        raise ValueError('complete metric population or summary chronology differs')
    selection = verify_summary(result, rows)
    write(out/'independent_counts.json', rows)
    write(out/'first_pair_numerical_checks.json', numerical_checks)
    replay_summary = dict(probability_matrices=108, probability_rows=864,
        same_prepared_tolerance=1.e-12, probability_tolerance=1.e-4, reference_tolerance=1.e-8,
        **{k: max(r[k] for r in numerical_checks) for k in ('same_prepared_probability_error',
            'reference_probability_error', 'hybrid_probability_error', 'reference_convergence_error')},
        **{k: sum(r[k] for r in numerical_checks) for k in ('same_prepared_argmax_changes', 'reference_argmax_changes')})
    verification = dict(passed=True, feasibility_lock_sha256=lock_sha,
        feasibility_forward_complete_sha256=seal_sha, feasibility_summary_sha256=sha(run/'summary.json'),
        feasibility_completion_sha256=sha(run/'completion.json'), independent_audit_lock_sha256=sha(out/'lock.json'),
        source_hashes={n: lock['source_hashes'][n] for n in SOURCE_NAMES}, probability_arrays=5184,
        new_probability_arrays=1728, reused_point_probability_arrays=3456, old_full1_T0_hash_matches=t0_matches,
        all_original_point_arrays_recounted=True, determinism_evidence_checked=True,
        independent_new_forward=False, numerical_contract=numerical, first_pair_numerical_replay=replay_summary,
        maximum_probability_sum_error=max_normalization, **selection,
        stop_candidate=not selection['feasible_grid_indices'], scientific_gate_passed=False,
        final_primary_selected=False, regional_labels_decoded=False, sensor_decoded=False,
        validation_or_test_read=False, original_design_created_unix=lock['created_unix'],
        started_unix=started, finished_unix=time.time(), seconds=time.time()-started)
    write(out/'verification.json', verification)
    write(out/'completion.json', dict(files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()},
        passed=True, scientific_gate_passed=False, finished_unix=time.time()))
    print(json.dumps({k: v for k, v in verification.items() if k not in ('source_hashes', 'numerical_contract')}, indent=2))
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'results/round17_feasibility')
    parser.add_argument('--out', type=Path, default=ROOT/'results/round17_feasibility_independent_audit')
    args = parser.parse_args()
    audit(args.run, args.out)


if __name__ == '__main__':
    main()


