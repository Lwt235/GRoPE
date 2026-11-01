"""Independent complete audit of the R16 retrospective TRAIN diagnostics.

No candidate probability is generated. The original, already audited Stage A
arrays are the reference. Geometry preparation and the float32 content product
are deliberately shared with frozen Stage A, so this is not an independent
reconstruction of mu/C. All diagnostic calculations use the independent R16
metrics module; neither the new producer nor its metrics module is imported.
The producer and this auditor must be jointly frozen before any R16 analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments import round15_feasibility as A

ROOT = Path(__file__).resolve().parents[1]
SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
QUERIES = ('g8', 'g16', 'g24')
COUNTS = dict(g8=432, g16=108, g24=48)
FAMILIES = ('quartic_full', 'quartic_point', 'gaussian_point')
PAIRED_FAMILIES = ('full_vs_quartic_point', 'full_vs_gaussian_point')
GRID = tuple(dict(h=h, temperature=t) for h in (.25, .5, 1., 2.) for t in (0., 5., 10.))
STATIC_FIELDS = ('mu_px', 'cov_reference8', 'key_px', 'content32', 'target_px', 'valid')
REQUIRED_SOURCES = (
    'ROUND16_MECHANISM_PROTOCOL.md', 'docs/round15_pre_result_mechanism_hypotheses.md',
    'experiments/round16_mechanism.py', 'tests/test_round16_mechanism.py',
    'experiments/round16_mechanism_metrics.py', 'tests/test_round16_mechanism_metrics.py',
    'experiments/round16_mechanism_audit_metrics.py', 'tests/test_round16_mechanism_audit_metrics.py',
    'experiments/round16_mechanism_audit.py', 'tests/test_round16_mechanism_audit.py',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def relative_path(root, value):
    """Accept only canonical, contained relative archive/source names."""
    name = Path(value)
    if name.is_absolute() or '..' in name.parts or str(name) != value:
        raise ValueError('noncanonical relative artifact path: ' + str(value))
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('artifact escapes root: ' + value)
    return path


def absolute_path(value):
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError('canonical absolute upstream path required: ' + str(value))
    return path


def repository_path(value):
    """Original Stage A entries may use canonical repository-relative paths."""
    path = Path(value)
    return absolute_path(value) if path.is_absolute() else relative_path(ROOT, value)


def original_input_digest(prior, input_path):
    path = input_path.resolve()
    candidates = [str(path)]
    if path.is_relative_to(ROOT.resolve()):
        candidates.append(str(path.relative_to(ROOT.resolve())))
    values = [prior['inputs'][key] for key in candidates if key in prior['inputs']]
    if not values or len(set(values)) != 1:
        raise ValueError('missing or conflicting original input commitment')
    return values[0]


def verify_files(root, files, cache=None, *, absolute=False):
    if not isinstance(files, dict) or not files:
        raise ValueError('empty file commitment')
    for name, expected in files.items():
        path = absolute_path(name) if absolute else relative_path(root, name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('invalid SHA256 commitment')
        key = str(path.resolve())
        actual = cache.get(key) if cache is not None else None
        if actual is None:
            actual = sha(path)
            if cache is not None:
                cache[key] = actual
        if actual != expected:
            raise ValueError('changed committed artifact: ' + str(path))


def compare_array(actual, expected, name):
    """Strict dtype/shape and discrete equality; floats use abs 1e-10."""
    if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
        raise ValueError('expected arrays: ' + name)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('array dtype/shape differs: ' + name)
    if actual.dtype.kind in 'biu':
        if not np.array_equal(actual, expected):
            raise ValueError('discrete array differs: ' + name)
        return 0.
    if actual.dtype.kind != 'f':
        raise ValueError('unsupported diagnostic array dtype: ' + name)
    if not np.allclose(actual, expected, rtol=0., atol=1.e-10, equal_nan=True):
        raise ValueError('floating array differs: ' + name)
    finite = np.isfinite(expected) & np.isfinite(actual)
    return float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.


def compare_tree(actual, expected, name='', *, additive=False):
    """Exact keys/discrete values; additive and derived tolerances differ."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('mapping fields differ: ' + name)
        error = 0.
        for key in expected:
            error = max(error, compare_tree(actual[key], expected[key], name + '/' + key,
                additive=additive or key == 'additive'))
        return error
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('sequence population differs: ' + name)
        return max((compare_tree(a, e, name + '/' + str(i), additive=additive)
                    for i, (a, e) in enumerate(zip(actual, expected))), default=0.)
    if isinstance(expected, (bool, int, str)) or expected is None:
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('discrete value differs: ' + name)
        return 0.
    if isinstance(expected, float):
        if type(actual) is not float or not math.isfinite(expected) or not math.isfinite(actual):
            raise ValueError('invalid floating scalar: ' + name)
        error = abs(actual - expected)
        bound = 1.e-8 + 1.e-12 * abs(expected) if additive else 1.e-10
        if error > bound:
            raise ValueError('floating scalar differs: ' + name)
        return error
    raise ValueError('unsupported scalar type: ' + name)


def identity(row):
    return row['scene'], tuple(row['pair']), row['family'], row['grid_index'], row['query']


def population(rows, entries, families):
    expected = {(e['scene'], tuple(e['pair']), f, i, q)
                for e in entries for f in families for i in range(12) for q in QUERIES}
    found = {}
    for row in rows:
        key = identity(row)
        if key in found or key not in expected:
            raise ValueError('duplicate or unexpected diagnostic identity')
        i = row['grid_index']
        if type(i) is not int or row['config'] != GRID[i]:
            raise ValueError('changed diagnostic configuration')
        found[key] = row
    if set(found) != expected:
        raise ValueError('incomplete diagnostic population')
    return found


def check_cohort(entries):
    if not isinstance(entries, list) or len(entries) != 48:
        raise ValueError('expected all 48 old TRAIN pairs')
    keys = [(e['scene'], tuple(e['pair'])) for e in entries]
    if len(set(keys)) != 48 or set(e['scene'] for e in entries) != set(SCENES):
        raise ValueError('changed old TRAIN cohort')
    for scene in SCENES:
        group = [e for e in entries if e['scene'] == scene]
        if len(group) != 16 or len({Path(e['path']).name for e in group}) != 16:
            raise ValueError('incomplete or colliding old TRAIN pair paths')
    for entry in entries:
        repository_path(entry['path']); repository_path(entry['target_path'])


def verify_manifest(directory, completion, cache=None, *, exact=True):
    verify_files(directory, completion['files'], cache)
    if exact:
        actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
        if actual != set(completion['files']) | {'completion.json'}:
            raise ValueError('unbound or missing completed artifact')


def verify_chronology(lock, stage, started, summary, complete, audit_started, lock_sha):
    times = [lock['created_unix'], stage['started_unix'], stage['finished_unix'],
             summary['created_unix'], complete['finished_unix'], audit_started]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in times):
        raise ValueError('invalid chronology timestamp')
    if times != sorted(times) or started['created_unix'] != stage['started_unix']:
        raise ValueError('joint freeze/analysis/summary chronology violated')
    if started['lock_sha256'] != lock_sha or stage['lock_sha256'] != lock_sha:
        raise ValueError('analysis is not bound to the joint source lock')


def require_bound(lock, path):
    value = str(path.resolve())
    if value not in lock['upstream_files'] or lock['upstream_files'][value] != sha(path):
        raise ValueError('missing upstream binding: ' + value)


def verify_contract(run, audit_started):
    """Metadata and all file hashes are checked before decoding diagnostics."""
    lock, complete, summary = (read(run / name) for name in
                               ('lock.json', 'completion.json', 'summary.json'))
    lock_sha = sha(run / 'lock.json')
    if lock_sha != (run / 'lock.sha256').read_text().strip():
        raise ValueError('changed joint lock')
    if (lock.get('joint_producer_and_independent_audit_freeze') is not True
            or lock.get('diagnostic_only') is not True or lock.get('new_predictions') is not False
            or lock.get('new_data') is not False or lock.get('scientific_gate_passed') is not False):
        raise ValueError('diagnostic scope changed')
    expected_counts = dict(expected_rows=5184, expected_paired_rows=3456,
                           expected_query_states=1016064, expected_static_queries=28224)
    for key, value in expected_counts.items():
        if type(lock[key]) is not int or lock[key] != value:
            raise ValueError('changed predeclared population')
    if lock['fields'] != list(A.FIELDS) or lock['runtime'] != A.configure_runtime('cpu'):
        raise ValueError('changed forward information/runtime')
    if not set(REQUIRED_SOURCES).issubset(lock['source_hashes']):
        raise ValueError('joint lock omitted required producer/auditor sources')
    inherited = A.sources()
    if any(lock['source_hashes'].get(k) != v for k, v in inherited.items()):
        raise ValueError('shared frozen Stage A source dependency changed')
    cache = {}
    verify_files(ROOT, lock['source_hashes'], cache)
    verify_files(run / 'source_snapshot', lock['source_hashes'], cache)
    verify_files(ROOT, lock['upstream_files'], cache, absolute=True)
    if complete['lock_sha256'] != lock_sha:
        raise ValueError('completion lock mismatch')
    verify_manifest(run, complete, cache)
    if (summary['diagnostic_only'] is not True or summary['new_predictions'] is not False
            or summary['scientific_gate_passed'] is not False
            or summary['independent_audit_pending'] is not True):
        raise ValueError('summary overclaims diagnostic scope')
    prior_root = absolute_path(lock['stage_a_root'])
    prior = read(prior_root / 'lock.json')
    require_bound(lock, prior_root / 'lock.json')
    require_bound(lock, prior_root / 'completion.json')
    require_bound(lock, prior_root / 'decision.json')
    if sha(prior_root / 'lock.json') != (prior_root / 'lock.sha256').read_text().strip():
        raise ValueError('Stage A lock mismatch')
    prior_complete = read(prior_root / 'completion.json')
    if (prior_complete['lock_sha256'] != sha(prior_root / 'lock.json')
            or prior_complete['summary_sha256'] != sha(prior_root / 'summary.json')):
        raise ValueError('Stage A completion chain mismatch')
    # Stage A legitimately gained decision/report files after its completion.
    verify_manifest(prior_root, prior_complete, cache, exact=False)
    for path, digest in prior_complete['files'].items():
        if lock['upstream_files'].get(str((prior_root / path).resolve())) != digest:
            raise ValueError('Stage A completion artifact omitted from joint lock')
    if lock['entries'] != prior['target_contract']['entries']:
        raise ValueError('joint lock cohort differs from sealed Stage A')
    check_cohort(lock['entries'])
    stage_a_decision = read(prior_root / 'decision.json')
    stage_a_audit = repository_path(stage_a_decision['audit_path'])
    require_bound(lock, stage_a_audit)
    if stage_a_decision['audit_sha256'] != sha(stage_a_audit):
        raise ValueError('Stage A independent audit binding differs')
    verification = read(stage_a_audit)
    if (verification['passed'] is not True or verification['scientific_gate_passed'] is not False
            or stage_a_decision['validation_authorized'] is not False
            or stage_a_decision['scientific_gate_passed'] is not False):
        raise ValueError('accepted Stage A audit/scope required')
    for name in ('lock', 'forward_complete', 'summary', 'completion'):
        key = 'feasibility_' + name + '_sha256'
        expected = sha(prior_root / (name + '.json'))
        if stage_a_decision[key] != expected or verification[key] != expected:
            raise ValueError('Stage A independent audit chain differs')
    decision_path = ROOT / 'results/round15_regional_post_result_decision.json'
    require_bound(lock, decision_path)
    decision = read(decision_path)
    if (decision['decision_type'] != 'post_result_development_futility'
            or decision['action'] != 'stop_fixed_quartic_candidate_before_old_validation'
            or decision['scientific_gate_passed'] is not False
            or decision['general_research_goal_achieved'] is not False
            or decision['old_validation']['source_freeze_performed'] is not False):
        raise ValueError('retained R15 stop decision changed')
    for binding in (*decision['bindings'].values(), decision['report']):
        path = absolute_path(binding['path'])
        if lock['upstream_files'].get(str(path)) != binding['sha256']:
            raise ValueError('R15 decision binding omitted from joint lock')
    for entry in lock['entries']:
        path = repository_path(entry['path']) / 'inputs.npz'
        recorded = original_input_digest(prior, path)
        if lock['upstream_files'].get(str(path)) != recorded:
            raise ValueError('unbound old forward input')
        if lock['upstream_files'].get(str(repository_path(entry['target_path']))) != entry['target_sha256']:
            raise ValueError('unbound old center target')
    stages = {}
    for scene in SCENES:
        directory = run / 'analysis' / scene.replace('/', '__')
        stage = read(directory / 'completion.json')
        if stage['stage'] != 'analysis' or stage['scene'] != scene:
            raise ValueError('analysis stage identity mismatch')
        if stage['diagnostic_only'] is not True or stage['new_predictions'] is not False:
            raise ValueError('analysis scope mismatch')
        verify_manifest(directory, stage, cache)
        verify_chronology(lock, stage, read(directory / 'started.json'), summary,
                          complete, audit_started, lock_sha)
        if (stage['row_count'], stage['paired_row_count'], stage['query_state_count'],
                stage['static_query_count']) != (1728, 1152, 338688, 9408):
            raise ValueError('analysis shard population mismatch')
        stages[scene] = stage
    return lock, prior, summary, stages


def reference_center(probability, key_px, target, valid):
    n_query = len(probability)
    if (probability.dtype != np.float64 or probability.shape != (n_query, 432)
            or not np.isfinite(probability).all() or (probability < 0).any()
            or np.max(np.abs(probability.sum(1) - 1.)) > 1.e-12):
        raise ValueError('invalid sealed native probability')
    if (key_px.shape != (432, 2) or target.shape != (n_query, 2)
            or valid.shape != (n_query,) or valid.dtype != np.bool_
            or not np.isfinite(target[valid]).all()):
        raise ValueError('invalid center reference population')
    winner = probability.argmax(axis=1)
    delta = key_px[winner[valid]] - target[valid]
    correct = int((np.einsum('ij,ij->i', delta, delta) <= 64.).sum())
    return winner, dict(pck_count=correct, n_point=int(valid.sum()), n_query=n_query,
                        argmax_sha256=array_sha(winner))


def temperature_identity(p0, p, content, temperature):
    """Verify old probability algebra; this never creates a candidate output."""
    logits = np.log(p0) + (np.float32(temperature) * content).astype(np.float64)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    error = float(np.max(np.abs(e / e.sum(axis=1, keepdims=True) - p)))
    if not math.isfinite(error) or error > 1.e-12:
        raise ValueError('old temperature identity differs')
    return error


@torch.no_grad()
def audit(run, out):
    from experiments import round16_mechanism_audit_metrics as M

    run, out = run.resolve(), out.resolve()
    if out.exists():
        raise ValueError('refusing audit overwrite or partial-output reuse')
    started = time.time()
    # Retain failure evidence too. The command line wrapper writes failure.json
    # if a later verification fails; no successful completion is then written.
    out.mkdir(parents=True)
    write(out / 'started.json', dict(created_unix=started, run=str(run),
                                    lock_sha256=sha(run / 'lock.json')))
    lock, prior, saved_summary, stages = verify_contract(run, started)
    initial_metadata = {name: sha(run / name) for name in ('lock.json', 'summary.json', 'completion.json')}
    prior_root = Path(lock['stage_a_root'])
    parts = A.S.read_parts(prior_root)
    tparts = A.S.M.partition_tensors(parts, 'cpu')
    # Independent fixed lattice confirms center units; the full mu/C pathway
    # below intentionally uses frozen Stage A preparation.
    yy, xx = np.meshgrid(np.arange(18) * 8 + 3.5, np.arange(24) * 8 + 3.5, indexing='ij')
    independent_key_px = np.column_stack((xx.ravel(), yy.ravel()))
    key_px = parts['g8']['mean'] * 8
    if not np.array_equal(key_px, independent_key_px):
        raise ValueError('native g8 center convention changed')
    all_rows, all_paired = [], []
    errors = dict(raw_arrays=0., additive_stats=0., summary=0., temperature_identity=0.)
    states = static_queries = probability_arrays = diagnostic_archives = 0
    expected_probability_fields = {f'{f}__c{i:02d}__{q}'
                                   for f in FAMILIES for i in range(12) for q in QUERIES}
    for scene in SCENES:
        slug = scene.replace('/', '__')
        entries = sorted((e for e in lock['entries'] if e['scene'] == scene),
                         key=lambda e: Path(e['path']).name)
        directory = run / 'analysis' / slug
        rows = population(read(directory / 'rows.json'), entries, FAMILIES)
        paired = population(read(directory / 'paired_rows.json'), entries, PAIRED_FAMILIES)
        original_dir = prior_root / 'forward' / slug
        old_records = population(read(original_dir / 'records.json'), entries, FAMILIES)
        old_metrics = population(read(prior_root / 'evaluation' / slug / 'rows.json'), entries, FAMILIES)
        scene_reconstruction = 0.
        for pair_index, entry in enumerate(entries):
            input_path = repository_path(entry['path']) / 'inputs.npz'
            # Only the same four already-authorized forward fields are decoded.
            observations = A.S.A.observation_tensors(A.load_forward_inputs(input_path), 'cpu')
            key = A.S.M.prepare_key(observations, tparts['g8'])
            labels = A.load_center_targets(repository_path(entry['target_path']))
            archive_path = original_dir / (Path(entry['path']).name + '.npz')
            archive_digest = sha(archive_path)
            with np.load(archive_path, allow_pickle=False) as probabilities:
                if set(probabilities.files) != expected_probability_fields:
                    raise ValueError('Stage A probability archive population differs')
                for q in QUERIES:
                    query = A.prepare_query(observations, tparts[q], True)
                    static = dict(mu_px=(query['mean'] * 8).numpy(), cov_reference8=query['cov'].numpy(),
                        key_px=key_px, content32=(query['features'] @ key['features'].T).numpy(),
                        target_px=labels[q + '_point_xy'], valid=labels[q + '_point_visible'])
                    if len(static['mu_px']) != COUNTS[q]:
                        raise ValueError('shared preparation changed native query population')
                    relative = f'analysis/{slug}/queries/{Path(entry["path"]).name}__{q}.npz'
                    banks = {}
                    expected_fields = set(STATIC_FIELDS)
                    with np.load(relative_path(run, relative), allow_pickle=False) as diagnostic:
                        for field in STATIC_FIELDS:
                            errors['raw_arrays'] = max(errors['raw_arrays'],
                                compare_array(diagnostic[field], static[field], relative + ':' + field))
                        for family in FAMILIES:
                            for index, config in enumerate(GRID):
                                rowid = (scene, tuple(entry['pair']), family, index, q)
                                saved, record, old = rows[rowid], old_records[rowid], old_metrics[rowid]
                                pfield = f'{family}__c{index:02d}__{q}'
                                p0field = f'{family}__c{3 * (index // 3):02d}__{q}'
                                p, p0 = probabilities[pfield], probabilities[p0field]
                                psha, p0sha = array_sha(p), array_sha(p0)
                                p0record = old_records[(scene, tuple(entry['pair']), family, 3 * (index // 3), q)]
                                if (record['field'] != pfield or record['array_sha256'] != psha
                                        or p0record['array_sha256'] != p0sha
                                        or record['archive_sha256'] != archive_digest
                                        or record['archive'] != archive_path.name
                                        or record['shape'] != list(p.shape) or record['dtype'] != 'float64'
                                        or record['input_sha256'] != lock['upstream_files'][str(input_path)]
                                        or any(old.get(k) != v for k, v in record.items())
                                        or old['target_sha256'] != entry['target_sha256']):
                                    raise ValueError('old probability provenance differs')
                                winner, center = reference_center(p, independent_key_px, static['target_px'], static['valid'])
                                for name, value in center.items():
                                    if saved[name] != value or old[name] != value or type(saved[name]) is not type(value):
                                        raise ValueError('original center count/argmax changed: ' + name)
                                if (saved['diagnostic_path'] != relative or saved['probability_sha256'] != psha
                                        or saved['p0_sha256'] != p0sha
                                        or saved['input_sha256'] != record['input_sha256']
                                        or saved['target_sha256'] != entry['target_sha256']):
                                    raise ValueError('diagnostic row provenance differs')
                                reconstruction = temperature_identity(p0, p, static['content32'], config['temperature'])
                                scene_reconstruction = max(scene_reconstruction, reconstruction)
                                errors['temperature_identity'] = max(errors['temperature_identity'], reconstruction)
                                expected = M.analyze(p0, p, static['content32'], static['mu_px'],
                                    static['cov_reference8'], key_px, static['target_px'], static['valid'], **config)
                                if float(np.max(expected['identity_error'])) > 1.e-10:
                                    raise ValueError('winner log-odds identity failed')
                                if (not np.array_equal(expected['winner'], winner)
                                        or int((expected['correct'] == 1).sum()) != center['pck_count']):
                                    raise ValueError('independent diagnostic center definition differs')
                                if set(saved['diagnostic_array_hashes']) != set(expected):
                                    raise ValueError('diagnostic array commitments incomplete')
                                prefix = f'{family}__c{index:02d}__'
                                for name, value in expected.items():
                                    field = prefix + name
                                    expected_fields.add(field)
                                    actual = diagnostic[field]
                                    if saved['diagnostic_array_hashes'][name] != array_sha(actual):
                                        raise ValueError('diagnostic array hash differs')
                                    errors['raw_arrays'] = max(errors['raw_arrays'],
                                        compare_array(actual, value, relative + ':' + field))
                                item = dict(scene=scene, pair=entry['pair'], family=family, grid_index=index,
                                            query=q, config=config, stats=M.stats(expected))
                                errors['additive_stats'] = max(errors['additive_stats'],
                                    compare_tree(saved['stats'], item['stats'], str(rowid), additive=True))
                                all_rows.append(item)
                                banks[family, index] = expected
                                probability_arrays += 1; states += len(p)
                        if set(diagnostic.files) != expected_fields:
                            raise ValueError('diagnostic archive has unexpected or missing fields')
                    for family in ('quartic_point', 'gaussian_point'):
                        for index, config in enumerate(GRID):
                            name = 'full_vs_' + family
                            rowid = (scene, tuple(entry['pair']), name, index, q)
                            item = dict(scene=scene, pair=entry['pair'], family=name, grid_index=index,
                                query=q, config=config,
                                stats=M.paired_stats(banks['quartic_full', index], banks[family, index]))
                            errors['additive_stats'] = max(errors['additive_stats'],
                                compare_tree(paired[rowid]['stats'], item['stats'], str(rowid), additive=True))
                            all_paired.append(item)
                    static_queries += COUNTS[q]; diagnostic_archives += 1
            print('MECHANISM_AUDIT', scene, pair_index + 1, '/16', flush=True)
        compare_tree(stages[scene]['maximum_reconstruction_error'], scene_reconstruction,
                     scene + '/maximum_reconstruction_error')
    if (probability_arrays, len(all_paired), states, static_queries, diagnostic_archives) != (
            5184, 3456, 1016064, 28224, 144):
        raise ValueError('incomplete independent reconstruction population')
    population(all_rows, lock['entries'], FAMILIES)
    population(all_paired, lock['entries'], PAIRED_FAMILIES)
    independent = dict(metrics=M.summarize(all_rows), paired_metrics=M.summarize(all_paired),
        row_count=len(all_rows), paired_row_count=len(all_paired), query_state_count=states,
        static_query_count=static_queries, maximum_reconstruction_error=errors['temperature_identity'])
    for name, expected in independent.items():
        errors['summary'] = max(errors['summary'], compare_tree(saved_summary[name], expected, name))
    # Full source recheck closes the audit interval without rereading gigabytes.
    verify_files(ROOT, lock['source_hashes'])
    verify_files(run / 'source_snapshot', lock['source_hashes'])
    verify_files(run, initial_metadata)
    write(out / 'independent_summary.json', independent)
    verification = dict(passed=True, lock_sha256=sha(run / 'lock.json'),
        mechanism_lock_sha256=sha(run / 'lock.json'), mechanism_summary_sha256=sha(run / 'summary.json'),
        mechanism_completion_sha256=sha(run / 'completion.json'),
        independent_summary_sha256=sha(out / 'independent_summary.json'),
        source_hashes=lock['source_hashes'], started_unix=started, finished_unix=time.time(),
        probability_arrays=probability_arrays, paired_rows=len(all_paired),
        query_state_count=states, static_query_count=static_queries, diagnostic_archives=diagnostic_archives,
        maximum_errors=errors, all_original_stage_a_pck_and_argmax_checked=True,
        all_diagnostic_arrays_independently_recomputed=True, all_additives_and_summaries_checked=True,
        shared_frozen_stage_a_preparation=True, independently_reconstructed_mu_cov=False,
        independent_diagnostic_metrics=True, diagnostic_only=True, new_predictions=False,
        new_data=False, new_parameters_selected=False, scientific_gate_passed=False,
        regional_labels_decoded=False, sensor_decoded=False, validation_or_test_read=False,
        seconds=time.time() - started)
    write(out / 'verification.json', verification)
    write(out / 'completion.json', dict(passed=True, lock_sha256=sha(run / 'lock.json'),
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()},
        finished_unix=time.time()))
    print(json.dumps({k: v for k, v in verification.items() if k != 'source_hashes'}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('audit',))
    parser.add_argument('--run', type=Path, default=ROOT / 'results/round16_mechanism')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/round16_mechanism_independent_audit')
    args = parser.parse_args()
    try:
        audit(args.run, args.out)
    except Exception as error:
        # Preserve the original exception/traceback and any partial audit output.
        # An existing completed audit is never mutated by an overwrite attempt.
        directory = args.out.resolve()
        if directory.exists() and not (directory / 'completion.json').exists() and not (directory / 'failure.json').exists():
            write(directory / 'failure.json', dict(exception=type(error).__name__, message=str(error),
                recorded_unix=time.time(), passed=False, diagnostic_only=True, scientific_gate_passed=False))
        raise


if __name__ == '__main__':
    main()
