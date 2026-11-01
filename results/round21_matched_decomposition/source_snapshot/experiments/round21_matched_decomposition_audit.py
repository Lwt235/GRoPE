"""R21 independent provenance, mathematics and paired-metric audit.

The all-matrix replay deliberately imports production at that step only and
is labelled reproducibility. Independent NumPy moments/GL references and
metric reductions never use the R21 production mathematical helpers.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil
import os
import time
import numpy as np  # The accepted runtime requires NumPy before Torch.
import torch
from threadpoolctl import threadpool_limits
from experiments import round20_anchor_audit as BASE
from experiments import round21_matched_decomposition_audit_metrics as M
from experiments import round21_matched_decomposition_audit_numeric as N

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/round21_matched_decomposition'
DEFAULT_OUT = ROOT / 'results/round21_matched_decomposition_independent_audit'
R20_ROOT = ROOT / 'results/round20_anchor_fixed'
R20_ACCEPTANCE = ROOT / 'results/round20_anchor_root_acceptance_20261031.json'
SCENES, CONDITIONS, COUNTS = M.SCENES, M.CONDITIONS, M.COUNTS
FAMILIES, GEOMETRIES, PARTITIONS = M.FAMILIES, M.GEOMETRIES, M.QUERIES
FIELDS = BASE.FIELDS
THREAD_ENV = BASE.THREAD_ENV
RAW_BYTES, RAW_MASS_BYTES = 780337152, 390168576
NUMERICAL_POLICY = dict(gaussian_order=32, reference_gaussian_orders=[64, 96], reference_query_rows=8,
    reference_pairs='first metadata pair in each scene', reference_matrices=72, reference_probability_rows=576,
    reference_unique_query_rows=72, probability_error=1e-4, reference_convergence=1e-8,
    same_raw_mass_arithmetic=1e-12, normalization_error=1e-12, anchor_error_px=1e-9,
    moment_normalized_error=1e-4, content_error=1e-5, geometry_dtype='float64', anchor_dtype='float64',
    new_covariance_dtype='float64', old_moment_dtype='float32', content_bias_dtype='float32', floor=1e-30)
SOURCE_ROOTS = ('ROUND21_MATCHED_DECOMPOSITION_PROTOCOL.md', 'docs/round21_matched_decomposition_design_review.md',
    'experiments/round13_mixture_mechanism.py',
    'models/round21_matched_decomposition.py', 'experiments/round21_matched_decomposition.py',
    'experiments/round21_matched_decomposition_metrics.py', 'tests/test_round21_matched_decomposition.py',
    'experiments/round21_matched_decomposition_audit.py', 'experiments/round21_matched_decomposition_audit_metrics.py',
    'experiments/round21_matched_decomposition_audit_numeric.py', 'tests/test_round21_matched_decomposition_audit.py',
    'tests/test_round21_matched_decomposition_audit_metrics.py', 'tests/test_round21_matched_decomposition_audit_numeric.py')
sha, read, write, array_sha = BASE.sha, BASE.read, BASE.write, BASE.array_sha
inventory, check_inventory, runtime = BASE.inventory, BASE.check_inventory, BASE.runtime
compare, check_file, identity, failure = BASE.compare, BASE.check_file, BASE.identity, BASE.failure


def source_closure():
    found, pending = set(), list(SOURCE_ROOTS)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / name
        if not path.is_file():
            raise ValueError('missing jointly frozen source: ' + name)
        found.add(name)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('relative imports require explicit closure review')
                if node.module:
                    modules = [node.module] + [node.module + '.' + a.name for a in node.names]
            for module in modules:
                parts = module.split('.')
                candidates = [Path(*parts).with_suffix('.py'), Path(*parts) / '__init__.py']
                candidates += [Path(*parts[:i]) / '__init__.py' for i in range(1, len(parts))]
                pending.extend(str(p) for p in candidates if (ROOT / p).is_file() and str(p) not in found)
    return sorted(found)


def role_configs():
    return {g + '_' + s: dict(geometry=g, content_policy=s, h=.25, temperature=10.)
            for g in GEOMETRIES for s in ('zero', 'content')}


def check_current_sources(lock):
    if runtime() != lock['runtime']:
        raise ValueError('audit runtime changed')
    for name, digest in lock['source_hashes'].items():
        check_file(ROOT / name, digest)


def inherited_replay_rows(old_lock):
    """Rebuild the four complete historical contracts from accepted metadata."""
    inherited = BASE.inherited_rows(old_lock)
    evaluation = read(R20_ROOT / 'evaluation/rows.json')
    conditions = {c['condition'] for c in CONDITIONS}; output = {}
    for role, source in (('mu_c_zero', 'regional_gaussian_full1'), ('full4_zero', 'regional_gaussian_full4')):
        rows = [dict(r) for r in inherited[source]['regional'] if r['condition'] in conditions]
        for row in rows:
            if row['config'] != dict(h=.25, temperature=0.) or row['replayed_array_sha256'] != row['array_sha256']:
                raise ValueError('old T0 complete commitment changed')
            path = ROOT / 'results/round15_regional' / row['sample_path'] if 'sample_path' in row else None
            row.update(replay_kind='r15_commitment', source_archive_path=str(path) if path else None,
                source_archive_sha256=(read(ROOT / 'results/round15_regional/completion.json')['files'][row['sample_path']] if path else None),
                source_field=row.get('sample_field'))
        output[role] = rows
    for role, source in (('a_c_content', 'mean_only_matched_tau'), ('a_ca_content', 'anchor_trace')):
        rows = [dict(r) for r in evaluation if r['family'] == source and r['condition'] in conditions]
        for row in rows:
            row.update(replay_kind='r20_saved', source_archive_path=str(R20_ROOT / row['archive_path']),
                       source_archive_sha256=row['archive_sha256'], source_field=source)
        output[role] = rows
    expected = {(*identity(e), c) for e in old_lock['entries'] for c in conditions}
    for rows in output.values():
        if len(rows) != 144 or {(*identity(r), r['condition']) for r in rows} != expected:
            raise ValueError('historical complete replay population changed')
    if sum(r['source_archive_path'] is not None for f in ('mu_c_zero', 'full4_zero') for r in output[f]) != 18:
        raise ValueError('exactly eighteen actually retained historical T0 arrays required')
    return output


def check_producer_lock(run):
    lock = read(run / 'lock.json')
    check_file(run / 'lock.json', (run / 'lock.sha256').read_text().strip())
    expected = dict(families=list(FAMILIES), geometries=list(GEOMETRIES), role_configs=role_configs(),
        contrasts=list(M.CONTRASTS), conditions=list(CONDITIONS), counts=COUNTS, expected_valid_counts=M.VALID_COUNTS,
        field_allowlist=list(FIELDS), numerical_policy=NUMERICAL_POLICY, numerical_references=N.REFERENCE_POLICY,
        pairs=48, unique_query_rows=28224, matrices=1152, raw_matrices=576, raw_probability_bytes=RAW_BYTES,
        raw_mass_bytes=RAW_MASS_BYTES, actual_target_hashes_read=False, labels_decoded=False,
        scientific_gate_passed=False, diagnostic_only=True, no_candidate_selection=True)
    for key, value in expected.items():
        compare(lock[key], value, 0., '/lock/' + key)
    M._cohort(lock['entries'])
    if set(lock['source_hashes']) != set(source_closure()):
        raise ValueError('joint transitive source closure changed')
    check_current_sources(lock)
    for name, digest in lock['source_hashes'].items():
        check_file(run / 'source_snapshot' / name, digest)
    denied = {str(Path(e[n]).resolve()) for e in lock['entries'] for n in ('label_path', 'target_path')}
    for filename, digest in lock['bindings'].items():
        path = Path(filename).resolve()
        if str(path) in denied or path.name == 'labels.npz' or 'targets' in path.parts:
            raise ValueError('actual label archive forbidden in forward-audit bindings')
        check_file(path, digest)
    def bound(path, digest=None):
        path = str(Path(path).resolve()); actual = lock['bindings'].get(path)
        if actual is None or digest is not None and actual != digest:
            raise ValueError('missing inherited binding: ' + path)
        return actual
    # Recheck the already accepted parent chain; no label archive is touched.
    old = BASE.check_producer_lock(R20_ROOT)
    receipt = read(R20_ACCEPTANCE)
    if not all(receipt.get(k) is True for k in ('accepted_for_development_interpretation',
        'full_forward_and_final_independent_audits_passed', 'strict_forward_before_label_chronology_checked')):
        raise ValueError('accepted complete R20 evidence required')
    bound(R20_ACCEPTANCE)
    for path, digest in receipt['files'].items():
        bound(ROOT / path, digest)
    compare(lock['entries'], old['entries'], 0., '/original-entries')
    compare(lock['source_function'], old['source_function'], 0., '/original-center-function')
    for path, digest in old['bindings'].items():
        bound(path, digest)
    for directory in (R20_ROOT, ROOT / 'results/round20_anchor_independent_audit'):
        for name, digest in read(directory / 'lock.json')['source_hashes'].items():
            bound(ROOT / name, digest); bound(directory / 'source_snapshot' / name, digest)
    if lock['partitions_path'] != old['partitions_path'] or lock['partitions_sha256'] != old['partitions_sha256']:
        raise ValueError('old native partition identity changed')
    bound(lock['partitions_path'], lock['partitions_sha256'])
    bound(R20_ROOT / 'inherited_rows.json', old['inherited_rows_sha256'])
    check_file(run / 'inherited_replay_rows.json', lock['inherited_replay_rows_sha256'])
    replay = inherited_replay_rows(old)
    compare(read(run / 'inherited_replay_rows.json'), replay, 0., '/historical-replay-contract')
    for rows in replay.values():
        for row in rows:
            if row['source_archive_path'] is not None:
                bound(row['source_archive_path'], row['source_archive_sha256'])
    return lock


def freeze(run, out):
    if out.exists() or (run / 'forward').exists() or (run / 'forward_complete.json').exists():
        raise ValueError('joint independent freeze must precede every real forward')
    lock = check_producer_lock(run)
    if Path(lock['audit_root']).resolve() != out.resolve():
        raise ValueError('different declared independent audit directory')
    out.mkdir(parents=True)
    for name in lock['source_hashes']:
        target = out / 'source_snapshot' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write(out / 'lock.json', dict(created_unix=time.time(), run=str(run.resolve()), runtime=runtime(),
        producer_lock_sha256=sha(run / 'lock.json'), source_hashes=lock['source_hashes'],
        numerical_references=N.REFERENCE_POLICY, labels_decoded=False, scientific_gate_passed=False))
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')


def check_audit_lock(run, out):
    design = read(out / 'lock.json')
    check_file(out / 'lock.json', (out / 'lock.sha256').read_text().strip())
    lock = check_producer_lock(run)
    if (design['run'] != str(run.resolve()) or design['runtime'] != runtime()
            or design['producer_lock_sha256'] != sha(run / 'lock.json')
            or design['source_hashes'] != lock['source_hashes'] or design['numerical_references'] != N.REFERENCE_POLICY
            or design['created_unix'] < lock['created_unix'] or Path(lock['audit_root']).resolve() != out.resolve()):
        raise ValueError('independent joint freeze changed')
    for name, digest in lock['source_hashes'].items():
        check_file(out / 'source_snapshot' / name, digest)
    return lock, design


def validate_records(records, entries):
    expected = {(*identity(e), c['condition'], f) for e in entries for c in CONDITIONS for f in FAMILIES}
    ids = [(*identity(r), r['condition'], r['family']) for r in records]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('complete unique 1152 probability records required')
    lookup, conditions = {identity(e): e for e in entries}, {c['condition']: c for c in CONDITIONS}
    for row in records:
        entry, c = lookup[identity(row)], conditions[row['condition']]
        geometry = role_configs()[row['family']]['geometry']
        path = 'forward/' + row['scene'].replace('/', '__') + '/arrays/' + entry['pair_name'] + '/' + row['condition'] + '.npz'
        if (row['config'] != role_configs()[row['family']] or any(row[k] != c[k] for k in ('query', 'key', 'scenario'))
                or row['shape'] != [COUNTS[c['query']], 432] or row['dtype'] != 'float64'
                or row['input_sha256'] != entry['input_sha256'] or row['pair_name'] != entry['pair_name']
                or row['archive_path'] != path or row['raw_mass_field'] != geometry + '__raw_mass'):
            raise ValueError('fixed diagnostic identity/configuration changed')


def check_seal(run, lock, design):
    complete, sealing = check_inventory(run / 'forward'), check_inventory(run / 'seal')
    seal = read(run / 'forward_complete.json')
    expected = dict(producer_lock_sha256=sha(run / 'lock.json'), forward_completion_sha256=sha(run / 'forward/completion.json'),
        forward_records_sha256=sha(run / 'forward/records.json'), support_records_sha256=sha(run / 'forward/support_records.json'),
        pairs=48, matrices=1152, raw_matrices=576, raw_probability_bytes=RAW_BYTES, raw_mass_bytes=RAW_MASS_BYTES,
        all_probabilities_retained=True, all_raw_masses_retained=True, actual_target_hashes_read=False, labels_decoded=False)
    for k, v in expected.items():
        if seal.get(k) != v:
            raise ValueError('global seal field changed: ' + k)
    if sealing['forward_complete_sha256'] != sha(run / 'forward_complete.json') or sealing.get('all_raw_masses_retained') is not True:
        raise ValueError('global seal commitment changed')
    records, supports = read(run / 'forward/records.json'), read(run / 'forward/support_records.json')
    validate_records(records, lock['entries'])
    if len(supports) != 48 or [identity(s) for s in supports] != [identity(e) for e in lock['entries']]:
        raise ValueError('complete ordered support evidence required')
    expected_files = {'records.json', 'support_records.json'}
    for scene in SCENES:
        folder = run / 'forward' / scene.replace('/', '__'); done = check_inventory(folder); start = read(folder / 'started.json')
        if (done['records'] != 384 or done['producer_lock_sha256'] != sha(run / 'lock.json')
                or done['scene'] != scene or done['labels_decoded'] is not False or done['actual_target_hashes_read'] is not False
                or start['created_unix'] != done['started_unix'] or start['scene'] != scene
                or start['producer_lock_sha256'] != sha(run / 'lock.json')
                or not design['created_unix'] <= done['started_unix'] <= done['finished_unix'] <= seal['created_unix']):
            raise ValueError('scene source/population/chronology changed')
        if read(folder / 'records.json') != [r for r in records if r['scene'] == scene] or read(folder / 'support_records.json') != [s for s in supports if s['scene'] == scene]:
            raise ValueError('scene/global metadata differs')
        names = {'started.json', 'records.json', 'support_records.json'}
        names.update(str((run / r['archive_path']).relative_to(folder)) for r in records if r['scene'] == scene)
        names.update(str((run / r['archive_path']).relative_to(folder)) for r in supports if r['scene'] == scene)
        if set(done['files']) != names:
            raise ValueError('unsealed extra/missing scene files')
        expected_files.update(str((folder / n).relative_to(run / 'forward')) for n in done['files'])
        expected_files.add(str((folder / 'completion.json').relative_to(run / 'forward')))
    if (set(complete['files']) != expected_files or complete['matrices'] != 1152
            or complete.get('all_raw_masses_retained') is not True or complete['labels_decoded'] is not False
            or complete['actual_target_hashes_read'] is not False):
        raise ValueError('global forward inventory/population changed')
    return records, supports, seal


def legacy_array_sha(array):
    import hashlib
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def inherited_probability(probability, source):
    digest = (legacy_array_sha if source['replay_kind'] == 'r15_commitment' else array_sha)(probability)
    if digest != source['array_sha256']:
        raise ValueError('historical complete probability commitment differs')
    saved = source['source_archive_path'] is not None
    if saved:
        check_file(source['source_archive_path'], source['source_archive_sha256'])
        with np.load(source['source_archive_path'], allow_pickle=False) as a:
            if not np.array_equal(probability, a[source['source_field']]):
                raise ValueError('actually retained historical probability differs')
    return dict(kind=source['replay_kind'], historical_array_sha256=digest, commitment_exact=True, saved_array_checked=saved)


def load_independent_support(entry, record, run, parts):
    check_file(entry['input_path'], entry['input_sha256'])
    with np.load(entry['input_path'], allow_pickle=False) as a:
        inputs = {n: a[n] for n in FIELDS}
    check_file(run / record['archive_path'], record['archive_sha256'])
    with np.load(run / record['archive_path'], allow_pickle=False) as a:
        saved = {n: a[n] for n in a.files}
    if {n: array_sha(v) for n, v in saved.items()} != record['array_sha256']:
        raise ValueError('saved support canonical identity changed')
    queries, key, schema = N.reference_banks(inputs, parts)
    checks = []
    fields = ('mu32', 'cov32', 'a64', 'anchor_cov64', 'delta64', 'q_features32', 'k_features32',
              'content32', 'tau32', 'bias32', 'full4_nodes32', 'full4_node_cov32', 'full4_weights32',
              'mixture_mean64', 'mixture_cov64', 'mean_residual64', 'cov_residual64', 'bounds_px', 'a_px', 'a_front', 'a_fov')
    expected_fields = {q + '__' + n for q in PARTITIONS for n in fields} | {
        'key__' + n for n in ('mass', 'cell_bounds', 'cell_owner', 'features', 'bounds_px')}
    if set(saved) != expected_fields:
        raise ValueError('extra/missing role-owned support fields')
    key_bounds = schema['g8_bounds']
    for name, value in (('bounds_px', key_bounds), ('cell_bounds', (key_bounds - .5) / 8.),
                        ('cell_owner', np.arange(432, dtype=np.int64))):
        if saved['key__' + name].dtype != value.dtype or not np.array_equal(saved['key__' + name], value):
            raise ValueError('independent literal native key geometry changed: ' + name)
    if not np.array_equal(saved['key__mass'], key['mass']):
        raise ValueError('independent native key mass changed')
    if (saved['key__features'].dtype != np.float32 or saved['key__features'].shape != key['features'].shape
            or not np.isfinite(saved['key__features']).all()
            or np.max(abs(saved['key__features'] - key['features'])) > 1e-5):
        raise ValueError('independent native key pooled features changed')
    with np.load(entry['support_path'], allow_pickle=False) as old, np.load(entry['r19_archive_path'], allow_pickle=False) as r19:
        for q in PARTITIONS:
            get = lambda n: saved[q + '__' + n]
            ref = queries[q]
            if not np.array_equal(get('bounds_px'), schema[q + '_bounds']) or get('bounds_px').dtype != np.int64:
                raise ValueError('literal native query bounds changed')
            for name, target in (('mu32', old[q + '__mean']), ('cov32', old[q + '__cov'])):
                if get(name).dtype != np.float32 or array_sha(get(name)) != array_sha(target):
                    raise ValueError('original accepted full1 moments changed')
            for name in ('a_px', 'a_front', 'a_fov'):
                if array_sha(get(name)) != array_sha(r19[q + '__' + name]):
                    raise ValueError('accepted original designated center changed')
            error = float(np.max(abs(get('a_px') - ref['anchor_px'])))
            if error > 1e-9 or get('a_px').dtype != np.float64 or not np.array_equal(get('a_front'), ref['front']) or not np.array_equal(get('a_fov'), ref['fov']):
                raise ValueError('independent all-row center/flags differ')
            a = get('a_px') / 8.; ca = N.second_moment(get('mu32'), get('cov32'), a)
            tau = N.REF.independent_temperature(ca, .25, 10.).astype(np.float32)
            for name, value in (('a64', a), ('delta64', get('mu32').astype(np.float64) - a), ('anchor_cov64', ca),
                                ('tau32', tau), ('bias32', tau[:, None] * get('content32'))):
                if get(name).dtype != value.dtype or not np.array_equal(get(name), value):
                    raise ValueError('exact common anchor/temperature arithmetic differs: ' + name)
            content = ref['features'] @ key['features'].T
            comparisons = [('mu32', ref['mean']), ('cov32', ref['cov']), ('anchor_cov64', ref['anchor_cov']),
                ('full4_nodes32', ref['nodes']), ('full4_node_cov32', ref['node_cov']), ('full4_weights32', ref['weights'])]
            errors = {name: float(np.max(abs(get(name) - value) / (1 + abs(value)))) for name, value in comparisons}
            feature_error = max(float(np.max(abs(get('q_features32') - ref['features']))), float(np.max(abs(get('k_features32') - key['features']))))
            content_error = float(np.max(abs(get('content32') - content)))
            mm, mc = N.mixture_moments(get('full4_nodes32'), get('full4_node_cov32'), get('full4_weights32'))
            residuals = {'mean': mm - get('mu32'), 'cov': mc - get('cov32')}
            normalized = {name: float(np.max(abs(value) / (1 + abs(get('mu32' if name == 'mean' else 'cov32'))))) for name, value in residuals.items()}
            for name, value in (('mixture_mean64', mm), ('mixture_cov64', mc), ('mean_residual64', residuals['mean']), ('cov_residual64', residuals['cov'])):
                if get(name).dtype != np.float64 or np.max(abs(get(name) - value) / (1 + abs(value))) > 1e-12:
                    raise ValueError('reported double mixture residual differs')
            if max(errors.values()) > 1e-4 or max(normalized.values()) > 1e-4 or max(feature_error, content_error) > 1e-5:
                raise ValueError('independent complete support/content/moment check failed')
            checks.append(dict(query=q, rows=COUNTS[q], max_projection_error_px=error, moment_errors=errors,
                full4_to_full1_normalized_residual=normalized, feature_error=feature_error, content_error=content_error,
                full4_to_full1_mean_max_absolute_residual=float(np.max(abs(residuals['mean']))),
                full4_to_full1_cov_max_absolute_residual=float(np.max(abs(residuals['cov']))), components_repaired=False))
    return inputs, saved, queries, key, schema, checks


def forward_audit(run, out):
    directory = out / 'forward_audit'
    if directory.exists() or (out / 'forward_verification.json').exists():
        raise ValueError('preserve previous forward audit attempt')
    directory.mkdir(parents=True); started = time.time(); write(directory / 'started.json', dict(created_unix=started))
    try:
        lock, design = check_audit_lock(run, out)
        if (run / 'evaluation').exists() or (run / 'summary.json').exists():
            raise ValueError('forward audit must precede actual label evaluation')
        records, supports, seal = check_seal(run, lock, design)
        if started < seal['created_unix']:
            raise ValueError('numerical audit started before global seal')
        # Shared production is imported only for clearly labelled exact replay.
        from experiments import round21_matched_decomposition as P
        parts, tensors = P.load_parts(lock)
        history = {(*identity(r), r['condition'], role): r for role, rr in read(run / 'inherited_replay_rows.json').items() for r in rr}
        by_pair = {identity(e): [r for r in records if identity(r) == identity(e)] for e in lock['entries']}
        support_map = {identity(r): r for r in supports}
        first = {s: next(identity(e) for e in lock['entries'] if e['scene'] == s) for s in SCENES}
        checks, stat_checks, replays, old_checks = [], [], [], []
        raw_count = 0
        with torch.no_grad():
            for entry in lock['entries']:
                sr = support_map[identity(entry)]
                expected_path = 'forward/' + entry['scene'].replace('/', '__') + '/arrays/' + entry['pair_name'] + '/supports.npz'
                if sr['archive_path'] != expected_path or sr['input_sha256'] != entry['input_sha256'] or sr['pair_name'] != entry['pair_name']:
                    raise ValueError('support record source identity differs')
                inputs, saved, rq, rk, schema, stats = load_independent_support(entry, sr, run, parts)
                stat_checks.append(dict(scene=entry['scene'], pair=entry['pair'], checks=stats,
                    rows=sum(s['rows'] for s in stats), max_projection_error_px=max(s['max_projection_error_px'] for s in stats)))
                banks, retained = P.prepare_pair(entry, parts, tensors)
                if {n: array_sha(v) for n, v in retained.items()} != sr['array_sha256']:
                    raise ValueError('shared-source full support replay differs')
                compare(sr['mixture_moment_residuals'], {q: P.MODEL.moment_diagnostics(banks[q]) for q in PARTITIONS}, 0.)
                for condition in CONDITIONS:
                    q, c = condition['query'], condition['condition']
                    selected = {r['family']: r for r in by_pair[identity(entry)] if r['condition'] == c}
                    path = run / selected[FAMILIES[0]]['archive_path']; check_file(path, selected[FAMILIES[0]]['archive_sha256'])
                    with np.load(path, allow_pickle=False) as archive:
                        expected = {n for f in FAMILIES for n in (f, f + '__argmax')} | {g + '__raw_mass' for g in GEOMETRIES}
                        if set(archive.files) != expected:
                            raise ValueError('saved eight-probability/four-raw schema changed')
                        for geometry in GEOMETRIES:
                            raw = P.MODEL.raw_mass(banks[q], geometry); raw_np = raw.numpy(); raw_count += 1
                            if not np.array_equal(archive[geometry + '__raw_mass'], raw_np):
                                raise ValueError('shared-source full raw-mass replay differs')
                            ps, diagnostic = P.MODEL.probabilities(banks[q], geometry, raw)
                            for state in ('zero', 'content'):
                                family = geometry + '_' + state; row = selected[family]
                                actual, winner = archive[family], archive[family + '__argmax']
                                replay = ps[state].numpy()
                                if (actual.dtype != np.float64 or actual.shape != (COUNTS[q], 432)
                                        or not np.array_equal(actual, replay) or winner.dtype != np.int32
                                        or not np.array_equal(winner, actual.argmax(1))
                                        or array_sha(actual) != row['array_sha256'] or array_sha(winner) != row['argmax_sha256']
                                        or array_sha(raw_np) != row['raw_mass_sha256'] or row['archive_sha256'] != sha(path)):
                                    raise ValueError('shared full probability/winner/hash replay differs')
                                compare(row['numerical'], diagnostic, 0.)
                                source = history.get((*identity(entry), c, family))
                                historical = inherited_probability(actual, source) if source else None
                                compare(row['inherited_replay'], historical, 0.)
                                if source:
                                    old_checks.append(dict(scene=entry['scene'], pair=entry['pair'], condition=c, family=family, **historical))
                                replays.append(dict(scene=entry['scene'], pair=entry['pair'], condition=c, family=family,
                                    array_sha256=row['array_sha256'], argmax_sha256=row['argmax_sha256'], raw_mass_sha256=row['raw_mass_sha256']))
                                if identity(entry) == first[entry['scene']]:
                                    indices = N.selected_indices(COUNTS[q])
                                    sample = {n: saved[q + '__' + n][indices] for n in ('mu32', 'cov32', 'a64', 'anchor_cov64',
                                        'full4_nodes32', 'full4_node_cov32', 'full4_weights32', 'content32', 'tau32', 'bias32')}
                                    sample['raw_mass64'] = raw_np[indices]
                                    check, evidence = N.numeric_condition(rq[q], rk, (schema['g8_bounds'] - .5) / 8., family, actual[indices], sample)
                                    check.update(scene=entry['scene'], pair=entry['pair'], condition=c, query=q, key='g8')
                                    destination = directory / 'numeric_arrays' / entry['scene'].replace('/', '__') / (c + '__' + family + '.npz')
                                    destination.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(destination, **evidence)
                                    check.update(archive_path=str(destination.relative_to(out)), archive_sha256=sha(destination)); checks.append(check)
                print('R21_REPLAY_AUDIT', entry['scene'], entry['pair_name'], flush=True)
        if (len(replays), raw_count, len(checks), sum(r['rows'] for r in stat_checks), len(old_checks)) != (1152, 576, 72, 28224, 576):
            raise ValueError('incomplete mathematical/statistical/historical replay coverage')
        if sum(r['kind'] == 'r15_commitment' and r['saved_array_checked'] for r in old_checks) != 18:
            raise ValueError('eighteen retained old T0 arrays not all rechecked')
        for name, value in (('anchor_checks', stat_checks), ('replay_records', replays), ('numerical_checks', checks), ('historical_replays', old_checks)):
            write(directory / (name + '.json'), value)
        passed = all(c['passed'] for c in checks); check_current_sources(lock)
        verification = dict(passed=passed, producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
            producer_forward_complete_sha256=sha(run / 'forward_complete.json'), producer_forward_completion_sha256=sha(run / 'forward/completion.json'),
            pairs=48, replayed_probability_matrices=1152, replayed_raw_matrices=576, independently_projected_rows=28224,
            numerical_matrices=72, numerical_probability_rows=576, numerical_unique_queries=72,
            numerical_failure_count=sum(not c['passed'] for c in checks), maximum_projection_error_px=max(x['max_projection_error_px'] for x in stat_checks),
            historical_t0_commitments=288, historical_t0_saved_arrays=18, historical_r20_saved_arrays=288,
            shared_source_replay_is_independent_mathematics=False, independent_reference_policy=N.REFERENCE_POLICY,
            actual_target_hashes_read=False, labels_decoded=False, scientific_gate_passed=False, diagnostic_only=True,
            started_unix=started, finished_unix=time.time())
        write(out / 'forward_verification.json', verification)
        write(directory / 'completion.json', dict(passed=passed, files=inventory(directory),
            verification_sha256=sha(out / 'forward_verification.json'), finished_unix=time.time()))
        if not passed:
            raise RuntimeError('frozen numerical audit failed; stop before labels and retain every check')
    except Exception as error:
        if not (directory / 'completion.json').exists():
            failure(directory, error, started)
        raise


def require_final_guards(run, out):
    lock, design = check_audit_lock(run, out); records, supports, seal = check_seal(run, lock, design)
    done = check_inventory(out / 'forward_audit'); check_file(out / 'forward_verification.json', done['verification_sha256'])
    forward = read(out / 'forward_verification.json')
    if (forward.get('passed') is not True or forward['producer_lock_sha256'] != sha(run / 'lock.json')
            or forward['producer_forward_complete_sha256'] != sha(run / 'forward_complete.json')
            or forward['audit_lock_sha256'] != sha(out / 'lock.json')
            or forward['actual_target_hashes_read'] is not False or forward['labels_decoded'] is not False):
        raise ValueError('complete numerical audit required before target bytes')
    completed, evaluation = check_inventory(run), check_inventory(run / 'evaluation')
    check_file(run / 'summary.json', evaluation['summary_sha256']); summary = read(run / 'summary.json')
    if (evaluation['labels_decoded'] is not True or evaluation['records'] != 1152 or evaluation['paired_rows'] != 1440
            or not seal['created_unix'] <= forward['started_unix'] <= forward['finished_unix'] <= read(run / 'evaluation/started.json')['created_unix']
            or completed['producer_lock_sha256'] != sha(run / 'lock.json')
            or summary['producer_lock_sha256'] != sha(run / 'lock.json')
            or summary['producer_forward_complete_sha256'] != sha(run / 'forward_complete.json')
            or summary['forward_audit_sha256'] != sha(out / 'forward_verification.json')
            or any(summary[k] is not v for k, v in (('scientific_gate_passed', False), ('diagnostic_only', True), ('no_candidate_selection', True)))):
        raise ValueError('final source/chronology/diagnostic contract differs')
    return lock, records, summary, evaluation


def audit(run, out):
    directory = out / 'final_audit'
    if directory.exists() or (out / 'verification.json').exists() or (out / 'completion.json').exists():
        raise ValueError('preserve previous final audit attempt')
    directory.mkdir(parents=True); started = time.time(); write(directory / 'started.json', dict(created_unix=started))
    try:
        lock, records, summary, evaluation = require_final_guards(run, out)
        measured_rows, measured_pairs = read(run / 'evaluation/rows.json'), read(run / 'evaluation/paired_rows.json')
        validate_records(measured_rows, lock['entries'])
        measured = {(*identity(r), r['condition'], r['family']): r for r in measured_rows}
        paired = {(*identity(r), r['condition'], r['contrast']): r for r in measured_pairs}
        if len(paired) != 1440 or len(measured_pairs) != 1440:
            raise ValueError('complete unique ten-contrast population required')
        old = {(*identity(r), r['condition'], role): r for role, rr in read(run / 'inherited_replay_rows.json').items() for r in rr}
        parts = M.REGION.literal_parts(); rows, pairs, label_sources, historical, historical_report = [], [], {}, [], []
        max_error = 0.
        for entry in lock['entries']:
            for path, digest in ((entry['label_path'], entry['label_sha256']), (entry['target_path'], entry['target_sha256'])):
                check_file(path, digest); label_sources[path] = digest
            with np.load(entry['label_path'], allow_pickle=False) as archive:
                labels = {n: archive[n] for n in ('gt_xy', 'gt_visible')}
            with np.load(entry['target_path'], allow_pickle=False) as archive:
                centers = {n: archive[n] for q in PARTITIONS for n in (q + '_point_xy', q + '_point_visible')}
            for condition in CONDITIONS:
                q, c = condition['query'], condition['condition']
                direct, common = M.REGION.independent_counts(labels, parts, q, 'g8')
                selected = {r['family']: r for r in records if identity(r) == identity(entry) and r['condition'] == c}
                with np.load(run / selected[FAMILIES[0]]['archive_path'], allow_pickle=False) as archive:
                    current, winners = {}, {}
                    for family in FAMILIES:
                        record = selected[family]; p, winner = archive[family], archive[family + '__argmax']
                        if array_sha(p) != record['array_sha256'] or array_sha(winner) != record['argmax_sha256']:
                            raise ValueError('sealed predictions changed before independent metrics')
                        region = M.region_metrics(p, parts, condition, direct, common)
                        center = M.center_metrics(winner, parts['g8']['bounds'], centers[q + '_point_xy'], centers[q + '_point_visible'])
                        row = dict(record, **region, center=center, label_sha256=entry['label_sha256'], target_sha256=entry['target_sha256'])
                        max_error = max(max_error, compare(measured[(*identity(entry), c, family)], row, 1e-8, '/metric-row'))
                        rows.append(row); current[family] = row; winners[family] = winner
                        source = old.get((*identity(entry), c, family))
                        if source:
                            keys = ('mass', 'n_query', 'n_key') + tuple(m + '_sum' for m in M.METRICS)
                            discrepancy = compare({k: row[k] for k in keys}, {k: source[k] for k in keys}, 1e-8, '/historical-metrics')
                            if source['replay_kind'] == 'r20_saved':
                                discrepancy = max(discrepancy, compare(row['center'], source['center'], 1e-8, '/historical-center'))
                            historical.append(dict(scene=entry['scene'], pair=entry['pair'], condition=c, family=family,
                                kind=source['replay_kind'], maximum_discrepancy=discrepancy))
                            historical_report.append(dict(scene=entry['scene'], pair=entry['pair'], condition=c, family=family,
                                regional_sum_differences={m + '_sum': row[m + '_sum'] - source[m + '_sum'] for m in M.METRICS},
                                center_checked=source['replay_kind'] == 'r20_saved'))
                    for contrast in M.CONTRASTS:
                        left, right = current[contrast['from_role']], current[contrast['to_role']]
                        item = dict({k: left[k] for k in ('scene', 'pair', 'condition', 'query', 'key', 'mass', 'n_query', 'n_key', 'label_sha256', 'target_sha256')},
                            contrast=contrast['name'], from_role=contrast['from_role'], to_role=contrast['to_role'],
                            **{m + '_delta_sum': right[m + '_sum'] - left[m + '_sum'] for m in M.METRICS},
                            center=M.paired_center(winners[contrast['from_role']], winners[contrast['to_role']], parts['g8']['bounds'],
                                                   centers[q + '_point_xy'], centers[q + '_point_visible']))
                        max_error = max(max_error, compare(paired[(*identity(entry), c, contrast['name'])], item, 1e-8, '/paired-row'))
                        pairs.append(item)
            print('R21_METRIC_AUDIT', entry['scene'], entry['pair_name'], flush=True)
        compare(evaluation['label_sources'], label_sources, 0.)
        compare(read(run / 'evaluation/inherited_metric_checks.json'), historical_report, 1e-8, '/historical-metric-report')
        roles = {}
        for family in FAMILIES:
            rr = [r for r in rows if r['family'] == family]
            cc = [dict(scene=r['scene'], pair=r['pair'], condition=r['condition'], **r['center']) for r in rr]
            roles[family] = M.summarize_role(rr, cc, lock['entries'])
        contrasts = {c['name']: M.summarize_contrast([r for r in pairs if r['contrast'] == c['name']], lock['entries'], c) for c in M.CONTRASTS}
        independent = dict(by_role=roles, by_contrast=contrasts, interactions=M.interactions(contrasts))
        summary_error = compare({k: summary[k] for k in independent}, independent, 1e-10, '/summary')
        for name, value in (('independent_summary', independent), ('independent_rows', rows), ('independent_paired_rows', pairs), ('historical_metric_checks', historical)):
            write(out / (name + '.json'), value)
        if len(historical) != 576:
            raise ValueError('complete four-role historical metric reproduction required')
        check_current_sources(lock)
        verification = dict(passed=True, producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
            producer_forward_complete_sha256=sha(run / 'forward_complete.json'), producer_summary_sha256=sha(run / 'summary.json'),
            producer_completion_sha256=sha(run / 'completion.json'), producer_evaluation_completion_sha256=sha(run / 'evaluation/completion.json'),
            forward_verification_sha256=sha(out / 'forward_verification.json'), independent_summary_sha256=sha(out / 'independent_summary.json'),
            pairs=48, matrices=1152, paired_rows=1440, contrasts=10, interactions=3, exact_center_counts=M.VALID_COUNTS,
            historical_metric_reproductions=576, maximum_row_metric_discrepancy=max_error, maximum_summary_discrepancy=summary_error,
            scientific_gate_passed=False, diagnostic_only=True, no_candidate_selection=True, started_unix=started, finished_unix=time.time())
        write(out / 'verification.json', verification)
        write(directory / 'completion.json', dict(passed=True, files=inventory(directory), verification_sha256=sha(out / 'verification.json'), finished_unix=time.time()))
        write(out / 'completion.json', dict(passed=True, files=inventory(out), producer_completion_sha256=sha(run / 'completion.json'),
            scientific_gate_passed=False, diagnostic_only=True, finished_unix=time.time()))
    except Exception as error:
        failure(directory, error, started)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'forward-audit', 'audit'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN); parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if any(os.environ.get(k) != '4' for k in THREAD_ENV):
        raise ValueError('launch all three thread environments as four')
    torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(limits=4):
        {'freeze': freeze, 'forward-audit': forward_audit, 'audit': audit}[args.stage](args.run.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
