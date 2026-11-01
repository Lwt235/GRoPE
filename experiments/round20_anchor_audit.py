"""R20 independent provenance/numerical/metric audit with a strict label barrier.

All-matrix replay explicitly shares the production source. Independent center,
support/integration and metric paths are separate frozen audit modules. Old
controls are inherited accepted rows, never represented as new forward work.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from experiments import round19_center_semantics_audit as H
from experiments import round20_anchor_audit_metrics as M
from experiments import round20_anchor_audit_numeric as N


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/round20_anchor_fixed'
DEFAULT_OUT = ROOT / 'results/round20_anchor_independent_audit'
SCENES, CONDITIONS, COUNTS = M.SCENES, M.CONDITIONS, M.COUNTS
FAMILIES, PARTITIONS = N.FAMILIES, N.PARTITIONS
FIELDS = ('q_to_k_xy', 'q_to_k_pixel_cov', 'q_features', 'k_features',
          'q_predicted_depth', 'q_K', 'k_K', 'q_to_k_transform')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
SOURCE_ROOTS = ('ROUND20_ANCHOR_FIXED_PROTOCOL.md', 'docs/round20_anchor_preparation_draft.md',
    'docs/round20_anchor_fixed_design_review.md', 'models/round20_anchor_pe.py',
    'tests/test_round20_anchor_pe.py', 'experiments/round20_anchor_fixed.py',
    'experiments/round20_anchor_fixed_metrics.py', 'tests/test_round20_anchor_fixed.py',
    'experiments/round20_anchor_audit.py', 'experiments/round20_anchor_audit_metrics.py',
    'experiments/round20_anchor_audit_numeric.py', 'tests/test_round20_anchor_audit.py',
    'tests/test_round20_anchor_audit_metrics.py', 'tests/test_round20_anchor_audit_numeric.py',
    'experiments/round13_mixture_mechanism.py')
sha, read, write, array_sha = H.sha, H.read, H.write, H.array_sha
inventory, check_inventory = H.inventory, H.check_inventory
RAW_BYTES = 3074609664
NUMERICAL_POLICY = dict(gaussian_order=32, quartic_order=24, reference_gaussian_orders=[64, 96],
    reference_query_rows=8, reference_pairs='first metadata pair in each scene', reference_matrices=504,
    reference_probability_rows=4032, probability_error=1e-4, reference_convergence=1e-8,
    same_raw_mass_arithmetic=1e-12, normalization_error=1e-12, anchor_error_px=1e-9,
    moment_normalized_error=1e-4, content_error=1e-5, geometry_dtype='float64', anchor_dtype='float64',
    new_covariance_dtype='float64', old_moment_dtype='float32', content_bias_dtype='float32', floor=1e-30)


def role_configs():
    result = {}
    for name in FAMILIES:
        old = ('quartic_point' if name.startswith('anchor_quartic_') else
               'gaussian_fixed' if name.startswith('anchor_fixed_') else
               'gaussian_point' if name.startswith('anchor_point_') else 'gaussian_full1')
        h, t = N.CONFIGS[name]
        result[name] = dict(h=h, temperature=t, old_family=old,
            content_policy='trace' if name in N.TRACE_FAMILIES else 'constant')
    return result


def runtime():
    if any(os.environ.get(k) != '4' for k in THREAD_ENV):
        raise ValueError('fixed four-thread process environment required')
    if (torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 4
            or not torch.are_deterministic_algorithms_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError('fixed deterministic CPU-four-thread Torch runtime required')
    pools = threadpool_info()
    if any(p['num_threads'] != 4 for p in pools):
        raise ValueError('actual BLAS/OpenMP pool thread count changed')
    build = io.StringIO()
    with redirect_stdout(build):
        np.__config__.show()
    return dict(device='cpu', cpu_threads=4, torch_interop_threads=4, python=sys.version,
        executable=sys.executable, platform=platform.platform(), numpy=np.__version__, torch=str(torch.__version__),
        scipy=importlib.metadata.version('scipy'),
        threadpoolctl=importlib.metadata.version('threadpoolctl'), thread_environment={k: os.environ[k] for k in THREAD_ENV},
        threadpools=pools, numpy_build_sha256=hashlib.sha256(build.getvalue().encode()).hexdigest(),
        torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(), deterministic=True, tf32=False)


def source_closure():
    found, pending = set(), list(SOURCE_ROOTS)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / name
        if not path.is_file():
            raise ValueError('missing joint source: ' + name)
        found.add(name)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('relative local import needs explicit review')
                if node.module:
                    modules = [node.module] + [node.module + '.' + a.name for a in node.names]
            for module in modules:
                pieces = module.split('.')
                paths = [Path(*pieces).with_suffix('.py'), Path(*pieces) / '__init__.py']
                paths += [Path(*pieces[:i]) / '__init__.py' for i in range(1, len(pieces))]
                pending.extend(str(p) for p in paths if (ROOT / p).is_file() and str(p) not in found)
    return sorted(found)


def compare(actual, expected, tolerance=1e-10, path='root'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('different keys at ' + path)
        return max((compare(actual[k], v, tolerance, path + '/' + k) for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('different list at ' + path)
        return max((compare(a, b, tolerance, path + '/' + str(i)) for i, (a, b) in enumerate(zip(actual, expected))), default=0.)
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise ValueError('invalid float at ' + path)
        error = abs(actual - expected)
        # Fraction displays, hashes, integer and screen decisions never get slack.
        if error > (0. if path.endswith('/value') else tolerance):
            raise ValueError('numeric mismatch at ' + path)
        return error
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError('exact mismatch at ' + path)
    return 0.


def check_file(path, digest):
    if sha(path) != digest:
        raise ValueError('changed file: ' + str(path))


def check_current_sources(lock):
    if runtime() != lock['runtime']:
        raise ValueError('runtime changed during audit phase')
    for name, digest in lock['source_hashes'].items():
        check_file(ROOT / name, digest)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def identity(row):
    return row['scene'], tuple(row['pair'])


def expected_old_controls(selection):
    picked = selection['selection']
    regional = ROOT / 'results/round15_regional'
    stage_a = ROOT / 'results/round15_feasibility'
    declaration = {}
    def add(name, spec, directory=regional, center=None):
        declaration[name] = dict(family=spec['family'], grid_index=spec['grid_index'], config=spec['config'],
            regional_rows=[str(directory / 'evaluation' / s.replace('/', '__') / 'rows.json') for s in SCENES],
            center_rows=[] if center is None else [str(center / 'evaluation' / s.replace('/', '__') / 'rows.json') for s in SCENES])
    for family, spec in picked['controls'].items():
        add('regional_' + family, spec)
    for name, spec in selection['legacy_configurations'].items():
        add('legacy_' + name, spec)
    for name, spec in (('strong_point', picked['strong_point']), ('legacy_point', selection['legacy_configurations']['point']),
        ('strong_quad', picked['strong_quadrature']), ('legacy_quad16', selection['legacy_configurations']['quad16']),
        ('historical_primary', picked['historical_primary'])):
        add(name, spec)
    for name, family in (('point_localizer', 'gaussian_point'), ('quartic_localizer', 'quartic_point')):
        add(name, dict(family=family, **selection['point_localization_selections'][family]), center=stage_a)
    add('legacy_point_localizer', selection['legacy_configurations']['point_pck'], center=stage_a)
    add('old_trace', picked['primary'], ROOT / 'results/round17_regional', ROOT / 'results/round17_feasibility')
    return declaration


def inherited_rows(lock):
    """Join complete selected old rows to accepted observations/targets independently."""
    entries = {identity(e): e for e in lock['entries']}
    output, cache = {}, {}
    for name, spec in lock['old_controls'].items():
        output[name] = {}
        for kind in ('regional', 'center'):
            result = []
            for filename in spec[kind + '_rows']:
                if filename not in cache:
                    cache[filename] = read(filename)
                for row in cache[filename]:
                    if (row['family'], row['grid_index']) != (spec['family'], spec['grid_index']):
                        continue
                    entry = entries.get(identity(row))
                    if (entry is None or row['input_sha256'] != entry['input_sha256']
                            or row['config'] != spec['config'] or row.get('dtype') != 'float64'
                            or not row.get('array_sha256')):
                        raise ValueError('inherited row observation/configuration identity changed')
                    item = dict(row)
                    if kind == 'regional':
                        if (row['label_sha256'] != entry['label_sha256']
                                or row['replayed_array_sha256'] != row['array_sha256']):
                            raise ValueError('inherited regional labels/replay identity changed')
                    else:
                        if row['target_sha256'] != entry['target_sha256']:
                            raise ValueError('inherited center source is not the exact R13 target')
                        item.update(key='g8', condition=row['query'] + '_g8')
                    result.append(item)
            output[name][kind] = result
        # Both region and center populations are independently verified by reductions.
        center_conditions = tuple(q + '_g8' for q in ('g8', 'g16', 'g24')) if spec['center_rows'] else ()
        M.summarize_role(output[name]['regional'], output[name]['center'], lock['entries'], center_conditions=center_conditions)
    return output


def check_producer_lock(run):
    lock = read(run / 'lock.json')
    check_file(run / 'lock.json', (run / 'lock.sha256').read_text().strip())
    expected = dict(families=list(FAMILIES), role_configs=role_configs(), conditions=list(CONDITIONS),
        counts=COUNTS, expected_valid_counts=M.VALID_COUNTS, field_allowlist=list(FIELDS),
        numerical_policy=NUMERICAL_POLICY, numerical_references=N.REFERENCE_POLICY,
        pairs=48, unique_query_rows=39888, matrices=8064, raw_probability_bytes=RAW_BYTES,
        actual_target_hashes_read=False, labels_decoded=False, scientific_gate_passed=False)
    for key, value in expected.items():
        compare(lock[key], value, 0., '/lock/' + key)
    if lock['runtime'] != runtime() or set(lock['source_hashes']) != set(source_closure()):
        raise ValueError('joint source closure or runtime changed')
    for name, digest in lock['source_hashes'].items():
        check_file(ROOT / name, digest)
        check_file(run / 'source_snapshot' / name, digest)
    forbidden = {str(absolute(e[k]).resolve()) for e in lock['entries'] for k in ('label_path', 'target_path')}
    for path, digest in lock['bindings'].items():
        resolved = absolute(path).resolve()
        if str(resolved) in forbidden or resolved.name == 'labels.npz' or 'targets' in resolved.parts:
            raise ValueError('actual label archive forbidden in pre-audit bindings')
        check_file(resolved, digest)
    def bound(path, expected=None):
        path = str(absolute(path).resolve())
        actual = lock['bindings'].get(path)
        if actual is None or (expected is not None and actual != expected):
            raise ValueError('missing inherited accepted binding: ' + path)
        return actual
    r19 = ROOT / 'results/round19_center_semantics'
    acceptance_path = ROOT / 'results/round19_center_root_acceptance_20261031.json'
    accepted = read(acceptance_path)
    if (accepted.get('accepted_for_exploratory_interpretation') is not True
            or accepted.get('full_forward_and_final_independent_audits_passed') is not True
            or accepted.get('strict_forward_before_label_chronology_checked') is not True):
        raise ValueError('accepted R19 provenance required')
    bound(acceptance_path)
    for path, digest in accepted['files'].items():
        bound(path, digest)
    old_lock = read(r19 / 'lock.json')
    if lock['source_function'] != old_lock['source_function']:
        raise ValueError('historical center source contract changed')
    for path, digest in old_lock['bindings'].items():
        bound(path, digest)
    for directory in (r19, ROOT / 'results/round19_center_semantics_independent_audit'):
        old_completed = read(directory / 'completion.json')
        if old_completed.get('passed') is not True:
            raise ValueError('accepted R19 completion required')
        for relative, digest in old_completed['files'].items():
            bound(directory / relative, digest)
        old_design = read(directory / 'lock.json')
        for name, digest in old_design['source_hashes'].items():
            # Historical notes/tests retain their old snapshots; only actually
            # imported code must additionally enter the new transitive closure.
            check_file(ROOT / name, digest)
            bound(directory / 'source_snapshot' / name, digest)
    records19 = {identity(r): r for r in read(r19 / 'forward/records.json')}
    M._cohort(lock['entries'])
    if [identity(e) for e in lock['entries']] != [identity(e) for e in old_lock['entries']]:
        raise ValueError('original ordered old-TRAIN cohort changed')
    for current, origin in zip(lock['entries'], old_lock['entries']):
        meta = read(origin['manifest_path'])
        record = records19[identity(origin)]
        expected_entry = dict(origin, label_path=str(Path(origin['path']) / 'labels.npz'),
            label_sha256=meta['label_sha256'], r19_archive_path=str(r19 / record['archive_path']),
            r19_archive_sha256=record['archive_sha256'])
        compare(current, expected_entry, 0., '/inherited-entry')
        bound(current['r19_archive_path'], current['r19_archive_sha256'])
    regional = ROOT / 'results/round17_regional'
    selection = read(regional / 'selection.json')
    bound(regional / 'selection.json', (regional / 'selection.sha256').read_text().strip())
    bound(regional / 'selection.sha256')
    bound(selection['audit_path'], selection['audit_sha256'])
    accepted_regional = read(absolute(selection['audit_path']))
    if accepted_regional.get('passed') is not True:
        raise ValueError('old regional independent audit required')
    for name in ('lock', 'forward_complete', 'summary', 'completion'):
        digest = selection['regional_' + name + '_sha256']
        bound(regional / (name + '.json'), digest)
        if accepted_regional['regional_' + name + '_sha256'] != digest:
            raise ValueError('old regional audit binding mismatch')
    compare(lock['old_controls'], expected_old_controls(selection), 0., '/old-controls')
    for spec in lock['old_controls'].values():
        for path in spec['regional_rows'] + spec['center_rows']:
            directory = Path(path).parents[2]
            bound(directory / 'completion.json')
            bound(path, read(directory / 'completion.json')['files'][str(Path(path).relative_to(directory))])
    for path, digest in read(ROOT / 'results/round17_feasibility/lock.json')['baseline_bindings'].items():
        bound(path, digest)
    parts_expected = read(regional / 'lock.json')['partitions_sha256']
    if Path(lock['partitions_path']).resolve() != regional / 'partitions.npz' or lock['partitions_sha256'] != parts_expected:
        raise ValueError('original native partitions changed')
    bound(lock['partitions_path'], parts_expected)
    for name in ('old_controls', 'inherited_rows'):
        check_file(run / (name + '.json'), lock[name + '_sha256'])
    compare(read(run / 'old_controls.json'), lock['old_controls'], 0.)
    compare(read(run / 'inherited_rows.json'), inherited_rows(lock), 0.)
    return lock


def freeze(run, out):
    if out.exists() or (run / 'forward').exists() or (run / 'forward_complete.json').exists():
        raise ValueError('joint freeze must precede any real forward attempt')
    lock = check_producer_lock(run)
    if Path(lock['audit_root']).resolve() != out.resolve():
        raise ValueError('different independent audit directory')
    out.mkdir(parents=True)
    for name in lock['source_hashes']:
        destination = out / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write(out / 'lock.json', dict(created_unix=time.time(), run=str(run.resolve()), runtime=runtime(),
        producer_lock_sha256=sha(run / 'lock.json'), source_hashes=lock['source_hashes'],
        numerical_references=N.REFERENCE_POLICY, labels_decoded=False, scientific_gate_passed=False))
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')


def check_audit_lock(run, out):
    design = read(out / 'lock.json')
    check_file(out / 'lock.json', (out / 'lock.sha256').read_text().strip())
    if (Path(design['run']).resolve() != run.resolve() or design['producer_lock_sha256'] != sha(run / 'lock.json')
            or design['runtime'] != runtime() or design['numerical_references'] != N.REFERENCE_POLICY):
        raise ValueError('audit source/runtime/reference contract changed')
    lock = check_producer_lock(run)
    if (design['source_hashes'] != lock['source_hashes'] or design['created_unix'] < lock['created_unix']
            or Path(lock['audit_root']).resolve() != out.resolve()):
        raise ValueError('joint audit freeze identity changed')
    for name, digest in design['source_hashes'].items():
        check_file(out / 'source_snapshot' / name, digest)
    return lock, design


def validate_records(records, entries):
    expected = {(*identity(e), c['condition'], f) for e in entries for c in CONDITIONS for f in FAMILIES}
    got = {(*identity(r), r['condition'], r['family']) for r in records}
    if len(got) != len(records) or got != expected:
        raise ValueError('complete 8064-matrix population required')
    cohort, conditions = {identity(e): e for e in entries}, {c['condition']: c for c in CONDITIONS}
    for row in records:
        c, entry = conditions[row['condition']], cohort[identity(row)]
        h, temperature = N.CONFIGS[row['family']]
        path = 'forward/' + row['scene'].replace('/', '__') + '/arrays/' + entry['pair_name'] + '/' + c['condition'] + '.npz'
        if (row['config'] != dict(h=h, temperature=temperature)
                or any(row[k] != c[k] for k in ('scenario', 'query', 'key'))
                or row['shape'] != [COUNTS[c['query']], COUNTS[c['key']]] or row['dtype'] != 'float64'
                or row['input_sha256'] != entry['input_sha256'] or row['pair_name'] != entry['pair_name']
                or row['archive_path'] != path):
            raise ValueError('forward record configuration/provenance changed')


def check_seal(run, lock, design):
    completed = check_inventory(run / 'forward')
    seal_stage = check_inventory(run / 'seal')
    seal = read(run / 'forward_complete.json')
    if (seal['producer_lock_sha256'] != sha(run / 'lock.json')
            or seal['forward_completion_sha256'] != sha(run / 'forward/completion.json')
            or seal['forward_records_sha256'] != sha(run / 'forward/records.json')
            or seal['support_records_sha256'] != sha(run / 'forward/support_records.json')
            or seal_stage['forward_complete_sha256'] != sha(run / 'forward_complete.json')
            or seal['pairs'] != 48 or seal['matrices'] != 8064 or seal['raw_probability_bytes'] != RAW_BYTES
            or seal['all_probabilities_retained'] is not True
            or seal['actual_target_hashes_read'] is not False or seal['labels_decoded'] is not False
            or completed['actual_target_hashes_read'] is not False or completed['labels_decoded'] is not False
            or completed['matrices'] != 8064):
        raise ValueError('global full-forward seal mismatch')
    records, supports = read(run / 'forward/records.json'), read(run / 'forward/support_records.json')
    validate_records(records, lock['entries'])
    if len(supports) != 48 or [identity(s) for s in supports] != [identity(e) for e in lock['entries']]:
        raise ValueError('complete ordered support evidence required')
    for support, entry in zip(supports, lock['entries']):
        expected_path = 'forward/' + entry['scene'].replace('/', '__') + '/arrays/' + entry['pair_name'] + '/supports.npz'
        if (support['archive_path'] != expected_path or support['pair_name'] != entry['pair_name']
                or support['input_sha256'] != entry['input_sha256']):
            raise ValueError('support archive/provenance identity changed')
    expected_files = {'records.json', 'support_records.json'}
    for scene in SCENES:
        folder = run / 'forward' / scene.replace('/', '__')
        stage = check_inventory(folder)
        start = read(folder / 'started.json')
        if (stage['scene'] != scene or stage['producer_lock_sha256'] != sha(run / 'lock.json')
                or stage['records'] != 2688 or stage['labels_decoded'] is not False
                or stage['actual_target_hashes_read'] is not False
                or start['created_unix'] != stage['started_unix'] or start['scene'] != scene
                or start['producer_lock_sha256'] != sha(run / 'lock.json')
                or not design['created_unix'] <= stage['started_unix'] <= stage['finished_unix'] <= seal['created_unix']):
            raise ValueError('scene forward chronology/source mismatch')
        if read(folder / 'records.json') != [r for r in records if r['scene'] == scene]:
            raise ValueError('scene/global probability record mismatch')
        if read(folder / 'support_records.json') != [r for r in supports if r['scene'] == scene]:
            raise ValueError('scene/global support record mismatch')
        declared_scene_files = {'started.json', 'records.json', 'support_records.json'}
        declared_scene_files.update(str((run / r['archive_path']).relative_to(folder))
            for r in records if r['scene'] == scene)
        declared_scene_files.update(str((run / r['archive_path']).relative_to(folder))
            for r in supports if r['scene'] == scene)
        if set(stage['files']) != declared_scene_files:
            raise ValueError('extra/missing native scene probability or support files')
        expected_files.update(str((folder / name).relative_to(run / 'forward')) for name in stage['files'])
        expected_files.add(str((folder / 'completion.json').relative_to(run / 'forward')))
    if set(completed['files']) != expected_files:
        raise ValueError('unexpected unsealed forward files')
    return records, supports, seal


def failure(directory, error, started):
    write(directory / 'failure.json', dict(passed=False, started_unix=started, finished_unix=time.time(),
        error_type=type(error).__name__, error=str(error)))
    if not (directory / 'completion.json').exists():
        write(directory / 'completion.json', dict(passed=False, files=inventory(directory), finished_unix=time.time()))


def independent_anchors(entry, support_record, run):
    with np.load(entry['input_path'], allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in FIELDS}
    check_file(run / support_record['archive_path'], support_record['archive_sha256'])
    with np.load(run / support_record['archive_path'], allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    if {name: array_sha(value) for name, value in saved.items()} != support_record['array_sha256']:
        raise ValueError('support canonical array identity changed')
    parts = M.REGION.literal_parts()
    errors, rows, expected_fields = [], 0, set()
    with np.load(entry['support_path'], allow_pickle=False) as old, np.load(entry['r19_archive_path'], allow_pickle=False) as r19:
        for q in PARTITIONS:
            prefix = q + '__'
            bounds = saved[prefix + 'bounds_px']
            if (bounds.dtype != np.int64 or not np.array_equal(bounds, parts[q]['bounds'])
                    or not np.array_equal(bounds, old[prefix + 'bounds'])):
                raise ValueError('native query bounds changed')
            a, front, inside = H.center_reference(inputs, bounds)
            actual_a = saved[prefix + 'a_px']
            if actual_a.dtype != np.float64 or actual_a.shape != a.shape or not np.isfinite(actual_a).all():
                raise ValueError('invalid all-row anchor')
            error = float(np.max(abs(actual_a - a)))
            if error > 1e-9:
                raise ValueError('independent all-row anchor disagreement')
            errors.append(error); rows += len(a)
            for name, reference in (('a_front', front), ('a_fov', inside)):
                if saved[prefix + name].dtype != np.bool_ or not np.array_equal(saved[prefix + name], reference):
                    raise ValueError('independent anchor flag mismatch')
            for name in ('mean', 'cov'):
                value = saved[prefix + name]
                if value.dtype != np.float32 or array_sha(value) != array_sha(old[prefix + name]):
                    raise ValueError('inherited original moments changed')
            anchor = actual_a / 8.
            delta = saved[prefix + 'mean'].astype(np.float64) - anchor
            ca = N.fixed_mean_second_moment(saved[prefix + 'mean'], saved[prefix + 'cov'], anchor)
            tau = N.GAUSSIAN.independent_temperature(ca, .25, 10.).astype(np.float32)
            for name, reference in (('a', anchor), ('delta', delta), ('anchor_cov', ca), ('tau32', tau)):
                if saved[prefix + name].dtype != reference.dtype or not np.array_equal(saved[prefix + name], reference):
                    raise ValueError('declared anchor moment/unit/temperature mismatch: ' + name)
            if q != 'mixed':
                for name in ('a_px', 'a_front', 'a_fov'):
                    if array_sha(saved[prefix + name]) != array_sha(r19[prefix + name]):
                        raise ValueError('accepted R19 anchor identity changed')
            for name in ('q_features', 'k_features'):
                feature = saved[prefix + name]
                if feature.dtype != np.float32 or feature.ndim != 2 or len(feature) != COUNTS[q] or not np.isfinite(feature).all():
                    raise ValueError('invalid executed pooled content bank')
            expected_fields.update(prefix + name for name in ('bounds_px', 'a_px', 'a_front', 'a_fov',
                'mean', 'cov', 'a', 'delta', 'anchor_cov', 'q_features', 'k_features', 'tau32'))
    if set(saved) != expected_fields:
        raise ValueError('extra/missing saved support fields')
    return inputs, saved, max(errors), rows


def prepared_sample(query, terms, retained, q, indices):
    def numpy(tensor):
        return tensor.detach().cpu().numpy()
    return dict(mean64=numpy(query['mean'])[indices], cov64=numpy(query['cov'])[indices],
        old_mean32=retained[q + '__mean'][indices], original_cov32=retained[q + '__cov'][indices],
        anchor_cov64=retained[q + '__anchor_cov'][indices], content32=numpy(terms['content32'])[indices],
        raw_mass64=numpy(terms['raw_mass'])[indices], tau32=numpy(terms['tau32'])[indices],
        bias32=numpy(terms['bias32'])[indices])


def forward_audit(run, out):
    directory = out / 'forward_audit'
    if directory.exists() or (out / 'forward_verification.json').exists():
        raise ValueError('preserve previous complete/failed forward audit attempt')
    directory.mkdir(parents=True)
    started = time.time(); write(directory / 'started.json', dict(created_unix=started))
    try:
        lock, design = check_audit_lock(run, out)
        if (run / 'evaluation').exists() or (run / 'summary.json').exists():
            raise ValueError('independent numerical audit must precede label evaluation')
        records, supports, seal = check_seal(run, lock, design)
        if started < seal['created_unix']:
            raise ValueError('audit began before global seal')
        # Explicit shared-production path: this is reproducibility, not independent math.
        from experiments import round20_anchor_fixed as P
        parts, tensors = P.load_parts(lock)
        literal_parts = M.REGION.literal_parts()
        for q in PARTITIONS:
            if not np.array_equal(parts[q]['bounds'], literal_parts[q]['bounds']):
                raise ValueError('original partition archive differs from literal geometry')
        by_pair = {identity(e): [] for e in lock['entries']}
        for record in records:
            by_pair[identity(record)].append(record)
        support_map = {identity(r): r for r in supports}
        first = {scene: next(identity(e) for e in lock['entries'] if e['scene'] == scene) for scene in SCENES}
        checks, anchors, replays = [], [], []
        with torch.no_grad():
            for entry in lock['entries']:
                inputs, saved_support, discrepancy, rows = independent_anchors(entry, support_map[identity(entry)], run)
                queries, keys, retained = P.prepare_pair(entry, parts, tensors)
                if {k: array_sha(v) for k, v in retained.items()} != support_map[identity(entry)]['array_sha256']:
                    raise ValueError('shared-source support replay differs')
                anchors.append(dict(scene=entry['scene'], pair=entry['pair'], rows=rows,
                    max_projection_error_px=discrepancy, exact_flags=True, r19_exact_reuse=True))
                numerical_pair = identity(entry) == first[entry['scene']]
                if numerical_pair:
                    reference_queries, reference_keys, schema = N.reference_banks(inputs, literal_parts)
                pair_records = by_pair[identity(entry)]
                for condition in CONDITIONS:
                    q, k, c = condition['query'], condition['key'], condition['condition']
                    selected = {r['family']: r for r in pair_records if r['condition'] == c}
                    archive_path = run / selected[FAMILIES[0]]['archive_path']
                    check_file(archive_path, selected[FAMILIES[0]]['archive_sha256'])
                    with np.load(archive_path, allow_pickle=False) as arrays:
                        if set(arrays.files) != {name for f in FAMILIES for name in (f, f + '__argmax')}:
                            raise ValueError('saved probability archive field mismatch')
                        for family in FAMILIES:
                            record = selected[family]; probability = arrays[family]; winner = arrays[family + '__argmax']
                            if (probability.dtype != np.float64 or probability.shape != tuple(record['shape'])
                                    or not np.isfinite(probability).all() or np.any(probability < 0)
                                    or np.max(abs(probability.sum(1) - 1)) > 1e-12
                                    or winner.dtype != np.int32 or not np.array_equal(winner, probability.argmax(1))
                                    or array_sha(probability) != record['array_sha256'] or array_sha(winner) != record['argmax_sha256']
                                    or record['archive_sha256'] != selected[FAMILIES[0]]['archive_sha256']):
                                raise ValueError('saved full probability/winner identity invalid')
                            h, temperature = N.CONFIGS[family]
                            terms = P.MODEL.prepared_terms(queries[q][family], keys[k][family], family, h, temperature)
                            replay = (terms['geometry'] + terms['bias32'].double()).softmax(-1).numpy()
                            if array_sha(replay) != record['array_sha256'] or array_sha(replay.argmax(1).astype(np.int32)) != record['argmax_sha256']:
                                raise ValueError('shared-source all-matrix replay differs')
                            expected_diagnostics = dict(terms['diagnostics'], all_finite=bool(np.isfinite(replay).all()),
                                max_normalization_error=float((torch.from_numpy(replay).sum(-1) - 1).abs().max()))
                            compare(record['numerical'], expected_diagnostics, 0., '/replay-diagnostics')
                            replays.append(dict(scene=entry['scene'], pair=entry['pair'], condition=c, family=family,
                                array_sha256=record['array_sha256'], argmax_sha256=record['argmax_sha256']))
                            if numerical_pair:
                                indices = N.selected_indices(COUNTS[q])
                                prepared = prepared_sample(queries[q][family], terms, retained, q, indices)
                                bounds = (schema[k + '_bounds'].astype(np.float64) - .5) / 8.
                                check, evidence = N.numeric_condition(reference_queries[q], reference_keys[k], bounds,
                                    family, probability[indices], prepared)
                                check.update(scene=entry['scene'], pair=entry['pair'], condition=c, query=q, key=k)
                                path = directory / 'numeric_arrays' / entry['scene'].replace('/', '__') / (c + '__' + family + '.npz')
                                path.parent.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(path, **evidence)
                                check.update(archive_path=str(path.relative_to(out)), archive_sha256=sha(path))
                                checks.append(check)
                print('R20_REPLAY_AUDIT', entry['scene'], entry['pair_name'], flush=True)
        if len(replays) != 8064 or sum(r['rows'] for r in anchors) != 39888 or len(checks) != 504:
            raise ValueError('incomplete independent/replay coverage')
        write(directory / 'anchor_checks.json', anchors)
        write(directory / 'replay_records.json', replays)
        write(directory / 'numerical_checks.json', checks)
        passed = all(c['passed'] for c in checks)
        check_current_sources(lock)
        verification = dict(passed=passed, producer_lock_sha256=sha(run / 'lock.json'),
            producer_forward_complete_sha256=sha(run / 'forward_complete.json'), audit_lock_sha256=sha(out / 'lock.json'),
            producer_forward_completion_sha256=sha(run / 'forward/completion.json'), pairs=48,
            replayed_probability_matrices=8064, independently_projected_rows=39888,
            numerical_matrices=504, numerical_probability_rows=4032, numerical_failure_count=sum(not c['passed'] for c in checks),
            maximum_projection_error_px=max(r['max_projection_error_px'] for r in anchors),
            shared_source_replay_is_independent_mathematics=False, independent_reference_policy=N.REFERENCE_POLICY,
            labels_decoded=False, actual_target_hashes_read=False, scientific_gate_passed=False,
            started_unix=started, finished_unix=time.time())
        write(out / 'forward_verification.json', verification)
        write(directory / 'completion.json', dict(passed=passed, files=inventory(directory),
            verification_sha256=sha(out / 'forward_verification.json'), finished_unix=time.time()))
        if not passed:
            raise RuntimeError('complete frozen numerical audit failed; preserve all checks and stop before labels')
        print('R20_FORWARD_AUDIT_PASS', flush=True)
    except Exception as error:
        if not (directory / 'completion.json').exists():
            failure(directory, error, started)
        raise


def audit(run, out):
    directory = out / 'final_audit'
    if directory.exists() or (out / 'verification.json').exists() or (out / 'completion.json').exists():
        raise ValueError('preserve previous complete/failed final audit attempt')
    directory.mkdir(parents=True)
    started = time.time(); write(directory / 'started.json', dict(created_unix=started))
    try:
        lock, design = check_audit_lock(run, out)
        records, supports, seal = check_seal(run, lock, design)
        forward_stage = check_inventory(out / 'forward_audit')
        check_file(out / 'forward_verification.json', forward_stage['verification_sha256'])
        forward = read(out / 'forward_verification.json')
        if (forward.get('passed') is not True or forward['producer_lock_sha256'] != sha(run / 'lock.json')
                or forward['producer_forward_complete_sha256'] != sha(run / 'forward_complete.json')
                or forward['audit_lock_sha256'] != sha(out / 'lock.json')
                or forward['actual_target_hashes_read'] is not False or forward['labels_decoded'] is not False):
            raise ValueError('complete accepted forward audit required before labels')
        completed, evaluation = check_inventory(run), check_inventory(run / 'evaluation')
        check_file(run / 'summary.json', evaluation['summary_sha256'])
        summary = read(run / 'summary.json')
        evaluation_start = read(run / 'evaluation/started.json')['created_unix']
        if (evaluation['labels_decoded'] is not True or evaluation['records'] != 8064
                or completed['producer_lock_sha256'] != sha(run / 'lock.json')
                or not seal['created_unix'] <= forward['started_unix'] <= forward['finished_unix'] <= evaluation_start
                or summary['producer_lock_sha256'] != sha(run / 'lock.json')
                or summary['producer_forward_complete_sha256'] != sha(run / 'forward_complete.json')
                or summary['forward_audit_sha256'] != sha(out / 'forward_verification.json')
                or summary['scientific_gate_passed'] is not False or summary['validation_or_fresh_read'] is not False
                or summary['inherited_baselines'] is not True or summary['new_baselines_optimized_for_a'] is not False):
            raise ValueError('evaluation completion/chronology/evidence attribution changed')
        measured_rows = read(run / 'evaluation/rows.json')
        validate_records(measured_rows, lock['entries'])
        measured = {(*identity(r), r['condition'], r['family']): r for r in measured_rows}
        grouped = {identity(e): [] for e in lock['entries']}
        for r in records:
            grouped[identity(r)].append(r)
        parts = M.REGION.literal_parts()
        rows, max_error, label_sources = [], 0., {}
        for entry in lock['entries']:
            # No actual target bytes are accessed above this complete guard chain.
            for path, digest in ((entry['label_path'], entry['label_sha256']), (entry['target_path'], entry['target_sha256'])):
                check_file(path, digest); label_sources[path] = digest
            with np.load(entry['label_path'], allow_pickle=False) as archive:
                labels = {name: archive[name] for name in ('gt_xy', 'gt_visible')}
            with np.load(entry['target_path'], allow_pickle=False) as archive:
                centers = {name: archive[name] for q in ('g8', 'g16', 'g24') for name in (q + '_point_xy', q + '_point_visible')}
            for condition in CONDITIONS:
                q, k, c = condition['query'], condition['key'], condition['condition']
                direct, common = M.REGION.independent_counts(labels, parts, q, k)
                selected = [r for r in grouped[identity(entry)] if r['condition'] == c]
                with np.load(run / selected[0]['archive_path'], allow_pickle=False) as arrays:
                    for record in selected:
                        family = record['family']; p = arrays[family]; winner = arrays[family + '__argmax']
                        if array_sha(p) != record['array_sha256'] or array_sha(winner) != record['argmax_sha256']:
                            raise ValueError('sealed arrays changed before independent metrics')
                        regional = M.region_metrics(p, parts, condition, direct, common)
                        center = None if q == 'mixed' else M.center_metrics(winner, parts[k]['bounds'],
                            centers[q + '_point_xy'], centers[q + '_point_visible'])
                        row = dict(record, **regional, center=center, label_sha256=entry['label_sha256'],
                                   target_sha256=entry['target_sha256'] if center else None)
                        discrepancy = compare(measured[(*identity(entry), c, family)], row, 1e-8, '/metric-row')
                        max_error = max(max_error, discrepancy); rows.append(row)
            print('R20_METRIC_AUDIT', entry['scene'], entry['pair_name'], flush=True)
        compare(evaluation['label_sources'], label_sources, 0., '/label-sources')
        center_conditions = tuple(c['condition'] for c in CONDITIONS if c['query'] != 'mixed')
        by_role = {}
        for family in FAMILIES:
            selected = [r for r in rows if r['family'] == family]
            center_rows = [dict(scene=r['scene'], pair=r['pair'], condition=r['condition'], **r['center'])
                           for r in selected if r['center'] is not None]
            by_role[family] = M.summarize_role(selected, center_rows, lock['entries'], center_conditions=center_conditions)
        inherited = inherited_rows(lock)
        old_by_role = {name: M.summarize_role(v['regional'], v['center'], lock['entries'],
            center_conditions=tuple(q + '_g8' for q in ('g8', 'g16', 'g24')) if v['center'] else ())
            for name, v in inherited.items()}
        independent = dict(by_role=by_role, old_by_role=old_by_role, screen=M.screen(by_role, old_by_role))
        summary_error = compare({key: summary[key] for key in independent}, independent, 1e-10, '/summary')
        write(out / 'independent_summary.json', independent)
        write(out / 'independent_rows.json', rows)
        check_current_sources(lock)
        verification = dict(passed=True, producer_lock_sha256=sha(run / 'lock.json'),
            producer_forward_complete_sha256=sha(run / 'forward_complete.json'),
            producer_summary_sha256=sha(run / 'summary.json'), producer_completion_sha256=sha(run / 'completion.json'),
            producer_evaluation_completion_sha256=sha(run / 'evaluation/completion.json'),
            forward_verification_sha256=sha(out / 'forward_verification.json'), audit_lock_sha256=sha(out / 'lock.json'),
            independent_summary_sha256=sha(out / 'independent_summary.json'), matrices=8064, pairs=48,
            exact_center_counts=M.VALID_COUNTS, screen_comparisons=60, maximum_row_metric_discrepancy=max_error,
            maximum_summary_discrepancy=summary_error, screen_passed=independent['screen']['passed'],
            inherited_baselines=True, scientific_gate_passed=False, started_unix=started, finished_unix=time.time())
        write(out / 'verification.json', verification)
        write(directory / 'completion.json', dict(passed=True, files=inventory(directory),
            verification_sha256=sha(out / 'verification.json'), finished_unix=time.time()))
        write(out / 'completion.json', dict(passed=True, files=inventory(out),
            producer_completion_sha256=sha(run / 'completion.json'), scientific_gate_passed=False, finished_unix=time.time()))
        print('R20_FINAL_AUDIT_PASS', flush=True)
    except Exception as error:
        failure(directory, error, started)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'forward-audit', 'audit'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if any(os.environ.get(k) != '4' for k in THREAD_ENV):
        raise ValueError('launch with all three thread environment variables set to four')
    torch.set_num_threads(4); torch.set_num_interop_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(limits=4):
        {'freeze': freeze, 'forward-audit': forward_audit, 'audit': audit}[args.stage](args.run.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
