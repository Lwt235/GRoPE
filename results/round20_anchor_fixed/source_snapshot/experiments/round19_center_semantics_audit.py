"""Independent audit of the fixed R19 old-TRAIN center-semantics diagnostic.

No producer/R13 mathematics imports. Center rays, bilinear interpolation,
nearest native cells and all metrics are reconstructed. Old means and winners
are inherited via audited provenance, not recomputed as new evidence.
This is an exploratory information diagnostic, never a scientific gate.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import json
import math
import importlib.metadata
import io
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from contextlib import redirect_stdout

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
QUERIES = ('g8', 'g16', 'g24')
COUNTS = {'g8': 432, 'g16': 108, 'g24': 48}
VALID_COUNTS = {'g8': 9521, 'g16': 2463, 'g24': 1096}
FIELDS = ('q_predicted_depth', 'q_K', 'k_K', 'q_to_k_transform')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
ROLES = ('mu', 'a', 'primary')
COMPARISONS = {'a_minus_mu': ('a', 'mu'), 'a_minus_primary': ('a', 'primary'),
               'mu_minus_primary': ('mu', 'primary')}
FLAGS = ('a_front', 'a_fov', 'mu_fov')
AUDIT_SOURCES = ('experiments/round19_center_semantics_audit.py',
                 'tests/test_round19_center_semantics_audit.py')
SOURCE_NAMES = AUDIT_SOURCES
COORD_ATOL = 1e-9
ADDITIVE_ATOL, DERIVED_ATOL, FLOAT_RTOL = 1e-8, 1e-10, 1e-12
DEFAULT_RUN = ROOT / 'results/round19_center_semantics'
DEFAULT_OUT = ROOT / 'results/round19_center_semantics_independent_audit'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)),
                        sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header + b'\n' + value.tobytes()).hexdigest()


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def native_bounds(stride):
    return np.asarray([(x, y, x + stride, y + stride)
                       for y in range(0, 144, stride)
                       for x in range(0, 192, stride)], dtype=np.int64)


def key_centers():
    return np.asarray([(x + 3.5, y + 3.5)
                       for y in range(0, 144, 8)
                       for x in range(0, 192, 8)], dtype=np.float64)


def fov(points):
    p = np.asarray(points)
    return (np.isfinite(p).all(axis=1) & (p[:, 0] >= -.5)
            & (p[:, 0] < 191.5) & (p[:, 1] >= -.5) & (p[:, 1] < 143.5))


def center_reference(inputs, bounds):
    """Independently interpolate predicted depth and solve center rays.

    Deliberately differs from production's matrix inverse and vectorized
    bilinear loop; no labels or source-visibility values are read.
    """
    raw = np.asarray(inputs['q_predicted_depth'], dtype=np.float64)
    if raw.shape != (144, 192):
        raise ValueError('unexpected predicted depth shape')
    depth = np.where(np.isnan(raw), 10., raw)
    depth = np.where(np.isposinf(depth), 20., depth)
    depth = np.where(np.isneginf(depth), .1, depth)
    depth = np.minimum(20., np.maximum(.1, depth))
    b = np.asarray(bounds)
    if b.dtype != np.int64 or b.ndim != 2 or b.shape[1] != 4:
        raise ValueError('int64 native bounds required')
    if np.any(b[:, :2] < 0) or np.any(b[:, 2:] > [192, 144]) or np.any(b[:, 2:] <= b[:, :2]):
        raise ValueError('out-of-raster bounds')
    camera_fields = ('q_K', 'k_K', 'q_to_k_transform')
    if any(np.asarray(inputs[k]).dtype != np.float64 for k in camera_fields):
        raise ValueError('historical camera archive must be float64')
    qk, kk, transform = [np.asarray(inputs[k]) for k in camera_fields]
    if qk.shape != (3, 3) or kk.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError('camera shape mismatch')
    if not all(np.isfinite(a).all() for a in (qk, kk, transform)):
        raise ValueError('nonfinite camera')
    pixel_centers = (b[:, :2].astype(np.float64) + b[:, 2:] - 1.) * .5
    samples = []
    for x, y in pixel_centers:
        ix, iy = math.floor(x), math.floor(y)
        u, v = x - ix, y - iy
        # Group by source row instead of the producer's four dy/dx additions.
        top = (1. - u) * depth[iy, ix] + u * depth[iy, ix + 1]
        bottom = (1. - u) * depth[iy + 1, ix] + u * depth[iy + 1, ix + 1]
        samples.append((1. - v) * top + v * bottom)
    hom = np.column_stack((pixel_centers, np.ones(len(b))))
    rays = np.linalg.solve(qk, hom.T).T
    source = rays * np.asarray(samples)[:, None]
    transformed = (transform[:3, :3] @ source.T).T + transform[:3, 3]
    front = transformed[:, 2] >= .1
    transformed[:, 2] = np.maximum(.1, transformed[:, 2])
    projected = (kk @ transformed.T).T
    result = np.clip(projected[:, :2] / projected[:, 2:3], -1024., 1024.)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite reference projection')
    return result, front, fov(result)


def nearest_reference(points):
    p = np.asarray(points)
    if p.dtype != np.float64 or p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('finite float64 pixel centers required')
    centers = key_centers()
    output = []
    for point in p:
        differences = centers - point
        squared = differences[:, 0]**2 + differences[:, 1]**2
        output.append(int(np.argmin(squared)))
    return np.asarray(output, dtype=np.int32)


def fraction(value):
    return dict(numerator=value.numerator, denominator=value.denominator, float=float(value))


def continuous_reference(errors):
    count = len(errors)
    total = math.fsum(map(float, errors))
    squared = math.fsum(float(v) * float(v) for v in errors)
    ordered = sorted(map(float, errors))
    middle = (ordered[(count - 1) // 2] + ordered[count // 2]) * .5 if count else None
    return dict(error_sum_px=total, error_squared_sum_px2=squared,
                mean_error_px=total / count if count else None, median_error_px=middle,
                rmse_px=math.sqrt(squared / count) if count else None)


def pair_metrics(arrays, query, target, valid):
    n = COUNTS[query]
    target, valid = np.asarray(target), np.asarray(valid)
    if (target.shape != (n, 2) or valid.shape != (n,) or valid.dtype != np.bool_
            or not np.isfinite(target[valid]).all()):
        raise ValueError('exact original target population required')
    prefix = query + '__'
    winners = {role: arrays[prefix + field] for role, field in
               (('mu', 'nearest_mu'), ('a', 'nearest_a'), ('primary', 'primary_argmax'))}
    if any(w.shape != (n,) or w.dtype != np.int32 or np.any((w < 0) | (w >= 432))
           for w in winners.values()):
        raise ValueError('invalid native winner')
    centers = key_centers()
    correct = {}
    for role, winner in winners.items():
        d = centers[winner[valid]] - target[valid]
        c = np.zeros(n, dtype=bool)
        c[valid] = d[:, 0]**2 + d[:, 1]**2 <= 64.
        correct[role] = c
    comparisons = {}
    for name, (new, old) in COMPARISONS.items():
        comparisons[name] = dict(benefit=int(np.count_nonzero(correct[new] & ~correct[old] & valid)),
            harm=int(np.count_nonzero(correct[old] & ~correct[new] & valid)),
            flip_all=int(np.count_nonzero(winners[new] != winners[old])),
            flip_valid=int(np.count_nonzero((winners[new] != winners[old]) & valid)))
    continuous, errors = {}, {}
    count = int(valid.sum())
    for role in ('a', 'mu'):
        d = arrays[prefix + role + '_px'][valid] - target[valid]
        squared = d[:, 0]**2 + d[:, 1]**2
        errors[role] = np.asarray([math.sqrt(x) for x in squared], dtype=np.float64)
        continuous[role] = continuous_reference(errors[role])
    transitions = {}
    for pr in (0, 1):
        for mu in (0, 1):
            for a in (0, 1):
                mask = valid & (correct['primary'] == bool(pr)) & (correct['mu'] == bool(mu)) & (correct['a'] == bool(a))
                transitions[f'primary{pr}_mu{mu}_a{a}'] = int(mask.sum())
    flags = {}
    for name in FLAGS:
        mask = arrays[prefix + name]
        if mask.shape != (n,) or mask.dtype != np.bool_:
            raise ValueError('invalid input-only flag')
        flags[name] = {str(flag).lower(): dict(n_query=int((mask == flag).sum()),
            n_point=int(((mask == flag) & valid).sum())) for flag in (True, False)}
    return dict(n_query=n, n_point=count, correct={r: int(v.sum()) for r, v in correct.items()},
                comparisons=comparisons, continuous=continuous, correctness_transitions=transitions, flags=flags), errors


def combine_metrics(rows, errors):
    if not rows:
        raise ValueError('empty population')
    count = sum(r['n_point'] for r in rows)
    continuous = {role: continuous_reference(np.concatenate([e[role] for e in errors]))
                  for role in ('a', 'mu')}
    return dict(n_query=sum(r['n_query'] for r in rows), n_point=count,
        correct={r: sum(row['correct'][r] for row in rows) for r in ROLES},
        comparisons={name: {k: sum(r['comparisons'][name][k] for r in rows)
                             for k in ('benefit', 'harm', 'flip_all', 'flip_valid')}
                     for name in COMPARISONS}, continuous=continuous,
        correctness_transitions={name: sum(r['correctness_transitions'][name] for r in rows)
                                 for name in rows[0]['correctness_transitions']},
        flags={name: {side: {k: sum(r['flags'][name][side][k] for r in rows)
                             for k in ('n_query', 'n_point')} for side in ('true', 'false')}
               for name in FLAGS})


def aggregate(rows, errors):
    identities = [(r['scene'], tuple(r['pair']), r['query']) for r in rows]
    if len(rows) != 144 or len(set(identities)) != 144:
        raise ValueError('exact 48-pair three-query population required')
    by_scene = {}
    for scene in SCENES:
        by_scene[scene] = {}
        for query in QUERIES:
            subset = [r for r in rows if (r['scene'], r['query']) == (scene, query)]
            if len(subset) != 16 or any(r['n_query'] != COUNTS[query] for r in subset):
                raise ValueError('missing or invalid scene/query population')
            by_scene[scene][query] = combine_metrics(subset,
                [errors[(r['scene'], tuple(r['pair']), r['query'])] for r in subset])
    macro, exact = {}, {}
    for query in QUERIES:
        pck = {role: sum((Fraction(by_scene[s][query]['correct'][role],
                    by_scene[s][query]['n_point']) for s in SCENES), Fraction()) / 3 for role in ROLES}
        delta = {name: pck[new] - pck[old] for name, (new, old) in COMPARISONS.items()}
        exact[query] = delta['a_minus_mu']
        macro[query] = dict(pck={r: fraction(v) for r, v in pck.items()},
                            deltas={n: fraction(v) for n, v in delta.items()})
    scene_strict = {s: by_scene[s]['g24']['correct']['a'] > by_scene[s]['g24']['correct']['mu'] for s in SCENES}
    conditions = dict(q24_macro_gain_at_least_1_200=exact['g24'] >= Fraction(1, 200),
        q24_all_scenes_strict=all(scene_strict.values()),
        q8_loss_at_most_1_200=exact['g8'] >= Fraction(-1, 200),
        q16_loss_at_most_1_200=exact['g16'] >= Fraction(-1, 200))
    passed = all(conditions.values())
    return dict(by_scene=by_scene, macro=macro, screen=dict(conditions=conditions,
        by_scene_q24_strict=scene_strict, passed=passed,
        next_action='consider_separately_frozen_gaussian_hypothesis' if passed else 'stop_current_anchor_information_branch',
        exploratory_only=True, scientific_gate_passed=False))


def compare(actual, reference, path=''):
    """Exact discrete values; only declared continuous statistics get tolerance."""
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or set(actual) != set(reference):
            raise ValueError('different keys at ' + path)
        return max((compare(actual[k], v, path + '/' + k) for k, v in reference.items()), default=0.)
    if isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):
            raise ValueError('different list at ' + path)
        return max((compare(a, b, path + '/' + str(i)) for i, (a, b) in enumerate(zip(actual, reference))), default=0.)
    if isinstance(reference, float):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise ValueError('invalid numeric value at ' + path)
        error = abs(actual - reference)
        if path.endswith('/float'):
            if actual != reference:
                raise ValueError('exact Fraction float mismatch at ' + path)
            return 0.
        additive = path.endswith('/error_sum_px') or path.endswith('/error_squared_sum_px2')
        tolerance = (ADDITIVE_ATOL if additive else DERIVED_ATOL) + FLOAT_RTOL * abs(reference)
        if error > tolerance:
            raise ValueError('continuous mismatch at ' + path)
        return error
    if type(actual) is not type(reference) or actual != reference:
        raise ValueError('exact value mismatch at ' + path)
    return 0.


JOINT_SOURCES = ('ROUND19_CENTER_SEMANTICS_PROTOCOL.md',
    'docs/round19_center_semantics_preparation_draft.md',
    'docs/round18_post_r17_design_review.md',
    'docs/round19_center_semantics_design_review.md', 'experiments/__init__.py',
    'experiments/round19_center_semantics.py',
    'experiments/round19_center_semantics_audit.py',
    'tests/test_round19_center_semantics_audit.py',
    'experiments/round13_mixture_mechanism.py')
SCREEN_POLICY = dict(q24_gain=dict(numerator=1, denominator=200),
    q8_q16_lower_bound=dict(numerator=-1, denominator=200),
    q24_each_scene_strict=True, exploratory_only=True)


def legacy_array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def runtime():
    if any(os.environ.get(k) != '1' for k in THREAD_ENV):
        raise ValueError('single-thread environment required')
    pools = threadpool_info()
    if any(p['num_threads'] != 1 for p in pools):
        raise ValueError('single-thread BLAS required')
    output = io.StringIO()
    with redirect_stdout(output):
        np.__config__.show()
    return dict(device='cpu', cpu_threads=1, python=sys.version, executable=sys.executable,
        numpy=np.__version__, threadpoolctl=importlib.metadata.version('threadpoolctl'),
        platform=platform.platform(), thread_environment={k: os.environ[k] for k in THREAD_ENV},
        numpy_build_sha256=hashlib.sha256(output.getvalue().encode()).hexdigest(), threadpools=pools)


def inventory(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p != directory / 'completion.json'}


def check_inventory(directory):
    completion = read(directory / 'completion.json')
    if completion.get('passed') is not True or completion['files'] != inventory(directory):
        raise ValueError('incomplete or changed inventory: ' + str(directory))
    return completion


def check_file(path, expected):
    if sha(path) != expected:
        raise ValueError('file hash mismatch: ' + str(path))


def entry_identity(entry):
    return entry['scene'], tuple(entry['pair'])


def check_producer_lock(run):
    """Verify a closed metadata chain without hashing/decoding any target bytes."""
    lock = read(run / 'lock.json')
    check_file(run / 'lock.json', (run / 'lock.sha256').read_text().strip())
    required = dict(scenes=list(SCENES), queries=list(QUERIES), counts=COUNTS,
        expected_valid_counts=VALID_COUNTS, field_allowlist=list(FIELDS), screen_policy=SCREEN_POLICY,
        rows=28224, pairs=48, labels_decoded=False, actual_target_hashes_read=False,
        exploratory_only=True, scientific_gate_passed=False, new_gaussian_predictions=False)
    for key, value in required.items():
        compare(lock[key], value, '/lock/' + key)
    if lock['runtime'] != runtime() or set(lock['source_hashes']) != set(JOINT_SOURCES):
        raise ValueError('joint source or runtime contract mismatch')
    for name, digest in lock['source_hashes'].items():
        check_file(ROOT / name, digest)
        check_file(run / 'source_snapshot' / name, digest)
    source_path = ROOT / 'experiments/round13_mixture_mechanism.py'
    source = source_path.read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name == 'predicted_centers']
    if len(nodes) != 1:
        raise ValueError('ambiguous historical center source')
    expected_function = dict(path=str(source_path.relative_to(ROOT)), name='predicted_centers',
        file_sha256=sha(source_path),
        source_sha256=hashlib.sha256(ast.get_source_segment(source, nodes[0]).encode()).hexdigest(),
        ast_sha256=hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest())
    compare(lock['source_function'], expected_function, '/source_function')
    # These are accepted metadata only. No target archive enters this hash loop.
    feasibility = ROOT / 'results/round17_feasibility'
    size = ROOT / 'results/round17_size_temperature_attribution'
    size_audit = ROOT / 'results/round17_size_temperature_independent_audit'
    old = read(feasibility / 'lock.json')
    target_contract = old['target_contract']
    old_entries = target_contract['entries']
    forbidden = {str(absolute(e['target_path']).resolve()) for e in old_entries}
    bindings = lock['bindings']
    for path, digest in bindings.items():
        resolved = str(absolute(path).resolve())
        if (resolved in forbidden or Path(path).name == 'labels.npz'
                or 'targets' in Path(resolved).parts):
            raise ValueError('target bytes forbidden before seal and forward audit')
        check_file(resolved, digest)

    def require_bound(path, expected=None):
        key = str(absolute(path).resolve())
        if key not in bindings or (expected is not None and bindings[key] != expected):
            raise ValueError('missing or incorrect inherited binding: ' + key)
        return bindings[key]

    acceptance_path = ROOT / 'results/round17_size_root_acceptance_20261031.json'
    acceptance = read(acceptance_path)
    if (acceptance.get('accepted_for_descriptive_reporting') is not True
            or acceptance.get('primary_selection_changed') is not False
            or acceptance.get('checked_current_sources_and_snapshots') is not True):
        raise ValueError('accepted unchanged primary provenance required')
    require_bound(acceptance_path)
    for path, digest in acceptance['files'].items():
        require_bound(path, digest)
    accepted = read(size_audit / 'verification.json')
    if accepted.get('passed') is not True or accepted.get('selection_changed') is not False:
        raise ValueError('inherited size audit not accepted')
    for name in ('lock', 'forward_complete', 'summary', 'completion'):
        require_bound(size / (name + '.json'), accepted['producer_' + name + '_sha256'])
    historical = {}
    for directory in (feasibility, size, size_audit):
        old_lock = read(directory / 'lock.json')
        historical[str(directory)] = old_lock
        require_bound(directory / 'lock.json', (directory / 'lock.sha256').read_text().strip())
        require_bound(directory / 'lock.sha256')
        for name, digest in old_lock['source_hashes'].items():
            require_bound(ROOT / name, digest)
            require_bound(directory / 'source_snapshot' / name, digest)
    size_lock = historical[str(size)]
    for path, digest in size_lock['prior']['files'].items():
        require_bound(path, digest)
    regional = read(ROOT / 'results/round17_regional/lock.json')
    stage_a = regional['stage_a']
    for name in ('lock', 'forward_complete', 'summary', 'completion'):
        require_bound(feasibility / (name + '.json'), stage_a['feasibility_' + name + '_sha256'])
    require_bound(feasibility / 'decision.json', stage_a['decision_sha256'])
    require_bound(stage_a['audit_path'], stage_a['audit_sha256'])
    if read(absolute(stage_a['audit_path'])).get('passed') is not True:
        raise ValueError('inherited Stage A audit failed')
    if target_contract != stage_a['target_contract'] or target_contract != size_lock['prior']['target_contract']:
        raise ValueError('target metadata contract changed')
    target_root = absolute(target_contract['root'])
    require_bound(target_root / 'lock.json', target_contract['lock_sha256'])
    require_bound(target_root / 'completion.json', target_contract['completion_sha256'])
    require_bound(target_contract['audit_path'], target_contract['audit_sha256'])
    target_audit = read(absolute(target_contract['audit_path']))
    if (target_audit.get('passed') is not True
            or target_audit['target_lock_sha256'] != target_contract['lock_sha256']
            or target_audit['target_completion_sha256'] != target_contract['completion_sha256']):
        raise ValueError('target audit binding mismatch')
    target_files = read(target_root / 'completion.json')['files']
    size_files = read(size / 'completion.json')['files']
    manifests = {entry_identity(e): e for e in old['training_manifest_records']}
    prior = {entry_identity(e): e for e in size_lock['entries']}
    records = {}
    for scene in SCENES:
        relative = 'forward/' + scene.replace('/', '__') + '/records.json'
        require_bound(size / relative, size_files[relative])
        for r in read(size / relative):
            if r['role'] == 'trace_full' and r['key'] == 'g8' and r['query'] in QUERIES:
                identity = (*entry_identity(r), r['query'])
                if identity in records:
                    raise ValueError('duplicate inherited winner')
                records[identity] = r
    entries = lock['entries']
    if (len(entries) != 48 or len({entry_identity(e) for e in entries}) != 48
            or [entry_identity(e) for e in entries] != [entry_identity(e) for e in old_entries]
            or any(sum(e['scene'] == s for e in entries) != 16 for s in SCENES) or len(records) != 144):
        raise ValueError('inherited cohort mismatch')
    for entry, origin in zip(entries, old_entries):
        identity = entry_identity(entry)
        base = absolute(origin['path']).resolve()
        previous, manifest = prior[identity], manifests[identity]
        input_path = base / 'inputs.npz'
        input_sha = old['inputs'][str(input_path.relative_to(ROOT))]
        support = size / 'forward' / entry['scene'].replace('/', '__') / 'arrays' / base.name / 'supports.npz'
        expected = dict(scene=origin['scene'], pair=origin['pair'], pair_name=base.name, path=str(base),
            input_path=str(input_path), input_sha256=input_sha,
            manifest_path=str(absolute(manifest['path'])), manifest_sha256=manifest['sha256'],
            support_path=str(support), support_sha256=size_files[str(support.relative_to(size))],
            target_path=str(absolute(origin['target_path'])), target_sha256=origin['target_sha256'], winners={})
        if (absolute(previous['path']).resolve() != base or previous['input_sha256'] != input_sha
                or previous['manifest_sha256'] != manifest['sha256']
                or previous['target_path'] != origin['target_path']
                or previous['target_sha256'] != origin['target_sha256']):
            raise ValueError('cohort provenance join failed')
        require_bound(manifest['path'], manifest['sha256'])
        meta = read(absolute(manifest['path']))
        if meta['split'] != 'train' or meta['forward_uses_sensor']:
            raise ValueError('predicted TRAIN source required')
        require_bound(input_path, input_sha)
        require_bound(support, expected['support_sha256'])
        target_relative = str(absolute(origin['target_path']).relative_to(target_root))
        if target_files[target_relative] != origin['target_sha256']:
            raise ValueError('expected target hash not inherited from completion')
        for query in QUERIES:
            record = records[(*identity, query)]
            path = size / record['archive_path']
            digest = size_files[record['archive_path']]
            require_bound(path, digest)
            if (record['input_sha256'] != input_sha or record['shape'] != [COUNTS[query], 432]
                    or record['config'] != dict(h=.25, temperature=10.)
                    or record['array_sha256'] != record['primary_regional_array_sha256']):
                raise ValueError('old primary parameter/array identity changed')
            expected['winners'][query] = dict(path=str(path), sha256=digest,
                argmax_sha256=record['argmax_sha256'], primary_regional_array_sha256=record['primary_regional_array_sha256'])
        compare(entry, expected, '/entry/' + base.name)
    return lock


def freeze(run, out):
    if out.exists() or (run / 'forward').exists() or (run / 'forward_complete.json').exists():
        raise ValueError('joint audit freeze must precede all real forward attempts')
    producer = check_producer_lock(run)
    if Path(producer['audit_root']).resolve() != out.resolve():
        raise ValueError('producer names a different audit directory')
    out.mkdir(parents=True)
    sources = {name: sha(ROOT / name) for name in JOINT_SOURCES}
    for name in sources:
        destination = out / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write(out / 'lock.json', dict(version=1, created_unix=time.time(), run=str(run.resolve()),
        producer_lock_sha256=sha(run / 'lock.json'), source_hashes=sources, runtime=runtime(),
        expected_rows=28224, expected_pairs=48, labels_decoded=False,
        exploratory_only=True, scientific_gate_passed=False))
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')


def check_audit_lock(run, out):
    design = read(out / 'lock.json')
    check_file(out / 'lock.json', (out / 'lock.sha256').read_text().strip())
    if (design['producer_lock_sha256'] != sha(run / 'lock.json')
            or Path(design['run']).resolve() != run.resolve()
            or design['runtime'] != runtime() or set(design['source_hashes']) != set(JOINT_SOURCES)):
        raise ValueError('audit design/source/runtime changed')
    for name, digest in design['source_hashes'].items():
        check_file(ROOT / name, digest)
        check_file(out / 'source_snapshot' / name, digest)
    producer = check_producer_lock(run)
    if producer['source_hashes'] != design['source_hashes'] or producer['created_unix'] > design['created_unix']:
        raise ValueError('joint chronological source freeze mismatch')
    if Path(producer['audit_root']).resolve() != out.resolve():
        raise ValueError('audit path changed')
    return producer, design


def check_seal(run, lock, design):
    seal = read(run / 'forward_complete.json')
    completion = check_inventory(run / 'forward')
    records = read(run / 'forward/records.json')
    if (seal['producer_lock_sha256'] != sha(run / 'lock.json')
            or seal['forward_completion_sha256'] != sha(run / 'forward/completion.json')
            or seal['forward_records_sha256'] != sha(run / 'forward/records.json')
            or seal['pairs'] != 48 or seal['rows'] != 28224
            or seal['labels_decoded'] is not False or seal['actual_target_hashes_read'] is not False
            or completion['labels_decoded'] is not False or completion['actual_target_hashes_read'] is not False
            or completion['producer_lock_sha256'] != sha(run / 'lock.json')
            or completion['pairs'] != 48 or completion['rows'] != 28224
            or not design['created_unix'] <= completion['started_unix'] <= completion['finished_unix'] <= seal['created_unix']):
        raise ValueError('incomplete or nonchronological global forward seal')
    started_record = read(run / 'forward/started.json')
    if (started_record['created_unix'] != completion['started_unix']
            or started_record['producer_lock_sha256'] != sha(run / 'lock.json')):
        raise ValueError('forward start record does not bind frozen source')
    if (len(records) != 48 or [(r['scene'], r['pair'], r['pair_name']) for r in records]
            != [(e['scene'], e['pair'], e['pair_name']) for e in lock['entries']]):
        raise ValueError('forward row identity/order mismatch')
    expected_files = {'started.json', 'records.json'}
    for entry, record in zip(lock['entries'], records):
        relative = 'forward/arrays/' + entry['scene'].replace('/', '__') + '/' + entry['pair_name'] + '.npz'
        if record['archive_path'] != relative or record['n_query'] != 588:
            raise ValueError('forward archive path/population mismatch')
        check_file(run / relative, record['archive_sha256'])
        expected_files.add(str(Path(relative).relative_to('forward')))
    if set(completion['files']) != expected_files:
        raise ValueError('unexpected or missing forward files')
    return records, seal


def forward_pair_audit(entry, record, run):
    with np.load(entry['input_path'], allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in FIELDS}
    with np.load(run / record['archive_path'], allow_pickle=False) as archive:
        actual = {name: archive[name] for name in archive.files}
    if {k: array_sha(v) for k, v in actual.items()} != record['array_sha256']:
        raise ValueError('forward canonical array hash mismatch')
    expected = {'key_bounds_px': native_bounds(8)}
    max_error = 0.
    with np.load(entry['support_path'], allow_pickle=False) as support:
        for query in QUERIES:
            bounds, mean = support[query + '__bounds'], support[query + '__mean']
            if (bounds.dtype != np.int64 or not np.array_equal(bounds, native_bounds(int(query[1:])))
                    or mean.dtype != np.float32 or mean.shape != (COUNTS[query], 2) or not np.isfinite(mean).all()):
                raise ValueError('inherited mean/bounds schema mismatch')
            mu = mean.astype(np.float64) * 8.
            a, front, inside = center_reference(inputs, bounds)
            with np.load(entry['winners'][query]['path'], allow_pickle=False) as archive:
                winner = archive['trace_full__argmax']
            if (winner.dtype != np.int32 or winner.shape != (COUNTS[query],)
                    or np.any(winner < 0) or np.any(winner >= 432)
                    or legacy_array_sha(winner) != entry['winners'][query]['argmax_sha256']):
                raise ValueError('inherited argmax hash/schema mismatch')
            prefix = query + '__'
            saved_a = actual[prefix + 'a_px']
            if saved_a.dtype != np.float64 or saved_a.shape != a.shape or not np.isfinite(saved_a).all():
                raise ValueError('invalid saved source center')
            discrepancy = float(np.max(np.abs(saved_a - a)))
            if discrepancy > COORD_ATOL:
                raise ValueError('independent source-center reconstruction failed')
            max_error = max(max_error, discrepancy)
            # Independent reference and saved coordinates must give exactly the same tie decisions.
            nearest_a = nearest_reference(a)
            if not np.array_equal(nearest_a, nearest_reference(saved_a)):
                raise ValueError('projection roundoff changes nearest-cell winner')
            values = dict(bounds_px=bounds, query_index=np.arange(COUNTS[query], dtype=np.int32),
                a_px=saved_a, mu_px=mu, delta_px=saved_a - mu,
                nearest_a=nearest_a, nearest_mu=nearest_reference(mu), primary_argmax=winner,
                a_finite=np.isfinite(a).all(axis=1), mu_finite=np.isfinite(mu).all(axis=1),
                a_front=front, a_fov=inside, mu_fov=fov(mu))
            expected.update({prefix + key: value for key, value in values.items()})
    if set(actual) != set(expected):
        raise ValueError('unexpected/missing saved array fields')
    for name, value in expected.items():
        if actual[name].dtype != value.dtype or not np.array_equal(actual[name], value):
            raise ValueError('independent array mismatch: ' + name)
    return max_error


def stage_failure(directory, error, started):
    write(directory / 'failure.json', dict(passed=False, started_unix=started,
        finished_unix=time.time(), error_type=type(error).__name__, error=str(error)))
    write(directory / 'completion.json', dict(passed=False, files=inventory(directory), finished_unix=time.time()))


def forward_audit(run, out):
    directory = out / 'forward_audit'
    if directory.exists() or (out / 'forward_verification.json').exists():
        raise ValueError('preserve existing forward audit attempt')
    directory.mkdir(parents=True)
    started = time.time()
    write(directory / 'started.json', dict(created_unix=started))
    try:
        lock, design = check_audit_lock(run, out)
        if (run / 'evaluation').exists() or (run / 'summary.json').exists():
            raise ValueError('forward audit must precede all target evaluation')
        records, seal = check_seal(run, lock, design)
        maximum = max(forward_pair_audit(e, r, run) for e, r in zip(lock['entries'], records))
        verification = dict(passed=True, producer_lock_sha256=sha(run / 'lock.json'),
            producer_forward_complete_sha256=sha(run / 'forward_complete.json'),
            audit_lock_sha256=sha(out / 'lock.json'), producer_forward_completion_sha256=sha(run / 'forward/completion.json'),
            rows=28224, pairs=48, maximum_projection_error_px=maximum,
            exact_all_row_nearest_indices=True, inherited_means_and_winners_verified=True,
            historical_gaussian_integration_recomputed=False, labels_decoded=False, actual_target_hashes_read=False,
            scientific_gate_passed=False, started_unix=started, finished_unix=time.time())
        if started < seal['created_unix']:
            raise ValueError('forward audit began before global seal')
        write(out / 'forward_verification.json', verification)
        write(directory / 'completion.json', dict(passed=True, files=inventory(directory),
            verification_sha256=sha(out / 'forward_verification.json'), finished_unix=time.time()))
        print('FORWARD_AUDIT_PASS', flush=True)
    except Exception as error:
        stage_failure(directory, error, started)
        raise


def audit(run, out):
    directory = out / 'final_audit'
    if directory.exists() or (out / 'verification.json').exists() or (out / 'completion.json').exists():
        raise ValueError('preserve existing final audit attempt')
    directory.mkdir(parents=True)
    started = time.time()
    write(directory / 'started.json', dict(created_unix=started))
    try:
        lock, design = check_audit_lock(run, out)
        records, seal = check_seal(run, lock, design)
        forward_completed = check_inventory(out / 'forward_audit')
        check_file(out / 'forward_verification.json', forward_completed['verification_sha256'])
        forward = read(out / 'forward_verification.json')
        if (forward.get('passed') is not True or forward['producer_lock_sha256'] != sha(run / 'lock.json')
                or forward['producer_forward_complete_sha256'] != sha(run / 'forward_complete.json')
                or forward['audit_lock_sha256'] != sha(out / 'lock.json')
                or forward['labels_decoded'] is not False or forward['actual_target_hashes_read'] is not False):
            raise ValueError('accepted forward audit required before targets')
        producer_completion = check_inventory(run)
        evaluation = check_inventory(run / 'evaluation')
        check_file(run / 'summary.json', evaluation['summary_sha256'])
        producer_summary = read(run / 'summary.json')
        evaluation_started = read(run / 'evaluation/started.json')['created_unix']
        if (producer_completion['producer_lock_sha256'] != sha(run / 'lock.json')
                or evaluation['labels_decoded'] is not True or evaluation['records'] != 144
                or not seal['created_unix'] <= forward['started_unix'] <= forward['finished_unix'] <= evaluation_started
                or producer_summary['producer_lock_sha256'] != sha(run / 'lock.json')
                or producer_summary['producer_forward_complete_sha256'] != sha(run / 'forward_complete.json')
                or producer_summary['forward_audit_sha256'] != sha(out / 'forward_verification.json')
                or producer_summary['scientific_gate_passed'] is not False
                or producer_summary['new_gaussian_predictions'] is not False
                or producer_summary['validation_or_fresh_read'] is not False):
            raise ValueError('producer evaluation chronology/bindings mismatch')
        rows, errors = [], {}
        for entry, record in zip(lock['entries'], records):
            # The first actual target-byte access is below this complete guard chain.
            check_file(entry['target_path'], entry['target_sha256'])
            with np.load(run / record['archive_path'], allow_pickle=False) as arrays, \
                    np.load(entry['target_path'], allow_pickle=False) as labels:
                for query in QUERIES:
                    metrics, distances = pair_metrics(arrays, query, labels[query + '_point_xy'],
                                                       labels[query + '_point_visible'])
                    rows.append(dict(scene=entry['scene'], pair=entry['pair'], query=query, **metrics))
                    errors[(*entry_identity(entry), query)] = distances
        if any(sum(r['n_point'] for r in rows if r['query'] == q) != VALID_COUNTS[q] for q in QUERIES):
            raise ValueError('original valid center population changed')
        row_discrepancy = compare(read(run / 'evaluation/rows.json'), rows, '/rows')
        independent = aggregate(rows, errors)
        discrepancy = compare({k: producer_summary[k] for k in independent}, independent, '/summary')
        independent.update(pairs=48, rows=28224, scientific_gate_passed=False, exploratory_only=True,
            new_gaussian_predictions=False, validation_or_fresh_read=False)
        write(out / 'independent_summary.json', independent)
        write(out / 'independent_rows.json', rows)
        verification = dict(passed=True, producer_lock_sha256=sha(run / 'lock.json'),
            producer_forward_complete_sha256=sha(run / 'forward_complete.json'),
            producer_summary_sha256=sha(run / 'summary.json'), producer_completion_sha256=sha(run / 'completion.json'),
            producer_evaluation_completion_sha256=sha(run / 'evaluation/completion.json'),
            audit_lock_sha256=sha(out / 'lock.json'), forward_verification_sha256=sha(out / 'forward_verification.json'),
            independent_summary_sha256=sha(out / 'independent_summary.json'),
            rows=28224, pairs=48, metric_records=144, exact_valid_counts=VALID_COUNTS,
            exact_discrete_metrics_and_screen=True, maximum_continuous_discrepancy=max(row_discrepancy, discrepancy),
            screen_passed=independent['screen']['passed'], scientific_gate_passed=False,
            started_unix=started, finished_unix=time.time())
        write(out / 'verification.json', verification)
        write(directory / 'completion.json', dict(passed=True, files=inventory(directory),
            verification_sha256=sha(out / 'verification.json'), finished_unix=time.time()))
        write(out / 'completion.json', dict(passed=True, files=inventory(out),
            producer_completion_sha256=sha(run / 'completion.json'), scientific_gate_passed=False,
            finished_unix=time.time()))
        print('FINAL_AUDIT_PASS', flush=True)
    except Exception as error:
        stage_failure(directory, error, started)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'forward-audit', 'audit'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for name in THREAD_ENV:
        os.environ[name] = '1'
    with threadpool_limits(limits=1):
        {'freeze': freeze, 'forward-audit': forward_audit, 'audit': audit}[args.stage](
            args.run.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
