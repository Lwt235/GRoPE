"""R23 legal-input preparation and fixed-budget applicability checks only.

Actual observation arrays are first decoded by the separately frozen prepare
stage. Every prepared input is independently audited before the frozen R22
model may run. No general numerical reference or task labels are used.
"""
from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np

from experiments import round22_ray_preflight as OLD
from experiments import round23_ray_contract as C
from models import round22_ray_pushforward as MODEL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/'results/round23_ray_applicability'
DEFAULT_AUDIT = ROOT/'results/round23_ray_applicability_independent_audit'
sha, read, write, array_sha = OLD.sha, OLD.read, OLD.write, OLD.array_sha
runtime, check_runtime = OLD.runtime, OLD.check_runtime
inventory, verify_inventory = OLD.inventory, OLD.verify_inventory
PREPARED_FIELDS = frozenset(('input__mu', 'input__factor', 'input__bounds', 'input__H', 'input__offset',
    'source__mu', 'source__factor', 'source__covariance', 'target__covariance',
    'query_bounds_px', 'query_index'))


def source_names():
    found, pending = set(), list(C.SOURCE_FILES)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT/name
        if not path.is_file():
            raise ValueError('joint source missing: '+name)
        found.add(name)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('relative source import requires review')
                if node.module:
                    names = [node.module]+[node.module+'.'+n.name for n in node.names]
            for module in names:
                parts = module.split('.')
                paths = [Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py']
                paths += [Path(*parts[:i])/'__init__.py' for i in range(1, len(parts))]
                pending.extend(str(p) for p in paths if (ROOT/p).is_file() and str(p) not in found)
    return sorted(found)


def inherited_bindings(entries):
    """Hash bytes/metadata only. Never decode an observation archive here."""
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('inherited evidence changed: '+str(path))
        if str(path) in bindings and bindings[str(path)] != digest:
            raise ValueError('conflicting source binding')
        bindings[str(path)] = digest

    for name in C.BINDINGS:
        bind(ROOT/name)
    receipt = read(ROOT/C.BINDINGS[0])
    if not all(receipt.get(k) is True for k in ('accepted_for_numerical_interpretation',
            'numerical_preflight_passed', 'all_sources_current_and_both_snapshots_verified')):
        raise ValueError('accepted bounded R22 preflight is required')
    for name, digest in receipt['files'].items():
        bind(ROOT/name, digest)
    decision = read(ROOT/'results/round22_ray_post_result_decision_20261031.json')
    if decision['root_acceptance_sha256'] != sha(ROOT/C.BINDINGS[0]):
        raise ValueError('post-result decision does not bind accepted R22 evidence')
    population_receipt = read(ROOT/'results/round20_anchor_root_acceptance_20261031.json')
    bind(C.METADATA_PATH, population_receipt['files']['results/round20_anchor_fixed/lock.json'])
    oldlock = read(ROOT/'results/round22_ray_preflight/lock.json')
    for name, digest in oldlock['source_hashes'].items():
        for prefix in (ROOT, ROOT/'results/round22_ray_preflight/source_snapshot',
                       ROOT/'results/round22_ray_independent_audit/source_snapshot'):
            bind(prefix/name, digest)
    for entry in entries:
        bind(entry['input_path'], entry['input_sha256'])
        bind(entry['manifest_path'], entry['manifest_sha256'])
        meta = read(entry['manifest_path'])
        if (meta['scene'] != entry['scene'] or meta['pair'] != entry['pair']
                or meta['split'] != 'train' or meta['forward_uses_sensor'] is not False
                or meta['input_sha256'] != entry['input_sha256']):
            raise ValueError('legal old TRAIN provenance differs')
    return bindings


def freeze(out, audit_root=DEFAULT_AUDIT):
    if out.exists():
        raise ValueError('refusing existing R23 output directory')
    entries, specs = C.entries(), C.query_specs()
    bindings = inherited_bindings(entries)
    sources = {n:sha(ROOT/n) for n in source_names()}
    observed_runtime = check_runtime()
    out.mkdir(parents=True)
    write(out/'query_specs.json', specs)
    for name, digest in sources.items():
        path = out/'source_snapshot'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, path)
        if sha(path) != digest:
            raise ValueError('source changed during freeze')
    lock = dict(version=1, created_unix=time.time(), source_hashes=sources, runtime=observed_runtime,
        entries=entries, query_specs_sha256=sha(out/'query_specs.json'), policy=C.POLICY, gates=C.GATES,
        input_keys=list(C.INPUT_KEYS), image_hw=list(C.IMAGE_HW), audit_root=str(audit_root.resolve()),
        bindings=bindings, case_count=len(specs), key_count=len(C.key_bounds()), labels_read=False,
        general_numerical_accuracy_established=False, scientific_gate_passed=False)
    write(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    verify(out)


def verify(out):
    lock = read(out/'lock.json')
    expected = dict(version=1, runtime=check_runtime(), entries=C.entries(), policy=C.POLICY,
        gates=C.GATES, input_keys=list(C.INPUT_KEYS), image_hw=list(C.IMAGE_HW),
        case_count=len(C.query_specs()), key_count=len(C.key_bounds()), labels_read=False,
        general_numerical_accuracy_established=False, scientific_gate_passed=False)
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip() or any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('frozen R23 metadata/runtime differs')
    if set(lock['source_hashes']) != set(source_names()):
        raise ValueError('source closure differs')
    for name, digest in lock['source_hashes'].items():
        if sha(ROOT/name) != digest or sha(out/'source_snapshot'/name) != digest:
            raise ValueError('current/snapshot source changed: '+name)
    if (sha(out/'query_specs.json') != lock['query_specs_sha256']
            or read(out/'query_specs.json') != C.query_specs()
            or inherited_bindings(lock['entries']) != lock['bindings']):
        raise ValueError('fixed population/provenance changed')
    return lock


def require_audit_freeze(out, lock):
    directory = Path(lock['audit_root'])
    design = read(directory/'lock.json')
    expected = dict(producer_lock_sha256=sha(out/'lock.json'), query_specs_sha256=lock['query_specs_sha256'],
                    source_hashes=lock['source_hashes'], runtime=lock['runtime'])
    if (sha(directory/'lock.json') != (directory/'lock.sha256').read_text().strip()
            or any(design.get(k) != v for k, v in expected.items())
            or design['created_unix'] < lock['created_unix']):
        raise ValueError('independent pre-input freeze differs')
    for name, digest in lock['source_hashes'].items():
        if sha(directory/'source_snapshot'/name) != digest:
            raise ValueError('independent snapshot changed')
    return sha(directory/'lock.json')


def validate_inputs(inputs, image_hw=C.IMAGE_HW):
    if set(inputs) != set(C.INPUT_KEYS):
        raise ValueError('exact six-field predicted-input allowlist required')
    height, width = image_hw
    expected = dict(q_points3=(height*width, 3), q_pixel_cov3=(height*width, 3, 3),
        q_K=(3, 3), k_K=(3, 3), q_to_k_transform=(4, 4), q_predicted_depth=(height, width))
    for name, shape in expected.items():
        value = np.asarray(inputs[name])
        if value.shape != shape:
            raise ValueError('invalid source shape: '+name)
        # Original predicted depth may contain the old declared nan/inf cases;
        # the independently audited cached 3D observations must be finite.
        if name != 'q_predicted_depth' and (value.dtype != np.float64 or not np.isfinite(value).all()):
            raise ValueError('finite float64 source required: '+name)
    for name in ('q_K', 'k_K'):
        if not np.array_equal(inputs[name][2], [0., 0., 1.]) or np.linalg.det(inputs[name]) == 0:
            raise ValueError('standard invertible pinhole intrinsics required')
    transform = inputs['q_to_k_transform']
    rotation = transform[:3, :3]
    if not np.array_equal(transform[3], [0., 0., 0., 1.]):
        raise ValueError('standard homogeneous camera transform required')
    # Preserve the supplied camera exactly. Camera storage/measurement residuals
    # are diagnostics, not an undeclared population-selection threshold.
    return dict(rotation_orthogonality_max_absolute=float(np.max(abs(rotation @ rotation.T-np.eye(3)))),
                rotation_determinant=float(np.linalg.det(rotation)))


def prepare_query(inputs, spec, *, image_hw=C.IMAGE_HW, key_bounds=None):
    """QR-compress the explicit pixel scatter/within factors; never clip rank."""
    height, width = image_hw
    bounds = np.asarray(spec['query_bounds'], np.int64)
    x0, y0, x1, y1 = map(int, bounds)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('invalid prescribed query rectangle')
    points = inputs['q_points3'].reshape(height, width, 3)[y0:y1, x0:x1].reshape(-1, 3)
    count = len(points)
    mean = points.mean(axis=0)
    basis = np.linalg.solve(inputs['q_K'], np.eye(3))[:, :2]/math.sqrt(12.)
    within = points[:, 2, None, None]*basis[None]
    factors = np.concatenate(((points-mean).T, within.transpose(1, 0, 2).reshape(3, -1)), axis=1)/math.sqrt(count)
    _, qr_upper = np.linalg.qr(factors.T, mode='reduced')
    source_factor = qr_upper.T
    source_covariance = factors @ factors.T
    rotation, translation = inputs['q_to_k_transform'][:3, :3], inputs['q_to_k_transform'][:3, 3]
    result = dict(input__mu=rotation @ mean+translation, input__factor=rotation @ source_factor,
        input__bounds=np.asarray(C.key_bounds() if key_bounds is None else key_bounds, np.float64),
        input__H=inputs['k_K'][:2, :2]/8., input__offset=inputs['k_K'][:2, 2]/8.,
        source__mu=mean, source__factor=source_factor, source__covariance=source_covariance,
        target__covariance=rotation @ source_covariance @ rotation.T,
        query_bounds_px=bounds, query_index=np.asarray(spec['query_index'], np.int64))
    if set(result) != PREPARED_FIELDS or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('nonfinite prepared geometry; no repair')
    return result


def _stage_failure(out, stage, error):
    directory = out/stage
    write(directory/'failure.json', dict(error_type=type(error).__name__, error=str(error),
        traceback=traceback.format_exc(), finished_unix=time.time(), complete_case_inventory=False,
        labels_read=False, general_numerical_accuracy_established=False, scientific_gate_passed=False))
    if (directory/'completion.json').exists():
        shutil.copyfile(directory/'completion.json', directory/'completion_before_failure.json')
    files = {k:v for k, v in inventory(directory).items() if k != 'completion.json'}
    write(directory/'completion.json', dict(files=files, failure_sha256=sha(directory/'failure.json'),
        complete_case_inventory=False, finished_unix=time.time(), scientific_gate_passed=False))


def prepare(out):
    lock = verify(out)
    audit_digest = require_audit_freeze(out, lock)
    directory = out/'prepared'
    if directory.exists() or (out/'prepared_complete.json').exists():
        raise ValueError('preserve every preparation attempt')
    directory.mkdir()
    started, clock = time.time(), time.perf_counter()
    write(directory/'started.json', dict(started_unix=started, producer_lock_sha256=sha(out/'lock.json'), audit_lock_sha256=audit_digest))
    try:
        (directory/'arrays').mkdir()
        banks, failures, load_seconds, camera_diagnostics = {}, {}, {}, {}
        for entry in lock['entries']:
            tick = time.perf_counter()
            try:
                if sha(entry['input_path']) != entry['input_sha256']:
                    raise ValueError('actual observation hash differs before decoding')
                with np.load(entry['input_path'], allow_pickle=False) as archive:
                    inputs = {k:archive[k] for k in lock['input_keys']}
                camera_diagnostics[entry['scene']] = validate_inputs(inputs)
                banks[entry['scene']] = inputs
            except Exception as error:
                failures[entry['scene']] = type(error).__name__+': '+str(error)
            load_seconds[entry['scene']] = time.perf_counter()-tick
        rows = []
        for spec in read(out/'query_specs.json'):
            tick = time.perf_counter()
            arrays = dict(query_bounds_px=np.asarray(spec['query_bounds'], np.int64),
                          query_index=np.asarray(spec['query_index'], np.int64))
            row = dict(spec, status='failed', passed=False,
                       camera_transform_diagnostics=camera_diagnostics.get(spec['scene']))
            try:
                if spec['scene'] in failures:
                    raise ValueError(failures[spec['scene']])
                arrays = prepare_query(banks[spec['scene']], spec)
                row.update(status='prepared', passed=True)
            except Exception as error:
                row['failure_reason'] = type(error).__name__+': '+str(error)
            row['preparation_seconds'] = time.perf_counter()-tick
            path = directory/'arrays'/(spec['id']+'.npz')
            np.savez_compressed(path, **arrays)
            row.update(archive_path=str(path.relative_to(out)), archive_sha256=sha(path),
                       array_sha256={k:array_sha(v) for k, v in arrays.items()})
            rows.append(row)
            write(directory/'records.json', rows)
        verify(out)
        if require_audit_freeze(out, lock) != audit_digest:
            raise ValueError('independent design changed during preparation')
        elapsed = time.perf_counter()-clock
        write(directory/'completion.json', dict(files=inventory(directory), complete_case_inventory=True,
            all_inputs_prepared=all(r['passed'] for r in rows), finished_unix=time.time(),
            preparation_phase_seconds=elapsed, archive_load_and_validation_seconds=load_seconds,
            scientific_gate_passed=False))
        seal = dict(producer_lock_sha256=sha(out/'lock.json'), audit_lock_sha256=audit_digest,
            prepared_records_sha256=sha(directory/'records.json'), prepared_completion_sha256=sha(directory/'completion.json'),
            query_specs_sha256=lock['query_specs_sha256'], case_ids=[r['id'] for r in rows], case_count=len(rows),
            all_inputs_prepared=all(r['passed'] for r in rows), started_unix=started, finished_unix=time.time(),
            preparation_phase_seconds=elapsed, labels_read=False, predictions_executed=False,
            general_numerical_accuracy_established=False, scientific_gate_passed=False)
        write(out/'prepared_complete.json', seal)
    except (Exception, KeyboardInterrupt) as error:
        _stage_failure(out, 'prepared', error)
        raise


def require_input_audit(out, lock):
    directory = Path(lock['audit_root'])
    audit_digest = require_audit_freeze(out, lock)
    verification = read(directory/'input_verification.json')
    completion = read(directory/'input_audit/completion.json')
    expected = dict(passed=True, producer_lock_sha256=sha(out/'lock.json'), audit_lock_sha256=audit_digest,
        producer_prepared_complete_sha256=sha(out/'prepared_complete.json'),
        producer_prepared_completion_sha256=sha(out/'prepared/completion.json'),
        query_specs_sha256=lock['query_specs_sha256'], checked_queries=lock['case_count'],
        source_input_hashes={e['input_path']:e['input_sha256'] for e in lock['entries']},
        labels_read=False, independent_probability_reference=False,
        general_numerical_accuracy_established=False, scientific_gate_passed=False)
    if (any(verification.get(k) != v for k, v in expected.items())
            or completion.get('passed') is not True
            or completion.get('verification_sha256') != sha(directory/'input_verification.json')):
        raise ValueError('complete independent input audit required before any model call')
    verify_inventory(directory/'input_audit', completion['files'], exclude=('completion.json',))
    prepared = read(out/'prepared/completion.json')
    verify_inventory(out/'prepared', prepared['files'], exclude=('completion.json',))
    seal = read(out/'prepared_complete.json')
    specs = read(out/'query_specs.json')
    audit_created = read(directory/'lock.json')['created_unix']
    if (seal['producer_lock_sha256'] != sha(out/'lock.json') or seal['audit_lock_sha256'] != audit_digest
            or seal['query_specs_sha256'] != lock['query_specs_sha256']
            or seal['prepared_completion_sha256'] != sha(out/'prepared/completion.json')
            or seal['prepared_records_sha256'] != sha(out/'prepared/records.json')
            or seal['case_ids'] != [s['id'] for s in specs] or seal['case_count'] != len(specs)
            or seal['all_inputs_prepared'] is not True
            or not audit_created <= seal['started_unix'] <= seal['finished_unix']
                <= verification['started_unix'] <= verification['finished_unix'] <= time.time()):
        raise ValueError('complete independently checked prepared population required')
    return sha(directory/'input_verification.json'), verification


def run(out):
    lock = verify(out)
    audit_digest = require_audit_freeze(out, lock)
    input_audit_digest, input_audit = require_input_audit(out, lock)
    directory = out/'forward'
    if directory.exists() or (out/'forward_complete.json').exists() or (out/'completion.json').exists():
        raise ValueError('preserve every applicability attempt')
    directory.mkdir()
    started, clock = time.time(), time.perf_counter()
    deadline = clock+lock['gates']['production_max_phase_wall_seconds']
    write(directory/'started.json', dict(started_unix=started, producer_lock_sha256=sha(out/'lock.json'),
        audit_lock_sha256=audit_digest, input_verification_sha256=input_audit_digest))
    try:
        (directory/'arrays').mkdir()
        rows = []
        for prepared_row in read(out/'prepared/records.json'):
            tick = time.perf_counter()
            bank = {}
            try:
                path = out/prepared_row['archive_path']
                if sha(path) != prepared_row['archive_sha256']:
                    raise ValueError('prepared input changed after independent audit')
                with np.load(path, allow_pickle=False) as archive:
                    bank = {k:archive[k] for k in PREPARED_FIELDS}
                if {k:array_sha(v) for k, v in bank.items()} != prepared_row['array_sha256']:
                    raise ValueError('prepared array identity changed')
                result = MODEL.integrate(**{k:bank['input__'+k] for k in ('mu', 'factor', 'bounds', 'H', 'offset')},
                    policy=lock['policy'], deadline=deadline)
            except Exception as error:
                result = dict(status='failed', passed=False, failure_reason=type(error).__name__+': '+str(error), work={})
            wall = time.perf_counter()-tick
            arrays = {k:v for k, v in result.items() if isinstance(v, np.ndarray)}
            path = directory/'arrays'/(prepared_row['id']+'.npz')
            np.savez_compressed(path, **arrays)
            row = {k:v for k, v in result.items() if not isinstance(v, np.ndarray)}
            row.update(id=prepared_row['id'], scene=prepared_row['scene'], pair=prepared_row['pair'],
                query_size=prepared_row['query_size'], query_index=prepared_row['query_index'],
                prepared_archive_path=prepared_row['archive_path'], prepared_archive_sha256=prepared_row['archive_sha256'],
                archive_path=str(path.relative_to(out)), archive_sha256=sha(path),
                array_sha256={k:array_sha(v) for k, v in arrays.items()}, call_and_input_load_seconds=wall,
                peak_rss_kib_process=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            rows.append(row)
            write(directory/'records.json', rows)
            print('APPLICABILITY_CASE', row['id'], row['status'], flush=True)
        phase_seconds = time.perf_counter()-clock
        phase_finished = time.time()
        verify(out)
        if require_input_audit(out, lock)[0] != input_audit_digest:
            raise ValueError('independent input audit changed during applicability phase')
        if [r['id'] for r in rows] != [s['id'] for s in read(out/'query_specs.json')]:
            raise ValueError('complete fixed population required')
        phase_passed = phase_seconds <= lock['gates']['production_max_phase_wall_seconds']
        completed = all(r['passed'] for r in rows) and phase_passed
        write(directory/'completion.json', dict(files=inventory(directory), complete_case_inventory=True,
            applicability_calls_completed=completed, phase_wall_passed=phase_passed, phase_wall_seconds=phase_seconds,
            finished_unix=time.time(), scientific_gate_passed=False))
        write(out/'forward_complete.json', dict(producer_lock_sha256=sha(out/'lock.json'), audit_lock_sha256=audit_digest,
            producer_prepared_complete_sha256=sha(out/'prepared_complete.json'), input_verification_sha256=input_audit_digest,
            forward_records_sha256=sha(directory/'records.json'), forward_completion_sha256=sha(directory/'completion.json'),
            query_specs_sha256=lock['query_specs_sha256'], case_ids=[r['id'] for r in rows], case_count=len(rows),
            started_unix=started, phase_finished_unix=phase_finished, finished_unix=time.time(),
            all_case_attempts_retained=True, applicability_calls_completed=completed,
            phase_wall_passed=phase_passed, phase_wall_seconds=phase_seconds, labels_read=False,
            general_numerical_accuracy_established=False, scientific_gate_passed=False))
        write(out/'completion.json', dict(files=inventory(out), finished_unix=time.time(),
            forward_complete_sha256=sha(out/'forward_complete.json'), complete_case_inventory=True,
            applicability_calls_completed=completed, post_phase_seconds_before_completion_write=time.perf_counter()-clock-phase_seconds,
            labels_read=False, general_numerical_accuracy_established=False, scientific_gate_passed=False))
    except (Exception, KeyboardInterrupt) as error:
        _stage_failure(out, 'forward', error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'prepare', 'run'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--audit-root', type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.out.resolve(), args.audit_root.resolve())
    else:
        (prepare if args.command == 'prepare' else run)(args.out.resolve())


if __name__ == '__main__':
    main()
