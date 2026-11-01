"""R28 authenticated R23 geometry reuse and fixed-budget applicability only.

The independent input gate precedes every immutable R26 model call. No source
patch is reconstructed, and no original observation or label archive is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np

from experiments import round28_rms_contract as C
from models import round26_rms_projective_pe as MODEL

DEFAULT_OUT = C.ROOT/'results/round28_rms_applicability'
DEFAULT_AUDIT = C.ROOT/'results/round28_rms_applicability_independent_audit'
SCOPE = dict(general_numerical_accuracy_established=False,
    independent_probability_reference=False, labels_read=False, efficacy_tested=False,
    candidate_selected=False, scientific_gate_passed=False, goal_complete=False)
PREPARED_FIELDS = frozenset({'r23__'+name for name in C.OLD_FIELDS} |
    {'mapping_matrix', 'input__mu', 'input__factor', 'input__bounds', 'input__content_bias'})
MODEL_FIELDS = ('mu', 'factor', 'bounds', 'content_bias')
IDENTITY_FIELDS = ('id', 'scene', 'pair', 'pair_name', 'query_size', 'query_index', 'query_bounds')


def freeze(out, audit_root=DEFAULT_AUDIT):
    out, audit_root = Path(out), Path(audit_root)
    if out.exists():
        raise ValueError('refusing an existing R28 attempt')
    bindings, specs = C.inherited_evidence()
    sources = {name:C.sha(C.ROOT/name) for name in C.source_names()}
    runtime = C.check_runtime()
    out.mkdir(parents=True)
    C.write(out/'query_specs.json', specs)
    for name, digest in sources.items():
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT/name, target)
        if C.sha(target) != digest:
            raise ValueError('source changed during freeze')
    C.write(out/'lock.json', dict(version=1, created_unix=time.time(), source_hashes=sources,
        runtime=runtime, policy=C.POLICY, gates=C.GATES, bindings=bindings,
        query_specs_sha256=C.sha(out/'query_specs.json'), case_ids=[s['id'] for s in specs],
        case_count=72, key_count=432, audit_root=str(audit_root.resolve()), **SCOPE))
    (out/'lock.sha256').write_text(C.sha(out/'lock.json')+'\n')
    verify(out)


def verify(out):
    out = Path(out)
    lock = C.read(out/'lock.json')
    bindings, specs = C.inherited_evidence()
    expected = dict(version=1, runtime=C.check_runtime(), policy=C.POLICY, gates=C.GATES,
        bindings=bindings, case_ids=[s['id'] for s in specs], case_count=72, key_count=432,
        query_specs_sha256=C.sha(out/'query_specs.json'), **SCOPE)
    if (C.sha(out/'lock.json') != (out/'lock.sha256').read_text().strip()
            or any(lock.get(k) != v for k,v in expected.items())
            or C.read(out/'query_specs.json') != specs):
        raise ValueError('frozen R28 population/runtime/provenance differs')
    if set(lock['source_hashes']) != set(C.source_names()):
        raise ValueError('joint source closure differs')
    for name,digest in lock['source_hashes'].items():
        if C.sha(C.ROOT/name) != digest or C.sha(out/'source_snapshot'/name) != digest:
            raise ValueError('current or producer snapshot source differs: '+name)
    return lock


def require_audit_freeze(out, lock):
    directory = Path(lock['audit_root'])
    design = C.read(directory/'lock.json')
    expected = dict(run=str(out.resolve()), producer_root=str(out.resolve()),
        producer_lock_sha256=C.sha(out/'lock.json'), query_specs_sha256=lock['query_specs_sha256'],
        source_hashes=lock['source_hashes'], runtime=lock['runtime'], **SCOPE)
    if (C.sha(directory/'lock.json') != (directory/'lock.sha256').read_text().strip()
            or any(design.get(k) != v for k,v in expected.items())
            or not lock['created_unix'] <= design['created_unix'] <= time.time()):
        raise ValueError('complete independent pre-input freeze required')
    for name,digest in lock['source_hashes'].items():
        if C.sha(directory/'source_snapshot'/name) != digest:
            raise ValueError('independent source snapshot differs: '+name)
    return C.sha(directory/'lock.json')


def load_old(spec):
    path = Path(spec['source_archive_path'])
    if C.sha(path) != spec['archive_sha256']:
        raise ValueError('old prepared archive bytes changed')
    # Archive metadata may contain additional historical fields; decode only
    # these eight, without touching source__* or any original observations.
    with np.load(path, allow_pickle=False) as archive:
        old = {name:archive[name] for name in C.OLD_FIELDS}
    if any(C.array_sha(value) != spec['array_sha256'][name] for name,value in old.items()):
        raise ValueError('old prepared canonical array identity changed')
    return old


def map_geometry(old, spec):
    """Map target camera Gaussian into the original reference-8 coordinates."""
    if set(old) != set(C.OLD_FIELDS):
        raise ValueError('exact eight-field inherited input boundary required')
    shapes = dict(input__mu=(3,), input__factor=(3,3), input__bounds=(432,4),
        input__H=(2,2), input__offset=(2,), target__covariance=(3,3),
        query_bounds_px=(4,), query_index=())
    for name,shape in shapes.items():
        value = np.asarray(old[name])
        dtype = np.int64 if name in ('query_bounds_px','query_index') else np.float64
        if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('invalid inherited dtype/shape/finite field: '+name)
    if (int(old['query_index']) != spec['query_index']
            or not np.array_equal(old['query_bounds_px'], np.asarray(spec['query_bounds'], np.int64))):
        raise ValueError('fixed query identity differs')
    a = np.eye(3, dtype=np.float64)
    a[:2,:2] = old['input__H']
    a[:2,2] = old['input__offset']
    with np.errstate(over='raise', invalid='raise'):
        mean, factor = a @ old['input__mu'], a @ old['input__factor']
    result = {'r23__'+name:value.copy() for name,value in old.items()}
    result.update(mapping_matrix=a, input__mu=mean, input__factor=factor,
        input__bounds=old['input__bounds'].copy(), input__content_bias=np.zeros(432, np.float64))
    if set(result) != PREPARED_FIELDS or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('nonfinite homogeneous mapping; no repair')
    return result


def stage_failure(out, stage, error):
    directory = out/stage
    C.write(directory/'failure.json', dict(error_type=type(error).__name__, error=str(error),
        traceback=traceback.format_exc(), finished_unix=time.time(), complete_case_inventory=False, **SCOPE))
    if (directory/'completion.json').exists():
        shutil.copyfile(directory/'completion.json', directory/'completion_before_failure.json')
    files = {name:digest for name,digest in C.inventory(directory).items() if name != 'completion.json'}
    C.write(directory/'completion.json', dict(files=files, failure_sha256=C.sha(directory/'failure.json'),
        complete_case_inventory=False, all_inputs_prepared=False, applicability_calls_completed=False,
        finished_unix=time.time(), **SCOPE))


def prepare(out):
    out = Path(out)
    lock = verify(out)
    audit_digest = require_audit_freeze(out, lock)
    directory = out/'prepared'
    if directory.exists() or (out/'prepared_complete.json').exists():
        raise ValueError('preserve every prior geometry preparation attempt')
    directory.mkdir()
    started, clock = time.time(), time.perf_counter()
    try:
        C.write(directory/'started.json', dict(started_unix=started,
            producer_lock_sha256=C.sha(out/'lock.json'), audit_lock_sha256=audit_digest, **SCOPE))
        (directory/'arrays').mkdir()
        rows = []
        for spec in C.read(out/'query_specs.json'):
            tick = time.perf_counter()
            arrays = {}
            row = {name:spec[name] for name in IDENTITY_FIELDS}
            row.update(source_archive_path=spec['source_archive_path'],
                source_archive_sha256=spec['archive_sha256'], status='failed', passed=False)
            try:
                old = load_old(spec)
                arrays = {'r23__'+name:value.copy() for name,value in old.items()}
                arrays = map_geometry(old, spec)
                row.update(status='prepared', passed=True)
            except Exception as error:
                row['failure_reason'] = type(error).__name__+': '+str(error)
            path = directory/'arrays'/(spec['id']+'.npz')
            if path.exists():
                raise ValueError('refusing to overwrite a prepared case')
            np.savez_compressed(path, **arrays)
            row.update(preparation_seconds=time.perf_counter()-tick,
                archive_path=str(path.relative_to(out)), archive_sha256=C.sha(path),
                array_sha256={name:C.array_sha(v) for name,v in arrays.items()})
            rows.append(row)
            C.write(directory/'records.json', rows)
        phase_seconds = time.perf_counter()-clock
        verify(out)
        if require_audit_freeze(out, lock) != audit_digest:
            raise ValueError('independent source freeze changed during preparation')
        if [r['id'] for r in rows] != lock['case_ids']:
            raise ValueError('complete fixed preparation population required')
        C.write(directory/'completion.json', dict(files=C.inventory(directory), complete_case_inventory=True,
            all_inputs_prepared=all(r['passed'] for r in rows), preparation_phase_seconds=phase_seconds,
            finished_unix=time.time(), **SCOPE))
        C.write(out/'prepared_complete.json', dict(producer_lock_sha256=C.sha(out/'lock.json'),
            audit_lock_sha256=audit_digest, prepared_records_sha256=C.sha(directory/'records.json'),
            prepared_completion_sha256=C.sha(directory/'completion.json'), query_specs_sha256=lock['query_specs_sha256'],
            case_ids=[r['id'] for r in rows], case_count=len(rows), all_inputs_prepared=all(r['passed'] for r in rows),
            started_unix=started, finished_unix=time.time(), preparation_phase_seconds=phase_seconds,
            predictions_executed=False, **SCOPE))
    except (Exception, KeyboardInterrupt) as error:
        stage_failure(out, 'prepared', error)
        raise


def require_input_audit(out, lock):
    directory = Path(lock['audit_root'])
    digest = require_audit_freeze(out, lock)
    seal = C.read(out/'prepared_complete.json')
    done = C.read(out/'prepared/completion.json')
    C.verify_inventory(out/'prepared', done['files'], exclude=('completion.json',))
    specs, rows = C.read(out/'query_specs.json'), C.read(out/'prepared/records.json')
    expected_seal = dict(producer_lock_sha256=C.sha(out/'lock.json'), audit_lock_sha256=digest,
        prepared_records_sha256=C.sha(out/'prepared/records.json'),
        prepared_completion_sha256=C.sha(out/'prepared/completion.json'), query_specs_sha256=lock['query_specs_sha256'],
        case_ids=lock['case_ids'], case_count=lock['case_count'], all_inputs_prepared=True,
        predictions_executed=False, **SCOPE)
    expected_files = {'started.json','records.json'} | {'arrays/'+s['id']+'.npz' for s in specs}
    if (any(seal.get(k) != v for k,v in expected_seal.items())
            or done.get('complete_case_inventory') is not True or done.get('all_inputs_prepared') is not True
            or set(done['files']) != expected_files or [r['id'] for r in rows] != lock['case_ids']
            or not all(r['passed'] is True and r['status'] == 'prepared' for r in rows)):
        raise ValueError('complete successful global geometry seal required')
    for row,spec in zip(rows,specs):
        if (row['archive_path'] != 'prepared/arrays/'+spec['id']+'.npz'
                or C.sha(out/row['archive_path']) != row['archive_sha256']
                or set(row['array_sha256']) != PREPARED_FIELDS
                or any(row[k] != spec[k] for k in IDENTITY_FIELDS)
                or row['source_archive_path'] != spec['source_archive_path']
                or row['source_archive_sha256'] != spec['archive_sha256']):
            raise ValueError('prepared source identity differs')
    verification = C.read(directory/'input_verification.json')
    completion = C.read(directory/'input_audit/completion.json')
    expected = dict(passed=True, checked_queries=lock['case_count'],
        producer_lock_sha256=C.sha(out/'lock.json'), audit_lock_sha256=digest,
        producer_prepared_complete_sha256=C.sha(out/'prepared_complete.json'),
        producer_prepared_completion_sha256=C.sha(out/'prepared/completion.json'),
        query_specs_sha256=lock['query_specs_sha256'],
        source_prepared_hashes={s['source_archive_path']:s['archive_sha256'] for s in specs}, **SCOPE)
    if (any(verification.get(k) != v for k,v in expected.items())
            or completion.get('passed') is not True
            or completion.get('verification_sha256') != C.sha(directory/'input_verification.json')):
        raise ValueError('complete independent input audit required before any model call')
    C.verify_inventory(directory/'input_audit', completion['files'], exclude=('completion.json',))
    checks = C.read(directory/'input_audit/checks.json')
    start, design = C.read(out/'prepared/started.json'), C.read(directory/'lock.json')
    if ([r['id'] for r in checks] != lock['case_ids'] or not all(r['passed'] is True for r in checks)
            or start['started_unix'] != seal['started_unix']
            or start['producer_lock_sha256'] != expected['producer_lock_sha256']
            or start['audit_lock_sha256'] != digest
            or not design['created_unix'] <= seal['started_unix'] <= seal['finished_unix']
                <= verification['started_unix'] <= verification['finished_unix'] <= time.time()):
        raise ValueError('all input checks and input-before-forward chronology required')
    return C.sha(directory/'input_verification.json'), rows


def append_record(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
        stream.flush()


def run_case(out, prepared_row, policy, deadline):
    """Retain a single attempted or deadline-excluded case without fake output."""
    directory = out/'forward'
    started, tick = time.time(), time.perf_counter()
    arrays, attempted = {}, False
    try:
        if time.perf_counter() >= deadline:
            raise TimeoutError('production phase deadline before input load or model call')
        source = out/prepared_row['archive_path']
        if C.sha(source) != prepared_row['archive_sha256']:
            raise ValueError('prepared archive changed after independent audit')
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != PREPARED_FIELDS:
                raise ValueError('prepared field population differs')
            bank = {name:archive[name] for name in PREPARED_FIELDS}
        if {name:C.array_sha(value) for name,value in bank.items()} != prepared_row['array_sha256']:
            raise ValueError('prepared canonical arrays changed after input audit')
        arrays = {'input__'+name:bank['input__'+name] for name in MODEL_FIELDS}
        if time.perf_counter() >= deadline:
            raise TimeoutError('production phase deadline before model call')
        attempted = True
        append_record(directory/'call_starts.jsonl', dict(id=prepared_row['id'], started_unix=time.time()))
        result = MODEL.integrate(**{name:arrays['input__'+name] for name in MODEL_FIELDS},
            policy=policy, deadline=deadline)
    except Exception as error:
        result = dict(status='failed', passed=False, failure_reason=type(error).__name__+': '+str(error), work={})
    arrays.update({name:value for name,value in result.items() if isinstance(value, np.ndarray)})
    if result['passed']:
        arrays['argmax'] = np.asarray(np.argmax(result['probability']), np.int32)
    metadata = {name:value for name,value in result.items() if not isinstance(value, np.ndarray)}
    hashes = {name:C.array_sha(value) for name,value in arrays.items()}
    append_record(directory/'model_results.jsonl', dict(metadata, id=prepared_row['id'], attempted=attempted,
        available_array_sha256=hashes))
    path = directory/'arrays'/(prepared_row['id']+'.npz')
    if path.exists():
        raise ValueError('refusing to overwrite a model outcome')
    np.savez_compressed(path, **arrays)
    wall = time.perf_counter()-tick
    return dict(metadata, **{name:prepared_row[name] for name in IDENTITY_FIELDS},
        prepared_archive_path=prepared_row['archive_path'], prepared_archive_sha256=prepared_row['archive_sha256'],
        archive_path=str(path.relative_to(out)), archive_sha256=C.sha(path), array_sha256=hashes,
        attempted=attempted, started_unix=started, finished_unix=time.time(), wall_seconds=wall,
        call_and_input_load_seconds=wall, peak_rss_kib_process=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        peak_rss_scope='cumulative process high-water mark; not exclusive per-case memory')


def run(out):
    out = Path(out)
    lock = verify(out)
    audit_digest = require_audit_freeze(out, lock)
    input_digest, prepared_rows = require_input_audit(out, lock)
    directory = out/'forward'
    if directory.exists() or (out/'forward_complete.json').exists() or (out/'completion.json').exists():
        raise ValueError('preserve every prior applicability attempt')
    directory.mkdir()
    started, clock = time.time(), time.perf_counter()
    deadline = clock+lock['gates']['production_max_phase_wall_seconds']
    try:
        C.write(directory/'started.json', dict(started_unix=started, producer_lock_sha256=C.sha(out/'lock.json'),
            audit_lock_sha256=audit_digest, input_verification_sha256=input_digest, **SCOPE))
        (directory/'arrays').mkdir()
        (directory/'call_starts.jsonl').touch(exist_ok=False)
        (directory/'model_results.jsonl').touch(exist_ok=False)
        rows = []
        for prepared_row in prepared_rows:
            row = run_case(out, prepared_row, lock['policy'], deadline)
            rows.append(row)
            C.write(directory/'records.json', rows)
            print('APPLICABILITY_CASE', row['id'], row['status'], flush=True)
        elapsed, phase_finished = time.perf_counter()-clock, time.time()
        verify(out)
        if (require_audit_freeze(out, lock) != audit_digest
                or require_input_audit(out, lock)[0] != input_digest
                or [r['id'] for r in rows] != lock['case_ids']):
            raise ValueError('input audit/source/population changed during applicability')
        phase_passed = elapsed <= lock['gates']['production_max_phase_wall_seconds']
        completed = all(r['passed'] is True for r in rows) and phase_passed
        C.write(directory/'completion.json', dict(files=C.inventory(directory), complete_case_inventory=True,
            applicability_calls_completed=completed, phase_wall_passed=phase_passed, phase_wall_seconds=elapsed,
            finished_unix=time.time(), **SCOPE))
        C.write(out/'forward_complete.json', dict(producer_lock_sha256=C.sha(out/'lock.json'),
            audit_lock_sha256=audit_digest, producer_prepared_complete_sha256=C.sha(out/'prepared_complete.json'),
            input_verification_sha256=input_digest, query_specs_sha256=lock['query_specs_sha256'],
            forward_records_sha256=C.sha(directory/'records.json'),
            forward_completion_sha256=C.sha(directory/'completion.json'), case_ids=[r['id'] for r in rows],
            case_count=len(rows), all_case_attempts_retained=True, all_available_partial_arrays_retained=True,
            attempted_calls=sum(r['attempted'] for r in rows), applicability_calls_completed=completed,
            started_unix=started, phase_finished_unix=phase_finished, finished_unix=time.time(),
            phase_wall_seconds=elapsed, phase_wall_passed=phase_passed,
            phase_timing_scope='started through final forward records write; provenance finalization separate', **SCOPE))
        C.write(out/'completion.json', dict(files=C.inventory(out),
            forward_complete_sha256=C.sha(out/'forward_complete.json'), complete_case_inventory=True,
            applicability_calls_completed=completed, finished_unix=time.time(),
            post_phase_bookkeeping_seconds_before_completion_write=time.perf_counter()-clock-elapsed,
            total_seconds_before_completion_write=time.perf_counter()-clock, **SCOPE))
    except (Exception, KeyboardInterrupt) as error:
        stage_failure(out, 'forward', error)
        if (out/'completion.json').exists():
            shutil.copyfile(out/'completion.json', out/'completion_before_failure.json')
        C.write(out/'completion.json', dict(files={n:h for n,h in C.inventory(out).items() if n != 'completion.json'},
            complete_case_inventory=False, applicability_calls_completed=False, finished_unix=time.time(), **SCOPE))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','prepare','run'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--audit-root', type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.out.resolve(), args.audit_root.resolve())
    else:
        (prepare if args.command == 'prepare' else run)(args.out.resolve())


if __name__ == '__main__':
    main()
