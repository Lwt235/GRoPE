"""R33 sealed real inputs and fixed-budget independent reference calls.

The independent input gate precedes all 144 reference calls. No production
model is imported, and old prediction arrays are never decoded here.
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

from experiments import round33_log_real_contract as C
from experiments import round32_log_conditional_reference as REF

DEFAULT_OUT = C.ROOT/'results/round33_log_real_reference'
DEFAULT_AUDIT = C.ROOT/'results/round33_log_real_reference_independent_audit'
SCOPE = C.SCOPE

PREPARED_FIELDS = frozenset(C.OLD_FIELDS)
INPUT_FIELDS = ('mu', 'factor', 'bounds', 'content_bias')
IDENTITY_FIELDS = ('id','scene','pair','pair_name','query_size','query_index','query_bounds')


def call_id(base, order):
    return base+'__order'+str(order)


def ordered_calls(specs):
    return [call_id(s['id'],o) for s in specs for o in C.REFERENCE_POLICY['orders']]


def freeze(out, audit_root=DEFAULT_AUDIT):
    out, audit_root = Path(out), Path(audit_root)
    if out.exists():
        raise ValueError('refusing an existing R33 attempt')
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
        runtime=runtime, policy=C.POLICY, reference_policy=C.REFERENCE_POLICY, gates=C.GATES, fail_fast=C.FAIL_FAST, bindings=bindings, real_inputs_read=False, independent_real_reference=False,
        query_specs_sha256=C.sha(out/'query_specs.json'),
        case_ids=[s['id'] for s in specs], ordered_call_ids=ordered_calls(specs),
        case_count=C.CASE_COUNT, call_count=C.CALL_COUNT, audit_root=str(audit_root.resolve()), **SCOPE))
    (out/'lock.sha256').write_text(C.sha(out/'lock.json')+'\n')
    verify(out)


def verify(out):
    out = Path(out)
    lock = C.read(out/'lock.json')
    bindings, specs = C.inherited_evidence()
    expected = dict(version=1, runtime=C.check_runtime(), policy=C.POLICY, reference_policy=C.REFERENCE_POLICY, gates=C.GATES, fail_fast=C.FAIL_FAST,
        bindings=bindings, real_inputs_read=False, independent_real_reference=False, case_ids=[s['id'] for s in specs],
        ordered_call_ids=ordered_calls(specs),case_count=C.CASE_COUNT,call_count=C.CALL_COUNT,
        query_specs_sha256=C.sha(out/'query_specs.json'), **SCOPE)
    if (C.sha(out/'lock.json') != (out/'lock.sha256').read_text().strip()
            or any(lock.get(k) != v for k,v in expected.items())
            or C.read(out/'query_specs.json') != specs):
        raise ValueError('frozen R33 population/runtime/provenance differs')
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
        source_hashes=lock['source_hashes'], runtime=lock['runtime'], real_inputs_read=False, independent_real_reference=False, **SCOPE)
    if (C.sha(directory/'lock.json') != (directory/'lock.sha256').read_text().strip()
            or any(design.get(k) != v for k,v in expected.items())
            or not lock['created_unix'] <= design['created_unix'] <= time.time()):
        raise ValueError('complete independent pre-input freeze required')
    for name,digest in lock['source_hashes'].items():
        if C.sha(directory/'source_snapshot'/name) != digest:
            raise ValueError('independent source snapshot differs: '+name)
    return C.sha(directory/'lock.json')


def load_old(spec, read_state=None):
    path = Path(spec['source_archive_path'])
    if C.sha(path) != spec['source_archive_sha256']:
        raise ValueError('old prepared archive bytes changed')
    # Decode only the four real inputs. Every old production output remains
    # untouched until the independently owned post-seal comparison stage.
    old = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in C.OLD_FIELDS:
            old[name] = archive[name]
            if read_state is not None:
                read_state['real_inputs_read'] = True
    if any(C.array_sha(value) != spec['array_sha256'][name] for name,value in old.items()):
        raise ValueError('old prepared canonical array identity changed')
    return old


def copy_inputs(old):
    """Copy exact float64 reference inputs without mapping, repair or rounding."""
    if set(old) != PREPARED_FIELDS:
        raise ValueError('exact four-field inherited input boundary required')
    bounds = np.asarray(old['input__bounds'])
    if bounds.shape != (C.KEY_COUNT,4):
        raise ValueError('fixed native rectangle population required')
    shapes = dict(input__mu=(3,), input__factor=(3,3), input__bounds=(len(bounds),4),
        input__content_bias=(len(bounds),))
    for name,shape in shapes.items():
        value = np.asarray(old[name])
        if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('invalid inherited dtype/shape/finite field: '+name)
    result = {name:value.copy() for name,value in old.items()}
    if any(C.array_sha(result[name]) != C.array_sha(old[name]) for name in PREPARED_FIELDS):
        raise ValueError('copied model input bytes changed')
    return result


def stage_failure(out, stage, error, real_inputs_read=False, independent_real_reference=False):
    directory = out/stage
    C.write(directory/'failure.json', dict(error_type=type(error).__name__, error=str(error),
        traceback=traceback.format_exc(), finished_unix=time.time(), complete_case_inventory=False,
        real_inputs_read=real_inputs_read, independent_real_reference=independent_real_reference, **SCOPE))
    if (directory/'completion.json').exists():
        shutil.copyfile(directory/'completion.json', directory/'completion_before_failure.json')
    files = {name:digest for name,digest in C.inventory(directory).items() if name != 'completion.json'}
    C.write(directory/'completion.json', dict(files=files, failure_sha256=C.sha(directory/'failure.json'),
        complete_case_inventory=False, all_inputs_prepared=False, reference_calls_completed=False,
        finished_unix=time.time(), real_inputs_read=real_inputs_read,
        independent_real_reference=independent_real_reference, **SCOPE))


def prepare(out):
    out = Path(out)
    lock = verify(out)
    audit_digest = require_audit_freeze(out, lock)
    directory = out/'prepared'
    if directory.exists() or (out/'prepared_complete.json').exists():
        raise ValueError('preserve every prior input preparation attempt')
    directory.mkdir()
    started, clock = time.time(), time.perf_counter()
    read_any = False
    try:
        C.write(directory/'started.json', dict(started_unix=started,
            producer_lock_sha256=C.sha(out/'lock.json'), audit_lock_sha256=audit_digest, real_inputs_read=False, independent_real_reference=False, **SCOPE))
        (directory/'arrays').mkdir()
        rows = []
        for spec in C.read(out/'query_specs.json'):
            tick = time.perf_counter()
            arrays = {}
            row = {name:spec[name] for name in IDENTITY_FIELDS}
            row.update(source_archive_path=spec['source_archive_path'],
                source_archive_sha256=spec['source_archive_sha256'], policy_overrides=spec['policy_overrides'],
                status='failed', passed=False,
                real_inputs_read=False, independent_real_reference=False, **SCOPE)
            try:
                old = load_old(spec, row)
                arrays = {name:value.copy() for name,value in old.items()}
                arrays = copy_inputs(old)
                row.update(status='prepared', passed=True)
            except Exception as error:
                row['failure_reason'] = type(error).__name__+': '+str(error)
            finally:
                read_any = read_any or row['real_inputs_read']
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
            finished_unix=time.time(), real_inputs_read=any(r['real_inputs_read'] for r in rows), independent_real_reference=False, **SCOPE))
        C.write(out/'prepared_complete.json', dict(producer_lock_sha256=C.sha(out/'lock.json'),
            audit_lock_sha256=audit_digest, prepared_records_sha256=C.sha(directory/'records.json'),
            prepared_completion_sha256=C.sha(directory/'completion.json'), query_specs_sha256=lock['query_specs_sha256'],
            case_ids=[r['id'] for r in rows], case_count=len(rows), all_inputs_prepared=all(r['passed'] for r in rows),
            started_unix=started, finished_unix=time.time(), preparation_phase_seconds=phase_seconds,
            predictions_executed=False, reference_calls_executed=False, real_inputs_read=read_any, independent_real_reference=False, **SCOPE))
    except (Exception, KeyboardInterrupt) as error:
        stage_failure(out, 'prepared', error, read_any)
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
        predictions_executed=False, reference_calls_executed=False, real_inputs_read=True, independent_real_reference=False, **SCOPE)
    expected_files = {'started.json','records.json'} | {'arrays/'+s['id']+'.npz' for s in specs}
    if (any(seal.get(k) != v for k,v in expected_seal.items())
            or done.get('complete_case_inventory') is not True or done.get('all_inputs_prepared') is not True
            or done.get('real_inputs_read') is not True
            or done.get('independent_real_reference') is not False
            or any(done.get(k) != v for k,v in SCOPE.items())
            or set(done['files']) != expected_files or [r['id'] for r in rows] != lock['case_ids']
            or not all(r['passed'] is True and r['status'] == 'prepared' and r['real_inputs_read'] is True for r in rows)):
        raise ValueError('complete successful global input seal required')
    for row,spec in zip(rows,specs):
        if (row['archive_path'] != 'prepared/arrays/'+spec['id']+'.npz'
                or C.sha(out/row['archive_path']) != row['archive_sha256']
                or set(row['array_sha256']) != PREPARED_FIELDS
                or any(row[k] != spec[k] for k in IDENTITY_FIELDS)
                or row['source_archive_path'] != spec['source_archive_path']
                or row['source_archive_sha256'] != spec['source_archive_sha256']
                or row['policy_overrides'] != spec['policy_overrides']
                or row.get('independent_real_reference') is not False
                or any(row.get(k) != v for k,v in SCOPE.items())):
            raise ValueError('prepared source identity differs')
    verification = C.read(directory/'input_verification.json')
    completion = C.read(directory/'input_audit/completion.json')
    expected = dict(passed=True, checked_cases=lock['case_count'],
        producer_lock_sha256=C.sha(out/'lock.json'), audit_lock_sha256=digest,
        producer_prepared_complete_sha256=C.sha(out/'prepared_complete.json'),
        producer_prepared_completion_sha256=C.sha(out/'prepared/completion.json'),
        query_specs_sha256=lock['query_specs_sha256'],
        source_prepared_hashes={s['source_archive_path']:s['source_archive_sha256'] for s in specs},
        all_input_bytes_exact=True, real_inputs_read=True, independent_real_reference=False, **SCOPE)
    if (any(verification.get(k) != v for k,v in expected.items())
            or completion.get('passed') is not True
            or completion.get('real_inputs_read') is not True
            or completion.get('independent_real_reference') is not False
            or any(completion.get(k) != v for k,v in SCOPE.items())
            or completion.get('verification_sha256') != C.sha(directory/'input_verification.json')):
        raise ValueError('complete independent input audit required before any reference call')
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


def run_case(out, prepared_row, order, policy, reference_policy, budget, deadline,
             phase_state=None, call_index=0, stop_trigger=None):
    """Archive one scheduled reference outcome, including all actual partials."""
    directory = out/'forward'
    started, tick = time.time(), time.perf_counter()
    arrays, attempted = {}, False
    before = int(budget.used)
    state = dict(real_inputs_read=False, independent_real_reference=False)
    identity = dict(**{name:prepared_row[name] for name in IDENTITY_FIELDS}, order=int(order),
        call_id=call_id(prepared_row['id'], order),call_index=int(call_index))
    stage, not_attempted_reason = 'input_decode', None
    try:
        if stop_trigger is not None:
            stage, not_attempted_reason = 'fail_fast', 'fail_fast'
            raise RuntimeError('not attempted: fail-fast after '+stop_trigger['call_id'])
        if time.perf_counter() >= deadline:
            stage, not_attempted_reason = 'phase_deadline', 'phase_deadline'
            raise TimeoutError('reference phase deadline before input load or call')
        source = out/prepared_row['archive_path']
        if C.sha(source) != prepared_row['archive_sha256']:
            raise ValueError('prepared archive changed after independent audit')
        bank = {}
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != PREPARED_FIELDS:
                raise ValueError('prepared field population differs')
            for name in C.OLD_FIELDS:
                bank[name] = archive[name]
                state['real_inputs_read'] = True
                if phase_state is not None:
                    phase_state['real_inputs_read'] = True
        if {name:C.array_sha(value) for name,value in bank.items()} != prepared_row['array_sha256']:
            raise ValueError('prepared canonical arrays changed after input audit')
        arrays = {'input__'+name:bank['input__'+name] for name in INPUT_FIELDS}
        if time.perf_counter() >= deadline:
            stage, not_attempted_reason = 'phase_deadline', 'phase_deadline'
            raise TimeoutError('reference phase deadline before call')
        append_record(directory/'call_starts.jsonl', dict(identity, started_unix=time.time(),
            budget_used_before=before, real_inputs_read=True,
            independent_real_reference=True, **SCOPE))
        attempted = True
        stage = 'reference'
        state['independent_real_reference'] = True
        if phase_state is not None:
            phase_state['independent_real_reference'] = True
        result = REF.integrate(**{name:arrays['input__'+name] for name in INPUT_FIELDS},
            policy=policy, reference_policy=reference_policy, order=order,
            budget=budget, deadline=deadline)
    except Exception as error:
        # ReferenceFailure carries a snapshot of all computed evidence. A
        # pre-call IO/identity error has no reference work and creates no output.
        result = dict(getattr(error, 'reference_metadata', {}))
        result.update(status='failed', passed=False,
            failure_reason=type(error).__name__+': '+str(error),
            work=getattr(error, 'reference_work', {}))
        arrays.update(getattr(error, 'available_arrays', {}))
        if not attempted and not_attempted_reason is None:
            not_attempted_reason = stage
    arrays.update({name:value for name,value in result.items() if isinstance(value, np.ndarray)})
    metadata = {name:value for name,value in result.items() if not isinstance(value, np.ndarray)}
    if 'wall_seconds' in metadata:
        metadata['reference_reported_wall_seconds'] = metadata.pop('wall_seconds')
    after = int(budget.used)
    charged = after-before
    reported = metadata.get('work', {}).get('cell_node_evaluations', 0)
    consistent = (type(reported) is int and reported == charged and charged >= 0)
    # Do not manufacture missing reference work or silently alter its ledger.
    # This outcome can be delivered, but it cannot satisfy the numerical gate.
    if not consistent:
        metadata.update(status='failed', passed=False,
            failure_reason=metadata.get('failure_reason', '')+'; reference work differs from shared budget delta')
        stage = 'reference'
    metadata.update(failure_stage=None if metadata['passed'] else stage,
        not_attempted_reason=not_attempted_reason, stop_trigger=stop_trigger)
    metadata.update(budget_used_before=before, budget_used_after=after,
        charged_cell_node_evaluations=charged, reference_work_consistent=consistent)
    hashes = {name:C.array_sha(value) for name,value in arrays.items()}
    append_record(directory/'reference_results.jsonl', dict(metadata, **identity,
        attempted=attempted, available_array_sha256=hashes, **state, **SCOPE))
    path = directory/'arrays'/(identity['call_id']+'.npz')
    if path.exists():
        raise ValueError('refusing to overwrite a reference outcome')
    np.savez_compressed(path, **arrays)
    wall = time.perf_counter()-tick
    return dict(metadata, **identity,
        policy_overrides=prepared_row['policy_overrides'],
        prepared_archive_path=prepared_row['archive_path'], prepared_archive_sha256=prepared_row['archive_sha256'],
        archive_path=str(path.relative_to(out)), archive_sha256=C.sha(path), array_sha256=hashes,
        attempted=attempted, **state, **SCOPE, started_unix=started, finished_unix=time.time(),
        wall_seconds=wall, call_and_input_load_seconds=wall,
        peak_rss_kib_process=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        peak_rss_scope='cumulative process high-water mark; not exclusive per-call memory')


def run(out):
    out = Path(out)
    lock = verify(out)
    audit_digest = require_audit_freeze(out, lock)
    input_digest, prepared_rows = require_input_audit(out, lock)
    directory = out/'forward'
    if directory.exists() or (out/'forward_complete.json').exists() or (out/'completion.json').exists():
        raise ValueError('preserve every prior reference attempt')
    directory.mkdir()
    started, clock = time.time(), time.perf_counter()
    reference_policy = lock['reference_policy']
    deadline = clock+reference_policy['max_phase_wall_seconds']
    budget = REF.NodeBudget(reference_policy['max_total_cell_node_evaluations'])
    state = dict(real_inputs_read=False, independent_real_reference=False)
    try:
        C.write(directory/'started.json', dict(started_unix=started,
            producer_lock_sha256=C.sha(out/'lock.json'), audit_lock_sha256=audit_digest,
            input_verification_sha256=input_digest, **state, **SCOPE))
        (directory/'arrays').mkdir()
        (directory/'call_starts.jsonl').touch(exist_ok=False)
        (directory/'reference_results.jsonl').touch(exist_ok=False)
        rows, stop_trigger = [], None
        for prepared_row in prepared_rows:
            policy = dict(lock['policy'], **prepared_row['policy_overrides'])
            for order in reference_policy['orders']:
                row = run_case(out, prepared_row, order, policy, reference_policy, budget, deadline,
                    state,call_index=len(rows),stop_trigger=stop_trigger)
                rows.append(row)
                if stop_trigger is None and row['passed'] is False:
                    stop_trigger = dict(call_id=row['call_id'],index=row['call_index'],
                        reason=row['failure_reason'],archive_sha256=row['archive_sha256'],stage=row['failure_stage'])
                C.write(directory/'records.json', rows)
                print('REFERENCE_CALL', row['call_id'], row['status'], flush=True)
        elapsed, phase_finished = time.perf_counter()-clock, time.time()
        verify(out)
        if (require_audit_freeze(out, lock) != audit_digest
                or require_input_audit(out, lock)[0] != input_digest
                or [r['call_id'] for r in rows] != lock['ordered_call_ids']):
            raise ValueError('input audit/source/population changed during reference phase')
        phase_passed = elapsed <= reference_policy['max_phase_wall_seconds']
        nodes_passed = (budget.used <= reference_policy['max_total_cell_node_evaluations']
            and sum(r['charged_cell_node_evaluations'] for r in rows) == budget.used
            and all(r['reference_work_consistent'] for r in rows))
        completed = all(r['attempted'] and r['passed'] is True for r in rows) and phase_passed and nodes_passed
        common = dict(reference_calls_completed=completed, phase_wall_passed=phase_passed,
            phase_wall_seconds=elapsed, total_cell_node_evaluations=int(budget.used),
            phase_node_budget_passed=nodes_passed, stop_trigger=stop_trigger,
            executed_failed_calls=sum(r['attempted'] and not r['passed'] for r in rows),
            fail_fast_skipped_calls=sum(r['not_attempted_reason']=='fail_fast' for r in rows),
            unattempted_failure_calls=sum(not r['attempted'] and r['not_attempted_reason']!='fail_fast' for r in rows),
            **state, **SCOPE)
        C.write(directory/'completion.json', dict(files=C.inventory(directory), complete_case_inventory=True,
            finished_unix=time.time(), **common))
        C.write(out/'forward_complete.json', dict(producer_lock_sha256=C.sha(out/'lock.json'),
            audit_lock_sha256=audit_digest, producer_prepared_complete_sha256=C.sha(out/'prepared_complete.json'),
            input_verification_sha256=input_digest, query_specs_sha256=lock['query_specs_sha256'],
            forward_records_sha256=C.sha(directory/'records.json'),
            forward_completion_sha256=C.sha(directory/'completion.json'),
            case_ids=lock['case_ids'], ordered_call_ids=[r['call_id'] for r in rows],
            case_count=lock['case_count'], call_count=len(rows), all_case_attempts_retained=True,
            all_available_partial_arrays_retained=True, attempted_calls=sum(r['attempted'] for r in rows),
            started_unix=started, phase_finished_unix=phase_finished, finished_unix=time.time(),
            phase_timing_scope='started through final forward records write; provenance finalization separate',
            **common))
        C.write(out/'completion.json', dict(files=C.inventory(out),
            forward_complete_sha256=C.sha(out/'forward_complete.json'), complete_case_inventory=True,
            finished_unix=time.time(), post_phase_bookkeeping_seconds_before_completion_write=time.perf_counter()-clock-elapsed,
            total_seconds_before_completion_write=time.perf_counter()-clock, **common))
    except (Exception, KeyboardInterrupt) as error:
        stage_failure(out, 'forward', error, **state)
        if (out/'completion.json').exists():
            shutil.copyfile(out/'completion.json', out/'completion_before_failure.json')
        C.write(out/'completion.json', dict(files={n:h for n,h in C.inventory(out).items() if n != 'completion.json'},
            complete_case_inventory=False, reference_calls_completed=False, finished_unix=time.time(),
            total_cell_node_evaluations=int(budget.used), **state, **SCOPE))
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


