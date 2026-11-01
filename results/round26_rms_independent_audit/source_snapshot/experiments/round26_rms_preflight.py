"""Freeze and seal the fixed R26 synthetic-only production computations."""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback

import numpy as np  # Keep the declared NumPy-first numerical runtime.
import scipy
import scipy.special  # Match the reference's loaded numerical runtime before freezing.
from threadpoolctl import threadpool_info

from experiments import round26_rms_contract as C
from experiments import round26_rms_fixtures as FIXTURES
from models import round26_rms_projective_pe as MODEL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/'results/round26_rms_preflight'
DEFAULT_AUDIT = ROOT/'results/round26_rms_independent_audit'
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)),
                        sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header+b'\n'+value.tobytes()).hexdigest()


def source_names():
    found, pending = set(), list(C.SOURCE_FILES)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT/name
        if not path.is_file():
            raise ValueError('missing joint source: '+name)
        found.add(name)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('relative import requires explicit source review')
                if node.module:
                    names = [node.module]+[node.module+'.'+n.name for n in node.names]
            for module in names:
                parts = module.split('.')
                paths = [Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py']
                paths += [Path(*parts[:i])/'__init__.py' for i in range(1, len(parts))]
                pending.extend(str(p) for p in paths if (ROOT/p).is_file() and str(p) not in found)
    return sorted(found)


def runtime():
    builds = {}
    for name, function in (('numpy', np.show_config), ('scipy', scipy.show_config)):
        stream = io.StringIO()
        with redirect_stdout(stream):
            function()
        builds[name+'_build_sha256'] = hashlib.sha256(stream.getvalue().encode()).hexdigest()
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), threads={n:os.environ.get(n) for n in THREAD_ENV},
                threadpools=threadpool_info(), **builds)


def check_runtime():
    result = runtime()
    if any(result['threads'][k] != '4' for k in THREAD_ENV):
        raise ValueError('the frozen OMP/MKL/OpenBLAS=4 environment is required')
    if any(p['num_threads'] != 4 for p in result['threadpools']):
        raise ValueError('loaded numerical thread pools must use four threads')
    return result


def inventory(directory):
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def verify_inventory(directory, files, exclude=()):
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual != set(files) | set(exclude):
        raise ValueError('inventory population differs: '+str(directory))
    for name, digest in files.items():
        if sha(directory/name) != digest:
            raise ValueError('inventory digest differs: '+name)


def inherited_bindings():
    """Accepted R25 outcome and prospective design; metadata bytes only."""
    bindings = {str(ROOT/n):sha(ROOT/n) for n in C.BINDINGS}
    receipt = read(ROOT/C.BINDINGS[0])
    if (receipt.get('accepted_for_numerical_interpretation') is not True
            or receipt.get('accepted_for_development_interpretation') is not True
            or receipt.get('numerical_gate_passed') is not True
            or receipt.get('necessary_efficacy_screen_passed') is not False
            or receipt.get('scientific_gate_passed') is not False):
        raise ValueError('accepted R25 necessary efficacy failure is required')
    for name, digest in receipt['files'].items():
        if Path(name).suffix == '.npz':
            raise ValueError('actual arrays forbidden during metadata freeze')
        if sha(ROOT/name) != digest:
            raise ValueError('inherited accepted evidence changed: '+name)
        bindings[str(ROOT/name)] = digest
    decision = read(ROOT/C.BINDINGS[1])
    selection = decision['next_selected_work']
    if (decision['root_acceptance_sha256'] != sha(ROOT/C.BINDINGS[0])
            or selection['round'] != 'R26'
            or selection['real_input_or_labels_authorized'] is not False
            or selection['efficacy_candidate_selected'] is not False
            or selection['design_sha256'] != sha(ROOT/selection['design_path'])):
        raise ValueError('R26 selection does not bind accepted failure/synthetic-only scope')
    bindings[str(ROOT/selection['design_path'])] = selection['design_sha256']
    return bindings


def population():
    rows = FIXTURES.fixtures()
    base, identities = [r for r in rows if r['kind'] == 'base'], [r for r in rows if r['kind'] == 'identity']
    if (rows != base+identities or [r['id'] for r in base] != list(FIXTURES.BASE_IDS)
            or [r['id'] for r in identities] != list(FIXTURES.IDENTITY_IDS)
            or len(base) != 18 or len(identities) != 6
            or len({r['id'] for r in rows}) != 24
            or any(r['identity']['parent'] not in FIXTURES.BASE_IDS for r in identities)):
        raise ValueError('fixed disjoint base then identity inventory required')
    return rows, [r['id'] for r in base], [r['id'] for r in identities]


def freeze(out, audit_root=DEFAULT_AUDIT):
    out, audit_root = Path(out), Path(audit_root)
    if out.exists():
        raise ValueError('refusing existing R26 directory')
    sources = {n:sha(ROOT/n) for n in source_names()}
    bindings = inherited_bindings()
    rows, base_ids, identity_ids = population()
    observed_runtime = check_runtime()
    out.mkdir(parents=True)
    write(out/'fixtures.json', rows)
    for name, digest in sources.items():
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        if sha(target) != digest:
            raise ValueError('source changed during freeze')
    write(out/'lock.json', dict(version=1, created_unix=time.time(), source_hashes=sources,
        runtime=observed_runtime, policy=C.POLICY, reference_policy=C.REFERENCE_POLICY, gates=C.GATES,
        fixtures_sha256=sha(out/'fixtures.json'), base_ids=base_ids, identity_ids=identity_ids,
        audit_root=str(audit_root.resolve()), bindings=bindings, synthetic_only=True,
        real_inputs_read=False, labels_read=False, scientific_gate_passed=False, goal_complete=False))
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    verify(out)


def verify(out):
    out = Path(out)
    lock = read(out/'lock.json')
    rows, base_ids, identity_ids = population()
    expected = dict(version=1, runtime=check_runtime(), policy=C.POLICY,
        reference_policy=C.REFERENCE_POLICY, gates=C.GATES, base_ids=base_ids, identity_ids=identity_ids,
        synthetic_only=True, real_inputs_read=False, labels_read=False, scientific_gate_passed=False,
        goal_complete=False)
    if (sha(out/'lock.json') != (out/'lock.sha256').read_text().strip()
            or any(lock.get(k) != v for k, v in expected.items())):
        raise ValueError('frozen R26 contract/runtime changed')
    if set(lock['source_hashes']) != set(source_names()):
        raise ValueError('joint source closure changed')
    for name, digest in lock['source_hashes'].items():
        if sha(ROOT/name) != digest or sha(out/'source_snapshot'/name) != digest:
            raise ValueError('joint source changed: '+name)
    if (sha(out/'fixtures.json') != lock['fixtures_sha256'] or read(out/'fixtures.json') != rows
            or inherited_bindings() != lock['bindings']):
        raise ValueError('frozen fixtures/accepted predecessor changed')
    return lock


def require_audit_freeze(out, lock):
    directory = Path(lock['audit_root'])
    audit = read(directory/'lock.json')
    expected = dict(producer_lock_sha256=sha(out/'lock.json'), fixtures_sha256=lock['fixtures_sha256'],
                    source_hashes=lock['source_hashes'], runtime=lock['runtime'])
    if (sha(directory/'lock.json') != (directory/'lock.sha256').read_text().strip()
            or any(audit.get(k) != v for k, v in expected.items())
            or not lock['created_unix'] <= audit['created_unix'] <= time.time()):
        raise ValueError('independent pre-forward freeze differs')
    for name, digest in lock['source_hashes'].items():
        if sha(directory/'source_snapshot'/name) != digest:
            raise ValueError('independent source snapshot changed')
    return sha(directory/'lock.json')


def append_record(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
        stream.flush()


def run_case(out, fixture, policy, deadline):
    """A single fixed attempt; retain inputs/partials even on model rejection."""
    started, tick = time.time(), time.perf_counter()
    policy = dict(policy)
    overrides = fixture.get('policy_overrides', {})
    if set(overrides)-{'h', 'floor_density'}:
        raise ValueError('only frozen screen-unit overrides are legal')
    policy.update(overrides)
    inputs = {k:np.asarray(fixture[k], np.float64)
              for k in ('mu', 'factor', 'bounds', 'content_bias')}
    attempted = False
    if time.perf_counter() >= deadline:
        result = dict(status='failed', passed=False, failure_reason='production phase deadline before model call', work={})
    else:
        attempted = True
        append_record(out/'call_starts.jsonl', dict(id=fixture['id'], started_unix=time.time()))
        try:
            result = MODEL.integrate(**inputs, policy=policy, deadline=deadline)
        except Exception as error:
            result = dict(status='failed', passed=False,
                          failure_reason=type(error).__name__+': '+str(error), work={})
    arrays = {'input__'+k:v for k, v in inputs.items()}
    arrays.update({k:v for k, v in result.items() if isinstance(v, np.ndarray)})
    if result['passed']:
        arrays['argmax'] = np.asarray(np.argmax(result['probability']), np.int32)
    metadata = {k:v for k, v in result.items() if not isinstance(v, np.ndarray)}
    # Journal returned status/work before an archive-save exception could hide it.
    append_record(out/'model_results.jsonl', dict(metadata, id=fixture['id'], attempted=attempted,
        available_array_sha256={k:array_sha(v) for k, v in arrays.items()}))
    path = out/'arrays'/(fixture['id']+'.npz')
    if path.exists():
        raise ValueError('refusing case archive overwrite')
    np.savez_compressed(path, **arrays)
    row = dict(metadata, id=fixture['id'], kind=fixture['kind'], identity=fixture.get('identity'),
        content_temperature=fixture['content_temperature'], bias_kind=fixture['bias_kind'],
        policy_overrides=overrides, archive_path=str(path.relative_to(out)), archive_sha256=sha(path),
        array_sha256={k:array_sha(v) for k, v in arrays.items()}, attempted=attempted,
        started_unix=started, finished_unix=time.time(), wall_seconds=time.perf_counter()-tick,
        peak_rss_kib_process=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        peak_rss_scope='cumulative process high-water mark; not exclusive per-case memory')
    return row


def _run_started(out):
    lock = verify(out)
    audit_digest = require_audit_freeze(out, lock)
    if any((out/n).exists() for n in ('started.json', 'completion.json', 'arrays', 'records.json',
                                     'forward_complete.json', 'call_starts.jsonl', 'model_results.jsonl')):
        raise ValueError('refusing to overwrite a started formal attempt')
    started, clock = time.time(), time.perf_counter()
    deadline = clock+lock['gates']['production_max_phase_wall_seconds']
    write(out/'started.json', dict(started_unix=started, producer_lock_sha256=sha(out/'lock.json'),
        audit_lock_sha256=audit_digest, synthetic_only=True, real_inputs_read=False, labels_read=False))
    (out/'arrays').mkdir()
    (out/'call_starts.jsonl').touch(exist_ok=False)
    (out/'model_results.jsonl').touch(exist_ok=False)
    rows = []
    for fixture in read(out/'fixtures.json'):
        row = run_case(out, fixture, lock['policy'], deadline)
        rows.append(row)
        write(out/'records.json', rows)
        print('PRODUCTION_CASE', fixture['id'], row['status'], flush=True)
    elapsed, phase_finished = time.perf_counter()-clock, time.time()
    verify(out)
    if require_audit_freeze(out, lock) != audit_digest:
        raise ValueError('independent freeze changed during forward')
    if [r['id'] for r in rows] != lock['base_ids']+lock['identity_ids']:
        raise ValueError('complete fixed population required')
    seal = dict(producer_lock_sha256=sha(out/'lock.json'), audit_lock_sha256=audit_digest,
        fixtures_sha256=lock['fixtures_sha256'], records_sha256=sha(out/'records.json'),
        case_ids=[r['id'] for r in rows], case_count=len(rows), all_case_attempts_retained=True,
        attempted_calls=sum(r['attempted'] for r in rows),
        all_successful_arrays_retained=True, all_available_partial_arrays_retained=True,
        started_unix=started, phase_finished_unix=phase_finished,
        finished_unix=time.time(), phase_wall_seconds=elapsed,
        phase_wall_passed=elapsed <= lock['gates']['production_max_phase_wall_seconds'],
        all_production_calls_completed=all(r['passed'] is True for r in rows), synthetic_only=True,
        phase_timing_scope='started through final records.json write; provenance finalization separate',
        independent_numerical_verification_pending=True, real_inputs_read=False, labels_read=False,
        scientific_gate_passed=False, goal_complete=False)
    write(out/'forward_complete.json', seal)
    write(out/'completion.json', dict(files=inventory(out), producer_forward_complete_sha256=sha(out/'forward_complete.json'),
        finished_unix=time.time(), complete_case_inventory=True,
        all_production_calls_completed=seal['all_production_calls_completed'], phase_wall_passed=seal['phase_wall_passed'],
        post_phase_bookkeeping_seconds_before_completion_write=time.perf_counter()-clock-elapsed,
        total_seconds_before_completion_write=time.perf_counter()-clock, real_inputs_read=False,
        labels_read=False, scientific_gate_passed=False, goal_complete=False))


def run(out):
    """Archive started engineering failures and preserve their nonzero exit."""
    out = Path(out)
    existing = (out/'started.json').exists() or (out/'completion.json').exists()
    try:
        return _run_started(out)
    except (Exception, KeyboardInterrupt) as error:
        if not existing and (out/'started.json').exists():
            write(out/'failure.json', dict(error_type=type(error).__name__, error=str(error),
                traceback=traceback.format_exc(), finished_unix=time.time(), complete_case_inventory=False,
                real_inputs_read=False, labels_read=False, scientific_gate_passed=False, goal_complete=False))
            if (out/'completion.json').exists():
                shutil.copyfile(out/'completion.json', out/'completion_before_failure.json')
            files = {k:v for k, v in inventory(out).items() if k != 'completion.json'}
            write(out/'completion.json', dict(files=files, failure_sha256=sha(out/'failure.json'),
                complete_case_inventory=False, all_production_calls_completed=False, phase_wall_passed=False,
                finished_unix=time.time(), real_inputs_read=False, labels_read=False,
                scientific_gate_passed=False, goal_complete=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--audit-root', type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.out.resolve(), args.audit_root.resolve())
    else:
        run(args.out.resolve())


if __name__ == '__main__':
    main()


