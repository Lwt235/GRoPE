"""Freeze and seal the fixed R22 synthetic-only production computations."""
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
from threadpoolctl import threadpool_info

from experiments import round22_ray_contract as C
from models import round22_ray_pushforward as MODEL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/'results/round22_ray_preflight'
DEFAULT_AUDIT = ROOT/'results/round22_ray_independent_audit'
BINDINGS = ('results/round21_matched_root_acceptance_20261031.json',
            'results/round21_matched_post_result_decision_20261031.json')
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


def freeze(out, audit_root=DEFAULT_AUDIT):
    if out.exists():
        raise ValueError('refusing existing R22 directory')
    source_hashes = {n:sha(ROOT/n) for n in source_names()}
    bindings = {str(ROOT/n):sha(ROOT/n) for n in BINDINGS}
    base, identities = C.fixtures(), C.identity_fixtures()
    rows = base + identities
    if not base or not identities or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('complete disjoint base/identity case identities required')
    observed_runtime = check_runtime()
    out.mkdir(parents=True)
    write(out/'fixtures.json', rows)
    for name, digest in source_hashes.items():
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        if sha(target) != digest:
            raise ValueError('source changed while freezing: '+name)
    lock = dict(version=1, created_unix=time.time(), source_hashes=source_hashes,
        runtime=observed_runtime, policy=C.POLICY, reference_policy=C.REFERENCE_POLICY, gates=C.GATES,
        fixtures_sha256=sha(out/'fixtures.json'), base_ids=[r['id'] for r in base],
        identity_ids=[r['id'] for r in identities], audit_root=str(audit_root.resolve()), bindings=bindings,
        synthetic_only=True, real_inputs_read=False, labels_read=False, scientific_gate_passed=False)
    write(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    verify(out)


def verify(out):
    lock = read(out/'lock.json')
    base, identities = C.fixtures(), C.identity_fixtures()
    expected = dict(runtime=check_runtime(), policy=C.POLICY, reference_policy=C.REFERENCE_POLICY,
                    gates=C.GATES, base_ids=[r['id'] for r in base], identity_ids=[r['id'] for r in identities],
                    synthetic_only=True, real_inputs_read=False, labels_read=False, scientific_gate_passed=False)
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip():
        raise ValueError('producer lock changed')
    if any(lock[k] != value for k, value in expected.items()):
        raise ValueError('frozen R22 contract/runtime changed')
    if set(lock['source_hashes']) != set(source_names()):
        raise ValueError('joint source closure changed')
    for name, digest in lock['source_hashes'].items():
        if sha(ROOT/name) != digest or sha(out/'source_snapshot'/name) != digest:
            raise ValueError('joint source changed: '+name)
    if sha(out/'fixtures.json') != lock['fixtures_sha256'] or read(out/'fixtures.json') != base+identities:
        raise ValueError('fixed input specification changed')
    if set(lock['bindings']) != {str(ROOT/n) for n in BINDINGS}:
        raise ValueError('development decision bindings changed')
    for name, digest in lock['bindings'].items():
        if sha(name) != digest:
            raise ValueError('development decision changed: '+name)
    return lock


def require_audit_freeze(out, lock):
    directory = Path(lock['audit_root'])
    audit = read(directory/'lock.json')
    if sha(directory/'lock.json') != (directory/'lock.sha256').read_text().strip():
        raise ValueError('independent lock changed')
    expected = dict(producer_lock_sha256=sha(out/'lock.json'), fixtures_sha256=lock['fixtures_sha256'],
                    source_hashes=lock['source_hashes'], runtime=lock['runtime'])
    if any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError('independent freeze is not bound to this joint contract')
    for name, digest in lock['source_hashes'].items():
        if sha(directory/'source_snapshot'/name) != digest:
            raise ValueError('independent frozen source changed: '+name)
    return sha(directory/'lock.json')


def _run_started(out):
    lock = verify(out)
    audit_digest = require_audit_freeze(out, lock)
    if (out/'started.json').exists() or (out/'completion.json').exists() or (out/'arrays').exists():
        raise ValueError('refusing to overwrite a started or completed formal run')
    started_unix = time.time()
    phase_start = time.perf_counter()
    deadline = phase_start+C.GATES['production_max_phase_wall_seconds']
    write(out/'started.json', dict(started_unix=started_unix, producer_lock_sha256=sha(out/'lock.json'),
                                 audit_lock_sha256=audit_digest))
    (out/'arrays').mkdir()
    rows = []
    for fixture in read(out/'fixtures.json'):
        policy = dict(lock['policy'])
        overrides = fixture.get('policy_overrides', {})
        if set(overrides)-{'h', 'floor_density'}:
            raise ValueError('only frozen unit-conversion policy overrides are legal')
        policy.update(overrides)
        inputs = {k:np.asarray(fixture[k], np.float64) for k in ('mu', 'factor', 'bounds', 'H', 'offset')}
        started = time.perf_counter()
        try:
            result = MODEL.integrate(**inputs, policy=policy, deadline=deadline)
        except Exception as error:
            # Retain the engineering failure and the entire fixed population.
            # No retry, repair, or alternate integration path follows it.
            result = dict(status='failed', passed=False,
                          failure_reason=type(error).__name__+': '+str(error), work={})
        wall = time.perf_counter()-started
        arrays = {'input__'+k:v for k, v in inputs.items()}
        arrays.update({k:v for k, v in result.items() if isinstance(v, np.ndarray)})
        path = out/'arrays'/(fixture['id']+'.npz')
        np.savez_compressed(path, **arrays)
        row = {k:v for k, v in result.items() if not isinstance(v, np.ndarray)}
        row.update(id=fixture['id'], kind=fixture['kind'], identity=fixture.get('identity'),
            policy_overrides=overrides, archive_path=str(path.relative_to(out)), archive_sha256=sha(path),
            array_sha256={k:array_sha(v) for k, v in arrays.items()}, wall_seconds=wall,
            peak_rss_kib_process=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            peak_rss_scope='cumulative process high-water mark; not exclusive per-case memory')
        rows.append(row)
        write(out/'records.json', rows)
        print('PRODUCTION_CASE', fixture['id'], row['status'], flush=True)
    # The phase includes all per-case assembly and writes. Final source and
    # inventory verification are separately timed, outside the phase budget.
    elapsed = time.perf_counter()-phase_start
    phase_finished_unix = time.time()
    verify(out)
    if require_audit_freeze(out, lock) != audit_digest:
        raise ValueError('independent lock changed during production')
    if [r['id'] for r in rows] != lock['base_ids']+lock['identity_ids']:
        raise ValueError('incomplete fixed case inventory')
    seal = dict(producer_lock_sha256=sha(out/'lock.json'), audit_lock_sha256=audit_digest,
        fixtures_sha256=lock['fixtures_sha256'], records_sha256=sha(out/'records.json'),
        case_ids=[r['id'] for r in rows], case_count=len(rows),
        all_case_attempts_retained=True, all_successful_arrays_retained=True,
        started_unix=started_unix, phase_finished_unix=phase_finished_unix,
        finished_unix=time.time(), phase_wall_seconds=elapsed,
        phase_wall_passed=elapsed <= C.GATES['production_max_phase_wall_seconds'],
        all_production_calls_completed=all(r['passed'] is True for r in rows),
        synthetic_only=True, independent_numerical_verification_pending=True,
        real_inputs_read=False, labels_read=False, scientific_gate_passed=False)
    write(out/'forward_complete.json', seal)
    files = inventory(out)
    completion = dict(files=files, producer_forward_complete_sha256=sha(out/'forward_complete.json'),
        finished_unix=time.time(), complete_case_inventory=True,
        all_production_calls_completed=seal['all_production_calls_completed'],
        phase_wall_passed=seal['phase_wall_passed'], scientific_gate_passed=False,
        post_phase_bookkeeping_seconds_before_completion_write=time.perf_counter()-phase_start-elapsed,
        total_seconds_before_completion_write=time.perf_counter()-phase_start)
    write(out/'completion.json', completion)
    print('PRODUCTION_SEALED', seal['all_production_calls_completed'], seal['phase_wall_passed'], flush=True)


def run(out):
    """Retain incomplete engineering failures, while preserving nonzero exit."""
    existing = (out/'started.json').exists() or (out/'completion.json').exists()
    try:
        return _run_started(out)
    except (Exception, KeyboardInterrupt) as error:
        if not existing and (out/'started.json').exists():
            failure = dict(finished_unix=time.time(), error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(),
                complete_case_inventory=False, scientific_gate_passed=False,
                formal_run_replacement_allowed=False)
            try:
                write(out/'failure.json', failure)
                if (out/'completion.json').exists():
                    shutil.copyfile(out/'completion.json', out/'completion_before_failure.json')
                files = {k:v for k, v in inventory(out).items() if k != 'completion.json'}
                write(out/'completion.json', dict(files=files, finished_unix=time.time(),
                    complete_case_inventory=False, all_production_calls_completed=False,
                    phase_wall_passed=False, scientific_gate_passed=False,
                    failure_sha256=sha(out/'failure.json')))
            except (Exception, KeyboardInterrupt) as archive_error:
                print('FAILURE_ARCHIVE_ERROR', repr(archive_error), file=sys.stderr, flush=True)
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
