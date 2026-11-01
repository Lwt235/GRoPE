"""Seal the fixed R25 q24-to-g8 Student forward, without reading labels."""
from __future__ import annotations

import argparse
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np  # Preserve the declared NumPy-first runtime.
from experiments import round25_student_contract as C
import torch
from models import round24_student_t_pe as MODEL

N_QUERY, N_KEY, CHANNELS = 48, 432, 128
INPUT_FIELDS = ('a64', 'covariance_archive64', 'covariance64', 'tau32',
                'q_features32', 'k_features32', 'key_bounds_px', 'bounds64',
                'content32', 'bias32', 'content_bias64',
                'covariance_asymmetry_normalized', 'covariance_change_max_absolute')


def freeze(out, audit_root):
    return C.freeze(Path(out), kind='producer', audit_root=Path(audit_root))


def require_audit_freeze(out, lock):
    audit_root = Path(lock['audit_root'])
    audit = C.verify(audit_root)
    if (lock['kind'] != 'producer' or audit['kind'] != 'audit'
            or Path(audit['producer_root']).resolve() != Path(out).resolve()
            or Path(audit['audit_root']).resolve() != audit_root.resolve()
            or audit['created_unix'] < lock['created_unix']
            or any(audit[k] != lock[k] for k in ('source_hashes', 'runtime', 'entries', 'bindings'))):
        raise ValueError('independent joint freeze must precede production')
    return C.sha(audit_root/'lock.json')


def load_cache(entry):
    """Decode exactly six authenticated cache fields; no original inputs/labels."""
    path = Path(entry['cache_path'])
    if C.sha(path) != entry['cache_sha256']:
        raise ValueError('accepted cache archive hash changed')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name:archive[name] for name in C.FIELDS}
    if any(C.array_sha(arrays[name]) != entry['cache_array_sha256'][name] for name in C.FIELDS):
        raise ValueError('accepted cache array hash changed')
    return arrays


def prepare_inputs(cache):
    """Keep original Ca; upper-copy its operational Gaussian covariance."""
    expected = {
        'g24__a': (np.float64, (N_QUERY, 2)),
        'g24__anchor_cov': (np.float64, (N_QUERY, 2, 2)),
        'g24__tau32': (np.float32, (N_QUERY,)),
        'g24__q_features': (np.float32, (N_QUERY, CHANNELS)),
        'g8__k_features': (np.float32, (N_KEY, CHANNELS)),
        'g8__bounds_px': (np.int64, (N_KEY, 4))}
    if set(cache) != set(expected):
        raise ValueError('exact six-field cache allowlist required')
    for name, (dtype, shape) in expected.items():
        value = cache[name]
        if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('cache dtype/shape/finite mismatch: '+name)
    covariance = cache['g24__anchor_cov']
    scale = np.maximum(1., np.max(np.abs(covariance), axis=(1, 2)))
    normalized = covariance/scale[:, None, None]
    asymmetry = np.max(np.abs(normalized-normalized.transpose(0, 2, 1)), axis=(1, 2))
    if (asymmetry > 1e-12).any():
        raise ValueError('archived covariance exceeds original normalized symmetry tolerance')
    execution = covariance.copy()
    execution[:, 1, 0] = execution[:, 0, 1]
    change = np.max(np.abs(execution-covariance), axis=(1, 2))
    # Original overflow-resistant trace formula and dtype rounding, independently
    # checked again by the prelabel auditor. Do not alter the archived tau.
    diagonal = torch.from_numpy(covariance).diagonal(dim1=-2, dim2=-1)
    h2 = torch.tensor(.25**2, dtype=torch.float64)
    tau_scale = torch.maximum(diagonal.amax(-1), h2)
    relative_h2 = h2/tau_scale
    expected_tau = (10.*(relative_h2/(relative_h2+(diagonal/tau_scale[:, None]).sum(-1)))).float().numpy()
    if (not np.isfinite(expected_tau).all()
            or not np.array_equal(expected_tau, cache['g24__tau32'])):
        raise ValueError('archived trace temperature disagrees with fixed formula')
    literal_bounds = np.asarray([[x, y, x+8, y+8]
        for y in range(0, 144, 8) for x in range(0, 192, 8)], np.int64)
    if not np.array_equal(cache['g8__bounds_px'], literal_bounds):
        raise ValueError('native key cell identity/order changed')
    q, k = cache['g24__q_features'], cache['g8__k_features']
    with torch.no_grad():
        content = torch.from_numpy(q) @ torch.from_numpy(k).T
        bias = torch.from_numpy(cache['g24__tau32'])[:, None]*content
    if not bool(torch.isfinite(content).all() and torch.isfinite(bias).all()):
        raise ValueError('nonfinite full-matrix float32 content/bias')
    return dict(a64=cache['g24__a'], covariance_archive64=covariance,
        covariance64=execution, tau32=cache['g24__tau32'], q_features32=q,
        k_features32=k, key_bounds_px=cache['g8__bounds_px'],
        bounds64=(cache['g8__bounds_px'].astype(np.float64)-.5)/8.,
        content32=content.numpy(), bias32=bias.numpy(),
        content_bias64=bias.double().numpy(),
        covariance_asymmetry_normalized=asymmetry,
        covariance_change_max_absolute=change)


def _save_archive(out, path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('refusing archive overwrite')
    np.savez_compressed(path, **arrays)
    return dict(archive_path=str(path.relative_to(out)), archive_sha256=C.sha(path),
                array_sha256={name:C.array_sha(value) for name, value in arrays.items()})


def _identity(entry):
    return {k:entry[k] for k in ('scene', 'pair', 'pair_name')}


def _pair_inputs(out, entry, deadline):
    directory = out/'forward'/entry['scene'].replace('/', '__')/entry['pair_name']
    directory.mkdir(parents=True)
    tick, started = time.perf_counter(), time.time()
    record = dict(**_identity(entry), cache_path=entry['cache_path'],
        cache_sha256=entry['cache_sha256'], cache_array_sha256=entry['cache_array_sha256'],
        status='failed', input_valid=False, input_path=None, input_sha256=None,
        array_sha256={}, cache_read_attempted=False, started_unix=started)
    cache, prepared = None, None
    try:
        if time.perf_counter() >= deadline:
            raise TimeoutError('production phase deadline before cache read')
        record['cache_read_attempted'] = True
        cache = load_cache(entry)
        prepared = prepare_inputs(cache)
        saved = _save_archive(out, directory/'inputs.npz', prepared)
        record.update(status='ok', input_valid=True, input_path=saved['archive_path'],
            input_sha256=saved['archive_sha256'], array_sha256=saved['array_sha256'])
    except Exception as error:
        record['failure_reason'] = type(error).__name__+': '+str(error)
        prepared = None
        if cache is not None:
            record['available_cache'] = _save_archive(out, directory/'failed_cache.npz', cache)
    record.update(finished_unix=time.time(), wall_seconds=time.perf_counter()-tick)
    C.write(directory/'inputs.json', record)
    return directory, prepared, record


def _query(out, directory, entry, inputs, input_record, index, deadline):
    tick, started = time.perf_counter(), time.time()
    result, attempted = {}, False
    if inputs is None:
        result = dict(status='failed', passed=False, work={},
            failure_reason='input preparation failed: '+input_record['failure_reason'])
    elif time.perf_counter() >= deadline:
        result = dict(status='failed', passed=False, work={},
            failure_reason='production phase deadline before model call')
    else:
        attempted = True
        try:
            result = MODEL.integrate(inputs['a64'][index], inputs['covariance64'][index],
                inputs['bounds64'], inputs['content_bias64'][index],
                policy=C.POLICY, deadline=deadline)
        except Exception as error:
            result = dict(status='failed', passed=False, work={},
                failure_reason=type(error).__name__+': '+str(error))
    arrays = {name:value for name, value in result.items() if isinstance(value, np.ndarray)}
    if result['passed']:
        arrays['argmax'] = np.asarray(np.argmax(result['probability']), np.int32)
    saved = _save_archive(out, directory/('q%04d.npz'%index), arrays)
    record = {name:value for name, value in result.items() if not isinstance(value, np.ndarray)}
    record.update(**_identity(entry), query_index=index,
        input_path=input_record['input_path'], input_sha256=input_record['input_sha256'],
        **saved, model_status=result['status'], status='ok' if result['passed'] else 'failed',
        attempted=attempted, started_unix=started, finished_unix=time.time(),
        call_and_output_seconds=time.perf_counter()-tick,
        peak_rss_kib_process=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        peak_rss_scope='cumulative process high-water mark')
    C.write(directory/('q%04d.json'%index), record)
    return record


def _forward_started(out):
    if any((out/name).exists() for name in ('started.json', 'completion.json', 'forward_complete.json', 'forward')):
        raise ValueError('refusing to overwrite an attempted production phase')
    lock = C.verify(out)
    audit_sha = require_audit_freeze(out, lock)
    started, clock = time.time(), time.perf_counter()
    deadline = clock+lock['gates']['production_max_phase_wall_seconds']
    C.write(out/'started.json', dict(started_unix=started,
        producer_lock_sha256=C.sha(out/'lock.json'), audit_lock_sha256=audit_sha,
        labels_read=False, scientific_gate_passed=False, goal_complete=False))
    rows, inputs = [], []
    for entry in lock['entries']:
        directory, prepared, record = _pair_inputs(out, entry, deadline)
        inputs.append(record)
        for index in range(N_QUERY):
            rows.append(_query(out, directory, entry, prepared, record, index, deadline))
        print('PRODUCTION_PAIR', entry['scene'], entry['pair_name'],
              sum(r['status'] == 'ok' for r in rows[-N_QUERY:]), '/', N_QUERY, flush=True)
    forward = out/'forward'
    C.write(forward/'input_records.json', inputs)
    C.write(forward/'records.json', rows)
    C.write(forward/'completion.json', dict(files=C.inventory(forward),
        records_sha256=C.sha(forward/'records.json'),
        input_records_sha256=C.sha(forward/'input_records.json'),
        complete_case_inventory=True, pair_count=len(inputs), case_count=len(rows),
        labels_read=False, scientific_gate_passed=False, goal_complete=False))
    # This boundary includes every input, per-query output, aggregate record and
    # forward inventory write. Post-phase provenance verification is reported apart.
    elapsed, phase_finished = time.perf_counter()-clock, time.time()
    if C.verify(out) != lock or require_audit_freeze(out, lock) != audit_sha:
        raise ValueError('source/runtime/cohort or joint freeze changed during forward')
    seal = dict(producer_lock_sha256=C.sha(out/'lock.json'), audit_lock_sha256=audit_sha,
        forward_completion_sha256=C.sha(forward/'completion.json'),
        records_sha256=C.sha(forward/'records.json'),
        input_records_sha256=C.sha(forward/'input_records.json'),
        pair_count=len(inputs), case_count=len(rows), complete_case_inventory=True,
        all_inputs_prepared=all(r['input_valid'] for r in inputs),
        all_production_calls_completed=all(r['status'] == 'ok' for r in rows),
        attempted_calls=sum(r['attempted'] for r in rows),
        successful_calls=sum(r['status'] == 'ok' for r in rows),
        failed_calls=sum(r['status'] == 'failed' for r in rows),
        all_case_statuses_retained=True, all_available_raw_arrays_retained=True,
        started_unix=started, phase_finished_unix=phase_finished, finished_unix=time.time(),
        phase_wall_seconds=elapsed,
        phase_wall_passed=elapsed <= lock['gates']['production_max_phase_wall_seconds'],
        phase_timing_scope='started through forward/completion.json write; provenance finalization separate',
        real_inputs_read=any(r['cache_read_attempted'] for r in inputs), labels_read=False,
        independent_numerical_verification_pending=True, scientific_gate_passed=False, goal_complete=False)
    C.write(out/'forward_complete.json', seal)
    C.write(out/'completion.json', dict(files=C.inventory(out),
        producer_forward_complete_sha256=C.sha(out/'forward_complete.json'),
        complete_case_inventory=True, all_production_calls_completed=seal['all_production_calls_completed'],
        phase_wall_passed=seal['phase_wall_passed'], finished_unix=time.time(),
        post_phase_bookkeeping_seconds_before_completion_write=time.perf_counter()-clock-elapsed,
        total_seconds_before_completion_write=time.perf_counter()-clock,
        real_inputs_read=seal['real_inputs_read'], labels_read=False,
        scientific_gate_passed=False, goal_complete=False))
    return seal


def forward(out):
    out = Path(out)
    existed = any((out/name).exists() for name in ('started.json', 'completion.json', 'forward_complete.json', 'forward'))
    try:
        return _forward_started(out)
    except (Exception, KeyboardInterrupt) as error:
        if not existed and (out/'started.json').exists():
            C.write(out/'failure.json', dict(error_type=type(error).__name__, error=str(error),
                traceback=traceback.format_exc(), finished_unix=time.time(),
                complete_case_inventory=False, labels_read=False,
                scientific_gate_passed=False, goal_complete=False))
            if (out/'completion.json').exists():
                shutil.copyfile(out/'completion.json', out/'completion_before_failure.json')
            C.write(out/'completion.json', dict(
                files={n:h for n, h in C.inventory(out).items() if n != 'completion.json'},
                failure_sha256=C.sha(out/'failure.json'), complete_case_inventory=False,
                all_production_calls_completed=False, phase_wall_passed=False,
                finished_unix=time.time(), labels_read=False, scientific_gate_passed=False,
                goal_complete=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'forward'))
    parser.add_argument('--out', type=Path, default=C.DEFAULT_OUT)
    parser.add_argument('--audit-root', type=Path, default=C.DEFAULT_AUDIT)
    args = parser.parse_args()
    C.configure_runtime()
    if args.command == 'freeze':
        freeze(args.out.resolve(), args.audit_root.resolve())
    else:
        forward(args.out.resolve())


if __name__ == '__main__':
    main()
