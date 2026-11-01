"""R25 prelabel audit: inherited inputs and all-query conditional-t5 references.

The six accepted R20 cache fields are the complete input boundary. This audit
does not reconstruct cameras or features and does not import production model
mathematics. Torch's full-matrix float32 GEMM is deliberately shared to verify
the executed content arithmetic. Student integration uses the frozen R24
independent marginal/conditional-t5 implementation.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
from experiments import round25_student_contract as C
import torch
from scipy.special import logsumexp

from experiments import round24_student_reference as REF

APPROXIMATE_TIE_TOLERANCE = 1e-12


def upper_covariance(covariance):
    """Match the old Gaussian's C01 interpretation, without averaging entries."""
    value = np.asarray(covariance)
    if (value.dtype != np.float64 or value.ndim != 3 or value.shape[1:] != (2, 2)
            or not np.isfinite(value).all()):
        raise ValueError('finite float64 [Q,2,2] archived covariance required')
    result = value.copy()
    result[:, 1, 0] = value[:, 0, 1]
    return result


def trace_temperature(covariance, h=.25, temperature=10.):
    """Independent float64 stable trace ratio, rounded once to float32."""
    value = np.asarray(covariance)
    if (value.dtype != np.float64 or value.ndim != 3 or value.shape[1:] != (2, 2)
            or not np.isfinite(value).all() or not math.isfinite(h) or h <= 0
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError('finite covariance and fixed h/T required')
    h2 = h * h
    diagonal = np.diagonal(value, axis1=1, axis2=2)
    scale = np.maximum(diagonal.max(axis=1), h2)
    relative = h2 / scale
    denominator = relative + (diagonal / scale[:, None]).sum(axis=1)
    result = (temperature * (relative / denominator)).astype(np.float32)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('invalid trace temperature')
    return result


def content_arithmetic(query, key, tau):
    """One complete QxK Torch32 multiplication; never per-row GEMM."""
    query, key, tau = map(np.asarray, (query, key, tau))
    if (query.dtype != np.float32 or key.dtype != np.float32 or tau.dtype != np.float32
            or query.ndim != 2 or key.ndim != 2 or query.shape[1] != key.shape[1]
            or tau.shape != (len(query),) or not len(query) or not len(key)
            or not all(np.isfinite(a).all() for a in (query, key, tau))):
        raise ValueError('finite matching normalized inherited features and tau32 required')
    content = (torch.from_numpy(query) @ torch.from_numpy(key).T).numpy()
    bias = (torch.from_numpy(tau)[:, None] * torch.from_numpy(content)).numpy()
    if not np.isfinite(content).all() or not np.isfinite(bias).all():
        raise ValueError('nonfinite complete content arithmetic')
    return content, bias


def probabilities(raw, bounds, bias, floor_density):
    raw, bounds, bias = map(np.asarray, (raw, bounds, bias))
    if (raw.dtype != np.float64 or raw.shape != (len(bounds),)
            or bounds.dtype != np.float64 or bounds.shape != (len(raw), 4)
            or bias.dtype != np.float64 or bias.shape != raw.shape
            or not all(np.isfinite(a).all() for a in (raw, bounds, bias)) or np.any(raw < 0)):
        raise ValueError('finite float64 native raw/bounds/bias required')
    floor = np.prod(bounds[:, 2:] - bounds[:, :2], axis=1) * floor_density
    if not math.isfinite(floor_density) or floor_density <= 0 or np.any(floor <= 0) or not np.isfinite(floor).all():
        raise ValueError('positive finite single native area floor required')
    logits = np.log(raw + floor) + bias
    return np.exp(logits - logsumexp(logits)), raw + floor


def compare_raw(actual, reference, bounds, bias, *, gates, production=False,
                actual_probability=None, floor_density=1e-30):
    """Raw-relative checks precede floor; saved production P has its own gate."""
    actual, reference = map(np.asarray, (actual, reference))
    pa, _ = probabilities(actual, bounds, bias, floor_density)
    pr, _ = probabilities(reference, bounds, bias, floor_density)
    if actual_probability is not None:
        value = np.asarray(actual_probability)
        if value.dtype != np.float64 or value.shape != pa.shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('archived production probability is invalid')
        pa = value
    denominator = math.fsum(map(float, reference))
    if (not denominator > 0 or not math.isfinite(denominator) or np.any(reference <= 0)
            or (not production and np.any(actual <= 0))):
        raise ValueError('zero/nonfinite positive-mass reference is unresolved')
    raw_difference = np.abs(actual - reference)
    raw_relative = abs(math.fsum(map(float, actual)) - denominator) / denominator
    raw_l1 = math.fsum(map(float, raw_difference)) / denominator
    p_error = float(np.max(np.abs(pa - pr)))
    p_limit = gates['probability_max_absolute'] if production else gates['reference_probability_max_absolute']
    raw_limit = gates['raw_fov_relative'] if production else gates['reference_raw_relative']
    l1_limit = gates['raw_l1_over_reference_fov'] if production else gates['reference_raw_relative']
    relative = np.divide(raw_difference, reference, out=np.full_like(reference, np.inf), where=reference > 0)
    relative[(reference == 0) & (actual == 0)] = 0.
    detail = dict(passed=bool(p_error <= p_limit and raw_relative <= raw_limit and raw_l1 <= l1_limit),
        maximum_probability_error=p_error, raw_fov_actual=math.fsum(map(float, actual)),
        raw_fov_reference=denominator, raw_fov_relative=raw_relative,
        raw_l1_over_reference_fov=raw_l1, zero_reference_cells=int(np.count_nonzero(reference == 0)),
        maximum_per_cell_absolute_error=float(raw_difference.max()),
        maximum_finite_per_cell_relative_error=float(relative[np.isfinite(relative)].max()) if np.isfinite(relative).any() else None,
        infinite_per_cell_relative_errors=int(np.count_nonzero(~np.isfinite(relative))),
        actual_argmax=int(pa.argmax()), reference_argmax=int(pr.argmax()),
        argmax_changed=bool(pa.argmax() != pr.argmax()),
        actual_exact_maximizers=np.flatnonzero(pa == pa.max()).tolist(),
        reference_exact_maximizers=np.flatnonzero(pr == pr.max()).tolist(),
        approximate_tie_absolute_tolerance=APPROXIMATE_TIE_TOLERANCE,
        actual_approximate_maximizers=np.flatnonzero(pa.max() - pa <= APPROXIMATE_TIE_TOLERANCE).tolist(),
        reference_approximate_maximizers=np.flatnonzero(pr.max() - pr <= APPROXIMATE_TIE_TOLERANCE).tolist())
    return detail, dict(per_cell_absolute_error=raw_difference, per_cell_relative_error=relative,
                        per_cell_probability_error=np.abs(pa - pr))


def reference_work(bounds, policy, reference_policy, order):
    bounds = np.asarray(bounds)
    step = reference_policy['maximum_panel_side_in_h'] * policy['h']
    native = sum(max(1, math.ceil(float(b[2] - b[0]) / step)) for b in bounds)
    direct = max(1, math.ceil(float(bounds[:, 2].max() - bounds[:, 0].min()) / step))
    return (native + direct) * order


def input_arrays(cached, *, policy):
    """Construct this audit's inputs exclusively from six accepted cache arrays."""
    if set(cached) != set(C.FIELDS):
        raise ValueError('exact accepted six-field cache boundary required')
    a = cached['g24__a']
    cov = cached['g24__anchor_cov']
    query = cached['g24__q_features']
    key = cached['g8__k_features']
    bounds_px = cached['g8__bounds_px']
    tau_cached = cached['g24__tau32']
    if (a.dtype != np.float64 or a.shape != (48, 2) or cov.dtype != np.float64 or cov.shape != (48, 2, 2)
            or query.dtype != np.float32 or query.shape != (48, 128)
            or key.dtype != np.float32 or key.shape != (432, 128)
            or bounds_px.dtype != np.int64 or bounds_px.shape != (432, 4)
            or tau_cached.dtype != np.float32 or tau_cached.shape != (48,)
            or not all(np.isfinite(v).all() for v in cached.values())):
        raise ValueError('accepted cache dtype/shape changed')
    literal = np.asarray([(x, y, x + 8, y + 8) for y in range(0, 144, 8) for x in range(0, 192, 8)], np.int64)
    if not np.array_equal(bounds_px, literal):
        raise ValueError('literal 432 native half-pixel cells changed')
    tau = trace_temperature(cov, policy['h'], 10.)
    if not np.array_equal(tau, tau_cached):
        raise ValueError('accepted tau32 does not match inherited covariance')
    execution_cov = upper_covariance(cov)
    content, bias = content_arithmetic(query, key, tau)
    scale = np.maximum(1., np.max(np.abs(cov), axis=(1, 2)))
    normalized = cov / scale[:, None, None]
    asymmetry = np.max(np.abs(normalized - normalized.transpose(0, 2, 1)), axis=(1, 2))
    if np.any(asymmetry > 1e-12):
        raise ValueError('archived covariance exceeds inherited symmetry tolerance')
    change = np.max(np.abs(execution_cov - cov), axis=(1, 2))
    return dict(a64=a, covariance_archive64=cov, covariance64=execution_cov, tau32=tau,
        q_features32=query, k_features32=key, key_bounds_px=bounds_px,
        bounds64=(bounds_px.astype(np.float64) - .5) / 8., content32=content,
        bias32=bias, content_bias64=bias.astype(np.float64),
        covariance_asymmetry_normalized=asymmetry, covariance_change_max_absolute=change)


def load_cached(entry):
    path = Path(entry['cache_path'])
    if C.sha(path) != entry['cache_sha256']:
        raise ValueError('accepted R20 cache archive changed')
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in C.FIELDS}
    if {name: C.array_sha(value) for name, value in values.items()} != entry['cache_array_sha256']:
        raise ValueError('accepted whitelisted cache array identity changed')
    return values


def input_check(entry, actual, policy):
    expected = input_arrays(load_cached(entry), policy=policy)
    if set(actual) != set(expected):
        raise ValueError('pair input array schema differs')
    differing = [name for name in expected if actual[name].dtype != expected[name].dtype
                 or not np.array_equal(actual[name], expected[name])]
    if differing:
        raise ValueError('independent inherited input construction differs: ' + ','.join(differing))
    return dict(passed=True, scene=entry['scene'], pair=entry['pair'], pair_name=entry['pair_name'],
        cache_path=entry['cache_path'], cache_sha256=entry['cache_sha256'],
        fields_read=list(C.FIELDS), array_sha256={n: C.array_sha(v) for n, v in expected.items()},
        maximum_covariance_asymmetry_normalized=float(expected['covariance_asymmetry_normalized'].max()),
        maximum_covariance_upper_copy_change=float(expected['covariance_change_max_absolute'].max()),
        tau32_exact=True, full_matrix_torch32_content_exact=True, bias32_exact=True,
        camera_or_moment_reconstruction=False, labels_read=False)


def production_check(row, bank, inputs, index, policy, gates):
    if not row['passed']:
        return dict(passed=False, status='production_failed', failure_reason=row['failure_reason'])
    mean, covariance = inputs['a64'][index], inputs['covariance64'][index]
    bounds, bias = inputs['bounds64'], inputs['content_bias64'][index]
    _, _, _, fov, eigen = REF.validate(mean, covariance, bounds, bias, policy)
    raw, probability = bank['raw_mass'], bank['probability']
    expected, floored = probabilities(raw, bounds, bias, policy['floor_density'])
    if probability.dtype != np.float64 or probability.shape != raw.shape or not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError('invalid archived probability')
    normalization = abs(math.fsum(map(float, probability)) - 1.)
    arithmetic = float(np.max(np.abs(expected - probability)))
    side = policy['h'] * policy['maximum_tile_side_in_h']
    native_tiles = sum(int(np.prod(np.ceil((box[2:] - box[:2]) / side).astype(np.int64))) for box in bounds)
    fov_tiles = int(np.prod(np.ceil((fov[2:] - fov[:2]) / side).astype(np.int64)))
    work = dict(cell_node_evaluations=(native_tiles + fov_tiles) * policy['quadrature_order'] ** 2,
                native_tiles_completed=native_tiles, fov_tiles_completed=fov_tiles)
    raw_sum = math.fsum(map(float, raw))
    floor = np.prod(bounds[:, 2:] - bounds[:, :2], axis=1) * policy['floor_density']
    floor_sum = math.fsum(map(float, floor))
    direct = row['raw_fov_mass']
    if not math.isfinite(direct) or direct < 0:
        raise ValueError('direct FOV mass must be finite and nonnegative')
    direct_error = abs(direct - raw_sum)
    log_normalizer = float(logsumexp(np.log(floored) + bias))
    metadata = bool(row['degrees_of_freedom'] == 4 and row['quadrature_order'] == 24
        and row['quadrature_error_bound_available'] is False and row['tail_truncation_used'] is False
        and row['native_tile_count'] == native_tiles and row['fov_tile_count'] == fov_tiles and row['work'] == work
        and row['h'] == policy['h'] and row['floor_density'] == policy['floor_density']
        and row['admitted_covariance_min_eigenvalue'] == float(eigen[0])
        and np.isclose(row['raw_mass_sum'], raw_sum, rtol=1e-13, atol=0)
        and np.isclose(row['total_area_floor'], floor_sum, rtol=1e-13, atol=0)
        and np.isclose(row['normalizer'], math.fsum(map(float, floored)), rtol=1e-13, atol=0)
        and abs(row['log_content_normalizer'] - log_normalizer) <= gates['normalization_absolute']
        and row['zero_raw_cells'] == int(np.count_nonzero(raw == 0))
        and row['all_raw_mass_zero'] is bool(np.all(raw == 0))
        and row['area_prior_used'] is bool(np.all(raw == 0) and np.all(bias == bias[0]))
        and row['raw_mass_below_total_floor'] is bool(np.sum(raw) < np.sum(floor))
        and row['argmax_index'] == int(probability.argmax())
        and row['argmax_exact_ties'] == int(np.count_nonzero(probability == probability.max()))
        and row['argmax_near_ties_at_1e_12'] == int(np.count_nonzero(probability.max() - probability <= APPROXIMATE_TIE_TOLERANCE)))
    winner = bank['argmax']
    passed = bool(metadata and normalization <= gates['normalization_absolute']
        and arithmetic <= gates['normalization_absolute'] and np.array_equal(floored, bank['floored_mass'])
        and direct_error <= gates['direct_fov_additivity_absolute']
        and work['cell_node_evaluations'] <= policy['max_cell_node_evaluations']
        and winner.dtype == np.int32 and winner.shape == () and int(winner) == int(probability.argmax()))
    return dict(passed=passed, status='checked', normalization_error=normalization,
        same_raw_probability_error=arithmetic, metadata_consistent=metadata,
        raw_fov_mass=raw_sum, direct_fov_mass=direct, direct_fov_additivity_error=direct_error,
        total_floor=floor_sum, raw_to_total_floor=raw_sum / floor_sum,
        raw_below_total_floor=bool(raw_sum < floor_sum), all_raw_mass_zero=bool(np.all(raw == 0)),
        zero_raw_cells=int(np.count_nonzero(raw == 0)),
        exact_maximizers=np.flatnonzero(probability == probability.max()).tolist(),
        approximate_maximizers=np.flatnonzero(probability.max() - probability <= APPROXIMATE_TIE_TOLERANCE).tolist(),
        work_expected=work)


def joint_locks(run, out):
    lock, design = C.verify(run), C.verify(out)
    if (lock['kind'] != 'producer' or design['kind'] != 'audit'
            or Path(lock['audit_root']).resolve() != out.resolve()
            or Path(design['producer_root']).resolve() != run.resolve()
            or lock['source_hashes'] != design['source_hashes'] or lock['runtime'] != design['runtime']
            or lock['entries'] != design['entries']
            or not lock['created_unix'] <= design['created_unix'] <= time.time()):
        raise ValueError('joint producer/audit locks differ')
    return lock, design


def freeze(run, out):
    if any((run / name).exists() for name in ('started.json', 'records.json', 'arrays', 'forward', 'completion.json', 'forward_complete.json')):
        raise ValueError('independent freeze must precede actual cache reads and predictions')
    return C.freeze(out, kind='audit', producer_root=run, audit_root=out)


def read_bank(path, expected_hashes):
    with np.load(path, allow_pickle=False) as archive:
        bank = {name: archive[name] for name in archive.files}
    if {name: C.array_sha(value) for name, value in bank.items()} != expected_hashes:
        raise ValueError('saved array canonical identities differ')
    return bank


def verify_seal(run, out):
    """Authenticate the complete producer population before decoding its arrays."""
    lock, design = joint_locks(run, out)
    complete = C.read(run / 'completion.json')
    C.verify_inventory(run, complete['files'], exclude=('completion.json',))
    forward = run / 'forward'
    forward_complete = C.read(forward / 'completion.json')
    C.verify_inventory(forward, forward_complete['files'], exclude=('completion.json',))
    seal = C.read(run / 'forward_complete.json')
    started = C.read(run / 'started.json')
    inputs, rows = C.read(forward / 'input_records.json'), C.read(forward / 'records.json')
    expected = dict(producer_lock_sha256=C.sha(run / 'lock.json'), audit_lock_sha256=C.sha(out / 'lock.json'),
        forward_completion_sha256=C.sha(forward / 'completion.json'), records_sha256=C.sha(forward / 'records.json'),
        input_records_sha256=C.sha(forward / 'input_records.json'), pair_count=48, case_count=2304,
        complete_case_inventory=True, all_case_statuses_retained=True, all_available_raw_arrays_retained=True,
        labels_read=False, scientific_gate_passed=False, goal_complete=False)
    if any(seal.get(k) != v for k, v in expected.items()) or len(inputs) != 48 or len(rows) != 2304:
        raise ValueError('complete sealed 48-pair/2304-query population required')
    if (complete.get('complete_case_inventory') is not True
            or complete['producer_forward_complete_sha256'] != C.sha(run / 'forward_complete.json')
            or forward_complete['records_sha256'] != expected['records_sha256']
            or forward_complete['input_records_sha256'] != expected['input_records_sha256']
            or forward_complete['pair_count'] != 48 or forward_complete['case_count'] != 2304
            or forward_complete.get('complete_case_inventory') is not True
            or started['producer_lock_sha256'] != expected['producer_lock_sha256']
            or started['audit_lock_sha256'] != expected['audit_lock_sha256']
            or not design['created_unix'] <= started['started_unix'] == seal['started_unix'] <= seal['phase_finished_unix'] <= seal['finished_unix']
            or not math.isfinite(seal['phase_wall_seconds']) or seal['phase_wall_seconds'] < 0
            or seal['phase_wall_passed'] != (seal['phase_wall_seconds'] <= lock['gates']['production_max_phase_wall_seconds'])
            or complete['phase_wall_passed'] != seal['phase_wall_passed']):
        raise ValueError('producer chronology, phase or seal binding differs')
    expected_forward = {'records.json', 'input_records.json'}
    previous = seal['started_unix']
    for position, (entry, record) in enumerate(zip(lock['entries'], inputs)):
        identity = {k: entry[k] for k in ('scene', 'pair', 'pair_name')}
        directory = Path(entry['scene'].replace('/', '__')) / entry['pair_name']
        prefix = Path('forward') / directory
        if (any(record.get(k) != v for k, v in identity.items())
                or record['cache_path'] != entry['cache_path'] or record['cache_sha256'] != entry['cache_sha256']
                or record['cache_array_sha256'] != entry['cache_array_sha256']
                or type(record['input_valid']) is not bool or type(record['cache_read_attempted']) is not bool
                or record['status'] != ('ok' if record['input_valid'] else 'failed')
                or not previous <= record['started_unix'] <= record['finished_unix']
                or not math.isfinite(record['wall_seconds']) or record['wall_seconds'] < 0
                or C.read(run / prefix / 'inputs.json') != record):
            raise ValueError('ordered pair input identity/status/chronology differs')
        expected_forward.add(str(directory / 'inputs.json'))
        if record['input_valid']:
            if (record['cache_read_attempted'] is not True or record['input_path'] != str(prefix / 'inputs.npz')
                    or C.sha(run / record['input_path']) != record['input_sha256']):
                raise ValueError('prepared input archive binding differs')
            expected_forward.add(str(directory / 'inputs.npz'))
        else:
            if (not record.get('failure_reason') or record['input_path'] is not None
                    or record['input_sha256'] is not None or record['array_sha256'] != {}):
                raise ValueError('failed input must preserve reason without fabricated archive')
            if 'available_cache' in record:
                cache = record['available_cache']
                if (cache['archive_path'] != str(prefix / 'failed_cache.npz')
                        or C.sha(run / cache['archive_path']) != cache['archive_sha256']
                        or set(cache['array_sha256']) != set(C.FIELDS)):
                    raise ValueError('failed preparation cache retention differs')
                expected_forward.add(str(directory / 'failed_cache.npz'))
        previous = record['finished_unix']
        for index, row in enumerate(rows[position * 48:(position + 1) * 48]):
            name = 'q%04d' % index
            if (any(row.get(k) != v for k, v in identity.items()) or row['query_index'] != index
                    or type(row['query_index']) is not int or type(row['passed']) is not bool
                    or type(row['attempted']) is not bool
                    or row['status'] != ('ok' if row['passed'] else 'failed')
                    or row['model_status'] != ('complete' if row['passed'] else 'failed')
                    or row['input_path'] != record['input_path'] or row['input_sha256'] != record['input_sha256']
                    or row['archive_path'] != str(prefix / (name + '.npz'))
                    or row['archive_sha256'] != C.sha(run / row['archive_path'])
                    or C.read(run / prefix / (name + '.json')) != row
                    or not previous <= row['started_unix'] <= row['finished_unix'] <= seal['phase_finished_unix']
                    or not math.isfinite(row['call_and_output_seconds']) or row['call_and_output_seconds'] < 0
                    or type(row['peak_rss_kib_process']) is not int or row['peak_rss_kib_process'] < 0
                    or any(type(v) is not int or v < 0 for v in row['work'].values())):
                raise ValueError('ordered query identity/status/work/chronology differs')
            if row['passed'] and (not row['attempted'] or not record['input_valid']):
                raise ValueError('successful output must follow prepared input and attempted call')
            if not row['passed'] and not row.get('failure_reason'):
                raise ValueError('failed query requires explicit reason')
            if not row['attempted'] and (row['work'] != {} or row['array_sha256'] != {}):
                raise ValueError('unattempted query must not fabricate work or output')
            expected_forward |= {str(directory / (name + '.npz')), str(directory / (name + '.json'))}
            previous = row['finished_unix']
    if set(forward_complete['files']) != expected_forward:
        raise ValueError('forward file population differs from fixed pair/query schema')
    expected_root = {'lock.json', 'lock.sha256', 'started.json', 'forward_complete.json', 'forward/completion.json'}
    expected_root |= {'source_snapshot/' + n for n in lock['source_hashes']}
    expected_root |= {'forward/' + n for n in expected_forward}
    if set(complete['files']) != expected_root:
        raise ValueError('producer inventory contains unbound extra/missing files')
    status = dict(all_inputs_prepared=all(r['input_valid'] for r in inputs),
        all_production_calls_completed=all(r['passed'] for r in rows), attempted_calls=sum(r['attempted'] for r in rows),
        successful_calls=sum(r['passed'] for r in rows), failed_calls=sum(not r['passed'] for r in rows))
    if any(seal.get(k) != v for k, v in status.items()) or complete['all_production_calls_completed'] != status['all_production_calls_completed']:
        raise ValueError('population outcome metadata inconsistent')
    return lock, seal, inputs, rows


def query_bank(run, row):
    bank = read_bank(run / row['archive_path'], row['array_sha256'])
    if row['passed']:
        if set(bank) != {'raw_mass', 'probability', 'floored_mass', 'argmax'}:
            raise ValueError('successful query arrays differ')
    elif not set(bank) <= {'partial_raw_mass', 'partial_raw_fov_mass'}:
        raise ValueError('failed query arrays differ')
    for name, value in bank.items():
        dtype, shape = (np.int32, ()) if name == 'argmax' else (np.float64, (1,) if name == 'partial_raw_fov_mass' else (432,))
        if value.dtype != dtype or value.shape != shape:
            raise ValueError('query array dtype/shape differs: ' + name)
    return bank


def append_record(path, record):
    import json
    with path.open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()


def reference_attempt(inputs, index, order, *, policy, reference_policy, gates,
                      phase, cumulative, before_call=None):
    """One fixed order; exceptions retain actual ledger or a stated upper charge."""
    tick = time.perf_counter()
    evidence, attempted, reserved, charged = {}, False, 0, False
    record = dict(order=order, passed=False, attempted=False,
                  work=dict(cell_node_evaluations=0), work_accounting='not_attempted')
    try:
        if inputs is None:
            raise ValueError('sealed producer input unavailable or failed independent input audit')
        bounds, bias = inputs['bounds64'], inputs['content_bias64'][index]
        expected = reference_work(bounds, policy, reference_policy, order)
        if time.perf_counter() - phase >= reference_policy['max_phase_wall_seconds']:
            raise RuntimeError('fixed independent reference phase deadline exhausted before call')
        if cumulative + expected > reference_policy['max_total_cell_node_evaluations']:
            raise RuntimeError('fixed independent reference cell-node budget exhausted before call')
        # Precharge the complete fixed call. An asynchronous interruption cannot
        # erase its work; an ordinary failure replaces this reservation with
        # the frozen reference's precise attempted-batch ledger when available.
        attempted, reserved, charged = True, expected, True
        cumulative += expected
        if before_call is not None:
            before_call(dict(order=order, reserved_cell_node_evaluations=expected,
                             cumulative_reserved_cell_node_evaluations=cumulative))
        result = REF.integrate(inputs['a64'][index], inputs['covariance64'][index], bounds, bias,
            policy=policy, reference_policy=reference_policy, order=order,
            deadline=phase + reference_policy['max_phase_wall_seconds'])
        consumed = result['work']['cell_node_evaluations']
        cumulative += consumed - reserved
        reserved = consumed
        record.update(work=dict(cell_node_evaluations=consumed), work_accounting='reference_ledger')
        raw = result.pop('raw_mass')
        evidence['raw_mass'] = raw
        if type(consumed) is not int or consumed != expected:
            raise ValueError('independent reference fixed work counter differs')
        if (raw.dtype != np.float64 or raw.shape != (len(bounds),) or not np.isfinite(raw).all()
                or np.any(raw <= 0) or not math.isfinite(result['raw_fov_mass']) or result['raw_fov_mass'] <= 0):
            raise ValueError('nonpositive/nonfinite mathematically positive reference mass is unresolved')
        probability, _ = probabilities(raw, bounds, bias, policy['floor_density'])
        evidence['probability'] = probability
        direct_error = abs(result['raw_fov_mass'] - math.fsum(map(float, raw)))
        normalization = abs(math.fsum(map(float, probability)) - 1.)
        elapsed = time.perf_counter() - phase
        record.update(**result, passed=bool(normalization <= gates['normalization_absolute']
            and direct_error <= gates['direct_fov_additivity_absolute']
            and cumulative <= reference_policy['max_total_cell_node_evaluations']
            and elapsed <= reference_policy['max_phase_wall_seconds']),
            normalization_error=normalization, direct_fov_additivity_error=direct_error,
            all_reference_cells_positive=True)
    except Exception as error:
        ledger = getattr(error, 'reference_work', None)
        if charged:
            consumed = ledger['cell_node_evaluations'] if ledger is not None else reserved
            cumulative += consumed - reserved
            record.update(work=dict(cell_node_evaluations=consumed),
                work_accounting=('reference_ledger' if ledger is not None or record['work_accounting'] == 'reference_ledger'
                                 else 'conservative_full_call_reservation'))
        if hasattr(error, 'partial_raw_mass'):
            evidence['partial_raw_mass'] = error.partial_raw_mass
        record.update(passed=False, failure_reason=type(error).__name__ + ': ' + str(error))
    record.update(attempted=attempted, cumulative_cell_node_evaluations=cumulative,
                  wall_seconds=time.perf_counter() - tick, phase_elapsed_seconds=time.perf_counter() - phase)
    return record, evidence, cumulative


def _save_evidence(out, relative, values):
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('refusing to overwrite independent reference evidence')
    np.savez_compressed(path, **values)
    return dict(archive_path=relative, archive_sha256=C.sha(path),
                array_sha256={name: C.array_sha(value) for name, value in values.items()})


def _complete(out, verification, summary, phase, elapsed):
    C.write(out / 'summary.json', summary)
    verification.update(summary_sha256=C.sha(out / 'summary.json'))
    C.write(out / 'forward_verification.json', verification)
    C.write(out / 'completion.json', dict(passed=verification['passed'],
        forward_verification_sha256=C.sha(out / 'forward_verification.json'),
        files={n: h for n, h in C.inventory(out).items() if n != 'completion.json'},
        labels_read=False, scientific_gate_passed=False, goal_complete=False, finished_unix=time.time(),
        post_phase_bookkeeping_seconds_before_completion_write=time.perf_counter() - phase - elapsed,
        total_seconds_before_completion_write=time.perf_counter() - phase))


def audit(run, out):
    run, out = Path(run), Path(out)
    if any((out / n).exists() for n in ('audit_started.json', 'completion.json', 'forward_verification.json')):
        raise ValueError('refusing to overwrite a prelabel audit attempt')
    started, phase = time.time(), time.perf_counter()
    C.write(out / 'audit_started.json', dict(started_unix=started, labels_read=False))
    inputs_checked, production, references, comparisons = [], [], [], []
    cumulative, calls, reservation = 0, 0, 0
    try:
        lock, seal, input_records, rows = verify_seal(run, out)
        if started < seal['finished_unix']:
            raise ValueError('independent audit must follow complete global production seal')
        for position, (entry, input_record) in enumerate(zip(lock['entries'], input_records)):
            identity = {name: entry[name] for name in ('scene', 'pair', 'pair_name')}
            arrays = None
            try:
                if not input_record['input_valid']:
                    raise ValueError(input_record['failure_reason'])
                actual = read_bank(run / input_record['input_path'], input_record['array_sha256'])
                checked = input_check(entry, actual, lock['policy'])
                arrays = actual
            except Exception as error:
                checked = dict(**identity, passed=False, failure_reason=type(error).__name__ + ': ' + str(error),
                    labels_read=False, input_valid=input_record['input_valid'])
            inputs_checked.append(checked)
            append_record(out / 'input_checks.jsonl', checked)
            for index, row in enumerate(rows[position * 48:(position + 1) * 48]):
                uid = '%04d_q%04d' % (position, index)
                base = dict(**identity, id=uid, query_index=index)
                bank = None
                try:
                    bank = query_bank(run, row)
                    if arrays is None:
                        raise ValueError('input audit failed or sealed input unavailable')
                    check = production_check(row, bank, arrays, index, lock['policy'], lock['gates'])
                except Exception as error:
                    check = dict(passed=False, status='audit_failure', failure_reason=type(error).__name__ + ': ' + str(error))
                check.update(base)
                production.append(check)
                append_record(out / 'production_checks.jsonl', check)
                values, order_records = {}, {}
                for order in lock['reference_policy']['orders']:
                    def before_call(value):
                        nonlocal reservation, calls
                        reservation = value['reserved_cell_node_evaluations']
                        calls += 1
                        append_record(out / 'reference_call_starts.jsonl', dict(**base, **value))
                    rec, evidence, cumulative = reference_attempt(arrays, index, order,
                        policy=lock['policy'], reference_policy=lock['reference_policy'], gates=lock['gates'],
                        phase=phase, cumulative=cumulative, before_call=before_call)
                    reservation = 0
                    append_record(out / 'reference_results.jsonl', dict(**base, **rec,
                        available_array_sha256={n: C.array_sha(v) for n, v in evidence.items()}))
                    saved = _save_evidence(out, 'reference_arrays/' + uid + '_%d.npz' % order, evidence)
                    rec.update(base, **saved)
                    references.append(rec)
                    append_record(out / 'reference_records.jsonl', rec)
                    order_records[order] = rec
                    if rec['passed']:
                        values[order] = evidence
                comparison = dict(**base, passed=False)
                error_arrays = {}
                if set(values) == set(lock['reference_policy']['orders']):
                    try:
                        low, high = lock['reference_policy']['orders']
                        comparison['reference'], error = compare_raw(values[low]['raw_mass'], values[high]['raw_mass'],
                            arrays['bounds64'], arrays['content_bias64'][index], gates=lock['gates'],
                            floor_density=lock['policy']['floor_density'])
                        error_arrays.update({'reference_' + k: v for k, v in error.items()})
                        if check['passed']:
                            comparison['production'], error = compare_raw(bank['raw_mass'], values[high]['raw_mass'],
                                arrays['bounds64'], arrays['content_bias64'][index], gates=lock['gates'], production=True,
                                actual_probability=bank['probability'], floor_density=lock['policy']['floor_density'])
                            error_arrays.update({'production_' + k: v for k, v in error.items()})
                        comparison['passed'] = bool(comparison['reference']['passed'] and comparison.get('production', {}).get('passed', False))
                    except Exception as error:
                        comparison['failure_reason'] = type(error).__name__ + ': ' + str(error)
                else:
                    comparison['failure_reason'] = 'one or both fixed independent reference orders failed'
                comparison.update(_save_evidence(out, 'comparison_arrays/' + uid + '.npz', error_arrays))
                comparisons.append(comparison)
                append_record(out / 'comparisons.jsonl', comparison)
            # The append-only records above preserve each completed attempt on
            # interruption; compact canonical tables are checkpointed per pair.
            for name, values_ in (('input_checks', inputs_checked), ('production_checks', production),
                                  ('reference_records', references), ('comparisons', comparisons)):
                C.write(out / (name + '.json'), values_)
            print('AUDIT_PAIR', entry['scene'], entry['pair_name'],
                  sum(r['passed'] for r in comparisons[-48:]), '/48', flush=True)
        joint_locks(run, out)
        elapsed, phase_finished = time.perf_counter() - phase, time.time()
        budget = bool(elapsed <= lock['reference_policy']['max_phase_wall_seconds']
                      and cumulative <= lock['reference_policy']['max_total_cell_node_evaluations'])
        full = bool(len(inputs_checked) == 48 and len(production) == len(comparisons) == 2304
                    and len(references) == 4608 and calls == 4608)
        passed = bool(full and budget and seal['phase_wall_passed'] and seal['all_inputs_prepared']
            and seal['all_production_calls_completed'] and all(r['passed'] for r in inputs_checked + production + references + comparisons))
        summary = dict(numerical_gate_passed=passed, passed=passed, checked_pairs=len(inputs_checked),
            checked_queries=len(production), reference_call_count=calls, reference_attempt_record_count=len(references),
            all_query_reference_population_checked=full, reference_cell_node_evaluations=cumulative,
            reference_phase_wall_seconds=elapsed, reference_budget_passed=budget,
            production_phase_wall_seconds=seal['phase_wall_seconds'], production_phase_wall_passed=seal['phase_wall_passed'],
            input_failure_count=sum(not r['passed'] for r in inputs_checked),
            production_failure_count=sum(not r['passed'] for r in production),
            reference_failure_count=sum(not r['passed'] for r in references),
            comparison_failure_count=sum(not r['passed'] for r in comparisons),
            reference_calls_not_attempted=sum(not r['attempted'] for r in references),
            real_inputs_read=any(r['input_valid'] for r in input_records), labels_read=False,
            scientific_gate_passed=False, goal_complete=False, candidate_selected=False,
            started_unix=started, phase_finished_unix=phase_finished, finished_unix=time.time(),
            next_action='root_acceptance_before_old_train_labels' if passed else 'stop_before_labels_preserve_all_failures',
            note='Accepted six-field R20 preparation is inherited, not rederived. Full fixed old population numerical evidence; no general quadrature certificate.')
        verification = dict(passed=passed, numerical_gate_passed=passed, delivery_integrity_passed=True,
            checked_pairs=len(inputs_checked), checked_queries=len(production), reference_call_count=calls,
            reference_attempt_record_count=len(references), all_query_reference_population_checked=full,
            producer_lock_sha256=C.sha(run / 'lock.json'), audit_lock_sha256=C.sha(out / 'lock.json'),
            producer_forward_complete_sha256=C.sha(run / 'forward_complete.json'),
            producer_records_sha256=C.sha(run / 'forward/records.json'), producer_completion_sha256=C.sha(run / 'completion.json'),
            source_count=len(lock['source_hashes']), real_inputs_read=summary['real_inputs_read'],
            labels_read=False, scientific_gate_passed=False, goal_complete=False, finished_unix=time.time())
        _complete(out, verification, summary, phase, elapsed)
        return verification
    except (Exception, KeyboardInterrupt) as error:
        # The reservation is an explicit upper charge for an interrupted call;
        # its start journal identifies the query/order. No incomplete audit
        # can unlock label access.
        elapsed = time.perf_counter() - phase
        C.write(out / 'failure.json', dict(error_type=type(error).__name__, message=str(error),
            reference_cell_node_evaluations_completed=cumulative,
            interrupted_call_reserved_upper_cell_nodes=reservation,
            started_unix=started, finished_unix=time.time(), labels_read=False))
        summary = dict(passed=False, numerical_gate_passed=False, checked_pairs=len(inputs_checked),
            checked_queries=len(production), reference_call_count=calls, reference_attempt_record_count=len(references),
            all_query_reference_population_checked=False, labels_read=False, scientific_gate_passed=False,
            goal_complete=False, failure_sha256=C.sha(out / 'failure.json'), reference_phase_wall_seconds=elapsed)
        verification = dict(summary, delivery_integrity_passed=False, finished_unix=time.time())
        _complete(out, verification, summary, phase, elapsed)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'audit'))
    parser.add_argument('--run', type=Path, default=C.DEFAULT_OUT)
    parser.add_argument('--out', type=Path, default=C.DEFAULT_AUDIT)
    args = parser.parse_args()
    C.configure_runtime()
    if args.command == 'freeze':
        freeze(args.run.resolve(), args.out.resolve())
    else:
        audit(args.run.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
