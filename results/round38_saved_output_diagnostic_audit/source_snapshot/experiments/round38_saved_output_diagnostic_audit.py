"""Independent algebra/geometry audit of frozen R38 saved-output diagnostics.

This module contains no model or integration imports.  Only the producer's
authenticated, allowlisted source reader is shared; statistical calculations
below are independent, scalar implementations.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
import traceback

import numpy as np

from experiments import round38_rms_anchor_contract as C


ALGEBRA_TOL = 2e-12
FLOAT_ATOL = 1e-9
FLOAT_RTOL = 1e-10
STAGES = ('raw', 'floor', 'logits', 'log_probability', 'probability')
STAT_PATHS = (
    'projection.old_anchor_distance_px', 'depth.a_z', 'depth.sigma_z', 'depth.rms_depth',
    'depth.normalized_mean', 'depth.normalized_std', 'depth.relative_std',
    'depth.front_mass', 'depth.tau32', 'distribution.raw_entropy',
    'distribution.raw_peak', 'distribution.floor_entropy',
    'distribution.floor_peak', 'distribution.log_raw_to_floor_total',
    'content.geometry_loss', 'content.content_gain', 'content.logit_advantage',
    'content.bookkeeping_residual',
) + tuple('stages.' + stage + '.' + field for stage in STAGES for field in (
    'top_two_gap', 'anchor_log_gap', 'rectangle_distance_px',
    'min_rectangle_distance_px', 'cell_center_distance_px'))
SCOPE = dict(labels_read=False, new_model_calls=0, new_reference_calls=0,
             new_candidate_selected=False, new_task_metrics_computed=False,
             independent_probability_reference=False, scientific_gate_passed=False,
             goal_complete=False, continuous_density_mode_tested=False,
             tau_mismatch_causality_established=False)


def _lse(values):
    numbers = [float(v) for v in values]
    maximum = max(numbers)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in numbers))


def _logadd(a, b):
    high, low = max(a, b), min(a, b)
    if low == -math.inf:
        return high
    return high + math.log1p(math.exp(low - high))


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _ratio(a, b):
    if b == 0:
        return None
    # Python division returns inf on overflow, but does not raise for finite
    # nonzero denominators.  Undefined/overflow states are retained as null.
    return _finite(float(a) / float(b))


def _quantile(numbers, q):
    """Fixed linear interpolation: position (n-1)*q in sorted observations."""
    values = sorted(float(v) for v in numbers)
    if not values:
        return None
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    fraction = position - low
    return values[low] + fraction * (values[high] - values[low])


def _distribution(values):
    valid = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in valid):
        raise ValueError('nonfinite summary observation')
    return dict(n=len(valid),
                mean=math.fsum(valid) / len(valid) if valid else None,
                min=min(valid) if valid else None,
                max=max(valid) if valid else None,
                p50=_quantile(valid, .5), p90=_quantile(valid, .9),
                p99=_quantile(valid, .99))


def _rect_distance(point, box):
    dx = max(float(box[0]) - point[0], 0., point[0] - float(box[2]))
    dy = max(float(box[1]) - point[1], 0., point[1] - float(box[3]))
    return 8. * math.hypot(dx, dy)


def _contains(point, box):
    return (float(box[0]) <= point[0] <= float(box[2])
            and float(box[1]) <= point[1] <= float(box[3]))


def _maximizers(values):
    scores = [float(v) for v in values]
    maximum = max(scores)
    indices = [i for i, value in enumerate(scores) if value == maximum]
    ordered = sorted(scores, reverse=True)
    gap = ordered[0] - ordered[1] if len(ordered) > 1 else None
    return indices, gap


def _entropy(logs):
    logz = _lse(logs)
    weights = [math.exp(float(v) - logz) for v in logs]
    entropy = -math.fsum(p * (float(v) - logz)
                         for p, v in zip(weights, logs) if p > 0)
    return entropy, max(weights)


def compare(expected, actual, path='root'):
    """Compare exact identities/states/ties and tolerant scalar algebra only."""
    errors = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [path + ': expected dictionary']
        if set(expected) != set(actual):
            errors.append(path + ': field membership differs')
        for key in sorted(set(expected) & set(actual)):
            errors.extend(compare(expected[key], actual[key], path + '.' + key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [path + ': list type/length differs']
        for index, (left, right) in enumerate(zip(expected, actual)):
            errors.extend(compare(left, right, path + '[' + str(index) + ']'))
    elif type(expected) is float:
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual)
                or not math.isclose(expected, actual,
                                    rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL)):
            errors.append(path + ': numeric value differs')
    elif type(actual) is not type(expected) or actual != expected:
        errors.append(path + ': exact value/type differs')
    return errors


def independent_row(payload):
    """Reconstruct one complete row without any producer statistics helpers."""
    spec, record = payload['spec'], payload['record']
    arrays, pair = payload['arrays'], payload['pair']
    mu = np.asarray(arrays['input__mu'])
    factor = np.asarray(arrays['input__factor'])
    bounds = np.asarray(arrays['input__bounds'])
    bias = np.asarray(arrays['input__content_bias'])
    raw = np.asarray(arrays['log_raw_mass'])
    floor = np.asarray(arrays['log_floored_mass'])
    logp = np.asarray(arrays['log_probability'])
    probability = np.asarray(arrays['probability'])
    n = len(bounds)
    if (mu.shape != (3,) or factor.shape != (3, 3)
            or bounds.shape != (n, 4) or n < 2
            or any(v.shape != (n,) for v in (bias, raw, floor, logp, probability))
            or any(v.dtype != np.float64 for v in
                   (mu, factor, bounds, bias, raw, floor, logp, probability))):
        raise ValueError('canonical float64 source shapes required')
    if (not all(np.isfinite(v).all() for v in
                (mu, factor, bounds, bias, floor, logp, probability))
            or np.any(probability < 0)
            or np.any(bounds[:, 2:] <= bounds[:, :2])):
        raise ValueError('invalid finite source values or rectangle sides')
    if (not np.array_equal(mu, pair['anchor_homogeneous'])
            or not np.array_equal(factor, pair['anchored_factor'])
            or not np.array_equal(bounds, pair['key_bounds'])
            or not np.array_equal(bounds, (np.asarray(pair['key_bounds_px']) - .5) / 8.)):
        raise ValueError('saved pair/query identity differs')
    old = np.asarray(pair['old_a2'])
    if old.shape != (2,) or not np.isfinite(old).all():
        raise ValueError('finite old two-dimensional anchor required')
    tau = float(pair['tau32'])
    if not math.isfinite(tau):
        raise ValueError('finite original temperature required')
    az = float(mu[2])
    sigma = math.hypot(*(float(v) for v in factor[2]))
    rms = math.hypot(az, *(float(v) for v in factor[2]))
    if not math.isfinite(sigma) or not math.isfinite(rms):
        raise ValueError('depth norm overflow')
    constant = all(float(v) == 0 for v in factor[2])
    branch = ('zero_depth_rms' if rms == 0 else
              'nonpositive_constant_depth' if constant and az <= 0 else
              'positive_constant_depth_gaussian' if constant else 'variable_depth')
    zero = branch in ('zero_depth_rms', 'nonpositive_constant_depth')
    if (record['structural_branch'] != branch or record['structural_zero'] is not zero
            or ('rms_depth' in record and record['rms_depth'] != rms)):
        raise ValueError('saved structural branch inconsistent with inputs')
    if (not np.isneginf(raw).all() if zero else not np.isfinite(raw).all()):
        raise ValueError('raw logs inconsistent with structural branch')
    front = float(record['front_mass'])
    if not math.isfinite(front) or not 0 <= front <= 1:
        raise ValueError('front mass outside probability range')

    xy = None
    projection_status = 'zero_depth'
    if az != 0:
        coordinates = [float(mu[0]) / az, float(mu[1]) / az]
        if all(math.isfinite(v) for v in coordinates):
            xy, projection_status = coordinates, 'finite'
        else:
            projection_status = 'nonfinite_division'
    fov = [min(float(b[0]) for b in bounds), min(float(b[1]) for b in bounds),
           max(float(b[2]) for b in bounds), max(float(b[3]) for b in bounds)]
    anchor_cells = [] if xy is None else [i for i, b in enumerate(bounds) if _contains(xy, b)]
    if xy is None:
        fov_status, boundary = 'undefined', None
    elif not _contains(xy, fov):
        fov_status, boundary = 'outside', False
    else:
        outer = xy[0] in (fov[0], fov[2]) or xy[1] in (fov[1], fov[3])
        fov_status = 'outer_boundary' if outer else 'interior'
        boundary = any(xy[0] in (float(bounds[i, 0]), float(bounds[i, 2]))
                       or xy[1] in (float(bounds[i, 1]), float(bounds[i, 3]))
                       for i in anchor_cells)
    primary = az > 0 and xy is not None and fov_status in ('interior', 'outer_boundary') and not zero
    delta = None if xy is None else [8. * (xy[j] - float(old[j])) for j in range(2)]
    old_distance = None if delta is None else math.hypot(*delta)
    if delta is not None and (any(not math.isfinite(v) for v in delta)
                              or not math.isfinite(old_distance)):
        raise ValueError('old anchor distance overflow')
    relative = _ratio(sigma, az) if az > 0 else None
    relative_status = ('nonpositive_mean' if az <= 0 else
                       'defined' if relative is not None else 'nonfinite_division')

    logfloors = []
    for box in bounds:
        area_floor = (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])) * 1e-30
        if not math.isfinite(area_floor) or area_floor <= 0:
            raise ValueError('positive finite native area floor required')
        logfloors.append(math.log(area_floor))
    logits = [float(g) + float(b) for g, b in zip(floor, bias)]
    logz = _lse(logits)
    floor_error = max(abs(float(g) - _logadd(float(r), f))
                      for r, f, g in zip(raw, logfloors, floor))
    logp_error = max(abs(float(p) - (v - logz)) for p, v in zip(logp, logits))
    material_error = max(abs(float(p) - math.exp(float(lp))) for p, lp in zip(probability, logp))
    normalization_error = abs(math.fsum(float(p) for p in probability) - 1.)
    if max(floor_error, logp_error, material_error, normalization_error) > ALGEBRA_TOL:
        raise ValueError('saved floor/content/materialization algebra fails absolute tolerance')

    def stage(scores, raw_zero=False, logarithmic=True):
        result = dict(defined=not raw_zero, first=None, maximizers=[], tie_count=None,
                      top_two_gap=None, anchor_set_contains=None, anchor_first_contains=None,
                      anchor_log_gap=None, rectangle_distance_px=None,
                      min_rectangle_distance_px=None, cell_center_distance_px=None)
        if raw_zero:
            return result
        winners, gap = _maximizers(scores)
        first = winners[0]
        result.update(first=first, maximizers=winners, tie_count=len(winners), top_two_gap=gap)
        if primary:
            result.update(anchor_set_contains=any(i in anchor_cells for i in winners),
                          anchor_first_contains=first in anchor_cells,
                          anchor_log_gap=(max(float(v) for v in scores)
                                          - max(float(scores[i]) for i in anchor_cells))
                                         if logarithmic else None,
                          rectangle_distance_px=_rect_distance(xy, bounds[first]),
                          min_rectangle_distance_px=min(_rect_distance(xy, bounds[i]) for i in winners),
                          cell_center_distance_px=8. * math.hypot(
                              xy[0] - (float(bounds[first, 0]) + float(bounds[first, 2])) / 2.,
                              xy[1] - (float(bounds[first, 1]) + float(bounds[first, 3])) / 2.))
        return result

    stages = dict(raw=stage(raw, raw_zero=zero), floor=stage(floor),
                  logits=stage(logits), log_probability=stage(logp),
                  probability=stage(probability, logarithmic=False))
    saved_argmax = np.asarray(arrays['argmax'])
    if (saved_argmax.shape != () or saved_argmax.dtype != np.int32
            or int(saved_argmax) != stages['probability']['first']):
        raise ValueError('saved actual argmax differs from first materialized winner')
    g, b = stages['floor']['first'], stages['logits']['first']
    geometry_loss = float(floor[g]) - float(floor[b])
    content_gain = float(bias[b]) - float(bias[g])
    advantage = logits[b] - logits[g]
    entropy_raw, peak_raw = (None, None) if zero else _entropy(raw)
    entropy_floor, peak_floor = _entropy(floor)
    return dict(
        **{key: spec[key] for key in ('id', 'scene', 'pair', 'pair_id', 'query_index')},
        status='ok',
        projection=dict(depth_sign='positive' if az > 0 else 'negative' if az < 0 else 'zero',
                        projection_status=projection_status, xy_ref8=xy, fov_status=fov_status,
                        cell_boundary=boundary, anchor_cells=anchor_cells,
                        primary_anchor_defined=bool(primary), old_anchor_delta_px=delta,
                        old_anchor_distance_px=old_distance),
        depth=dict(a_z=az, sigma_z=sigma, rms_depth=rms, normalized_mean=_ratio(az, rms),
                   normalized_std=_ratio(sigma, rms), relative_std=relative,
                   relative_std_status=relative_status, front_mass=front, tau32=tau,
                   structural_branch=branch, structural_zero=zero),
        distribution=dict(raw_entropy=entropy_raw, raw_peak=peak_raw,
                          floor_entropy=entropy_floor, floor_peak=peak_floor,
                          log_raw_to_floor_total=None if zero else _lse(raw) - _lse(logfloors)),
        arithmetic=dict(floor_error=floor_error, log_probability_error=logp_error,
                        materialization_error=material_error, normalization_error=normalization_error,
                        pair_identity_verified=True,
                        raw_to_floor_winner_changed=None if zero else stages['raw']['first'] != stages['floor']['first'],
                        raw_to_floor_maximizers_changed=None if zero else stages['raw']['maximizers'] != stages['floor']['maximizers'],
                        logits_to_log_probability_winner_changed=stages['logits']['first'] != stages['log_probability']['first'],
                        log_probability_to_probability_winner_changed=stages['log_probability']['first'] != stages['probability']['first'],
                        saved_argmax_verified=True),
        stages=stages,
        content=dict(winner_changed=g != b, geometry_loss=geometry_loss, content_gain=content_gain,
                     logit_advantage=advantage, bookkeeping_residual=advantage - (content_gain - geometry_loss),
                     set_transition=(''.join('1' if stages[name]['anchor_set_contains'] else '0'
                                             for name in ('floor', 'logits'))) if primary else None,
                     first_transition=(''.join('1' if stages[name]['anchor_first_contains'] else '0'
                                               for name in ('floor', 'logits'))) if primary else None))


def _path(row, dotted):
    value = row
    for component in dotted.split('.'):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _counts(values):
    result = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def independent_group(rows):
    successful = [r for r in rows if r['status'] == 'ok']
    categorical_paths = (
        'projection.depth_sign', 'projection.projection_status',
        'projection.fov_status', 'projection.cell_boundary', 'projection.primary_anchor_defined',
        'depth.structural_branch', 'depth.relative_std_status', 'content.winner_changed',
        'content.set_transition', 'content.first_transition')
    event_names = ('raw_to_floor_winner_changed', 'raw_to_floor_maximizers_changed',
                   'logits_to_log_probability_winner_changed',
                   'log_probability_to_probability_winner_changed')
    return dict(
        population=len(rows), successful_rows=len(successful),
        status_counts=_counts(r['status'] for r in rows),
        categories={path: _counts(_path(r, path) for r in successful)
                    for path in categorical_paths},
        numerics={path: _distribution(_path(r, path) for r in successful)
                  for path in STAT_PATHS},
        stage_counts={stage: {name: _counts(r['stages'][stage][name] for r in successful)
                             for name in ('defined', 'anchor_set_contains', 'anchor_first_contains')}
                      for stage in STAGES},
        arithmetic_events={name: _counts(r['arithmetic'][name] for r in successful)
                           for name in event_names})


def independent_summary(rows):
    result = dict(quantile_method='linear_at_(n-1)*q', numerical_paths=list(STAT_PATHS),
                  all=independent_group(rows),
                  by_scene={scene: independent_group([r for r in rows if r['scene'] == scene])
                            for scene in C.SCENES}, **SCOPE)
    for kind in ('set_transition', 'first_transition'):
        result[kind] = {state: independent_group([r for r in rows if _path(r, 'content.' + kind) == state])
                        for state in ('11', '10', '01', '00')}
    return result


def _empty(spec, status, reason):
    return dict(**{key: spec[key] for key in ('id', 'scene', 'pair', 'pair_id', 'query_index')},
                status=status, reason=reason)


def audit(producer_root, audit_root):
    # Importing the reader is allowed; no producer statistic is called below.
    from experiments import round38_saved_output_diagnostic as reader

    producer_root, audit_root = Path(producer_root).resolve(), Path(audit_root).resolve()
    if (audit_root / 'started.json').exists() or (audit_root / 'completion.json').exists():
        raise ValueError('preserve previous independent diagnostic attempt')
    started, tick = time.time(), time.perf_counter()
    deadline = tick + 120.
    before = C.read(audit_root / 'before_execution.json')
    specs = before['specs']
    rows, checks, reads, failure = [], [], [], None
    summary_errors = []
    producer_completion_sha = None
    C.write(audit_root / 'started.json', dict(started_unix=started,
            before_execution_sha256=C.sha(audit_root / 'before_execution.json'), **SCOPE))
    try:
        before = reader.verify_before(audit_root)
        if (before['role'] != 'audit' or before['producer_root'] != str(producer_root)
                or before['audit_root'] != str(audit_root)
                or before['stage_names'] != list(STAGES)
                or before['stat_paths'] != list(STAT_PATHS)):
            raise ValueError('independent audit freeze or statistic schema differs')
        producer_completion = C.read(producer_root / 'completion.json')
        C.verify_inventory(producer_root, producer_completion['files'], exclude=('completion.json',))
        producer_completion_sha = C.sha(producer_root / 'completion.json')
        if (producer_completion.get('passed') is not True
                or producer_completion.get('complete_case_inventory') is not True
                or producer_completion.get('completed_queries') != 2304
                or producer_completion.get('total_queries') != 2304
                or producer_completion.get('phase_wall_passed') is not True
                or producer_completion['before_execution_sha256'] != C.sha(producer_root / 'before_execution.json')
                or any(producer_completion.get(k) != v for k, v in SCOPE.items())):
            raise ValueError('complete passed producer diagnostic required for audit')
        actual_rows = C.read(producer_root / 'rows.json')
        actual_summary = C.read(producer_root / 'summary.json')
        if (len(specs) != 2304 or len(actual_rows) != 2304
                or [row['id'] for row in actual_rows] != [spec['id'] for spec in specs]
                or any(row['status'] != 'ok' for row in actual_rows)):
            raise ValueError('producer ordered complete row population differs')
        with reader.restricted_access(before):
            iterator = reader.iter_payloads(before, reads)
            for index, spec in enumerate(specs):
                try:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('independent diagnostic deadline before row')
                    payload = next(iterator)
                    if payload['spec'] != spec or payload['arrays']['input__bounds'].shape != (432, 4):
                        raise ValueError('independent fixed population or 432 native keys differs')
                    expected = independent_row(payload)
                    differences = compare(expected, actual_rows[index])
                    if differences:
                        raise ValueError('independent row disagreement: ' + '; '.join(differences))
                    rows.append(expected)
                    checks.append(dict(id=spec['id'], status='checked', passed=True, differences=[]))
                except BaseException as error:
                    failure = dict(index=index, id=spec['id'], type=type(error).__name__,
                                   message=str(error), traceback=traceback.format_exc())
                    rows.append(_empty(spec, 'failed', failure['message']))
                    checks.append(dict(id=spec['id'], status='failed', passed=False,
                                       differences=[failure['message']]))
                    break
            if failure is None:
                try:
                    next(iterator)
                    raise ValueError('independent source iterator has extra positions')
                except StopIteration:
                    pass
        if failure is None:
            summary_errors = compare(independent_summary(rows), actual_summary, 'summary')
            if summary_errors:
                raise ValueError('independent summary disagreement: ' + '; '.join(summary_errors))
    except BaseException as error:
        failure = dict(index=len(rows), id=specs[len(rows)]['id'] if len(rows) < len(specs) else None,
                       type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if len(rows) < len(specs):
            spec = specs[len(rows)]
            rows.append(_empty(spec, 'failed', failure['message']))
            checks.append(dict(id=spec['id'], status='failed', passed=False, differences=[failure['message']]))
    while len(rows) < len(specs):
        spec = specs[len(rows)]
        rows.append(_empty(spec, 'unattempted', 'after_first_independent_diagnostic_failure'))
        checks.append(dict(id=spec['id'], status='unattempted', passed=False, differences=[]))
    if [r['id'] for r in rows] != [s['id'] for s in specs]:
        raise ValueError('independent retained row inventory differs')
    C.write(audit_root / 'rows.json', rows)
    C.write(audit_root / 'checks.json', checks)
    C.write(audit_root / 'summary.json', independent_summary(rows))
    C.write(audit_root / 'summary_verification.json', dict(passed=failure is None,
            differences=summary_errors, completed_rows=len(rows), **SCOPE))
    C.write(audit_root / 'reads.json', reads)
    if failure is not None:
        C.write(audit_root / 'failure.json', failure)
    wall = time.perf_counter() - tick
    time_ok = wall <= 120.
    passed = bool(failure is None and time_ok and len(rows) == 2304
                  and len(checks) == 2304 and all(r['passed'] for r in checks))
    inventory_tick = time.perf_counter()
    files = C.inventory(audit_root)
    inventory_wall = time.perf_counter() - inventory_tick
    C.write(audit_root / 'completion.json', dict(passed=passed,
            complete_case_inventory=len(rows) == 2304, completed_queries=sum(c['passed'] for c in checks),
            total_queries=len(rows), started_unix=started, finished_unix=time.time(),
            phase_wall_seconds=wall, phase_wall_passed=time_ok,
            final_inventory_wall_seconds=inventory_wall,
            before_execution_sha256=C.sha(audit_root / 'before_execution.json'),
            producer_completion_sha256=producer_completion_sha,
            replay_relative_tolerance=FLOAT_RTOL, replay_absolute_tolerance=FLOAT_ATOL,
            saved_algebra_absolute_tolerance=ALGEBRA_TOL, failure=failure,
            archives_access_attempted=len(reads),
            archives_opened=sum(bool(r['archive_opened']) for r in reads),
            archives_verified=sum(r['status'] == 'verified' for r in reads),
            fields_decoded=sum(len(r['decoded_fields']) for r in reads), files=files, **SCOPE))
    print('R38_SAVED_DIAGNOSTIC_AUDIT', passed, sum(c['passed'] for c in checks), flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('audit',))
    parser.add_argument('--out', type=Path, default=C.ROOT / 'results/round38_saved_output_diagnostic')
    parser.add_argument('--audit-out', type=Path, default=C.ROOT / 'results/round38_saved_output_diagnostic_audit')
    args = parser.parse_args()
    C.configure_runtime()
    if not audit(args.out, args.audit_out):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
