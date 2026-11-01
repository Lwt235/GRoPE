"""Separate R22 provenance, reference, raw-mass and identity audit.

This process never imports the production mathematical implementation. Its
only numerical inputs are frozen synthetic descriptors and sealed arrays.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import scipy
from scipy.integrate import quad as _runtime_load_quad  # Match loaded runtime libraries, not production math.
from scipy.special import log_ndtr, ndtr
from threadpoolctl import threadpool_info
from experiments import round22_ray_contract as C
from experiments import round22_ray_reference as REF

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/round22_ray_preflight'
DEFAULT_OUT = ROOT / 'results/round22_ray_independent_audit'
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
BINDINGS = ('results/round21_matched_root_acceptance_20261031.json',
            'results/round21_matched_post_result_decision_20261031.json')
APPROXIMATE_TIE_TOLERANCE = 1e-12  # Descriptive only, never a pass/fail threshold.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)), sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header + b'\n' + value.tobytes()).hexdigest()


def inventory(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def verify_inventory(directory):
    complete = read(directory / 'completion.json')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual != set(complete['files']) | {'completion.json'}:
        raise ValueError('inventory population differs: ' + str(directory))
    for name, digest in complete['files'].items():
        if sha(directory / name) != digest:
            raise ValueError('inventory hash differs: ' + name)
    return complete


def source_names():
    found, pending = set(), list(C.SOURCE_FILES)
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
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('relative import requires explicit source review')
                if node.module:
                    modules = [node.module] + [node.module + '.' + n.name for n in node.names]
            for module in modules:
                parts = module.split('.')
                paths = [Path(*parts).with_suffix('.py'), Path(*parts) / '__init__.py']
                paths += [Path(*parts[:i]) / '__init__.py' for i in range(1, len(parts))]
                pending.extend(str(p) for p in paths if (ROOT / p).is_file() and str(p) not in found)
    return sorted(found)


def runtime():
    builds = {}
    for name, function in (('numpy', np.show_config), ('scipy', scipy.show_config)):
        stream = io.StringIO()
        with redirect_stdout(stream):
            function()
        builds[name + '_build_sha256'] = hashlib.sha256(stream.getvalue().encode()).hexdigest()
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(), threads={n: os.environ.get(n) for n in THREAD_ENV}, threadpools=threadpool_info(), **builds)


def check_runtime():
    result = runtime()
    if any(result['threads'][k] != '4' for k in THREAD_ENV) or any(p['num_threads'] != 4 for p in result['threadpools']):
        raise ValueError('frozen four-thread numerical runtime required')
    return result


def verify_producer_lock(run):
    lock = read(run / 'lock.json')
    if sha(run / 'lock.json') != (run / 'lock.sha256').read_text().strip():
        raise ValueError('producer lock changed')
    base, identities = C.fixtures(), C.identity_fixtures()
    expected = dict(version=1, runtime=check_runtime(), policy=C.POLICY, reference_policy=C.REFERENCE_POLICY,
        gates=C.GATES, base_ids=[r['id'] for r in base], identity_ids=[r['id'] for r in identities],
        synthetic_only=True, real_inputs_read=False, labels_read=False, scientific_gate_passed=False)
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError('frozen synthetic contract/runtime differs')
    if set(lock['source_hashes']) != set(source_names()):
        raise ValueError('source closure differs')
    for name, digest in lock['source_hashes'].items():
        if sha(ROOT / name) != digest or sha(run / 'source_snapshot' / name) != digest:
            raise ValueError('current/snapshot source changed: ' + name)
    if sha(run / 'fixtures.json') != lock['fixtures_sha256'] or read(run / 'fixtures.json') != base + identities:
        raise ValueError('fixed fixture descriptor changed')
    if set(lock['bindings']) != {str(ROOT / n) for n in BINDINGS}:
        raise ValueError('unexpected development provenance binding')
    for name, digest in lock['bindings'].items():
        if Path(name).suffix == '.npz' or sha(name) != digest:
            raise ValueError('forbidden/changed external binding')
    receipt = read(ROOT / BINDINGS[0])
    if receipt.get('accepted_for_development_interpretation') is not True or receipt.get('full_forward_and_final_independent_audits_passed') is not True:
        raise ValueError('accepted R21 development receipt required')
    return lock


def freeze(run, out):
    if out.exists() or any((run / n).exists() for n in ('started.json', 'arrays', 'records.json', 'completion.json')):
        raise ValueError('independent freeze must precede every formal case')
    lock = verify_producer_lock(run)
    if Path(lock['audit_root']).resolve() != out.resolve():
        raise ValueError('different declared independent audit directory')
    out.mkdir(parents=True)
    for name, digest in lock['source_hashes'].items():
        target = out / 'source_snapshot' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha(target) != digest:
            raise ValueError('source changed while copying independent snapshot')
    write(out / 'lock.json', dict(created_unix=time.time(), producer_lock_sha256=sha(run / 'lock.json'),
        fixtures_sha256=lock['fixtures_sha256'], source_hashes=lock['source_hashes'], runtime=lock['runtime'],
        run=str(run.resolve()), real_inputs_read=False, labels_read=False, scientific_gate_passed=False))
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')


def verify_joint_lock(run, out):
    lock, design = verify_producer_lock(run), read(out / 'lock.json')
    if sha(out / 'lock.json') != (out / 'lock.sha256').read_text().strip():
        raise ValueError('independent lock changed')
    expected = dict(producer_lock_sha256=sha(run / 'lock.json'), fixtures_sha256=lock['fixtures_sha256'],
                    source_hashes=lock['source_hashes'], runtime=lock['runtime'], run=str(run.resolve()))
    if any(design.get(k) != v for k, v in expected.items()) or design['created_unix'] < lock['created_unix']:
        raise ValueError('independent joint freeze differs')
    for name, digest in lock['source_hashes'].items():
        if sha(out / 'source_snapshot' / name) != digest:
            raise ValueError('independent snapshot changed: ' + name)
    return lock, design


def verify_seal(run, out):
    lock, design = verify_joint_lock(run, out)
    complete, seal = verify_inventory(run), read(run / 'forward_complete.json')
    ids = lock['base_ids'] + lock['identity_ids']
    expected = dict(producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
        fixtures_sha256=lock['fixtures_sha256'], records_sha256=sha(run / 'records.json'), case_ids=ids, case_count=len(ids),
        all_case_attempts_retained=True, all_successful_arrays_retained=True, synthetic_only=True,
        real_inputs_read=False, labels_read=False, scientific_gate_passed=False)
    if any(seal.get(k) != v for k, v in expected.items()):
        raise ValueError('global production seal differs')
    records, fixtures = read(run / 'records.json'), read(run / 'fixtures.json')
    if [r['id'] for r in records] != ids or len(set(ids)) != len(ids):
        raise ValueError('complete ordered production population required')
    started = read(run / 'started.json')
    if (started['producer_lock_sha256'] != sha(run / 'lock.json') or started['audit_lock_sha256'] != sha(out / 'lock.json')
            or not design['created_unix'] <= started['started_unix'] == seal['started_unix'] <= seal['phase_finished_unix'] <= seal['finished_unix']
            or complete['producer_forward_complete_sha256'] != sha(run / 'forward_complete.json')
            or seal['phase_wall_passed'] != (seal['phase_wall_seconds'] <= C.GATES['production_max_phase_wall_seconds'])
            or complete['phase_wall_passed'] != seal['phase_wall_passed']
            or seal['all_production_calls_completed'] != all(r['passed'] is True for r in records)
            or complete['all_production_calls_completed'] != seal['all_production_calls_completed']):
        raise ValueError('production chronology/status differs')
    expected_files = {'fixtures.json', 'lock.json', 'lock.sha256', 'started.json', 'records.json', 'forward_complete.json'}
    expected_files.update('source_snapshot/' + n for n in lock['source_hashes'])
    expected_files.update('arrays/' + name + '.npz' for name in ids)
    if set(complete['files']) != expected_files:
        raise ValueError('unexpected/missing sealed production file')
    arrays = {}
    for fixture, row in zip(fixtures, records):
        path = 'arrays/' + fixture['id'] + '.npz'
        if (row['kind'] != fixture['kind'] or row['identity'] != fixture.get('identity')
                or row['policy_overrides'] != fixture.get('policy_overrides', {}) or row['archive_path'] != path
                or row['archive_sha256'] != sha(run / path)):
            raise ValueError('saved case identity changed')
        with np.load(run / path, allow_pickle=False) as a:
            bank = {name: a[name] for name in a.files}
        if {n: array_sha(v) for n, v in bank.items()} != row['array_sha256']:
            raise ValueError('saved case array hash changed')
        for name in ('mu', 'factor', 'bounds', 'H', 'offset'):
            if (bank['input__' + name].dtype != np.float64
                    or not np.array_equal(bank['input__' + name], np.asarray(fixture[name], np.float64))):
                raise ValueError('input array differs from frozen descriptor')
        expected_names = {'input__' + n for n in ('mu', 'factor', 'bounds', 'H', 'offset')}
        if row['passed'] is True:
            expected_names |= {'raw_mass', 'probability', 'floored_mass', 'quadrature_error_estimate', 'inner_error_estimate_mass'}
            valid_names = set(bank) == expected_names
        else:
            partial = {'partial_raw_mass', 'partial_raw_fov_mass', 'partial_quadrature_error_estimate', 'partial_fov_quadrature_error_estimate'}
            valid_names = expected_names <= set(bank) <= expected_names | partial
        if not valid_names:
            raise ValueError('unexpected/missing successful case array')
        arrays[fixture['id']] = bank
    return lock, seal, records, fixtures, arrays


def probabilities(raw, bounds, floor_density):
    raw, bounds = np.asarray(raw), np.asarray(bounds)
    if (raw.dtype != np.float64 or raw.ndim != 1 or raw.shape != (len(bounds),)
            or not np.isfinite(raw).all() or np.any(raw < 0)):
        raise ValueError('finite nonnegative float64 raw vector required')
    areas = np.prod(bounds[:, 2:] - bounds[:, :2], axis=1)
    if np.any(areas <= 0) or not np.isfinite(floor_density) or floor_density <= 0:
        raise ValueError('positive finite native floor required')
    floor = areas * floor_density; floored = raw + floor
    return floored / math.fsum(map(float, floored)), floor, floored


def compare_raw(actual, reference, bounds, floor_density, gates, prefix, actual_probability=None):
    """Raw-tail ratios are computed before the floor, including tiny FOVs."""
    pa, floor, _ = probabilities(actual, bounds, floor_density)
    pr, _, _ = probabilities(reference, bounds, floor_density)
    if actual_probability is not None:
        pa = np.asarray(actual_probability)
        if pa.dtype != np.float64 or pa.shape != actual.shape or not np.isfinite(pa).all() or np.any(pa < 0):
            raise ValueError('valid archived probability required')
    difference = abs(actual - reference); total = math.fsum(map(float, reference))
    raw_l1 = math.fsum(map(float, difference)); actual_sum = math.fsum(map(float, actual))
    zero = total == 0.; exact_zero = bool(np.count_nonzero(actual) == 0) if zero else None
    fov_relative = abs(actual_sum - total) / total if total else None
    l1_relative = raw_l1 / total if total else None
    per_cell_relative = [float(d / r) if r > 0 else (0. if d == 0 else None) for d, r in zip(difference, reference)]
    p_error = float(np.max(abs(pa - pr)))
    passed = bool(p_error <= gates[prefix + '_probability_absolute'] and
        (exact_zero if zero else fov_relative <= gates[prefix + '_raw_fov_relative']
         and l1_relative <= gates[prefix + '_raw_l1_over_reference_fov']))
    actual_ties = np.flatnonzero(pa == pa.max()).tolist(); reference_ties = np.flatnonzero(pr == pr.max()).tolist()
    return dict(passed=passed, raw_fov_actual=actual_sum, raw_fov_reference=total, raw_fov_relative=fov_relative,
        raw_l1=raw_l1, raw_l1_over_reference_fov=l1_relative, per_cell_absolute_error=difference.tolist(),
        per_cell_relative_error=per_cell_relative, zero_reference_cells=int(np.count_nonzero(reference == 0)),
        zero_reference_nonzero_actual_cells=int(np.count_nonzero((reference == 0) & (actual != 0))),
        reference_exact_zero=zero, exact_zero_actual=exact_zero, maximum_probability_error=p_error,
        actual_argmax=int(pa.argmax()), reference_argmax=int(pr.argmax()), argmax_changed=bool(pa.argmax() != pr.argmax()),
        actual_exact_maximizers=actual_ties, reference_exact_maximizers=reference_ties,
        actual_approximate_maximizers=np.flatnonzero(abs(pa - pa.max()) <= APPROXIMATE_TIE_TOLERANCE).tolist(),
        reference_approximate_maximizers=np.flatnonzero(abs(pr - pr.max()) <= APPROXIMATE_TIE_TOLERANCE).tolist(),
        approximate_tie_absolute_tolerance=APPROXIMATE_TIE_TOLERANCE, total_floor=math.fsum(map(float, floor)),
        actual_raw_below_floor=bool(actual_sum < floor.sum()), reference_raw_below_floor=bool(total < floor.sum()))


def production_check(fixture, row, bank, gates, policy):
    if row['passed'] is not True or row['status'] != 'complete':
        return dict(passed=False, status='production_failed', failure_reason=row.get('failure_reason'))
    raw, p, bounds = bank['raw_mass'], bank['probability'], bank['input__bounds']
    expected, floor, floored = probabilities(raw, bounds, policy['floor_density'])
    if p.dtype != np.float64 or p.shape != raw.shape or not np.isfinite(p).all() or np.any(p < 0):
        return dict(passed=False, status='invalid_probability')
    norm = abs(math.fsum(map(float, p)) - 1.); arithmetic = float(np.max(abs(expected - p)))
    floor_error = float(np.max(abs(floored - bank['floored_mass'])))
    total_floor, raw_sum = math.fsum(map(float, floor)), math.fsum(map(float, raw))
    direct = float(row['raw_fov_mass']); direct_error = abs(direct - raw_sum)
    sd = float(np.linalg.norm(bank['input__factor'][2])); tail = float(2 * ndtr(-policy['depth_extent'])) if sd > 0 else 0.
    amplification = 2 * tail / total_floor
    estimate = bank['quadrature_error_estimate']
    work = row['work']
    mu = bank['input__mu']
    exact_zero = bool(sd == 0. and mu[2] <= 0.)
    expected_front = REF.front_mass(mu, bank['input__factor'])
    expected_log_front = float(log_ndtr(mu[2] / sd)) if sd else (0. if mu[2] > 0 else None)
    positive_consistent = bool(np.isclose(row['positive_mass'], expected_front, atol=0., rtol=gates['normalization_absolute'])
        and (row['log_positive_mass'] is None if expected_log_front is None else
             row['log_positive_mass'] is not None and np.isclose(row['log_positive_mass'], expected_log_front, atol=0., rtol=gates['normalization_absolute']))
        and row['exact_zero_positive_mass'] is exact_zero
        and row['positive_mass_underflow'] is bool(expected_front == 0. and not exact_zero))
    inner_estimate = bank['inner_error_estimate_mass']
    outer_estimate = row['outer_error_estimate']
    metadata = bool(np.isclose(row['raw_mass_sum'], raw_sum, atol=0., rtol=1e-13)
        and np.isclose(row['total_area_floor'], total_floor, atol=0., rtol=1e-13)
        and np.isclose(row['normalizer'], math.fsum(map(float, floored)), atol=0., rtol=1e-13)
        and row['depth_tail_bound'] == tail and row['certified_tail_amplification'] == 2 * tail / float(row['total_area_floor'])
        and row['quadrature_error_is_estimate'] is True and row['tail_bound_excludes_quadrature_and_roundoff'] is True
        and estimate.dtype == np.float64 and estimate.shape == raw.shape and np.isfinite(estimate).all() and np.all(estimate >= 0)
        and inner_estimate.dtype == np.float64 and inner_estimate.shape == raw.shape and np.isfinite(inner_estimate).all()
        and np.all(inner_estimate >= 0) and np.isfinite(outer_estimate) and outer_estimate >= 0
        and np.array_equal(estimate, inner_estimate + outer_estimate)
        and row['fov_quadrature_error_estimate'] == row['fov_inner_error_estimate'] + outer_estimate
        and row['zero_raw_cells'] == int(np.count_nonzero(raw == 0))
        and row['area_prior_used'] is bool(np.all(raw == 0))
        and row['raw_mass_below_total_floor'] is bool(np.sum(raw, dtype=np.float64) < row['total_area_floor'])
        and row['argmax_index'] == int(p.argmax())
        and row['argmax_exact_ties'] == int(np.count_nonzero(p == p.max()))
        and row['argmax_near_ties_at_1e_12'] == int(np.count_nonzero(p.max() - p <= APPROXIMATE_TIE_TOLERANCE))
        and positive_consistent)
    budget = bool(work['depth_evaluations'] <= policy['max_depth_evaluations']
        and work['inner_evaluations'] <= policy['max_inner_evaluations']
        and work['outer_intervals'] <= policy['outer_limit']
        and len(row['depth_breakpoints']) <= policy['max_breakpoints'])
    passed = bool(norm <= gates['normalization_absolute'] and arithmetic <= gates['normalization_absolute']
        and floor_error == 0. and direct_error <= gates['direct_fov_additivity_absolute']
        and amplification <= gates['amplified_analytic_tail'] and metadata and budget)
    return dict(passed=passed, status='checked', normalization_error=norm, same_raw_probability_error=arithmetic,
        floor_array_absolute_error=floor_error, raw_cell_sum=raw_sum, direct_fov_mass=direct,
        direct_fov_additivity_error=direct_error, total_floor=total_floor, depth_tail_bound=tail,
        certified_tail_amplification=amplification, metadata_consistent=metadata, work_budget_passed=budget,
        positive_mass_consistent=positive_consistent, independently_computed_positive_mass=expected_front,
        independently_computed_log_positive_mass=expected_log_front,
        zero_raw_cells=int(np.count_nonzero(raw == 0)), zero_raw_fov=raw_sum == 0., raw_fov_below_floor=raw_sum < total_floor,
        exact_maximizers=np.flatnonzero(p == p.max()).tolist(),
        approximate_maximizers=np.flatnonzero(abs(p - p.max()) <= APPROXIMATE_TIE_TOLERANCE).tolist(),
        approximate_tie_absolute_tolerance=APPROXIMATE_TIE_TOLERANCE)


def identity_check(fixture, actual, parent, gates):
    relation = fixture['identity']['relation']; identity = fixture['identity']
    expected = {name: parent[name] for name in ('raw_mass', 'probability')}
    if relation == 'permutation':
        expected = {name: value[identity['permutation']] for name, value in expected.items()}
    elif relation == 'merge':
        expected = {name: np.asarray([math.fsum(map(float, value[group])) for group in identity['groups']]) for name, value in expected.items()}
    elif relation not in ('equal', 'point_limit'):
        raise ValueError('unfrozen identity relation')
    errors = {name + '_absolute_error': float(np.max(abs(actual[name] - value))) for name, value in expected.items()}
    tolerance = gates['fixed_point_limit_absolute'] if relation == 'point_limit' else gates['integrated_identity_absolute']
    return dict(id=fixture['id'], parent=identity['parent'], relation=relation,
        passed=all(v <= tolerance for v in errors.values()), tolerance=tolerance, **errors,
        finite_point_limit_is_independent_reference=False)


def audit(run, out):
    if any((out / name).exists() for name in ('audit_started.json', 'summary.json', 'completion.json')):
        raise ValueError('preserve every previous formal audit attempt')
    started_unix, phase = time.time(), time.perf_counter()
    write(out / 'audit_started.json', dict(started_unix=started_unix))
    try:
        lock, seal, records, fixtures, arrays = verify_seal(run, out)
        if started_unix < seal['finished_unix']:
            raise ValueError('independent reference phase must follow the global production seal')
        row_map = {r['id']: r for r in records}; fixture_map = {r['id']: r for r in fixtures}
        production, reference_records, comparisons, identities = [], [], [], []
        reference_budget = lock['reference_policy']; cumulative = 0; reference_failures = 0
        reference_directory = out / 'reference_arrays'; reference_directory.mkdir()
        for fixture in fixtures:
            policy = dict(lock['policy'], **fixture.get('policy_overrides', {}))
            row, bank = row_map[fixture['id']], arrays[fixture['id']]
            try:
                check = production_check(fixture, row, bank, lock['gates'], policy)
            except (ValueError, KeyError, FloatingPointError, OverflowError) as error:
                check = dict(passed=False, status='production_array_failure', failure_reason=str(error))
            production.append(dict(id=fixture['id'], **check))
            write(out / 'production_checks.json', production)
        production_map = {r['id']: r for r in production}
        for name in lock['base_ids']:
            fixture, bank = fixture_map[name], arrays[name]
            evidence, references = {}, {}
            for order in reference_budget['orders']:
                try:
                    if time.perf_counter() - phase >= reference_budget['max_phase_wall_seconds'] or cumulative >= reference_budget['max_total_cell_node_evaluations']:
                        raise RuntimeError('fixed total reference budget exhausted; no retry')
                    result = REF.integrate(**{n: bank['input__' + n] for n in ('mu', 'factor', 'bounds', 'H', 'offset')},
                        policy=lock['policy'], kind=fixture['kind'], order=order, reference_policy=reference_budget)
                    raw = result.pop('raw_mass'); probability, floor, _ = probabilities(raw, bank['input__bounds'], lock['policy']['floor_density'])
                    cumulative += result['work']['cell_node_evaluations']
                    elapsed = time.perf_counter() - phase
                    amplification = 2 * result['analytic_tail']['raw_fov_upper'] / math.fsum(map(float, floor))
                    norm = abs(math.fsum(map(float, probability)) - 1.)
                    passed = bool(cumulative <= reference_budget['max_total_cell_node_evaluations'] and elapsed <= reference_budget['max_phase_wall_seconds']
                        and amplification <= lock['gates']['amplified_analytic_tail'] and norm <= lock['gates']['normalization_absolute'])
                    reference_records.append(dict(id=name, passed=passed, **result, normalization_error=norm,
                        certified_tail_amplification=amplification, cumulative_cell_node_evaluations=cumulative, phase_elapsed_seconds=elapsed,
                        work_scope='native cells only; FOV mass is their sum; no extra direct-FOV quadrature'))
                    evidence['raw_mass' + str(order)] = raw; evidence['probability' + str(order)] = probability
                    references[order] = (raw, passed)
                except Exception as error:
                    reference_failures += 1
                    reference_records.append(dict(id=name, order=order, passed=False, failure_reason=type(error).__name__ + ': ' + str(error),
                        cumulative_cell_node_evaluations=cumulative, phase_elapsed_seconds=time.perf_counter() - phase))
                write(out / 'reference_records.json', reference_records)
                # Preserve every available order even if a later order/check
                # encounters an engineering error. No order is rerun.
                path = reference_directory / (name + '.npz'); np.savez_compressed(path, **evidence)
            path = reference_directory / (name + '.npz'); np.savez_compressed(path, **evidence)
            check = dict(id=name, archive_path=str(path.relative_to(out)), archive_sha256=sha(path),
                array_sha256={n: array_sha(v) for n, v in evidence.items()}, passed=False)
            if set(references) == set(reference_budget['orders']):
                low, high = reference_budget['orders']
                check['reference'] = compare_raw(references[low][0], references[high][0], bank['input__bounds'], lock['policy']['floor_density'], lock['gates'], 'reference')
                if production_map[name]['passed']:
                    check['production'] = compare_raw(bank['raw_mass'], references[high][0], bank['input__bounds'],
                        lock['policy']['floor_density'], lock['gates'], 'production', actual_probability=bank['probability'])
                check['passed'] = bool(all(value[1] for value in references.values()) and check['reference']['passed']
                    and check.get('production', {}).get('passed', False) and production_map[name]['passed'])
            comparisons.append(check)
            write(out / 'comparisons.json', comparisons)
            print('INDEPENDENT_REFERENCE', name, check['passed'], flush=True)
        for name in lock['identity_ids']:
            fixture = fixture_map[name]; parent = fixture['identity']['parent']
            if production_map[name]['passed'] and production_map[parent]['passed']:
                identities.append(identity_check(fixture, arrays[name], arrays[parent], lock['gates']))
            else:
                identities.append(dict(id=name, parent=parent, passed=False, failure_reason='production identity/base call failed'))
            write(out / 'identities.json', identities)
        verify_joint_lock(run, out)
        elapsed = time.perf_counter() - phase
        phase_finished_unix = time.time()
        budget_passed = bool(elapsed <= reference_budget['max_phase_wall_seconds'] and cumulative <= reference_budget['max_total_cell_node_evaluations'])
        passed = bool(seal['phase_wall_passed'] and seal['all_production_calls_completed'] and budget_passed
            and all(r['passed'] for r in production + reference_records + comparisons + identities))
        summary = dict(numerical_preflight_passed=passed, scientific_gate_passed=False, synthetic_only=True,
            real_inputs_read=False, labels_read=False, candidate_selected=False, base_case_count=len(lock['base_ids']),
            identity_count=len(lock['identity_ids']), reference_call_count=len(reference_records),
            independent_reference_orders=reference_budget['orders'], reference_phase_wall_seconds=elapsed,
            reference_cell_node_evaluations=cumulative, reference_budget_passed=budget_passed,
            production_phase_wall_seconds=seal['phase_wall_seconds'], production_phase_wall_passed=seal['phase_wall_passed'],
            production_failure_count=sum(not r['passed'] for r in production),
            reference_call_failure_count=sum(not r['passed'] for r in reference_records), reference_exception_count=reference_failures,
            base_comparison_failure_count=sum(not r['passed'] for r in comparisons), identity_failure_count=sum(not r['passed'] for r in identities),
            next_action=('fixed_synthetic_numerics_only_passed_separate_development_protocol_required' if passed else
                         'stop_fixed_implementation_preserve_all_failures_no_real_data'),
            started_unix=started_unix, phase_finished_unix=phase_finished_unix, finished_unix=time.time(),
            note='Canonical finite cases only. Convergence is not a rigorous total error bound; no scientific or uniform-domain pass.')
        for filename, value in (('production_checks.json', production), ('reference_records.json', reference_records),
                                ('comparisons.json', comparisons), ('identities.json', identities), ('summary.json', summary)):
            write(out / filename, value)
        verification = dict(delivery_integrity_passed=True, numerical_preflight_passed=passed, scientific_gate_passed=False,
            producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
            producer_completion_sha256=sha(run / 'completion.json'), producer_forward_complete_sha256=sha(run / 'forward_complete.json'),
            fixtures_sha256=lock['fixtures_sha256'], summary_sha256=sha(out / 'summary.json'),
            producer_case_count=len(records), independent_reference_case_count=len(comparisons), identity_count=len(identities),
            source_count=len(lock['source_hashes']), started_unix=started_unix, finished_unix=time.time())
        write(out / 'verification.json', verification)
        files = inventory(out)
        write(out / 'completion.json', dict(files=files, delivery_integrity_passed=True,
            numerical_preflight_passed=passed, scientific_gate_passed=False, finished_unix=time.time(),
            post_phase_bookkeeping_seconds_before_completion_write=time.perf_counter() - phase - elapsed,
            total_seconds_before_completion_write=time.perf_counter() - phase))
    except Exception as error:
        write(out / 'failure.json', dict(error_type=type(error).__name__, message=str(error), started_unix=started_unix, finished_unix=time.time()))
        write(out / 'completion.json', dict(files=inventory(out), delivery_integrity_passed=False,
            numerical_preflight_passed=False, scientific_gate_passed=False, finished_unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'audit'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN); parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    {'freeze': freeze, 'audit': audit}[args.command](args.run.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
