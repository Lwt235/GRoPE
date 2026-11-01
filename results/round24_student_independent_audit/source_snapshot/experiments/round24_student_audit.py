"""Independent R24 provenance, conditional-t5 reference and saved-array audit.

No production mathematical implementation is imported or called. Metadata
hash/runtime helpers follow the existing audited pattern; the numerical
reference and floor/native-bias arithmetic are separate implementations.
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
from scipy.special import logsumexp
from threadpoolctl import threadpool_info
from experiments import round24_student_contract as C
from experiments import round24_student_reference as REF
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/round24_student_preflight'
DEFAULT_OUT = ROOT / 'results/round24_student_independent_audit'
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
APPROXIMATE_TIE_TOLERANCE = 1e-12

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


def inherited_bindings():
    result = {str(ROOT / n): sha(ROOT / n) for n in C.BINDINGS}
    receipt = read(ROOT / C.BINDINGS[0])
    if (receipt.get('accepted_for_applicability_interpretation') is not True
            or receipt.get('applicability_calls_completed') is not False or receipt.get('scientific_gate_passed') is not False):
        raise ValueError('accepted R23 applicability failure required')
    for name, digest in receipt['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('accepted predecessor evidence changed')
        result[str(ROOT / name)] = digest
    decision = read(ROOT / C.BINDINGS[1])
    if (decision['root_acceptance_sha256'] != sha(ROOT / C.BINDINGS[0])
            or decision['next_selected_work']['round'] != 'R24'
            or decision['next_selected_work']['real_input_or_labels_authorized'] is not False):
        raise ValueError('synthetic-only decision provenance differs')
    return result


def verify_producer_lock(run):
    lock = read(run / 'lock.json'); fixtures = C.fixtures()
    if sha(run / 'lock.json') != (run / 'lock.sha256').read_text().strip():
        raise ValueError('producer lock changed')
    expected = dict(version=1, runtime=check_runtime(), policy=C.POLICY, reference_policy=C.REFERENCE_POLICY, gates=C.GATES,
        base_ids=[r['id'] for r in fixtures if r['kind'] == 'base'], identity_ids=[r['id'] for r in fixtures if r['kind'] == 'identity'],
        synthetic_only=True, real_inputs_read=False, labels_read=False, scientific_gate_passed=False, goal_complete=False)
    if any(lock.get(k) != value for k, value in expected.items()) or len(expected['base_ids']) != 36 or len(expected['identity_ids']) != 6:
        raise ValueError('fixed synthetic population/policy/runtime changed')
    if set(lock['source_hashes']) != set(source_names()):
        raise ValueError('joint transitive source closure differs')
    for name, digest in lock['source_hashes'].items():
        if sha(ROOT / name) != digest or sha(run / 'source_snapshot' / name) != digest:
            raise ValueError('current/frozen source changed: ' + name)
    if (sha(run / 'fixtures.json') != lock['fixtures_sha256'] or read(run / 'fixtures.json') != fixtures
            or lock['bindings'] != inherited_bindings()):
        raise ValueError('fixed descriptors/provenance changed')
    return lock


def freeze(run, out):
    if out.exists() or any((run / name).exists() for name in ('started.json', 'records.json', 'arrays', 'completion.json')):
        raise ValueError('independent freeze must precede every formal case')
    lock = verify_producer_lock(run)
    if Path(lock['audit_root']).resolve() != out.resolve():
        raise ValueError('different declared audit directory')
    out.mkdir(parents=True)
    for name, digest in lock['source_hashes'].items():
        target = out / 'source_snapshot' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha(target) != digest:
            raise ValueError('source changed while freezing independent snapshot')
    write(out / 'lock.json', dict(created_unix=time.time(), run=str(run.resolve()), producer_lock_sha256=sha(run / 'lock.json'),
        fixtures_sha256=lock['fixtures_sha256'], source_hashes=lock['source_hashes'], runtime=lock['runtime'],
        real_inputs_read=False, labels_read=False, scientific_gate_passed=False, goal_complete=False))
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')


def verify_joint_lock(run, out):
    lock, design = verify_producer_lock(run), read(out / 'lock.json')
    expected = dict(run=str(run.resolve()), producer_lock_sha256=sha(run / 'lock.json'), fixtures_sha256=lock['fixtures_sha256'],
                    source_hashes=lock['source_hashes'], runtime=lock['runtime'])
    if (sha(out / 'lock.json') != (out / 'lock.sha256').read_text().strip() or any(design.get(k) != v for k, v in expected.items())
            or not lock['created_unix'] <= design['created_unix'] <= time.time()):
        raise ValueError('independent joint lock differs')
    for name, digest in lock['source_hashes'].items():
        if sha(out / 'source_snapshot' / name) != digest:
            raise ValueError('independent snapshot changed')
    return lock, design


def verify_seal(run, out):
    lock, design = verify_joint_lock(run, out)
    complete, seal = verify_inventory(run), read(run / 'forward_complete.json')
    records, fixtures = read(run / 'records.json'), read(run / 'fixtures.json')
    ids = lock['base_ids'] + lock['identity_ids']
    expected = dict(producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
        fixtures_sha256=lock['fixtures_sha256'], records_sha256=sha(run / 'records.json'), case_ids=ids, case_count=42,
        all_case_attempts_retained=True, all_successful_arrays_retained=True, synthetic_only=True,
        real_inputs_read=False, labels_read=False, scientific_gate_passed=False, goal_complete=False)
    if any(seal.get(k) != v for k, v in expected.items()) or [r['id'] for r in records] != ids or len(set(ids)) != 42:
        raise ValueError('complete ordered 42-case seal required')
    started = read(run / 'started.json')
    if (started['producer_lock_sha256'] != sha(run / 'lock.json') or started['audit_lock_sha256'] != sha(out / 'lock.json')
            or not design['created_unix'] <= started['started_unix'] == seal['started_unix'] <= seal['phase_finished_unix'] <= seal['finished_unix']
            or complete['producer_forward_complete_sha256'] != sha(run / 'forward_complete.json')
            or complete['complete_case_inventory'] is not True
            or seal['phase_wall_passed'] != (seal['phase_wall_seconds'] <= lock['gates']['production_max_phase_wall_seconds'])
            or complete['phase_wall_passed'] != seal['phase_wall_passed']
            or seal['all_production_calls_completed'] != all(r['passed'] is True for r in records)
            or complete['all_production_calls_completed'] != seal['all_production_calls_completed']):
        raise ValueError('case chronology/status differs')
    expected_files = {'lock.json', 'lock.sha256', 'fixtures.json', 'started.json', 'records.json', 'forward_complete.json'}
    expected_files |= {'source_snapshot/' + name for name in lock['source_hashes']} | {'arrays/' + name + '.npz' for name in ids}
    if set(complete['files']) != expected_files:
        raise ValueError('unsealed extra/missing production files')
    banks = {}
    for fixture, row in zip(fixtures, records):
        path = 'arrays/' + fixture['id'] + '.npz'
        if (row['archive_path'] != path or row['archive_sha256'] != sha(run / path) or row['kind'] != fixture['kind']
                or row['identity'] != fixture.get('identity') or row['policy_overrides'] != fixture['policy_overrides']
                or row['content_temperature'] != fixture['content_temperature']):
            raise ValueError('saved fixture identity changed')
        with np.load(run / path, allow_pickle=False) as archive:
            bank = {name: archive[name] for name in archive.files}
        if {name: array_sha(value) for name, value in bank.items()} != row['array_sha256']:
            raise ValueError('saved array canonical identity changed')
        for name in ('mu', 'covariance', 'bounds', 'content_bias'):
            if bank['input__' + name].dtype != np.float64 or not np.array_equal(bank['input__' + name], np.asarray(fixture[name], np.float64)):
                raise ValueError('input array differs from frozen descriptor')
        input_fields = {'input__' + name for name in ('mu', 'covariance', 'bounds', 'content_bias')}
        success = {'raw_mass', 'probability', 'floored_mass'}
        partial = {'partial_raw_mass', 'partial_raw_fov_mass'}
        if type(row.get('passed')) is not bool or row['status'] != ('complete' if row['passed'] else 'failed'):
            raise ValueError('explicit coherent status required')
        if row['passed']:
            if set(bank) != input_fields | success:
                raise ValueError('successful-array schema differs')
        elif not input_fields <= set(bank) <= input_fields | partial or not row.get('failure_reason'):
            raise ValueError('failed case must retain reason and allowed partial arrays only')
        for name in set(bank) - input_fields:
            shape = (1,) if name == 'partial_raw_fov_mass' else (len(fixture['bounds']),)
            if bank[name].dtype != np.float64 or bank[name].shape != shape:
                raise ValueError('output dtype/shape differs: ' + name)
        if (not math.isfinite(row['wall_seconds']) or row['wall_seconds'] < 0 or type(row['peak_rss_kib_process']) is not int
                or row['peak_rss_kib_process'] < 0 or any(type(v) is not int or v < 0 for v in row['work'].values())):
            raise ValueError('retained work/time/RSS metadata invalid')
        banks[fixture['id']] = bank
    return lock, seal, records, fixtures, banks


def probabilities(raw, bounds, bias, floor_density):
    raw, bounds, bias = map(np.asarray, (raw, bounds, bias))
    if (raw.dtype != np.float64 or raw.shape != (len(bounds),) or bias.dtype != np.float64 or bias.shape != raw.shape
            or not np.isfinite(raw).all() or np.any(raw < 0) or not np.isfinite(bias).all()):
        raise ValueError('finite nonnegative raw and native bias required')
    floor = np.prod(bounds[:, 2:] - bounds[:, :2], axis=1) * floor_density
    if np.any(floor <= 0) or not np.isfinite(floor).all():
        raise ValueError('positive native area floor required')
    floored = raw + floor; logits = np.log(floored) + bias
    normalizer = float(logsumexp(logits))
    return np.exp(logits - normalizer), floor, floored, normalizer


def compare_raw(actual, reference, bounds, bias, floor_density, gates, reference_comparison=False, archived_probability=None):
    pa, _, _, _ = probabilities(actual, bounds, bias, floor_density)
    pr, _, _, _ = probabilities(reference, bounds, bias, floor_density)
    if archived_probability is not None:
        pa = np.asarray(archived_probability)
        if pa.dtype != np.float64 or pa.shape != actual.shape or not np.isfinite(pa).all() or np.any(pa < 0):
            raise ValueError('finite archived probability required')
    total = math.fsum(map(float, reference)); difference = abs(actual - reference)
    if not math.isfinite(total) or total <= 0:
        raise ValueError('zero/nonfinite mathematical-positive reference is unresolved, never skipped')
    fov_error = abs(math.fsum(map(float, actual)) - total) / total
    l1 = math.fsum(map(float, difference)) / total; p_error = float(np.max(abs(pa - pr)))
    relative_gate = gates['reference_raw_relative'] if reference_comparison else gates['raw_fov_relative']
    l1_gate = gates['reference_raw_relative'] if reference_comparison else gates['raw_l1_over_reference_fov']
    p_gate = gates['reference_probability_max_absolute'] if reference_comparison else gates['probability_max_absolute']
    return dict(passed=bool(fov_error <= relative_gate and l1 <= l1_gate and p_error <= p_gate),
        raw_fov_actual=math.fsum(map(float, actual)), raw_fov_reference=total, raw_fov_relative=fov_error,
        raw_l1_over_reference_fov=l1, maximum_probability_error=p_error,
        per_cell_absolute_error=difference.tolist(), per_cell_relative_error=[float(d / r) if r else None for d, r in zip(difference, reference)],
        zero_reference_cells=int(np.count_nonzero(reference == 0)), actual_argmax=int(pa.argmax()), reference_argmax=int(pr.argmax()),
        argmax_changed=bool(pa.argmax() != pr.argmax()), actual_exact_maximizers=np.flatnonzero(pa == pa.max()).tolist(),
        reference_exact_maximizers=np.flatnonzero(pr == pr.max()).tolist(),
        actual_approximate_maximizers=np.flatnonzero(abs(pa - pa.max()) <= APPROXIMATE_TIE_TOLERANCE).tolist(),
        reference_approximate_maximizers=np.flatnonzero(abs(pr - pr.max()) <= APPROXIMATE_TIE_TOLERANCE).tolist(),
        approximate_tie_absolute_tolerance=APPROXIMATE_TIE_TOLERANCE)


def production_check(row, bank, policy, gates):
    if row['passed'] is not True:
        return dict(passed=False, status='production_failed', failure_reason=row.get('failure_reason'))
    mean, covariance, bounds, bias = [bank['input__' + n] for n in ('mu', 'covariance', 'bounds', 'content_bias')]
    _, _, _, fov, eigen = REF.validate(mean, covariance, bounds, bias, policy)
    raw, probability = bank['raw_mass'], bank['probability']
    expected, floor, floored, content_normalizer = probabilities(raw, bounds, bias, policy['floor_density'])
    if probability.dtype != np.float64 or probability.shape != raw.shape or not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError('invalid saved probability')
    normalization = abs(math.fsum(map(float, probability)) - 1.)
    arithmetic = float(np.max(abs(expected - probability)))
    max_side = policy['h'] * policy['maximum_tile_side_in_h']
    native_tiles = sum(int(np.prod(np.ceil((box[2:] - box[:2]) / max_side).astype(np.int64))) for box in bounds)
    fov_tiles = int(np.prod(np.ceil((fov[2:] - fov[:2]) / max_side).astype(np.int64)))
    work_expected = dict(cell_node_evaluations=(native_tiles + fov_tiles) * policy['quadrature_order'] ** 2,
                         native_tiles_completed=native_tiles, fov_tiles_completed=fov_tiles)
    raw_sum, floor_sum = math.fsum(map(float, raw)), math.fsum(map(float, floor))
    if not math.isfinite(row['raw_fov_mass']) or row['raw_fov_mass'] < 0:
        raise ValueError('direct FOV mass must be finite and nonnegative')
    direct_error = abs(row['raw_fov_mass'] - raw_sum)
    metadata = bool(row['degrees_of_freedom'] == 4 and row['quadrature_order'] == 24
        and row['quadrature_error_bound_available'] is False and row['tail_truncation_used'] is False
        and row['native_tile_count'] == native_tiles and row['fov_tile_count'] == fov_tiles and row['work'] == work_expected
        and row['h'] == policy['h'] and row['floor_density'] == policy['floor_density']
        and row['admitted_covariance_min_eigenvalue'] == float(eigen[0])
        and np.isclose(row['raw_mass_sum'], raw_sum, rtol=1e-13, atol=0)
        and np.isclose(row['total_area_floor'], floor_sum, rtol=1e-13, atol=0)
        and np.isclose(row['normalizer'], math.fsum(map(float, floored)), rtol=1e-13, atol=0)
        and np.isclose(row['log_content_normalizer'], content_normalizer, rtol=0, atol=gates['normalization_absolute'])
        and row['zero_raw_cells'] == int(np.count_nonzero(raw == 0))
        and row['all_raw_mass_zero'] is bool(np.all(raw == 0))
        and row['area_prior_used'] is bool(np.all(raw == 0) and np.all(bias == bias[0]))
        and row['raw_mass_below_total_floor'] is bool(np.sum(raw) < np.sum(floor))
        and row['argmax_index'] == int(probability.argmax())
        and row['argmax_exact_ties'] == int(np.count_nonzero(probability == probability.max()))
        and row['argmax_near_ties_at_1e_12'] == int(np.count_nonzero(probability.max() - probability <= APPROXIMATE_TIE_TOLERANCE)))
    passed = bool(metadata and normalization <= gates['normalization_absolute'] and arithmetic <= gates['normalization_absolute']
        and np.array_equal(floored, bank['floored_mass']) and direct_error <= gates['direct_fov_additivity_absolute']
        and work_expected['cell_node_evaluations'] <= policy['max_cell_node_evaluations'])
    return dict(passed=passed, status='checked', normalization_error=normalization, same_raw_probability_error=arithmetic,
        metadata_consistent=metadata, raw_fov_mass=raw_sum, direct_fov_mass=row['raw_fov_mass'], direct_fov_additivity_error=direct_error,
        raw_below_total_floor=bool(raw_sum < floor_sum), all_raw_mass_zero=bool(np.all(raw == 0)),
        total_floor=floor_sum, exact_maximizers=np.flatnonzero(probability == probability.max()).tolist(), work_expected=work_expected)


def identity_check(fixture, actual, parent, gates):
    declaration = fixture['identity']; expected = {n: parent[n] for n in ('raw_mass', 'probability')}
    if declaration['relation'] == 'permutation':
        expected = {n: v[declaration['permutation']] for n, v in expected.items()}
    elif declaration['relation'] == 'merge':
        expected = {n: np.asarray([math.fsum(map(float, v[g])) for g in declaration['groups']]) for n, v in expected.items()}
    elif declaration['relation'] != 'equal':
        raise ValueError('unfrozen identity relation')
    errors = {n + '_absolute_error': float(np.max(abs(actual[n] - value))) for n, value in expected.items()}
    return dict(id=fixture['id'], parent=declaration['parent'], relation=declaration['relation'],
                passed=all(v <= gates['integrated_identity_absolute'] for v in errors.values()), **errors)


def reference_work(bounds, h, order, policy):
    """Precharge native x panels and the separately integrated enclosing FOV."""
    bounds = np.asarray(bounds); side = h * policy['maximum_panel_side_in_h']
    widths = bounds[:, 2] - bounds[:, 0]
    enclosing_width = float(bounds[:, 2].max() - bounds[:, 0].min())
    return (sum(int(np.ceil(w / side)) for w in widths) + int(np.ceil(enclosing_width / side))) * order


def audit(run, out):
    if any((out / n).exists() for n in ('audit_started.json', 'completion.json', 'summary.json')):
        raise ValueError('preserve every prior formal audit attempt')
    started, phase = time.time(), time.perf_counter(); write(out / 'audit_started.json', dict(started_unix=started))
    try:
        lock, seal, records, fixtures, banks = verify_seal(run, out)
        if started < seal['finished_unix']:
            raise ValueError('independent references must follow the complete global seal')
        fixture_map = {f['id']: f for f in fixtures}; production, references, comparisons, identities = [], [], [], []
        for fixture, row in zip(fixtures, records):
            policy = dict(lock['policy'], **fixture['policy_overrides'])
            try:
                check = production_check(row, banks[fixture['id']], policy, lock['gates'])
            except Exception as error:
                check = dict(passed=False, status='saved_output_failure', failure_reason=type(error).__name__ + ': ' + str(error))
            production.append(dict(id=fixture['id'], **check)); write(out / 'production_checks.json', production)
        production_map = {r['id']: r for r in production}; cumulative = 0
        directory = out / 'reference_arrays'; directory.mkdir()
        for name in lock['base_ids']:
            bank, fixture = banks[name], fixture_map[name]; values, evidence = {}, {}
            for order in lock['reference_policy']['orders']:
                charged, call_work = False, 0
                try:
                    if time.perf_counter() - phase >= lock['reference_policy']['max_phase_wall_seconds'] or cumulative >= lock['reference_policy']['max_total_cell_node_evaluations']:
                        raise RuntimeError('fixed global reference budget exhausted; no retry')
                    expected_work = reference_work(bank['input__bounds'], lock['policy']['h'], order, lock['reference_policy'])
                    if cumulative + expected_work > lock['reference_policy']['max_total_cell_node_evaluations']:
                        raise RuntimeError('next fixed reference would exceed total cell-node budget; no retry')
                    result = REF.integrate(**{n: bank['input__' + n] for n in ('mu', 'covariance', 'bounds', 'content_bias')},
                        policy=lock['policy'], reference_policy=lock['reference_policy'], order=order,
                        deadline=phase + lock['reference_policy']['max_phase_wall_seconds'])
                    call_work = result['work']['cell_node_evaluations']; cumulative += call_work; charged = True
                    if result['work']['cell_node_evaluations'] != expected_work:
                        raise ValueError('independent reference work counter differs from native/direct-FOV panels')
                    raw = result.pop('raw_mass'); probability, _, _, _ = probabilities(raw, bank['input__bounds'], bank['input__content_bias'], lock['policy']['floor_density'])
                    elapsed = time.perf_counter() - phase
                    direct_error = abs(result['raw_fov_mass'] - math.fsum(map(float, raw)))
                    norm = abs(math.fsum(map(float, probability)) - 1.)
                    positive = bool(math.fsum(map(float, raw)) > 0)
                    passed = bool(positive and cumulative <= lock['reference_policy']['max_total_cell_node_evaluations']
                        and elapsed <= lock['reference_policy']['max_phase_wall_seconds'] and norm <= lock['gates']['normalization_absolute']
                        and direct_error <= lock['gates']['direct_fov_additivity_absolute'])
                    references.append(dict(id=name, passed=passed, **result, cumulative_cell_node_evaluations=cumulative,
                        phase_elapsed_seconds=elapsed, normalization_error=norm, reference_positive_raw_mass=positive,
                        direct_fov_additivity_error=direct_error))
                    evidence['raw_mass' + str(order)] = raw; evidence['probability' + str(order)] = probability
                    values[order] = (raw, passed)
                except Exception as error:
                    failed_work = dict(cell_node_evaluations=call_work) if charged else getattr(error, 'reference_work', dict(cell_node_evaluations=0))
                    if not charged: cumulative += failed_work['cell_node_evaluations']
                    if hasattr(error, 'partial_raw_mass'):
                        evidence['partial_raw_mass' + str(order)] = error.partial_raw_mass
                    references.append(dict(id=name, order=order, passed=False, failure_reason=type(error).__name__ + ': ' + str(error),
                        work=failed_work, cumulative_cell_node_evaluations=cumulative, phase_elapsed_seconds=time.perf_counter() - phase))
                write(out / 'reference_records.json', references)
                np.savez_compressed(directory / (name + '.npz'), **evidence)
            path = directory / (name + '.npz')
            check = dict(id=name, passed=False, archive_path=str(path.relative_to(out)), archive_sha256=sha(path),
                         array_sha256={n: array_sha(v) for n, v in evidence.items()})
            if set(values) == set(lock['reference_policy']['orders']):
                try:
                    low, high = lock['reference_policy']['orders']
                    check['reference'] = compare_raw(values[low][0], values[high][0], bank['input__bounds'], bank['input__content_bias'],
                        lock['policy']['floor_density'], lock['gates'], reference_comparison=True)
                    if production_map[name]['passed']:
                        check['production'] = compare_raw(bank['raw_mass'], values[high][0], bank['input__bounds'], bank['input__content_bias'],
                            lock['policy']['floor_density'], lock['gates'], archived_probability=bank['probability'])
                    check['passed'] = bool(all(v[1] for v in values.values()) and check['reference']['passed'] and check.get('production', {}).get('passed', False))
                except (ValueError, FloatingPointError, OverflowError) as error:
                    check['failure_reason'] = type(error).__name__ + ': ' + str(error)
            comparisons.append(check); write(out / 'comparisons.json', comparisons)
            print('INDEPENDENT_REFERENCE', name, check['passed'], flush=True)
        for name in lock['identity_ids']:
            fixture = fixture_map[name]; parent = fixture['identity']['parent']
            if production_map[name]['passed'] and production_map[parent]['passed']:
                identities.append(identity_check(fixture, banks[name], banks[parent], lock['gates']))
            else:
                identities.append(dict(id=name, parent=parent, passed=False, failure_reason='identity/base production failed'))
            write(out / 'identities.json', identities)
        verify_joint_lock(run, out)
        elapsed, phase_finished = time.perf_counter() - phase, time.time()
        resource_passed = bool(elapsed <= lock['reference_policy']['max_phase_wall_seconds']
            and cumulative <= lock['reference_policy']['max_total_cell_node_evaluations'])
        passed = bool(seal['all_production_calls_completed'] and seal['phase_wall_passed'] and resource_passed
            and len(references) == 72 and len(comparisons) == 36 and len(identities) == 6
            and all(r['passed'] for r in production + references + comparisons + identities))
        summary = dict(numerical_preflight_passed=passed, scientific_gate_passed=False, goal_complete=False,
            real_inputs_read=False, labels_read=False, candidate_selected=False, base_case_count=36, identity_count=6,
            reference_call_count=len(references), reference_cell_node_evaluations=cumulative, reference_phase_wall_seconds=elapsed,
            reference_budget_passed=resource_passed, production_phase_wall_seconds=seal['phase_wall_seconds'],
            production_phase_wall_passed=seal['phase_wall_passed'], production_failure_count=sum(not r['passed'] for r in production),
            reference_call_failure_count=sum(not r['passed'] for r in references), base_comparison_failure_count=sum(not r['passed'] for r in comparisons),
            identity_failure_count=sum(not r['passed'] for r in identities), started_unix=started,
            phase_finished_unix=phase_finished, finished_unix=time.time(),
            next_action=('fixed_synthetic_numerics_only_passed_separate_protocol_required' if passed else 'stop_fixed_implementation_preserve_all_failures'),
            note='No Student domain truncation or rigorous quadrature/roundoff certificate. Fixed synthetic cases only; every scientific gate remains open.')
        write(out / 'summary.json', summary)
        write(out / 'verification.json', dict(delivery_integrity_passed=True, numerical_preflight_passed=passed,
            producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
            producer_completion_sha256=sha(run / 'completion.json'), producer_forward_complete_sha256=sha(run / 'forward_complete.json'),
            summary_sha256=sha(out / 'summary.json'), source_count=len(lock['source_hashes']), producer_case_count=42,
            independent_reference_case_count=36, reference_call_count=len(references), identity_count=6,
            real_inputs_read=False, labels_read=False, scientific_gate_passed=False, goal_complete=False, finished_unix=time.time()))
        files = inventory(out)
        write(out / 'completion.json', dict(files=files, delivery_integrity_passed=True, numerical_preflight_passed=passed,
            scientific_gate_passed=False, goal_complete=False, real_inputs_read=False, labels_read=False, finished_unix=time.time(),
            post_phase_bookkeeping_seconds_before_completion_write=time.perf_counter() - phase - elapsed,
            total_seconds_before_completion_write=time.perf_counter() - phase))
    except (Exception, KeyboardInterrupt) as error:
        write(out / 'failure.json', dict(error_type=type(error).__name__, message=str(error), started_unix=started, finished_unix=time.time()))
        write(out / 'completion.json', dict(files=inventory(out), delivery_integrity_passed=False,
            numerical_preflight_passed=False, scientific_gate_passed=False, real_inputs_read=False, labels_read=False, goal_complete=False, finished_unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'audit')); parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT); args = parser.parse_args()
    {'freeze': freeze, 'audit': audit}[args.command](args.run.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
