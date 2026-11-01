"""R23 independent input/statistics and output-status audit, without labels.

No production model, QR construction, raw-mass reference or replay is called.
Frozen R22 independent arithmetic/status checks are reused only on saved
arrays; importing their module does not execute its reference integrator.
"""
from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
import shutil
import time
import numpy as np
from experiments import round22_ray_audit as BASE
from experiments import round23_ray_contract as C

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/round23_ray_applicability'
DEFAULT_OUT = ROOT / 'results/round23_ray_applicability_independent_audit'
SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
INPUT_KEYS = ('q_points3', 'q_pixel_cov3', 'q_K', 'k_K', 'q_to_k_transform', 'q_predicted_depth')
sha, read, write, array_sha = BASE.sha, BASE.read, BASE.write, BASE.array_sha
inventory, verify_inventory, runtime = BASE.inventory, BASE.verify_inventory, BASE.check_runtime


def source_names():
    found, pending = set(), list(C.SOURCE_FILES)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / name
        if not path.is_file():
            raise ValueError('missing joint R23 source: ' + name)
        found.add(name)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('relative imports require explicit review')
                if node.module:
                    modules = [node.module] + [node.module + '.' + n.name for n in node.names]
            for module in modules:
                parts = module.split('.')
                paths = [Path(*parts).with_suffix('.py'), Path(*parts) / '__init__.py']
                paths += [Path(*parts[:i]) / '__init__.py' for i in range(1, len(parts))]
                pending.extend(str(p) for p in paths if (ROOT / p).is_file() and str(p) not in found)
    return sorted(found)


def pixel_bounds(size):
    if type(size) is not int or size not in (8, 16, 24):
        raise ValueError('one of the three original native query sizes required')
    return np.asarray([[x, y, x + size, y + size] for y in range(0, 144, size) for x in range(0, 192, size)], np.int64)


def metadata_population():
    """Rebuild the fixed population independently from accepted old metadata."""
    old = read(C.METADATA_PATH)['entries']; entries, specs = [], []
    fields = ('scene', 'pair', 'pair_name', 'input_path', 'input_sha256', 'manifest_path', 'manifest_sha256')
    for scene in SCENES:
        original = next(r for r in old if r['scene'] == scene)
        row = {k: original[k] for k in fields}
        expected = ROOT / 'results/round10_partition_data/train' / scene.replace('/', '__') / '000_0000030_0000045/inputs.npz'
        if row['pair'] != [30, 45] or row['pair_name'] != expected.parent.name or Path(row['input_path']).resolve() != expected.resolve():
            raise ValueError('only the original metadata-first TRAIN pair is authorized')
        entries.append(row)
        for size in (8, 16, 24):
            bounds = pixel_bounds(size)
            indices = [int(i * (len(bounds) - 1) / 7) for i in range(8)]
            for index in indices:
                specs.append(dict(id=f"{scene.replace('/', '__')}__g{size}__q{index:04d}", scene=scene,
                    pair=[30, 45], pair_name=row['pair_name'], query_size=size, query_index=index, query_bounds=bounds[index].tolist()))
    if len(specs) != 72 or len({s['id'] for s in specs}) != 72:
        raise ValueError('the complete unique 72-query population is required')
    return entries, specs


def normalized_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('matching finite statistical arrays required')
    return float(np.max(abs(actual - expected) / (1 + abs(expected))))


def dense_reference(inputs):
    """Backproject sanitized predicted depth with solves; never use labels."""
    if set(inputs) != set(INPUT_KEYS):
        raise ValueError('only the six legal source inputs may enter the audit')
    for name, shape in (('q_points3', (144 * 192, 3)), ('q_pixel_cov3', (144 * 192, 3, 3)),
                        ('q_K', (3, 3)), ('k_K', (3, 3)), ('q_to_k_transform', (4, 4))):
        value = np.asarray(inputs[name])
        if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('invalid stored legal geometry: ' + name)
    depth = np.asarray(inputs['q_predicted_depth'])
    if depth.shape != (144, 192) or depth.dtype.kind != 'f':
        raise ValueError('original predicted depth raster required')
    depth = np.nan_to_num(depth.astype(np.float64), nan=10., posinf=20., neginf=.1).clip(.1, 20.)
    yy, xx = np.mgrid[:144, :192]
    hom = np.column_stack((xx.ravel(), yy.ravel(), np.ones(144 * 192)))
    for name in ('q_K', 'k_K'):
        if not np.array_equal(inputs[name][2], [0., 0., 1.]):
            raise ValueError('canonical pinhole intrinsics required')
    if not np.array_equal(inputs['q_to_k_transform'][3], [0., 0., 0., 1.]):
        raise ValueError('homogeneous query-to-key transform required')
    rays = np.linalg.solve(inputs['q_K'], hom.T).T
    points = rays * depth.ravel()[:, None]
    inverse_columns = np.linalg.solve(inputs['q_K'], np.eye(3))[:, :2]
    within = depth.ravel()[:, None, None] ** 2 * (inverse_columns @ inverse_columns.T)[None] / 12.
    errors = dict(points3=normalized_error(inputs['q_points3'], points),
                  within_covariance=normalized_error(inputs['q_pixel_cov3'], within))
    rotation = inputs['q_to_k_transform'][:3, :3]
    camera_diagnostics = dict(rotation_orthogonality_max_absolute=float(np.max(abs(rotation @ rotation.T - np.eye(3)))),
                              rotation_determinant=float(np.linalg.det(rotation)))
    return dict(points=inputs['q_points3'].reshape(144, 192, 3),
        covariance=inputs['q_pixel_cov3'].reshape(144, 192, 3, 3), errors=errors,
        transform=inputs['q_to_k_transform'], camera_diagnostics=camera_diagnostics,
        H=inputs['k_K'][:2, :2] / 8., offset=inputs['k_K'][:2, 2] / 8.)


def patch_reference(dense, bounds):
    bounds = np.asarray(bounds)
    if (bounds.dtype != np.int64 or bounds.shape != (4,) or np.any(bounds[:2] < 0)
            or np.any(bounds[2:] > [192, 144]) or np.any(bounds[2:] <= bounds[:2])):
        raise ValueError('literal nonempty integer query rectangle required')
    x0, y0, x1, y1 = bounds
    points = dense['points'][y0:y1, x0:x1].reshape(-1, 3)
    mean = np.mean(points, axis=0); delta = points - mean
    covariance = dense['covariance'][y0:y1, x0:x1].mean((0, 1)) + delta.T @ delta / len(points)
    rotation, translation = dense['transform'][:3, :3], dense['transform'][:3, 3]
    return dict(source_mean=mean, source_covariance=covariance, target_mean=rotation @ mean + translation,
        target_covariance=rotation @ covariance @ rotation.T, H=dense['H'], offset=dense['offset'],
        bounds=(pixel_bounds(8).astype(np.float64) - .5) / 8.)


def geometry_check(spec, bank, dense, tolerance):
    expected_fields = {'input__mu', 'input__factor', 'input__bounds', 'input__H', 'input__offset',
        'source__mu', 'source__factor', 'source__covariance', 'target__covariance', 'query_bounds_px', 'query_index'}
    if set(bank) != expected_fields:
        raise ValueError('prepared geometry schema differs')
    if (bank['query_index'].dtype != np.int64 or bank['query_index'].shape != () or int(bank['query_index']) != spec['query_index']
            or bank['query_bounds_px'].dtype != np.int64 or not np.array_equal(bank['query_bounds_px'], spec['query_bounds'])):
        raise ValueError('prepared native query identity differs')
    ref = patch_reference(dense, bank['query_bounds_px'])
    for name in expected_fields - {'query_bounds_px', 'query_index'}:
        if bank[name].dtype != np.float64 or not np.isfinite(bank[name]).all():
            raise ValueError('prepared geometry must be finite float64: ' + name)
    for name in ('source__factor', 'input__factor'):
        if bank[name].shape != (3, 3):
            raise ValueError('the untrimmed three-column QR factor is required')
    source_covariance = bank['source__factor'] @ bank['source__factor'].T
    target_covariance = bank['input__factor'] @ bank['input__factor'].T
    comparisons = [('source__mu', ref['source_mean']), ('input__mu', ref['target_mean']),
        ('source__covariance', ref['source_covariance']), ('target__covariance', ref['target_covariance']),
        ('input__H', ref['H']), ('input__offset', ref['offset']), ('input__bounds', ref['bounds'])]
    errors = {name: normalized_error(bank[name], value) for name, value in comparisons}
    errors.update(source_factor_covariance=normalized_error(source_covariance, ref['source_covariance']),
        target_factor_covariance=normalized_error(target_covariance, ref['target_covariance']),
        factor_target_transform=normalized_error(bank['input__factor'], dense['transform'][:3, :3] @ bank['source__factor']))
    exact_projection = all(np.array_equal(bank[n], ref[r]) for n, r in
        (('input__H', 'H'), ('input__offset', 'offset'), ('input__bounds', 'bounds')))
    return dict(passed=bool(max(errors.values()) <= tolerance and max(dense['errors'].values()) <= tolerance and exact_projection),
        normalized_errors=errors, source_dense_errors=dense['errors'], exact_H_offset_bounds=exact_projection,
        supplied_camera_diagnostics=dense['camera_diagnostics'],
        source_covariance_min_eigenvalue=float(np.linalg.eigvalsh(ref['source_covariance']).min()),
        target_covariance_min_eigenvalue=float(np.linalg.eigvalsh(ref['target_covariance']).min()),
        source_factor_columns=3, target_factor_columns=3, covariance_repair_used=False,
        general_numerical_accuracy_established=False, independent_probability_reference=False), ref


def verify_producer_lock(run):
    lock = read(run / 'lock.json')
    if sha(run / 'lock.json') != (run / 'lock.sha256').read_text().strip():
        raise ValueError('producer lock changed')
    original_acceptance = read(ROOT / 'results/round20_anchor_root_acceptance_20261031.json')
    if (original_acceptance.get('accepted_for_development_interpretation') is not True
            or sha(C.METADATA_PATH) != original_acceptance['files']['results/round20_anchor_fixed/lock.json']):
        raise ValueError('original TRAIN population metadata differs from accepted R20 lock')
    entries, specs = metadata_population()
    expected = dict(version=1, runtime=runtime(), entries=entries, policy=C.POLICY, gates=C.GATES,
        input_keys=list(INPUT_KEYS), image_hw=list(C.IMAGE_HW), case_count=72, key_count=432,
        labels_read=False, general_numerical_accuracy_established=False, scientific_gate_passed=False)
    if any(lock.get(k) != value for k, value in expected.items()):
        raise ValueError('fixed source/population/runtime contract differs')
    if read(run / 'query_specs.json') != specs or sha(run / 'query_specs.json') != lock['query_specs_sha256']:
        raise ValueError('fixed native query population changed')
    if set(lock['source_hashes']) != set(source_names()):
        raise ValueError('joint source closure changed')
    for name, digest in lock['source_hashes'].items():
        if sha(ROOT / name) != digest or sha(run / 'source_snapshot' / name) != digest:
            raise ValueError('current/frozen source changed: ' + name)
    required = {str(ROOT / n) for n in C.BINDINGS}
    required.update(e[n] for e in entries for n in ('input_path', 'manifest_path'))
    receipt = read(ROOT / C.BINDINGS[0])
    if not all(receipt.get(k) is True for k in ('accepted_for_numerical_interpretation',
        'numerical_preflight_passed', 'all_sources_current_and_both_snapshots_verified')):
        raise ValueError('accepted bounded R22 synthetic receipt required')
    for name, digest in receipt['files'].items():
        path = (ROOT / name).resolve()
        required.add(str(path))
        if sha(path) != digest or lock['bindings'].get(str(path)) != digest:
            raise ValueError('accepted R22 receipt binding changed')
    decision = read(ROOT / 'results/round22_ray_post_result_decision_20261031.json')
    if decision.get('root_acceptance_sha256') != sha(ROOT / C.BINDINGS[0]):
        raise ValueError('post-result decision is not bound to the accepted receipt')
    previous = read(ROOT / 'results/round22_ray_preflight/lock.json')
    for name, digest in previous['source_hashes'].items():
        for path in (ROOT / name, ROOT / 'results/round22_ray_preflight/source_snapshot' / name,
                     ROOT / 'results/round22_ray_independent_audit/source_snapshot' / name):
            required.add(str(path))
            if lock['bindings'].get(str(path)) != digest:
                raise ValueError('frozen R22 source provenance changed')
    if set(lock['bindings']) != required:
        raise ValueError('unexpected/missing legal source binding')
    for path, digest in lock['bindings'].items():
        if sha(path) != digest:
            raise ValueError('source binding changed: ' + path)
    for entry in entries:
        if (lock['bindings'][entry['input_path']] != entry['input_sha256']
                or lock['bindings'][entry['manifest_path']] != entry['manifest_sha256']):
            raise ValueError('original accepted TRAIN archive identity changed')
        manifest = read(entry['manifest_path'])
        if (manifest.get('split') != 'train' or manifest.get('forward_uses_sensor') is not False
                or manifest.get('scene') != entry['scene'] or manifest.get('pair') != entry['pair']
                or manifest.get('input_sha256') != entry['input_sha256']):
            raise ValueError('only original no-sensor TRAIN forward archives allowed')
    return lock


def freeze(run, out):
    if out.exists() or any((run / n).exists() for n in ('prepared', 'prepared_complete.json', 'forward', 'started.json', 'completion.json')):
        raise ValueError('independent freeze must precede geometry preparation')
    lock = verify_producer_lock(run)
    if Path(lock['audit_root']).resolve() != out.resolve():
        raise ValueError('independent audit destination changed')
    out.mkdir(parents=True)
    for name in lock['source_hashes']:
        path = out / 'source_snapshot' / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    write(out / 'lock.json', dict(created_unix=time.time(), producer_lock_sha256=sha(run / 'lock.json'),
        source_hashes=lock['source_hashes'], runtime=lock['runtime'], query_specs_sha256=lock['query_specs_sha256'],
        run=str(run.resolve()), general_numerical_accuracy_established=False, scientific_gate_passed=False, labels_read=False))
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')


def verify_joint_lock(run, out):
    lock, design = verify_producer_lock(run), read(out / 'lock.json')
    expected = dict(producer_lock_sha256=sha(run / 'lock.json'), source_hashes=lock['source_hashes'], runtime=lock['runtime'],
                    query_specs_sha256=lock['query_specs_sha256'], run=str(run.resolve()))
    if (sha(out / 'lock.json') != (out / 'lock.sha256').read_text().strip()
            or any(design.get(k) != value for k, value in expected.items()) or design['created_unix'] < lock['created_unix']):
        raise ValueError('independent freeze changed')
    for name, digest in lock['source_hashes'].items():
        if sha(out / 'source_snapshot' / name) != digest:
            raise ValueError('independent source snapshot changed')
    return lock, design


def verify_prepared(run, out):
    lock, design = verify_joint_lock(run, out)
    complete, seal = verify_inventory(run / 'prepared'), read(run / 'prepared_complete.json')
    records, specs = read(run / 'prepared/records.json'), read(run / 'query_specs.json')
    expected = dict(producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
        prepared_records_sha256=sha(run / 'prepared/records.json'), prepared_completion_sha256=sha(run / 'prepared/completion.json'),
        query_specs_sha256=lock['query_specs_sha256'], case_ids=[s['id'] for s in specs], case_count=72,
        labels_read=False, predictions_executed=False, general_numerical_accuracy_established=False, scientific_gate_passed=False)
    if any(seal.get(k) != value for k, value in expected.items()) or [r['id'] for r in records] != expected['case_ids']:
        raise ValueError('complete ordered prepared geometry seal required')
    if not design['created_unix'] <= seal['started_unix'] <= seal['finished_unix']:
        raise ValueError('geometry preparation preceded independent freeze')
    start = read(run / 'prepared/started.json')
    if (start['started_unix'] != seal['started_unix'] or start['producer_lock_sha256'] != sha(run / 'lock.json')
            or start['audit_lock_sha256'] != sha(out / 'lock.json')
            or seal['all_inputs_prepared'] != all(r['passed'] is True for r in records)
            or complete['all_inputs_prepared'] != seal['all_inputs_prepared'] or complete['complete_case_inventory'] is not True):
        raise ValueError('prepared start/status inventory differs')
    expected_files = {'started.json', 'records.json'} | {'arrays/' + s['id'] + '.npz' for s in specs}
    if set(complete['files']) != expected_files:
        raise ValueError('unsealed/missing prepared geometry files')
    for row, spec in zip(records, specs):
        path = 'prepared/arrays/' + spec['id'] + '.npz'
        if (row['archive_path'] != path or row['archive_sha256'] != sha(run / path)
                or any(row[n] != spec[n] for n in ('id', 'scene', 'pair', 'query_size', 'query_index'))):
            raise ValueError('prepared query source identity changed')
    return lock, seal, records, specs


def load_prepared(run, row):
    with np.load(run / row['archive_path'], allow_pickle=False) as archive:
        bank = {n: archive[n] for n in archive.files}
    if {n: array_sha(v) for n, v in bank.items()} != row['array_sha256']:
        raise ValueError('prepared array canonical hashes changed')
    return bank


def finish_failure(directory, error, started):
    write(directory / 'failure.json', dict(error_type=type(error).__name__, message=str(error), started_unix=started, finished_unix=time.time()))
    write(directory / 'completion.json', dict(passed=False, files=inventory(directory), finished_unix=time.time(),
        labels_read=False, general_numerical_accuracy_established=False, scientific_gate_passed=False))


def input_audit(run, out):
    directory = out / 'input_audit'
    if directory.exists() or (out / 'input_verification.json').exists():
        raise ValueError('preserve previous input-audit attempt')
    directory.mkdir(); started = time.time(); write(directory / 'started.json', dict(created_unix=started))
    checks = []
    try:
        lock, seal, records, specs = verify_prepared(run, out)
        if (run / 'forward').exists() or (run / 'started.json').exists() or started < seal['finished_unix']:
            raise ValueError('input audit must follow geometry seal and precede every model call')
        dense, dense_failures = {}, {}
        for entry in lock['entries']:
            try:
                with np.load(entry['input_path'], allow_pickle=False) as archive:
                    legal = {name: archive[name] for name in INPUT_KEYS}
                dense[entry['scene']] = dense_reference(legal)
            except Exception as error:
                dense_failures[entry['scene']] = type(error).__name__ + ': ' + str(error)
        references = {}
        for spec, row in zip(specs, records):
            try:
                if row['passed'] is not True or spec['scene'] in dense_failures:
                    raise ValueError(dense_failures.get(spec['scene'], row.get('failure_reason', 'geometry preparation failed')))
                check, ref = geometry_check(spec, load_prepared(run, row), dense[spec['scene']], lock['gates']['input_normalized_absolute'])
                for name, value in ref.items(): references[spec['id'] + '__' + name] = value
            except Exception as error:
                check = dict(passed=False, failure_reason=type(error).__name__ + ': ' + str(error), normalized_errors={})
            checks.append(dict(id=spec['id'], scene=spec['scene'], query_size=spec['query_size'], query_index=spec['query_index'], **check))
            write(directory / 'checks.json', checks)
        np.savez_compressed(directory / 'independent_geometry.npz', **references)
        verify_joint_lock(run, out)
        passed = bool(len(checks) == 72 and all(c['passed'] for c in checks))
        verification = dict(passed=passed, producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
            producer_prepared_complete_sha256=sha(run / 'prepared_complete.json'),
            producer_prepared_completion_sha256=sha(run / 'prepared/completion.json'), query_specs_sha256=lock['query_specs_sha256'],
            checked_queries=len(checks), source_input_hashes={e['input_path']: e['input_sha256'] for e in lock['entries']},
            maximum_normalized_input_error=max((max(list(c['normalized_errors'].values()) + list(c.get('source_dense_errors', {}).values()), default=0.) for c in checks), default=0.),
            general_numerical_accuracy_established=False, independent_probability_reference=False, scientific_gate_passed=False,
            labels_read=False, started_unix=started, finished_unix=time.time())
        write(out / 'input_verification.json', verification)
        write(directory / 'completion.json', dict(passed=passed, files=inventory(directory),
            verification_sha256=sha(out / 'input_verification.json'), finished_unix=time.time(),
            labels_read=False, general_numerical_accuracy_established=False, scientific_gate_passed=False))
        if not passed:
            raise RuntimeError('independent input audit failed; no model calls authorized')
    except Exception as error:
        if not (directory / 'completion.json').exists(): finish_failure(directory, error, started)
        raise


def verify_input_audit(run, out):
    lock, prepared, records, specs = verify_prepared(run, out)
    done = verify_inventory(out / 'input_audit'); verification = read(out / 'input_verification.json')
    expected = dict(passed=True, producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
        producer_prepared_complete_sha256=sha(run / 'prepared_complete.json'),
        producer_prepared_completion_sha256=sha(run / 'prepared/completion.json'), query_specs_sha256=lock['query_specs_sha256'],
        checked_queries=72, general_numerical_accuracy_established=False, independent_probability_reference=False,
        scientific_gate_passed=False, labels_read=False,
        source_input_hashes={e['input_path']: e['input_sha256'] for e in lock['entries']})
    if (done['passed'] is not True or done['verification_sha256'] != sha(out / 'input_verification.json')
            or any(verification.get(k) != value for k, value in expected.items())
            or not prepared['finished_unix'] <= verification['started_unix'] <= verification['finished_unix']):
        raise ValueError('complete independent input audit required before outputs')
    return lock, prepared, records, specs, verification


def validate_forward_row(row, outputs, key_count=432):
    """A failed attempt is retained evidence only with an explicit reason."""
    if type(row.get('passed')) is not bool or not isinstance(row.get('work'), dict):
        raise ValueError('explicit Boolean status and retained work dictionary required')
    expected_status = 'complete' if row['passed'] else 'failed'
    if row.get('status') != expected_status:
        raise ValueError('call status conflicts with its success flag')
    if not row['passed'] and (not isinstance(row.get('failure_reason'), str) or not row['failure_reason'].strip()):
        raise ValueError('every failed call requires its retained failure reason')
    if any(not isinstance(name, str) or type(value) is not int or value < 0 for name, value in row['work'].items()):
        raise ValueError('finite nonnegative integer work counters required')
    elapsed = row.get('call_and_input_load_seconds')
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('every call requires finite nonnegative timing')
    if type(row.get('peak_rss_kib_process')) is not int or row['peak_rss_kib_process'] < 0:
        raise ValueError('retained process high-water RSS required')
    success = {'raw_mass', 'probability', 'floored_mass', 'quadrature_error_estimate', 'inner_error_estimate_mass'}
    partial = {'partial_raw_mass', 'partial_raw_fov_mass', 'partial_quadrature_error_estimate', 'partial_fov_quadrature_error_estimate'}
    if row['passed']:
        if set(outputs) != success:
            raise ValueError('complete success-array schema required')
    elif not set(outputs) <= partial:
        raise ValueError('failed calls may retain partial arrays only')
    for name, value in outputs.items():
        shape = () if name in ('partial_raw_fov_mass', 'partial_fov_quadrature_error_estimate') else (key_count,)
        if value.dtype != np.float64 or value.shape != shape:
            raise ValueError('retained output dtype/shape differs: ' + name)
    return True


def audit(run, out):
    directory = out / 'final_audit'
    if directory.exists() or (out / 'verification.json').exists() or (out / 'completion.json').exists():
        raise ValueError('preserve previous final applicability audit')
    directory.mkdir(); started = time.time(); write(directory / 'started.json', dict(created_unix=started))
    checks = []
    try:
        lock, prepared, prepared_records, specs, inputs = verify_input_audit(run, out)
        complete = verify_inventory(run); forward = verify_inventory(run / 'forward'); seal = read(run / 'forward_complete.json')
        records = read(run / 'forward/records.json'); ids = [s['id'] for s in specs]
        expected = dict(producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
            producer_prepared_complete_sha256=sha(run / 'prepared_complete.json'),
            input_verification_sha256=sha(out / 'input_verification.json'), forward_records_sha256=sha(run / 'forward/records.json'),
            forward_completion_sha256=sha(run / 'forward/completion.json'), query_specs_sha256=lock['query_specs_sha256'],
            case_ids=ids, case_count=72, all_case_attempts_retained=True, labels_read=False,
            general_numerical_accuracy_established=False, scientific_gate_passed=False)
        if any(seal.get(k) != value for k, value in expected.items()) or [r['id'] for r in records] != ids:
            raise ValueError('complete ordered 72-call forward seal required')
        if (not inputs['finished_unix'] <= seal['started_unix'] <= seal['phase_finished_unix'] <= seal['finished_unix'] <= started
                or seal['phase_wall_passed'] != (seal['phase_wall_seconds'] <= lock['gates']['production_max_phase_wall_seconds'])):
            raise ValueError('input-audit/model-call chronology or phase status differs')
        expected_forward_files = {'started.json', 'records.json'} | {'arrays/' + s['id'] + '.npz' for s in specs}
        if set(forward['files']) != expected_forward_files:
            raise ValueError('unsealed/missing forward case files')
        root_files = {'lock.json', 'lock.sha256', 'query_specs.json', 'prepared_complete.json', 'forward_complete.json'}
        root_files.update('source_snapshot/' + name for name in lock['source_hashes'])
        root_files.update('prepared/' + name for name in read(run / 'prepared/completion.json')['files'])
        root_files.add('prepared/completion.json')
        root_files.update('forward/' + name for name in forward['files']); root_files.add('forward/completion.json')
        if set(complete['files']) != root_files:
            raise ValueError('unexpected/missing final producer file')
        forward_start = read(run / 'forward/started.json')
        if (forward_start['started_unix'] != seal['started_unix'] or forward_start['producer_lock_sha256'] != sha(run / 'lock.json')
                or forward_start['audit_lock_sha256'] != sha(out / 'lock.json')
                or forward_start['input_verification_sha256'] != sha(out / 'input_verification.json')
                or complete['forward_complete_sha256'] != sha(run / 'forward_complete.json')
                or complete['complete_case_inventory'] is not True or forward['complete_case_inventory'] is not True
                or complete['applicability_calls_completed'] != seal['applicability_calls_completed']
                or forward['applicability_calls_completed'] != seal['applicability_calls_completed']
                or forward['phase_wall_passed'] != seal['phase_wall_passed']):
            raise ValueError('forward start/status completion differs')
        for spec, row, source in zip(specs, records, prepared_records):
            path = 'forward/arrays/' + spec['id'] + '.npz'
            if (row['archive_path'] != path or row['archive_sha256'] != sha(run / path)
                    or row['prepared_archive_path'] != source['archive_path'] or row['prepared_archive_sha256'] != source['archive_sha256']
                    or any(row[n] != spec[n] for n in ('id', 'scene', 'pair', 'query_size', 'query_index'))):
                raise ValueError('forward/prepared array identity differs')
            geometry = load_prepared(run, source)
            with np.load(run / path, allow_pickle=False) as archive:
                outputs = {n: archive[n] for n in archive.files}
            if {n: array_sha(v) for n, v in outputs.items()} != row['array_sha256']:
                raise ValueError('forward output canonical identity differs')
            validate_forward_row(row, outputs)
            bank = dict(geometry, **outputs)
            try:
                check = BASE.production_check({}, row, bank, lock['gates'], lock['policy'])
            except (ValueError, KeyError, OverflowError, FloatingPointError) as error:
                check = dict(passed=False, status='output_arithmetic_or_metadata_failure', failure_reason=str(error))
            checks.append(dict(id=spec['id'], scene=spec['scene'], query_size=spec['query_size'], query_index=spec['query_index'],
                retained_call_valid=True, model_call_completed=row['passed'] is True,
                call_and_input_load_seconds=row['call_and_input_load_seconds'], work=row['work'], **check))
            write(directory / 'checks.json', checks)
        verify_joint_lock(run, out)
        delivery_passed = bool(len(checks) == 72 and all(r['passed'] or not r['model_call_completed'] for r in checks))
        all_completed = bool(seal['phase_wall_passed'] and all(r['model_call_completed'] and r['passed'] for r in checks))
        expected_completed = bool(seal['phase_wall_passed'] and all(r['passed'] is True for r in records))
        if seal['applicability_calls_completed'] != expected_completed:
            raise ValueError('producer completion flag differs from retained call statuses/budget')
        summary = dict(applicability_calls_completed=all_completed, delivery_integrity_passed=delivery_passed,
            general_numerical_accuracy_established=False, independent_probability_reference=False, scientific_gate_passed=False,
            efficacy_tested=False, labels_read=False, candidate_selected=False, checked_queries=72, native_key_count=432,
            completed_call_count=sum(r['model_call_completed'] for r in checks), failed_call_count=sum(not r['model_call_completed'] for r in checks),
            output_arithmetic_or_metadata_failure_count=sum(r['model_call_completed'] and not r['passed'] for r in checks),
            production_phase_wall_seconds=seal['phase_wall_seconds'], production_phase_wall_passed=seal['phase_wall_passed'],
            next_action=('fixed_input_applicability_only_separate_numerical_reference_protocol_required' if all_completed else
                         'stop_fixed_real_input_implementation_preserve_every_failure'),
            note='Input moments and saved-array arithmetic/status only; no independent probability-mass accuracy, efficacy, or new confirmation evidence.',
            started_unix=started, finished_unix=time.time())
        write(out / 'summary.json', summary)
        write(out / 'verification.json', dict(delivery_integrity_passed=delivery_passed, applicability_calls_completed=all_completed,
            producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
            producer_prepared_complete_sha256=sha(run / 'prepared_complete.json'), input_verification_sha256=sha(out / 'input_verification.json'),
            producer_forward_complete_sha256=sha(run / 'forward_complete.json'), producer_completion_sha256=sha(run / 'completion.json'),
            summary_sha256=sha(out / 'summary.json'), checked_queries=72, general_numerical_accuracy_established=False,
            independent_probability_reference=False, scientific_gate_passed=False, labels_read=False, finished_unix=time.time()))
        write(directory / 'completion.json', dict(files=inventory(directory), delivery_integrity_passed=delivery_passed,
            verification_sha256=sha(out / 'verification.json'), finished_unix=time.time()))
        write(out / 'completion.json', dict(files=inventory(out), delivery_integrity_passed=delivery_passed,
            applicability_calls_completed=all_completed, general_numerical_accuracy_established=False, scientific_gate_passed=False, finished_unix=time.time()))
    except Exception as error:
        if not (directory / 'completion.json').exists(): finish_failure(directory, error, started)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'input-audit', 'audit'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN); parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    {'freeze': freeze, 'input-audit': input_audit, 'audit': audit}[args.command](args.run.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
