"""Stage A of the predeclared R15 old-training calibration, never validation.

All 5,184 float64 probability arrays are retained. Three complete forward
shards must be sealed before any center target is decoded. An independent
audit, not this producer, authorizes either stopping or the full regional
calibration. This file contains no regional metric or final model selection.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments import round12_stress_probe as S
from experiments import round11_target_conflict as T
from models.round10_partition_pe import prepare_support, pool_content

ROOT = Path(__file__).resolve().parents[1]
SCENES = T.TRAIN_SCENES
QUERIES = ('g8', 'g16', 'g24')
QUERY_COUNTS = dict(g8=432, g16=108, g24=48)
FAMILIES = ('quartic_full', 'quartic_point', 'gaussian_point')
GRID = tuple(dict(h=h, temperature=t) for h in (.25, .5, 1., 2.) for t in (0., 5., 10.))
LEGACY_INDEX = 4  # h=.5, T=5, already in the declared grid.
FIELDS = ('q_to_k_xy', 'q_to_k_pixel_cov', 'q_features', 'k_features')
QUARTIC_MODULE = 'models.round15_quartic_tiled'
NUMERICAL_POLICY = dict(quadrature_order=24, query_chunk_size=16, key_chunk_size=32,
                        maximum_tile_side=1., probability_dtype='float64',
                        preparation_dtype='float32', content_dtype='float32',
                        density_floor=1.e-30)
RAW_PROBABILITY_BYTES = 3 * 12 * 48 * sum(QUERY_COUNTS.values()) * 432 * 8
PCK_TOLERANCE = .005
PCK_EXACT_TOLERANCE = Fraction(1, 200)
NUMERICAL_ROOT = ROOT / 'results/round15_quartic_tiled24_numerical_audit_frozen'
FORWARD_PREFLIGHT_ROOT = ROOT / 'results/round15_train_numerical24'
COST_REPORT = ROOT / 'results/round15_quartic_preflight24/cost_cpu.json'
PREVIOUS_FAILURE_ROOT = ROOT / 'results/round15_train_numerical'
PREVIOUS_FAILURE_LOCK_SHA256 = '03fbb72ed8f4d5bbdc7cb71ae37bb1ac277fabd1a02ea939582381490c4093e9'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def verify_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('changed frozen file: ' + str(path))


def sources():
    # Includes the transitive frozen partition/content/Gaussian preparation.
    result = S.sources()
    for name in ('ROUND15_CANDIDATE_PROTOCOL.md', 'experiments/round15_feasibility.py',
                 'tests/test_round15_feasibility.py', 'experiments/round11_target_conflict.py',
                 'models/round15_quartic_pe.py', 'models/round15_quartic_tiled.py',
                 'ROUND15_NUMERICAL24_PROTOCOL.md', 'ROUND15_TRAIN_NUMERICAL24_PROTOCOL.md',
                 'experiments/round15_train_numerical24.py', 'tests/test_round15_train_numerical24.py',
                 'experiments/round15_quartic_preflight24.py', 'tests/test_round15_quartic_preflight24.py',
                 'ROUND15_TRAIN_NUMERICAL_PROTOCOL.md', 'experiments/round15_train_numerical.py',
                 'tests/test_round15_train_numerical.py'):
        result[name] = sha(ROOT / name)
    return result


def named_probability(family, index, query):
    return f'{family}__c{index:02d}__{query}'


def expected_arrays():
    return {named_probability(f, i, q): (QUERY_COUNTS[q], 432)
            for f in FAMILIES for i in range(len(GRID)) for q in QUERIES}


def target_contract(targets, audit, records):
    """Metadata only: target archives are not hashed or decoded before forward."""
    lock, complete, verification = (read(targets / 'lock.json'),
                                    read(targets / 'completion.json'), read(audit))
    lock_hash = sha(targets / 'lock.json')
    if lock_hash != (targets / 'lock.sha256').read_text().strip():
        raise ValueError('changed target lock')
    if complete['lock_sha256'] != lock_hash or lock['training_manifest_records'] != records:
        raise ValueError('target completion/cohort mismatch')
    verify_files({str(ROOT / p): h for p, h in lock['source_hashes'].items()})
    if verification['passed'] is not True:
        raise ValueError('independent target audit must pass')
    if (verification['target_lock_sha256'] != lock_hash
            or verification['target_completion_sha256'] != sha(targets / 'completion.json')
            or verification['target_summary_sha256'] != complete['files']['summary.json']):
        raise ValueError('target independent audit binding mismatch')
    entries = []
    for record in records:
        path = Path(record['path'])
        name = 'targets/' + record['scene'].replace('/', '__') + '__' + path.parent.name + '.npz'
        entries.append(dict(scene=record['scene'], pair=record['pair'], path=str(path.parent),
                            target_path=str(targets / name), target_sha256=complete['files'][name]))
    if {e['target_path'] for e in entries} != {str(targets / p) for p in complete['files']
                                             if p.startswith('targets/')}:
        raise ValueError('unexpected target archive population')
    return dict(root=str(targets), audit_path=str(audit), lock_sha256=lock_hash,
                completion_sha256=sha(targets / 'completion.json'), audit_sha256=sha(audit),
                entries=entries)


def numerical_contract(directory):
    """Require the final independent synthetic tiling audit, not just unit tests."""
    lock = read(directory / 'lock.json')
    lock_hash = sha(directory / 'lock.json')
    if lock_hash != (directory / 'lock.sha256').read_text().strip():
        raise ValueError('numerical lock mismatch')
    report = Path(lock['report'])
    if sha(report / 'verification.json') != lock['report_sha256'] or sha(report / 'completion.json') != lock['completion_sha256']:
        raise ValueError('numerical report changed')
    result, complete = read(report / 'verification.json'), read(report / 'completion.json')
    verify_files({str(report / p): h for p, h in complete['files'].items()})
    for source, digest in lock['source_hashes'].items():
        if sha(ROOT / source) != digest or sha(directory / 'sources' / source) != digest:
            raise ValueError('numerical source mismatch')
    if (result['passed'] is not True or result['quadrature_order'] != 24
            or lock['quadrature_order'] != 24
            or result['maximum_tile_side'] != 1. or result['internal_floor_used']
            or result['real_observations_read'] or result['labels_read']):
        raise ValueError('required fixed numerical policy has not passed')
    return dict(root=str(directory), lock_sha256=lock_hash,
                report_sha256=sha(report / 'verification.json'),
                completion_sha256=sha(report / 'completion.json'))


def previous_failure_contract():
    """Retain and verify the failed GL16 evidence; it cannot become a pass."""
    directory = PREVIOUS_FAILURE_ROOT
    lock, summary, complete = (read(directory / name) for name in
                               ('lock.json', 'summary.json', 'completion.json'))
    digest = sha(directory / 'lock.json')
    if digest != PREVIOUS_FAILURE_LOCK_SHA256 or (directory / 'lock.sha256').read_text().strip() != digest:
        raise ValueError('historical GL16 failure lock changed')
    if (summary['passed'] is not False or complete['passed'] is not False
            or summary['reference_converged'] is not True or lock['integration_order'] != 16
            or summary['labels_read'] or summary['validation_or_test_read']):
        raise ValueError('historical GL16 failure must remain failed and label-free')
    if (summary['lock_sha256'] != digest or complete['lock_sha256'] != digest
            or complete['summary_sha256'] != sha(directory / 'summary.json')):
        raise ValueError('historical GL16 failure chain mismatch')
    verify_files({str(directory / path): value for path, value in complete['files'].items()})
    for path, value in lock['source_hashes'].items():
        if sha(ROOT / path) != value or sha(directory / 'sources' / path) != value:
            raise ValueError('historical GL16 source changed')
    return dict(root=str(directory), lock_sha256=digest, summary_sha256=sha(directory / 'summary.json'),
                completion_sha256=sha(directory / 'completion.json'), cohort=lock['cohort'], original_passed=False)


def forward_preflight_contract(directory):
    """Independent first-training-pair numerical checks, metadata only."""
    lock, summary, complete = (read(directory / name) for name in
                               ('lock.json', 'summary.json', 'completion.json'))
    digest = sha(directory / 'lock.json')
    if digest != (directory / 'lock.sha256').read_text().strip():
        raise ValueError('forward numerical lock mismatch')
    if complete['lock_sha256'] != digest or summary['lock_sha256'] != digest:
        raise ValueError('forward numerical chain mismatch')
    verify_files({str(directory / p): h for p, h in complete['files'].items()})
    if complete['files'].get('summary.json') != sha(directory / 'summary.json'):
        raise ValueError('unbound forward numerical summary')
    previous = previous_failure_contract()
    if (summary['passed'] is not True or complete['passed'] is not True
            or complete['summary_sha256'] != sha(directory / 'summary.json')
            or summary['labels_read'] or summary['validation_or_test_read']
            or lock['integration_order'] != 24 or summary['integration_order'] != 24
            or summary['historical_GL16_failure_retained'] is not True
            or summary['previous_failure'] != previous or lock['previous_failure'] != previous):
        raise ValueError('independent label-free training numerical preflight required')
    old = read(PREVIOUS_FAILURE_ROOT / 'lock.json')
    for field in ('cohort', 'npz_fields_read', 'strides', 'query_indices', 'methods', 'widths',
                  'temperatures', 'reference_orders', 'tolerances', 'geometry_rows', 'probability_rows'):
        if old[field] != lock[field]:
            raise ValueError('numerical repair changed cohort, configuration or tolerance')
    for path, expected in lock['source_hashes'].items():
        if sha(ROOT / path) != expected or sha(directory / 'sources' / path) != expected:
            raise ValueError('changed forward numerical source')
    return dict(root=str(directory), lock_sha256=digest,
                summary_sha256=sha(directory / 'summary.json'),
                completion_sha256=sha(directory / 'completion.json'), previous_failure=previous)


def verify_preflight_cohort(preflight, records, data, numerical):
    lock = read(Path(preflight['root']) / 'lock.json')
    if str(Path(lock['data_root']).resolve()) != str(data) or lock['data_completion_sha256'] != sha(data / 'completion.json'):
        raise ValueError('numerical preflight data/cohort mismatch')
    if set(lock['npz_fields_read']) != set(FIELDS):
        raise ValueError('numerical preflight uses different forward information')
    if (lock['parent']['lock_sha256'] != numerical['lock_sha256']
            or lock['parent']['report_sha256'] != numerical['report_sha256']
            or lock['integration_order'] != NUMERICAL_POLICY['quadrature_order']
            or lock['widths'] != [.25, .5, 1., 2.] or lock['temperatures'] != [0., 5., 10.]
            or lock['strides'] != [8, 16, 24]):
        raise ValueError('numerical preflight operator/grid mismatch')
    expected = [min((r for r in records if r['scene'] == scene), key=lambda r: r['path']) for scene in SCENES]
    if len(lock['cohort']) != 3:
        raise ValueError('numerical preflight must cover three first-training pairs')
    for record, actual in zip(expected, lock['cohort']):
        manifest = Path(record['path'])
        meta = read(manifest)
        if (actual['scene'] != record['scene'] or actual['pair'] != record['pair']
                or Path(actual['manifest_path']) != manifest or actual['manifest_sha256'] != record['sha256']
                or Path(actual['input_path']) != manifest.parent / 'inputs.npz'
                or actual['input_sha256'] != meta['input_sha256']):
            raise ValueError('numerical preflight first-training-pair identity mismatch')


def cost_contract(path, numerical):
    value = read(path)
    if value['labels_read'] or value['real_inputs_read'] or value['order'] != 24:
        raise ValueError('expected fixed synthetic-only cost preflight')
    lock_path = path.parent / 'lock.json'
    lock = read(lock_path)
    if value['lock_sha256'] != sha(lock_path) or sha(lock_path) != (path.parent / 'lock.sha256').read_text().strip():
        raise ValueError('cost preflight lock mismatch')
    if (lock['order'] != 24 or lock['numerical_report_sha256'] != numerical['report_sha256']
            or lock['numerical_completion_sha256'] != numerical['completion_sha256']):
        raise ValueError('cost and independent numerical operator differ')
    verify_files({str(ROOT / p): h for p, h in lock['source_hashes'].items()})
    verify_files({str(path.parent / 'source_snapshot' / p): h for p, h in lock['source_hashes'].items()})
    return dict(path=str(path), sha256=sha(path), lock_path=str(lock_path), lock_sha256=sha(lock_path))


def freeze(out, data, targets, target_audit, numerical_root, forward_preflight, cost_report):
    if out.exists():
        raise ValueError('refusing freeze overwrite')
    paths = [data, targets, target_audit, numerical_root, forward_preflight, cost_report]
    data, targets, target_audit, numerical_root, forward_preflight, cost_report = [p.resolve() for p in paths]
    records = T.data_records(data)
    if len({(r['scene'], tuple(r['pair'])) for r in records}) != 48:
        raise ValueError('duplicate training pair')
    target = target_contract(targets, target_audit, records)
    numerical = numerical_contract(numerical_root)
    preflight = forward_preflight_contract(forward_preflight)
    verify_preflight_cohort(preflight, records, data, numerical)
    cost = cost_contract(cost_report, numerical)
    inputs, manifests = {}, {}
    for record in records:
        path = Path(record['path'])
        if sha(path) != record['sha256']:
            raise ValueError('changed training manifest')
        meta = read(path)
        if (meta['split'] != 'train' or meta['forward_uses_sensor']
                or meta['scene'] != record['scene'] or meta['pair'] != record['pair']):
            raise ValueError('unexpected training identity or forward information')
        inputs[str(path.parent / 'inputs.npz')] = meta['input_sha256']
        manifests[str(path)] = record['sha256']
    verify_files(inputs)
    out.mkdir(parents=True)
    parts = S.M.partitions()
    np.savez_compressed(out / 'partitions.npz', **{name + '__' + key: value
        for name, part in parts.items() for key, value in part.items()})
    hashes = sources()
    for name in hashes:
        destination = out / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    lock = dict(created_unix=time.time(), source_hashes=hashes, data_root=str(data),
                data_completion_sha256=sha(data / 'completion.json'), training_manifest_records=records,
                target_contract=target, numerical_contract=numerical, forward_preflight=preflight,
                cost_contract=cost, inputs=inputs, manifests=manifests,
                partitions_sha256=sha(out / 'partitions.npz'), families=list(FAMILIES),
                grid=list(GRID), queries=list(QUERIES), numerical_policy=NUMERICAL_POLICY,
                raw_probability_bytes=RAW_PROBABILITY_BYTES, pair_count=48, probability_array_count=5184,
                field_allowlist=list(FIELDS), legacy_point_index=LEGACY_INDEX, pck_tolerance=PCK_TOLERANCE,
                pck_tolerance_exact=[PCK_EXACT_TOLERANCE.numerator, PCK_EXACT_TOLERANCE.denominator],
                stage_order=['forward_all', 'seal_forward', 'evaluate_centers', 'summarize', 'independent_audit'],
                empty_feasible_set_action='stop_unchanged_candidate', regional_calibration_roles=8,
                regional_calibration_configurations_per_role=12, regional_calibration_conditions=14,
                all_final_probabilities_retained=True, all_forward_before_any_target_decode=True,
                center_labels_decoded=False, regional_labels_decoded=False, fresh_test_read=False,
                torch_version=torch.__version__, numpy_version=np.__version__,
                deterministic_algorithms=True, tf32=False, cpu_threads=4)
    write(out / 'lock.json', lock)
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')
    print('FROZEN', sha(out / 'lock.json'), flush=True)


def verify_lock(out):
    lock = read(out / 'lock.json')
    if sha(out / 'lock.json') != (out / 'lock.sha256').read_text().strip():
        raise ValueError('feasibility lock changed')
    if lock['source_hashes'] != sources() or lock['grid'] != list(GRID) or lock['families'] != list(FAMILIES):
        raise ValueError('source or configuration changed after freeze')
    for name, digest in lock['source_hashes'].items():
        if sha(out / 'source_snapshot' / name) != digest:
            raise ValueError('source snapshot changed')
    if lock['numerical_policy'] != NUMERICAL_POLICY or lock['queries'] != list(QUERIES):
        raise ValueError('operator policy changed')
    if lock['torch_version'] != torch.__version__ or lock['numpy_version'] != np.__version__:
        raise ValueError('software version changed')
    data = Path(lock['data_root'])
    if sha(data / 'completion.json') != lock['data_completion_sha256'] or T.data_records(data) != lock['training_manifest_records']:
        raise ValueError('training cohort changed')
    verify_files(lock['inputs']); verify_files(lock['manifests'])
    target = lock['target_contract']
    if target_contract(Path(target['root']), Path(target['audit_path']), lock['training_manifest_records']) != target:
        raise ValueError('target metadata changed')
    if numerical_contract(Path(lock['numerical_contract']['root'])) != lock['numerical_contract']:
        raise ValueError('numerical provenance changed')
    if forward_preflight_contract(Path(lock['forward_preflight']['root'])) != lock['forward_preflight']:
        raise ValueError('training numerical provenance changed')
    verify_preflight_cohort(lock['forward_preflight'], lock['training_manifest_records'], data, lock['numerical_contract'])
    if cost_contract(Path(lock['cost_contract']['path']), lock['numerical_contract']) != lock['cost_contract']:
        raise ValueError('cost provenance changed')
    if sha(out / 'partitions.npz') != lock['partitions_sha256']:
        raise ValueError('partitions changed')
    return lock


def scene_entries(lock, scene):
    if scene not in SCENES:
        raise ValueError('only fixed old-training scenes are authorized')
    entries = sorted((e for e in lock['target_contract']['entries'] if e['scene'] == scene),
                     key=lambda e: Path(e['path']).name)
    if len(entries) != 16:
        raise ValueError('expected 16 old-training pairs')
    return entries


def load_forward_inputs(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in FIELDS}


def prepare_query(observations, part, full):
    query = prepare_support(observations['q_to_k_xy'] / 8,
        observations['q_to_k_pixel_cov'] / 64, part['owner'], part['bounds'], part['parent'],
        include_full_moments=full)
    if not full:
        query['cov'] = query['mean'].new_zeros((len(query['mean']), 2, 2))
    query['features'] = pool_content(observations['q_features'], part['atoms'], len(part['mean']))
    return query


def condition_predictions(full, point, key):
    """One native-cell mass per family/h, then the three frozen temperatures.

    No labels accepted. Point computes no global covariance; full and point
    share their legitimate mean/content definition, not a costly preparation.
    """
    quartic = importlib.import_module(QUARTIC_MODULE).quartic_cell_mass
    output, diagnostics = {}, []
    content = full['features'] @ key['features'].T  # inherited float32 content dot
    if not torch.equal(full['mean'], point['mean']) or not torch.equal(full['features'], point['features']):
        raise ValueError('full and point means/content differ')
    for family in FAMILIES:
        query = full if family == 'quartic_full' else point
        for hi, h in enumerate((.25, .5, 1., 2.)):
            if family == 'gaussian_point':
                raw = S.M.tiled_cell_mass(query['mean'], query['cov'].double(), key, h)
            else:
                # The current stage uses only g8 native keys, with one bound/key.
                if len(key['cell_owner']) != len(key['mass']) or not torch.equal(
                        key['cell_owner'], torch.arange(len(key['mass']), device=key['cell_owner'].device)):
                    raise ValueError('feasibility requires original native g8 keys')
                raw = quartic(query['mean'], query['cov'], key['cell_bounds'], bandwidth=h,
                    quadrature_order=24, query_chunk_size=16, key_chunk_size=32)
            if raw.dtype != torch.float64 or not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
                raise ValueError('invalid raw mass')
            fov = raw.sum(-1)
            diagnostics.append(dict(family=family, h=h, n_query=len(raw),
                raw_zero_cells=int((raw == 0).sum()), raw_zero_fov_rows=int((fov == 0).sum()),
                raw_tiny_fov_rows=int((fov < 1.e-20).sum()), raw_fov_min=float(fov.min()),
                raw_fov_max=float(fov.max())))
            geometry = (raw + 1.e-30 * (key['mass'].double() / 64)[None]).log()
            for ti, temperature in enumerate((0., 5., 10.)):
                probability = (geometry + temperature * content).softmax(-1)
                if not bool(torch.isfinite(probability).all()) or float((probability.sum(-1)-1).abs().max()) > 1.e-12:
                    raise ValueError('invalid final probability')
                output[(family, hi * 3 + ti)] = probability
    return output, diagnostics


def configure_runtime(device):
    if device != 'cpu' and not device.startswith('cuda:'):
        raise ValueError('explicit CPU or CUDA device required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.startswith('cuda:'):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.cuda.set_device(device)
    return dict(device=device, device_name=torch.cuda.get_device_name() if device.startswith('cuda:') else 'cpu',
                torch_version=torch.__version__, numpy_version=np.__version__,
                deterministic_algorithms=True, tf32=False, cpu_threads=4)


def pair_predictions(inputs, partition, device):
    observations = S.A.observation_tensors(inputs, device)
    key = S.M.prepare_key(observations, partition['g8'])
    arrays, diagnostics = {}, []
    for q in QUERIES:
        full = prepare_query(observations, partition[q], True)
        point = prepare_query(observations, partition[q], False)
        probabilities, numeric = condition_predictions(full, point, key)
        for (family, index), probability in probabilities.items():
            arrays[named_probability(family, index, q)] = probability.cpu().numpy()
        diagnostics.extend(dict(query=q, **row) for row in numeric)
    return arrays, diagnostics


def validate_arrays(arrays):
    if set(arrays) != set(expected_arrays()):
        raise ValueError('missing or extra native probability arrays')
    for name, shape in expected_arrays().items():
        probability = arrays[name]
        if probability.shape != shape or probability.dtype != np.float64:
            raise ValueError('probability shape/dtype mismatch')
        if not np.isfinite(probability).all() or (probability < 0).any():
            raise ValueError('invalid probability')
        if np.max(np.abs(probability.sum(-1) - 1.)) > 1.e-12:
            raise ValueError('unnormalized probability')


def complete_stage(directory, out, stage, scene, started, **metadata):
    files = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()}
    write(directory / 'completion.json', dict(stage=stage, scene=scene, started_unix=started,
        finished_unix=time.time(), lock_sha256=sha(out / 'lock.json'), files=files, **metadata))


def verify_stage(directory, out, stage, scene):
    result = read(directory / 'completion.json')
    if result['stage'] != stage or result['scene'] != scene or result['lock_sha256'] != sha(out / 'lock.json'):
        raise ValueError('stage identity mismatch')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p.name != 'completion.json'}
    if actual != set(result['files']):
        raise ValueError('extra or missing stage artifact')
    verify_files({str(directory / p): h for p, h in result['files'].items()})
    return result


@torch.no_grad()
def forward(out, scene, device):
    lock = verify_lock(out)
    entries = scene_entries(lock, scene)
    directory = out / 'forward' / scene.replace('/', '__')
    if directory.exists():
        raise ValueError('refusing overwrite/partial forward')
    if shutil.disk_usage(out).free < RAW_PROBABILITY_BYTES + 512 * 1024**2:
        raise ValueError('insufficient free space for full retained Stage A probabilities')
    runtime = configure_runtime(device)
    directory.mkdir(parents=True)
    started = time.time()
    write(directory / 'started.json', dict(created_unix=started, lock_sha256=sha(out / 'lock.json'), **runtime))
    partition = S.M.partition_tensors(S.read_parts(out), device)
    records, numerical = [], []
    for pair_index, entry in enumerate(entries):
        path = Path(entry['path'])
        inputs = load_forward_inputs(path / 'inputs.npz')
        arrays, diagnostics = pair_predictions(inputs, partition, device)
        validate_arrays(arrays)
        if pair_index == 0:
            repeated, _ = pair_predictions(inputs, partition, device)
            if any(array_sha(arrays[k]) != array_sha(repeated[k]) for k in arrays):
                raise ValueError('actual-input repeated forward is not bitwise deterministic')
            write(directory / 'determinism.json', dict(scene=scene, pair=entry['pair'],
                array_hashes={k: array_sha(v) for k, v in arrays.items()}, passed=True,
                repeated_array_count=len(arrays), labels_decoded=False))
            del repeated
        name = path.name + '.npz'
        np.savez_compressed(directory / name, **arrays)
        archive_hash = sha(directory / name)
        for family in FAMILIES:
            for i, cfg in enumerate(GRID):
                for q in QUERIES:
                    field = named_probability(family, i, q)
                    records.append(dict(scene=scene, pair=entry['pair'], family=family, grid_index=i,
                        query=q, config=cfg, archive=name, archive_sha256=archive_hash, field=field,
                        shape=list(arrays[field].shape), dtype=str(arrays[field].dtype),
                        array_sha256=array_sha(arrays[field]), input_sha256=lock['inputs'][str(path / 'inputs.npz')]))
        numerical.extend(dict(scene=scene, pair=entry['pair'], **row) for row in diagnostics)
        print('FORWARD', scene, pair_index + 1, '/16', flush=True)
    write(directory / 'records.json', records)
    write(directory / 'numerical.json', numerical)
    complete_stage(directory, out, 'forward', scene, started, record_count=len(records), pair_count=16,
                   labels_decoded=False, all_probabilities_retained=True, runtime=runtime)


def forward_chain(out):
    result = {}
    for scene in SCENES:
        directory = out / 'forward' / scene.replace('/', '__')
        complete = verify_stage(directory, out, 'forward', scene)
        if complete['record_count'] != 16 * 108 or complete['labels_decoded']:
            raise ValueError('incomplete forward or prior label read')
        result[scene] = dict(completion_sha256=sha(directory / 'completion.json'),
                             finished_unix=complete['finished_unix'])
    return result


def verify_population(rows, entries):
    expected = {(e['scene'], tuple(e['pair']), f, i, q)
                for e in entries for f in FAMILIES for i in range(12) for q in QUERIES}
    identities = [(r['scene'], tuple(r['pair']), r['family'], r['grid_index'], r['query']) for r in rows]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError('missing, duplicate or unauthorized probability population')
    populations = {}
    for row in rows:
        if row['config'] != GRID[row['grid_index']]:
            raise ValueError('configuration identity mismatch')
        if 'n_point' in row:
            identity = (row['scene'], tuple(row['pair']), row['query'])
            current = (row['n_query'], row['n_point'])
            if row['n_query'] != QUERY_COUNTS[row['query']] or not 0 <= row['pck_count'] <= row['n_point'] <= row['n_query']:
                raise ValueError('invalid center metric population')
            if identity in populations and populations[identity] != current:
                raise ValueError('center population differs between configurations')
            populations[identity] = current


def seal_forward(out):
    lock = verify_lock(out)
    if (out / 'forward_complete.json').exists():
        raise ValueError('refusing forward seal overwrite')
    chain = forward_chain(out)
    all_records = []
    for scene in SCENES:
        directory = out / 'forward' / scene.replace('/', '__')
        records = read(directory / 'records.json')
        for entry in scene_entries(lock, scene):
            name = Path(entry['path']).name + '.npz'
            archive_hash = sha(directory / name)
            with np.load(directory / name, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            validate_arrays(arrays)
            matching = [r for r in records if r['pair'] == entry['pair']]
            for row in matching:
                if (row['archive'] != name or row['archive_sha256'] != archive_hash
                        or row['field'] != named_probability(row['family'], row['grid_index'], row['query'])
                        or row['array_sha256'] != array_sha(arrays[row['field']])
                        or row['shape'] != list(arrays[row['field']].shape) or row['dtype'] != 'float64'
                        or row['input_sha256'] != lock['inputs'][str(Path(entry['path']) / 'inputs.npz')]):
                    raise ValueError('forward array/record provenance mismatch')
        repeated = read(directory / 'determinism.json')
        first_pair = scene_entries(lock, scene)[0]['pair']
        first = {r['field']: r['array_sha256'] for r in records if r['pair'] == first_pair}
        if repeated['passed'] is not True or repeated['pair'] != first_pair or repeated['array_hashes'] != first:
            raise ValueError('first-pair determinism evidence mismatch')
        all_records.extend(records)
    verify_population(all_records, lock['target_contract']['entries'])
    write(out / 'forward_complete.json', dict(lock_sha256=sha(out / 'lock.json'),
        forward_chain=chain, array_count=len(all_records), raw_probability_bytes=RAW_PROBABILITY_BYTES,
        sealed_unix=time.time(), labels_decoded=False, all_probabilities_retained=True))


def verify_forward_complete(out):
    value = read(out / 'forward_complete.json')
    if (value['lock_sha256'] != sha(out / 'lock.json') or value['forward_chain'] != forward_chain(out)
            or value['array_count'] != 5184 or value['labels_decoded']):
        raise ValueError('global forward seal mismatch')
    return value


def load_center_targets(path):
    # Deliberately do not decode g*_counts or open sensor files here.
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for q in QUERIES
                for name in (q + '_point_xy', q + '_point_visible')}


def center_metric(probability, key_centers, point, valid):
    if probability.dtype != np.float64 or probability.ndim != 2 or len(key_centers) != probability.shape[1]:
        raise ValueError('invalid native prediction')
    if point.shape != (len(probability), 2) or valid.shape != (len(probability),) or valid.dtype != np.bool_:
        raise ValueError('invalid independent center target')
    if not np.isfinite(probability).all() or (probability < 0).any() or np.max(np.abs(probability.sum(-1)-1)) > 1.e-12:
        raise ValueError('invalid normalized probability')
    if not np.isfinite(point[valid]).all():
        raise ValueError('valid center is nonfinite')
    winner = np.argmax(probability, axis=-1)  # first native index wins ties, unchanged.
    distance2 = np.square(key_centers[winner][valid] - point[valid]).sum(-1)
    return dict(pck_count=int((distance2 <= 64.).sum()), n_point=int(valid.sum()),
                n_query=len(probability), argmax_sha256=array_sha(winner))


def evaluate(out, scene):
    lock = verify_lock(out)
    entries = scene_entries(lock, scene)
    seal = verify_forward_complete(out)  # Must precede every target hash/decode.
    directory = out / 'evaluation' / scene.replace('/', '__')
    if directory.exists():
        raise ValueError('refusing evaluation overwrite')
    started = time.time()
    if seal['sealed_unix'] > started:
        raise ValueError('forward-before-label chronology violated')
    directory.mkdir(parents=True)
    key_centers = S.read_parts(out)['g8']['mean'] * 8
    rows, label_sources = [], {}
    forward_dir = out / 'forward' / scene.replace('/', '__')
    records = read(forward_dir / 'records.json')
    for entry in entries:
        target = Path(entry['target_path'])
        if sha(target) != entry['target_sha256']:
            raise ValueError('independent center target changed')
        labels = load_center_targets(target)
        label_sources[str(target)] = entry['target_sha256']
        name = Path(entry['path']).name + '.npz'
        with np.load(forward_dir / name, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        validate_arrays(arrays)
        for record in (r for r in records if r['pair'] == entry['pair']):
            q = record['query']; probability = arrays[record['field']]
            if array_sha(probability) != record['array_sha256']:
                raise ValueError('probability changed before center evaluation')
            row = center_metric(probability, key_centers, labels[q + '_point_xy'], labels[q + '_point_visible'])
            rows.append(dict(**record, **row, target_sha256=entry['target_sha256']))
    verify_population(rows, entries)
    write(directory / 'rows.json', rows)
    complete_stage(directory, out, 'evaluation', scene, started, record_count=len(rows),
        forward_complete_sha256=sha(out / 'forward_complete.json'), label_sources=label_sources,
        center_labels_decoded=True, regional_labels_decoded=False)


def summarize_rows(rows, entries):
    verify_population(rows, entries)
    metrics, exact = {}, {}
    for family in FAMILIES:
        values, exact_values = [], []
        for i in range(12):
            scene_values, scene_fractions = {}, {}
            for scene in SCENES:
                scene_values[scene], scene_fractions[scene] = {}, {}
                for q in QUERIES:
                    subset = [r for r in rows if (r['family'], r['grid_index'], r['scene'], r['query']) == (family, i, scene, q)]
                    denominator = sum(r['n_point'] for r in subset)
                    if denominator <= 0:
                        raise ValueError('empty scene/stride center population')
                    fraction = Fraction(sum(r['pck_count'] for r in subset), denominator)
                    scene_fractions[scene][q] = fraction
                    scene_values[scene][q] = dict(pck8=float(fraction),
                                                 n_point=denominator, n_query=sum(r['n_query'] for r in subset))
            query_fractions = {q: sum(scene_fractions[s][q] for s in SCENES) / len(SCENES) for q in QUERIES}
            macro = sum(query_fractions.values()) / len(QUERIES)
            by_query = {q: float(value) for q, value in query_fractions.items()}
            values.append(dict(grid_index=i, config=GRID[i], by_scene_query=scene_values,
                               by_query=by_query, equal_scene_equal_scale_pck8=float(macro)))
            exact_values.append(dict(by_query=query_fractions, macro=macro))
        metrics[family] = values
        exact[family] = exact_values
    selections = {family: max(range(12), key=lambda i: exact[family][i]['macro'])
                  for family in ('gaussian_point', 'quartic_point')}
    references = dict(gaussian_point=( 'gaussian_point', selections['gaussian_point']),
                      quartic_point=('quartic_point', selections['quartic_point']),
                      legacy_point_pck=('gaussian_point', LEGACY_INDEX))
    feasible, constraints = [], []
    for i in range(12):
        checks = []
        for q in QUERIES:
            candidate = exact['quartic_full'][i]['by_query'][q]
            for name, (family, index) in references.items():
                reference = exact[family][index]['by_query'][q]
                checks.append(dict(query=q, reference=name, reference_family=family, reference_grid_index=index,
                    candidate_pck8=float(candidate), reference_pck8=float(reference),
                    difference_pp=float(100 * (candidate-reference)),
                    passed=candidate + PCK_EXACT_TOLERANCE >= reference))
        passed = all(row['passed'] for row in checks)
        constraints.append(dict(grid_index=i, config=GRID[i], feasible=passed, checks=checks))
        if passed:
            feasible.append(i)
    return dict(metrics=metrics, point_localization_selections={f: dict(grid_index=i, config=GRID[i])
                for f, i in selections.items()}, legacy_point=dict(grid_index=LEGACY_INDEX, config=GRID[LEGACY_INDEX]),
                feasibility=constraints, feasible_grid_indices=feasible, feasible=bool(feasible),
                decision_pending_independent_audit='complete_full_regional_calibration' if feasible else 'stop_unchanged_candidate',
                final_primary_selected=False, regional_calibration_done=False,
                scientific_gate_passed=False, fresh_test_read=False)


def summarize(out):
    lock = verify_lock(out)
    verify_forward_complete(out)
    if (out / 'summary.json').exists() or (out / 'completion.json').exists():
        raise ValueError('refusing summary overwrite')
    rows = []
    for scene in SCENES:
        directory = out / 'evaluation' / scene.replace('/', '__')
        completion = verify_stage(directory, out, 'evaluation', scene)
        if completion['forward_complete_sha256'] != sha(out / 'forward_complete.json'):
            raise ValueError('evaluation forward seal changed')
        rows.extend(read(directory / 'rows.json'))
    summary = summarize_rows(rows, lock['target_contract']['entries'])
    summary.update(lock_sha256=sha(out / 'lock.json'), forward_complete_sha256=sha(out / 'forward_complete.json'),
                   pair_count=48, row_count=len(rows), created_unix=time.time())
    write(out / 'summary.json', summary)
    files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    write(out / 'completion.json', dict(lock_sha256=sha(out / 'lock.json'), files=files,
        summary_sha256=sha(out / 'summary.json'), forward_complete_sha256=sha(out / 'forward_complete.json'),
        finished_unix=time.time(), center_labels_decoded=True, regional_labels_decoded=False,
        fresh_test_read=False, independent_audit_required=True))
    print('FEASIBLE CONFIGURATIONS', summary['feasible_grid_indices'], 'AUDIT PENDING', flush=True)


def audit_decision(out, verification_path):
    verify_lock(out)
    complete, summary, audit = read(out / 'completion.json'), read(out / 'summary.json'), read(verification_path)
    verify_files({str(out / path): digest for path, digest in complete['files'].items()})
    expected = dict(feasibility_lock_sha256=sha(out / 'lock.json'),
                    feasibility_forward_complete_sha256=sha(out / 'forward_complete.json'),
                    feasibility_summary_sha256=sha(out / 'summary.json'),
                    feasibility_completion_sha256=sha(out / 'completion.json'))
    if audit['passed'] is not True or any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError('independent feasibility audit/hash binding required')
    if audit['feasible_grid_indices'] != summary['feasible_grid_indices'] or audit['scientific_gate_passed'] is not False:
        raise ValueError('independent feasibility conclusion mismatch')
    result = dict(**expected, audit_path=str(verification_path.resolve()), audit_sha256=sha(verification_path),
        feasible_grid_indices=summary['feasible_grid_indices'],
        action='complete_full_regional_calibration' if summary['feasible'] else 'stop_unchanged_candidate',
        primary_selected=False, validation_authorized=False, scientific_gate_passed=False,
        regional_design=dict(families=8, configurations_per_family=12, conditions=14, unchanged_J=True),
        created_unix=time.time())
    write(out / 'decision.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'forward', 'seal-forward', 'evaluate', 'summarize', 'accept-audit'))
    parser.add_argument('--out', type=Path, default=ROOT / 'results/round15_feasibility')
    parser.add_argument('--data', type=Path, default=ROOT / 'results/round10_partition_data')
    parser.add_argument('--targets', type=Path, default=ROOT / 'results/round13_target_conflict')
    parser.add_argument('--target-audit', type=Path, default=ROOT / 'results/round13_target_independent_audit/verification.json')
    parser.add_argument('--numerical-root', type=Path, default=NUMERICAL_ROOT)
    parser.add_argument('--forward-preflight', type=Path, default=FORWARD_PREFLIGHT_ROOT)
    parser.add_argument('--cost-report', type=Path, default=COST_REPORT)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--scene', choices=SCENES)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.stage == 'freeze':
        if any(v is None for v in (args.numerical_root, args.forward_preflight, args.cost_report)):
            parser.error('freeze requires --numerical-root, --forward-preflight and --cost-report')
        freeze(args.out, args.data, args.targets, args.target_audit, args.numerical_root,
               args.forward_preflight, args.cost_report)
    elif args.stage == 'forward':
        forward(args.out, args.scene, args.device)
    elif args.stage == 'seal-forward':
        seal_forward(args.out)
    elif args.stage == 'evaluate':
        evaluate(args.out, args.scene)
    elif args.stage == 'summarize':
        summarize(args.out)
    else:
        if args.audit is None:
            parser.error('accept-audit requires --audit')
        print(json.dumps(audit_decision(args.out, args.audit), indent=2))


if __name__ == '__main__':
    main()
