"""Independent complete replay/target/numerical audit of R17 old validation.

Preparation/operator replay is explicitly shared with the locked producer.
Targets, metrics, aggregation, gates and fixed numerical references are independent.
Freeze reads source and accepted metadata only; audit requires the complete
global forward seal before it can hash or decode any validation label bytes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from experiments import round15_regional_audit as H
from experiments import round15_regional_audit_metrics as REGION
from experiments import round17_validation_audit_metrics as M
from experiments import round17_validation_audit_numeric as N
from experiments import round17_regional_audit as UPSTREAM
from experiments import round17_regional_audit_path_repair as REPAIR
from experiments import round12_stress_result_audit as SENSOR


ROOT = Path(__file__).resolve().parents[1]
SCENES = ('harvard_c8/hv_c8_3', 'brown_bm_4/brown_bm_4')
TRAIN_SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
TIMING_SCENES = (*TRAIN_SCENES, *SCENES)
FAMILIES = ('trace_full', 'quartic_point', 'gaussian_point', 'gaussian_fixed',
            'gaussian_full1', 'gaussian_full4', 'gaussian_quad4', 'gaussian_quad16')
LEGACY = dict(point=('gaussian_point', 1., 5.), point_pck=('gaussian_point', .5, 5.),
              fixed=('gaussian_fixed', .5, 5.), full1=('gaussian_full1', .25, 0.),
              full4=('gaussian_full4', .5, 5.), quad4=('gaussian_quad4', .5, 5.),
              quad16=('gaussian_quad16', .5, 5.))
GRID = tuple(dict(h=h, temperature=t) for h in (.25, .5, 1., 2.) for t in (0., 5., 10.))
NUMERICAL_POLICY = dict(gaussian_quadrature_order=32, probability_dtype='float64',
    preparation_dtype='float32', content_bias_dtype='float32', temperature_formula_dtype='float64',
    floor=1e-30, quartic_control_quadrature_order=24)
CONDITIONS = tuple(dict(scenario=s, state=state, query=q, key=k, condition=q+'_'+k)
                   for s, state, q, k in N.CONDITIONS)
LABEL_FIELDS = ('gt_xy', 'gt_visible', *(q+'_point_'+f for q in N.PARTITIONS for f in ('xy', 'visible', 'status')))
OWN_SOURCES = ('ROUND17_VALIDATION_PROTOCOL.md', 'docs/round17_validation_implementation_plan.md',
    'experiments/round17_validation_audit.py', 'tests/test_round17_validation_audit.py',
    'experiments/round17_validation_audit_metrics.py', 'tests/test_round17_validation_audit_metrics.py',
    'experiments/round17_validation.py', 'experiments/round17_validation_metrics.py',
    'models/round17_validation_pe.py', 'models/round15_validation_pe.py',
    'tests/test_round17_validation_pe.py')
sha, read, write, array_sha, verify_files, compare_tree = (H.sha, H.read, H.write, H.array_sha, H.verify_files, H.compare_tree)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def source_files():
    return {p:sha(ROOT/p) for p in sorted(set(OWN_SOURCES).union(N.SOURCE_NAMES, H.OWN_SOURCES, UPSTREAM.OWN_SOURCES,
        UPSTREAM.H.SOURCE_NAMES, REPAIR.REPAIR_SOURCES))}


def assemble_roles(selected):
    """Reconstruct first-occurrence aliases from accepted metadata, no retuning."""
    selection = selected['selection']
    if set(selection['controls']) != set(FAMILIES[1:]):
        raise ValueError('regional control family population changed')
    specs = {}
    def add(alias, spec, family):
        cfg = spec['config']
        if (spec['family'] != family or cfg not in GRID
                or spec['grid_index'] != GRID.index(cfg)):
            raise ValueError('invalid accepted family/grid identity')
        specs[alias] = dict(family=family, **cfg)
    add('primary', selection['primary'], 'trace_full')
    if selection['primary']['grid_index'] not in selection['eligible_indices']:
        raise ValueError('accepted primary is outside center-feasible set')
    for family in FAMILIES[1:]:
        add('regional_'+family, selection['controls'][family], family)
    for family in ('gaussian_point', 'quartic_point'):
        add('localizer_'+family, dict(family=family, **selected['point_localization_selections'][family]), family)
    for alias, (family, h, temperature) in LEGACY.items():
        prior = selected['legacy_configurations'][alias]
        add('legacy_'+alias, prior, family)
        if specs['legacy_'+alias] != dict(family=family, h=h, temperature=temperature):
            raise ValueError('legacy protected configuration changed')
    specs['matched_constant_temperature'] = dict(specs['primary'], family='gaussian_full1')
    specs['matched_isotropic'] = dict(specs['primary'], family='trace_isotropic_matched')
    aliases, operators, seen = {}, {}, {}
    for alias, cfg in specs.items():
        identity = cfg['family'], cfg['h'], cfg['temperature']
        if identity not in seen:
            key = f'o{len(seen):02d}'
            seen[identity] = key
            operators[key] = cfg
        aliases[alias] = seen[identity]
    for kind, choices in (('strong_point', ('gaussian_point', 'quartic_point')),
                          ('strong_quadrature', ('gaussian_quad4', 'gaussian_quad16'))):
        chosen = selection[kind]
        if chosen['family'] not in choices or chosen != selection['controls'][chosen['family']]:
            raise ValueError('strong reference is not the accepted regional control')
    N.validate_operators(operators)
    return dict(aliases=aliases, operators=operators,
        strong_point='regional_'+selection['strong_point']['family'],
        strong_quad='regional_'+selection['strong_quadrature']['family'], primary='primary')


def verify_regional(producer):
    """Verify accepted Stage B metadata; never decode a calibration probability."""
    prior = producer['regional']; directory = absolute(prior['root'])
    lock = read(directory/'lock.json'); selected = read(directory/'selection.json')
    summary = read(directory/'summary.json'); complete = read(directory/'completion.json')
    if (sha(directory/'lock.json') != (directory/'lock.sha256').read_text().strip()
            or sha(directory/'selection.json') != (directory/'selection.sha256').read_text().strip()
            or sha(directory/'selection.json') != prior['selection_sha256']):
        raise ValueError('accepted Stage B lock/selection hash changed')
    verify_files(lock['source_hashes']); verify_files(lock['source_hashes'], directory/'source_snapshot')
    audit_path = absolute(prior['audit_path']); audit = read(audit_path)
    if selected['audit_path'] != prior['audit_path'] or sha(audit_path) != prior['audit_sha256'] or selected['audit_sha256'] != sha(audit_path):
        raise ValueError('accepted Stage B independent audit changed')
    for name in ('lock', 'forward_complete', 'summary', 'completion'):
        digest = sha(directory/(name+'.json')); field = f'regional_{name}_sha256'
        if prior[field] != digest or audit[field] != digest or selected[field] != digest:
            raise ValueError('accepted Stage B artifact binding changed: '+name)
    if (complete['lock_sha256'] != sha(directory/'lock.json')
            or complete['summary_sha256'] != sha(directory/'summary.json')
            or complete['forward_complete_sha256'] != sha(directory/'forward_complete.json')
            or audit['passed'] is not True or audit['scientific_gate_passed'] is not False
            or selected['selection'] != summary['selection'] or audit['selection'] != summary['selection']
            or selected['point_localization_selections'] != lock['stage_a']['point_localization_selections']
            or selected['legacy_configurations'] != lock['legacy']['configurations']
            or selected['train_screen'] != summary['train_screen'] or audit['train_screen'] != summary['train_screen']
            or prior['train_screen'] != summary['train_screen']
            or summary['train_screen']['passed'] is not True
            or summary['train_screen']['fixed_primary'] != summary['selection']['primary']
            or summary['train_screen']['next_action'] != 'prepare_old_validation_protocol'
            or selected['next_action'] != 'prepare_old_validation_protocol'
            or selected['validation_run'] or selected['scientific_gate_passed']):
        raise ValueError('independently accepted Stage B contract required')
    # Its own independently accepted audit already bound every calibration
    # array; the new audit binds those immutable completion/audit commitments.
    audit_root = audit_path.parent
    if sha(audit_root/'lock.json') != audit['independent_audit_lock_sha256']:
        raise ValueError('Stage B audit design changed')
    audit_lock = read(audit_root/'lock.json')
    # R17 froze this auditor in the producer's joint source_snapshot. The later
    # execution lock has no fabricated second snapshot directory.
    if (audit_lock['joint_source_lock_sha256'] != sha(directory/'lock.json')
            or audit_lock['original_design_created_unix'] != lock['created_unix']
            or any(lock['source_hashes'].get(n) != h for n, h in audit_lock['source_hashes'].items())):
        raise ValueError('Stage B joint pre-forward audit design changed')
    verify_files(audit_lock['source_hashes']); verify_files(audit_lock['source_hashes'], directory/'source_snapshot')
    audit_complete = read(audit_root/'completion.json')
    if audit_complete['files'][audit_path.name] != sha(audit_path) or audit_complete['passed'] is not True:
        raise ValueError('Stage B audit completion missing')
    verify_files(audit_complete['files'], audit_root)
    verify_regional_repair(prior, directory, lock, audit_path, audit_lock, audit, selected)
    # Independent accepted TRAIN/source/numerical chain, including arithmetic
    # extension; the corrected R12 policy verifies only its real three snapshots.
    _, _, old_summary = UPSTREAM.verify_previous(lock)
    REPAIR.verify_legacy(lock, old_summary)
    roles = assemble_roles(selected)
    if roles != prior['roles'] or roles != producer['roles']:
        raise ValueError('validation roles differ from accepted selections')
    return lock, roles


def verify_regional_repair(prior, directory, regional_lock, audit_path, audit_lock, audit, selected):
    """Bind the honest post-freeze snapshot-policy repair and unchanged design."""
    repair_root = audit_path.parent.parent
    repaired = REPAIR.verify_repair(repair_root)
    repair_completion = read(repair_root/'completion.json')
    digest = sha(repair_root/'lock.json')
    expected_receipt = dict(repair_lock_sha256=digest, repair_frozen_unix=repaired['created_unix'],
        repair_source_hashes=repaired['source_hashes'],
        original_joint_lock_sha256=repaired['original_contract']['joint_lock_sha256'],
        path_repair_scope=repaired['scope'], path_repair_frozen_before_regional_forward=False,
        original_failure_bound_files=repaired['original_contract']['bound_files'])
    if (audit.get('path_repair') != expected_receipt or audit_lock.get('path_repair') != expected_receipt
            or repaired['original_contract']['run'] != str(directory.resolve())
            or repaired['original_contract']['joint_lock_sha256'] != sha(directory/'lock.json')
            or repaired['original_contract']['original_source_hashes'] != regional_lock['source_hashes']
            or repaired['original_contract']['original_design_created_unix'] != regional_lock['created_unix']
            or repaired['audit_output'] != str(audit_path.parent.resolve())
            or repair_completion['passed'] is not True or repair_completion['scientific_gate_passed']
            or repair_completion['repair_lock_sha256'] != digest
            or repair_completion['verification_path'] != str(audit_path.resolve())
            or repair_completion['verification_sha256'] != sha(audit_path)
            or not repaired['created_unix'] <= audit['started_unix'] <= audit['finished_unix']
                    <= repair_completion['finished_unix'] <= selected['created_unix']):
        raise ValueError('regional snapshot repair source/acceptance chronology changed')
    actual = {str(p.relative_to(repair_root)) for p in repair_root.rglob('*')
              if p.is_file() and p != repair_root/'completion.json'}
    if set(repair_completion['files']) != actual:
        raise ValueError('regional repair completion file population changed')
    verify_files(repair_completion['files'], repair_root)
    # The new producer must explicitly commit the repair, not just its nested
    # successful verification. Exact field names are shared with its contract.
    expected = dict(root=str(repair_root.resolve()), lock_sha256=digest,
        completion_sha256=sha(repair_root/'completion.json'),
        audit_completion_sha256=sha(audit_path.parent/'completion.json'),
        source_hashes=repaired['source_hashes'], scope=repaired['scope'], frozen_before_regional_forward=False)
    if prior['audit_repair'] != expected:
        raise ValueError('validation did not bind the accepted engineering repair')
    return expected


def verify_target_metadata(producer):
    """Bind the old independently checked label archive without opening bytes."""
    target = producer['targets']; directory = absolute(target['root'])
    lock = read(directory/'lock.json'); completion = read(directory/'completion.json')
    audit = read(absolute(target['audit_path']))
    if (sha(directory/'lock.json') != target['lock_sha256']
            or sha(directory/'lock.json') != (directory/'lock.sha256').read_text().strip()
            or sha(directory/'completion.json') != target['completion_sha256']
            or sha(absolute(target['audit_path'])) != target['audit_sha256']
            or audit['passed'] is not True or audit['scientific_gate_passed'] is not False
            or audit['stress_lock_sha256'] != target['lock_sha256']
            or audit['stress_completion_sha256'] != target['completion_sha256']
            or audit['stress_summary_sha256'] != completion['files']['summary.json']
            or completion['lock_sha256'] != target['lock_sha256']
            or target['old_scientific_gate_passed'] is not False or target['target_archive_bytes_read'] is not False):
        raise ValueError('old target audit/completion binding changed')
    verify_files(lock['source_hashes'])
    REPAIR.verify_historical_snapshots(directory, lock)
    old_design = ROOT/'results/round12_stress_audit_source_lock.json'
    if sha(old_design) != target['audit_source_lock_sha256'] or audit['audit_source_lock_sha256'] != sha(old_design):
        raise ValueError('old target independent audit design changed')
    design = read(old_design); verify_files(design['source_hashes'])
    if sha(absolute(design['snapshot'])) != design['snapshot_sha256']:
        raise ValueError('old target audit snapshot changed')
    verify_files(target['metadata_files'])
    entries = target['entries']
    ids = [(e['scene'], tuple(e['pair'])) for e in entries]
    if (entries != producer['entries'] or len(ids) != 32 or len(set(ids)) != 32
            or set(s for s, _ in ids) != set(SCENES)
            or any(sum(s == scene for s, _ in ids) != 16 for scene in SCENES)):
        raise ValueError('old validation cohort changed')
    inherited = {(e['scene'], tuple(e['pair'])):e for e in lock['entries'] if e['scene'] in SCENES}
    if set(inherited) != set(ids):
        raise ValueError('cohort is not exactly the old stress population')
    for scene in SCENES:
        stage = directory/'evaluation'/scene.replace('/', '__')
        stage_complete = read(stage/'completion.json')
        provenance = read(stage/'label_provenance.json')
        provenance_map = {tuple(r['pair']):r for r in provenance}
        if (len(provenance_map) != 16 or len(provenance) != 16
                or completion['files'][str((stage/'completion.json').relative_to(directory))] != sha(stage/'completion.json')
                or stage_complete['files']['label_provenance.json'] != sha(stage/'label_provenance.json')):
            raise ValueError('old target metadata population/chain changed')
        for entry in (e for e in entries if e['scene'] == scene):
            source = absolute(entry['path']); original = inherited[(scene, tuple(entry['pair']))]
            if absolute(original['path']) != source or sha(source/'manifest.json') != entry['manifest_sha256']:
                raise ValueError('source manifest identity changed')
            meta = read(source/'manifest.json'); prov = provenance_map[tuple(entry['pair'])]
            expected_path = stage/'labels'/(source.name+'.npz')
            digest = stage_complete['files'][str(expected_path.relative_to(stage))]
            if (meta['scene'] != scene or meta['pair'] != entry['pair'] or meta['split'] != 'val'
                    or meta['forward_uses_sensor'] or meta['input_sha256'] != entry['input_sha256']
                    or meta['label_sha256'] != entry['old_label_sha256']
                    or prov != entry['target_provenance'] or prov['scene'] != scene
                    or prov['old_labels_sha256'] != meta['label_sha256']
                    or prov['sensor_sha256'] != meta['sensor_sources']
                    or prov['source_selection_sha256'] != meta['source_selection_sha256']
                    or prov['projected_centroid_used']
                    or absolute(entry['target_path']) != expected_path or entry['target_sha256'] != digest
                    or completion['files'][str(expected_path.relative_to(directory))] != digest):
                raise ValueError('old target pair/provenance/expected hash changed')
    return entries


def freeze(out, run):
    if out.exists():
        raise ValueError('refusing independent validation audit overwrite')
    run = run.resolve()
    if (run/'forward').exists() or (run/'forward_complete.json').exists():
        raise ValueError('independent design must be frozen before validation forward')
    producer = read(run/'lock.json')
    if sha(run/'lock.json') != (run/'lock.sha256').read_text().strip():
        raise ValueError('producer lock mismatch')
    verify_files(producer['source_hashes']); verify_files(producer['source_hashes'], run/'source_snapshot')
    _, roles = verify_regional(producer); verify_target_metadata(producer)
    sources = source_files(); now = time.time()
    if any(producer['source_hashes'].get(n) != h for n, h in sources.items()):
        raise ValueError('producer joint source closure omits independent/shared replay dependency')
    if now < producer['created_unix']:
        raise ValueError('producer/audit chronology is invalid')
    count = len(roles['operators'])
    out.mkdir(parents=True)
    for name in sources:
        destination = out/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    write(out/'lock.json', dict(created_unix=now, run=str(run), source_hashes=sources,
        producer_sources=producer['source_hashes'], producer_lock_sha256=sha(run/'lock.json'),
        expected_pairs=32, expected_rows=32*68*count, expected_retained_arrays=2176+136*(count-1),
        expected_numerical_matrices=136*count, expected_geometry_records=224,
        expected_center_labels=32*(432+108+243+48), expected_timing_rows=45*count,
        expected_replayed_arrays=2176*count, expected_attribution_supports=704,
        expected_attribution_queries=176928, shared_production_operator_replay=True,
        independent_forward_implementation=False,
        validation_effects_read=False, validation_labels_decoded=False, scientific_gate_passed=False,
        metric_sum_tolerance=1e-8, summary_tolerance=1e-10, probability_tolerance=1e-4,
        reference_convergence_tolerance=1e-8,
        coverage='All 2176U probabilities share-operator replayed and independently measured; all targets, projections, centers, aggregation/timing/gates; 136U independent numerical matrices.'))
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def verify_stage(run, directory, stage, scene, producer):
    complete = read(directory/'completion.json')
    files = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'completion.json'}
    if (complete['stage'] != stage or complete['scene'] != scene
            or complete['lock_sha256'] != sha(run/'lock.json') or complete['runtime'] != producer['runtime']
            or set(complete['files']) != files):
        raise ValueError('stage source/runtime/artifact identity changed')
    verify_files(complete['files'], directory)
    return complete


def identity(row):
    return row['scene'], tuple(row['pair']), row['operator'], row['scenario'], row['condition']


def check_numerical_diagnostics(value, shape, key_mass):
    expected = {'raw_zero_cells','raw_zero_fov_rows','raw_tiny_fov_rows','raw_fov_min','raw_fov_max',
                'raw_floor_fov_mass','all_finite','max_normalization_error'}
    if set(value) != expected:
        raise ValueError('raw numerical diagnostic fields changed')
    nq, nk = shape
    counts = [value[name] for name in ('raw_zero_cells','raw_zero_fov_rows','raw_tiny_fov_rows')]
    if (any(type(v) is not int for v in counts) or not 0 <= counts[0] <= nq*nk
            or not 0 <= counts[1] <= counts[2] <= nq or counts[0] < counts[1]*nk
            or not np.isfinite([value['raw_fov_min'],value['raw_fov_max'],value['raw_floor_fov_mass'],
                                value['max_normalization_error']]).all()
            or not 0 <= value['raw_fov_min'] <= value['raw_fov_max']
            or value['raw_floor_fov_mass'] != float(1e-30*(np.asarray(key_mass,dtype=float)/64).sum())
            or value['all_finite'] is not True or not 0 <= value['max_normalization_error'] <= 1e-12):
        raise ValueError('invalid raw FOV/floor/normalization diagnostics')


def validate_forward_rows(rows, entries, roles, parts):
    expected = {(e['scene'], tuple(e['pair']), op, c['scenario'], c['condition'])
                for e in entries for op in roles['operators'] for c in CONDITIONS}
    mapping = {identity(row):row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != expected:
        raise ValueError('missing/duplicate/unexpected committed forward row')
    entries_by_id = {(e['scene'], tuple(e['pair'])):e for e in entries}
    specs = {(c['scenario'], c['condition']):c for c in CONDITIONS}
    for row in rows:
        spec = specs[row['scenario'], row['condition']]
        entry = entries_by_id[row['scene'], tuple(row['pair'])]
        if (any(row[k] != v for k, v in spec.items()) or row['config'] != roles['operators'][row['operator']]
                or row['shape'] != [len(parts[row['query']]['mass']), len(parts[row['key']]['mass'])]
                or row['dtype'] != 'float64' or row['input_sha256'] != entry['input_sha256']
                or len(row['array_sha256']) != 64 or len(row['argmax_sha256']) != 64
                or row['numerical']['all_finite'] is not True
                or not 0 <= row['numerical']['max_normalization_error'] <= 1e-12):
            raise ValueError('committed forward configuration/shape/source/numerics changed')
        check_numerical_diagnostics(row['numerical'], row['shape'], parts[row['key']]['mass'])
    return mapping


def check_measured_row(forward, row, entry, primary, direct, labels):
    if any(row.get(k) != v for k, v in forward.items()):
        raise ValueError('evaluation changed a committed forward record')
    if (row['replayed_array_sha256'] != forward['array_sha256'] or row['target_sha256'] != entry['target_sha256']
            or row['probability_source'] != ('retained_primary' if row['operator'] == primary else 'exact_control_replay')
            or row['n_query'] != forward['shape'][0] or row['n_key'] != forward['shape'][1]
            or row['mass'] != int(direct.sum())
            or row['n_point'] != int(labels[row['query']+'_point_visible'].sum())):
        raise ValueError('evaluation replay/label/population mismatch')


def check_probability(probability, record, labels, parts, direct, common):
    q, k = record['query'], record['key']
    if (probability.dtype != np.float64 or list(probability.shape) != record['shape']
            or not np.isfinite(probability).all() or np.any(probability < 0)
            or np.max(np.abs(probability.sum(-1)-1)) > 1e-12
            or array_sha(probability) != record['array_sha256']
            or array_sha(probability.argmax(-1).astype(np.int32)) != record['argmax_sha256']):
        raise ValueError('retained native probability/hash/argmax mismatch')
    metric = REGION.metric_reference(probability, direct, common, parts[q], parts[k],
                                    'scale' if record['scenario'] == 'scale' else 'partition')
    centers = (parts[k]['bounds'][:, :2]+parts[k]['bounds'][:, 2:]-1)/2
    correct = (np.linalg.norm(centers[probability.argmax(-1)]-labels[q+'_point_xy'], axis=-1) <= 8)
    metric.update(pck_count=int((correct & labels[q+'_point_visible']).sum()),
                  n_point=int(labels[q+'_point_visible'].sum()))
    return metric


def check_replayed_probability(probability, diagnostics, record):
    """Require the actual complete operator replay, including unsaved controls."""
    value = np.asarray(probability)
    if (value.dtype != np.float64 or list(value.shape) != record['shape']
            or not np.isfinite(value).all() or np.any(value < 0)
            or np.max(abs(value.sum(-1)-1)) > 1.e-12
            or array_sha(value) != record['array_sha256']
            or array_sha(value.argmax(-1).astype(np.int32)) != record['argmax_sha256']
            or diagnostics != record['numerical']):
        raise ValueError('complete shared operator replay/hash/diagnostic mismatch')
    return value


def _numpy(value):
    return value.detach().cpu().numpy()


def trace_attribution(query_banks, roles, trace_temperature):
    """Check all actual prepared queries, transparently sharing production math."""
    cfg = roles['operators'][roles['aliases']['primary']]
    iso_cfg = roles['operators'][roles['aliases']['matched_isotropic']]
    if (cfg['family'] != 'trace_full' or iso_cfg != dict(cfg, family='trace_isotropic_matched')):
        raise ValueError('trace attribution roles/configurations changed')
    output = []
    for name, bank in query_banks.items():
        full, iso = bank['trace_full'], bank['trace_isotropic_matched']
        cov, original, iso_original, iso_cov = map(_numpy,
            (full['cov'], full['tau_cov'], iso['tau_cov'], iso['cov']))
        expected_iso = np.trace(original.astype(np.float64), axis1=-2, axis2=-1)[:, None, None]*np.eye(2)/2
        if (cov.dtype != np.float32 or original.dtype != np.float32 or iso_original.dtype != np.float32
                or iso_cov.dtype != np.float64 or not np.array_equal(cov, original)
                or not np.array_equal(original, iso_original) or not np.array_equal(iso_cov, expected_iso)
                or not np.array_equal(_numpy(full['mean']), _numpy(iso['mean']))
                or not np.array_equal(_numpy(full['features']), _numpy(iso['features']))):
            raise ValueError('matched isotropic role changed original full covariance/content')
        tau = _numpy(trace_temperature(full['tau_cov'], cfg['h'], cfg['temperature']))
        iso_tau = _numpy(trace_temperature(iso['tau_cov'], cfg['h'], cfg['temperature']))
        if tau.dtype != np.float32 or iso_tau.dtype != np.float32 or not np.array_equal(tau, iso_tau):
            raise ValueError('matched isotropic effective tau is not bitwise identical')
        output.append(dict(query=name, query_count=len(cov), config=dict(cfg),
            full_covariance_sha256=array_sha(original), isotropic_original_covariance_sha256=array_sha(iso_original),
            isotropic_geometry_covariance_sha256=array_sha(iso_cov), tau32_sha256=array_sha(tau),
            isotropic_tau32_sha256=array_sha(iso_tau), maximum_tau_difference=0.,
            same_original_covariance=True, tau32_exact=True, shared_production_replay=True,
            independent_support_preparation=False))
    return output


def prepared_numeric_reference(backend, queries, keys, config, record):
    """Recompute actual intermediates, never pretend that they were stored.

    The independent numerical module receives NumPy arrays and imports no
    production arithmetic. Its same-mass and integration checks are separate.
    """
    import torch
    family = config['family']
    if family not in ('trace_full', 'trace_isotropic_matched'):
        raise ValueError('prepared trace reference requested for a non-trace control')
    query, key = queries[family], keys[family]
    with torch.no_grad():
        probability, diagnostics = backend.predict_prepared(queries, keys, config)
        probability = check_replayed_probability(probability, diagnostics, record)
        # Match the complete original content GEMM and integral chunking before
        # taking the eight fixed rows. No subset GEMM or geometric shortcut.
        content = query['features'] @ key['features'].T
        raw = backend.P.OLD._gaussian_mass(query['mean'], query['cov'].double(), key, config['h'])
        tau = backend.P.TRACE.trace_temperature(query['tau_cov'], config['h'], config['temperature'])
        bias = backend.P.TRACE.content_bias(query['tau_cov'], config['h'], config['temperature'], content)
    indices = np.linspace(0, len(probability)-1, 8, dtype=np.int64)
    return dict(indices=indices, mean=_numpy(query['mean'])[indices], cov=_numpy(query['cov'])[indices],
        tau_cov=_numpy(query['tau_cov'])[indices], content=_numpy(content)[indices], raw=_numpy(raw)[indices],
        tau32=_numpy(tau)[indices], bias32=_numpy(bias)[indices], probability=probability[indices],
        full_probability_sha256=array_sha(probability), committed_probability_sha256=record['array_sha256'],
        recomputed_not_original_intermediates=True)


def load_inputs(entry):
    path = absolute(entry['path'])/'inputs.npz'
    if sha(path) != entry['input_sha256']:
        raise ValueError('forward input archive changed')
    with np.load(path, allow_pickle=False) as archive:
        return {k:archive[k] for k in N.FIELDS}


def load_labels(entry):
    old_path = absolute(entry['path'])/'labels.npz'; target = absolute(entry['target_path'])
    verify_files({str(old_path):entry['old_label_sha256'], str(target):entry['target_sha256']})
    with np.load(old_path, allow_pickle=False) as old, np.load(target, allow_pickle=False) as saved:
        added = {'g24_point_'+f for f in ('xy', 'visible', 'status')}
        if set(saved.files) != set(old.files) | added:
            raise ValueError('old target archive fields changed')
        labels = {k:saved[k] for k in saved.files}
        for key in old.files:
            np.testing.assert_array_equal(labels[key], old[key])
    return labels


def check_sensor_labels(meta, labels, provenance, parts):
    """Decode the two fixed sensors once; replay all native centers independently."""
    from experiments.round6_region_data import read_scene_selection
    selection = absolute(meta['source_selection'])
    verify_files({str(selection):meta['source_selection_sha256'], **meta['camera_files'],
                  **meta['source_index_files'], **meta['sensor_sources']})
    rows = [r for r in read(selection)['scenes'] if r['scene'] == meta['scene']]
    if len(rows) != 1:
        raise ValueError('ambiguous sensor scene')
    frames = read_scene_selection(selection.parent, rows[0]); qi, ki = meta['pair']; q, k = frames[qi], frames[ki]
    if (set(meta['sensor_sources']) != {str(q.depth_path), str(k.depth_path)}
            or provenance['sensor_sha256'] != meta['sensor_sources']
            or provenance['query_rgb_index'] != qi or provenance['key_rgb_index'] != ki
            or provenance['query_sensor'] != str(q.depth_path) or provenance['key_sensor'] != str(k.depth_path)):
        raise ValueError('sensor timestamp/camera/point provenance changed')
    qdepth, kdepth = SENSOR.decode_sensor(q.depth_path), SENSOR.decode_sensor(k.depth_path)
    counts = {}
    for name, part in parts.items():
        centers = (part['bounds'][:, :2]+part['bounds'][:, 2:]-1)/2
        xy, visible, status = SENSOR.center_reference(centers, qdepth, kdepth, q.K_native, k.K_native, q.c2w, k.c2w)
        np.testing.assert_allclose(labels[name+'_point_xy'], xy, atol=1e-10, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(labels[name+'_point_visible'], visible)
        np.testing.assert_array_equal(labels[name+'_point_status'], status)
        counts[name] = len(centers)
    return counts


def check_projection(saved, reference, scene, pair):
    expected = dict(scene=scene, pair=pair, **reference)
    for key in ('zero_float32_xy_equal', 'zero_float32_cov_equal'):
        expected.pop(key, None)  # Additional audit diagnostic; no hidden new gate.
    compare_tree(saved, expected, tolerance=1e-12)


def retained_path(entry, condition):
    return Path('forward')/entry['scene'].replace('/', '__')/'arrays'/Path(entry['path']).name/(condition['scenario']+'__'+condition['condition']+'.npz')


def expected_execution_order(entries, roles):
    opids = list(roles['operators']); output = []
    for scene_index, entry in enumerate(entries):
        for condition_index, (q, k) in enumerate(N.ORIGINAL):
            for repeat in range(18):
                start = (scene_index*9+condition_index+repeat) % len(opids)
                output.extend(dict(scene=entry['scene'], condition=q+'_'+k, repeat=repeat, operator=op)
                              for op in opids[start:]+opids[:start])
    return output


def check_timing_sources(rows, entries, roles):
    by_scene = {entry['scene']:entry for entry in entries}
    for row in rows:
        entry = by_scene[row['scene']]
        if (row['pair'] != entry['pair'] or row['input_sha256'] != entry['input_sha256']
                or row['config'] != roles['operators'][row['operator']]):
            raise ValueError('timed operator/input identity changed')


def check_numerical_policy(producer, regional_lock):
    if (producer['numerical_policy'] != dict(regional_lock['numerical_policy'], quartic_control_quadrature_order=24)
            or producer['numerical_policy'] != NUMERICAL_POLICY):
        raise ValueError('fixed Gaussian GL32/quartic-control GL24/dtype/floor policy changed')


def decision_fields(numeric_passed, gates):
    if type(numeric_passed) is not bool or type(gates['passed']) is not bool:
        raise ValueError('audit and metric gate decisions must be explicit booleans')
    return dict(passed=numeric_passed, metric_gates_passed=gates['passed'],
                scientific_gate_passed=numeric_passed and gates['passed'])


def verify_chronology(producer, design, seal, forwards, evaluations, timing):
    if set(forwards) != set(SCENES) or set(evaluations) != set(SCENES):
        raise ValueError('both complete scenes are required')
    for scene in SCENES:
        f, e = forwards[scene], evaluations[scene]
        if not (producer['created_unix'] <= design['created_unix'] <= f['started_unix']
                <= f['finished_unix'] <= seal['sealed_unix'] <= e['started_unix'] <= e['finished_unix']):
            raise ValueError('audit/forward/seal/label chronology changed')
    if not max(e['finished_unix'] for e in evaluations.values()) <= timing['started_unix'] <= timing['finished_unix']:
        raise ValueError('timing ran before evaluation completed')


def verify_zero_preflight(producer, regional_lock):
    saved = producer['zero_path_preflight']; rows = []
    for scene in TRAIN_SCENES:
        entry = min((e for e in regional_lock['entries'] if e['scene'] == scene), key=lambda e:Path(e['path']).name)
        inputs = load_inputs(entry); raw, diagnostics = N.reference_observations(inputs, 'zero')
        fields = {}
        for name in N.REF.FIELDS:
            cached, reprojected = np.asarray(inputs[name], np.float32), np.asarray(raw[name], np.float32)
            fields[name] = dict(cached_sha256=array_sha(cached), reprojected_sha256=array_sha(reprojected),
                                exactly_equal=bool(np.array_equal(cached, reprojected)))
        rows.append(dict(scene=scene, pair=entry['pair'], input_sha256=entry['input_sha256'], fields=fields,
            zero_xy_max_error=diagnostics['zero_xy_max_error'], zero_cov_max_error=diagnostics['zero_cov_max_error']))
    expected = dict(passed=all(v['exactly_equal'] for r in rows for v in r['fields'].values()),
                    rows=rows, old_training_only=True, labels_decoded=False)
    compare_tree(saved, expected)
    if expected['passed'] is not True:
        raise ValueError('declared three-training-pair zero preflight failed')
    return expected


def audit(out):
    started = time.time()
    if (out/'verification.json').exists() or (out/'completion.json').exists():
        raise ValueError('refusing independent audit result overwrite')
    design = read(out/'lock.json')
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip() or design['source_hashes'] != source_files():
        raise ValueError('independent audit lock/source changed')
    verify_files(design['source_hashes'], out/'source_snapshot')
    run = absolute(design['run']); producer = read(run/'lock.json')
    if (sha(run/'lock.json') != design['producer_lock_sha256']
            or sha(run/'lock.json') != (run/'lock.sha256').read_text().strip()
            or producer['source_hashes'] != design['producer_sources']):
        raise ValueError('producer source/lock changed')
    verify_files(producer['source_hashes']); verify_files(producer['source_hashes'], run/'source_snapshot')
    regional_lock, roles = verify_regional(producer)
    check_numerical_policy(producer, regional_lock)
    entries = verify_target_metadata(producer)
    operators = roles['operators']; primary = roles['aliases']['primary']; count = len(operators)
    expected_bytes = 1112516352+69532272*(count-1)
    if (producer['conditions'] != list(CONDITIONS) or producer['field_allowlist'] != list(N.FIELDS)
            or producer['label_allowlist'] != list(LABEL_FIELDS)
            or producer['retained_probability_bytes'] != expected_bytes
            or producer['primary_all_arrays_retained'] is not True
            or producer['new_q16_protection'] is not True or producer['matched_isotropic_attribution_only'] is not True
            or producer['matched_constant_temperature_attribution_only'] is not True
            or producer['validation_labels_decoded'] or producer['fresh_test_read'] or producer['scientific_gate_passed']
            or any(producer['runtime'][k] != v for k, v in dict(device='cpu', device_name='cpu',
                    deterministic_algorithms=True, tf32=False, cpu_threads=4).items())
            or producer['timing_policy'] != dict(warmups=3, repeats=15, cpu_threads=4,
                rotating_operator_order=True, conditions=9, scene_count=5, include_role_preparation=True)):
        raise ValueError('validation numerical/population/cost policy changed')
    complete = read(run/'completion.json'); seal = read(run/'forward_complete.json')
    if (complete['lock_sha256'] != sha(run/'lock.json') or complete['summary_sha256'] != sha(run/'summary.json')
            or complete['independent_audit_pending'] is not True
            or seal['lock_sha256'] != sha(run/'lock.json') or seal['labels_decoded']
            or seal['record_count'] != 32*68*count or seal['retained_probability_bytes'] != expected_bytes):
        raise ValueError('producer completion or global forward seal changed')
    # No label decoding precedes these global phase and completion checks.
    actual_files = {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and p != run/'completion.json'}
    if actual_files != set(complete['files']):
        raise ValueError('producer completion has extra/missing artifacts')
    verify_files(complete['files'], run)
    forwards, evaluations = {}, {}
    for scene in SCENES:
        slug = scene.replace('/', '__')
        f = verify_stage(run, run/'forward'/slug, 'forward', scene, producer)
        e = verify_stage(run, run/'evaluation'/slug, 'evaluation', scene, producer)
        if (f['labels_decoded'] or f['record_count'] != 16*68*count or e['record_count'] != 16*68*count
                or e['forward_complete_sha256'] != sha(run/'forward_complete.json')
                or e['all_replay_hashes_exact'] is not True
                or seal['forward_chain'][scene] != dict(completion_sha256=sha(run/'forward'/slug/'completion.json'),
                                                       finished_unix=f['finished_unix'])):
            raise ValueError('scene forward/evaluation seal mismatch')
        forwards[scene], evaluations[scene] = f, e
        fstart, estart = read(run/'forward'/slug/'started.json'), read(run/'evaluation'/slug/'started.json')
        if (fstart != dict(created_unix=f['started_unix'], runtime=producer['runtime'], lock_sha256=sha(run/'lock.json'))
                or estart != dict(created_unix=e['started_unix'], forward_complete_sha256=sha(run/'forward_complete.json'))):
            raise ValueError('stage start record changed')
    timed = verify_stage(run, run/'timing', 'timing', 'all', producer)
    verify_chronology(producer, design, seal, forwards, evaluations, timed)
    expected_evaluations = {s:dict(completion_sha256=sha(run/'evaluation'/s.replace('/', '__')/'completion.json'),
                                   finished_unix=evaluations[s]['finished_unix']) for s in SCENES}
    if timed['evaluation_chain'] != expected_evaluations or timed['cells'] != 45*count:
        raise ValueError('paired timing evaluation chain/population changed')
    parts = REGION.literal_parts()
    if sha(run/'partitions.npz') != producer['partitions_sha256']:
        raise ValueError('saved partition file changed')
    with np.load(run/'partitions.npz', allow_pickle=False) as archive:
        expected_fields = {name+'__'+field for name in parts for field in ('bounds','owner','atoms','parent','mean','cov','mass','cell_bounds','cell_owner')}
        if set(archive.files) != expected_fields:
            raise ValueError('saved partition field population changed')
        for name, part in parts.items():
            for field in ('bounds', 'owner', 'atoms', 'parent', 'mean', 'cov', 'mass'):
                np.testing.assert_array_equal(archive[name+'__'+field], part[field])
            children, owners = [], []
            for i, (x0,y0,x1,y1) in enumerate(part['bounds']):
                cells = [(x0,y0,x1,y1)] if x1-x0 <= 16 and y1-y0 <= 16 else [
                    (x,y,x+12,y+12) for y in (y0,y0+12) for x in (x0,x0+12)]
                children.extend(cells); owners.extend([i]*len(cells))
            np.testing.assert_array_equal(archive[name+'__cell_bounds'], (np.asarray(children)-.5)/8)
            np.testing.assert_array_equal(archive[name+'__cell_owner'], owners)
    preflight = verify_zero_preflight(producer, regional_lock)
    expected_timing_entries = []
    for scene in TIMING_SCENES:
        population = regional_lock['entries'] if scene in TRAIN_SCENES else entries
        first = min((e for e in population if e['scene'] == scene), key=lambda e:Path(e['path']).name)
        expected_timing_entries.append({k:first[k] for k in ('scene','pair','path','input_sha256','manifest_sha256')})
    if producer['timing_entries'] != expected_timing_entries:
        raise ValueError('timing did not use each original first metadata pair')
    # Import only after every source/phase check above. No producer target,
    # metric, selector or gate function is used by this independent audit.
    from experiments import round17_validation as SHARED
    import torch
    if SHARED.runtime() != producer['runtime']:
        raise ValueError('shared operator replay runtime changed')
    with np.load(run/'partitions.npz', allow_pickle=False) as archive:
        saved_parts = {q:{name.split('__',1)[1]:archive[name] for name in archive.files if name.startswith(q+'__')}
                       for q in parts}
    partition_tensors = SHARED.S.M.partition_tensors(saved_parts, 'cpu')
    all_rows, rebuilt_rows, numerical, projections, attribution = [], [], [], [], []
    retained_count = retained_bytes = count_tables = center_count = 0
    replay_count = attribution_queries = 0
    maximum_metric_error = 0.
    for scene in SCENES:
        slug = scene.replace('/', '__'); directory = run/'forward'/slug
        scene_entries = sorted((e for e in entries if e['scene'] == scene), key=lambda e:Path(e['path']).name)
        if producer['sample_pairs'][scene] != scene_entries[0]['pair']:
            raise ValueError('numerical/retained sample pair changed')
        committed = read(directory/'records.json'); measured = read(run/'evaluation'/slug/'rows.json')
        fm = validate_forward_rows(committed, scene_entries, roles, parts)
        em = {identity(row):row for row in measured}
        if len(em) != len(measured) or set(em) != set(fm):
            raise ValueError('evaluation row population differs from committed forward')
        geometry = read(directory/'geometry.json')
        geom = {(tuple(r['pair']), r['state']):r for r in geometry}
        if (len(geometry) != 16*7 or len(geom) != len(geometry)
                or set(geom) != {(tuple(e['pair']), s) for e in scene_entries for s in N.STATES}):
            raise ValueError('perturbed projection population changed')
        repeat = read(directory/'repeat.json')
        expected_repeat = [{k:r[k] for k in ('operator','scenario','condition','array_sha256')}
                           for r in committed if r['pair'] == scene_entries[0]['pair']]
        if repeat != expected_repeat or len(repeat) != 68*count:
            raise ValueError('first-pair exact preparation/forward repeat incomplete')
        seen_paths = set()
        for index, entry in enumerate(scene_entries):
            source = absolute(entry['path']); meta = read(source/'manifest.json')
            inputs = load_inputs(entry); labels = load_labels(entry)
            per_partition = check_sensor_labels(meta, labels, entry['target_provenance'], parts)
            center_count += sum(per_partition.values())
            state_banks = {}
            for state in N.STATES:
                _, reference = N.reference_observations(inputs, state)
                check_projection(geom[tuple(entry['pair']), state], reference, scene, entry['pair'])
                projections.append(dict(scene=scene, pair=entry['pair'], **reference))
                with torch.no_grad():
                    qb, kb, diagnostic = SHARED.prepare_state(inputs, state, partition_tensors, operators)
                    if dict(scene=scene, pair=entry['pair'], **diagnostic) != geom[tuple(entry['pair']), state]:
                        raise ValueError('shared preparation projection differs from forward commitment')
                    matched = trace_attribution(qb, roles, SHARED.P.TRACE.trace_temperature)
                expected_queries = set(N.PARTITIONS if state == 'zero' else N.PARTITIONS[:3])
                if {r['query'] for r in matched} != expected_queries:
                    raise ValueError('trace attribution lost native query partitions')
                attribution.extend(dict(scene=scene, pair=entry['pair'], state=state, **r) for r in matched)
                attribution_queries += sum(r['query_count'] for r in matched)
                state_banks[state] = qb, kb
            target_tables = {}
            for c in CONDITIONS:
                q, k = c['query'], c['key']
                if (q, k) not in target_tables:
                    target_tables[q,k] = REGION.independent_counts(labels, parts, q, k)
                    count_tables += 1
                direct, common = target_tables[q,k]
                if direct.sum() != common.sum() or int(direct.sum()) != int(labels['gt_visible'].sum()):
                    raise ValueError('independent native/common target population changed')
                relative = retained_path(entry, c); seen_paths.add(str(relative))
                expected_ops = set(operators) if index == 0 else {primary}
                with np.load(run/relative, allow_pickle=False) as archive:
                    if set(archive.files) != expected_ops:
                        raise ValueError('retained primary/control sample population changed')
                    for op in operators:
                        row_id = scene, tuple(entry['pair']), op, c['scenario'], c['condition']
                        fr, er = fm[row_id], em[row_id]
                        check_measured_row(fr, er, entry, primary, direct, labels)
                        qb, kb = state_banks[c['state']]
                        with torch.no_grad():
                            replayed, diagnostic = SHARED.predict_prepared(qb[q], kb[k], operators[op])
                        replayed = check_replayed_probability(replayed, diagnostic, fr)
                        replay_count += 1
                        if op in expected_ops:
                            if fr.get('sample_path') != str(relative) or fr.get('sample_field') != op:
                                raise ValueError('retained array record/path binding changed')
                            probability = archive[op]
                            if (probability.dtype != np.float64 or probability.shape != replayed.shape
                                    or not np.array_equal(probability, replayed)):
                                raise ValueError('retained probability differs from complete shared replay')
                            retained_count += 1; retained_bytes += probability.nbytes
                        elif 'sample_path' in fr or 'sample_field' in fr:
                            raise ValueError('unsaved control falsely marked retained')
                        metric = check_probability(replayed, fr, labels, parts, direct, common)
                        for field, expected in metric.items():
                            error = compare_tree(er[field], expected, 1e-8, field)
                            maximum_metric_error = max(maximum_metric_error, error)
                        rebuilt_rows.append(dict(er, **metric))
            if index == 0:
                def loader(scenario, condition, operator, current_entry=entry):
                    path = run/retained_path(current_entry, dict(scenario=scenario, condition=condition))
                    with np.load(path, allow_pickle=False) as archive:
                        return archive[operator]
                def prepared_loader(scenario, condition, operator):
                    spec = next(c for c in CONDITIONS if c['scenario'] == scenario and c['condition'] == condition)
                    qb, kb = state_banks[spec['state']]
                    row = fm[scene, tuple(entry['pair']), operator, scenario, condition]
                    return prepared_numeric_reference(SHARED, qb[spec['query']], kb[spec['key']], operators[operator], row)
                result = N.audit_pair(inputs, parts, loader, operators, prepared_loader=prepared_loader)
                numerical.extend(dict(scene=scene, pair=entry['pair'], **r) for r in result['checks'])
                write(out/(slug+'_numeric.json'), result)
            print('VALIDATION_AUDIT_PAIR', scene, index+1, '/16', flush=True)
        actual_paths = {str(p.relative_to(run)) for p in (directory/'arrays').rglob('*') if p.is_file()}
        if actual_paths != seen_paths:
            raise ValueError('extra/missing native probability archive')
        all_rows.extend(measured)
    metrics = M.summarize(all_rows, entries, roles)
    rebuilt_metrics = M.summarize(rebuilt_rows, entries, roles)
    rebuilt_metric_error = compare_tree(metrics, rebuilt_metrics)
    timing_rows = read(run/'timing/rows.json')
    check_timing_sources(timing_rows, expected_timing_entries, roles)
    timing = M.aggregate_timing(timing_rows, expected_timing_entries, roles)
    if read(run/'timing/execution_order.json') != expected_execution_order(expected_timing_entries, roles):
        raise ValueError('paired cyclic operator execution order changed')
    shared = read(run/'timing/shared.json')
    if (len(shared['rows']) != 5 or shared['primary_time_includes_role_preparation'] is not True
            or shared['excludes_rgb_depth_predictor'] is not True or not isinstance(shared['max_rss_kib'], int)
            or shared['max_rss_kib'] <= 0 or 'not per-role allocator peak' not in shared['memory_scope']):
        raise ValueError('shared projection/memory evidence changed')
    for row, entry in zip(shared['rows'], expected_timing_entries):
        if (any(row[k] != entry[k] for k in ('scene','pair','input_sha256'))
                or not 0 <= row['projection_seconds'] <= row['export_seconds']
                or not np.isfinite(row['export_seconds'])):
            raise ValueError('shared projection provenance/time invalid')
    for row in timing_rows:
        if (len(row['load_average']) != 18 or np.asarray(row['load_average']).shape != (18,3)
                or not np.isfinite(row['load_average']).all() or np.any(np.asarray(row['load_average']) < 0)
                or len(row['prepared_tensor_bytes']) != 18
                or any(type(v) is not int or v <= 0 for v in row['prepared_tensor_bytes'])):
            raise ValueError('timing load or tensor-byte evidence missing')
    gates = M.gates(metrics, roles, timing)
    rebuilt_gates = M.gates(rebuilt_metrics, roles, timing)
    compare_tree(gates, rebuilt_gates)
    summary = read(run/'summary.json')
    maximum_summary_error = max(compare_tree(summary['metrics'], metrics),
        compare_tree(summary['timing'], timing), compare_tree(read(run/'timing/summary.json'), timing),
        compare_tree(summary['gates'], gates))
    if (summary['roles'] != roles or summary['lock_sha256'] != sha(run/'lock.json')
            or summary['forward_complete_sha256'] != sha(run/'forward_complete.json')
            or summary['pair_count'] != 32 or summary['row_count'] != 32*68*count
            or summary['independent_audit_pending'] is not True or summary['scientific_gate_passed']
            or summary['actual_value_run'] or summary['optimized_gs_run'] or summary['fresh_test_read']
            or not timed['finished_unix'] <= summary['created_unix'] <= complete['finished_unix']):
        raise ValueError('summary completion/authorization claim changed')
    coverage = dict(expected_pairs=len(entries), expected_rows=len(all_rows), expected_retained_arrays=retained_count,
        expected_numerical_matrices=len(numerical), expected_geometry_records=len(projections),
        expected_center_labels=center_count, expected_timing_rows=len(timing_rows),
        expected_replayed_arrays=replay_count, expected_attribution_supports=len(attribution),
        expected_attribution_queries=attribution_queries)
    if any(design[k] != v for k, v in coverage.items()) or retained_bytes != expected_bytes or count_tables != 32*14:
        raise ValueError('independent audit coverage is incomplete')
    numeric_passed = all(c['passed'] for c in numerical)
    decisions = decision_fields(numeric_passed, gates)
    verification = dict(**decisions,
        **{f'validation_{name}_sha256':sha(run/(name+'.json')) for name in ('lock','forward_complete','summary','completion')},
        independent_audit_lock_sha256=sha(out/'lock.json'), roles=roles,
        all_metric_rows=len(all_rows), retained_probability_arrays=retained_count, retained_probability_bytes=retained_bytes,
        independent_sensor_centers=center_count, independently_reconstructed_count_tables=count_tables,
        independently_reprojected_pair_states=len(projections), numerical_matrices=len(numerical),
        numerical_failure_count=sum(not r['passed'] for r in numerical),
        maximum_all_row_metric_error=maximum_metric_error, maximum_summary_error=maximum_summary_error,
        maximum_independently_recomputed_aggregate_error=rebuilt_metric_error,
        maximum_probability_error=max(r['max_probability_error'] for r in numerical),
        maximum_reference_convergence_error=max(r['reference_convergence_error'] for r in numerical
                                                 if r['reference_convergence_error'] is not None),
        numerical_argmax_changes=sum(r['argmax_change_count'] for r in numerical),
        gates=gates, timing_rows=len(timing_rows),
        shared_operator_replayed_arrays=replay_count, independent_all_metric_recomputation=True,
        shared_production_operator_replay=True, independent_forward_implementation=False,
        independent_all_raw_probability_replay=False, all_retained_probability_metrics_recomputed=True,
        trace_attribution_supports=len(attribution), trace_attribution_query_count=attribution_queries,
        trace_attribution_tau32_exact=True, trace_attribution_maximum_tau_difference=0.,
        attribution_uses_shared_preparation=True, attribution_does_not_establish_effect=True,
        numerical_sample='two fixed first pairs, all 68 conditions/all unique operators, 8 linspace queries/all native keys',
        fresh_test_read=False, actual_value_run=False, optimized_gs_run=False, seconds=time.time()-started)
    write(out/'projection_reference.json', projections)
    write(out/'zero_preflight_reference.json', preflight)
    write(out/'trace_attribution_replay.json', attribution)
    write(out/'independent_summary.json', dict(metrics=rebuilt_metrics, timing=timing, gates=rebuilt_gates))
    write(out/'verification.json', verification)
    write(out/'completion.json', dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
        finished_unix=time.time(), **decisions))
    print('INDEPENDENT_VALIDATION_'+('PASSED' if numeric_passed else 'FAILED'),
          len(all_rows), retained_count, len(numerical), 'SCIENTIFIC_GATE', decisions['scientific_gate_passed'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'audit'))
    parser.add_argument('--out', type=Path, default=ROOT/'results/round17_validation_independent_audit')
    parser.add_argument('--run', type=Path, default=ROOT/'results/round17_validation')
    args = parser.parse_args()
    freeze(args.out, args.run) if args.stage == 'freeze' else audit(args.out)


if __name__ == '__main__':
    main()
