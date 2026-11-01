"""Independent Stage B artifact, sampled-array, numerical and selection audit.

All metric aggregation is independently recomputed. Only the preregistered
three first-pair raw samples are retained and independently reintegrated.
This file never imports the Stage B producer or changes a selected operator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments import round15_regional_audit_metrics as M
from experiments import round15_regional_audit_numeric as N

ROOT = Path(__file__).resolve().parents[1]
SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
FAMILIES = ('quartic_full', 'quartic_point', 'gaussian_point', 'gaussian_fixed',
            'gaussian_full1', 'gaussian_full4', 'gaussian_quad4', 'gaussian_quad16')
GRID = tuple(dict(h=h, temperature=t) for h in (.25, .5, 1., 2.) for t in (0., 5., 10.))
ORIGINAL = tuple((q, k) for q in ('g8', 'g16', 'mixed') for k in ('g8', 'g16', 'mixed'))
SCALE = (('g8','g24'), ('g24','g8'), ('g16','g24'), ('g24','g16'), ('g24','g24'))
CONDITIONS = tuple((scenario, q, k) for scenario, pairs in (('partition', ORIGINAL), ('scale', SCALE))
                   for q, k in pairs)
METRIC_FIELDS = ('kl_sum', 'js_sum', 'direct_kl_sum', 'direct_js_sum')
OWN_SOURCES = ('ROUND15_REGIONAL_AUDIT_PROTOCOL.md', 'experiments/round15_regional_audit.py',
    'tests/test_round15_regional_audit.py', 'experiments/round15_regional_audit_metrics.py',
    'tests/test_round15_regional_audit_metrics.py', 'experiments/round15_regional_audit_numeric.py',
    'tests/test_round15_regional_audit_numeric.py')


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def verify_files(files, base=ROOT):
    for name, digest in files.items():
        path = Path(name)
        path = path if path.is_absolute() else Path(base)/path
        if sha(path) != digest:
            raise ValueError('changed frozen artifact: '+str(path))


def identity(row):
    return row['scene'], tuple(row['pair']), row['family'], row['grid_index'], row['condition']


def compare_tree(actual, expected, tolerance=1e-10, path='root'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('summary keys differ: '+path)
        errors = [compare_tree(actual[k], v, tolerance, path+'.'+k) for k, v in expected.items()]
        return max(errors, default=0.)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('summary list differs: '+path)
        return max((compare_tree(a, e, tolerance, path+f'[{i}]')
                    for i, (a, e) in enumerate(zip(actual, expected))), default=0.)
    if isinstance(expected, (float, np.floating)):
        if not np.isfinite(actual) or abs(actual-expected) > tolerance:
            raise ValueError('summary numeric value differs: '+path)
        return abs(actual-expected)
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        raise ValueError('summary identity differs: '+path)
    return 0.


def check_replayed_row(forward, measured, entry, shape):
    if any(measured.get(k) != v for k, v in forward.items()):
        raise ValueError('evaluation changed a committed forward record')
    index = forward['grid_index']
    if (forward['config'] != GRID[index] or forward['shape'] != list(shape)
            or forward['dtype'] != 'float64' or forward['input_sha256'] != entry['input_sha256']
            or measured['replayed_array_sha256'] != forward['array_sha256']
            or measured['label_sha256'] != entry['label_sha256']
            or measured['n_query'] != shape[0] or measured['n_key'] != shape[1]):
        raise ValueError('replay/configuration/provenance mismatch')


def freeze(out, run):
    if out.exists():
        raise ValueError('refusing independent audit freeze overwrite')
    run = run.resolve(); producer = read(run/'lock.json')
    if sha(run/'lock.json') != (run/'lock.sha256').read_text().strip():
        raise ValueError('producer lock mismatch')
    # No Stage B result or label is decoded while freezing the independent design.
    paths = sorted(set(OWN_SOURCES).union(N.SOURCE_NAMES))
    sources = {name: sha(ROOT/name) for name in paths}
    verify_files(producer['source_hashes'])
    out.mkdir(parents=True)
    for name in sources:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    write(out/'lock.json', dict(created_unix=time.time(), run=str(run), source_hashes=sources,
        producer_sources=producer['source_hashes'], producer_lock_sha256=sha(run/'lock.json'),
        expected_rows=64512, expected_saved_arrays=4032, expected_pairs=48,
        stage_b_results_opened=False, regional_labels_decoded=False, scientific_gate_passed=False,
        metric_sum_tolerance=1e-8, summary_tolerance=1e-10,
        numeric_probability_tolerance=1e-4, quartic_reference_convergence_tolerance=1e-8,
        coverage='All identities, label masses, aggregation and selection; all 4032 retained first-pair arrays; declared independent numerical subset.'))
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def audit(out):
    started = time.time(); lock = read(out/'lock.json')
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip():
        raise ValueError('independent audit lock mismatch')
    verify_files(lock['source_hashes']); verify_files(lock['source_hashes'], out/'source_snapshot')
    run = Path(lock['run']); producer = read(run/'lock.json')
    if sha(run/'lock.json') != lock['producer_lock_sha256'] or producer['source_hashes'] != lock['producer_sources']:
        raise ValueError('producer provenance changed')
    verify_files(lock['producer_sources']); verify_files(lock['producer_sources'], run/'source_snapshot')
    completion = read(run/'completion.json'); verify_files(completion['files'], run)
    if (completion['lock_sha256'] != sha(run/'lock.json')
            or completion['summary_sha256'] != sha(run/'summary.json')
            or completion['forward_complete_sha256'] != sha(run/'forward_complete.json')
            or completion['all_replay_hashes_exact'] is not True
            or completion['scientific_gate_passed'] or completion['validation_run']):
        raise ValueError('invalid producer completion chain')
    previous = producer['stage_a']; previous_root = Path(previous['root'])
    for name in ('lock','forward_complete','summary','completion'):
        if sha(previous_root/(name+'.json')) != previous[f'feasibility_{name}_sha256']:
            raise ValueError('Stage A artifact chain changed: '+name)
    previous_complete = read(previous_root/'completion.json')
    verify_files(previous_complete['files'], previous_root)
    previous_lock = read(previous_root/'lock.json')
    verify_files(previous_lock['source_hashes']); verify_files(previous_lock['source_hashes'], previous_root/'source_snapshot')
    if (sha(previous_root/'decision.json') != previous['decision_sha256']
            or sha(previous['audit_path']) != previous['audit_sha256']):
        raise ValueError('Stage A eligibility provenance changed')
    previous_decision = read(previous_root/'decision.json')
    if (previous_decision['action'] != 'complete_full_regional_calibration'
            or previous_decision['feasible_grid_indices'] != previous['eligible_indices']
            or read(previous['audit_path'])['passed'] is not True):
        raise ValueError('Stage A did not independently authorize this calibration')
    inherited_manifests = {(r['scene'], tuple(r['pair'])):r for r in previous['training_manifest_records']}
    if previous['training_manifest_records'] != previous_lock['training_manifest_records'] or len(inherited_manifests) != 48:
        raise ValueError('Stage B changed the inherited training cohort')
    legacy = producer['legacy']; legacy_root = Path(legacy['root'])
    if sha(legacy_root/'lock.json') != legacy['lock_sha256']:
        raise ValueError('legacy lock changed')
    legacy_lock = read(legacy_root/'lock.json')
    if legacy['upstream_files'] != legacy_lock['inherited']['files']:
        raise ValueError('legacy dependency chain changed')
    verify_files(legacy['upstream_files']); verify_files(legacy_lock['source_hashes'])
    family_names = dict(point='gaussian_point', fixed='gaussian_fixed', full='gaussian_full1',
                        full4='gaussian_full4', quad4='gaussian_quad4', quad16='gaussian_quad16')
    expected_configurations = {}
    for role in ('point','point_pck','fixed','full1','full4','quad4','quad16'):
        old = legacy_lock['configurations'][role]
        cfg = {name:old[name] for name in ('h','temperature')}
        expected_configurations[role] = dict(family=family_names[old['method']], grid_index=GRID.index(cfg), config=cfg)
    if legacy['configurations'] != expected_configurations:
        raise ValueError('legacy reference configuration changed')
    seal = read(run/'forward_complete.json')
    if (seal['record_count'] != 64512 or seal['raw_sample_count'] != 4032
            or seal['regional_labels_decoded'] or seal['all_probabilities_retained']
            or seal['lock_sha256'] != sha(run/'lock.json')):
        raise ValueError('incomplete or misrepresented global forward seal')
    parts = M.literal_parts()
    if sha(run/'partitions.npz') != producer['partitions_sha256']:
        raise ValueError('partition artifact changed')
    with np.load(run/'partitions.npz', allow_pickle=False) as saved_parts:
        for name, part in parts.items():
            for field in ('bounds','owner','atoms','mass','parent','mean','cov'):
                value = part[field]
                if not np.array_equal(value, saved_parts[name+'__'+field]):
                    raise ValueError('partition differs from independent literal definition: '+name+'/'+field)
    result = read(run/'summary.json')
    all_rows = []; sample_arrays = 0; count_checks = 0; maximum_metric_error = 0.; numerical = []; numeric_ok = True
    for scene in SCENES:
        slug = scene.replace('/', '__'); forward = run/'forward'/slug; evaluation = run/'evaluation'/slug
        fc, ec = read(forward/'completion.json'), read(evaluation/'completion.json')
        verify_files(fc['files'], forward); verify_files(ec['files'], evaluation)
        if (fc['scene'] != scene or ec['scene'] != scene or fc['stage'] != 'forward' or ec['stage'] != 'evaluation'
                or fc['lock_sha256'] != sha(run/'lock.json') or ec['lock_sha256'] != sha(run/'lock.json')
                or fc['regional_labels_decoded'] or ec['all_replay_hashes_exact'] is not True
                or ec['new_regional_labels_decoded'] is not True
                or fc['runtime'] != producer['runtime'] or ec['runtime'] != producer['runtime']
                or ec['forward_complete_sha256'] != sha(run/'forward_complete.json')
                or seal['forward_chain'][scene]['completion_sha256'] != sha(forward/'completion.json')):
            raise ValueError('scene stage provenance mismatch')
        if not producer['created_unix'] <= lock['created_unix'] <= fc['started_unix'] <= fc['finished_unix'] <= seal['sealed_unix'] <= ec['started_unix'] <= ec['finished_unix']:
            raise ValueError('regional experiment/audit chronology mismatch')
        committed, measured = read(forward/'records.json'), read(evaluation/'rows.json')
        fm, em = {identity(r): r for r in committed}, {identity(r): r for r in measured}
        if len(fm) != 21504 or len(fm) != len(committed) or len(em) != len(measured) or set(fm) != set(em):
            raise ValueError('missing or duplicate scene record')
        stage_a_rows = read(previous_root/'forward'/slug/'records.json')
        stage_a_map = {(tuple(r['pair']), r['family'], r['grid_index'], r['query']):r for r in stage_a_rows}
        overlap_count = 0
        for row in committed:
            if row['key'] == 'g8' and row['query'] in ('g8','g16','g24') and row['family'] in FAMILIES[:3]:
                prior_row = stage_a_map[(tuple(row['pair']), row['family'], row['grid_index'], row['query'])]
                if any(row[field] != prior_row[field] for field in ('array_sha256','input_sha256','shape','dtype','config')):
                    raise ValueError('regional operator no longer matches the center-feasible operator')
                overlap_count += 1
        if overlap_count != 1728:
            raise ValueError('incomplete Stage A probability overlap')
        entries = sorted((e for e in producer['entries'] if e['scene'] == scene), key=lambda e: Path(e['path']).name)
        if len(entries) != 16 or producer['sample_pairs'][scene] != entries[0]['pair']:
            raise ValueError('metadata sample/cohort changed')
        repeats = read(forward/'determinism.json')
        repeated = {(r['condition'], r['family'], r['grid_index']): r['array_sha256'] for r in repeats['records']}
        expected_repeat = {(r['condition'], r['family'], r['grid_index']): r['array_sha256']
                           for r in committed if r['pair'] == entries[0]['pair']}
        if (repeats['passed'] is not True or repeats['labels_decoded'] or repeats['pair'] != entries[0]['pair']
                or len(repeats['records']) != 1344 or repeated != expected_repeat):
            raise ValueError('first-pair bitwise repeat is incomplete')
        for pair_index, entry in enumerate(entries):
            path = absolute(entry['path'])
            verify_files({str(path/'manifest.json'):entry['manifest_sha256'],
                          str(path/'inputs.npz'):entry['input_sha256'], str(path/'labels.npz'):entry['label_sha256']})
            metadata = read(path/'manifest.json'); prior = inherited_manifests[(scene, tuple(entry['pair']))]
            if (metadata['split'] != 'train' or metadata['forward_uses_sensor']
                    or metadata['scene'] != scene or metadata['pair'] != entry['pair']
                    or metadata['input_sha256'] != entry['input_sha256'] or metadata['label_sha256'] != entry['label_sha256']
                    or absolute(prior['path']) != path/'manifest.json' or prior['sha256'] != entry['manifest_sha256']):
                raise ValueError('training pair metadata does not match the inherited cohort')
            with np.load(path/'labels.npz', allow_pickle=False) as archive:
                labels = {name: archive[name] for name in ('gt_xy', 'gt_visible')}
            samples = {}
            for scenario, q, k in CONDITIONS:
                condition = q+'_'+k; shape = (len(parts[q]['mass']), len(parts[k]['mass']))
                direct, common = M.independent_counts(labels, parts, q, k); count_checks += 1
                if int(direct.sum()) != int(common.sum()):
                    raise ValueError('independent target population differs by partition')
                sample = forward/'samples'/path.name/(condition+'.npz')
                archive = np.load(sample, allow_pickle=False) if pair_index == 0 else None
                if archive is not None:
                    expected_fields = {f'{f}__c{i:02d}' for f in FAMILIES for i in range(12)}
                    if set(archive.files) != expected_fields:
                        raise ValueError('incomplete saved sample')
                    samples[condition] = sample
                try:
                    for family in FAMILIES:
                        for index in range(12):
                            row_id = (scene, tuple(entry['pair']), family, index, condition)
                            fr, er = fm[row_id], em[row_id]
                            check_replayed_row(fr, er, entry, shape)
                            if (fr['scenario'] != scenario or fr['query'] != q or fr['key'] != k
                                    or er['mass'] != int(direct.sum())):
                                raise ValueError('condition or independently reconstructed mass differs')
                            if archive is not None:
                                field = f'{family}__c{index:02d}'
                                if fr['sample_field'] != field or fr['sample_path'] != str(sample.relative_to(run)):
                                    raise ValueError('saved array identity differs')
                                probability = archive[field]
                                if probability.dtype != np.float64 or probability.shape != shape:
                                    raise ValueError('saved sample shape/dtype mismatch')
                                if array_sha(probability) != fr['array_sha256']:
                                    raise ValueError('sample does not match forward commitment')
                                independent = M.measure(probability, parts, q, k, scenario, direct, common)
                                for metric in METRIC_FIELDS:
                                    error = abs(independent[metric]-er[metric]); maximum_metric_error = max(maximum_metric_error, error)
                                    if error > 1e-8:
                                        raise ValueError('independent sample metric differs: '+str((row_id, metric, error)))
                                sample_arrays += 1
                            elif 'sample_path' in fr or 'sample_field' in fr:
                                raise ValueError('undeclared sample')
                            all_rows.append(er)
                finally:
                    if archive is not None:
                        archive.close()
            if pair_index == 0:
                with np.load(path/'inputs.npz', allow_pickle=False) as archive:
                    inputs = {name: archive[name] for name in ('q_to_k_xy','q_to_k_pixel_cov','q_features','k_features')}
                numeric = N.numeric_pair(inputs, parts, samples, result['selection'])
                write(out/('numerical_'+slug+'.json'), numeric)
                if numeric['matrix_count'] != len(numeric['checks']):
                    raise ValueError('incomplete independent numerical sample')
                numeric_ok = numeric_ok and numeric['passed'] is True
                expected_numeric = {(q, k, family, index) for _, q, k in CONDITIONS for family in FAMILIES
                    for index in (range(12) if family.startswith('quartic_') else
                                  sorted({4, result['selection']['controls'][family]['grid_index']}))}
                found_numeric = [(c['query'],c['key'],c['family'],c['grid_index']) for c in numeric['checks']]
                if len(found_numeric) != len(set(found_numeric)) or set(found_numeric) != expected_numeric:
                    raise ValueError('numerical sample differs from the fixed coverage contract')
                numerical.extend(dict(scene=scene, pair=entry['pair'], **check) for check in numeric['checks'])
        print('INDEPENDENT_REGIONAL_SCENE', scene, 'complete', flush=True)
    independent = M.summarize_and_select(all_rows, producer['entries'], previous['eligible_indices'])
    maximum_summary_error = compare_tree(result['metrics'], independent['metrics'])
    maximum_selection_error = compare_tree(result['selection'], independent['selection'])
    if (len(all_rows) != 64512 or sample_arrays != 4032 or count_checks != 672
            or result['row_count'] != 64512 or result['pair_count'] != 48
            or result['selection_pending_independent_audit'] is not True or result['scientific_gate_passed']
            or result['validation_run'] or result['fresh_confirmation'] or result['all_replay_hashes_exact'] is not True
            or result['stage_a_point_localization_selections'] != previous['point_localization_selections']
            or result['legacy_point_localization'] != previous['legacy_point']):
        raise ValueError('incomplete coverage or overclaimed result')
    expected_legacy = {name:dict(**spec, summary=result['metrics'][spec['family']][spec['grid_index']])
                       for name, spec in producer['legacy']['configurations'].items()}
    compare_tree(result['legacy_references'], expected_legacy)
    verification = dict(passed=numeric_ok, **{f'regional_{name}_sha256':sha(run/(name+'.json'))
        for name in ('lock','forward_complete','summary','completion')},
        independent_audit_lock_sha256=sha(out/'lock.json'), selection=result['selection'],
        all_metric_rows=len(all_rows), saved_probability_arrays=sample_arrays,
        independently_reconstructed_count_tables=count_checks, numerical_matrices=len(numerical), numerical_checks=numerical,
        maximum_sample_metric_error=maximum_metric_error, maximum_summary_error=maximum_summary_error,
        maximum_selection_error=maximum_selection_error, scientific_gate_passed=False,
        numerical_failure_count=sum(not c['passed'] for c in numerical),
        independent_all_raw_probability_replay=False, declared_first_pair_sample=True,
        validation_or_fresh_test_read=False, seconds=time.time()-started)
    write(out/'verification.json', verification)
    write(out/'completion.json', dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
        finished_unix=time.time(), passed=numeric_ok))
    print('INDEPENDENT_REGIONAL_'+('PASSED' if numeric_ok else 'FAILED'), len(all_rows), sample_arrays, len(numerical), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('freeze','audit'))
    parser.add_argument('--out', type=Path, default=ROOT/'results/round15_regional_independent_audit')
    parser.add_argument('--run', type=Path, default=ROOT/'results/round15_regional')
    args = parser.parse_args()
    freeze(args.out, args.run) if args.stage == 'freeze' else audit(args.out)


if __name__ == '__main__':
    main()
