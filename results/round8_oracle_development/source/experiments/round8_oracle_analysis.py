"""Independent NumPy analysis of OLD-validation oracle diagnostics.

This module imports neither a model nor the training metric implementation.
It never discovers or evaluates test data. Two development scans do not support
a generalization confidence interval, so all paired effects are descriptive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)
ROLES = ('pred_pred', 'gt_route', 'gt_value', 'gt_both', 'singleview')
SCENES = ('brown_bm_4/brown_bm_4', 'harvard_c8/hv_c8_3')
SUM_FIELDS = ('count', 'boundary_count', 'absrel_sum', 'abs_sum', 'sq_sum',
              'log_sq_sum', 'delta1_count', 'boundary_absrel_sum', 'boundary_abs_sum')
METRICS = ('AbsRel', 'MAE', 'RMSE', 'log_RMSE', 'delta1',
           'boundary_AbsRel', 'boundary_MAE')
COUNT_FIELDS = ('count', 'boundary_count', 'delta1_count')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolved(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def identity(row):
    return row['scene'], int(row['q_index']), int(row['k_index'])


def numpy_rows(prediction, target, valid, boundary, ids):
    """Compute sums after selecting the population, including zero-valid pairs."""
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid, boundary = np.asarray(valid, bool), np.asarray(boundary, bool)
    if pred.ndim != 2 or pred.shape != target.shape or valid.shape != pred.shape or boundary.shape != pred.shape:
        raise ValueError('prediction/label shape mismatch')
    if len(ids) != len(pred) or len({identity(row) for row in ids}) != len(ids):
        raise ValueError('identity count mismatch or duplicate identity')
    if not np.isfinite(pred).all() or np.any(pred <= 0):
        raise ValueError('predictions must be finite and positive')
    if np.any(valid & (~np.isfinite(target) | (target <= 0))):
        raise ValueError('valid targets must be finite and positive')
    boundary = boundary & valid
    rows = []
    for index, row_identity in enumerate(ids):
        p, t = pred[index, valid[index]], target[index, valid[index]]
        error = np.abs(p - t)
        pb, tb = pred[index, boundary[index]], target[index, boundary[index]]
        boundary_error = np.abs(pb - tb)
        rows.append(dict(row_identity,
            count=int(len(p)), boundary_count=int(len(pb)),
            absrel_sum=float(np.sum(error / t)), abs_sum=float(np.sum(error)),
            sq_sum=float(np.sum(error ** 2)), log_sq_sum=float(np.sum((np.log(p) - np.log(t)) ** 2)),
            delta1_count=int(np.count_nonzero(np.maximum(p / t, t / p) < 1.25)),
            boundary_absrel_sum=float(np.sum(boundary_error / tb)),
            boundary_abs_sum=float(np.sum(boundary_error))))
    return rows


def numpy_macro(rows, positions_per_pair=432):
    """Ratio of sums within a scan, then equal available-scan weight.

    An empty stratum in a scan is represented by None, never by a perfect zero.
    The all-query primary population must contain both scans (checked upstream).
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['scene']].append(row)
    by_scene = {}
    for scene, group in sorted(grouped.items()):
        sums = {field: math.fsum(float(row[field]) for row in group) for field in SUM_FIELDS}
        for field in COUNT_FIELDS:
            sums[field] = int(sums[field])
        n, b = sums['count'], sums['boundary_count']
        by_scene[scene] = dict(
            AbsRel=sums['absrel_sum'] / n if n else None,
            MAE=sums['abs_sum'] / n if n else None,
            RMSE=math.sqrt(sums['sq_sum'] / n) if n else None,
            log_RMSE=math.sqrt(sums['log_sq_sum'] / n) if n else None,
            delta1=sums['delta1_count'] / n if n else None,
            boundary_AbsRel=sums['boundary_absrel_sum'] / b if b else None,
            boundary_MAE=sums['boundary_abs_sum'] / b if b else None, **sums)
    metrics = {}
    for metric in METRICS:
        available = [row[metric] for row in by_scene.values() if row[metric] is not None]
        metrics[metric] = float(np.mean(available)) if available else None
    return dict(metrics=metrics, by_scene=by_scene,
        total_count=sum(row['count'] for row in rows),
        total_positions=len(rows) * positions_per_pair)


def compare_tree(actual, reported, tolerance=1e-8):
    """Strict schema and numeric check, including NaN rejection."""
    failures = []
    maximum = 0.
    def visit(a, b, path):
        nonlocal maximum
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                failures.append(path + ':keys')
                return
            for name in a:
                visit(a[name], b[name], path + '/' + str(name))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                failures.append(path + ':length')
                return
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + '/' + str(index))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            error = abs(float(a) - float(b))
            if not math.isfinite(error):
                failures.append(path + ':nonfinite')
            else:
                maximum = max(maximum, error)
                if error > tolerance:
                    failures.append(path + ':numeric')
        elif a != b:
            failures.append(path + ':value')
    visit(actual, reported, '')
    return dict(passed=not failures, max_absolute_error=maximum, mismatches=failures)


def population_masks(labels):
    valid = np.asarray(labels['valid'], bool)
    covisible = np.asarray(labels['oracle_covisible'], bool)
    if valid.shape != covisible.shape:
        raise ValueError('oracle covisibility shape mismatch')
    if np.any(covisible & ~valid):
        raise ValueError('oracle covisibility includes invalid query labels')
    masks = dict(all_query=valid, oracle_covisible=valid & covisible,
                 oracle_fallback=valid & ~covisible)
    if 'gt_key_coverage' in labels:
        coverage = np.asarray(labels['gt_key_coverage'], float)
        if coverage.shape != valid.shape or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
            raise ValueError('GT key coverage must be finite in [0,1]')
        for lower, upper in ((0., .25), (.25, .5), (.5, .75), (.75, 1.)):
            right = coverage <= upper if upper == 1. else coverage < upper
            # Source coverage is interpretable only where the sensor route exists.
            masks[f'key_coverage_{lower:g}_{upper:g}'] = covisible & (coverage >= lower) & right
    return masks


def paired_description(candidate, control, seeds=SEEDS, scenes=SCENES, higher_is_better=False):
    a, b = np.asarray(candidate, float), np.asarray(control, float)
    if a.shape != (len(seeds), len(scenes)) or b.shape != a.shape:
        raise ValueError('paired seed/scene shape mismatch')
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise ValueError('paired metric availability differs')
    finite = np.isfinite(a) & np.isfinite(b)
    delta = a - b
    def mean(values):
        values = np.asarray(values)
        return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None
    am, bm, dm = mean(a), mean(b), mean(delta)
    seed_delta = {str(seed): mean(delta[index]) for index, seed in enumerate(seeds)}
    scene_delta = {scene: mean(delta[:, index]) for index, scene in enumerate(scenes)}
    per_seed = {str(seed): dict(candidate=mean(a[index]), control=mean(b[index]),
        difference=seed_delta[str(seed)],
        relative_difference=(seed_delta[str(seed)] / mean(b[index]))
        if seed_delta[str(seed)] is not None and mean(b[index]) > 0 else None)
        for index, seed in enumerate(seeds)}
    return dict(descriptive_only=True, ci95=None,
        candidate_mean=am, control_mean=bm, difference=dm,
        relative_difference=dm / bm if bm is not None and bm > 0 else None,
        complete_seed_scene_matrix=bool(finite.all()), available_cells=int(finite.sum()),
        higher_is_better=higher_is_better,
        improved_seed_count=sum(value is not None and (value > 0 if higher_is_better else value < 0) for value in seed_delta.values()),
        improved_scene_count=sum(value is not None and (value > 0 if higher_is_better else value < 0) for value in scene_delta.values()),
        per_seed=per_seed, per_scene_difference=scene_delta,
        difference_matrix=matrix_json(delta))


def matrix_json(values):
    return [[float(x) if np.isfinite(x) else None for x in row] for row in np.asarray(values)]


def effect_description(values, seeds=SEEDS, scenes=SCENES):
    values = np.asarray(values, float)
    def mean(x):
        x = np.asarray(x)
        return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else None
    return dict(mean=mean(values), per_seed={str(s): mean(values[i]) for i, s in enumerate(seeds)},
        per_scene={s: mean(values[:, i]) for i, s in enumerate(scenes)},
        matrix=matrix_json(values), descriptive_only=True)


def analyze_predictions(predictions, labels, ids, seeds=SEEDS, scenes=SCENES):
    """Analyze all roles and raw arrays; no training metrics are used here."""
    seeds, scenes = tuple(seeds), tuple(scenes)
    if seeds != SEEDS or scenes != SCENES:
        raise ValueError('requires five fixed seeds and the two old validation scenes')
    if set(predictions) != set(ROLES):
        raise ValueError('incomplete role matrix')
    if set(row['scene'] for row in ids) != set(scenes):
        raise ValueError('unexpected scene population')
    masks = population_masks(labels)
    aggregates, arrays, rows_by_role = {}, {}, {}
    for role in ROLES:
        if set(int(s) for s in predictions[role]) != set(seeds) or len(predictions[role]) != len(seeds):
            raise ValueError('incomplete seed matrix: ' + role)
        aggregates[role], arrays[role], rows_by_role[role] = {}, {}, {}
        for stratum, valid in masks.items():
            aggregates[role][stratum] = {}
            arrays[role][stratum] = {metric: np.full((len(seeds), len(scenes)), np.nan) for metric in METRICS}
            for seed_index, seed in enumerate(seeds):
                prediction = predictions[role].get(seed, predictions[role].get(str(seed)))
                rows = numpy_rows(prediction, labels['target'], valid, labels['boundary'], ids)
                summary = numpy_macro(rows, positions_per_pair=np.asarray(labels['target']).shape[1])
                aggregates[role][stratum][str(seed)] = summary
                if stratum == 'all_query':
                    rows_by_role[role][str(seed)] = rows
                    if any(summary['by_scene'][scene]['count'] == 0 for scene in scenes):
                        raise ValueError('empty all-query primary population')
                for scene_index, scene in enumerate(scenes):
                    for metric in METRICS:
                        value = summary['by_scene'][scene][metric]
                        arrays[role][stratum][metric][seed_index, scene_index] = np.nan if value is None else value
    pairs = (('gt_route', 'singleview'), ('gt_route', 'pred_pred'),
             ('gt_value', 'pred_pred'), ('gt_both', 'pred_pred'),
             ('gt_both', 'gt_route'), ('gt_both', 'gt_value'), ('pred_pred', 'singleview'),
             ('gt_value', 'singleview'), ('gt_both', 'singleview'))
    comparisons = {stratum: {f'{a}_minus_{b}': {metric:
        paired_description(arrays[a][stratum][metric], arrays[b][stratum][metric], seeds, scenes, higher_is_better=metric=='delta1')
        for metric in METRICS} for a, b in pairs} for stratum in masks}
    factorial = {}
    for stratum in masks:
        factorial[stratum] = {}
        for metric in METRICS:
            p, r, v, j = (arrays[role][stratum][metric] for role in ROLES[:4])
            effects = dict(route_at_predicted_value=r-p, route_at_sensor_value=j-v,
                value_at_predicted_route=v-p, value_at_sensor_route=j-r,
                interaction=j-r-v+p, mean_route_effect=(r-p+j-v)/2,
                mean_value_effect=(v-p+j-r)/2)
            factorial[stratum][metric] = {name: effect_description(value, seeds, scenes) for name, value in effects.items()}
    def gate(role):
        comparison = comparisons['all_query'][role+'_minus_singleview']['AbsRel']
        criteria = dict(complete_primary_population=comparison['complete_seed_scene_matrix'],
            relative_AbsRel_improvement_at_least_2_percent=comparison['relative_difference'] is not None and comparison['relative_difference'] <= -.02,
            at_least_4_of_5_seed_macros_improve=comparison['improved_seed_count'] >= 4,
            both_validation_scene_seed_means_improve=comparison['improved_scene_count'] == 2)
        return dict(passed=all(criteria.values()), criteria=criteria, comparison=comparison,
            interpretation='development continuation gate, not a confirmatory significance test')
    models = {role: {stratum: {metric: effect_description(arrays[role][stratum][metric], seeds, scenes)
        for metric in METRICS} for stratum in masks} for role in ROLES}
    population = {name: dict(count=int(mask.sum()), fraction_of_all_valid=float(mask.sum()/masks['all_query'].sum()),
        by_scene={scene: int(mask[[i for i, row in enumerate(ids) if row['scene']==scene]].sum()) for scene in scenes})
        for name, mask in masks.items()}
    return dict(scope='old development validation only; no fresh confirmation', roles=ROLES, seeds=seeds, scenes=scenes,
        models=models, per_role_stratum_seed=aggregates, comparisons=comparisons,
        factorial=factorial, population=population,
        primary_gate=gate('gt_route'),
        diagnostic_gates={role:gate(role) for role in ('gt_value','gt_both')},
        limitations=['Oracle sensor geometry/depth are privileged diagnostic inputs and cannot be reported as deployable performance.',
            'Oracle replacement is not a strict upper bound; route quantization and the retained prediction head can limit its benefit.',
            'Five paired seeds quantify training variation on two already-used validation scans, not population generalization.',
            'Checkpoints are selected separately by minimum all-query validation AbsRel; factorial contrasts are diagnostic, not strict causal effects.',
            'Covisibility and key-coverage partitions use sensor data for analysis; all valid queries remain in the primary metric.',
            'Sensor routing includes privileged route-availability/visibility information; its benefit cannot be attributed only to corrected coordinates.',
            'An empty stratum/scene is missing rather than zero; stratum macros average only scenes with observations.']), rows_by_role


def load_validation_sources(sources, oracle_sources):
    """Validate paths/manifests before opening any arrays; forbid test inputs."""
    arrays, ids, records = defaultdict(list), [], []
    for source in sources:
        directory = resolved(source['path']).resolve()
        if directory.parent.name != 'val' or 'test' in directory.parts:
            raise ValueError('only old validation sources are allowed')
        manifest_path = directory/'manifest.json'
        if sha(manifest_path) != source['manifest_sha256']:
            raise ValueError('validation manifest hash changed')
        manifest = json.loads(manifest_path.read_text())
        if manifest['split'] != 'val' or manifest['scene'] not in SCENES:
            raise ValueError('not one of the two old validation scenes')
        for filename, field in (('inputs.npz','input_sha256'), ('labels.npz','label_sha256')):
            if sha(directory/filename) != source[field] or manifest[field] != source[field]:
                raise ValueError('validation array hash changed: '+filename)
        with np.load(directory/'labels.npz', allow_pickle=False) as data:
            for name in ('target', 'valid', 'boundary'):
                if name not in data:
                    raise ValueError('missing diagnostic label: '+name)
                arrays[name].append(data[name].copy())
        ids.extend(dict(scene=manifest['scene'], q_index=int(q), k_index=int(k), pair_id=f'{q}:{k}') for q,k in manifest['pairs'])
        records.append(dict(path=str(directory), manifest_sha256=sha(manifest_path), input_sha256=source['input_sha256'], label_sha256=source['label_sha256']))
    if set(row['scene'] for row in ids) != set(SCENES) or len(ids) != 128 or len({identity(row) for row in ids}) != 128:
        raise ValueError('requires both complete 64-pair old validation scenes')
    if any(sum(row['scene'] == scene for row in ids) != 64 for scene in SCENES):
        raise ValueError('requires 64 pairs per old validation scene')
    arrays = {name: np.concatenate(value) for name, value in arrays.items()}
    if arrays['target'].shape != (128,432):
        raise ValueError('expected 128 pairs and 432 query centers')
    oracle_records, oracle_rows = [], {}
    for source in oracle_sources:
        directory = resolved(source['path']).resolve()
        if directory.parent.name != 'val' or 'test' in directory.parts:
            raise ValueError('only old validation oracle sources are allowed')
        manifest_path = directory/'manifest.json'
        if sha(manifest_path) != source['manifest_sha256']:
            raise ValueError('oracle manifest hash changed')
        manifest = json.loads(manifest_path.read_text())
        if manifest['split'] != 'val' or manifest['scene'] not in SCENES:
            raise ValueError('oracle source is not one of the old validation scenes')
        if sha(directory/'oracle.npz') != source['oracle_sha256'] or manifest['oracle_sha256'] != source['oracle_sha256']:
            raise ValueError('oracle array hash changed')
        with np.load(directory/'oracle.npz', allow_pickle=False) as data:
            valid = data['oracle_valid'].astype(bool)
            indices, weights = data['oracle_indices'], data['oracle_weights'].astype(float)
            fractions = data['k_sensor_fraction'].astype(float)
            if valid.shape != (len(manifest['pairs']),432) or indices.shape != valid.shape+(4,) or weights.shape != indices.shape or fractions.shape != valid.shape:
                raise ValueError('oracle routing shapes differ')
            if not np.issubdtype(indices.dtype, np.integer) or np.any((indices<0)|(indices>=432)):
                raise ValueError('oracle key indices outside token grid')
            if not np.isfinite(weights).all() or np.any(weights<0) or not np.isfinite(fractions).all() or np.any((fractions<0)|(fractions>1)):
                raise ValueError('invalid oracle weights or sensor coverage')
            if not np.allclose(weights.sum(-1)[valid], 1., rtol=0, atol=1e-6):
                raise ValueError('available oracle route must have unit weight')
            gathered = np.take_along_axis(fractions[:,None,:], indices, axis=2)
            coverage = np.sum(gathered*weights,axis=-1)
            coverage = np.where(valid,coverage,0).clip(0,1)
        for index,(q,k) in enumerate(manifest['pairs']):
            key=(manifest['scene'],int(q),int(k))
            if key in oracle_rows:
                raise ValueError('duplicate oracle pair identity')
            oracle_rows[key]=(valid[index],coverage[index])
        oracle_records.append(dict(path=str(directory),manifest_sha256=sha(manifest_path),oracle_sha256=source['oracle_sha256']))
    if set(oracle_rows) != {identity(row) for row in ids}:
        raise ValueError('oracle and label pair identities differ')
    arrays['oracle_covisible'] = np.stack([oracle_rows[identity(row)][0] for row in ids])
    arrays['gt_key_coverage'] = np.stack([oracle_rows[identity(row)][1] for row in ids])
    return arrays, ids, dict(base=records,oracle=oracle_records)


def inspect_training_metadata(metadata, role, seed, run, expected_sources, batch_hashes, parameter_counts):
    if metadata['role'] != role or metadata['seed'] != seed or metadata['steps'] != 1000:
        raise ValueError('unexpected training role, seed or budget')
    if metadata['data_sources'] != expected_sources:
        raise ValueError('unpaired training/validation source archives')
    for filename, key in (('best.pt','checkpoint_sha256'),('validation.json','validation_sha256'),('val_predictions.npz','prediction_sha256')):
        if sha(run/filename) != metadata[key]:
            raise ValueError('changed training artifact: '+str(run/filename))
    for source, digest in metadata['source_hashes'].items():
        if sha(resolved(source)) != digest:
            raise ValueError('changed training source: '+source)
    expected_steps = list(range(0,1001,100))
    history = metadata['history']
    if [entry['step'] for entry in history] != expected_steps:
        raise ValueError('unexpected checkpoint evaluation schedule')
    values = [entry['validation']['metrics']['AbsRel'] for entry in history]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('nonfinite checkpoint-selection metric')
    best = min(history, key=lambda entry: entry['validation']['metrics']['AbsRel'])
    if best['step'] != metadata['best_step'] or abs(best['validation']['metrics']['AbsRel'] - metadata['best_validation_AbsRel']) > 1e-12:
        raise ValueError('checkpoint is not the earliest minimum validation AbsRel')
    if seed in batch_hashes and batch_hashes[seed] != metadata['batch_sequence_sha256']:
        raise ValueError('unpaired training batch sequences')
    batch_hashes[seed] = metadata['batch_sequence_sha256']
    parameter_counts.add(metadata['parameter_count'])


def write_readme(path, result):
    lines = ['# Round 8 oracle diagnostics (old validation only)', '',
        'Independent NumPy reconstruction from saved predictions; five paired seeds and two already-used validation scans. No confidence interval or generalization claim.', '',
        '| Role | AbsRel | MAE (m) | RMSE (m) | Boundary AbsRel |',
        '|---|---:|---:|---:|---:|']
    for role in ROLES:
        metrics = result['models'][role]['all_query']
        values = [metrics[name]['mean'] for name in ('AbsRel','MAE','RMSE','boundary_AbsRel')]
        lines.append('| '+role+' | '+' | '.join('NA' if value is None else f'{value:.6f}' for value in values)+' |')
    primary = result['primary_gate']['comparison']
    relative = primary['relative_difference']
    lines += ['', f"Development continuation gate: **{'PASS' if result['primary_gate']['passed'] else 'FAIL'}**.",
        f"Sensor routing with original predicted V versus singleview: relative AbsRel difference {relative:+.3%}; {primary['improved_seed_count']}/5 seed macros and {primary['improved_scene_count']}/2 scene means improve.", '',
        'Negative differences indicate improvement for error metrics; positive delta1 differences indicate improvement.', '',
        'Factorial differences, coverage partitions, seed/scene matrices, exact population counts and audit records are in `summary.json`.', '',
        'Oracle inputs are privileged diagnostics, not deployable results or strict upper bounds. Checkpoint selection uses the same development validation, so the factorial contrasts are not strict causal effects.']
    path.write_text('\n'.join(lines)+'\n')


def analyze_directory(development, out):
    development, out = Path(development), Path(out)
    if out.exists():
        raise ValueError('refuse overwrite analysis output')
    metadata0 = json.loads((development/'pred_pred'/'11'/'run.json').read_text())
    sources = metadata0['data_sources']
    labels, ids, data_records = load_validation_sources(sources['val'], sources['oracle_val'])
    predictions, reported, audits, batch_hashes, capacities = {}, {}, [], {}, set()
    for role in ROLES:
        predictions[role], reported[role] = {}, {}
        for seed in SEEDS:
            run = development/role/str(seed)
            metadata = json.loads((run/'run.json').read_text())
            inspect_training_metadata(metadata, role, seed, run, sources, batch_hashes, capacities)
            with np.load(run/'val_predictions.npz', allow_pickle=False) as data:
                if set(data.files) != {'depth'}:
                    raise ValueError('prediction archive must contain only depth')
                predictions[role][seed] = data['depth'].copy()
            reported[role][seed] = json.loads((run/'validation.json').read_text())
            audits.append(dict(role=role, seed=seed, run_sha256=sha(run/'run.json'),
                checkpoint_sha256=metadata['checkpoint_sha256'], prediction_sha256=metadata['prediction_sha256'],
                validation_sha256=metadata['validation_sha256'], best_step=metadata['best_step'],
                batch_sequence_sha256=metadata['batch_sequence_sha256']))
    if len(capacities) != 1:
        raise ValueError('role parameter counts differ')
    result, rows = analyze_predictions(predictions, labels, ids)
    metric_checks = []
    for role in ROLES:
        for seed in SEEDS:
            actual = dict(rows=rows[role][str(seed)], summary=result['per_role_stratum_seed'][role]['all_query'][str(seed)])
            record = reported[role][seed]
            check = compare_tree(actual, {name:record[name] for name in ('rows','summary')})
            metric_checks.append(dict(role=role, seed=seed, **check))
            if not check['passed']:
                raise ValueError('independent metrics disagree with trainer: '+role+'/'+str(seed)+' '+str(check))
    result['verification'] = dict(passed=True, independently_recomputed_prediction_archives=len(audits),
        parameter_count=next(iter(capacities)), data_sources=data_records, run_audits=audits,
        metric_checks=metric_checks, max_metric_absolute_error=max(row['max_absolute_error'] for row in metric_checks),
        analysis_sha256=sha(__file__), source_model_or_trainer_imports=False,
        any_test_data_read=False, prediction_sha256_checks=True, input_label_source_sha256_checks=True)
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (out/'source_snapshot.py').write_text(Path(__file__).read_text())
    write_readme(out/'README.md', result)
    print(json.dumps(dict(out=str(out), passed=result['primary_gate']['passed'],
        verification=result['verification']['passed'], comparison=result['primary_gate']['comparison']), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze_directory(args.development, args.out)


if __name__ == '__main__':
    main()
