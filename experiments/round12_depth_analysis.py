"""NumPy-only audit and gates for the old-development Round12 depth task."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROLES = ('full4', 'iso4_matched', 'full1', 'quad4', 'quad16', 'point', 'point_pck', 'fixed', 'singleview')
SEEDS = (11, 22, 33, 44, 55)
POINT_CONTROLS = ('point', 'point_pck', 'fixed')
COMPRESSION_CONTROLS = ('full1', 'quad4', 'quad16')
METRICS = ('AbsRel', 'MAE', 'RMSE', 'log_RMSE', 'delta1', 'boundary_AbsRel', 'boundary_MAE')
ADDITIVE = ('count', 'boundary_count', 'absrel_sum', 'abs_sum', 'sq_sum', 'log_sq_sum',
            'delta1_count', 'boundary_absrel_sum', 'boundary_abs_sum')
PARAMETER_NAMES = {'qk.weight', 'query.weight', 'query.bias', 'value.weight', 'value.bias',
                   'head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_close(a, b, where='', tolerance=1e-10):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError('dictionary mismatch '+where)
        return max((check_close(a[k], b[k], where+'/'+k, tolerance) for k in a), default=0.)
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError('list mismatch '+where)
        return max((check_close(x, y, where+'/'+str(i), tolerance) for i, (x, y) in enumerate(zip(a, b))), default=0.)
    if isinstance(a, (float, int)) and not isinstance(a, bool):
        if not isinstance(b, (float, int)) or not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('numeric mismatch '+where)
        error = abs(float(a)-float(b))
        if error > tolerance*max(1., abs(a), abs(b)):
            raise ValueError('numeric mismatch '+where)
        return error
    if type(a) is not type(b) or a != b:
        raise ValueError('metadata mismatch '+where)
    return 0.


def raw_metrics(prediction, target, valid, boundary, identities):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool); boundary = np.asarray(boundary, dtype=bool) & valid
    if prediction.shape != target.shape or valid.shape != target.shape or len(identities) != len(target):
        raise ValueError('prediction/label/identity shape mismatch')
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError('invalid predicted depth')
    if not np.isfinite(target[valid]).all() or (target[valid] <= 0).any():
        raise ValueError('invalid sensor target on valid population')
    target = np.where(valid, target, 1.)
    delta = np.abs(prediction-target); relative = delta/target
    log_error = (np.log(prediction)-np.log(target))**2
    ratio = np.maximum(prediction/target, target/prediction)
    arrays = dict(count=valid.sum(-1), boundary_count=boundary.sum(-1),
        absrel_sum=np.where(valid, relative, 0.).sum(-1), abs_sum=np.where(valid, delta, 0.).sum(-1),
        sq_sum=np.where(valid, delta**2, 0.).sum(-1), log_sq_sum=np.where(valid, log_error, 0.).sum(-1),
        delta1_count=((ratio < 1.25) & valid).sum(-1),
        boundary_absrel_sum=np.where(boundary, relative, 0.).sum(-1),
        boundary_abs_sum=np.where(boundary, delta, 0.).sum(-1))
    rows = [dict(identity, **{key: float(value[i]) for key, value in arrays.items()})
            for i, identity in enumerate(identities)]
    by_scene = {}
    for scene in sorted({row['scene'] for row in rows}):
        chosen = [row for row in rows if row['scene'] == scene]
        sums = {key: math.fsum(row[key] for row in chosen) for key in ADDITIVE}
        n, b = sums['count'], sums['boundary_count']
        if n <= 0:
            raise ValueError('scene without valid labels')
        by_scene[scene] = dict(AbsRel=sums['absrel_sum']/n, MAE=sums['abs_sum']/n,
            RMSE=math.sqrt(sums['sq_sum']/n), log_RMSE=math.sqrt(sums['log_sq_sum']/n),
            delta1=sums['delta1_count']/n, boundary_AbsRel=sums['boundary_absrel_sum']/b if b else None,
            boundary_MAE=sums['boundary_abs_sum']/b if b else None, **sums)
    metrics = {key: float(np.mean([row[key] for row in by_scene.values() if row[key] is not None]))
               if any(row[key] is not None for row in by_scene.values()) else None for key in METRICS}
    return dict(rows=rows, summary=dict(metrics=metrics, by_scene=by_scene,
                total_count=float(valid.sum()), total_positions=int(valid.size)))


def comparison(candidate, control):
    candidate, control = np.asarray(candidate, float), np.asarray(control, float)
    if candidate.shape != (5, 2) or control.shape != candidate.shape:
        raise ValueError('five-seed two-scene matrices required')
    if not np.isfinite(candidate).all() or not np.isfinite(control).all() or control.mean() <= 0:
        raise ValueError('finite positive-error control required')
    delta = candidate-control
    return dict(candidate_mean=float(candidate.mean()), control_mean=float(control.mean()),
                relative_change=float(candidate.mean()/control.mean()-1.),
                per_seed_difference=delta.mean(1).tolist(), per_scene_difference=delta.mean(0).tolist(),
                improved_seeds=int((delta.mean(1) < 0).sum()), improved_scenes=int((delta.mean(0) < 0).sum()))


def development_gate(matrices, timings):
    """Choices and thresholds are fixed before any task score; no intervals."""
    point = min(POINT_CONTROLS, key=lambda role: float(np.mean(matrices[role]['AbsRel'])))
    compression = min(COMPRESSION_CONTROLS, key=lambda role: float(np.mean(matrices[role]['AbsRel'])))
    pos = comparison(matrices['full4']['AbsRel'], matrices[point]['AbsRel'])
    single = comparison(matrices['full4']['AbsRel'], matrices['singleview']['AbsRel'])
    comp = comparison(matrices['full4']['AbsRel'], matrices[compression]['AbsRel'])
    rmse_point = comparison(matrices['full4']['RMSE'], matrices[point]['RMSE'])
    rmse_comp = comparison(matrices['full4']['RMSE'], matrices[compression]['RMSE'])
    if any(not math.isfinite(timings[role]) or timings[role] <= 0 for role in ROLES):
        raise ValueError('finite fresh paired timing required for every role')
    ratio = timings['full4']/timings[compression]
    accurate = comp['relative_change'] <= -.02 and ratio <= 1.25
    efficient = comp['relative_change'] <= .01 and ratio <= .5
    conditions = dict(point_gain_at_least_2pct=pos['relative_change'] <= -.02,
        point_both_scenes=pos['improved_scenes'] == 2, point_four_seeds=pos['improved_seeds'] >= 4,
        singleview_strict_gain=single['relative_change'] < 0,
        singleview_both_scenes=single['improved_scenes'] == 2, singleview_four_seeds=single['improved_seeds'] >= 4,
        compression_accuracy_or_efficiency=accurate or efficient,
        point_rmse_guard=rmse_point['relative_change'] <= .02,
        compression_rmse_guard=rmse_comp['relative_change'] <= .02)
    return dict(passed=all(conditions.values()), conditions=conditions, point_control=point,
                compression_control=compression, versus_point=pos, versus_singleview=single,
                versus_compression=comp, rmse_versus_point=rmse_point, rmse_versus_compression=rmse_comp,
                timing_ratio=ratio, accuracy_branch=accurate, efficiency_branch=efficient,
                confidence_intervals=None, scope='repeated old development; not independent confirmation')


def expected_batch_hash(seed):
    rng = np.random.default_rng(seed); digest = hashlib.sha256()
    for _ in range(1000):
        digest.update(rng.integers(576, size=8).astype('<i8').tobytes())
    return digest.hexdigest()


def analyze(runs, out):
    runs, out = Path(runs), Path(out)
    if out.exists():
        raise ValueError('refuse existing analysis directory')
    lock_path = runs/'lock.json'; lock = json.loads(lock_path.read_text()); lock_sha = sha(lock_path)
    if (runs/'lock.sha256').read_text().strip() != lock_sha:
        raise ValueError('training lock changed')
    if lock['roles'] != list(ROLES) or lock['seeds'] != list(SEEDS) or lock['test_access_authorized'] is not False:
        raise ValueError('development lock mismatch')
    for name, digest in lock['source_hashes'].items():
        if sha(ROOT/name) != digest or sha(runs/'source'/name) != digest:
            raise ValueError('frozen source changed')
    promotion = lock['stress_promotion']; stress_root = Path(promotion['root'])
    for name, digest in promotion['artifacts_sha256'].items():
        if sha(stress_root/name) != digest:
            raise ValueError('stress promotion result changed')
    audit_path = Path(promotion['audit_path'])
    if sha(audit_path) != promotion['audit_sha256']:
        raise ValueError('independent stress audit changed')
    audit = json.loads(audit_path.read_text())
    stress_summary = json.loads((stress_root/'summary.json').read_text())
    if (stress_summary['gates']['passed'] is not True or audit.get('passed') is not True
            or audit.get('stress_lock_sha256') != promotion['artifacts_sha256']['lock.json']
            or audit.get('stress_summary_sha256') != promotion['artifacts_sha256']['summary.json']
            or audit.get('stress_completion_sha256') != promotion['artifacts_sha256']['completion.json']):
        raise ValueError('failed/unbound stress promotion evidence')
    identities, targets, valid, boundary, raw_depth = [], [], [], [], []
    for record in lock['old_data']['val']:
        directory = Path(record['path'])
        for file, key in (('manifest.json', 'manifest_sha256'), ('inputs.npz', 'input_sha256'), ('labels.npz', 'label_sha256')):
            if sha(directory/file) != record[key]:
                raise ValueError('old validation data changed')
        with np.load(directory/'labels.npz', allow_pickle=False) as data:
            targets.append(data['target']); valid.append(data['valid']); boundary.append(data['boundary'])
        with np.load(directory/'inputs.npz', allow_pickle=False) as data:
            raw_depth.append(data['q_depth'].clip(.1, 20))
        identities.extend(dict(scene=record['scene'], q_index=q, k_index=k, pair_id=f'{q}:{k}') for q, k in record['pairs'])
    target = np.concatenate(targets); valid = np.concatenate(valid); boundary = np.concatenate(boundary)
    scenes = sorted({row['scene'] for row in identities})
    if len(identities) != 128 or len(scenes) != 2 or len({(r['scene'], r['pair_id']) for r in identities}) != 128:
        raise ValueError('development validation population incomplete')
    matrices, records, interventions, initializations, maximum_error = {}, {}, {}, {}, 0.
    for role in ROLES:
        matrices[role] = {metric: [] for metric in METRICS}; records[role] = {}; interventions[role] = {}
        for seed in SEEDS:
            directory = runs/role/str(seed); run = json.loads((directory/'run.json').read_text())
            if (run['role'] != role or run['seed'] != seed or run['steps'] != 1000 or run['parameter_count'] != 41345
                    or run['lock_sha256'] != lock_sha or run['source_hashes'] != lock['source_hashes']):
                raise ValueError('run identity/lock/capacity mismatch')
            expected_temperature = 5. if role == 'singleview' else lock['geometry']['configurations'][role]['temperature']
            if run['temperature'] != expected_temperature:
                raise ValueError('content temperature changed')
            for file, key in (('best.pt', 'checkpoint_sha256'), ('val_predictions.npz', 'prediction_sha256'),
                              ('validation.json', 'validation_sha256'), ('interventions.json', 'interventions_sha256')):
                if sha(directory/file) != run[key]:
                    raise ValueError('run artifact changed')
            if run['batch_sequence_sha256'] != expected_batch_hash(seed):
                raise ValueError('unpaired training batches')
            if seed in initializations and initializations[seed] != run['initialization_sha256']:
                raise ValueError('unpaired parameter initialization')
            initializations[seed] = run['initialization_sha256']
            history = run['history']
            if [entry['step'] for entry in history] != list(range(0, 1001, 100)) or len(run['training_losses']) != 1000:
                raise ValueError('incomplete training history')
            if not np.isfinite(run['training_losses']).all():
                raise ValueError('nonfinite training history')
            chosen = min(history, key=lambda entry: entry['validation']['metrics']['AbsRel'])
            if chosen['step'] != run['best_step'] or chosen['validation']['metrics']['AbsRel'] != run['best_validation_AbsRel']:
                raise ValueError('earliest validation selection mismatch')
            gradients = run['first_step_gradients']
            if set(gradients) != PARAMETER_NAMES:
                raise ValueError('incomplete parameter gradient evidence')
            for name, value in gradients.items():
                if not value['present'] or not value['finite'] or value['l1'] is None or not math.isfinite(value['l1']):
                    raise ValueError('invalid first-step gradient evidence')
                if (name != 'qk.weight' or run['temperature'] > 0) and value['l1'] <= 0:
                    raise ValueError('inactive actual learned parameter '+name)
            with np.load(directory/'val_predictions.npz', allow_pickle=False) as arrays:
                if set(arrays.files) != {'depth', 'zero_v', 'reverse_v'}:
                    raise ValueError('missing selected/intervention raw predictions')
                actual = raw_metrics(arrays['depth'], target, valid, boundary, identities)
                reported = json.loads((directory/'validation.json').read_text())
                maximum_error = max(maximum_error, check_close(actual, reported))
                check_close(actual['summary'], chosen['validation'])
                mode_records = json.loads((directory/'interventions.json').read_text())
                if set(mode_records) != {'zero_v', 'reverse_v'}:
                    raise ValueError('missing frozen V interventions')
                interventions[role][str(seed)] = {}
                for mode in ('zero_v', 'reverse_v'):
                    changed = raw_metrics(arrays[mode], target, valid, boundary, identities)
                    reported_change = mode_records[mode]
                    maximum_error = max(maximum_error, check_close(changed,
                        {key: reported_change[key] for key in ('rows', 'summary')}))
                    change = float(np.abs(arrays[mode].astype(float)-arrays['depth'].astype(float)).mean())
                    check_close(change, reported_change['mean_absolute_output_change_m'], tolerance=1e-7)
                    interventions[role][str(seed)][mode] = dict(summary=changed['summary'], mean_absolute_output_change_m=change)
                for metric in METRICS:
                    matrices[role][metric].append([actual['summary']['by_scene'][scene][metric] for scene in scenes])
            records[role][str(seed)] = dict(run_sha256=sha(directory/'run.json'), best_step=run['best_step'],
                prediction_sha256=run['prediction_sha256'], checkpoint_sha256=run['checkpoint_sha256'],
                first_step_gradients=gradients, training_seconds=run['seconds'])
    timing_path = runs/'timing.json'; timing = json.loads(timing_path.read_text())
    if timing['lock_sha256'] != lock_sha or timing['geometry_recomputed'] is not True:
        raise ValueError('timing must use this lock and fresh geometry')
    rows = timing['rows']; expected = {(role, seed, scene) for role in ROLES for seed in SEEDS for scene in
        sorted(row['scene'] for split in lock['old_data'].values() for row in split)}
    actual_keys = [(row['role'], row['seed'], row['scene']) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected:
        raise ValueError('incomplete paired timing population')
    expected_runs = {str((runs/role/str(seed)/'run.json').resolve()) for role in ROLES for seed in SEEDS}
    recorded_runs = {str(Path(path).resolve()): digest for path, digest in timing['run_hashes'].items()}
    if len(recorded_runs) != 45 or set(recorded_runs) != expected_runs:
        raise ValueError('timing does not bind all selected runs')
    for path, digest in recorded_runs.items():
        if sha(path) != digest:
            raise ValueError('timing training record changed')
    first_pairs = {record['scene']: record['pairs'][0] for split in lock['old_data'].values() for record in split}
    times = {}
    for role in ROLES:
        per_role = []
        for row in rows:
            if row['role'] != role:
                continue
            if row['warmups'] != 3 or len(row['samples_ms']) != 15:
                raise ValueError('timing repeat contract mismatch')
            if (row['pair'] != first_pairs[row['scene']]
                    or row['checkpoint_sha256'] != records[role][str(row['seed'])]['checkpoint_sha256']):
                raise ValueError('timing first-pair/checkpoint identity mismatch')
            if not np.isfinite(row['samples_ms']).all() or min(row['samples_ms']) <= 0:
                raise ValueError('nonpositive/nonfinite elapsed time')
            median = float(np.median(row['samples_ms']))
            check_close(median, row['median_ms']); per_role.append(median)
        times[role] = float(np.mean(per_role))
    check_close(times, timing['mean_median_ms'])
    common_rows = timing['common_dense_projection_cpu']
    if len(common_rows) != 5 or {row['scene'] for row in common_rows} != set(first_pairs):
        raise ValueError('missing common dense projection timings')
    for row in common_rows:
        if (row['pair'] != first_pairs[row['scene']] or len(row['samples_ms']) != 15
                or row['warmups'] != 1 or not np.isfinite(row['samples_ms']).all() or min(row['samples_ms']) <= 0):
            raise ValueError('common projection timing contract mismatch')
        check_close(float(np.median(row['samples_ms'])), row['median_ms'])
    gate = development_gate(matrices, times)
    raw = raw_metrics(np.concatenate(raw_depth), target, valid, boundary, identities)
    summary = dict(gate=gate, models={role: {metric: dict(matrix=values,
        mean=float(np.mean(values)) if all(x is not None for row in values for x in row) else None)
        for metric, values in metrics.items()} for role, metrics in matrices.items()},
        scene_order=scenes, seed_order=list(SEEDS), raw_baseline=raw['summary'], runs=records,
        frozen_value_interventions=interventions, timing=timing,
        lock_sha256=lock_sha, timing_sha256=sha(timing_path), analyzer_sha256=sha(__file__),
        independently_recomputed_raw_predictions=True, maximum_metric_absolute_error=maximum_error,
        valid_count=int(valid.sum()), boundary_count=int(boundary.sum()), total_positions=int(valid.size),
        content_temperatures={role:5. if role == 'singleview' else lock['geometry']['configurations'][role]['temperature'] for role in ROLES},
        stored_parameters_per_role=41345,
        zero_temperature_note='A T=0 role has zero data gradient for its 16,384 QK weights; equal stored capacity does not imply equal effective routing capacity.',
        independent_confirmation=False, historical_R7_R9_failures_unchanged=True)
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# Round12 actual-V depth development', '',
        'Gate: **'+('passed' if gate['passed'] else 'failed')+'**. Repeated old development only.', '',
        '| Role | Content T | AbsRel | RMSE | Incremental PE+V/head ms |', '|---|---:|---:|---:|---:|']
    for role in ROLES:
        lines.append(f"| {role} | {summary['content_temperatures'][role]:g} | {summary['models'][role]['AbsRel']['mean']:.7f} | {summary['models'][role]['RMSE']['mean']:.7f} | {times[role]:.4f} |")
    lines += ['', f"Strong position control: {gate['point_control']}; strongest compression control: {gate['compression_control']}.",
        '', 'All raw selected/intervention depths were independently recomputed against the same original query sensor labels. '
        'No new-scene generalization or confidence-interval claim is made. Shared predictors, dense projection and transfers '
        'are excluded from the incremental timing and disclosed separately.', '',
        'Every role stores 41,345 parameters. The frozen full1 T=0 control has zero routing gradient for its 16,384 QK weights; '
        'the remaining 24,961 parameters receive data gradients. Full4 and quad16 share T=5 and the same head. '
        'Per-parameter gradient evidence is retained; stored capacity alone does not establish effective-capacity equality.', '']
    (out/'README.md').write_text('\n'.join(lines)); print(json.dumps(gate, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    analyze(args.runs, args.out)


if __name__ == '__main__':
    main()
