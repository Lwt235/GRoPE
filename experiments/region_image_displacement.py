"""Fixed old-data association diagnostic; no model inference or new observations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/region_image_displacement/diagnostic1'
PARENT = ROOT / 'results/region_rotary_relation/attempt1'
PLAN = ROOT / 'docs/region_image_displacement_plan.md'
DATA_LOCK = ROOT / 'results/region_hrope_control/preparation/data_lock.json'
CAMERA = ROOT / 'results/region_hrope_control/camera_data/manifest.json'
SEEDS = (11, 22, 33)
ROLES = ('relative5', 'distance_mlp', 'geotransformer_rpe',
         'fixed_isotropic5', 'predicted_hrope', 'fixed_hrope')
COMPARISONS = ('fixed_hrope', 'distance_mlp', 'fixed_isotropic5',
               'geotransformer_rpe', 'predicted_hrope')
GROUPS = ('low', 'high', 'undefined')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    assert not path.exists(), f'Preserve existing artifact: {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def nullable(value):
    return float(value) if np.isfinite(value) else None


def freeze():
    assert not OUT.exists()
    data, camera = read(DATA_LOCK), read(CAMERA)
    parent = read(PARENT / 'status.json')
    assert parent['scientific_review_complete'] and parent['state'] == 'complete'
    caches = {name: dict(geometry=data['caches'][name], camera=camera['caches'][name])
              for name in ('fit0', 'eval_base')}
    for entry in caches.values():
        for row in entry.values():
            assert sha(row['path']) == row['sha256']
    sources = [Path(__file__), PLAN, DATA_LOCK, CAMERA, PARENT / 'result.json',
               PARENT / 'review.json', PARENT / 'status.json']
    write(OUT / 'lock.json', dict(created_unix=time.time(),
        source_sha256={str(p): sha(p) for p in sources}, caches=caches,
        roles=list(ROLES), seeds=list(SEEDS), comparisons=list(COMPARISONS),
        bootstrap_seeds=list(range(99001, 99006)), bootstrap_replicates=10000,
        minimum_pairs_per_group_per_scene=8, minimum_eligible_scenes=3,
        cutoff='Median finite fit0 target-weighted bidirectional image displacement.',
        old_data_only=True, new_training=False, new_inference=False,
        new_pose_estimation=False, fresh_confirmation=False, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('IMAGE_DISPLACEMENT_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    lock = read(OUT / 'lock.json')
    for path, digest in lock['source_sha256'].items():
        assert sha(path) == digest, path
    return lock


def load_data(name, lock):
    entry = lock['caches'][name]
    for row in entry.values():
        assert sha(row['path']) == row['sha256']
    with np.load(entry['geometry']['path'], allow_pickle=False) as z:
        data = {k: z[k] for k in ('ids', 'scenes', 'overlap')}
    with np.load(entry['camera']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], data['ids'])
        data['image_positions'] = z['image_positions']
    return data


def attributes(data):
    overlap = data['overlap'].astype(np.float64)
    xy = data['image_positions'].astype(np.float64)
    assert xy.shape == (len(overlap), 2, 64, 2)
    distance = np.linalg.norm(xy[:, 0, :, None] - xy[:, 1, None], axis=-1)
    count = np.zeros(len(overlap), dtype=np.int64)
    total = np.zeros(len(overlap), dtype=np.float64)
    nearest_correct = np.zeros(len(overlap), dtype=np.int64)
    for truth, delta in ((overlap, distance),
                         (overlap.swapaxes(1, 2), distance.swapaxes(1, 2))):
        valid = truth.max(-1) >= .3
        weights = truth * (truth >= .1)
        value = (weights * delta).sum(-1) / np.maximum(weights.sum(-1), 1e-12)
        count += valid.sum(-1)
        total += (value * valid).sum(-1)
        nearest = delta.argmin(-1)
        correct = np.take_along_axis(truth, nearest[..., None], axis=-1)[..., 0] >= .3
        nearest_correct += (correct & valid).sum(-1)
    displacement = np.divide(total, count, out=np.full(len(overlap), np.nan), where=count > 0)
    return dict(displacement=displacement, valid_queries=count,
                nearest_xy_correct_queries=nearest_correct)


def independent_attribute_error(data, attr):
    """Explicit per-query candidate loop for the first eight existing pairs."""
    maximum = 0.
    for b in range(min(8, len(data['ids']))):
        values, valid_count, nearest_correct = [], 0, 0
        for view in range(2):
            truth = data['overlap'][b].astype(np.float64)
            if view == 1:
                truth = truth.T
            for q in range(64):
                if max(truth[q]) < .3:
                    continue
                valid_count += 1
                distances = [math.dist(data['image_positions'][b, view, q].tolist(),
                                      data['image_positions'][b, 1-view, k].tolist())
                             for k in range(64)]
                included = [k for k in range(64) if truth[q, k] >= .1]
                denominator = math.fsum(float(truth[q, k]) for k in included)
                values.append(math.fsum(float(truth[q, k]) * distances[k]
                                        for k in included) / denominator)
                closest = min(range(64), key=distances.__getitem__)
                nearest_correct += int(truth[q, closest] >= .3)
        assert valid_count == attr['valid_queries'][b]
        assert nearest_correct == attr['nearest_xy_correct_queries'][b]
        if values:
            maximum = max(maximum, abs(math.fsum(values) / len(values) - attr['displacement'][b]))
        else:
            assert np.isnan(attr['displacement'][b])
    assert maximum < 1e-12
    return maximum


def calibrate():
    lock = check()
    assert not (OUT / 'calibration.json').exists()
    data = load_data('fit0', lock)
    attr = attributes(data)
    error = independent_attribute_error(data, attr)
    values = attr['displacement'][np.isfinite(attr['displacement'])]
    assert len(values) > 0
    threshold = float(np.median(values))
    np.savez(OUT / 'fit_attributes.npz', ids=data['ids'], scenes=data['scenes'], **attr)
    write(OUT / 'calibration.json', dict(completed_unix=time.time(), threshold=threshold,
        pairs=len(data['ids']), defined_pairs=len(values), undefined_pairs=len(data['ids'])-len(values),
        quantiles={str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
        independent_attribute_max_error=error, lock_sha256=sha(OUT / 'lock.json'),
        attributes_sha256=sha(OUT / 'fit_attributes.npz'),
        evaluation_outcomes_accessed=False, threshold_selected_without_model_results=True))
    print('IMAGE_DISPLACEMENT_CALIBRATED', threshold, 'defined', len(values), flush=True)


def group_summary(mask, scenes, attr):
    rows = []
    for scene in sorted(set(scenes[mask].tolist())):
        selected = mask & (scenes == scene)
        valid = int(attr['valid_queries'][selected].sum())
        correct = int(attr['nearest_xy_correct_queries'][selected].sum())
        rows.append(dict(scene=scene, pairs=int(selected.sum()), valid_queries=valid,
            valid_query_fraction=valid / (128 * int(selected.sum())),
            nearest_xy_recall=correct / valid if valid else None))
    return dict(pairs=int(mask.sum()), scenes=rows,
        scene_macro_valid_query_fraction=float(np.mean([x['valid_query_fraction'] for x in rows])) if rows else None,
        scene_macro_nearest_xy_recall=float(np.mean([x['nearest_xy_recall'] for x in rows
            if x['nearest_xy_recall'] is not None])) if any(x['nearest_xy_recall'] is not None for x in rows) else None)


def error_quantiles(values):
    return dict(median=nullable(np.median(values)),
                p90=nullable(np.quantile(values, .9, method='higher'))) if len(values) else None


def paired_conditional(left, right, seed, repeats):
    """Axes are seed, common eligible scene, and low/high group."""
    assert left.shape == right.shape and left.ndim == 3 and left.shape[2] == 2
    ns, nc, _ = left.shape
    difference = 100 * (left - right)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(ns, size=(repeats, ns))
    scenes = rng.integers(nc, size=(repeats, nc))
    draws = difference[seeds[:, :, None], scenes[:, None, :]].mean((1, 2))
    alternative = 100 * (left[seeds[:, :, None], scenes[:, None, :]].mean((1, 2))
                         - right[seeds[:, :, None], scenes[:, None, :]].mean((1, 2)))
    error = float(np.max(np.abs(draws - alternative)))
    # Rebuild selected draws through explicit seed and scene multiplicities.
    for i in range(min(64, repeats)):
        value = np.zeros(2)
        for a in range(ns):
            for b in range(nc):
                weight = int((seeds[i] == a).sum()) * int((scenes[i] == b).sum()) / (ns * nc)
                value += weight * 100 * (left[a, b] - right[a, b])
        error = max(error, float(np.max(np.abs(value - draws[i]))))
    assert error < 1e-10
    by_seed = difference.mean(1)
    by_scene = difference.mean(0)
    intervals = {name: np.quantile(values, [.025, .975]).tolist()
                 for name, values in [('low', draws[:, 0]), ('high', draws[:, 1]),
                                      ('high_minus_low', draws[:, 1] - draws[:, 0])]}
    return dict(mean_gain_pp=dict(low=float(difference[..., 0].mean()),
        high=float(difference[..., 1].mean()),
        high_minus_low=float((difference[..., 1] - difference[..., 0]).mean())),
        paired_seed_gain_pp=dict(low=by_seed[:, 0].tolist(), high=by_seed[:, 1].tolist(),
                                high_minus_low=(by_seed[:, 1] - by_seed[:, 0]).tolist()),
        paired_scene_gain_pp=dict(low=by_scene[:, 0].tolist(), high=by_scene[:, 1].tolist(),
                                 high_minus_low=(by_scene[:, 1] - by_scene[:, 0]).tolist()),
        crossed_intervals_95_pp=intervals, bootstrap_seed=seed, bootstrap_replicates=repeats,
        independent_bootstrap_max_error_pp=error)


def analyze():
    lock = check()
    calibration = read(OUT / 'calibration.json')
    assert calibration['lock_sha256'] == sha(OUT / 'lock.json')
    assert calibration['attributes_sha256'] == sha(OUT / 'fit_attributes.npz')
    assert not (OUT / 'result.json').exists()
    data = load_data('eval_base', lock)
    attr = attributes(data)
    attribute_error = independent_attribute_error(data, attr)
    finite = np.isfinite(attr['displacement'])
    group = np.full(len(finite), 'undefined', dtype='<U9')
    group[finite] = np.where(attr['displacement'][finite] > calibration['threshold'], 'high', 'low')
    scenes = data['scenes']
    scene_names = sorted(set(scenes.tolist()))
    counts = {scene: {g: int(((scenes == scene) & (group == g)).sum()) for g in GROUPS}
              for scene in scene_names}
    eligible = [s for s in scene_names if min(counts[s]['low'], counts[s]['high'])
                >= lock['minimum_pairs_per_group_per_scene']]
    eligible_mask = np.isin(scenes, eligible)
    np.savez(OUT / 'evaluation_attributes.npz', ids=data['ids'], scenes=scenes, group=group, **attr)
    populations = {g: dict(all_scenes=group_summary(group == g, scenes, attr),
                          common_eligible_scenes=group_summary((group == g) & eligible_mask, scenes, attr))
                   for g in GROUPS}
    parent = read(PARENT / 'result.json')
    assert len(parent['records']) == 20
    records, cubes, global_error = [], {}, 0.
    for record in parent['records']:
        receipt = record['receipt']
        assert sha(receipt['path']) == receipt['sha256']
        with np.load(receipt['path'], allow_pickle=False) as z:
            saved = {k: z[k] for k in ('ids', 'scenes', 'success', 'rmse_m', 'valid')}
        np.testing.assert_array_equal(saved['ids'], data['ids'])
        np.testing.assert_array_equal(saved['scenes'], scenes)
        whole = float(np.mean([saved['success'][scenes == s].mean() for s in scene_names]))
        global_error = max(global_error, abs(whole - record['macro_success_rate']))
        by_scene = {s: {g: float(saved['success'][(scenes == s) & (group == g)].mean())
                       if counts[s][g] else None for g in GROUPS} for s in scene_names}
        summaries = {}
        for g in GROUPS:
            keep = (group == g) & eligible_mask
            all_keep = group == g
            summaries[g] = dict(common_eligible_scene_macro_success=float(np.mean([
                by_scene[s][g] for s in eligible if by_scene[s][g] is not None]))
                if any(by_scene[s][g] is not None for s in eligible) else None,
                all_available_scene_macro_success=float(np.mean([
                    by_scene[s][g] for s in scene_names if by_scene[s][g] is not None]))
                    if any(by_scene[s][g] is not None for s in scene_names) else None,
                common_eligible_pairs=int(keep.sum()), all_pairs=int(all_keep.sum()),
                common_eligible_rmse_m=error_quantiles(saved['rmse_m'][keep]),
                all_group_rmse_m=error_quantiles(saved['rmse_m'][all_keep]),
                invalid_estimates=int((~saved['valid'][all_keep]).sum()))
        records.append(dict(role=record['role'], seed=record['seed'],
            whole_success=whole, by_scene=by_scene, groups=summaries, receipt_sha256=receipt['sha256']))
        if eligible:
            cubes[record['role'], record['seed']] = np.asarray([
                [by_scene[s][g] for g in ('low', 'high')] for s in eligible])
    assert global_error < 1e-12
    comparisons = []
    if eligible:
        left = np.stack([cubes['relative5', s] for s in SEEDS])
        for role, seed in zip(COMPARISONS, lock['bootstrap_seeds']):
            stats = paired_conditional(left, np.stack([cubes[role, s] for s in SEEDS]),
                                       seed, lock['bootstrap_replicates'])
            comparisons.append(dict(left='relative5', right=role, **stats))
    enough = len(eligible) >= lock['minimum_eligible_scenes']
    primary_signal = bool(enough and comparisons
        and min(comparisons[0]['paired_seed_gain_pp']['high']) > 0
        and min(comparisons[0]['paired_seed_gain_pp']['high_minus_low']) > 0
        and comparisons[0]['crossed_intervals_95_pp']['high'][0] > 0
        and comparisons[0]['crossed_intervals_95_pp']['high_minus_low'][0] > 0)
    write(OUT / 'result.json', dict(completed_unix=time.time(), lock_sha256=sha(OUT / 'lock.json'),
        calibration_sha256=sha(OUT / 'calibration.json'),
        evaluation_attributes_sha256=sha(OUT / 'evaluation_attributes.npz'),
        threshold=calibration['threshold'], counts_by_scene=counts, eligible_scenes=eligible,
        sufficient_common_scenes=enough, populations=populations, records=records,
        comparisons=comparisons, primary_conditional_numerical_signal=primary_signal,
        original_all_pair_result_unchanged=True, fresh_confirmation=False,
        scientific_review_pending=True, goal_complete=False))
    write(OUT / 'independent_checks.json', dict(passed=True, completed_unix=time.time(),
        result_sha256=sha(OUT / 'result.json'), global20_group_success_max_error=global_error,
        fit_attribute_max_error=calibration['independent_attribute_max_error'],
        evaluation_attribute_max_error=attribute_error,
        independent_attribute_scope='Explicit query/candidate formula on first eight fit and first eight evaluation pairs.',
        maximum_bootstrap_error_pp=max((x['independent_bootstrap_max_error_pp'] for x in comparisons), default=0.),
        all_prediction_and_pose_identities_matched=True,
        group_accounting_sum=sum(sum(x.values()) for x in counts.values()),
        new_model_inference=False, new_training=False, new_pose_estimation=False))
    print('IMAGE_DISPLACEMENT_ANALYZED', 'eligible_scenes', len(eligible),
          'primary_conditional_signal', primary_signal, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'calibrate', 'analyze'))
    globals()[parser.parse_args().action]()
