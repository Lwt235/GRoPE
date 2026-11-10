"""Fixed pose metrics, paired uncertainty, and independent CPU score review."""
import argparse
import io
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.round65_common import read, write, sha
from experiments.region_pose_utility import OUT, check, arrays


def paired_statistics(left, right, *, seed, replicates=10000):
    # Rows are paired training seeds; columns are whole scenes, not frame pairs.
    delta = 100. * (np.asarray(left, np.float64) - np.asarray(right, np.float64))
    assert delta.ndim == 2 and delta.shape[0] == 3
    srng, grng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    ns, ng = delta.shape
    sw = srng.multinomial(ns, np.full(ns, 1. / ns), size=replicates) / ns
    gw = grng.multinomial(ng, np.full(ng, 1. / ng), size=replicates) / ng
    draws = dict(seed=sw @ delta.mean(1), scenes=gw @ delta.mean(0),
                 crossed=np.einsum('bi,ij,bj->b', sw, delta, gw))
    intervals = {key: np.quantile(value, [.025, .975]).tolist() for key, value in draws.items()}
    # Independent resampling arithmetic: resample each absolute accuracy first.
    independently = 100. * (np.sum((sw @ left) * gw, 1) - np.sum((sw @ right) * gw, 1))
    error = float(np.max(np.abs(draws['crossed'] - independently)))
    assert error < 1e-10
    return dict(mean_gain_pp=float(delta.mean()), paired_seed_gains_pp=delta.mean(1).tolist(),
        by_scene_gain_pp=delta.mean(0).tolist(), intervals_95_pp=intervals,
        bootstrap_seed=seed, bootstrap_replicates=replicates, independent_draw_max_error=error)


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def load_estimate(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def freeze_analysis():
    lock = check()
    destination = OUT / 'analysis_lock.json'
    assert not destination.exists()
    write(destination, dict(created_unix=time.time(), study_lock_sha256=sha(OUT / 'lock.json'),
        source_sha256=sha(__file__), test_sha256=sha(Path(__file__).resolve().parents[1] / 'tests/test_region_pose_analysis.py'),
        cpu_checks_sha256=sha(OUT / 'analysis_cpu_checks.txt'),
        bootstrap_seeds=lock['plan']['bootstrap_seeds'], bootstrap_replicates=10000,
        fresh_confirmation=False, pose_scores_read=False))
    print('POSE_ANALYSIS_FROZEN', sha(destination), flush=True)


def summarize():
    lock = check()
    analysis_lock = read(OUT / 'analysis_lock.json')
    assert analysis_lock['source_sha256'] == sha(__file__)
    assert analysis_lock['study_lock_sha256'] == sha(OUT / 'lock.json')
    state = read(OUT / 'status.json')
    assert state['state'] == 'estimates_complete' and len(state['completed']) == 14
    assert not (OUT / 'result.json').exists()
    cache = arrays(lock['plan']['cache']['path'], ('ids', 'scenes', 'centers', 'world_centers', 'scales'))
    centers = cache['centers'].astype(np.float64)
    world = cache['world_centers'].astype(np.float64)
    scene_names = sorted(set(cache['scenes'].tolist()))
    gt = []
    for index in range(len(centers)):
        transforms = []
        for view in range(2):
            a, b = centers[index, view], world[index, view]
            rotation, _ = Rotation.align_vectors(b - b.mean(0), a - a.mean(0))
            matrix = np.eye(4)
            matrix[:3, :3] = rotation.as_matrix()
            matrix[:3, 3] = b.mean(0) - rotation.apply(a.mean(0))
            transforms.append(matrix)
        gt.append(np.linalg.solve(transforms[1], transforms[0]))
    gt = np.stack(gt)
    records, accuracy, maximum_errors = [], {}, dict(rmse_m=0., rotation_error_deg=0., centered_translation_error_m=0.)
    consensus_count_differences = 0
    for receipt in state['completed']:
        assert sha(receipt['path']) == receipt['sha256']
        row = load_estimate(receipt['path'])
        assert np.array_equal(row['ids'], cache['ids']) and np.array_equal(row['scenes'], cache['scenes'])
        assert len(row['success']) == 480
        if receipt['input_prediction']:
            prediction = receipt['input_prediction']
            assert sha(prediction['path']) == prediction['sha256']
            saved = arrays(prediction['path'], ('ids', 'logits'))
            assert np.array_equal(saved['ids'], cache['ids'])
            picked = saved['logits'].argmax(-1)
        elif receipt['role'] == 'gt_nearest':
            mapped = np.einsum('bij,bnj->bni', gt[:, :3, :3], centers[:, 0]) + gt[:, None, :3, 3]
            picked = np.square(mapped[:, :, None] - centers[:, None, 1]).sum(-1).argmin(-1)
        else:
            picked = None
        for index in range(480):
            if not row['valid'][index]:
                assert not row['success'][index] and np.isinf(row['rmse_m'][index])
                continue
            estimate = np.eye(4)
            estimate[:3, :3], estimate[:3, 3] = row['rotation'][index], row['translation'][index]
            np.testing.assert_allclose(estimate[:3, :3].T @ estimate[:3, :3], np.eye(3), atol=1e-8)
            assert abs(np.linalg.det(estimate[:3, :3]) - 1) < 1e-8
            points = np.column_stack((centers[index, 0], np.ones(64)))
            error = points @ (estimate - gt[index]).T
            rmse = float(np.linalg.norm(error[:, :3], 'fro') / np.sqrt(64) * cache['scales'][index])
            relative_rotation = estimate[:3, :3] @ gt[index, :3, :3].T
            angle = float(np.degrees(Rotation.from_matrix(relative_rotation).magnitude()))
            translation = float(np.linalg.norm((estimate - gt[index])[:3, 3]) * cache['scales'][index])
            for key, value in [('rmse_m', rmse), ('rotation_error_deg', angle), ('centered_translation_error_m', translation)]:
                maximum_errors[key] = max(maximum_errors[key], abs(value - row[key][index]))
            assert bool(rmse < .2) == row['success'][index], 'A primary decision changed in independent scoring.'
            if picked is not None:
                residual = points @ estimate.T
                actual_count = int((np.linalg.norm(residual[:, :3] - centers[index, 1, picked[index]], axis=1) <= .1).sum())
                consensus_count_differences += int(actual_count != row['consensus_count'][index])
        grouped = [float(row['success'][cache['scenes'] == scene].mean()) for scene in scene_names]
        accuracy[receipt['role'], receipt['seed']] = np.asarray(grouped)
        errors = {key: dict(median=finite_or_none(np.median(row[key])),
                           p90=finite_or_none(np.quantile(row[key], .9, method='higher')))
                  for key in ('rmse_m', 'rotation_error_deg', 'centered_translation_error_m')}
        records.append(dict(role=receipt['role'], seed=receipt['seed'], pairs=480,
            successful_pairs=int(row['success'].sum()), invalid_estimates=int((~row['valid']).sum()),
            macro_success_rate=float(np.mean(grouped)), micro_success_rate=float(row['success'].mean()),
            by_scene=dict(zip(scene_names, grouped)), all_pair_errors=errors,
            null_error_quantile_means_infinite_failed_estimates=True, receipt=receipt))
    assert maximum_errors['rmse_m'] < 1e-6
    assert maximum_errors['rotation_error_deg'] < 1e-5
    assert maximum_errors['centered_translation_error_m'] < 1e-6
    comparisons = []
    left = np.stack([accuracy['relative5', seed] for seed in (11, 22, 33)])
    for role, seed in zip(('distance_mlp', 'geotransformer_rpe', 'fixed_isotropic5'), (93001, 93002, 93003)):
        right = np.stack([accuracy[role, s] for s in (11, 22, 33)])
        stats = paired_statistics(left, right, seed=seed)
        signal = stats['mean_gain_pp'] >= 1 and min(stats['paired_seed_gains_pp']) > 0 and stats['intervals_95_pp']['crossed'][0] > 0
        comparisons.append(dict(left='relative5', right=role, **stats, numerical_signal=signal))
    write(OUT / 'result.json', dict(completed_unix=time.time(), study_lock_sha256=sha(OUT / 'lock.json'),
        analysis_lock_sha256=sha(OUT / 'analysis_lock.json'), scenes=scene_names, records=records,
        comparisons=comparisons, primary_numerical_signal=all(r['numerical_signal'] for r in comparisons[:2]),
        scientific_review_pending=True, fresh_confirmation=False, goal_complete=False))
    write(OUT / 'independent_checks.json', dict(completed_unix=time.time(), passed=True,
        result_sha256=sha(OUT / 'result.json'), implementation_sha256=sha(__file__),
        score_maximum_errors=maximum_errors, primary_success_decisions_changed=0,
        consensus_count_disagreements=consensus_count_differences,
        records=14, pairs_per_record=480, independent_ground_truth='SciPy alignment and homogeneous inverse composition',
        independent_uncertainty='Resample left and right absolute scene accuracies separately then subtract.',
        scientific_review_pending=True, goal_complete=False))
    print('POSE_RESULT_AND_INDEPENDENT_REVIEW_READY', sha(OUT / 'result.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze_analysis', 'summarize'))
    globals()[parser.parse_args().action]()
