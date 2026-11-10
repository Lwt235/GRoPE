"""Apply the unchanged pose estimator and independent analysis to a fixed new cohort."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.round65_common import read, write, sha
from experiments.region_pose_solver import estimate, recover_truth, score_transform
from experiments.region_pose_utility import arrays, configure_cuda
from experiments.region_pose_analysis import paired_statistics, finite_or_none, load_estimate
from experiments.region_pose_confirmation import OUT, check as check_evaluation, preserved


def freeze_inputs():
    evaluation = check_evaluation()
    state = read(OUT/'status.json')
    assert state['stage'] == 'inference_complete' and len(state['completed']) == 12
    data = read(OUT/'data_manifest.json')
    cache = data['caches']['base']
    assert cache['pairs'] > 0
    expected = {(m['role'], m['seed']):m['checkpoint_sha256'] for m in evaluation['models']}
    assert len({(m['role'],m['seed']) for m in state['completed']}) == 12
    for item in state['completed']:
        assert item['condition']=='base' and sha(item['path'])==item['sha256']
        assert item['checkpoint_sha256']==expected[item['role'],item['seed']]
    assert sha(cache['path'])==cache['sha256']
    preserved(OUT/'scoring_input_lock.json',dict(created_unix=time.time(),
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
        data_manifest_sha256=sha(OUT/'data_manifest.json'),
        inference_status_sha256=sha(OUT/'status.json'),
        plan=dict(evaluation['pose_rule'],cache=cache,predictions=state['completed']),
        note='Only derived input identities bound here; scientific rules and source frozen before new depth access.',
        new_fits=0,pose_scores_read=False,goal_complete=False))
    print('SCORING_INPUTS_BOUND',sha(OUT/'scoring_input_lock.json'),flush=True)


def check_scoring():
    check_evaluation()
    lock=read(OUT/'scoring_input_lock.json')
    assert lock['evaluation_lock_sha256']==sha(OUT/'evaluation_lock.json')
    assert lock['data_manifest_sha256']==sha(OUT/'data_manifest.json')
    assert lock['inference_status_sha256']==sha(OUT/'status.json')
    assert lock['plan']['alignment_rmse_threshold_m']==.2
    assert lock['plan']['consensus_radius_pair_units']==.1
    return lock


def run():
    lock = check_scoring()
    assert not (OUT / 'pose_status.json').exists(), 'Inspect a previous worker before resuming.'
    assert read(OUT / 'preflight.json')['passed']
    configure_cuda()
    plan = lock['plan']
    assert sha(plan['cache']['path']) == plan['cache']['sha256']
    data = arrays(plan['cache']['path'], ('ids', 'scenes', 'centers', 'world_centers', 'scales'))
    pair_count=len(data['ids'])
    assert pair_count==plan['cache']['pairs'] and len(np.unique(data['scenes']))>=2
    assert data['centers'].shape == data['world_centers'].shape == (pair_count, 2, 64, 3)
    centers = np.asarray(data['centers'], dtype=np.float64)
    world = np.asarray(data['world_centers'], dtype=np.float64)
    truth = [recover_truth(centers[i], world[i]) for i in range(pair_count)]
    truth_r, truth_t = np.stack([x[0] for x in truth]), np.stack([x[1] for x in truth])
    maximum_reconstruction_error = max(x[2] for x in truth)
    assert maximum_reconstruction_error < 2e-5
    np.savez(OUT / 'scoring_truth.npz', ids=data['ids'], scenes=data['scenes'], rotation=truth_r,
             translation=truth_t, scale=data['scales'])
    gpu_centers = torch.as_tensor(centers, device='cuda')
    state = dict(state='estimating', pid=os.getpid(), started_unix=time.time(),
        lock_sha256=sha(OUT / 'scoring_input_lock.json'), completed=[], active=None, goal_complete=False,
        maximum_scoring_coordinate_reconstruction_error=maximum_reconstruction_error,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])

    def save():
        state['updated_unix'] = time.time()
        write(OUT / 'pose_status.json', state)

    save()
    jobs = [dict(role='identity', seed=0), dict(role='gt_nearest', seed=0), *plan['predictions']]
    try:
        for job in jobs:
            role, seed = job['role'], job['seed']
            if 'path' in job:
                assert sha(job['path']) == job['sha256']
                prediction = arrays(job['path'], ('ids', 'logits'))
                assert np.array_equal(prediction['ids'], data['ids'])
                assert prediction['logits'].shape == (pair_count, 64, 64) and np.isfinite(prediction['logits']).all()
                indices = prediction['logits'].argmax(-1)
            elif role == 'gt_nearest':
                mapped = np.einsum('bij,bnj->bni', truth_r, centers[:, 0]) + truth_t[:, None]
                indices = np.square(mapped[:, :, None] - centers[:, None, 1]).sum(-1).argmin(-1)
            else:
                indices = None
            records = []
            started = time.time()
            state['active'] = dict(role=role, seed=seed, completed_pairs=0)
            save()
            for index in range(pair_count):
                if role == 'identity':
                    result = dict(rotation=np.eye(3), translation=np.zeros(3), valid=True,
                        consensus_count=0, truncated_squared_error=None, hypotheses_total=0,
                        distinct_target_hypotheses=0, nondegenerate_hypotheses=0, refits=0)
                else:
                    result = estimate(gpu_centers[index, 0], gpu_centers[index, 1],
                        torch.as_tensor(indices[index], device='cuda', dtype=torch.long),
                        radius=plan['consensus_radius_pair_units'], max_refits=plan['maximum_consensus_refits'])
                score = score_transform(centers[index, 0], result['rotation'], result['translation'],
                    truth_r[index], truth_t[index], data['scales'][index], valid=result['valid'])
                records.append(dict(**result, **score))
                if (index + 1) % 16 == 0:
                    state['active']['completed_pairs'] = index + 1
                    save()
            destination = OUT / 'estimates' / role
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / f'{seed}.npz'
            assert not path.exists()
            payload = {key: np.stack([r[key] for r in records]) for key in records[0]
                       if key != 'truncated_squared_error'}
            payload['truncated_squared_error'] = np.asarray([
                np.nan if row['truncated_squared_error'] is None else row['truncated_squared_error'] for row in records])
            np.savez(path, ids=data['ids'], scenes=data['scenes'], **payload)
            receipt = dict(role=role, seed=seed, path=str(path), sha256=sha(path),
                pairs=pair_count, seconds=time.time() - started, input_prediction=job if 'path' in job else None,
                interpretation='undeployable diagnostic' if role == 'gt_nearest' else 'identity baseline' if role == 'identity' else 'fixed learned matcher')
            state['completed'].append(receipt)
            save()
            print('POSE_ESTIMATES_COMPLETE', role, seed, receipt['seconds'], flush=True)
        state.update(state='estimates_complete', active=None, finished_unix=time.time(),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
        print('POSE_CONFIRMATION_ESTIMATES_COMPLETE', len(state['completed']), flush=True)
    except BaseException as error:
        state.update(state='failed', error_type=type(error).__name__, reason=str(error))
        save()
        raise


def summarize():
    lock = check_scoring()
    state = read(OUT / 'pose_status.json')
    assert state['state'] == 'estimates_complete' and len(state['completed']) == 14
    assert not (OUT / 'result.json').exists()
    cache = arrays(lock['plan']['cache']['path'], ('ids', 'scenes', 'centers', 'world_centers', 'scales'))
    pair_count=len(cache['ids'])
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
        assert len(row['success']) == pair_count
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
        for index in range(pair_count):
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
        records.append(dict(role=receipt['role'], seed=receipt['seed'], pairs=pair_count,
            successful_pairs=int(row['success'].sum()), invalid_estimates=int((~row['valid']).sum()),
            macro_success_rate=float(np.mean(grouped)), micro_success_rate=float(row['success'].mean()),
            by_scene=dict(zip(scene_names, grouped)), all_pair_errors=errors,
            null_error_quantile_means_infinite_failed_estimates=True, receipt=receipt))
    assert maximum_errors['rmse_m'] < 1e-6
    assert maximum_errors['rotation_error_deg'] < 1e-5
    assert maximum_errors['centered_translation_error_m'] < 1e-6
    comparisons = []
    left = np.stack([accuracy['relative5', seed] for seed in (11, 22, 33)])
    for role, seed in zip(('distance_mlp', 'geotransformer_rpe', 'fixed_isotropic5'), lock['plan']['bootstrap_seeds']):
        right = np.stack([accuracy[role, s] for s in (11, 22, 33)])
        stats = paired_statistics(left, right, seed=seed)
        signal = (role == 'fixed_isotropic5' or stats['mean_gain_pp'] >= 1) and min(stats['paired_seed_gains_pp']) > 0 and stats['intervals_95_pp']['crossed'][0] > 0
        comparisons.append(dict(left='relative5', right=role, **stats, numerical_signal=signal))
    write(OUT / 'result.json', dict(completed_unix=time.time(), evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
        scoring_input_lock_sha256=sha(OUT / 'scoring_input_lock.json'),
        analysis_source_sha256=sha(__file__), scenes=scene_names, records=records,
        comparisons=comparisons, primary_numerical_signal=all(r['numerical_signal'] for r in comparisons[:2]),
        scientific_review_pending=True, fresh_confirmation=True, goal_complete=False))
    write(OUT / 'independent_checks.json', dict(completed_unix=time.time(), passed=True,
        result_sha256=sha(OUT / 'result.json'), implementation_sha256=sha(__file__),
        score_maximum_errors=maximum_errors, primary_success_decisions_changed=0,
        consensus_count_disagreements=consensus_count_differences,
        records=14, pairs_per_record=pair_count, independent_ground_truth='SciPy alignment and homogeneous inverse composition',
        independent_uncertainty='Resample left and right absolute scene accuracies separately then subtract.',
        scientific_review_pending=True, goal_complete=False))
    print('POSE_CONFIRMATION_RESULT_READY', sha(OUT / 'result.json'), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze_inputs','run','summarize'))
    globals()[parser.parse_args().action]()
