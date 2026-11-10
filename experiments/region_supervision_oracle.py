"""Optimistic regional-label controls, never deployable PE results."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.round65_common import ROOT, read, write, sha
from experiments.region_pose_utility import arrays, configure_cuda
from experiments.region_pose_solver import estimate, score_transform

OUT = ROOT / 'results/region_supervision_oracle/diagnostic1'
PARENT = ROOT / 'results/region_rotary_relation/attempt1'
CONDITIONS = ROOT / 'results/region_image_displacement/diagnostic1'
PLAN = ROOT / 'docs/region_supervision_oracle_plan.md'
ROLES = ('overlap_nearest_tie', 'overlap_first_tie')


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def freeze():
    assert not OUT.exists()
    parent = read(PARENT / 'result.json')
    previous = read(PARENT / 'pose_analysis_lock.json')
    truth = ROOT / 'results/region_pose_utility/attempt1/scoring_truth.npz'
    sources = [Path(__file__), PLAN, ROOT / 'experiments/region_pose_solver.py',
        ROOT / 'experiments/region_pose_utility.py', ROOT / 'experiments/round65_common.py',
        PARENT / 'result.json', PARENT / 'review.json', PARENT / 'independent_checks.json',
        CONDITIONS / 'evaluation_attributes.npz', CONDITIONS / 'calibration.json']
    assert read(PARENT / 'review.json')['scientific_review_complete']
    write(OUT / 'lock.json', dict(created_unix=time.time(),
        source_sha256={str(p): sha(p) for p in sources}, cache=previous['cache'],
        truth=dict(path=str(truth), sha256=sha(truth)),
        reused_records=parent['records'], roles=list(ROLES),
        groups_path=str(CONDITIONS / 'evaluation_attributes.npz'),
        consensus_radius_pair_units=.1, maximum_consensus_refits=5,
        bootstrap_seeds=[99101, 99102, 99103], bootstrap_replicates=10000,
        comparisons=[['overlap_nearest_tie', 'gt_nearest'],
                     ['overlap_first_tie', 'gt_nearest'],
                     ['overlap_nearest_tie', 'overlap_first_tie']],
        new_fits=0, new_model_inference=0, new_pose_groups=2,
        ground_truth_used_to_construct_diagnostic_correspondences=True,
        deployable=False, fresh_confirmation=False, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('SUPERVISION_ORACLE_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    lock = read(OUT / 'lock.json')
    for path, digest in lock['source_sha256'].items():
        assert sha(path) == digest, path
    for entry in (lock['cache'], lock['truth']):
        assert sha(entry['path']) == entry['sha256']
    return lock


def load_inputs(lock):
    data = arrays(lock['cache']['path'], ('ids', 'scenes', 'centers', 'world_centers', 'scales', 'overlap'))
    truth = arrays(lock['truth']['path'], ('ids', 'rotation', 'translation', 'scale'))
    np.testing.assert_array_equal(data['ids'], truth['ids'])
    np.testing.assert_array_equal(data['scales'], truth['scale'])
    data['centers'] = data['centers'].astype(np.float64)
    return data, truth


def prepare():
    lock = check()
    assert not (OUT / 'correspondences.npz').exists()
    data, truth = load_inputs(lock)
    mapped = np.einsum('bij,bnj->bni', truth['rotation'], data['centers'][:, 0]) + truth['translation'][:, None]
    squared = np.square(mapped[:, :, None] - data['centers'][:, None, 1]).sum(-1)
    gt = squared.argmin(-1)
    overlap = data['overlap'].astype(np.float64)
    maximum = overlap.max(-1)
    valid = maximum >= .3
    tied = overlap == maximum[..., None]
    first = np.where(valid, overlap.argmax(-1), gt)
    nearest = np.where(valid, np.where(tied, squared, np.inf).argmin(-1), gt)
    for indices in (first, nearest):
        np.testing.assert_array_equal(indices[~valid], gt[~valid])
        picked = np.take_along_axis(overlap, indices[..., None], -1)[..., 0]
        np.testing.assert_array_equal(picked[valid], maximum[valid])
    # Independent rule enumeration, including both supervised and unconstrained rows.
    for b in range(min(16, len(gt))):
        for q in range(64):
            candidates = [k for k in range(64) if overlap[b, q, k] == max(overlap[b, q])]
            if max(overlap[b, q]) >= .3:
                assert first[b, q] == min(candidates)
                assert nearest[b, q] == min(candidates, key=lambda k: (squared[b, q, k], k))
            else:
                assert first[b, q] == nearest[b, q] == gt[b, q]
    np.savez(OUT / 'correspondences.npz', ids=data['ids'], scenes=data['scenes'],
        gt_nearest=gt, overlap_first_tie=first, overlap_nearest_tie=nearest,
        valid_region_queries=valid, maximum_overlap_tie_count=tied.sum(-1))
    write(OUT / 'preparation.json', dict(completed_unix=time.time(),
        lock_sha256=sha(OUT / 'lock.json'), correspondences_sha256=sha(OUT / 'correspondences.npz'),
        pairs=len(gt), source_queries=int(gt.size), valid_region_queries=int(valid.sum()),
        pairs_without_valid_source_region_queries=int((~valid.any(-1)).sum()),
        valid_queries_with_multiple_maxima=int((valid & (tied.sum(-1) > 1)).sum()),
        changed_from_gt_nearest={ROLES[0]: int((nearest != gt).sum()), ROLES[1]: int((first != gt).sum())},
        tie_variant_differences=int((nearest != first).sum()),
        unconstrained_queries_identical_to_gt_nearest=True,
        independent_rule_checked_pairs=min(16, len(gt)), all_queries_kept=True,
        deployable=False, scientific_review_pending=True, goal_complete=False))
    print('SUPERVISION_ORACLE_CORRESPONDENCES_READY', int(valid.sum()), flush=True)


def run():
    lock = check()
    prep = read(OUT / 'preparation.json')
    assert prep['lock_sha256'] == sha(OUT / 'lock.json')
    assert prep['correspondences_sha256'] == sha(OUT / 'correspondences.npz')
    assert not (OUT / 'execution_status.json').exists(), 'Inspect existing worker before any recovery.'
    configure_cuda()
    data, truth = load_inputs(lock)
    indices = arrays(OUT / 'correspondences.npz', ('ids', 'gt_nearest', *ROLES))
    np.testing.assert_array_equal(indices['ids'], data['ids'])
    gpu_centers = torch.as_tensor(data['centers'], device='cuda')
    gt_receipt = next(x['receipt'] for x in lock['reused_records'] if x['role'] == 'gt_nearest')
    assert sha(gt_receipt['path']) == gt_receipt['sha256']
    baseline = arrays(gt_receipt['path'], ('rotation', 'translation', 'valid', 'consensus_count', 'success'))
    state = dict(pid=os.getpid(), state='preflight', started_unix=time.time(),
        lock_sha256=sha(OUT / 'lock.json'), CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        active=None, completed=[], goal_complete=False)

    def save():
        state['updated_unix'] = time.time()
        write(OUT / 'execution_status.json', state)

    save()
    try:
        maximum = 0.
        for b in range(2):
            result = estimate(gpu_centers[b, 0], gpu_centers[b, 1],
                torch.as_tensor(indices['gt_nearest'][b], device='cuda', dtype=torch.long),
                radius=lock['consensus_radius_pair_units'], max_refits=lock['maximum_consensus_refits'])
            assert result['valid'] == baseline['valid'][b]
            assert result['consensus_count'] == baseline['consensus_count'][b]
            for key in ('rotation', 'translation'):
                maximum = max(maximum, float(np.abs(result[key] - baseline[key][b]).max()))
            score = score_transform(data['centers'][b, 0], result['rotation'], result['translation'],
                truth['rotation'][b], truth['translation'][b], data['scales'][b], valid=result['valid'])
            assert score['success'] == baseline['success'][b]
        assert maximum < 1e-8
        write(OUT / 'gpu_replay.json', dict(passed=True, pairs=2, maximum_transform_error=maximum,
            device=torch.cuda.get_device_name(0), CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
            solver_sha256=sha(ROOT / 'experiments/region_pose_solver.py'), completed_unix=time.time()))
        state['state'] = 'estimating'
        for role in ROLES:
            start = time.time()
            state['active'] = dict(role=role, completed_pairs=0)
            save()
            records = []
            for b in range(len(data['ids'])):
                result = estimate(gpu_centers[b, 0], gpu_centers[b, 1],
                    torch.as_tensor(indices[role][b], device='cuda', dtype=torch.long),
                    radius=lock['consensus_radius_pair_units'], max_refits=lock['maximum_consensus_refits'])
                score = score_transform(data['centers'][b, 0], result['rotation'], result['translation'],
                    truth['rotation'][b], truth['translation'][b], data['scales'][b], valid=result['valid'])
                records.append(dict(**result, **score))
                if (b + 1) % 32 == 0:
                    state['active']['completed_pairs'] = b + 1
                    save()
            payload = {k: np.stack([r[k] for r in records]) for k in records[0] if k != 'truncated_squared_error'}
            payload['truncated_squared_error'] = np.asarray([
                np.nan if r['truncated_squared_error'] is None else r['truncated_squared_error'] for r in records])
            path = OUT / (role + '.npz')
            assert not path.exists()
            np.savez(path, ids=data['ids'], scenes=data['scenes'], **payload)
            state['completed'].append(dict(role=role, path=str(path), sha256=sha(path),
                pairs=len(records), seconds=time.time()-start, deployable=False))
            save()
            print('SUPERVISION_ORACLE_ESTIMATED', role, time.time()-start, flush=True)
        state.update(state='estimates_complete', active=None, finished_unix=time.time(),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as error:
        state.update(state='failed', error_type=type(error).__name__, reason=str(error))
        save()
        raise


def summarize():
    lock = check()
    state = read(OUT / 'execution_status.json')
    assert state['state'] == 'estimates_complete' and len(state['completed']) == 2
    assert not (OUT / 'result.json').exists()
    data, _ = load_inputs(lock)
    grouping = arrays(lock['groups_path'], ('ids', 'group'))
    np.testing.assert_array_equal(grouping['ids'], data['ids'])
    correspondence = arrays(OUT / 'correspondences.npz', ('gt_nearest', *ROLES))
    scenes = sorted(set(data['scenes'].tolist()))
    gt = []
    for b in range(len(data['ids'])):
        transforms = []
        for view in range(2):
            a = data['centers'][b, view]
            w = data['world_centers'][b, view].astype(np.float64)
            rotation, _ = Rotation.align_vectors(w-w.mean(0), a-a.mean(0))
            matrix = np.eye(4)
            matrix[:3, :3] = rotation.as_matrix()
            matrix[:3, 3] = w.mean(0)-rotation.apply(a.mean(0))
            transforms.append(matrix)
        gt.append(np.linalg.solve(transforms[1], transforms[0]))
    gt = np.stack(gt)
    references = [x['receipt'] for x in lock['reused_records'] if x['role'] in ('gt_nearest', 'identity')]
    rows, accuracy = [], {}
    errors = dict(rmse_m=0., rotation_error_deg=0., centered_translation_error_m=0.)
    consensus_differences = 0
    for receipt in [*state['completed'], *references]:
        assert sha(receipt['path']) == receipt['sha256']
        row = arrays(receipt['path'], ('ids', 'scenes', 'rotation', 'translation', 'valid',
            'success', 'rmse_m', 'rotation_error_deg', 'centered_translation_error_m', 'consensus_count'))
        np.testing.assert_array_equal(row['ids'], data['ids'])
        np.testing.assert_array_equal(row['scenes'], data['scenes'])
        for b in range(len(data['ids'])):
            if not row['valid'][b]:
                assert not row['success'][b] and np.isinf(row['rmse_m'][b])
                continue
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = row['rotation'][b], row['translation'][b]
            np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8)
            assert abs(np.linalg.det(matrix[:3, :3]) - 1) < 1e-8
            points = np.column_stack((data['centers'][b, 0], np.ones(64)))
            residual = points @ (matrix - gt[b]).T
            rmse = float(np.linalg.norm(residual[:, :3], 'fro') / 8 * data['scales'][b])
            angle = float(np.degrees(Rotation.from_matrix(matrix[:3, :3] @ gt[b, :3, :3].T).magnitude()))
            translation = float(np.linalg.norm((matrix-gt[b])[:3, 3]) * data['scales'][b])
            for key, value in [('rmse_m', rmse), ('rotation_error_deg', angle), ('centered_translation_error_m', translation)]:
                errors[key] = max(errors[key], abs(value-row[key][b]))
            assert bool(rmse < .2) == row['success'][b]
            if receipt['role'] != 'identity':
                matched = data['centers'][b, 1, correspondence[receipt['role']][b]]
                count = int((np.linalg.norm((points @ matrix.T)[:, :3]-matched, axis=1) <= .1).sum())
                consensus_differences += int(count != row['consensus_count'][b])
        by_scene = {s: float(row['success'][data['scenes'] == s].mean()) for s in scenes}
        accuracy[receipt['role']] = np.asarray([by_scene[s] for s in scenes])
        groups = {}
        for group in ('low', 'high', 'undefined'):
            mask = grouping['group'] == group
            available = [s for s in scenes if np.any(mask & (data['scenes'] == s))]
            groups[group] = dict(pairs=int(mask.sum()),
                scene_macro_success=float(np.mean([row['success'][mask & (data['scenes'] == s)].mean() for s in available])),
                successful_pairs=int(row['success'][mask].sum()))
        rows.append(dict(role=receipt['role'], macro_success=float(np.mean(list(by_scene.values()))),
            successful_pairs=int(row['success'].sum()), invalid_estimates=int((~row['valid']).sum()),
            by_scene=by_scene, groups=groups,
            error_quantiles={k: dict(median=finite_or_none(np.median(row[k])),
                p90=finite_or_none(np.quantile(row[k], .9, method='higher'))) for k in errors},
            null_error_quantile_means_infinite=True,
            receipt=receipt, deployable=receipt['role'] == 'identity'))
    assert errors['rmse_m'] < 1e-6 and errors['centered_translation_error_m'] < 1e-6
    assert errors['rotation_error_deg'] < 1e-5 and consensus_differences == 0
    comparisons = []
    bootstrap_error = 0.
    for (left, right), seed in zip(lock['comparisons'], lock['bootstrap_seeds']):
        delta = 100 * (accuracy[left] - accuracy[right])
        index = np.random.default_rng(seed).integers(len(scenes), size=(lock['bootstrap_replicates'], len(scenes)))
        draws = delta[index].mean(-1)
        alternative = 100 * (accuracy[left][index].mean(-1) - accuracy[right][index].mean(-1))
        bootstrap_error = max(bootstrap_error, float(np.max(np.abs(draws-alternative))))
        comparisons.append(dict(left=left, right=right, mean_gain_pp=float(delta.mean()),
            by_scene_gain_pp=delta.tolist(), scene_interval_95_pp=np.quantile(draws, [.025, .975]).tolist(),
            bootstrap_seed=seed, bootstrap_replicates=lock['bootstrap_replicates']))
    assert bootstrap_error < 1e-10
    primary = comparisons[0]
    oracle_success = float(accuracy['overlap_nearest_tie'].mean())
    readout_signal = (primary['mean_gain_pp'] <= -10 and sum(v < 0 for v in primary['by_scene_gain_pp']) >= 4
                      and primary['scene_interval_95_pp'][1] < 0)
    correspondence_signal = oracle_success >= .75 and primary['mean_gain_pp'] > -10
    direction = 'prioritize_shared_readout_control' if readout_signal else (
        'prioritize_learned_correspondence' if correspondence_signal else 'mixed_diagnostic')
    write(OUT / 'result.json', dict(completed_unix=time.time(), lock_sha256=sha(OUT / 'lock.json'),
        preparation_sha256=sha(OUT / 'preparation.json'), records=rows, comparisons=comparisons,
        scenes=scenes, proposed_next_direction=direction, scientific_review_pending=True,
        new_fits=0, new_inference=0, new_pose_groups=2, fresh_confirmation=False, goal_complete=False))
    write(OUT / 'independent_checks.json', dict(passed=True, completed_unix=time.time(),
        result_sha256=sha(OUT / 'result.json'), score_maximum_errors=errors,
        success_decisions_changed=0, consensus_count_disagreements=consensus_differences,
        bootstrap_maximum_error_pp=bootstrap_error, pose_groups=4, pairs_per_group=480,
        independent_truth='SciPy alignment and homogeneous transform composition',
        scientific_review_pending=True, goal_complete=False))
    print('SUPERVISION_ORACLE_SUMMARY_READY', direction, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'prepare', 'run', 'summarize'))
    globals()[parser.parse_args().action]()
