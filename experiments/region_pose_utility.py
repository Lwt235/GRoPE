"""Frozen, zero-training regional pose-utility experiment on old five scenes."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from experiments.region_pose_solver import estimate, recover_truth, score_transform
from experiments.round65_common import ROOT, read, write, sha

BASE = ROOT / 'results/region_pose_utility'
PREP = BASE / 'preparation'
OUT = BASE / 'attempt1'


def arrays(path, fields):
    with zipfile.ZipFile(path) as archive:
        return {key: np.load(io.BytesIO(archive.read(key + '.npy')), allow_pickle=False) for key in fields}


def freeze():
    plan = read(PREP / 'plan.json')
    assert sha(ROOT / 'docs/region_pose_utility_plan.md') == plan['plan_sha256']
    assert '3 passed' in (PREP / 'cpu_checks.txt').read_text()
    for entry in [plan['cache'], *plan['predictions']]:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    sources = [Path(__file__).resolve(), ROOT / 'experiments/region_pose_solver.py',
               ROOT / 'tests/test_region_pose_solver.py', ROOT / 'experiments/round65_common.py']
    OUT.mkdir(exist_ok=False)
    write(OUT / 'lock.json', dict(plan=plan, plan_sha256=sha(PREP / 'plan.json'),
        cpu_checks_sha256=sha(PREP / 'cpu_checks.txt'), sources={str(p): sha(p) for p in sources},
        created_unix=time.time(), fresh_confirmation=False, new_fits=0, new_model_inference=0))
    print('POSE_UTILITY_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    lock = read(OUT / 'lock.json')
    assert sha(PREP / 'plan.json') == lock['plan_sha256']
    assert sha(ROOT / 'docs/region_pose_utility_plan.md') == lock['plan']['plan_sha256']
    for path, digest in lock['sources'].items():
        assert sha(path) == digest, path
    return lock


def configure_cuda():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(2 * 2**30 / torch.cuda.get_device_properties(0).total_memory)


def preflight():
    check()
    assert not (OUT / 'cuda_preflight.json').exists()
    configure_cuda()
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(93000)
    source = rng.normal(size=(64, 3))
    rotation = Rotation.from_rotvec([.3, -.4, .1]).as_matrix()
    translation = np.array([.2, .1, -.3])
    target = source @ rotation.T + translation + rng.normal(0, .002, source.shape)
    target[40:] = rng.normal(size=(24, 3)) * 4
    rows, timings = {}, {}
    for device in ('cpu', 'cuda'):
        started = time.time()
        rows[device] = estimate(torch.as_tensor(source, device=device), torch.as_tensor(target, device=device),
                                torch.arange(64, device=device))
        if device == 'cuda':
            torch.cuda.synchronize()
        timings[device] = time.time() - started
    assert rows['cpu']['valid'] and rows['cuda']['valid']
    assert rows['cpu']['consensus_count'] == rows['cuda']['consensus_count'] == 40
    r_error = float(np.abs(rows['cpu']['rotation'] - rows['cuda']['rotation']).max())
    t_error = float(np.abs(rows['cpu']['translation'] - rows['cuda']['translation']).max())
    assert max(r_error, t_error) < 1e-8
    assert rows['cuda']['hypotheses_total'] == 41664
    write(OUT / 'cuda_preflight.json', dict(passed=True, completed_unix=time.time(),
        source='Generated geometry only; no real pose scores read.',
        lock_sha256=sha(OUT / 'lock.json'), device=torch.cuda.get_device_name(0),
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        maximum_rotation_difference=r_error, maximum_translation_difference=t_error,
        consensus_count=40, hypotheses=41664, seconds=timings,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    print('POSE_PREFLIGHT_PASSED', timings, r_error, t_error, flush=True)


def run():
    lock = check()
    assert not (OUT / 'status.json').exists(), 'Inspect a previous worker before resuming.'
    preflight = read(OUT / 'cuda_preflight.json')
    assert preflight['passed'] and preflight['lock_sha256'] == sha(OUT / 'lock.json')
    configure_cuda()
    plan = lock['plan']
    assert sha(plan['cache']['path']) == plan['cache']['sha256']
    data = arrays(plan['cache']['path'], ('ids', 'scenes', 'centers', 'world_centers', 'scales'))
    assert len(data['ids']) == 480 and len(np.unique(data['scenes'])) == 5
    assert data['centers'].shape == data['world_centers'].shape == (480, 2, 64, 3)
    centers = np.asarray(data['centers'], dtype=np.float64)
    world = np.asarray(data['world_centers'], dtype=np.float64)
    truth = [recover_truth(centers[i], world[i]) for i in range(480)]
    truth_r, truth_t = np.stack([x[0] for x in truth]), np.stack([x[1] for x in truth])
    maximum_reconstruction_error = max(x[2] for x in truth)
    assert maximum_reconstruction_error < 2e-5
    np.savez(OUT / 'scoring_truth.npz', ids=data['ids'], scenes=data['scenes'], rotation=truth_r,
             translation=truth_t, scale=data['scales'])
    gpu_centers = torch.as_tensor(centers, device='cuda')
    state = dict(state='estimating', pid=os.getpid(), started_unix=time.time(),
        lock_sha256=sha(OUT / 'lock.json'), completed=[], active=None, goal_complete=False,
        maximum_scoring_coordinate_reconstruction_error=maximum_reconstruction_error,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])

    def save():
        state['updated_unix'] = time.time()
        write(OUT / 'status.json', state)

    save()
    jobs = [dict(role='identity', seed=0), dict(role='gt_nearest', seed=0), *plan['predictions']]
    try:
        for job in jobs:
            role, seed = job['role'], job['seed']
            if 'path' in job:
                assert sha(job['path']) == job['sha256']
                prediction = arrays(job['path'], ('ids', 'logits'))
                assert np.array_equal(prediction['ids'], data['ids'])
                assert prediction['logits'].shape == (480, 64, 64) and np.isfinite(prediction['logits']).all()
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
            for index in range(480):
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
                pairs=480, seconds=time.time() - started, input_prediction=job if 'path' in job else None,
                interpretation='undeployable diagnostic' if role == 'gt_nearest' else 'identity baseline' if role == 'identity' else 'fixed learned matcher')
            state['completed'].append(receipt)
            save()
            print('POSE_ESTIMATES_COMPLETE', role, seed, receipt['seconds'], flush=True)
        state.update(state='estimates_complete', active=None, finished_unix=time.time(),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
        print('POSE_UTILITY_ESTIMATES_COMPLETE', len(state['completed']), flush=True)
    except BaseException as error:
        state.update(state='failed', error_type=type(error).__name__, reason=str(error))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'preflight', 'run'))
    globals()[parser.parse_args().action]()
