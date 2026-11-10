"""Finish only unsaved scoring groups after the GPU telemetry interruption."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.region_hrope_confirmation_score import (
    OUT, DEV, SEEDS, check, read, write, sha, configure_cuda, arrays,
    recover_truth, estimate, score_transform,
)

RECOVERY = OUT / 'gpu_query_timeout_recovery1'
GPU = 'GPU-0a17e6c1-0ac8-cbed-cea7-d34fbb759d92'


def run_missing(seed):
    lock = check()
    bound = read(OUT / 'pose_inputs.json')
    assert bound['selected_sha256'] == sha(DEV / 'selected.json')
    folder = OUT / 'pose' / str(seed)
    previous = read(RECOVERY / f'original_pose_{seed}_status.json')
    assert not Path('/proc', str(previous['pid'])).exists()
    assert read(folder / 'status.json') == previous
    assert previous['pose_inputs_sha256'] == sha(OUT / 'pose_inputs.json')
    completed = previous['completed']
    assert {r['role'] for r in completed} == {
        'relative5', 'distance_mlp', 'geotransformer_rpe', 'fixed_isotropic5'}
    jobs = [p for p in bound['predictions'] if p['seed'] == seed]
    for receipt in completed:
        assert sha(receipt['path']) == receipt['sha256']
        assert receipt['input_prediction'] == next(p for p in jobs if p['role'] == receipt['role'])
    jobs = [p for p in jobs if p['role'] not in {r['role'] for r in completed}]
    assert len(jobs) == 1 and jobs[0]['role'] == 'predicted_hrope'
    assert not (folder / 'predicted_hrope.npz').exists()
    configure_cuda()
    assert sha(lock['cache']['path']) == lock['cache']['sha256']
    data = arrays(lock['cache']['path'], ('ids', 'scenes', 'centers', 'world_centers', 'scales'))
    pair_count = lock['cache']['pairs']
    assert len(data['ids']) == pair_count and len(np.unique(data['scenes'])) >= 2
    centers = data['centers'].astype(np.float64)
    truth = [recover_truth(centers[i], data['world_centers'][i]) for i in range(pair_count)]
    assert max(r[2] for r in truth) < 2e-5
    gpu_centers = torch.as_tensor(centers, device='cuda')
    state = dict(previous, state='estimating', pid=os.getpid(), resumed_unix=time.time(),
                 active=None, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
                 recovery=str(RECOVERY), recovery_source_sha256=sha(__file__))

    def save():
        state['updated_unix'] = time.time()
        write(folder / 'status.json', state)

    save()
    try:
        # Same prediction selection, estimator, scores, payload and receipt as run().
        for job in jobs:
            assert sha(job['path']) == job['sha256']
            prediction = arrays(job['path'], ('ids', 'logits'))
            np.testing.assert_array_equal(prediction['ids'], data['ids'])
            assert prediction['logits'].shape == (pair_count, 64, 64) and np.isfinite(prediction['logits']).all()
            indices = prediction['logits'].argmax(-1)
            records, started = [], time.time()
            state['active'] = dict(role=job['role'], completed_pairs=0)
            save()
            for index in range(pair_count):
                result = estimate(gpu_centers[index, 0], gpu_centers[index, 1],
                    torch.as_tensor(indices[index], device='cuda', dtype=torch.long),
                    radius=lock['consensus_radius_pair_units'], max_refits=lock['maximum_consensus_refits'])
                score = score_transform(centers[index, 0], result['rotation'], result['translation'],
                    truth[index][0], truth[index][1], data['scales'][index], valid=result['valid'])
                records.append(dict(**result, **score))
                if (index+1) % 32 == 0:
                    state['active']['completed_pairs'] = index+1
                    save()
            path = folder/(job['role']+'.npz')
            assert not path.exists()
            payload = {k: np.stack([r[k] for r in records]) for k in records[0] if k != 'truncated_squared_error'}
            payload['truncated_squared_error'] = np.asarray([
                np.nan if r['truncated_squared_error'] is None else r['truncated_squared_error'] for r in records])
            np.savez(path, ids=data['ids'], scenes=data['scenes'], **payload)
            receipt = dict(role=job['role'], seed=seed, path=str(path), sha256=sha(path),
                pairs=pair_count, seconds=time.time()-started, input_prediction=job,
                interpretation='camera-information-matched learned regional PE')
            state['completed'].append(receipt)
            save()
            print('HROPE_POSE_COMPLETE', job['role'], seed, receipt['seconds'], flush=True)
        state.update(state='estimates_complete', active=None, finished_unix=time.time(),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as error:
        state.update(state='failed', error_type=type(error).__name__, reason=str(error))
        save()
        raise


def device_snapshot():
    # Telemetry is needed only before launching. It never kills running workers.
    memory = subprocess.run(['nvidia-smi', '-i', GPU,
        '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=30)
    apps = subprocess.run(['nvidia-smi', '-i', GPU,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=30)
    fields = [v.strip() for v in memory.stdout.strip().split(',')]
    return dict(observed_unix=time.time(), uuid=fields[0], memory_mib=int(fields[1]),
                free_mib=int(fields[2]), utilization=int(fields[3]),
                pids=[int(v.strip()) for v in apps.stdout.splitlines() if v.strip()])


def finish():
    check()
    assert not (RECOVERY / 'queue_status.json').exists()
    repair = read(RECOVERY / 'repair.json')
    assert repair['pose_inputs_sha256'] == sha(OUT / 'pose_inputs.json')
    observations = []
    for _ in range(2):
        row = device_snapshot()
        assert not row['pids'] and row['memory_mib'] < 512 and row['utilization'] <= 10, row
        observations.append(row)
        time.sleep(3)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    probe = subprocess.run([sys.executable, '-c',
        'import numpy;import torch;assert torch.cuda.is_available();assert torch.cuda.device_count()==1;print((torch.ones(1,device="cuda")+1).item())'],
        env=env, capture_output=True, text=True, check=True, timeout=45)
    final = device_snapshot()
    assert not final['pids'] and final['free_mib'] >= 12288, final
    observations.append(final)
    state = dict(pid=os.getpid(), started_unix=time.time(), state='running', stage='pose',
                 active=[], completed=[], source_sha256=sha(__file__),
                 gpu_observations=observations, cuda_probe=probe.stdout.strip(),
                 evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'), goal_complete=False)
    active = []

    def save():
        state['observed_unix'] = time.time()
        state['active'] = [item for _, item, _ in active]
        write(RECOVERY / 'queue_status.json', state)

    for seed in SEEDS:
        log = RECOVERY / f'pose_{seed}.log'
        stream = log.open('x')
        process = subprocess.Popen([sys.executable, '-u', '-m',
            'experiments.region_hrope_confirmation_finish', 'worker', '--seed', str(seed)],
            env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        item = dict(seed=seed, pid=process.pid, log=str(log), gpu_uuid=GPU)
        active.append((process, item, stream))
        save()
        print('RECOVERY_LAUNCHED', item, flush=True)
    # No allocation or launch remains, so GPU telemetry is unnecessary here.
    while active:
        for process, item, stream in list(active):
            code = process.poll()
            if code is not None:
                stream.close()
                active.remove((process, item, stream))
                state['completed'].append(dict(item, exit_code=code, finished_unix=time.time()))
                print('RECOVERY_CHILD_EXIT', item['seed'], code, flush=True)
        save()
        if active:
            time.sleep(3)
    if any(r['exit_code'] for r in state['completed']):
        state.update(state='failed', reason='Worker failure; all artifacts preserved.')
        save()
        raise RuntimeError(state['reason'])
    for stage, command in [
        ('independent_summary', ['experiments.region_hrope_confirmation_score', 'summarize']),
        ('tables_and_figure', ['experiments.region_hrope_confirmation_report']),
    ]:
        state['stage'] = stage
        save()
        with (RECOVERY / f'{stage}.log').open('x') as stream:
            result = subprocess.run([sys.executable, '-u', '-m', *command],
                env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=stream, stderr=subprocess.STDOUT)
        state['completed'].append(dict(stage=stage, exit_code=result.returncode))
        if result.returncode:
            state.update(state='failed', reason=stage+' failed')
            save()
            raise RuntimeError(state['reason'])
    state.update(state='complete', stage='awaiting_scientific_review', scientific_review_pending=True)
    save()
    print('RECOVERY_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('worker', 'finish'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == 'worker':
        assert args.seed in SEEDS
        run_missing(args.seed)
    else:
        finish()
