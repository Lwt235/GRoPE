"""Resource controller for fixed scoring; never changes scientific settings."""
import os
import signal
import subprocess
import sys
import time

from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.round65_queue import occupancy
from experiments.region_hrope_control import OUT
from experiments.region_hrope_pose import check


def app_pids():
    raw = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    result = {}
    for line in raw.stdout.splitlines():
        if ',' in line:
            uuid, pid = [x.strip() for x in line.split(',')]
            result.setdefault(uuid, set()).add(int(pid))
    return result


def main():
    check()
    assert not (OUT/'pose_queue_status.json').exists(), 'Inspect previous controller before restarting.'
    active, completed, pending = [], [], list(SEEDS)
    stage = 'waiting_for_training_exports'
    identity = dict(pid=os.getpid(), started_unix=time.time(), source_sha256=sha(__file__),
        study_lock_sha256=sha(OUT/'lock.json'), maximum_workers=3,
        resource_policy='Prefer idle GPUs; tiny geometry workers may share a GPU occupied only by this controller. Never start on an externally occupied GPU.')

    def record(state, **extra):
        write(OUT/'pose_queue_status.json', dict(identity, stage=stage, state=state, pending=pending,
            active=[r[1] for r in active], completed=completed, observed_unix=time.time(),
            goal_complete=False, **extra))

    def cpu(module, action=None):
        command = [sys.executable, '-u', '-m', module]+([] if action is None else [action])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        result = subprocess.run(command, cwd=ROOT, env=env)
        assert result.returncode == 0, (module, action, result.returncode)

    try:
        while True:
            q = read(OUT/'queue_status.json')
            assert q['state'] != 'stopped', 'Training stopped; inspect preserved failure.'
            if q['state'] == 'exports_complete':
                break
            record('waiting_for_training_exports')
            time.sleep(5)
        stage = 'bind_inputs'
        record('running')
        cpu('experiments.region_hrope_pose', 'bind_inputs')
        stage = 'pose_estimation'
        previous = set()
        while pending or active:
            for process, item, stream in list(active):
                code = process.poll()
                if code is not None:
                    stream.close()
                    active.remove((process, item, stream))
                    completed.append(dict(item, exit_code=code, finished_unix=time.time()))
                    print('HROPE_POSE_WORKER_EXIT', item['seed'], code, flush=True)
            failures = [r for r in completed if r['exit_code']]
            if failures:
                record('finishing_active_after_failure', failures=failures)
                if not active:
                    raise RuntimeError('Scoring failure preserved; remaining jobs not launched.')
                time.sleep(3)
                continue
            if not pending and not active:
                break
            rows, apps = occupancy(), app_pids()
            free = {r['uuid'] for r in rows if r['free']}
            stable = free & previous
            owned = {}
            for _, item, _ in active:
                owned.setdefault(item['gpu']['uuid'], set()).add(item['pid'])
            available = [r for r in rows if r['uuid'] in stable]
            available += [r for r in rows if r['uuid'] in owned and
                          apps.get(r['uuid'], set()) <= owned[r['uuid']] and r['memory_mib'] < 8192]
            if pending and len(active) < 3 and available:
                gpu = available[0]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                probe = subprocess.run([sys.executable, '-c',
                    'import numpy;import torch;assert torch.cuda.is_available();assert torch.cuda.device_count()==1;print((torch.ones(1,device="cuda")+1).item())'],
                    env=env, capture_output=True, text=True, timeout=45)
                assert probe.returncode == 0, probe.stderr
                if app_pids().get(gpu['uuid'], set()) <= owned.get(gpu['uuid'], set()):
                    seed = pending.pop(0)
                    log = OUT/f'pose_seed{seed}_{time.time_ns()}.log'
                    stream = log.open('x')
                    command = [sys.executable, '-u', '-m', 'experiments.region_hrope_pose', 'run', '--seed', str(seed)]
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    item = dict(seed=seed, pid=process.pid, gpu=gpu, log=str(log), started_unix=time.time(),
                        cuda_probe=dict(exit_code=probe.returncode, stdout=probe.stdout.strip()),
                        sharing_with_owned_pids=sorted(owned.get(gpu['uuid'], set())))
                    active.append((process, item, stream))
                    print('HROPE_POSE_LAUNCHED', item, flush=True)
            previous = free
            record('running' if active else 'waiting_for_idle_gpu')
            time.sleep(3)
        stage = 'independent_scoring_and_summary'
        record('running')
        cpu('experiments.region_hrope_pose', 'summarize')
        stage = 'report_tables_and_figures'
        record('running')
        cpu('experiments.region_hrope_report')
        stage = 'awaiting_scientific_review'
        record('complete', scientific_review_pending=True)
    except BaseException as error:
        for process, item, stream in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        record('stopped', error_type=type(error).__name__, reason=str(error))
        raise


if __name__ == '__main__':
    main()
