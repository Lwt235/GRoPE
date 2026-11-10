"""Same fixed-H work with the user-authorized limited spare-GPU fallback."""
import os
import subprocess
import sys
import time

from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.round65_queue import occupancy
from experiments.region_hrope_pose_queue import app_pids
from experiments.region_fixed_hrope import OUT, check

TRAIN = 'experiments.region_fixed_hrope'
POSE = 'experiments.region_fixed_hrope_pose'
RECOVERY = OUT / 'execution_spare_adjustment1'

def memory_free():
    raw = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=30)
    return {a.strip():int(b.strip()) for a,b in (line.split(',') for line in raw.stdout.splitlines())}


def main():
    check()
    execution = read(RECOVERY / 'execution_lock.json')
    assert execution['source_sha256'] == sha(__file__)
    assert execution['study_lock_sha256'] == sha(OUT / 'lock.json')
    assert not (RECOVERY / 'queue_status.json').exists()
    active, completed, pending = [], [], []
    stage = 'starting'
    identity = dict(pid=os.getpid(), started_unix=time.time(), lock_sha256=sha(OUT / 'lock.json'))

    def record(state, **extra):
        write(RECOVERY / 'queue_status.json', dict(identity, state=state, stage=stage,
            active=[r[1] for r in active], completed=completed, pending=pending,
            observed_unix=time.time(), goal_complete=False, **extra))

    def batch(name, module, action, jobs, gpu=True):
        nonlocal stage, pending
        stage, pending = name, list(jobs)
        previous = set()
        previous_spare = set()
        while pending or active:
            for process, item, stream in list(active):
                code = process.poll()
                if code is not None:
                    stream.close(); active.remove((process, item, stream))
                    completed.append(dict(item, exit_code=code, finished_unix=time.time()))
                    print('FIXED_H_CHILD_EXIT', name, item.get('seed'), code, flush=True)
            if any(r['exit_code'] for r in completed):
                record('finishing_active_after_failure')
                if not active:
                    raise RuntimeError('Worker failed; artifacts retained and pending work not launched.')
                time.sleep(3); continue
            if not pending and not active:
                break
            # Query only when another worker can actually be launched.
            if not pending or len(active) >= (3 if gpu else 1):
                record('running'); time.sleep(3); continue
            owned = {}
            for _, item, _ in active:
                if item['gpu']:
                    owned.setdefault(item['gpu']['uuid'], set()).add(item['pid'])
            devices = [None]
            if gpu:
                try:
                    rows, apps, space = occupancy(), app_pids(), memory_free()
                    free = {r['uuid'] for r in rows if r['free']}
                    devices = [r for r in rows if r['uuid'] in free & previous]
                    devices += [r for r in rows if r['uuid'] in owned and
                        apps.get(r['uuid'], set()) <= owned[r['uuid']] and r['memory_mib'] < 8192]
                    spare = {r['uuid'] for r in rows if space[r['uuid']] >= 12288 and r['utilization'] <= 60}
                    external_already_shared = any(apps.get(uuid, set()) - pids for uuid, pids in owned.items())
                    if not devices and not external_already_shared:
                        candidates = [dict(r, shared_existing_compute=True, memory_free_mib=space[r['uuid']])
                            for r in rows if r['uuid'] in spare & previous_spare and r['uuid'] not in owned]
                        devices = sorted(candidates, key=lambda r: (r['utilization'], -r['memory_free_mib']))[:1]
                    previous = free
                    previous_spare = spare
                except (subprocess.SubprocessError, OSError) as error:
                    record('telemetry_retry', reason=str(error))
                    time.sleep(3); continue
            if devices:
                device = devices[0]
                env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                    CUDA_VISIBLE_DEVICES=device['uuid'] if device else '', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
                probe = None
                if device:
                    try:
                        test = subprocess.run([sys.executable, '-c',
                            'import numpy;import torch;assert torch.cuda.is_available();assert torch.cuda.device_count()==1;print((torch.ones(1,device="cuda")+1).item())'],
                            env=env, capture_output=True, text=True, timeout=45)
                        if test.returncode:
                            record('cuda_probe_retry', error=test.stderr[-1200:]); time.sleep(3); continue
                        if device.get('shared_existing_compute'):
                            current = next(r for r in occupancy() if r['uuid'] == device['uuid'])
                            ready = memory_free()[device['uuid']] >= 12288 and current['utilization'] <= 60
                        else:
                            ready = not (app_pids().get(device['uuid'], set()) - owned.get(device['uuid'], set()))
                        if not ready:
                            record('gpu_changed_before_launch'); time.sleep(3); continue
                        probe = dict(exit_code=0, stdout=test.stdout.strip())
                    except (subprocess.SubprocessError, OSError) as error:
                        record('telemetry_retry', reason=str(error)); time.sleep(3); continue
                job = pending.pop(0)
                command = [sys.executable, '-u', '-m', module]
                if action:
                    command.append(action)
                if 'seed' in job:
                    command += ['--seed', str(job['seed'])]
                log = RECOVERY / f'{name}_{job.get("seed")}_{time.time_ns()}.log'
                stream = log.open('x')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(job, pid=process.pid, stage=name, gpu=device, cuda_probe=probe,
                            log=str(log), started_unix=time.time())
                active.append((process, item, stream))
                print('FIXED_H_LAUNCHED', item, flush=True)
            record('running' if active else 'waiting_for_idle_gpu')
            time.sleep(3)

    try:
        batch('preflight', TRAIN, 'preflight', [{}])
        batch('train', TRAIN, 'train', [dict(seed=s) for s in SEEDS])
        batch('selections', TRAIN, 'selections', [{}], gpu=False)
        batch('export', TRAIN, 'export', [dict(seed=s) for s in SEEDS])
        batch('bind_inputs', POSE, 'bind_inputs', [{}], gpu=False)
        batch('pose', POSE, 'run', [dict(seed=s) for s in SEEDS])
        batch('independent_summary', POSE, 'summarize', [{}], gpu=False)
        batch('tables_and_figures', 'experiments.region_fixed_hrope_report', None, [{}], gpu=False)
        stage = 'awaiting_scientific_review'
        record('complete', scientific_review_pending=True)
    except BaseException as error:
        # Record live child identities; a controller exception is not child failure.
        record('controller_failed', reason=str(error), error_type=type(error).__name__,
               children_left_running=bool(active))
        raise


if __name__ == '__main__':
    main()
