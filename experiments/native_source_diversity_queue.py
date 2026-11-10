"""Live GPU scheduling for the nine sealed fits; never touch other jobs."""
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.native_source_diversity import OUT, ROOT, ROLES, SEEDS, check, select_all
from experiments.round65_common import read, write, sha


def resources():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    cards = []
    for line in raw.strip().splitlines():
        index, uuid, used, free, utilization = [v.strip() for v in line.split(',')]
        cards.append(dict(index=int(index), uuid=uuid, used_mib=int(used), free_mib=int(free),
                          utilization=int(utilization)))
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    apps = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        pid, uuid, memory = [v.strip() for v in line.split(',')]
        apps.append(dict(pid=int(pid), uuid=uuid))
    return cards, apps


def run():
    lock = check()
    target = OUT / 'queue_status.json'
    assert not target.exists(), 'Inspect any previous queue and child processes before recovery.'
    state = dict(pid=os.getpid(), started_unix=time.time(), state='running', active=[], completed=[],
        pending=[[r, s] for s in SEEDS for r in ROLES], failures=[],
        lock_sha256=sha(OUT / 'lock.json'), queue_source_sha256=sha(__file__),
        all_models_selected=False, scientific_review_complete=False, goal_complete=False)
    active = []
    def save(**updates):
        state.update(updates, updated_unix=time.time())
        state['active'] = [item['record'] for item in active]
        write(target, state)
    save()
    try:
        while state['pending'] or active:
            for item in active[:]:
                code = item['process'].poll()
                if code is None:
                    continue
                item['log'].close()
                record = dict(item['record'], exit_code=code, finished_unix=time.time())
                active.remove(item)
                job = [record['role'], record['seed']]
                write(OUT / 'queue_events' / (str(time.time_ns()) + '.json'), record)
                if code == 0:
                    assert (OUT / 'runs' / job[0] / str(job[1]) / 'run.json').exists()
                    state['completed'].append(job)
                    print('FIT_TERMINAL_COMPLETE', *job, flush=True)
                elif code == 75:
                    snapshot = OUT / 'runs' / job[0] / str(job[1]) / 'resume.pt'
                    assert snapshot.exists()
                    state['pending'].insert(0, job)
                    print('OWN_FIT_SHARD_ENDED', *job, flush=True)
                else:
                    state['failures'].append(record)
                    print('FIT_FAILED_NO_NEW_STARTS', *job, code, flush=True)
                save()
            if state['failures']:
                if not active:
                    save(state='failed', finished_unix=time.time())
                    return 1
                save(state='draining_after_failure')
                time.sleep(15)
                continue
            if state['pending'] and len(active) < lock['maximum_gpu_jobs']:
                cards, apps = resources()
                occupied = {item['record']['gpu']['index'] for item in active}
                own_pids = {item['process'].pid for item in active}
                shared_count = sum(any(a['uuid'] == item['record']['gpu']['uuid'] and a['pid'] not in own_pids
                                       for a in apps) for item in active)
                cards.sort(key=lambda c: (c['used_mib'] < 512 and c['utilization'] < 10, c['free_mib']), reverse=True)
                for card in cards:
                    if not state['pending'] or len(active) >= lock['maximum_gpu_jobs']:
                        break
                    if card['index'] in occupied or card['free_mib'] < lock['shared_minimum_free_mib']:
                        continue
                    shared = any(a['uuid'] == card['uuid'] and a['pid'] not in own_pids for a in apps)
                    shared = shared or card['used_mib'] >= 512 or card['utilization'] >= 10
                    if shared and shared_count >= lock['maximum_shared_gpu_jobs']:
                        continue
                    role, seed = state['pending'].pop(0)
                    directory = OUT / 'runs' / role / str(seed)
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / f'launch_{time.time_ns()}.log'
                    log = path.open('x')
                    environment = os.environ.copy()
                    environment.update(CUDA_VISIBLE_DEVICES=str(card['index']), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                        PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                    command = [sys.executable, '-u', '-m', 'experiments.native_source_diversity', 'train',
                               '--role', role, '--seed', str(seed)]
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    fields = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()
                    record = dict(role=role, seed=seed, pid=child.pid, start_ticks=int(fields[19]),
                        started_unix=time.time(), gpu=card, shared_at_launch=shared, log=str(path),
                        allocator_cap_bytes=lock['allocator_cap_bytes'], command=command)
                    active.append(dict(process=child, log=log, record=record))
                    occupied.add(card['index']); own_pids.add(child.pid)
                    shared_count += int(shared)
                    save()
                    print('FIT_STARTED', role, seed, 'gpu', card['index'], 'pid', child.pid, flush=True)
            save()
            if state['pending'] or active:
                time.sleep(15)
        select_all()
        save(state='ready_for_evaluation', all_models_selected=True, finished_unix=time.time())
        print('NINE_FITS_AND_SELECTION_COMPLETE_EVALUATION_PENDING', flush=True)
        return 0
    except BaseException as error:
        # Running owned fits are left intact and explicitly listed for recovery.
        save(state='controller_failed', error_type=type(error).__name__, reason=str(error),
             finished_unix=time.time())
        raise


if __name__ == '__main__':
    raise SystemExit(run())
