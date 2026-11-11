"""Two explicit disjoint queues; every GPU job rechecks shared-card headroom."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gaussian_memory/soft_read1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def wait_for_fit(seed):
    path = OUT / f'soft_{seed}/status.json'; deadline = time.monotonic() + 1800
    while not path.exists():
        if time.monotonic() > deadline: raise TimeoutError(f'Existing Soft seed {seed} did not finish in 30 minutes.')
        time.sleep(30)
    assert json.loads(path.read_text())['complete']


def main():
    p = argparse.ArgumentParser(); p.add_argument('worker', choices=('primary', 'auxiliary')); p.add_argument('--gpu', required=True)
    p.add_argument('--relocate-stopped-queue', action='store_true')
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / f'queue_{a.worker}.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    train = 'experiments.gaussian_memory_soft_train'; evaluate = 'experiments.gaussian_memory_soft_evaluate'
    def evaluation(stage, kind=None, seed=None):
        return [evaluate, stage] + (['--kind', kind] if kind else []) + (['--seed', str(seed)] if seed else [])
    if a.worker == 'primary':
        wait_for_fit(20261110)
        tasks = [[train, 'train', '--kind', 'soft', '--seed', '20261112'],
                 evaluation('static', 'soft', 20261112), evaluation('integrated', 'soft', 20261112),
                 evaluation('integrated', 'content', 20261112), evaluation('retention', 'soft'),
                 ['experiments.gaussian_memory_soft_cost']]
    else:
        wait_for_fit(20261111); wait_for_fit(20261110)
        tasks = []
        for seed in (20261111, 20261110):
            tasks += [evaluation('static', 'soft', seed), evaluation('integrated', 'soft', seed),
                      evaluation('integrated', 'content', seed)]
        tasks += [evaluation('interventions'), evaluation('retention', 'content')]
    dest = OUT / f'queue_{a.worker}.json'
    previous = json.loads(dest.read_text()) if dest.exists() else {}
    migrations = previous.get('gpu_relocations', [])
    if previous:
        assert previous['tasks'] == tasks
        if previous['gpu'] != a.gpu:
            assert a.relocate_stopped_queue and not previous['complete'] and previous.get('returncode', 0) != 0
            migrations = migrations + [{'from': previous['gpu'], 'to': a.gpu, 'at_task': previous['current_task'],
                                       'reason': 'verified old queue stopped; newly available GPU; settings unchanged'}]
    done = previous.get('completed_tasks', []); start = time.monotonic()
    state = {'worker': a.worker, 'gpu': a.gpu, 'tasks': tasks, 'completed_tasks': done, 'complete': False,
             'gpu_relocations': migrations}
    with (OUT / f'queue_{a.worker}.log').open('a', buffering=1) as log:
        for index, task in enumerate(tasks):
            if index in done: continue
            state.update(current_task=index, current_command=task); save(dest, state)
            command = [sys.executable, '-u', '-m', 'experiments.gaussian_memory_launch', '--gpu', a.gpu,
                       '--allow-shared', '--wait-for-gpu', *task]
            print(json.dumps({'worker': a.worker, 'start_task': index, 'command': task}), flush=True)
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(failed_task=index, returncode=result.returncode); save(dest, state)
                raise SystemExit(result.returncode)
            done.append(index); state['completed_tasks'] = done; save(dest, state)
        state.update(complete=True, seconds_this_invocation=time.monotonic() - start); save(dest, state)
    print(json.dumps({'worker': a.worker, 'complete': True}), flush=True)


if __name__ == '__main__': main()
