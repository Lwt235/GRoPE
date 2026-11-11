"""Disjoint fixed queues; every child uses the existing shared GPU guard."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gaussian_memory/state_read1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def tasks_for(kind):
    train = 'experiments.gaussian_memory_state_train'; evaluate = 'experiments.gaussian_memory_state_evaluate'
    tasks = []
    for seed in (20261110, 20261111, 20261112):
        tasks += [[train, 'train', '--kind', kind, '--seed', str(seed)],
                  [evaluate, 'evaluate', '--kind', kind, '--seed', str(seed), '--policy', 'corrective_clean']]
    for seed in (20261110, 20261111, 20261112):
        tasks += [[evaluate, 'evaluate', '--kind', kind, '--seed', str(seed), '--policy', 'static'],
                  [evaluate, 'evaluate', '--kind', kind, '--seed', str(seed), '--policy', 'rebuild_clean']]
    if kind == 'soft': tasks += [[evaluate, 'interventions']]
    tasks += [[evaluate, 'retention', '--kind', kind]]
    return tasks


def main():
    p = argparse.ArgumentParser(); p.add_argument('kind', choices=('content', 'soft')); p.add_argument('--gpu', required=True)
    args = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / f'queue_{args.kind}.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((OUT / 'preflight/complete.json').read_text())['complete']
    tasks = tasks_for(args.kind); path = OUT / f'queue_{args.kind}.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous: assert previous['tasks'] == tasks and previous['gpu'] == args.gpu
    done = previous.get('completed_tasks', []); start = time.monotonic()
    state = {'kind': args.kind, 'gpu': args.gpu, 'tasks': tasks, 'completed_tasks': done, 'complete': False}
    with (OUT / f'queue_{args.kind}.log').open('a', buffering=1) as log:
        for index, task in enumerate(tasks):
            if index in done: continue
            state.update(current_task=index, current_command=task); save(path, state)
            command = [sys.executable, '-u', '-m', 'experiments.gaussian_memory_launch', '--gpu', args.gpu,
                       '--allow-shared', '--wait-for-gpu', *task]
            print(json.dumps({'kind': args.kind, 'start_task': index, 'command': task}), flush=True)
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(failed_task=index, returncode=result.returncode); save(path, state)
                raise SystemExit(result.returncode)
            done.append(index); state['completed_tasks'] = done; save(path, state)
        state.update(complete=True, seconds_this_invocation=time.monotonic() - start); save(path, state)
    print(json.dumps({'kind': args.kind, 'complete': True}), flush=True)


if __name__ == '__main__': main()
