"""CPU-only queues; acquire a GPU only after the owning training stage completes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gaussian_memory/bound_read1'


def complete(name):
    path = OUT / name
    return path.exists() and json.loads(path.read_text()).get('complete', False)


def wait_for(names, report):
    while not all(complete(name) for name in names):
        report(stage='waiting_for_dependencies', waiting_for=[name for name in names if not complete(name)])
        time.sleep(30)


def main():
    p = argparse.ArgumentParser(); p.add_argument('worker', choices=('primary', 'aux'))
    p.add_argument('--gpu', required=True); args = p.parse_args()
    lockfile = (OUT / f'evaluation_queue_{args.worker}.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic(); status_path = OUT / f'evaluation_queue_{args.worker}.json'

    def report(**values):
        record = {'pid': os.getpid(), 'worker': args.worker, 'gpu': args.gpu,
                  'elapsed_s': time.monotonic() - start, 'updated_unix': time.time(), **values}
        temp = status_path.with_suffix('.tmp'); temp.write_text(json.dumps(record, indent=2) + '\n'); temp.replace(status_path)

    def launch(arguments, label):
        logpath = OUT / f'evaluation_queue_{args.worker}_{label}.log'
        command = [sys.executable, '-u', '-m', 'experiments.gaussian_memory_launch', '--gpu', args.gpu,
                   '--allow-shared', '--wait-for-gpu', 'experiments.gaussian_memory_bound_finish', *arguments]
        with logpath.open('a', buffering=1) as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            report(stage='gpu_launcher_running', child_pid=proc.pid, log=str(logpath), task=label)
            code = proc.wait()
        if code:
            report(stage='worker_exited_with_error', task=label, returncode=code, log=str(logpath), complete=False)
            raise SystemExit(code)
        report(stage='stage_complete', task=label, returncode=0)

    # These barriers avoid competing with the still-running training worker on the same GPU.
    wait_for([f'train_{args.worker}_status.json'], report)
    seeds = (20261111,) if args.worker == 'primary' else (20261112, 20261110)
    for seed in seeds:
        wait_for([f'{kind}_{seed}/status.json' for kind in ('local', 'content', 'coverage', 'bound')], report)
        if not complete(f'evaluation_seed_{seed}_complete.json'):
            launch(['seed', '--seed', str(seed)], f'seed{seed}')
    if args.worker == 'aux':
        wait_for([f'evaluation_seed_{seed}_complete.json' for seed in (20261110, 20261111, 20261112)], report)
        if not complete('evaluation_pipeline_complete.json'):
            launch(['final'], 'final')
    report(stage='queue_complete', complete=True)


if __name__ == '__main__': main()
