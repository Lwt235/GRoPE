"""Finite 12-fit GPU queue, each launch independently verifies live GPU occupancy."""
import json
import os
import subprocess
import sys
import time

import numpy as np

from experiments.gaussian_memory_native_pe_data import ROOT, OUT, KINDS, SEEDS, load_plan, read, write_json, sha256


def main():
    plan = load_plan(); assert read(OUT / 'preflight_final/complete.json')['complete']
    gpus = ['2', '3', '4', '5', '7']; pending = [(k, s) for s in SEEDS for k in KINDS]
    active = {}; done = []; failed = []; started = time.monotonic()
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    while pending or active:
        for gpu, job in list(active.items()):
            code = job['process'].poll()
            if code is None: continue
            job['log'].close(); row = {'gpu': gpu, 'kind': job['kind'], 'seed': job['seed'], 'exit_code': code,
                                      'seconds': time.monotonic() - job['start']}
            (done if code == 0 else failed).append(row); del active[gpu]
            print(json.dumps({'finished': row}), flush=True)
        for gpu in gpus:
            if gpu in active or not pending: continue
            kind, seed = pending.pop(0); log_path = OUT / f'{kind}_{seed}.log'; log = log_path.open('a', buffering=1)
            command = [sys.executable, '-u', '-m', 'experiments.gaussian_memory_launch', '--gpu', gpu, '--wait-for-gpu',
                       'experiments.gaussian_memory_native_pe_train', 'train', '--kind', kind, '--seeds', str(seed)]
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = {'process': child, 'log': log, 'kind': kind, 'seed': seed, 'start': time.monotonic()}
            print(json.dumps({'started': [gpu, kind, seed], 'pid': child.pid}), flush=True)
        state = {'complete': not pending and not active and not failed, 'active': [
            {'gpu': g, 'kind': j['kind'], 'seed': j['seed'], 'launcher_pid': j['process'].pid} for g, j in active.items()],
            'pending': pending, 'done': done, 'failed': failed, 'seconds': time.monotonic() - started,
            'plan_sha256': sha256(OUT / 'plan.json'), 'training_lock_sha256': sha256(OUT / 'training_lock.json'),
            'queue_source_sha256': sha256(ROOT / 'experiments/gaussian_memory_native_pe_queue.py')}
        write_json(OUT / 'queue.json', state)
        if active: time.sleep(5)
    write_json(OUT / 'queue_complete.json', state)
    if failed: raise SystemExit('One or more finite tasks failed; see queue_complete.json')


if __name__ == '__main__': main()
