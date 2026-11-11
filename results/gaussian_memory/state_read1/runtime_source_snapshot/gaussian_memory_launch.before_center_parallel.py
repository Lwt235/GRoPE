"""Launch one explicitly selected idle GPU job without disturbing other users."""
import argparse
import json
import os
import subprocess
import sys
import time

from experiments.gaussian_memory_setup import DATA, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True)
    p.add_argument('--allow-shared', action='store_true', help='one previously authorized shared worker only')
    p.add_argument('--wait-for-gpu', action='store_true', help='recheck every 30 seconds, up to 30 minutes')
    p.add_argument('module')
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    deadline = time.monotonic() + 1800
    while True:
        checks = []
        for _ in range(2):
            raw = subprocess.check_output(['nvidia-smi', '-i', a.gpu,
                '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                text=True, timeout=15).strip().split(', ')
            uuid, used, free, util = raw[0], *map(int, raw[1:])
            checks.append({'uuid': uuid, 'used_MiB': used, 'free_MiB': free, 'util_percent': util})
            acceptable = (free >= 20000 and util <= 60) if a.allow_shared else (used <= 512 and free >= 40000 and util <= 5)
            if not acceptable:
                break
            time.sleep(2)
        if acceptable and len(checks) == 2:
            break
        if not a.wait_for_gpu or time.monotonic() > deadline:
            raise SystemExit('Selected GPU no longer meets the occupancy rule; no job launched: ' + json.dumps(checks))
        print(json.dumps({'resource_wait': a.module, 'last_checks': checks}), flush=True)
        time.sleep(30)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MAX_JOBS='4', TORCH_CUDA_ARCH_LIST='8.6',
               TORCH_EXTENSIONS_DIR=str(DATA / 'torch_extensions'))
    env['PATH'] = '/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:' + env['PATH']
    print(json.dumps({'launch': a.module, 'args': a.args, 'gpu_checks': checks,
                      'shared_worker': a.allow_shared, 'memory_cap_GiB': 12 if a.allow_shared else None}), flush=True)
    if a.allow_shared:
        runner = '''import sys, runpy
import numpy as np
import torch
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(12 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__")
'''
        command = [sys.executable, '-u', '-c', runner, a.module, *a.args]
    else:
        command = [sys.executable, '-u', '-m', a.module, *a.args]
    result = subprocess.run(command, cwd=ROOT, env=env)
    raise SystemExit(result.returncode)


if __name__ == '__main__': main()
