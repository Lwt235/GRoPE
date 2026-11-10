"""Run the sealed inference on one live-selected GPU with a tighter memory cap."""
import argparse
import os
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('preflight', 'export'))
args = parser.parse_args()
lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'], text=True).strip().splitlines()
cards = [[part.strip() for part in line.split(',')] for line in lines]
candidates = [r for r in cards if int(r[3]) >= 8192]
if not candidates:
    print('NO_GPU_WITH_8_GIB_RESERVE', flush=True)
    raise SystemExit(75)
card = max(candidates, key=lambda r: int(r[3]))
os.environ['CUDA_VISIBLE_DEVICES'] = card[0]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
import torch
from experiments import modelnet_cross_source as experiment

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
lock = experiment.check()
# The sealed 16 GiB allocation ceiling is an upper bound, not a reservation.
# This execution uses a stricter 4 GiB ceiling to protect concurrent jobs.
experiment.CAP_BYTES = 4 * 2**30
assert experiment.CAP_BYTES <= lock['allocator_cap_bytes']
status_path = experiment.OUT / (args.phase + '_runtime.json')
status = dict(pid=os.getpid(), started_unix=time.time(), gpu=card, shared_at_launch=int(card[2]) > 512,
              CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
              effective_allocator_cap_bytes=experiment.CAP_BYTES,
              sealed_maximum_allocator_cap_bytes=lock['allocator_cap_bytes'],
              lock_sha256=experiment.sha(experiment.OUT/'lock.json'), completed=[])
experiment.write(status_path, dict(status, state='running'))
print('LIVE_GPU',card,'cap_gib',4,'phase',args.phase,flush=True)
try:
    if args.phase == 'preflight':
        for shard in range(experiment.PREFLIGHT_SHARDS):
            target = experiment.OUT/'preflight'/f'{shard}.json'
            if target.exists():
                prior = experiment.read(target)
                assert prior['passed'] and prior['lock_sha256'] == status['lock_sha256']
            else:
                experiment.preflight(shard)
            status['completed'].append(shard)
            experiment.write(status_path, dict(status,state='running'))
    else:
        experiment.ready()
        for model in lock['models']:
            experiment.export(model['role'], model['seed'])
            status['completed'].append([model['role'],model['seed']])
            experiment.write(status_path, dict(status,state='running'))
except Exception as error:
    experiment.write(status_path, dict(status,state='failed',finished_unix=time.time(),
                                       error_type=type(error).__name__))
    raise
experiment.write(status_path,dict(status,state='complete',finished_unix=time.time()))
print('GPU_PHASE_COMPLETE',args.phase,flush=True)
