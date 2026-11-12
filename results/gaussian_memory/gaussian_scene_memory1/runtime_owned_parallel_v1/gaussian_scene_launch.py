"""One12GiB shared scene worker under the user's standing spare-VRAM authorization."""
import argparse
import json
import os
import subprocess
import sys
import time

from experiments.gaussian_scene_data import DATA,ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',required=True)
    parser.add_argument('module',choices=['experiments.gaussian_scene_train','experiments.gaussian_scene_reference',
                                        'experiments.gaussian_scene_carrier','experiments.gaussian_scene_cost',
                                        'experiments.gaussian_scene_build','experiments.gaussian_scene_cache',
                                        'experiments.gaussian_scene_preflight'])
    parser.add_argument('args',nargs=argparse.REMAINDER);a=parser.parse_args()
    checks=[]
    for _ in range(2):
        row=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(', ')
        uuid=row[0];used,free,util=map(int,row[1:]);checks.append(dict(uuid=uuid,used_MiB=used,free_MiB=free,util_percent=util))
        if free<24000:raise SystemExit('Selected GPU no longer meets the occupancy rule; no job launched: '+json.dumps(checks))
        time.sleep(2)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(DATA/'torch_extensions'))
    env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH']
    runner='''import sys,runpy
import numpy as np
import torch
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
sys.argv=sys.argv[1:]
runpy.run_module(sys.argv[0],run_name="__main__")
'''
    print(json.dumps(dict(launch=a.module,args=a.args,gpu_checks=checks,shared_worker=True,memory_cap_GiB=12,
        authorization='User previously authorized a small number of spare-VRAM workers while GPUs are occupied; scene queue allows only one',
        timing_scope='Shared GPU; not an exclusive speed benchmark')),flush=True)
    result=subprocess.run([sys.executable,'-u','-c',runner,a.module,*a.args],cwd=ROOT,env=env)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
