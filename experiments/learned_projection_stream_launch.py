"""Lower-residency worker:7GiB cap with measured6GiB spare margin."""
import argparse
import json
import os
import subprocess
import sys
import time

from experiments.learned_projection_data import ROOT,DATA,OUT,write_json


def main():
    a=argparse.ArgumentParser();a.add_argument('--gpu',required=True)
    a.add_argument('module',choices=['experiments.learned_projection_stream_check','experiments.learned_projection_stream_train'])
    a.add_argument('args',nargs=argparse.REMAINDER);args=a.parse_args();checks=[]
    for _ in range(2):
        row=subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(', ')
        uuid=row[0];used,free,util=map(int,row[1:]);checks.append(dict(uuid=uuid,used_MiB=used,free_MiB=free,util_percent=util))
        if free<13312:raise SystemExit('Staged worker needs13GiB initial free memory; no job launched: '+json.dumps(checks))
        time.sleep(2)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(DATA/'torch_extensions'))
    env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH']
    runner='''import sys,runpy
import numpy as np
import torch
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory)
sys.argv=sys.argv[1:]
runpy.run_module(sys.argv[0],run_name="__main__")
'''
    print(json.dumps(dict(launch=args.module,args=args.args,gpu_checks=checks,memory_cap_GiB=7,
        reserve_scope='CPU-staged exact tensors;13GiB admission (7GiB cap plus6GiB margin); no change to model, data, optimizer or scientific budget')),flush=True)
    process=subprocess.Popen([sys.executable,'-u','-c',runner,args.module,*args.args],cwd=ROOT,env=env)
    while process.poll() is None:
        free=int(subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip())
        if free<6144:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            path=OUT/'memory_staging/yields'/f'{args.module.rsplit(".",1)[-1]}_{int(time.time())}.json'
            write_json(path,dict(gpu=args.gpu,free_MiB=free,stopped_only_owned_pid=process.pid,
                checkpoint_preserved=True,reason='Staged7GiB worker yields below6GiB free margin',module=args.module,args=args.args))
            print('Staged resource yield; checkpoint preserved; returncode75',flush=True)
            raise SystemExit(75)
        time.sleep(2)
    raise SystemExit(process.returncode)


if __name__=='__main__':main()
