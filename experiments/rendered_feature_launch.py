"""Seven-GiB own worker, with measured free-memory admission and yielding."""
import argparse
import json
import os
import subprocess
import sys
import time

from experiments.rendered_feature_data import ROOT, DATA, OUT, write_json


def main():
    a=argparse.ArgumentParser();a.add_argument('--gpu',required=True)
    a.add_argument('module',choices=['experiments.rendered_feature_prepare',
        'experiments.rendered_feature_train','experiments.rendered_feature_cost'])
    a.add_argument('args',nargs=argparse.REMAINDER);args=a.parse_args();checks=[]
    for _ in range(2):
        parts=subprocess.check_output(['nvidia-smi','-i',args.gpu,
            '--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(', ')
        uuid=parts[0];used,free,util=map(int,parts[1:])
        checks.append(dict(uuid=uuid,used_MiB=used,free_MiB=free,util_percent=util))
        if free<13312:
            print('no job launched: fewer than13GiB available '+json.dumps(checks),flush=True)
            raise SystemExit(75)
        time.sleep(2)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',
        TORCH_EXTENSIONS_DIR=str(DATA/'torch_extensions'))
    env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH']
    runner='''import sys,runpy
import numpy as np
import torch
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory)
sys.argv=sys.argv[1:]
runpy.run_module(sys.argv[0],run_name="__main__")
'''
    print(json.dumps(dict(module=args.module,args=args.args,checks=checks,
        memory_cap_GiB=7,yield_below_free_GiB=6)),flush=True)
    process=subprocess.Popen([sys.executable,'-u','-c',runner,args.module,*args.args],cwd=ROOT,env=env)
    while process.poll() is None:
        free=int(subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip())
        if free<6144:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            write_json(OUT/'resource_yields'/f'{int(time.time_ns())}.json',dict(
                gpu=args.gpu,free_MiB=free,stopped_only_owned_pid=process.pid,
                module=args.module,args=args.args,checkpoint_preserved=True,
                reason='Own worker yielded below6GiB free margin; other jobs untouched'))
            raise SystemExit(75)
        time.sleep(2)
    raise SystemExit(process.returncode)


if __name__=='__main__':main()
