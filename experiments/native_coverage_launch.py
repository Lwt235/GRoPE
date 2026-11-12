"""Measured headroom and yielding limited to this experiment's own child."""
import argparse
import json
import os
import subprocess
import sys
import time
from experiments.native_coverage_data import ROOT,DATA,OUT,write_json


def main():
    a=argparse.ArgumentParser();a.add_argument('--gpu',required=True);a.add_argument('--cap',type=float,default=11)
    a.add_argument('--margin',type=int,default=6144);a.add_argument('--admission',type=int,default=17408)
    a.add_argument('module');a.add_argument('args',nargs=argparse.REMAINDER);args=a.parse_args()
    assert args.module.startswith('experiments.native_coverage_')
    checks=[]
    for _ in range(2):
        fields=subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(', ')
        uuid,free,util=fields;free=int(free);checks.append(dict(uuid=uuid,free_MiB=free,utilization=int(util)))
        if free<max(args.cap*1024+args.margin,args.admission):raise SystemExit(75)
        time.sleep(2)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(DATA/'torch_extensions'))
    env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH']
    runner='import sys,runpy\nimport numpy as np\nimport torch\ntorch.set_num_threads(4)\ncap=float(sys.argv[1]);torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(0).total_memory)\nsys.argv=sys.argv[2:];runpy.run_module(sys.argv[0],run_name="__main__")\n'
    print(json.dumps(dict(module=args.module,args=args.args,checks=checks,memory_cap_GiB=args.cap)),flush=True)
    child=subprocess.Popen([sys.executable,'-u','-c',runner,str(args.cap),args.module,*args.args],cwd=ROOT,env=env)
    while child.poll() is None:
        free=int(subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip())
        if free<args.margin:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            write_json(OUT/'resource_yields'/f'{time.time_ns()}.json',dict(gpu=args.gpu,free_MiB=free,stopped_only_owned_pid=child.pid,module=args.module,args=args.args))
            write_json(OUT/'worker_exits'/f'{os.getpid()}.json',dict(returncode=75,module=args.module,args=args.args))
            raise SystemExit(75)
        time.sleep(3)
    write_json(OUT/'worker_exits'/f'{os.getpid()}.json',dict(returncode=child.returncode,module=args.module,args=args.args))
    raise SystemExit(child.returncode)


if __name__=='__main__':main()
