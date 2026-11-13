"""Own-child resource protection and total active-worker budget accounting."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from experiments.native_memory_context_data import *
from experiments.native_memory_context_resources import check


def active_seconds():
    total=0.
    for path in (OUT/'runtime').glob('*.json'):
        r=read(path)
        if r.get('runtime_session'):total+=r.get('seconds',time.time()-r['start_epoch'])
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--cap',type=float,default=12);p.add_argument('--admission',type=int,default=18432);p.add_argument('module');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    assert a.module.startswith('experiments.native_memory_context_');initialize();OUT.mkdir(parents=True,exist_ok=True);folder=OUT/'runtime';folder.mkdir(exist_ok=True)
    fields=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ');uuid,free,util=fields
    if int(free)<max(a.admission,a.cap*1024+6144):raise SystemExit(75)
    if active_seconds()>=8*3600:raise RuntimeError('Hard GPU worker-hour budget exhausted')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(DATA/'torch_extensions'))
    env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH']
    runner='import sys,runpy\nimport numpy\nimport torch\ntorch.set_num_threads(4)\ncap=float(sys.argv[1]);torch.cuda.set_per_process_memory_fraction(cap*2**30/torch.cuda.get_device_properties(0).total_memory)\nsys.argv=sys.argv[2:];runpy.run_module(sys.argv[0],run_name="__main__")\n'
    started=time.time();path=folder/f'{time.time_ns()}_{a.module.rsplit("_",1)[-1]}.json'
    record=dict(runtime_session=True,start_epoch=started,started_utc=now(),gpu=int(a.gpu),GPU_UUID=uuid,cap_GiB=a.cap,admission_free_MiB=int(free),module=a.module,args=a.args,parent_pid=os.getpid())
    write(path,record);child=subprocess.Popen([sys.executable,'-u','-c',runner,str(a.cap),a.module,*a.args],cwd=ROOT,env=env);record['pid']=child.pid;write(path,record)
    yielding=False;previous=check();last_space=time.monotonic()
    while child.poll() is None:
        free=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        space=check(previous) if time.monotonic()-last_space>=30 else previous
        if space is not previous:previous=space;last_space=time.monotonic()
        if free<6144 or active_seconds()>=8*3600 or space['stop_reasons']:
            yielding=True;record['yield_reason']='resource_margin' if free<6144 else 'disk_resource' if space['stop_reasons'] else 'hard_worker_time_cap';record['space_at_yield']=space;record['free_MiB_at_yield']=free;child.terminate();break
        time.sleep(3)
    if yielding:
        try:child.wait(timeout=240)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    record.update(returncode=75 if yielding else child.wait(),seconds=time.time()-started,process_exited=True,finished_utc=now());write(path,record)
    print(dict(session=str(path),seconds=record['seconds'],returncode=record['returncode']),flush=True)
    raise SystemExit(record['returncode'])


if __name__=='__main__':main()
