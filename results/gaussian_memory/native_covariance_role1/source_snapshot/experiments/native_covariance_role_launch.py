"""Monitor only our own child; enforce disk, GPU, calls and worker-time budget."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from experiments.native_covariance_role import OUT,CACHE,d,now,write,resources


def active_seconds():
    total=2.1  # Conservative initial CUDA-access probe wall time.
    for path in (OUT/'runtime').glob('*.json'):
        r=d.read(path)
        if r.get('runtime_session'):total+=r.get('seconds',time.time()-r['start_epoch'])
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('command',choices=['prepare','preflight','run']);p.add_argument('--seeds',type=int,nargs='+',default=[20261110]);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);folder=OUT/'runtime';folder.mkdir(exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    guard=(folder/f'gpu{a.gpu}.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid,free,util=[x.strip() for x in subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
    assert int(free)>=12288 and active_seconds()<3600
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    foreign=any(row.split(',')[0].strip()==uuid for row in apps.splitlines())
    with (folder/'admission.lock').open('a') as admission:
        fcntl.flock(admission,fcntl.LOCK_EX)
        running=[d.read(x) for x in folder.glob('*.json') if d.read(x).get('runtime_session') and not d.read(x).get('process_exited')]
        assert len(running)<2 and sum(bool(x.get('foreign_shared')) for x in running)+int(foreign)<=1
        space=resources();assert not space['stop_reasons'] and space['projected_headroom_bytes']>=8*2**30,space
        started=time.time();path=folder/f'{time.time_ns()}_{a.command}.json'
        record=dict(runtime_session=True,start_epoch=started,started_utc=now(),gpu=int(a.gpu),GPU_UUID=uuid,cap_GiB=8,admission_free_MiB=int(free),foreign_shared=foreign,command=a.command,seeds=a.seeds,parent_pid=os.getpid())
        write(path,record)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.DATA/'torch_extensions'))
    env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH']
    runner='import sys,runpy\nimport numpy\nimport torch\ntorch.set_num_threads(4)\ntorch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)\nsys.argv=sys.argv[1:];runpy.run_module(sys.argv[0],run_name="__main__")\n'
    child=subprocess.Popen([sys.executable,'-B','-u','-c',runner,'experiments.native_covariance_role',a.command,'--seeds',*[str(x) for x in a.seeds]],cwd=d.ROOT,env=env)
    record['pid']=child.pid;write(path,record);previous=space;last=time.monotonic();yielding=False
    while child.poll() is None:
        free=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if time.monotonic()-last>=30:previous=resources(previous);last=time.monotonic()
        if free<4096 or active_seconds()>=3590 or previous['stop_reasons']:
            yielding=True;record['yield_reason']=dict(free_MiB=free,active_seconds=active_seconds(),space=previous);child.terminate();break
        time.sleep(3)
    if yielding:
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    record.update(returncode=75 if yielding else child.wait(),seconds=time.time()-started,process_exited=True,finished_utc=now());write(path,record)
    print(dict(session=str(path),seconds=record['seconds'],returncode=record['returncode']),flush=True)
    raise SystemExit(record['returncode'])


if __name__=='__main__':main()
