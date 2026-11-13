"""Own-child GPU/disk/time watchdog; no scientific settings changes."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from experiments import camera_only_feature_data as d

def running_seconds():
    total=5.
    for p in (d.OUT/'runtime').glob('*.json'):
        r=d.read(p)
        if r.get('GPU_session'):total+=r.get('seconds',time.time()-r['start_epoch'])
    return total

def main():
    a=argparse.ArgumentParser();a.add_argument('--gpu',required=True);a.add_argument('--worker',required=True);a.add_argument('--fits',default='');a.add_argument('--preflight',action='store_true');a.add_argument('--baselines',action='store_true');a.add_argument('--cost',action='store_true');args=a.parse_args()
    folder=d.OUT/'runtime';folder.mkdir(exist_ok=True);gpu_guard=(folder/f'gpu{args.gpu}.lock').open('a');fcntl.flock(gpu_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid,free=[x.strip() for x in subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);foreign=any(line.split(',')[0].strip()==uuid for line in apps.splitlines())
    assert int(free)>=16384 and running_seconds()<6*3600
    with (folder/'admission.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        running=[d.read(p) for p in folder.glob('*.json') if d.read(p).get('GPU_session') and not d.read(p).get('process_exited')]
        assert len(running)<2 and sum(bool(x.get('foreign_shared')) for x in running)+int(foreign)<=1
        space=d.resources(admission=True);assert not space['stop_reasons'],space
        path=folder/f'{args.worker}_{time.time_ns()}.json';start=time.time();record=dict(GPU_session=True,worker=args.worker,gpu=args.gpu,GPU_UUID=uuid,start_epoch=start,started_utc=d.now(),foreign_shared=foreign,admission_free_MiB=int(free),cap_GiB=10,parent_pid=os.getpid())
        d.write(path,record)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.old.DATA/'torch_extensions'))
    env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH']
    command=[sys.executable,'-B','-u','-m','experiments.camera_only_feature_run','preflight' if args.preflight else 'worker','--worker',args.worker]
    if args.fits:command+=['--fits',args.fits]
    if args.baselines:command+=['--baselines']
    if args.cost:command+=['--cost']
    child=subprocess.Popen(command,cwd=d.ROOT,env=env);record['pid']=child.pid;d.write(path,record);last=0;peak=0;previous=space;yielding=False
    while child.poll() is None:
        if time.monotonic()-last>=30:
            previous=d.resources(previous);last=time.monotonic()
            current=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
            own=[int(row.split(',')[2].strip()) for row in current.splitlines() if row.split(',')[1].strip()==str(child.pid)]
            peak=max([peak]+own);free_now=int(subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if previous['stop_reasons'] or free_now<6144 or peak>10240 or running_seconds()>=6*3600-15:
                yielding=True;record['yield_reason']=dict(space=previous,free_MiB=free_now,own_peak_MiB=peak,worker_seconds=running_seconds());child.terminate()
        time.sleep(3)
    code=child.wait();record.update(process_exited=True,exit_code=code,seconds=time.time()-start,finished_utc=d.now(),peak_process_MiB_sampled=peak,resource_yield=yielding);d.write(path,record)
    print(dict(runtime=str(path),exit_code=code,seconds=record['seconds']),flush=True);raise SystemExit(code)

if __name__=='__main__':main()
