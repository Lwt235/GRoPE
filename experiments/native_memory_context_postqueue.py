"""Reporting-only sequential endpoint controller; never starts new fits."""
import os
import subprocess
import sys
import time
from experiments.native_memory_context_data import *
from experiments.gaussian_scene_orchestrate import occupancy

def run(name,args):
    folder=OUT/'postqueue';folder.mkdir(exist_ok=True);log=folder/f'{name}.log';write(folder/'status.json',dict(stage=name,state='running',pid=os.getpid(),epoch=time.time()))
    with log.open('a',buffering=1) as stream:
        result=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:write(folder/'status.json',dict(stage=name,state='failed',returncode=result.returncode,log=str(log)));raise SystemExit(result.returncode)

def main():
    folder=OUT/'postqueue';folder.mkdir(exist_ok=True)
    import fcntl
    with (OUT/'postqueue.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            s=read(OUT/'queue_status.json')
            if s['state']=='failed':write(folder/'status.json',dict(state='queue_failed',queue=s));return
            if s['state']=='complete' and not s['running']:break
            write(folder/'status.json',dict(state='waiting_unique_existing_queue',pid=os.getpid(),epoch=time.time(),complete=len(s['complete'])));time.sleep(20)
        run('score',['-m','experiments.native_memory_context_score'])
        while True:
            idle=[g for g in occupancy() if g[1]<=512 and g[2]>=20480 and g[3]<=5]
            if idle:break
            time.sleep(20)
        gpu=idle[0][0];run('cost',['-m','experiments.native_memory_context_launch','--gpu',str(gpu),'--cap','14','--admission','20480','experiments.native_memory_context_cost'])
        run('finish',['-m','experiments.native_memory_context_finish'])
        run('figures',['-m','experiments.native_memory_context_figures'])
        write(folder/'status.json',dict(state='fixed_endpoint_ready_for_actual_visual_review',pid=os.getpid(),epoch=time.time(),additional_training=False))
if __name__=='__main__':main()
