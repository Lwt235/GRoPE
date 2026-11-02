"""Launch only this experiment's seed shards, on repeatedly observed idle GPUs."""
import os
import json
import subprocess
import sys
import time
from pathlib import Path
from experiments.round50_data import OUT, ROOT, write
from experiments.round50_train import SEEDS, check_lock, seal, report


def occupancy():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(parts[0]):(int(parts[1]),int(parts[2])) for parts in (line.split(',') for line in out.splitlines())}


def run():
    check_lock(OUT)
    status_path=OUT/'queue_status.json'
    if status_path.exists():raise FileExistsError('retain earlier controller status')
    history=[];state=dict(state='running',active=[],children=history)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for stage in ('train','eval'):
        if stage=='eval':seal(OUT)
        first=occupancy();time.sleep(5);second=occupancy()
        available=[g for g in second if second[g][0]<128 and second[g][1]<5 and first.get(g,(1e6,100))[0]<128 and first[g][1]<5]
        if len(available)<len(SEEDS):raise RuntimeError('fewer than three idle GPUs; do not occupy another job')
        jobs=[]
        for seed,gpu in zip(SEEDS,available):
            # Recheck immediately before launch as well.
            if occupancy()[gpu][0]>=128:raise RuntimeError('GPU newly occupied')
            log=OUT/f'{stage}_{seed}.log';handle=open(log,'x')
            cmd=[sys.executable,'-m','experiments.round50_train',stage,'--seed',str(seed),'--gpu',str(gpu)]
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=env)
            rec=dict(stage=stage,seed=seed,gpu=gpu,pid=proc.pid,command=cmd,log=str(log),started=time.time())
            jobs.append((proc,handle,rec));print('START',stage,seed,'GPU',gpu,'PID',proc.pid,flush=True)
        while jobs:
            active=[];remaining=[]
            for proc,handle,rec in jobs:
                code=proc.poll()
                if code is None:active.append(rec);remaining.append((proc,handle,rec))
                else:
                    handle.close();rec.update(exit_code=code,finished=time.time());history.append(rec)
                    print('EXIT',rec['stage'],rec['seed'],code,flush=True)
                    if code!=0:
                        # No automatic relaunch/overwrite. Record peers and leave their
                        # already authorized bounded jobs to finish; never kill other jobs.
                        state['failure']=rec
            jobs=remaining;state.update(stage=stage,active=active);write(status_path,state)
            if jobs:time.sleep(10)
        if any(r['exit_code']!=0 for r in history):
            state.update(state='failed',active=[]);write(status_path,state);return 1
    report(OUT);check_lock(OUT)
    state.update(state='complete',active=[],finished=time.time());write(status_path,state)
    return 0


if __name__=='__main__':sys.exit(run())
