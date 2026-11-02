"""Execution-only refinement: split frozen fits by seed AND geometry regime."""
import argparse
import hashlib
import os
import subprocess
import sys
import time
import numpy as np
import torch
from experiments import round50_train as P
from experiments.round50_queue import occupancy
from experiments.round50_data import OUT, ROOT, write, sha


def worker(seed, regime, gpu):
    P.check_lock(OUT);device=P.setup(gpu)
    fit=P.load(OUT,'fit','base',device);val=P.load(OUT,'val','base',device)
    rng=np.random.default_rng(seed+50000)
    ix=rng.integers(0,len(fit['qrgb']),size=(P.CONFIG['steps'],P.CONFIG['batch']),dtype=np.int64)
    indices=torch.from_numpy(ix).to(device)
    tasks=[(r,role) for r in P.REGIMES for role in P.ROLES]
    for i in np.random.default_rng(seed+80000).permutation(len(tasks)):
        r,role=tasks[i]
        if r==regime:P.fit_one(OUT,role,seed,regime,fit,val,indices,device)
    P.check_lock(OUT);print('WORKER COMPLETE',seed,regime,flush=True)


def execute():
    P.check_lock(OUT);status_path=OUT/'queue_status.json'
    if status_path.exists():raise FileExistsError('retain existing queue state')
    script_sha=sha(__file__)
    write(OUT/'execution_plan.json',dict(script_sha256=script_sha,lock_sha256=sha(OUT/'lock.json'),
        reason='Six host GPUs idle; split by seed/regime using unchanged frozen fit_one and sampling/order.',
        source_science_changed=False,max_gpus=6,cpu_threads_per_worker=4,original_queue_unexecuted=True))
    (OUT/'source/experiments/round50_execute.py').write_bytes(open(__file__,'rb').read())
    state=dict(state='running',active=[],children=[],gpu_observations=[])
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    write(status_path,state)
    for stage in ('train','eval'):
        if stage=='eval':P.seal(OUT)
        pending=[(seed,regime) for seed in P.SEEDS for regime in P.REGIMES] if stage=='train' else [(seed,None) for seed in P.SEEDS]
        running=[]
        while pending or running:
            still=[]
            for proc,log,rec in running:
                code=proc.poll()
                if code is None:still.append((proc,log,rec))
                else:
                    log.close();rec.update(exit_code=code,finished=time.time());state['children'].append(rec)
                    print('EXIT',rec['tag'],code,flush=True)
            running=still
            failed=any(r['exit_code']!=0 for r in state['children'])
            if failed:pending=[]
            if pending:
                first=occupancy();time.sleep(5);second=occupancy()
                used={rec['gpu'] for _,_,rec in running}
                available=[g for g in second if g not in used and second[g][0]<128 and second[g][1]<5 and first.get(g,(1e6,100))[0]<128 and first[g][1]<5]
                state['gpu_observations'].append(dict(time=time.time(),first=first,second=second))
                while available and pending and len(running)<6:
                    gpu=available.pop(0)
                    if occupancy()[gpu][0]>=128:continue
                    seed,regime=pending.pop(0);tag=f'{stage}_{seed}'+(' under '+regime if regime else '')
                    logpath=OUT/(tag.replace(' ','_')+'.log');log=open(logpath,'x')
                    if stage=='train':cmd=[sys.executable,'-m','experiments.round50_execute','worker','--seed',str(seed),'--regime',regime,'--gpu',str(gpu)]
                    else:cmd=[sys.executable,'-m','experiments.round50_train','eval','--seed',str(seed),'--gpu',str(gpu)]
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    rec=dict(stage=stage,tag=tag,seed=seed,regime=regime,gpu=gpu,pid=proc.pid,log=str(logpath),command=cmd,started=time.time())
                    running.append((proc,log,rec));print('START',tag,'GPU',gpu,'PID',proc.pid,flush=True)
            state.update(stage=stage,active=[r for _,_,r in running],pending=pending)
            write(status_path,state)
            if running or pending:time.sleep(10)
        if any(r['exit_code']!=0 for r in state['children']):
            state.update(state='failed',active=[]);write(status_path,state);return 1
    P.report(OUT);P.check_lock(OUT)
    if sha(__file__)!=script_sha:raise ValueError('execution source changed')
    state.update(state='complete',active=[],finished=time.time());write(status_path,state)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['queue','worker'])
    p.add_argument('--seed',type=int);p.add_argument('--regime',choices=P.REGIMES);p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.action=='queue':sys.exit(execute())
    worker(a.seed,a.regime,a.gpu)
