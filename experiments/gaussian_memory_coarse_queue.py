"""Finite preprocessing/training queue; launch only on currently idle GPUs."""
import argparse
import json
import os
import subprocess
import sys
import time
import signal
import numpy as np
from experiments.gaussian_memory_coarse_data import *


def occupancy():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {r[0]:(int(r[1]),int(r[2]),int(r[3])) for line in raw.splitlines() if (r:=[x.strip() for x in line.split(',')])}


class ExistingProcess:
    """Monitor our live launcher after a scheduler-only replacement; never restart its fit."""
    def __init__(self,pid,args):self.pid=pid;self.args=args
    def poll(self):
        try:
            stat=Path(f'/proc/{self.pid}/stat').read_text();alive=stat.split(') ',1)[1][0]!='Z'
        except FileNotFoundError:alive=False
        if alive:return None
        kind=self.args[self.args.index('--kind')+1];seed=self.args[self.args.index('--seed')+1]
        path=OUT/'evaluation'/f'{kind}_{seed}.json'
        return 0 if path.exists() and read(path)['complete'] else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['cache','train']);parser.add_argument('--resume-parent',type=int);a=parser.parse_args();p=plan();verify(p)
    if a.action=='cache':pending=[['--shard',str(i),'--shards','4'] for i in range(4)];module='experiments.gaussian_memory_coarse_cache'
    else:
        assert read(OUT/'preflight/complete.json')['complete'] and read(OUT/'independent_input_checks.json')['complete'] and (OUT/'training_lock.json').exists()
        pending=[['train','--kind',k,'--seed',str(s)] for s in SEEDS for k in KINDS];module='experiments.gaussian_memory_coarse_train'
    active={};done=[];failed=[];begin=time.monotonic();env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    prior_seconds=0.
    if a.resume_parent:
        assert a.action=='train'
        old=read(OUT/'train_queue_before_resume.json');prior_seconds=old['seconds'];done=old['done'];failed=old['failed']
        pending=old['pending']
        for job in old['active']:
            active[job['gpu']]=dict(process=ExistingProcess(job['pid'],job['args']),log=open(os.devnull,'w'),args=job['args'],start=begin,shared=False,reattached=True)
        print(json.dumps(dict(scheduler_resumed=True,old_parent=a.resume_parent,reattached=[j['process'].pid for j in active.values()],pending=len(pending))),flush=True)
    while pending or active:
        for gpu,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();row=dict(gpu=gpu,args=job['args'],exit_code=code,seconds=time.monotonic()-job['start'])
            if job.get('reattached'):row['seconds_scope']='since scheduler reattachment; full training duration in fit status'
            (done if code==0 else failed).append(row);del active[gpu];print(json.dumps({'finished':row}),flush=True)
        occ=occupancy()
        for gpu in ['2','3','4','5','7','6']:
            if gpu in active or not pending or failed:continue
            used,free,util=occ[gpu]
            shared=gpu=='6'
            if shared:
                if any(j.get('shared',False) for j in active.values()) or free<20000 or util>60:continue
            elif used>512 or free<40000 or util>5:continue
            args=pending.pop(0);name='_'.join(args).replace('--','');logpath=OUT/'logs'/f'{a.action}_{name}.log';logpath.parent.mkdir(parents=True,exist_ok=True);log=logpath.open('a',buffering=1)
            command=[sys.executable,'-u','-m','experiments.gaussian_memory_launch','--gpu',gpu,'--wait-for-gpu',*(['--allow-shared'] if shared else []),module,*args]
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=dict(process=child,log=log,args=args,start=time.monotonic(),shared=shared);print(json.dumps(dict(started=args,gpu=gpu,pid=child.pid,shared=shared)),flush=True)
        status=dict(complete=not pending and not active and not failed,active=[dict(gpu=g,args=j['args'],pid=j['process'].pid,shared=j.get('shared',False)) for g,j in active.items()],pending=pending,done=done,failed=failed,seconds=prior_seconds+time.monotonic()-begin,plan_sha256=sha256(OUT/'plan.json'),scheduler_source_sha256=sha256(ROOT/'experiments/gaussian_memory_coarse_queue.py'))
        write_json(OUT/f'{a.action}_queue.json',status)
        if failed:
            # Finish active work, do not launch additional jobs after an implementation failure.
            if pending:write_json(OUT/f'{a.action}_unlaunched_after_failure.json',pending);pending=[]
        if pending or active:time.sleep(10)
    write_json(OUT/f'{a.action}_queue_complete.json',status)
    if a.resume_parent:
        # The previous process was stopped only as a controller. All of its original
        # fits are finished now; retire that controller without signalling workers.
        os.kill(a.resume_parent,signal.SIGTERM);os.kill(a.resume_parent,signal.SIGCONT)
    if failed:raise SystemExit('Finite job failed; inspect preserved logs')


if __name__=='__main__':main()
