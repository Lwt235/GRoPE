"""Run the frozen nine-fit experiment only on observed idle GPUs."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from experiments.round48_queue import occupancy
from experiments.round57_common import ROOT,OUT,NEW_ROLES,SEEDS,checked,read,sha,atomic_json

ACTIVE,DONE,INFO={},{},{}


def record(state,**extra):
    atomic_json(OUT/'queue_status.json',dict(INFO,state=state,observed_unix=time.time(),
        completed=list(DONE.values()),active=[x['meta'] for x in ACTIVE.values()],**extra))


def launch(stage,job,gpu):
    tag=stage if job is None else stage+'_'+'_'.join(str(v) for v in job)
    if stage in ('training_audit','metrics'):
        command=[sys.executable,'-m','experiments.round57_audit','training' if stage=='training_audit' else 'metrics']
    else:
        command=[sys.executable,'-m','experiments.round57_train',stage]
        if stage=='train':command+=['--role',job[0],'--seed',str(job[1])]
        elif stage=='export':command+=['--seed',str(job[0])]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'])
    log=OUT/(tag+'.log');stream=log.open('x')
    try:process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException:stream.close();raise
    meta=dict(tag=tag,stage=stage,job=job,pid=process.pid,process_group=process.pid,command=command,
        gpu=gpu,started_unix=time.time(),log=str(log))
    ACTIVE[tag]=dict(process=process,stream=stream,meta=meta)
    print('LAUNCHED',tag,process.pid,'gpu',None if gpu is None else gpu['index'],flush=True)


def collect():
    failures=[]
    for tag,item in list(ACTIVE.items()):
        code=item['process'].poll()
        if code is None:continue
        item['stream'].close();DONE[tag]=dict(item['meta'],exit_code=code,completed_unix=time.time());del ACTIVE[tag]
        print('EXIT',tag,code,flush=True)
        if code:failures.append((tag,code))
    if failures:raise RuntimeError('child failure '+str(failures))


def stop_owned():
    for item in ACTIVE.values():
        if item['process'].poll() is None:os.killpg(item['process'].pid,signal.SIGTERM)
    for tag,item in list(ACTIVE.items()):
        try:code=item['process'].wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(item['process'].pid,signal.SIGKILL);code=item['process'].wait(timeout=5)
        item['stream'].close();DONE[tag]=dict(item['meta'],exit_code=code,completed_unix=time.time(),stopped_by_driver=True)
        del ACTIVE[tag]


def main():
    checked()
    if (OUT/'queue_status.json').exists():raise ValueError('inspect existing controller handle before any restart')
    if not read(OUT/'preflight.json')['passed']:raise ValueError('preflight required')
    INFO.update(pid=os.getpid(),started_unix=time.time(),lock_sha256=sha(OUT/'lock.json'))
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    jobs=[(NEW_ROLES[(j+i)%3],seed) for j in range(3) for i,seed in enumerate(SEEDS)]
    phases=(('smoke',[None]),('train',jobs),('training_audit',[None]),('export',[(s,) for s in SEEDS]),('metrics',[None]))
    try:
        for stage,items in phases:
            pending=list(items);previous=set();idle_started=None
            print('PHASE',stage,flush=True)
            while pending or ACTIVE:
                collect()
                if not pending and not ACTIVE:break
                if stage in ('training_audit','metrics'):
                    if pending:launch(stage,pending.pop(0),None)
                    record('running',stage=stage,pending=pending)
                elif not pending or len(ACTIVE)>=6:
                    # No launch can occur now. Avoid redundant NVML queries,
                    # especially while a CUDA worker is in driver exit cleanup.
                    previous=set();record('running',stage=stage,pending=pending)
                else:
                    try:rows=occupancy()
                    except (subprocess.SubprocessError,ValueError) as error:
                        previous=set();record('observation_error',stage=stage,error=repr(error));time.sleep(5);continue
                    available={r['uuid'] for r in rows if r['free']};stable=available&previous
                    own={x['meta']['gpu']['uuid'] for x in ACTIVE.values()}
                    for gpu in rows:
                        if not pending or len(ACTIVE)>=6:break
                        if gpu['uuid'] in stable-own:
                            launch(stage,pending.pop(0),gpu);own.add(gpu['uuid'])
                    previous=available-own
                    record('running' if ACTIVE else 'waiting_for_gpu',stage=stage,pending=pending,occupancy=rows)
                if pending and not ACTIVE:
                    if idle_started is None:idle_started=time.monotonic()
                    if time.monotonic()-idle_started>3600:raise TimeoutError('one hour without an available GPU')
                else:idle_started=None
                time.sleep(5)
        record('complete',all_children_exit_zero=True,goal_complete=False)
        print('R57 execution complete; development evidence only.',flush=True)
    except BaseException as error:
        stop_owned();record('stopped',error=repr(error),goal_complete=False);raise


if __name__=='__main__':main()
