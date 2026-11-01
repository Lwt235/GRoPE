"""Run the frozen R44 stages on unoccupied GPUs, with real child exit records.

This controller does not change scientific settings or stop other users' jobs.
It waits at most one hour with no active child before yielding a queue timeout.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round44_spatial/attempt1'
ACTIVE={}
DONE=[]
INFO={}


def record(state,**extra):
    value=dict(INFO,state=state,observed_unix=time.time(),completed=DONE,
        active=[item['meta'] for item in ACTIVE.values()],**extra)
    tmp=OUT/'queue_status.tmp'
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(OUT/'queue_status.json')


def occupancy():
    args=['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']
    rows=subprocess.run(args,check=True,capture_output=True,text=True,timeout=10).stdout.strip().splitlines()
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=10).stdout.strip().splitlines()
    used={line.split(',')[0].strip() for line in apps if ',' in line}
    result=[]
    for line in rows:
        index,uuid,memory,util=[v.strip() for v in line.split(',')]
        result.append(dict(index=int(index),uuid=uuid,memory_mib=int(memory),utilization=int(util),
            free=uuid not in used and int(memory)<16 and int(util)==0))
    return result


def launch(stage,seed,gpu):
    tag=stage if seed is None else f'{stage}_{seed}'
    if stage in ('training_audit','metrics'):
        cmd=[sys.executable,'-m','experiments.round44_spatial_audit','training' if stage=='training_audit' else 'metrics']
    else:
        cmd=[sys.executable,'-m','experiments.round44_spatial_train',stage]
        if seed is not None:cmd+=['--seeds',str(seed)]
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if gpu is not None:env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu['uuid'])
    stream=(OUT/(tag+'.log')).open('x')
    try:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException:
        stream.close();raise
    meta=dict(tag=tag,stage=stage,seed=seed,pid=child.pid,process_group=child.pid,
        gpu=gpu,command=cmd,started_unix=time.time())
    ACTIVE[tag]=dict(process=child,stream=stream,meta=meta)
    print('LAUNCHED',tag,'pid',child.pid,'gpu',None if gpu is None else gpu['index'],flush=True)


def collect():
    failures=[]
    for tag,item in list(ACTIVE.items()):
        code=item['process'].poll()
        if code is None:continue
        item['stream'].close()
        DONE.append(dict(item['meta'],exit_code=code,completed_unix=time.time()))
        del ACTIVE[tag]
        print('EXIT',tag,code,flush=True)
        if code:failures.append((tag,code))
    if failures:raise RuntimeError('child failure '+str(failures))


def terminate_own_children():
    for item in ACTIVE.values():
        if item['process'].poll() is None:
            os.killpg(item['process'].pid,signal.SIGTERM)
    for tag,item in list(ACTIVE.items()):
        try:code=item['process'].wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(item['process'].pid,signal.SIGKILL);code=item['process'].wait(timeout=5)
        item['stream'].close()
        DONE.append(dict(item['meta'],exit_code=code,stopped_by_driver=True,completed_unix=time.time()))
        del ACTIVE[tag]


def main():
    if (OUT/'queue_status.json').exists():raise ValueError('existing queue; inspect its live handle before any restart')
    if not json.loads((OUT/'preflight.json').read_text())['passed']:raise ValueError('CPU checks required')
    source=Path(__file__).read_bytes()
    (OUT/'queue_source.py').write_bytes(source)
    INFO.update(pid=os.getpid(),started_unix=time.time(),source_sha256=hashlib.sha256(source).hexdigest(),
        lock_sha256=(OUT/'lock.sha256').read_text().strip())
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    phases=(('smoke',[None]),('train',[11,22,33,44,55]),('training_audit',[None]),
            ('export',[11,22,33,44,55]),('metrics',[None]))
    try:
        for stage,seeds in phases:
            pending=list(seeds);previous=set();idle_started=None
            print('PHASE',stage,flush=True)
            while pending or ACTIVE:
                collect()
                if not pending and not ACTIVE:break
                if stage in ('training_audit','metrics'):
                    if pending:launch(stage,pending.pop(0),None)
                    record('running',stage=stage,pending=pending)
                else:
                    try:rows=occupancy()
                    except (subprocess.SubprocessError,ValueError) as error:
                        previous=set();record('observation_error',stage=stage,pending=pending,error=repr(error))
                        time.sleep(5);continue
                    available={r['uuid'] for r in rows if r['free']}
                    stable=available&previous
                    own={item['meta']['gpu']['uuid'] for item in ACTIVE.values()}
                    for gpu in rows:
                        if not pending or len(ACTIVE)>=5:break
                        if gpu['uuid'] in stable-own:
                            launch(stage,pending.pop(0),gpu);own.add(gpu['uuid'])
                    previous=available-own
                    record('running' if ACTIVE else 'waiting_for_gpu',stage=stage,pending=pending,occupancy=rows)
                if pending and not ACTIVE:
                    if idle_started is None:idle_started=time.monotonic()
                    if time.monotonic()-idle_started>3600:
                        record('queue_timeout',stage=stage,pending=pending)
                        raise SystemExit(75)
                else:idle_started=None
                time.sleep(5)
        record('complete',all_children_exit_zero=True,goal_complete=False)
        print('R44 execution complete; inspect scientific results before any claim.',flush=True)
    except BaseException as error:
        terminate_own_children()
        record('stopped',error=repr(error),goal_complete=False)
        raise


if __name__=='__main__':main()
