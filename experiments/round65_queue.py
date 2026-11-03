"""Run the frozen experiment on observed idle GPUs; preserve external jobs."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from experiments.round65_common import ROOT,OUT,PREP,ROLES,SEEDS,checked,read,write,sha

ACTIVE,DONE,INFO={},{},{}


def occupancy():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=10)
    used={line.split(',')[0].strip() for line in apps.stdout.splitlines() if ',' in line}
    rows=[]
    for line in result.stdout.strip().splitlines():
        index,uuid,memory,util=[s.strip() for s in line.split(',')]
        rows.append(dict(index=int(index),uuid=uuid,memory_mib=int(memory),utilization=int(util),
                         free=uuid not in used and int(memory)<16 and int(util)==0))
    return rows


def record(state,**extra):
    write(OUT/'queue_status.json',dict(INFO,state=state,observed_unix=time.time(),
        completed=list(DONE.values()),active=[x['meta'] for x in ACTIVE.values()],**extra))


def launch(stage,job,gpu):
    tag=stage if job is None else stage+'_'+'_'.join(map(str,job))
    suffix=str(time.time_ns());key=tag+'_'+suffix
    if stage=='data':command=[sys.executable,'-m','experiments.round65_data']
    elif stage in ('selections','metrics'):command=[sys.executable,'-m','experiments.round65_audit',stage]
    else:
        command=[sys.executable,'-m','experiments.round65_run',stage]
        if stage=='train':command+=['--role',job[0],'--seed',str(job[1])]
        else:command+=['--seed',str(job[0])]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'])
    stream=(OUT/(key+'.log')).open('x')
    try:process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException:
        stream.close();raise
    meta=dict(tag=key,stage=stage,job=job,pid=process.pid,gpu=gpu,command=command,started_unix=time.time())
    ACTIVE[key]=dict(process=process,stream=stream,meta=meta)
    print('LAUNCHED',key,process.pid,None if gpu is None else gpu['index'],flush=True)


def collect():
    resume=[];failures=[]
    for key,item in list(ACTIVE.items()):
        code=item['process'].poll()
        if code is None:continue
        item['stream'].close();meta=dict(item['meta'],exit_code=code,completed_unix=time.time())
        DONE[key]=meta;del ACTIVE[key];print('EXIT',key,code,flush=True)
        if code==75 and meta['stage']=='train':resume.append(tuple(meta['job']))
        elif code:failures.append((key,code))
    if failures:raise RuntimeError('owned child failure '+str(failures))
    return resume


def stop_owned():
    for item in ACTIVE.values():
        if item['process'].poll() is None:os.killpg(item['process'].pid,signal.SIGTERM)
    for key,item in list(ACTIVE.items()):
        try:code=item['process'].wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(item['process'].pid,signal.SIGKILL);code=item['process'].wait(timeout=5)
        item['stream'].close();DONE[key]=dict(item['meta'],exit_code=code,stopped_by_driver=True)
        del ACTIVE[key]


def wait_for_acquisition():
    expected=read(OUT/'execution.json')['acquisition']
    marker=PREP/'acquisition_complete.json'
    while not marker.exists():
        proc=Path('/proc')/str(expected['pid'])
        try:
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
            ticks=(proc/'stat').read_text().split()[21]
        except FileNotFoundError:
            if marker.exists():break
            raise RuntimeError('acquisition handle is terminal/missing before completion; inspect original acquisition')
        if command!=expected['command'] or ticks!=expected['start_ticks']:
            raise RuntimeError('acquisition process identity changed; inspect before attaching a new handle')
        attempt=Path(expected['attempt']);progress=attempt/'progress.json'
        record('waiting_for_acquisition',acquisition_live_verified=True,
               acquisition_progress=read(progress) if progress.exists() else None)
        time.sleep(10)
    if read(marker)['catalog_sha256']!=sha(PREP/'train_catalog.json'):
        raise ValueError('acquired population mismatch')


def main():
    checked()
    if (OUT/'queue_status.json').exists():raise ValueError('inspect the existing controller before restart')
    if not read(OUT/'units.json')['passed'] or not read(OUT/'cuda_preflight.json')['passed']:
        raise ValueError('implementation preflights required')
    INFO.update(pid=os.getpid(),started_unix=time.time(),lock_sha256=sha(OUT/'lock.json'))
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    jobs=[(ROLES[(j+i)%len(ROLES)],seed) for j in range(len(ROLES)) for i,seed in enumerate(SEEDS)]
    phases=(('data',[None]),('train',jobs),('selections',[None]),
            ('export',[(seed,) for seed in SEEDS]),('metrics',[None]))
    try:
        wait_for_acquisition()
        for stage,pending in phases:
            pending,previous=list(pending),set();print('PHASE',stage,flush=True)
            while pending or ACTIVE:
                pending.extend(collect())
                if not pending and not ACTIVE:break
                if stage in ('data','metrics'):
                    if pending:launch(stage,pending.pop(0),None)
                    record('running',stage=stage,pending=pending)
                elif not pending or len(ACTIVE)>=6:
                    previous=set();record('running',stage=stage,pending=pending)
                else:
                    try:rows=occupancy()
                    except (subprocess.SubprocessError,ValueError) as error:
                        previous=set();record('observation_error',stage=stage,error=str(error));time.sleep(5);continue
                    available={r['uuid'] for r in rows if r['free']};stable=available&previous
                    owned={item['meta']['gpu']['uuid'] for item in ACTIVE.values()}
                    for gpu in rows:
                        if not pending or len(ACTIVE)>=6:break
                        if gpu['uuid'] in stable-owned:
                            launch(stage,pending.pop(0),gpu);owned.add(gpu['uuid'])
                    previous=available-owned
                    record('running' if ACTIVE else 'waiting_for_gpu',stage=stage,pending=pending,occupancy=rows)
                time.sleep(5)
        record('complete',task_execution_completed=True,scientific_review_pending=True,goal_complete=False)
        print('R65 execution complete; report and scientific/visual review remain.',flush=True)
    except BaseException as error:
        stop_owned();record('stopped',error_type=type(error).__name__,error=str(error),goal_complete=False);raise


if __name__=='__main__':main()
