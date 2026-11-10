"""Bounded, resumeless controller for the frozen unused-scene comparison."""
import os
import signal
import subprocess
import sys
import time
from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.round65_queue import occupancy
from experiments.region_hrope_pose_queue import app_pids
from experiments.region_hrope_confirmation import OUT, check

MAIN='experiments.region_hrope_confirmation'
SCORE='experiments.region_hrope_confirmation_score'


def memory_free():
    raw=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
    return {a.strip():int(b.strip()) for a,b in (line.split(',') for line in raw.stdout.splitlines())}


def main():
    check()
    assert not (OUT/'queue_status.json').exists(),'Inspect existing controller first.'
    active,completed,pending=[],[],[]
    stage='starting'
    identity=dict(pid=os.getpid(),started_unix=time.time(),source_sha256=sha(__file__),
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),maximum_gpu_workers=4)

    def record(state,**extra):
        write(OUT/'queue_status.json',dict(identity,stage=stage,state=state,pending=pending,
            active=[r[1] for r in active],completed=completed,observed_unix=time.time(),goal_complete=False,**extra))

    def batch(name,jobs,gpu=False,share_owned=False):
        nonlocal stage,pending
        stage,pending=name,list(jobs)
        previous=set();previous_spare=set()
        while pending or active:
            for process,item,stream in list(active):
                code=process.poll()
                if code is not None:
                    stream.close();active.remove((process,item,stream))
                    completed.append(dict(item,exit_code=code,finished_unix=time.time()))
                    print('CONFIRMATION_CHILD_EXIT',name,item.get('seed'),code,flush=True)
            failures=[r for r in completed if r['exit_code']]
            if failures:
                record('finishing_active_after_failure',failures=failures)
                if not active:raise RuntimeError('Child failure preserved; no pending jobs launched.')
                time.sleep(3);continue
            if not pending and not active:break
            owned={}
            for _,item,_ in active:
                if item['gpu']:owned.setdefault(item['gpu']['uuid'],set()).add(item['pid'])
            if gpu:
                rows,apps,space=occupancy(),app_pids(),memory_free()
                free={r['uuid'] for r in rows if r['free']}
                available=[r for r in rows if r['uuid'] in free & previous]
                if share_owned:
                    available += [r for r in rows if r['uuid'] in owned and
                        apps.get(r['uuid'],set())<=owned[r['uuid']] and r['memory_mib']<8192]
                spare={r['uuid'] for r in rows if space[r['uuid']]>=12288 and r['utilization']<=60}
                external_already_shared=any(apps.get(uuid,set())-pids for uuid,pids in owned.items())
                if not available and not external_already_shared:
                    candidates=[dict(r,shared_existing_compute=True,memory_free_mib=space[r['uuid']])
                        for r in rows if r['uuid'] in spare & previous_spare and r['uuid'] not in owned]
                    available=sorted(candidates,key=lambda r:(r['utilization'],-r['memory_free_mib']))[:1]
                previous=free
                previous_spare=spare
            else:available=[None]
            if pending and len(active)<(4 if gpu else 1) and available:
                device=available[0]
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                    CUDA_VISIBLE_DEVICES=device['uuid'] if device else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
                probe=None
                if device:
                    test=subprocess.run([sys.executable,'-c',
                        'import numpy;import torch;assert torch.cuda.is_available();assert torch.cuda.device_count()==1;print((torch.ones(1,device="cuda")+1).item())'],
                        env=env,capture_output=True,text=True,timeout=45)
                    assert test.returncode==0,test.stderr
                    probe=dict(exit_code=test.returncode,stdout=test.stdout.strip())
                    if device.get('shared_existing_compute'):
                        current=next(r for r in occupancy() if r['uuid']==device['uuid'])
                        ready=memory_free()[device['uuid']]>=12288 and current['utilization']<=60
                    else:
                        ready=app_pids().get(device['uuid'],set())<=owned.get(device['uuid'],set())
                    if not ready:
                        record('gpu_changed_before_launch');time.sleep(3);continue
                job=pending.pop(0)
                command=[sys.executable,'-u','-m',job['module']]
                if job.get('action'):command.append(job['action'])
                if job.get('seed') is not None:command+=['--seed',str(job['seed'])]
                log=OUT/f"{name}_{job.get('seed')}_{time.time_ns()}.log"
                stream=log.open('x')
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                item=dict(job,pid=process.pid,gpu=device,cuda_probe=probe,log=str(log),started_unix=time.time(),
                    sharing_with_owned_pids=sorted(owned.get(device['uuid'],set())) if device else [])
                active.append((process,item,stream))
                print('CONFIRMATION_LAUNCHED',item,flush=True)
            record('running' if active else 'waiting_for_idle_gpu')
            time.sleep(3)
    try:
        batch('preflight',[dict(module=MAIN,action='preflight')],gpu=True)
        batch('download',[dict(module=MAIN,action='download')])
        batch('prepare',[dict(module=MAIN,action='prepare')])
        batch('inference',[dict(module=MAIN,action='infer_all')],gpu=True)
        batch('bind_inputs',[dict(module=SCORE,action='bind_inputs')])
        batch('pose',[dict(module=SCORE,action='run',seed=s) for s in SEEDS]+
              [dict(module=SCORE,action='diagnostics')],gpu=True,share_owned=True)
        batch('independent_summary',[dict(module=SCORE,action='summarize')])
        batch('tables_and_figure',[dict(module='experiments.region_hrope_confirmation_report')])
        stage='awaiting_scientific_review';record('complete',scientific_review_pending=True)
    except BaseException as error:
        for process,item,stream in active:
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        record('stopped',error_type=type(error).__name__,reason=str(error));raise


if __name__=='__main__':main()
