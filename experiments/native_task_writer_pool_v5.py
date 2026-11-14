"""Same eighteen fits: exact resumes, local GPU yielding, explicit dormant storage."""
import fcntl,os,signal,subprocess,sys,time
from pathlib import Path
from experiments import native_task_writer_data as d
from experiments.native_task_writer_manage import gpu_rows,compute_pids
STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
def done(job):return all((d.OUT/'evaluation'/f'{job[0]}_{job[1]}_{m["id"]}.json').exists() for m in d.metas())
def laststep(job):
    p=d.OUT/'fits'/f'{job[0]}_{job[1]}'/'training.jsonl'
    return d.json.loads(p.read_text().splitlines()[-1])['step'] if p.exists() else 0

def main():
    with (d.OUT/'pool_v5.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[(a,s) for s in d.SEEDS for a in d.ARMS];active={};completed={j for j in jobs if done(j)};failed=[];yields=[];cooldown={};halt=False;attempt=time.time_ns();idle_since={};startup_retries={}
        assert all(x.get('actual_process_wait_returned') or x.get('actual_process_completion_verified') for x in [d.read(p) for p in (d.OUT/'workers').glob('*.json')]),'Never duplicate live prior workers'
        while True:
            for name,r in list(active.items()):
                code=r['child'].poll();info=r['info'];info.update(elapsed_seconds=time.monotonic()-r['start'],updated_utc=d.now(),exit_code=code)
                if code is not None:
                    info.update(exit_code=r['child'].wait(),ended_utc=d.now(),actual_process_wait_returned=True,actual_process_completion_verified=True);r['log'].close();del active[name]
                    if code==0 and done(r['job']):completed.add(r['job'])
                    elif code==75 and (d.latest(*r['job']).exists() or d.final(*r['job']).exists()):
                        y=dict(name=name,job=r['job'],exit_code=code,retained_step=laststep(r['job']),reason=info.get('stop_reason','worker_yield'));yields.append(y);d.journal(d.OUT/'engineering_yields.jsonl',dict(utc=d.now(),**y));cooldown[info['gpu_uuid']]=time.monotonic()+120
                    elif code==1 and 'MKL_THREADING_LAYER' in Path(info['log']).read_text() and 'model_loaded' not in Path(info['log']).read_text() and not (d.OUT/'calls'/f'{name}.jsonl').exists():
                        startup_retries[r['job']]=startup_retries.get(r['job'],0)+1
                        d.journal(d.OUT/'engineering_startup_failures.jsonl',dict(name=name,utc=d.now(),classification='pre_model_MKL_import_conflict',model_or_training_calls=0,all_wall_charged=True))
                        cooldown[info['gpu_uuid']]=time.monotonic()+60
                        if startup_retries[r['job']]>1:failed.append(dict(name=name,job=r['job'],exit_code=code));halt=True
                    else:failed.append(dict(name=name,job=r['job'],exit_code=code));halt=True
                d.write(d.OUT/'workers'/f'{name}.json',info)
            rr=d.resource();gpus=gpu_rows();apps=compute_pids()
            for g in gpus:
                truly_idle=g['used']<256 and g['util']<=2 and g['free']>=42*1024 and not apps.get(g['uuid'])
                if truly_idle:idle_since.setdefault(g['uuid'],time.monotonic())
                else:idle_since.pop(g['uuid'],None)
            global_reasons=rr['stop_reasons']+(['dispatcher_interrupted'] if STOP else [])+(['unexpected_worker_failure'] if halt else [])
            if global_reasons:halt=True
            pressured={g['uuid'] for g in gpus if g['free']<6144}
            for name,r in active.items():
                info=r['info'];g=next(g for g in gpus if g['uuid']==info['gpu_uuid']);d.journal(d.OUT/'gpu_resources'/f'{name}.jsonl',dict(utc=d.now(),**g,compute_pids=sorted(apps.get(g['uuid'],set()))))
                reasons=global_reasons or (['local_GPU_pressure'] if g['uuid'] in pressured else [])
                if reasons and not r.get('stop_requested'):
                    r['child'].send_signal(signal.SIGTERM);r['stop_requested']=d.now();info['stop_reason']=reasons;cooldown[g['uuid']]=time.monotonic()+120
            pending=[j for j in jobs if j not in completed and j not in [r['job'] for r in active.values()]]
            # Complete the most advanced existing unique fits before initializing more.
            pending.sort(key=lambda j:(not(d.latest(*j).exists() or d.final(*j).exists()),-laststep(j),jobs.index(j)))
            if not active and (not pending or halt):break
            admissions=[]
            if not halt:
                for g in sorted(gpus,key=lambda x:sum(r['info']['gpu_uuid']==x['uuid'] for r in active.values())):
                    owners={r['info']['pid'] for r in active.values() if r['info']['gpu_uuid']==g['uuid']};policy=d.read(d.OUT/'execution_policy.json');idle=not owners and g['uuid'] in idle_since and time.monotonic()-idle_since[g['uuid']]>=policy.get('stable_idle_seconds',60);shared=bool(owners) and len(owners)<policy.get('max_per_card',1) and apps.get(g['uuid'],set())<=owners and g['free']>=22*1024
                    if not pending or not(idle or shared) or time.monotonic()<cooldown.get(g['uuid'],0):continue
                    if len(active)+1>d.read(d.OUT/'execution_policy.json')['max_active']:continue
                    job=pending[0]
                    # Reserve every in-flight OR dormant full checkpoint, including
                    # brand-new jobs whose first full checkpoint has not yet formed.
                    reserved={j for j in jobs if j not in completed and (d.latest(*j).exists() or any(r['job']==j for r in active.values()))}
                    if not d.final(*job).exists():reserved.add(job)
                    final_only={j for j in jobs if j not in completed and d.final(*j).exists() and j not in reserved}
                    peak=2.69*(len(completed)+len(final_only))+7.15*len(reserved)+7.15+.10
                    admissions.append(dict(gpu=g['index'],prospective_job=job,completed=len(completed),final_only=len(final_only),active_and_dormant_full_states=len(reserved),peak_upper_GiB=peak))
                    if peak>80 or (len(active)+1)*35*2**30+30*2**30>rr['RAM_available_bytes']:continue
                    pending.pop(0);arm,seed=job;name=f'{arm}_{seed}_v5_{time.time_ns()}';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=g['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.DATA/'torch_extensions'));env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH'];logpath=d.OUT/'logs'/f'{name}.log';log=logpath.open('a',buffering=1);begin=time.monotonic();child=subprocess.Popen([sys.executable,'-u','-m','experiments.native_task_writer_run','fit','--worker',name,'--arm',arm,'--seed',str(seed)],cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    info=dict(name=name,phase='formal_pool_v5',arm=arm,seed=seed,pid=child.pid,manager_pid=os.getpid(),gpu_uuid=g['uuid'],GPU_index=g['index'],admission=g,started_utc=d.now(),elapsed_seconds=0.,exit_code=None,log=str(logpath),storage_peak_upper_GiB_at_admission=peak,reserved_full_states=[list(j) for j in sorted(reserved)],resume_step_before_launch=laststep(job))
                    active[name]=dict(info=info,job=job,child=child,log=log,start=begin);d.write(d.OUT/'workers'/f'{name}.json',info);print(dict(launched=name,gpu=g['index'],pid=child.pid,resume_step=laststep(job),storage_peak_upper_GiB=peak),flush=True)
            d.write(d.OUT/'queue_status.json',dict(updated_utc=d.now(),manager_pid=os.getpid(),phase='formal_pool_v5',active=[r['info'] for r in active.values()],completed=[list(j) for j in sorted(completed)],failed=failed,expected_engineering_yields=yields,pending=pending,resource=rr,admission_envelopes=admissions,storage_formula='(completed+final_only)2.69GiB+(active_or_dormant_full_states)7.15GiB+one_atomic7.15GiB+metadata0.10GiB',live_policy=d.read(d.OUT/'execution_policy.json'),local_GPU_yield_does_not_stop_other_cards=True,stable_idle_observation_seconds={k:time.monotonic()-v for k,v in idle_since.items()},explicit_worker_MKL_THREADING_LAYER='GNU'))
            time.sleep(10)
        result=dict(complete=not failed and not pending,completed=[list(j) for j in sorted(completed)],failed=failed,expected_engineering_yields=yields,all_GPU_process_completion_verified=True,all_children_waited=True,ended_utc=d.now());d.write(d.OUT/'formal_queue_complete.json',result);print(result,flush=True)
        if failed or pending:raise SystemExit(75)
if __name__=='__main__':main()
