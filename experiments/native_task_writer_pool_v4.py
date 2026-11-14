"""Same eighteen fits: exact resumes, local GPU yielding, explicit dormant storage."""
import datetime,fcntl,os,signal,subprocess,sys,time
from experiments.native_task_writer_pool_v2 import proc_state
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
    with (d.OUT/'pool_v4.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[(a,s) for s in d.SEEDS for a in d.ARMS];active={};completed={j for j in jobs if done(j)};failed=[];yields=[];cooldown={};halt=False;attempt=time.time_ns();idle_since={};supervisor_retired=False
        handoff=d.read(d.OUT/'dispatcher_v4_adoption_input.json');oldpid=handoff['old_manager_pid'];assert proc_state(oldpid)[0]=='T';yields=list(handoff.get('expected_engineering_yields',[]))
        for info in handoff['active']:
            info=dict(info,adopted_live_PID=True,original_dispatcher_pid=oldpid,adopted_by=os.getpid(),adopted_utc=d.now());job=(info['arm'],info['seed']);begin=datetime.datetime.fromisoformat(info['started_utc']).timestamp()
            active[info['name']]=dict(info=info,job=job,adopted=True,start_epoch=begin)
        assert {r['info']['pid'] for r in active.values()}==set(map(int,open(f'/proc/{oldpid}/task/{oldpid}/children').read().split())), 'Adopt every original live child exactly once'
        while True:
            for name,r in list(active.items()):
                info=r['info'];code=None
                if r.get('adopted'):
                    ps,status=proc_state(info['pid'])
                    if ps=='Z':code=os.waitstatus_to_exitcode(status);info.update(actual_process_wait_returned=False,actual_process_completion_verified=True,observed_host_wait_status=status,termination_proof='Stopped original dispatcher retains zombie; host /proc stat final wait status captured')
                    elif ps is None:raise RuntimeError(('Adopted PID vanished before final wait status',info['pid']))
                    elapsed=time.time()-r['start_epoch']
                else:
                    code=r['child'].poll();elapsed=time.monotonic()-r['start']
                    if code is not None:code=r['child'].wait();info.update(actual_process_wait_returned=True,actual_process_completion_verified=True)
                info.update(elapsed_seconds=elapsed,updated_utc=d.now(),exit_code=code)
                if code is not None:
                    info.update(exit_code=code,ended_utc=d.now())
                    if not r.get('adopted'):r['log'].close()
                    del active[name]
                    if code==0 and done(r['job']):completed.add(r['job'])
                    elif code==75 and (d.latest(*r['job']).exists() or d.final(*r['job']).exists()):
                        y=dict(name=name,job=r['job'],exit_code=code,retained_step=laststep(r['job']),reason=info.get('stop_reason','worker_yield'));yields.append(y);d.journal(d.OUT/'engineering_yields.jsonl',dict(utc=d.now(),**y));cooldown[info['gpu_uuid']]=time.monotonic()+120
                    else:failed.append(dict(name=name,job=r['job'],exit_code=code));halt=True
                d.write(d.OUT/'workers'/f'{name}.json',info)
            if not supervisor_retired and not any(r.get('adopted') for r in active.values()):
                assert proc_state(oldpid)[0]=='T';os.kill(oldpid,signal.SIGKILL);supervisor_retired=True
                d.write(d.OUT/f'dispatcher_retirement_{oldpid}.json',dict(utc=d.now(),old_dispatcher_pid=oldpid,all_its_GPU_children_already_exited=True,only_own_stopped_CPU_dispatcher_killed=True))
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
                    os.kill(info['pid'],signal.SIGTERM);r['stop_requested']=d.now();info['stop_reason']=reasons;cooldown[g['uuid']]=time.monotonic()+120
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
                    pending.pop(0);arm,seed=job;name=f'{arm}_{seed}_v4_{time.time_ns()}';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=g['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.DATA/'torch_extensions'));env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH'];logpath=d.OUT/'logs'/f'{name}.log';log=logpath.open('a',buffering=1);begin=time.monotonic();child=subprocess.Popen([sys.executable,'-u','-m','experiments.native_task_writer_run','fit','--worker',name,'--arm',arm,'--seed',str(seed)],cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    info=dict(name=name,phase='formal_pool_v4',arm=arm,seed=seed,pid=child.pid,manager_pid=os.getpid(),gpu_uuid=g['uuid'],GPU_index=g['index'],admission=g,started_utc=d.now(),elapsed_seconds=0.,exit_code=None,log=str(logpath),storage_peak_upper_GiB_at_admission=peak,reserved_full_states=[list(j) for j in sorted(reserved)],resume_step_before_launch=laststep(job))
                    active[name]=dict(info=info,job=job,child=child,log=log,start=begin);d.write(d.OUT/'workers'/f'{name}.json',info);print(dict(launched=name,gpu=g['index'],pid=child.pid,resume_step=laststep(job),storage_peak_upper_GiB=peak),flush=True)
            d.write(d.OUT/'queue_status.json',dict(updated_utc=d.now(),manager_pid=os.getpid(),phase='formal_pool_v4',active=[r['info'] for r in active.values()],completed=[list(j) for j in sorted(completed)],failed=failed,expected_engineering_yields=yields,pending=pending,resource=rr,admission_envelopes=admissions,storage_formula='(completed+final_only)2.69GiB+(active_or_dormant_full_states)7.15GiB+one_atomic7.15GiB+metadata0.10GiB',live_policy=d.read(d.OUT/'execution_policy.json'),local_GPU_yield_does_not_stop_other_cards=True,stable_idle_observation_seconds={k:time.monotonic()-v for k,v in idle_since.items()},retired_original_supervisor=supervisor_retired))
            time.sleep(10)
        result=dict(complete=not failed and not pending,completed=[list(j) for j in sorted(completed)],failed=failed,expected_engineering_yields=yields,all_GPU_process_completion_verified=True,adopted_GPU_exit_wait_status_captured=True,old_dispatcher_retired=supervisor_retired,all_direct_children_waited=True,ended_utc=d.now());d.write(d.OUT/'formal_queue_complete.json',result);print(result,flush=True)
        if failed or pending:raise SystemExit(75)
if __name__=='__main__':main()
