"""Live-PID dispatcher adoption and completed/active storage envelopes; no GPU restart."""
import datetime,fcntl,os,signal,subprocess,sys,time
from pathlib import Path
from experiments import native_task_writer_data as d
from experiments.native_task_writer_manage import gpu_rows,compute_pids

def proc_state(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists():return None,None
    fields=p.read_text().split(') ',1)[1].split();return fields[0],int(fields[49]) if len(fields)>49 else None

def done(arm,seed):return all((d.OUT/'evaluation'/f'{arm}_{seed}_{m["id"]}.json').exists() for m in d.metas())

def main():
    with (d.OUT/'pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);handoff=d.read(d.OUT/'dispatcher_adoption_input.json');oldpid=handoff['old_manager_pid'];assert proc_state(oldpid)[0]=='T'
        jobs=[(a,s) for s in d.SEEDS for a in d.ARMS];active={};completed=set();failed=[];stop=False;attempt=time.time_ns();old_names=set();supervisor_retired=False
        for info in handoff['queue']['active']:
            name=info['name'];pid=info['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
            if proc_state(pid)[0]!='Z':assert 'experiments.native_task_writer_run fit' in cmd and '--worker '+name in cmd
            info=dict(info,adopted_live_PID=True,original_dispatcher_pid=oldpid,adopted_by=os.getpid(),adopted_utc=d.now());old_names.add(name)
            begin=datetime.datetime.fromisoformat(info['started_utc']).timestamp();active[name]=dict(info=info,job=(info['arm'],info['seed']),adopted=True,start_epoch=begin)
        for job in jobs:
            if job not in [r['job'] for r in active.values()] and done(*job):completed.add(job)
        while True:
            for name,r in list(active.items()):
                info=r['info'];state,waitstatus=proc_state(info['pid']);code=None;exited=False
                if r['adopted']:
                    if state=='Z':
                        code=os.waitstatus_to_exitcode(waitstatus);exited=True;info.update(actual_process_wait_returned=False,actual_process_completion_verified=True,termination_proof='Stopped original dispatcher retains zombie; host /proc stat exposes final wait status',observed_host_wait_status=waitstatus)
                    elif state is None:raise RuntimeError(('Adopted PID vanished before exit-status capture',info['pid']))
                else:
                    code=r['child'].poll();exited=code is not None
                    if exited:code=r['child'].wait();info.update(actual_process_wait_returned=True,actual_process_completion_verified=True)
                info.update(elapsed_seconds=time.time()-r['start_epoch'],updated_utc=d.now(),exit_code=code)
                if exited:
                    info['ended_utc']=d.now();del active[name]
                    if not r['adopted']:r['log'].close()
                    if code==0 and done(*r['job']):completed.add(r['job'])
                    else:failed.append(dict(name=name,job=r['job'],exit_code=code));stop=True
                d.write(d.OUT/'workers'/f'{name}.json',info)
            if not supervisor_retired and not any(n in active for n in old_names):
                # All former children have final exit-status proofs; no live GPU
                # process is signalled or belongs to the retired dispatcher now.
                cmd=Path(f'/proc/{oldpid}/cmdline').read_bytes().decode().replace('\0',' ');assert 'experiments.native_task_writer_manage formal' in cmd
                os.kill(oldpid,signal.SIGKILL);supervisor_retired=True;d.write(d.OUT/'dispatcher_retirement.json',dict(utc=d.now(),old_dispatcher_pid=oldpid,only_own_stopped_CPU_dispatcher_killed=True,all_adopted_GPU_children_already_exited=True,adopted_exit_status_records=[str(d.OUT/'workers'/f'{n}.json') for n in sorted(old_names)]))
            rr=d.resource();gpus=gpu_rows();apps=compute_pids()
            for name,r in active.items():
                g=next(g for g in gpus if g['uuid']==r['info']['gpu_uuid']);d.journal(d.OUT/'gpu_resources'/f'{name}.jsonl',dict(utc=d.now(),**g))
                if (rr['stop_reasons'] or g['free']<6144 or stop) and not r.get('stop_requested'):
                    os.kill(r['info']['pid'],signal.SIGTERM);r['stop_requested']=d.now();r['info']['stop_reason']=rr['stop_reasons'] or ['GPU_pressure_or_sibling_failure'];stop=True
            pending=[j for j in jobs if j not in completed and j not in [r['job'] for r in active.values()]]
            if not active and (not pending or stop):break
            admission_rows=[]
            if not stop and not rr['stop_reasons']:
                for g in sorted(gpus,key=lambda x:sum(r['info']['gpu_uuid']==x['uuid'] for r in active.values())):
                    owners={r['info']['pid'] for r in active.values() if r['info']['gpu_uuid']==g['uuid']};idle=not owners and g['used']<256 and g['util']<=2 and g['free']>=42*1024;shared=bool(owners) and len(owners)<2 and apps.get(g['uuid'],set())<=owners and g['free']>=22*1024
                    if not pending or not (idle or shared):continue
                    n=len(active)+1;f=len(completed);peak=(2.69*f+7.15*n+7.15+.10)*2**30
                    admission_rows.append(dict(gpu=g['index'],completed=f,prospective_active=n,peak_upper_GiB=peak/2**30))
                    if peak>80*2**30:continue
                    if len(active)*35*2**30+30*2**30>rr['RAM_available_bytes']:continue
                    arm,seed=pending.pop(0);name=f'{arm}_{seed}_pool_{attempt}';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=g['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.DATA/'torch_extensions'));env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH'];logpath=d.OUT/'logs'/f'{name}.log';log=logpath.open('a',buffering=1);begin=time.time();child=subprocess.Popen([sys.executable,'-u','-m','experiments.native_task_writer_run','fit','--worker',name,'--arm',arm,'--seed',str(seed)],cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    info=dict(name=name,phase='formal_pool',arm=arm,seed=seed,pid=child.pid,manager_pid=os.getpid(),gpu_uuid=g['uuid'],GPU_index=g['index'],admission=g,started_utc=d.now(),elapsed_seconds=0.,exit_code=None,log=str(logpath),storage_peak_upper_GiB_at_admission=peak/2**30)
                    active[name]=dict(info=info,job=(arm,seed),adopted=False,start_epoch=begin,child=child,log=log);d.write(d.OUT/'workers'/f'{name}.json',info);print(dict(launched=name,gpu=g['index'],pid=child.pid,storage_peak_upper_GiB=peak/2**30),flush=True)
            d.write(d.OUT/'queue_status.json',dict(updated_utc=d.now(),manager_pid=os.getpid(),phase='formal_dynamic_pool',active=[r['info'] for r in active.values()],completed=sorted([list(j) for j in completed],key=str),failed=failed,pending=pending,resource=rr,admission_envelopes=admission_rows,storage_formula='completed2.69GiB+active7.15GiB+one_atomic7.15GiB+metadata0.10GiB',original_GPU_workers_not_paused_or_restarted=True))
            time.sleep(10)
        result=dict(complete=not failed and not pending,completed=[list(j) for j in completed],failed=failed,all_GPU_process_completion_verified=True,adopted_GPU_exit_wait_status_captured=True,old_dispatcher_retired=supervisor_retired,ended_utc=d.now());d.write(d.OUT/'formal_queue_complete.json',result);print(result,flush=True)
        if failed or pending:raise SystemExit(75)
if __name__=='__main__':main()
