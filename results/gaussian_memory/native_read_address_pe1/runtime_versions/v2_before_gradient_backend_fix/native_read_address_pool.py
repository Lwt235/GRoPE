"""One sequential scientific stage with dynamic safe-idle unique GPU jobs."""
import fcntl,os,signal,subprocess,sys,time
from experiments import native_read_address_data as d
from experiments.native_task_writer_manage import gpu_rows,compute_pids
STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def done(job):
    phase,arm,seed=job
    if phase=='prepare':return (d.OUT/'preparation'/f'complete_{seed}.json').exists()
    if phase=='preflight':return (d.OUT/'preflight.json').exists() and d.read(d.OUT/'preflight.json')['complete']
    return all((d.OUT/'evaluation'/f'{arm}_{seed}_{m["id"]}.json').exists() for m in d.metas())
def main():
    d.initialize()
    with (d.OUT/'pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert all(r.get('actual_process_wait_returned') for r in [d.read(p) for p in (d.OUT/'workers').glob('*.json')]),'Never duplicate live workers'
        active={};idle_since={};cooldown={};failures=[];yields=[];halt=False
        while True:
            for name,r in list(active.items()):
                code=r['child'].poll();info=r['info'];info.update(elapsed_seconds=time.monotonic()-r['start'],updated_utc=d.now(),exit_code=code)
                if code is not None:
                    info.update(exit_code=r['child'].wait(),actual_process_wait_returned=True,ended_utc=d.now());r['log'].close();del active[name]
                    if code==75:yields.append(dict(name=name,job=r['job'],reason=info.get('stop_reason','safe_unit_exit')));cooldown[info['gpu_uuid']]=time.monotonic()+120
                    elif code!=0 or not done(r['job']):failures.append(info);halt=True
                d.write(d.OUT/'workers'/f'{name}.json',info)
            resource=d.resource();rows=gpu_rows();apps=compute_pids()
            for g in rows:
                if g['used']<256 and g['util']<=2 and g['free']>=42*1024 and not apps.get(g['uuid']):idle_since.setdefault(g['uuid'],time.monotonic())
                else:idle_since.pop(g['uuid'],None)
            global_reasons=resource['stop_reasons']+(['dispatcher_interrupted'] if STOP else [])+(['worker_failure'] if halt else [])
            if global_reasons:halt=True
            for name,r in active.items():
                info=r['info'];g=next(g for g in rows if g['uuid']==info['gpu_uuid']);d.journal(d.OUT/'gpu_resources'/f'{name}.jsonl',dict(utc=d.now(),**g,compute_pids=sorted(apps.get(g['uuid'],set()))))
                reasons=global_reasons or (['local_GPU_pressure'] if g['free']<6144 else [])
                if reasons and not r.get('stop_requested'):r['child'].send_signal(signal.SIGTERM);r['stop_requested']=True;info['stop_reason']=reasons
            prepares=[('prepare',None,s) for s in d.SEEDS]
            if not all(done(j) for j in prepares):jobs=prepares;phase='preparation'
            elif not done(('preflight',None,d.SEEDS[0])):jobs=[('preflight',None,d.SEEDS[0])];phase='preflight'
            else:jobs=[('fit',a,s) for s in d.SEEDS for a in d.ARMS];phase='formal'
            pending=[j for j in jobs if not done(j) and j not in [r['job'] for r in active.values()]]
            if not halt:
                for g in rows:
                    if not pending or g['uuid'] not in idle_since or time.monotonic()-idle_since[g['uuid']]<60 or time.monotonic()<cooldown.get(g['uuid'],0):continue
                    if any(r['info']['gpu_uuid']==g['uuid'] for r in active.values()):continue
                    if (len(active)+1)*35*2**30+30*2**30>resource['RAM_available_bytes']:continue
                    job=pending.pop(0);task,arm,seed=job;name=f'{task}_{arm}_{seed}_{time.time_ns()}';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=g['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.DATA/'torch_extensions'));env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH'];cmd=[sys.executable,'-u','-m','experiments.native_read_address_run',task,'--worker',name,'--seed',str(seed)]
                    if arm:cmd+=['--arm',arm]
                    path=d.OUT/'logs'/f'{name}.log';log=path.open('a',buffering=1);begin=time.monotonic();child=subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);info=dict(name=name,phase=task,arm=arm,seed=seed,pid=child.pid,gpu_uuid=g['uuid'],GPU_index=g['index'],started_utc=d.now(),elapsed_seconds=0.,exit_code=None,log=str(path),admission=g)
                    active[name]=dict(child=child,job=job,info=info,log=log,start=begin);d.write(d.OUT/'workers'/f'{name}.json',info);print(dict(launched=name,gpu=g['index'],pid=child.pid),flush=True)
            d.write(d.OUT/'queue_status.json',dict(updated_utc=d.now(),manager_pid=os.getpid(),phase=phase,active=[r['info'] for r in active.values()],completed=[j for j in jobs if done(j)],pending=pending,failed=failures,expected_yields=yields,resource=resource,stable_idle_seconds={k:time.monotonic()-v for k,v in idle_since.items()},admission_policy='All live true-idle cards after60s; one unique job/card and35GiB CPU reserve/job; no inherited two-card cap. Local pressure yields own worker only.'))
            if not active and (halt or (phase=='formal' and not pending)):break
            time.sleep(10)
        complete=not halt and not failures and phase=='formal' and not pending;d.write(d.OUT/'formal_queue_complete.json',dict(complete=complete,all_children_actually_waited=True,failures=failures,yields=yields,ended_utc=d.now()));print(dict(complete=complete,all_children_actually_waited=True),flush=True)
        if not complete:raise SystemExit(75)
if __name__=='__main__':main()
