"""Unique round049 owner; S-first, all live safe cards, actual child waits."""
import os,sys,time,fcntl,subprocess,signal,shutil
from experiments import stac_budget_data as d
from experiments.stream_history_pool import cards
STOP=False
def stop(*_):
 global STOP;STOP=True
def main():
 signal.signal(signal.SIGTERM,stop);p=d.prepare();started=time.monotonic();active=[];idle={};halt=None;fail=[];peakdata=0;peakproject=0;first=p['catalog'][0]['id']
 auxiliary=d.read(d.OUT/'auxiliary_costs.json');aux=auxiliary['GPU_worker_seconds_upper_bound']
 with (d.OUT/'pool.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert all(d.read(x)['actual_wait_returned'] for x in (d.OUT/'workers').glob('*.json'))
  while True:
   live=[]
   for job in active:
    w=job['w'];w['elapsed_seconds']=time.monotonic()-job['start'];code=job['child'].poll()
    if code is not None:
     w.update(exit_code=job['child'].wait(),actual_wait_returned=True,ended_utc=d.now());job['log'].close()
     if code not in (0,75):fail.append(w);halt='worker_failure'
     if code==75 and not (d.DATA/'resume'/f"{w['arm']}_{w['id']}"/'manifest.json').exists():halt='missing_exact_checkpoint'
    else:live.append(job)
    d.write(d.OUT/'workers'/f"{w['name']}.json",w)
   active=live;workers=[d.read(x) for x in (d.OUT/'workers').glob('*.json')];total=aux+sum(w['elapsed_seconds'] for w in workers);profile=aux+sum(w['elapsed_seconds'] for w in workers if w['phase']=='preflight');data=d.size(d.DATA);project=d.size(d.OUT);peakdata=max(peakdata,data);peakproject=max(peakproject,project);rows=cards();available=int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024
   admitted=(d.OUT/'preflight_admission.json').exists()
   if not admitted and (d.OUT/'completed'/f'S_{first}.json').exists() and not active:
    s=sum(w['elapsed_seconds'] for w in workers);projected=aux+s*36*1.3
    admit=profile<3600 and projected<28800
    d.write(d.OUT/'preflight_admission.json',dict(admitted=admit,before_GT_scoring=True,profile_worker_seconds_upper_bound=profile,projection_seconds=projected,first_S_output_reused=True))
    if not admit:halt='first_fixed_stream_cost_gap'
    admitted=True
   admitted=admitted and d.read(d.OUT/'preflight_admission.json')['admitted']
   sdone=all((d.OUT/'completed'/f"S_{m['id']}.json").exists() for m in p['catalog'])
   jobs=[('S',m['id']) for m in p['catalog']] if admitted else [('S',first)]
   if sdone and not active and not (d.OUT/'capacity_lock.json').exists():
    # Exact capacity lock is performed by the separately versioned deterministic byte calculation, before any GT scoring.
    from experiments.stac_budget_capacity import lock_capacity
    lock_capacity();print(dict(capacity_locked=True),flush=True)
   if (d.OUT/'capacity_lock.json').exists():jobs=[(a,m['id']) for m in p['catalog'] for a in ('H','W')]
   pending=[(a,r) for a,r in jobs if not (d.OUT/'completed'/f'{a}_{r}.json').exists() and not any(v['w']['arm']==a and v['w']['id']==r for v in active)]
   if STOP:halt='requested_own_stop'
   if total+max(1,len(active))*120>=28800:halt='GPU_worker_budget_reserve'
   if not admitted and profile+max(1,len(active))*120>=3600:halt='preflight_budget_reserve'
   if data>48*2**30 or project>2*2**30:halt='storage_cap'
   own={v['w']['PID'] for v in active}
   for v in active:
    w=v['w'];g=next(x for x in rows if x['uuid']==w['GPU_UUID']);foreign=[x for x in g['apps'] if x['PID'] not in own];used=max([x['used_MiB'] for x in g['apps'] if x['PID']==w['PID']]+[0]);reason=halt or ('foreign_PID_arrived' if foreign else None)
    if used>40960:halt='worker_VRAM_cap';reason=halt
    if foreign:idle[g['uuid']]=time.monotonic()+60
    if reason and not v.get('signalled'):v['child'].send_signal(signal.SIGTERM);v['signalled']=True;w['stop_reason']=reason
   if not halt:
    for g in rows:
     if any(v['w']['GPU_UUID']==g['uuid'] for v in active):continue
     safe=not g['apps'] and g['used_MiB']<256 and g['util']<=2 and g['free_MiB']>=42*1024
     if not safe:idle[g['uuid']]=time.monotonic();continue
     idle.setdefault(g['uuid'],time.monotonic()-30)
     if time.monotonic()-idle[g['uuid']]<30 or not pending:continue
     # Worst exact paused-state backing reservation + fixed permanent outputs; no default card-count cap.
     reserve=(len(active)+1)*12*2**30
     if data+reserve>48*2**30 or shutil.disk_usage(d.DATA).free<reserve or available<(len(active)+1)*24*2**30:continue
     arm,rid=pending.pop(0);phase='preflight' if arm=='S' and rid==first else 'formal';name=f'{phase}_{arm}_{rid}_{time.time_ns()}';env=dict(os.environ,CUDA_VISIBLE_DEVICES=g['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PATH='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+os.environ['PATH'],CUDA_HOME='/usr/local/cuda-12.4',TORCH_CUDA_ARCH_LIST='8.6');path=d.OUT/'logs'/f'{name}.log';path.parent.mkdir(exist_ok=True);log=path.open('a');t=time.monotonic();child=subprocess.Popen([sys.executable,'-u','-m','experiments.stac_budget_run','--arm',arm,'--rid',rid,'--name',name,'--phase',phase],cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);w=dict(name=name,arm=arm,id=rid,phase=phase,PID=child.pid,GPU_index=g['index'],GPU_UUID=g['uuid'],started_utc=d.now(),elapsed_seconds=0,actual_wait_returned=False,exit_code=None,admission=g);active.append(dict(child=child,w=w,start=t,log=log));d.write(d.OUT/'workers'/f'{name}.json',w);print(dict(launched=name,PID=child.pid,GPU=g['index']),flush=True)
   d.write(d.OUT/'queue_status.json',dict(utc=d.now(),manager_PID=os.getpid(),active=[v['w'] for v in active],pending=[dict(arm=a,id=r) for a,r in pending],failed=fail,halt_reason=halt,GPU_worker_seconds_upper_bound=total,preflight_GPU_worker_seconds_upper_bound=profile,DATA_bytes=data,DATA_peak_sampled_bytes=peakdata,project_bytes=project,project_peak_sampled_bytes=peakproject,RAM_available_bytes=available,cards=rows,S_all_complete=sdone,GT_scoring_started=False))
   if not active and (halt or not pending):break
   time.sleep(1)
  d.write(d.OUT/'pool_complete.json',dict(complete=not halt,all_children_actual_waited=True,manager_PID=os.getpid(),halt_reason=halt,failed=fail,GPU_worker_seconds_upper_bound=total,preflight_GPU_worker_seconds_upper_bound=profile,wall_seconds=time.monotonic()-started,ended_utc=d.now()))
  if halt:raise SystemExit(75)
if __name__=='__main__':main()
