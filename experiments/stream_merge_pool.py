"""One owner, fixed36streams plus one Wcost, actualwait/live all-card admission."""
import os,sys,time,signal,fcntl,subprocess,shutil
from experiments import stream_merge_data as d
from experiments.stream_history_pool import cards
STOP=False
def stop(*_):
 global STOP;STOP=True
 d.write(d.OUT/'manager_signal.json',dict(PID=os.getpid(),utc=d.now(),signal=_[0],sender_unknown=True))
def main():
 signal.signal(signal.SIGTERM,stop);start=time.monotonic();p=d.prepare();d.DATA.mkdir(exist_ok=True);active=[];idle={};fail=[];halt=None;peakdata=d.size(d.DATA);peakproj=d.size(d.OUT);peakNVML=0;first=p['catalog'][0]['id'];profiles=[(a,first) for a in ['C','I','G','W']];formal=[(a,m['id']) for m in p['catalog'][1:] for a in ['C','I','G']]
 with (d.OUT/'pool.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert all(d.read(x)['actual_wait_returned'] for x in (d.OUT/'workers').glob('*.json'))
  while True:
   live=[]
   for v in active:
    w=v['w'];w['elapsed_seconds']=time.monotonic()-v['start'];code=v['child'].poll()
    if code is not None:
     w.update(exit_code=v['child'].wait(),actual_wait_returned=True,ended_utc=d.now());v['log'].close()
     if code not in (0,75):fail.append(w);halt='worker_failure'
     if code==75 and not (d.DATA/'resume'/f"{w['policy']}_{w['id']}"/'manifest.json').exists():halt='missing_exact_checkpoint'
    else:live.append(v)
    d.write(d.OUT/'workers'/f"{w['name']}.json",w)
   active=live;workers=[d.read(x) for x in (d.OUT/'workers').glob('*.json')];seconds=sum(x['elapsed_seconds'] for x in workers);profile=sum(x['elapsed_seconds'] for x in workers if x['phase']=='profile');data=d.size(d.DATA);proj=d.size(d.OUT);peakdata=max(peakdata,data);peakproj=max(peakproj,proj);rows=cards();available=int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024;admitted=(d.OUT/'profiling_admission.json').exists() and d.read(d.OUT/'profiling_admission.json')['admitted_before_scoring']
   if not admitted and all((d.OUT/'completed'/f'{a}_{r}.json').exists() for a,r in profiles) and not active:
    times={a:sum(w['elapsed_seconds'] for w in workers if w['phase']=='profile' and w['policy']==a) for a,_ in profiles};projected=seconds+11*sum(times[a] for a in ['C','I','G'])*1.2;admitted=profile<3600 and projected<28800;d.write(d.OUT/'profiling_admission.json',dict(admitted_before_scoring=admitted,registered_utc=d.now(),first_fixed_stream_seconds=times,profile_worker_seconds=profile,projected_total_worker_seconds=projected,first_C_I_G_outputs_reused=True,W_fixed_cost_replay_once=True,scientific_scores_not_read=True,peak_NVML_worker_MiB=peakNVML));
    if not admitted:halt='fixed_first_stream_cost_gap'
   if STOP:halt='requested_own_stop'
   if seconds+max(len(active),1)*180>28800:halt='total_worker_budget_reserve'
   if not admitted and profile+max(len(active),1)*120>3600:halt='preflight_worker_budget_reserve'
   if data>32*2**30 or proj>2**30:halt='storage_cap'
   own={v['w']['PID'] for v in active};settings=d.read(d.OUT/'resource_settings.json') if (d.OUT/'resource_settings.json').exists() else {};cohost=max(1,min(2,int(settings.get('max_own_flows_per_GPU',1))))
   for v in active:
    w=v['w'];g=next(x for x in rows if x['uuid']==w['GPU_UUID']);foreign=[x for x in g['apps'] if x['PID'] not in own];used=max([x['used_MiB'] for x in g['apps'] if x['PID']==w['PID']]+[0]);peakNVML=max(peakNVML,used);reason=halt or ('external_PID_arrived' if foreign else None)
    if used>40960:halt='VRAM_cap';reason=halt
    if foreign:idle[g['uuid']]=time.monotonic()+90
    if reason and not v.get('signalled'):v['child'].send_signal(signal.SIGTERM);v['signalled']=True;w['stop_reason']=reason
   jobs=profiles+formal if admitted else profiles;pending=[(a,r) for a,r in jobs if not (d.OUT/'completed'/f'{a}_{r}.json').exists() and not any(v['w']['policy']==a and v['w']['id']==r for v in active)];initialproof=(d.OUT/'progress'/f'C_{first}.json').exists() and d.read(d.OUT/'progress'/f'C_{first}.json')['frame']>=5
   if not initialproof:pending=[(a,r) for a,r in pending if a=='C' and r==first]
   if not halt:
    for g in rows:
     hosted=[v for v in active if v['w']['GPU_UUID']==g['uuid']]
     if len(hosted)>=cohost:continue
     foreign=[x for x in g['apps'] if x['PID'] not in own]
     owned_safe=bool(hosted) and not foreign and g['free_MiB']>14*1024 and not any(v.get('signalled') for v in hosted)
     safe=owned_safe or (not hosted and not g['apps'] and g['used_MiB']<256 and g['free_MiB']>42*1024 and g['util']<=2)
     if not safe:idle[g['uuid']]=time.monotonic();continue
     idle.setdefault(g['uuid'],time.monotonic())
     if not owned_safe and time.monotonic()-idle[g['uuid']]<30:continue
     if not pending:continue
     reserve=(len(active)+1)*(p['state_copy_bound_bytes']+256*2**20)+4*2**30
     if data+reserve>32*2**30 or shutil.disk_usage(d.DATA).free<reserve or available<16*2**30*(len(active)+1):continue
     policy,rid=pending.pop(0);phase='profile' if (policy,rid) in profiles else 'formal';name=f'{phase}_{policy}_{rid}_{time.time_ns()}';env=dict(os.environ,CUDA_VISIBLE_DEVICES=g['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PATH='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+os.environ['PATH'],CUDA_HOME='/usr/local/cuda-12.4');path=d.OUT/'logs'/f'{name}.log';path.parent.mkdir(exist_ok=True);log=path.open('a');s=time.monotonic();child=subprocess.Popen([sys.executable,'-u','-m','experiments.stream_merge_run','--policy',policy,'--rid',rid,'--name',name,'--phase',phase],cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);w=dict(name=name,policy=policy,id=rid,phase=phase,PID=child.pid,GPU_index=g['index'],GPU_UUID=g['uuid'],started_utc=d.now(),elapsed_seconds=0,exit_code=None,actual_wait_returned=False,admission=g);active.append(dict(child=child,w=w,start=s,log=log));own.add(child.pid);d.write(d.OUT/'workers'/f'{name}.json',w);print(dict(launched=name,PID=child.pid,GPU=g['index']),flush=True)
   d.write(d.OUT/'queue_status.json',dict(utc=d.now(),manager_PID=os.getpid(),active=[v['w'] for v in active],pending=[dict(policy=a,id=r) for a,r in pending],failures=fail,halt_reason=halt,GPU_worker_seconds=seconds,profiling_worker_seconds=profile,DATA_bytes=data,DATA_peak_sampled_bytes=peakdata,project_peak_sampled_bytes=peakproj,peak_NVML_worker_MiB=peakNVML,RAM_available_bytes=available,cards=rows,cost_admitted=admitted,initial_C_interface_complete=initialproof,resource_settings=settings,parallelism='Allstablegenuinelyidlecards;atmost2ownflows/card,12GiBreservationeachand14GiBfreebeforecohost;foreignPIDyieldsallownflows;dynamicsetting1drainswithoutinterrupt.32GiBDATA/hostRAM/fullrecoverypeak.'))
   if not active and (halt or not pending):break
   time.sleep(.7)
  d.write(d.OUT/'pool_complete.json',dict(complete=not halt and all((d.OUT/'completed'/f'{a}_{r}.json').exists() for a,r in profiles+formal),manager_PID=os.getpid(),actual_all_children_waited=True,halt_reason=halt,failures=fail,wall_seconds=time.monotonic()-start,GPU_worker_seconds=seconds,profile_worker_seconds=profile,DATA_peak_sampled_bytes=peakdata,project_peak_sampled_bytes=peakproj,peak_NVML_worker_MiB=peakNVML,ended_utc=d.now()))
  if halt:raise SystemExit(75)
if __name__=='__main__':main()
