"""Sole native-response queue: measured first stream, actual waits, safe idle cards."""
import os,sys,time,signal,fcntl,argparse,subprocess,shutil
from experiments import stream_response_data as d
from experiments.stream_history_pool import cards
STOP=False
def stop(*_):
 global STOP;STOP=True
 d.write(d.OUT/'manager_signal.json',dict(PID=os.getpid(),utc=d.now(),signal=_[0],sender_unknown=True))
def main():
 signal.signal(signal.SIGTERM,stop);start=time.monotonic();d.prepare();d.DATA.mkdir(exist_ok=True);p=d.plan();cap=24*2**30;reserve_full=p['checkpoint_full_payload_bound_bytes'];reserve_prefix=p['checkpoint_prefix31_payload_bound_bytes'];active=[];idle={};fail=[];halt=None;peakdata=d.size(d.DATA);peakproj=d.size(d.OUT);peakNVML=0
 with (d.OUT/'pool.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);previous=[d.read(x) for x in (d.OUT/'workers').glob('*.json')];assert all(x.get('actual_wait_returned') for x in previous)
  while True:
   live=[]
   for v in active:
    w=v['w'];w['elapsed_seconds']=time.monotonic()-v['start'];code=v['child'].poll()
    if code is not None:
     w.update(exit_code=v['child'].wait(),actual_wait_returned=True,ended_utc=d.now());v['log'].close()
     if code not in (0,75):fail.append(w);halt='worker_failure'
     if code==75 and not (d.DATA/'resume'/w['id']/'manifest.json').exists():halt='missing_recovery_state'
    else:live.append(v)
    d.write(d.OUT/'workers'/f"{w['name']}.json",w)
   active=live;workers=[d.read(x) for x in (d.OUT/'workers').glob('*.json')];seconds=sum(x['elapsed_seconds'] for x in workers);data=d.size(d.DATA);proj=d.size(d.OUT);peakdata=max(peakdata,data);peakproj=max(peakproj,proj);rows=cards();available=int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024
   first=p['catalog'][0]['id'];admitted=(d.OUT/'profiling_admission.json').exists() and d.read(d.OUT/'profiling_admission.json')['admitted_before_scoring']
   if not admitted and (d.OUT/'completed'/f'{first}.json').exists() and not active:
    ws=sum(x['elapsed_seconds'] for x in workers if x['id']==first);projected=seconds+11*ws*1.25;admitted=projected<14400 and peakNVML<40960;d.write(d.OUT/'profiling_admission.json',dict(admitted_before_scoring=admitted,registered_utc=d.now(),first_fixed_stream_seconds=ws,projected_total_worker_seconds=projected,first_outputs_reused=True,scientific_scores_not_read=True,peak_NVML_worker_MiB=peakNVML,DATA_peak_sampled_bytes=peakdata));
    if not admitted:halt='first_stream_cost_gap'
   if STOP:halt='requested_own_stop'
   if seconds+max(1,len(active))*180>=14400:halt='total_worker_budget_reserve'
   if data>cap or proj>2**30:halt='storage_cap'
   own={v['w']['PID'] for v in active};busy={v['w']['GPU_UUID'] for v in active}
   for v in active:
    w=v['w'];g=next(x for x in rows if x['uuid']==w['GPU_UUID']);foreign=[x for x in g['apps'] if x['PID'] not in own];used=max([x['used_MiB'] for x in g['apps'] if x['PID']==w['PID']]+[0]);peakNVML=max(peakNVML,used);reason=halt or ('external_PID_arrived' if foreign else None)
    if used>40960:reason='worker_VRAM_cap';halt=reason
    if foreign:idle[g['uuid']]=time.monotonic()+90
    if reason and not v.get('signalled'):w['stop_reason']=reason;v['child'].send_signal(signal.SIGTERM);v['signalled']=True
   pending=[x['id'] for x in p['catalog'] if not (d.OUT/'completed'/f"{x['id']}.json").exists() and not any(v['w']['id']==x['id'] for v in active)]
   if not admitted:pending=[r for r in pending if r==first]
   # Ready exact prefixes precede new streams; no cached teacher frames are repeated.
   pending.sort(key=lambda r:(not (d.DATA/'resume'/r/'manifest.json').exists(),[x['id'] for x in p['catalog']].index(r)))
   if not halt:
    for g in rows:
     if g['uuid'] in busy:continue
     safe=not g['apps'] and g['used_MiB']<256 and g['util']<=2 and g['free_MiB']>42*1024
     if not safe:idle[g['uuid']]=time.monotonic();continue
     idle.setdefault(g['uuid'],time.monotonic())
     if time.monotonic()-idle[g['uuid']]<30:continue
     if not pending or (not admitted and active):continue
     if available<32*2**30*(len(active)+1) or shutil.disk_usage(d.OUT).free<256*2**20:continue
     rid=pending[0];existing=d.size(d.DATA/'resume'/rid);otherres=sum(v['w']['checkpoint_reserve_bytes'] for v in active);future=500*2**20*(len(active)+1);budget=data-existing+otherres+reserve_full+future+128*2**20;until=95;reserve=reserve_full
     if budget>cap:
      # A third card may form a useful exact prefix while two full checkpoints fit.
      if existing or (d.OUT/'progress'/f'{rid}.json').exists():continue
      reserve=reserve_prefix;until=31;budget=data+otherres+reserve+future+128*2**20
     if budget>cap or shutil.disk_usage(d.DATA).free<reserve+future:continue
     if until==31 and any(v['w']['until']==31 for v in active):continue
     pending.pop(0);name=f'F_{rid}_{time.time_ns()}';env=dict(os.environ,CUDA_VISIBLE_DEVICES=g['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PATH='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+os.environ['PATH'],CUDA_HOME='/usr/local/cuda-12.4');path=d.OUT/'logs'/f'{name}.log';path.parent.mkdir(exist_ok=True);log=path.open('a');s=time.monotonic();child=subprocess.Popen([sys.executable,'-u','-m','experiments.stream_response_run','--rid',rid,'--name',name,'--until',str(until)],cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);w=dict(name=name,id=rid,PID=child.pid,GPU_index=g['index'],GPU_UUID=g['uuid'],started_utc=d.now(),elapsed_seconds=0,exit_code=None,actual_wait_returned=False,admission=g,until=until,checkpoint_reserve_bytes=reserve,projected_concurrent_storage_bound_bytes=budget);active.append(dict(child=child,w=w,start=s,log=log));busy.add(g['uuid']);d.write(d.OUT/'workers'/f'{name}.json',w);print(dict(launched=name,PID=child.pid,GPU=g['index'],until=until),flush=True)
   d.write(d.OUT/'queue_status.json',dict(utc=d.now(),manager_PID=os.getpid(),active=[v['w'] for v in active],pending=pending,failures=fail,halt_reason=halt,GPU_worker_seconds=seconds,DATA_bytes=data,DATA_peak_sampled_bytes=peakdata,project_peak_sampled_bytes=peakproj,peak_NVML_worker_MiB=peakNVML,RAM_available_bytes=available,cards=rows,cost_admitted=admitted,concurrency='Allstableidlecardswithfullcheckpoint/capture/permanentpeak;exactprefix31allowedifthirdfullstatewouldexceed24GiB. No inherited2worker cap.'))
   if not active and (halt or not pending):break
   time.sleep(.7)
  d.write(d.OUT/'pool_complete.json',dict(complete=not halt and all((d.OUT/'completed'/f"{x['id']}.json").exists() for x in p['catalog']),manager_PID=os.getpid(),actual_all_children_waited=True,halt_reason=halt,failures=fail,wall_seconds=time.monotonic()-start,GPU_worker_seconds=seconds,DATA_peak_sampled_bytes=peakdata,project_peak_sampled_bytes=peakproj,peak_NVML_worker_MiB=peakNVML,ended_utc=d.now()))
  if halt:raise SystemExit(75)
if __name__=='__main__':main()
