"""Unique bounded native-retention queue; dynamically use genuinely idle cards."""
import os,sys,time,signal,argparse,subprocess,fcntl,shutil
from experiments import stream_retention_data as d
from experiments.stream_history_pool import cards
STOP=False
def stop(signum,*_):
 global STOP;STOP=True
 d.write(d.OUT/'manager_signal.json',dict(PID=os.getpid(),utc=d.now(),signal=signum,sender_unknown=True))
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','formal']);a=p.parse_args();signal.signal(signal.SIGTERM,stop);start=time.monotonic();d.DATA.mkdir(exist_ok=True);d.OUT.mkdir(exist_ok=True)
 with (d.OUT/'pool.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);previous=[d.read(x) for x in (d.OUT/'workers').glob('*.json')];assert all(x.get('actual_wait_returned') for x in previous)
  if a.phase=='formal':assert d.read(d.OUT/'preflight.json')['complete'] and d.read(d.OUT/'byte_ceiling.json')['before_any_T_D_scientific_effect_scoring']
  jobs=[('W','fidelity_00')] if a.phase=='profile' else [(policy,m['id']) for m in d.plan()['catalog'] for policy in ('T','D')];active=[];idle={};failures=[];halt=None;peakdata=d.size(d.DATA);peakproj=d.size(d.OUT);peakVRAM=0
  while True:
   live=[]
   for v in active:
    w=v['record'];w['elapsed_seconds']=time.monotonic()-v['start'];code=v['child'].poll()
    if code is not None:
     w.update(exit_code=v['child'].wait(),actual_wait_returned=True,ended_utc=d.now());v['log'].close()
     if code not in (0,75):failures.append(w);halt='worker_failure'
    else:live.append(v)
    d.write(d.OUT/'workers'/f"{w['name']}.json",w)
   active=live;records=[d.read(x) for x in (d.OUT/'workers').glob('*.json')];seconds=sum(w['elapsed_seconds'] for w in records);profile_seconds=sum(w['elapsed_seconds'] for w in records if w['phase']=='profile');databytes=d.size(d.DATA);projbytes=d.size(d.OUT);peakdata=max(peakdata,databytes);peakproj=max(peakproj,projbytes);rows=cards();available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024
   if STOP:halt='manager_SIGTERM'
   if seconds>=7200-45:halt='total_worker_budget_reserve'
   if a.phase=='profile' and profile_seconds>=900-20:halt='preflight_cost_budget_reserve'
   if databytes>8*2**30 or projbytes>2**30:halt='storage_cap'
   own={v['record']['PID'] for v in active};busy={v['record']['GPU_UUID'] for v in active}
   for v in active:
    w=v['record'];g=next(g for g in rows if g['uuid']==w['GPU_UUID']);foreign=[x['PID'] for x in g['apps'] if x['PID'] not in own];used=max([x['used_MiB'] for x in g['apps'] if x['PID']==w['PID']]+[0]);peakVRAM=max(peakVRAM,used)
    if used>40960:halt='single_worker_VRAM_cap'
    reason=halt or ('external_PID_arrived' if foreign else None)
    if foreign:idle[g['uuid']]=time.monotonic()+90
    if reason and not v.get('signalled'):w['stop_reason']=reason;v['child'].send_signal(signal.SIGTERM);v['signalled']=True
   pending=[(p,r) for p,r in jobs if not (d.OUT/'completed'/f'{p}_{r}.json').exists() and not any(v['record']['policy']==p and v['record']['id']==r for v in active)]
   if not halt:
    for g in rows:
     if g['uuid'] in busy:continue
     safe=not g['apps'] and g['used_MiB']<256 and g['util']<=2 and g['free_MiB']>42*1024
     if not safe:idle[g['uuid']]=time.monotonic();continue
     idle.setdefault(g['uuid'],time.monotonic()-30 if g['index']==4 else time.monotonic())
     if time.monotonic()-idle[g['uuid']]<30:continue
     if not pending or len(active)>=(1 if a.phase=='profile' else 4):break
     if available<8*2**30*(len(active)+1) or shutil.disk_usage(d.DATA).free<8*2**30 or shutil.disk_usage(d.OUT).free<256*2**20:continue
     if databytes+(len(active)+1)*2**30>7*2**30:continue
     policy,rid=pending.pop(0);name=f'{a.phase}_{policy}_{rid}_{time.time_ns()}';env=dict(os.environ,CUDA_VISIBLE_DEVICES=g['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PATH='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+os.environ['PATH'],CUDA_HOME='/usr/local/cuda-12.4')
     path=d.OUT/'logs'/f'{name}.log';path.parent.mkdir(exist_ok=True);log=path.open('a');t=time.monotonic();child=subprocess.Popen([sys.executable,'-u','-m','experiments.stream_retention_run',a.phase,'--policy',policy,'--rid',rid,'--name',name],cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);w=dict(name=name,phase=a.phase,policy=policy,id=rid,PID=child.pid,GPU_index=g['index'],GPU_UUID=g['uuid'],started_utc=d.now(),elapsed_seconds=0,exit_code=None,actual_wait_returned=False,admission=g);active.append(dict(child=child,record=w,start=t,log=log));d.write(d.OUT/'workers'/f'{name}.json',w);print(dict(launched=name,PID=child.pid,GPU=g['index']),flush=True)
   d.write(d.OUT/'queue_status.json',dict(utc=d.now(),manager_PID=os.getpid(),phase=a.phase,active=[v['record'] for v in active],pending=[dict(policy=p,id=r) for p,r in pending],failures=failures,halt_reason=halt,GPU_worker_seconds=seconds,preflight_cost_seconds=profile_seconds,DATA_bytes=databytes,DATA_peak_sampled_bytes=peakdata,project_bytes=projbytes,project_peak_sampled_bytes=peakproj,peak_NVML_worker_MiB=peakVRAM,RAM_available_bytes=available,cards=rows,concurrency_scope='All genuinely idle cards, max4simultaneous checkpoint exits from8GiB DATA bound (not default2card cap); one independent stream per idlecard. Busy foreign compute is protected.'))
   if not active and (halt or not pending):break
   time.sleep(.5)
  d.write(d.OUT/f'pool_{a.phase}_complete.json',dict(complete=not halt and all((d.OUT/'completed'/f'{p}_{r}.json').exists() for p,r in jobs),manager_PID=os.getpid(),actual_all_children_waited=True,halt_reason=halt,failures=failures,ended_utc=d.now(),wall_seconds=time.monotonic()-start,DATA_peak_sampled_bytes=peakdata,project_peak_sampled_bytes=peakproj,peak_NVML_worker_MiB=peakVRAM))
  if halt:raise SystemExit(75)
if __name__=='__main__':main()
