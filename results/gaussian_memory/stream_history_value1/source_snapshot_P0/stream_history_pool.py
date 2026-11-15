"""One owner, fixed stream jobs; live host admission and full worker-time accounting."""
import os,sys,time,fcntl,subprocess,signal,argparse,shutil
from experiments import stream_history_data as d
STOP=False
def stop(*_):
 global STOP;STOP=True
def cards():
 apps={};rows=[]
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
  u,p,m=[x.strip() for x in line.split(',')];apps.setdefault(u,[]).append(dict(PID=int(p),used_MiB=int(m)))
 for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
  i,u,m,f,v=[x.strip() for x in line.split(',')];rows.append(dict(index=int(i),uuid=u,used_MiB=int(m),free_MiB=int(f),util=int(v),apps=apps.get(u,[])))
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','formal']);a=p.parse_args();start=time.monotonic();signal.signal(signal.SIGTERM,stop)
 d.OUT.mkdir(exist_ok=True);d.DATA.mkdir(exist_ok=True);(d.OUT/'workers').mkdir(exist_ok=True);(d.OUT/'logs').mkdir(exist_ok=True)
 with (d.OUT/'pool.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);old=[d.read(p) for p in (d.OUT/'workers').glob('*.json')];assert all(w.get('actual_wait_returned') for w in old),'Inspect existing child before resume'
  plan=d.plan();jobs=[plan['profiling_stream']] if a.phase=='profile' else [m['id'] for m in plan['catalog']];active=None;idle={};failures=[];peak_DATA=d.size(d.DATA);peak_project=d.size(d.OUT);peak_NVML=0;halt_reason=None
  while True:
   if active:
    w=active['record'];w['elapsed_seconds']=time.monotonic()-active['start'];code=active['child'].poll()
    if code is not None:
     w.update(exit_code=active['child'].wait(),actual_wait_returned=True,ended_utc=d.now());active['log'].close();d.write(d.OUT/'workers'/f"{w['name']}.json",w)
     if code not in (0,75):failures.append(w);halt_reason='worker_failure'
     active=None
    else:d.write(d.OUT/'workers'/f"{w['name']}.json",w)
   workers=[d.read(p) for p in (d.OUT/'workers').glob('*.json')];seconds=sum(w['elapsed_seconds'] for w in workers);profile_seconds=sum(w['elapsed_seconds'] for w in workers if w['phase']=='profile');data=d.size(d.DATA);project=d.size(d.OUT);peak_DATA=max(peak_DATA,data);peak_project=max(peak_project,project)
   available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024;rows=cards()
   if STOP:halt_reason='requested_own_stop'
   if seconds>=4*3600-60:halt_reason='GPU_total_budget_reserve'
   if a.phase=='profile' and profile_seconds>=1800-45:halt_reason='profiling_budget_reserve'
   if data>16*2**30 or project>2**30:halt_reason='storage_cap'
   pending=[rid for rid in jobs if not (d.OUT/'completed'/f'{rid}.json').exists() and (not active or active['record']['id']!=rid)]
   if active:
    w=active['record'];g=next(g for g in rows if g['uuid']==w['GPU_UUID']);foreign=[x['PID'] for x in g['apps'] if x['PID']!=w['PID']];used=max([x['used_MiB'] for x in g['apps'] if x['PID']==w['PID']]+[0]);peak_NVML=max(peak_NVML,used)
    if used>40*1024:halt_reason='single_worker_VRAM_cap'
    reason=halt_reason or ('external_PID_arrived' if foreign else None)
    if reason and not active.get('signalled'):w['stop_reason']=reason;active['child'].send_signal(signal.SIGTERM);active['signalled']=True
   elif not halt_reason and pending:
    for g in rows:
     safe=not g['apps'] and g['used_MiB']<256 and g['util']<=2 and g['free_MiB']>42*1024
     if not safe:idle[g['uuid']]=time.monotonic();continue
     idle.setdefault(g['uuid'],time.monotonic()-30) # startup snapshot independently checked; still inspect live PID here
     if time.monotonic()-idle[g['uuid']]<30 or available<32*2**30:continue
     if shutil.disk_usage(d.DATA).free<16*2**30 or shutil.disk_usage(d.OUT).free<256*2**20:continue
     rid=pending.pop(0);name=f"{a.phase}_{rid}_{time.time_ns()}";env=dict(os.environ,CUDA_VISIBLE_DEVICES=g['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PATH='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+os.environ['PATH'],CUDA_HOME='/usr/local/cuda-12.4',TORCH_EXTENSIONS_DIR='/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/torch_extensions',TORCH_CUDA_ARCH_LIST='8.6')
     log=(d.OUT/'logs'/f'{name}.log').open('a');t=time.monotonic();child=subprocess.Popen([sys.executable,'-u','-m','experiments.stream_history_run',a.phase,'--rid',rid,'--name',name],cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);w=dict(name=name,phase=a.phase,id=rid,PID=child.pid,GPU_index=g['index'],GPU_UUID=g['uuid'],started_utc=d.now(),elapsed_seconds=0,exit_code=None,actual_wait_returned=False,admission=g);active=dict(child=child,record=w,start=t,log=log);d.write(d.OUT/'workers'/f'{name}.json',w);print(dict(launched=name,PID=child.pid,GPU=g['index']),flush=True);break
   d.write(d.OUT/'queue_status.json',dict(utc=d.now(),manager_PID=os.getpid(),phase=a.phase,active=active['record'] if active else None,pending=pending,failures=failures,halt_reason=halt_reason,GPU_worker_seconds=seconds,profiling_seconds=profile_seconds,DATA_bytes=data,DATA_peak_sampled_bytes=peak_DATA,project_bytes=project,project_peak_sampled_bytes=peak_project,peak_NVML_worker_MiB=peak_NVML,RAM_available_bytes=available,cards=rows,concurrency_reason='One stream checkpoint can retain about11GiB of exact raw KV; concurrent recovery peaks plus permanent predictions must fit total16GiB. This is a storage admission choice, not a two-card default.',CPU_wait_not_GPU_worker_time=True))
   if not active and (halt_reason or not pending):break
   time.sleep(1)
  d.write(d.OUT/f'pool_{a.phase}_complete.json',dict(complete=not halt_reason and all((d.OUT/'completed'/f'{r}.json').exists() for r in jobs),halt_reason=halt_reason,failures=failures,actual_children_all_waited=True,manager_PID=os.getpid(),wall_seconds=time.monotonic()-start,ended_utc=d.now(),DATA_peak_sampled_bytes=peak_DATA,project_peak_sampled_bytes=peak_project,peak_NVML_worker_MiB=peak_NVML))
  if halt_reason:raise SystemExit(75)
if __name__=='__main__':main()
