"""New unique round036 owner; fixed jobs, actual waits, all-card resource guard."""
import os,sys,time,signal,subprocess,fcntl,shutil,argparse
from experiments import stream_factorization_data as d
from experiments.stream_history_pool import cards
STOP=False
def stop(signum,*_):
 global STOP;STOP=True
 d.write(d.OUT/'manager_signal.json',dict(PID=os.getpid(),utc=d.now(),signal=signum,sender_unknown=True))
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','formal']);a=p.parse_args();signal.signal(signal.SIGTERM,stop);start=time.monotonic();d.prepare();d.DATA.mkdir(exist_ok=True)
 with (d.OUT/'pool.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);old=[d.read(x) for x in (d.OUT/'workers').glob('*.json')];assert all(x['actual_wait_returned'] for x in old)
  if a.phase=='formal':assert d.read(d.OUT/'profiling_admission.json')['admitted_before_scoring']
  jobs=[m['id'] for m in d.plan()['catalog']][:1 if a.phase=='profile' else 12];active=None;idle={};fail=[];halt=None;peakdata=d.size(d.DATA);peakproject=d.size(d.OUT);peakNVML=0
  while True:
   if active:
    w=active['w'];w['elapsed_seconds']=time.monotonic()-active['start'];code=active['child'].poll()
    if code is not None:
     w.update(exit_code=active['child'].wait(),actual_wait_returned=True,ended_utc=d.now());active['log'].close()
     if code not in (0,75):halt='worker_failure';fail.append(w)
     d.write(d.OUT/'workers'/f"{w['name']}.json",w);active=None
    else:d.write(d.OUT/'workers'/f"{w['name']}.json",w)
   workers=[d.read(x) for x in (d.OUT/'workers').glob('*.json')];seconds=sum(w['elapsed_seconds'] for w in workers);profiles=sum(w['elapsed_seconds'] for w in workers if w['phase']=='profile');databytes=d.size(d.DATA);projbytes=d.size(d.OUT);peakdata=max(peakdata,databytes);peakproject=max(peakproject,projbytes);rows=cards();available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024
   if STOP:halt='manager_SIGTERM'
   if seconds>=14400-120:halt='total_budget_reserve'
   if a.phase=='profile' and profiles>=1800-90:halt='profile_budget_reserve'
   if databytes>32*2**30 or projbytes>2**30:halt='storage_cap'
   if active:
    w=active['w'];g=next(g for g in rows if g['uuid']==w['GPU_UUID']);foreign=[x for x in g['apps'] if x['PID']!=w['PID']];used=max([x['used_MiB'] for x in g['apps'] if x['PID']==w['PID']]+[0]);peakNVML=max(peakNVML,used)
    if used>40960:halt='worker_VRAM_cap'
    reason=halt or ('external_PID_arrived' if foreign else None)
    if foreign:idle[g['uuid']]=time.monotonic()+120
    if reason and not active.get('signalled'):w['stop_reason']=reason;active['child'].send_signal(signal.SIGTERM);active['signalled']=True
   pending=[r for r in jobs if not (d.OUT/'completed'/f'{r}.json').exists() and (not active or active['w']['id']!=r)]
   if not active and not halt and pending:
    resumable=[r for r in pending if (d.DATA/'resume'/r/'manifest.json').exists()];rid=resumable[0] if resumable else pending[0]
    for g in rows:
     safe=not g['apps'] and g['used_MiB']<256 and g['util']<=2 and g['free_MiB']>24*1024
     if not safe:idle[g['uuid']]=time.monotonic();continue
     idle.setdefault(g['uuid'],time.monotonic()-30)
     if time.monotonic()-idle[g['uuid']]<30 or available<45*2**30:continue
     required=max(0,21*2**30-d.size(d.DATA/'resume'/rid))
     if shutil.disk_usage(d.DATA).free<required or shutil.disk_usage(d.OUT).free<256*2**20:continue
     name=f'{a.phase}_{rid}_{time.time_ns()}';env=dict(os.environ,CUDA_VISIBLE_DEVICES=g['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PATH='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+os.environ['PATH'],CUDA_HOME='/usr/local/cuda-12.4');path=d.OUT/'logs'/f'{name}.log';path.parent.mkdir(exist_ok=True);log=path.open('a');s=time.monotonic();argv=[sys.executable,'-u','-m','experiments.stream_factorization_run','--rid',rid,'--name',name]+(['--profile'] if a.phase=='profile' else []);child=subprocess.Popen(argv,cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);w=dict(name=name,phase=a.phase,id=rid,PID=child.pid,GPU_index=g['index'],GPU_UUID=g['uuid'],started_utc=d.now(),elapsed_seconds=0,exit_code=None,actual_wait_returned=False,admission=g);active=dict(child=child,w=w,start=s,log=log);d.write(d.OUT/'workers'/f'{name}.json',w);print(dict(launched=name,PID=child.pid,GPU=g['index']),flush=True);break
   d.write(d.OUT/'queue_status.json',dict(utc=d.now(),manager_PID=os.getpid(),phase=a.phase,active=active['w'] if active else None,pending=pending,failures=fail,halt_reason=halt,GPU_worker_seconds=seconds,profiling_seconds=profiles,DATA_bytes=databytes,DATA_peak_sampled_bytes=peakdata,project_peak_sampled_bytes=peakproject,peak_NVML_worker_MiB=peakNVML,RAM_available_bytes=available,cards=rows,concurrency='Two full96 file-backed banks reserve18.3GiB per stream;32GiB DATA plus permanent predictions permits one live/resumable stream. Dynamicall8card admission, no arbitrarytwoGPUcap.'))
   if not active and (halt or not pending):break
   time.sleep(1)
  complete=not halt and all((d.OUT/'completed'/f'{r}.json').exists() for r in jobs);rec=dict(complete=complete,manager_PID=os.getpid(),actual_all_children_waited=True,halt_reason=halt,failures=fail,ended_utc=d.now(),wall_seconds=time.monotonic()-start,DATA_peak_sampled_bytes=peakdata,project_peak_sampled_bytes=peakproject,peak_NVML_worker_MiB=peakNVML)
  d.write(d.OUT/f'pool_{a.phase}_complete.json',rec)
  if a.phase=='profile' and complete:
   elapsed=sum(w['elapsed_seconds'] for w in workers if w['phase']=='profile');remaining=11*elapsed*1.25;admit=elapsed+remaining<14400 and elapsed<1800
   d.write(d.OUT/'profiling_admission.json',dict(admitted_before_scoring=admit,registered_utc=d.now(),first96_actual_worker_seconds=elapsed,remaining11_same_case_with25pct_envelope_seconds=remaining,total_projected_seconds=elapsed+remaining,profile_peak_NVML_MiB=peakNVML,DATA_peak_sampled_bytes=peakdata,scientific_scores_not_read=True,first_stream_outputs_reused=True))
  if halt:raise SystemExit(75)
if __name__=='__main__':main()
