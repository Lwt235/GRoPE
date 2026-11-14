"""Unique resource-admitted process manager; every stage contributes to one round budget."""
import os,sys,time,json,fcntl,subprocess,signal,argparse
from pathlib import Path
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_track_memory_pe1';DATA=Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/native_track_memory_pe_v1');STOP=False
def stop(*a):
    global STOP;STOP=True
def now():return __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def read(p):return json.loads(Path(p).read_text())
def gpus():
    rows=[]
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
        i,u,m,f,v=[s.strip() for s in line.split(',')];rows.append(dict(index=int(i),uuid=u,used_MiB=int(m),free_MiB=int(f),util=int(v)))
    apps={}
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines():
        u,p=[s.strip() for s in line.split(',')];apps.setdefault(u,set()).add(int(p))
    return rows,apps
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','preflight','fits','cost']);a=p.parse_args();OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True);(OUT/'workers').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True);signal.signal(signal.SIGTERM,stop)
    with (OUT/'pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert all(w.get('actual_wait_returned') for w in [read(p) for p in (OUT/'workers').glob('*.json')]),'Old child not waited; inspect before any recovery'
        jobs=list(range(3)) if a.phase=='prepare' else ([0] if a.phase in ('preflight','cost') else [(k,s) for s in (20261110,20261111,20261112) for k in ('P','S','G','L')]);active={};failed=[]
        def complete(job):
            if a.phase=='prepare':return (OUT/'prepare_shards'/f'{job}.json').exists()
            if a.phase=='preflight':return (OUT/'preflight.json').exists()
            if a.phase=='cost':return (OUT/'actual_cost.json').exists()
            return (OUT/'evaluation'/f'{job[0]}_{job[1]}.json').exists()
        try:
            while True:
                for name,w in list(active.items()):
                    code=w['child'].poll();w['info'].update(elapsed_seconds=time.monotonic()-w['start'],exit_code=code)
                    if code is not None:
                        w['info'].update(exit_code=w['child'].wait(),actual_wait_returned=True,ended_utc=now());w['log'].close();del active[name]
                        if code not in (0,75):failed.append(w['info'])
                        if code==0 and a.phase=='prepare':write(OUT/'prepare_shards'/f'{w["job"]}.json',dict(complete=True,worker=name,actual_exit_code=0))
                    write(OUT/'workers'/f'{name}.json',w['info'])
                records=[read(p) for p in (OUT/'workers').glob('*.json')];seconds=sum(w['elapsed_seconds'] for w in records);used=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file());project=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
                halt=STOP or bool(failed) or seconds>12*3600-180 or used+256*2**20>12*2**30 or project+64*2**20>2**30
                pending=[j for j in jobs if not complete(j) and not any(w['job']==j for w in active.values())];rows,apps=gpus()
                if not halt:
                    for g in rows:
                        if not pending:break
                        if g['used_MiB']>256 or g['free_MiB']<24000 or g['util']>2 or apps.get(g['uuid']) or any(w['info']['gpu_uuid']==g['uuid'] for w in active.values()) or avail<20*2**30:continue
                        j=pending.pop(0);name=f'{a.phase}_{j}_{time.time_ns()}'.replace(' ','');env=dict(os.environ,CUDA_VISIBLE_DEVICES=g['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PATH='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+os.environ['PATH'],CUDA_HOME='/usr/local/cuda-12.4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR='/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/torch_extensions',MAX_JOBS='2');cmd=[sys.executable,'-u','-m','experiments.native_track_memory_prepare','--shard',str(j),'--name',name] if a.phase=='prepare' else [sys.executable,'-u','-m','experiments.native_track_memory_run',a.phase,'--name',name]
                        if a.phase=='fits':cmd+=['--kind',j[0],'--seed',str(j[1])]
                        if a.phase=='cost':cmd=[sys.executable,'-u','-m','experiments.native_track_memory_cost','--name',name]
                        lf=(OUT/'logs'/f'{name}.log').open('a');t=time.monotonic();child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=lf,stderr=subprocess.STDOUT);info=dict(name=name,phase=a.phase,job=j,pid=child.pid,GPU_index=g['index'],gpu_uuid=g['uuid'],started_utc=now(),elapsed_seconds=0.,exit_code=None,admission=g,log=str(OUT/'logs'/f'{name}.log'));active[name]=dict(child=child,log=lf,info=info,start=t,job=j);write(OUT/'workers'/f'{name}.json',info);print(dict(launched=a.phase,job=j,pid=child.pid,GPU_index=g['index']),flush=True)
                for w in active.values():
                    g=next(g for g in rows if g['uuid']==w['info']['gpu_uuid'])
                    if (halt or g['free_MiB']<4096) and not w.get('signalled'):w['child'].send_signal(signal.SIGTERM);w['signalled']=True
                write(OUT/'queue_status.json',dict(utc=now(),phase=a.phase,manager_pid=os.getpid(),active=[w['info'] for w in active.values()],pending=pending,failed=failed,GPU_worker_seconds=seconds,DATA_bytes=used,project_bytes=project,RAM_available_bytes=avail,prepared_pairs=len(list((OUT/'prepared_pairs').glob('*.json'))),no_default_two_worker_cap=True))
                if not active and (halt or not pending):break
                time.sleep(2)
        finally:
            for w in active.values():
                if w['child'].poll() is None:w['child'].send_signal(signal.SIGTERM)
            for name,w in active.items():
                code=w['child'].wait();w['info'].update(elapsed_seconds=time.monotonic()-w['start'],exit_code=code,actual_wait_returned=True,ended_utc=now());w['log'].close();write(OUT/'workers'/f'{name}.json',w['info'])
        ok=not failed and not halt and all(complete(j) for j in jobs);write(OUT/f'pool_{a.phase}_complete.json',dict(complete=ok,actual_children_all_waited=True,ended_utc=now(),failed=failed));print(dict(complete=ok,phase=a.phase),flush=True)
        if not ok:raise SystemExit(75)
if __name__=='__main__':main()
