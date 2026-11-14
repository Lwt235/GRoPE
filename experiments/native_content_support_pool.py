"""Unique, live-resource-safe CPU preparation and three disjoint image shards."""
import os,sys,json,subprocess,time,fcntl,signal,datetime,resource
from pathlib import Path
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_content_support1';STOP=False
def stop(*a):
    global STOP;STOP=True
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def gpus():
    rows=[]
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
        i,u,used,free,util=[x.strip() for x in line.split(',')];rows.append(dict(index=int(i),uuid=u,used=int(used),free=int(free),util=int(util)))
    apps={}
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines():
        u,p=[x.strip() for x in line.split(',')];apps.setdefault(u,set()).add(int(p))
    return rows,apps
def resources():
    ws=[read(p) for p in (OUT/'workers').glob('*.json')];gpu=sum(w.get('elapsed_seconds',0) for w in ws if w['phase']!='prepare');cpu=sum(w.get('elapsed_seconds',0) for w in ws);files=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;rss=0
    for w in ws:
        if w.get('exit_code') is None:
            p=Path('/proc')/str(w['pid'])/'status'
            if p.exists():rss+=int(next(l.split()[1] for l in p.read_text().splitlines() if l.startswith('VmRSS:')))*1024
    return dict(utc=now(),GPU_worker_seconds=gpu,CPU_worker_seconds=cpu,storage_bytes=files,live_child_RSS_bytes=rss,RAM_available_bytes=mem,project_free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize)
def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'workers').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True);signal.signal(signal.SIGTERM,stop)
    envbase=os.environ.copy();envbase.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR='/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/torch_extensions');envbase['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+envbase['PATH']
    with (OUT/'pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert all(w.get('actual_wait_returned') for w in [read(p) for p in (OUT/'workers').glob('*.json')]),'Existing child not yet waited'
        active={};failed=[]
        def launch(phase,index,g=None):
            name=f'{phase}_{index}_{time.time_ns()}';env=envbase.copy();env['CUDA_VISIBLE_DEVICES']=g['uuid'] if g else '';cmd=[sys.executable,'-u','-m','experiments.native_content_support','prepare' if phase=='prepare' else 'worker','--index',str(index),'--name',name]
            if phase=='preflight':cmd.append('--preflight')
            log=(OUT/'logs'/f'{name}.log').open('a');t=time.monotonic();child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);info=dict(name=name,phase=phase,index=index,pid=child.pid,GPU_index=g['index'] if g else None,gpu_uuid=g['uuid'] if g else None,started_utc=now(),elapsed_seconds=0.,exit_code=None,admission=g,log=str(OUT/'logs'/f'{name}.log'));active[name]=dict(child=child,start=t,info=info,log=log);write(OUT/'workers'/f'{name}.json',info);print(dict(launched=phase,shard=index,pid=child.pid,GPU_index=info['GPU_index']),flush=True)
        while True:
            for name,w in list(active.items()):
                code=w['child'].poll();w['info'].update(elapsed_seconds=time.monotonic()-w['start'],exit_code=code,updated_utc=now())
                if code is not None:
                    w['info'].update(exit_code=w['child'].wait(),actual_wait_returned=True,ended_utc=now());w['log'].close();del active[name]
                    if code not in (0,75):failed.append(w['info'])
                write(OUT/'workers'/f'{name}.json',w['info'])
            r=resources();halt=STOP or bool(failed) or r['GPU_worker_seconds']>=2*3600-30 or r['CPU_worker_seconds']>=4*3600-30 or r['storage_bytes']+128*2**20>=3*2**30 or r['live_child_RSS_bytes']>=31*2**30 or r['project_free_bytes']<2**30
            if not (OUT/'configuration.json').exists():phase='prepare';pending=[0] if not active else []
            elif not (OUT/'preflight_complete.json').exists():phase='preflight';pending=[0] if not active else []
            else:
                phase='formal';images=read(OUT/'configuration.json')['images'];pending=[i for i in range(3) if any(not (OUT/'images'/c['key']/'complete.json').exists() for c in images[i::3]) and not any(w['info']['phase']=='formal' and w['info']['index']==i for w in active.values())]
            if not halt:
                if phase=='prepare' and pending:launch(phase,0)
                elif phase!='prepare':
                    rows,apps=gpus()
                    for g in rows:
                        if not pending:break
                        if g['used']>=256 or g['free']<16000 or g['util']>2 or apps.get(g['uuid']) or any(w['info']['gpu_uuid']==g['uuid'] for w in active.values()):continue
                        if r['RAM_available_bytes']<12*2**30 or (len(active)+1)*6*2**30+2*2**30>32*2**30:continue
                        launch(phase,pending.pop(0),g)
            if phase!='prepare':
                rows,apps=gpus()
                for w in active.values():
                    g=next(g for g in rows if g['uuid']==w['info']['gpu_uuid']) if w['info']['gpu_uuid'] else None
                    if (halt or (g and g['free']<6144)) and not w.get('signal_sent'):w['child'].send_signal(signal.SIGTERM);w['signal_sent']=True;w['info']['stop_reason']='round_budget_or_local_pressure';write(OUT/'workers'/f'{w["info"]["name"]}.json',w['info'])
            complete_images=len(list((OUT/'images').glob('*/complete.json'))) if (OUT/'images').exists() else 0
            write(OUT/'queue_status.json',dict(utc=now(),phase=phase,manager_pid=os.getpid(),active=[w['info'] for w in active.values()],pending=pending,completed_images=complete_images,failed=failed,resource=r,independent_shards=3,no_fixed_two_card_limit=True))
            if not active and (halt or (phase=='formal' and not pending)):break
            time.sleep(3)
        ok=not failed and not halt and complete_images==96;write(OUT/'pool_complete.json',dict(complete=ok,all_children_actual_wait=True,completed_images=complete_images,ended_utc=now(),failures=failed));print(dict(complete=ok,completed_images=complete_images),flush=True)
        if not ok:raise SystemExit(75)
if __name__=='__main__':main()
