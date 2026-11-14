"""One manager: all geometry committed before any feature reads, unique scene leases."""
import os,sys,subprocess,time,fcntl,signal,json,datetime
from pathlib import Path
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_crossview_content1';DATA=Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/native_crossview_content_v1');STOP=False
def stop(*a):
    global STOP;STOP=True
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def gpus():
    rows=[]
    for l in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
        i,u,m,f,v=[x.strip() for x in l.split(',')];rows.append(dict(index=int(i),uuid=u,used_MiB=int(m),free_MiB=int(f),util=int(v)))
    apps={}
    for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines():
        u,p=[x.strip() for x in l.split(',')];apps.setdefault(u,set()).add(int(p))
    return rows,apps
def main():
    OUT.mkdir(exist_ok=True);(OUT/'workers').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True);signal.signal(signal.SIGTERM,stop)
    with (OUT/'pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert all(w.get('actual_wait_returned') for w in [read(p) for p in (OUT/'workers').glob('*.json')]),'Unwaited old child: do not duplicate'
        envbase=os.environ.copy();envbase.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if not (OUT/'configuration.json').exists():
            t=time.monotonic();p=subprocess.Popen([sys.executable,'-u','-m','experiments.native_crossview_content','prepare'],cwd=ROOT,env=dict(envbase,CUDA_VISIBLE_DEVICES=''));code=p.wait();write(OUT/'prepare_runtime.json',dict(actual_exit_code=code,pid=p.pid,wall_seconds=time.monotonic()-t,actual_wait_returned=True));assert code==0
        cfg=read(OUT/'configuration.json');active={};failed=[];max_rss=0;begun=now()
        def launch(phase,rid,g):
            name=f'{phase}_{rid}_{time.time_ns()}';env=dict(envbase,CUDA_VISIBLE_DEVICES=g['uuid']);lf=(OUT/'logs'/f'{name}.log').open('a');t=time.monotonic();p=subprocess.Popen([sys.executable,'-u','-m','experiments.native_crossview_content',phase,'--id',rid,'--name',name],cwd=ROOT,env=env,stdout=lf,stderr=subprocess.STDOUT)
            info=dict(name=name,phase=phase,scene=rid,pid=p.pid,GPU_index=g['index'],gpu_uuid=g['uuid'],admission=g,started_utc=now(),elapsed_seconds=0.,exit_code=None,log=str(OUT/'logs'/f'{name}.log'));active[name]=dict(child=p,start=t,info=info,log=lf);write(OUT/'workers'/f'{name}.json',info);print(dict(launched=phase,scene=rid,pid=p.pid,GPU_index=g['index']),flush=True)
        while True:
            for name,w in list(active.items()):
                code=w['child'].poll();w['info'].update(elapsed_seconds=time.monotonic()-w['start'],exit_code=code)
                if code is not None:
                    w['info'].update(actual_wait_returned=True,exit_code=w['child'].wait(),ended_utc=now());w['log'].close();del active[name]
                    if code not in (0,75):failed.append(w['info'])
                write(OUT/'workers'/f'{name}.json',w['info'])
            records=[read(p) for p in (OUT/'workers').glob('*.json')];gpu=sum(w['elapsed_seconds'] for w in records);szdata=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file());szout=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;rss=0
            for w in active.values():
                p=Path('/proc')/str(w['info']['pid'])/'status'
                if p.exists():rss+=int(next(l.split()[1] for l in p.read_text().splitlines() if l.startswith('VmRSS:')))*1024
            max_rss=max(max_rss,rss);free=os.statvfs(DATA).f_bavail*os.statvfs(DATA).f_frsize;projectfree=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
            halt=STOP or bool(failed) or gpu>=3500 or szdata+32*2**20>=4*2**30 or szout+32*2**20>=2**30 or rss>=31*2**30 or free<2**30 or projectfree<2**30
            geometry_count=len(list((OUT/'geometry').glob('*.json')));measurement_count=len(list((OUT/'measurements').glob('*.json')))
            if geometry_count==264 and not (OUT/'geometry_lock.json').exists() and not active:
                import hashlib
                markers=[read(OUT/'geometry'/f'{c["key"]}.json') for c in cfg['cases']];assert all(x['complete'] and not x['feature_values_read'] for x in markers)
                write(OUT/'geometry_lock.json',dict(complete=True,pairs=264,query_source_pairs=192,source_source_pairs=72,feature_values_read=False,geometry_data_identities={x['key']:x['sha256'] for x in markers},configuration_sha256=hashlib.sha256((OUT/'configuration.json').read_bytes()).hexdigest(),locked_utc=now()))
            phase='features' if (OUT/'geometry_lock.json').exists() else 'geometry';folder='measurements' if phase=='features' else 'geometry';pending=[]
            for rid in sorted({c['id'] for c in cfg['cases']}):
                if any(not (OUT/folder/f'{c["key"]}.json').exists() for c in cfg['cases'] if c['id']==rid) and not any(w['info']['scene']==rid for w in active.values()):pending.append(rid)
            rows,apps=gpus()
            if not halt:
                for g in rows:
                    if not pending:break
                    if g['used_MiB']>=256 or g['free_MiB']<4096 or g['util']>2 or apps.get(g['uuid']) or any(w['info']['gpu_uuid']==g['uuid'] for w in active.values()):continue
                    if avail<8*2**30 or (len(active)+1)*3*2**30+2*2**30>32*2**30:continue
                    launch(phase,pending.pop(0),g)
            for w in active.values():
                g=next(x for x in rows if x['uuid']==w['info']['gpu_uuid']);foreign=apps.get(g['uuid'],set())-{w['info']['pid']}
                if (halt or (foreign and g['free_MiB']<4096)) and not w.get('signal_sent'):
                    w['child'].send_signal(signal.SIGTERM);w['signal_sent']=True;w['info']['stop_reason']='budget_or_foreign_pressure';write(OUT/'workers'/f'{w["info"]["name"]}.json',w['info'])
            resources=dict(GPU_worker_seconds=gpu,DATA_bytes=szdata,project_bytes=szout,live_child_RSS_bytes=rss,max_observed_concurrent_child_RSS_bytes=max_rss,RAM_available_bytes=avail,DATA_free_bytes=free,project_free_bytes=projectfree)
            write(OUT/'queue_status.json',dict(utc=now(),manager_pid=os.getpid(),phase=phase,active=[w['info'] for w in active.values()],pending=pending,geometry_completed=geometry_count,measurement_completed=measurement_count,failed=failed,resources=resources,no_default_two_card_cap=True))
            if not active and (halt or (phase=='features' and not pending)):break
            time.sleep(1)
        ok=not halt and not failed and measurement_count==264;write(OUT/'pool_complete.json',dict(complete=ok,all_children_actual_wait=True,geometry_completed=geometry_count,measurement_completed=measurement_count,workers=len(records),started_utc=begun,ended_utc=now(),resources=resources,failures=failed));print(dict(complete=ok,geometry=geometry_count,measurements=measurement_count),flush=True)
        if not ok:raise SystemExit(75)
if __name__=='__main__':main()
