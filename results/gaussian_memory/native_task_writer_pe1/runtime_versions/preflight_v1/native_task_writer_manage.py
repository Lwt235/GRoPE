"""Unique dynamic idle-GPU scheduler and full active-worker accounting."""
import argparse,fcntl,os,signal,subprocess,sys,time
from experiments import native_task_writer_data as d

def gpu_rows():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(x[0]),uuid=x[1],used=int(x[2]),free=int(x[3]),util=int(x[4])) for line in text.splitlines() for x in [[s.strip() for s in line.split(',')]]]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','formal']);a=parser.parse_args();d.OUT.mkdir(parents=True,exist_ok=True);d.CACHE.mkdir(parents=True,exist_ok=True)
    with (d.OUT/'manager.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[('preflight',None)] if a.phase=='preflight' else [(arm,seed) for seed in d.SEEDS for arm in d.ARMS]
        active={};completed=set();failed=[];started=d.now();attempt=time.time_ns();stop=False
        for arm,seed in list(jobs):
            if arm!='preflight' and all((d.OUT/'evaluation'/f'{arm}_{seed}_{m["id"]}.json').exists() for m in d.metas()):completed.add((arm,seed))
        while True:
            for name,record in list(active.items()):
                child=record['child'];info=record['info'];code=child.poll();info.update(elapsed_seconds=time.monotonic()-record['start'],updated_utc=d.now(),exit_code=code)
                d.write(d.OUT/'workers'/f'{name}.json',info)
                if code is not None:
                    info.update(actual_process_wait_returned=True,exit_code=child.wait(),ended_utc=d.now());d.write(d.OUT/'workers'/f'{name}.json',info);record['log'].close();del active[name]
                    if code==0:completed.add(record['job'])
                    else:failed.append(dict(job=record['job'],exit_code=code,name=name));stop=True
            rr=d.resource();gpus=gpu_rows();busy={r['info']['gpu_uuid'] for r in active.values()}
            for name,r in active.items():
                g=next(x for x in gpus if x['uuid']==r['info']['gpu_uuid']);d.journal(d.OUT/'gpu_resources'/f'{name}.jsonl',dict(utc=d.now(),**g))
                if (rr['stop_reasons'] or g['free']<6144 or stop) and not r.get('stop_requested'):
                    r['child'].send_signal(signal.SIGTERM);r['stop_requested']=time.monotonic();r['info']['stop_reason']=rr['stop_reasons'] or ['foreign_GPU_pressure_or_sibling_failure'];stop=True
            pending=[job for job in jobs if job not in completed and job not in [r['job'] for r in active.values()]]
            if not active and (not pending or stop):break
            if not stop and not rr['stop_reasons']:
                for g in gpus:
                    if not pending or g['uuid'] in busy or g['used']>=256 or g['util']>2 or g['free']<42*1024:continue
                    if len(active)*35*2**30+30*2**30>rr['RAM_available_bytes']:continue
                    arm,seed=pending.pop(0);name=f'{arm}_{seed}_{attempt}';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=g['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.DATA/'torch_extensions'))
                    env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH'];logpath=d.OUT/'logs'/f'{name}.log';logpath.parent.mkdir(exist_ok=True);log=logpath.open('a',buffering=1)
                    cmd=[sys.executable,'-u','-m','experiments.native_task_writer_run','preflight' if arm=='preflight' else 'fit','--worker',name]
                    if arm!='preflight':cmd+=['--arm',arm,'--seed',str(seed)]
                    begin=time.monotonic();child=subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    info=dict(name=name,phase=a.phase,arm=arm,seed=seed,pid=child.pid,manager_pid=os.getpid(),gpu_uuid=g['uuid'],GPU_index=g['index'],admission=g,started_utc=d.now(),elapsed_seconds=0,exit_code=None,log=str(logpath))
                    active[name]=dict(child=child,info=info,start=begin,job=(arm,seed),log=log);d.write(d.OUT/'workers'/f'{name}.json',info)
                    print(dict(launched=name,gpu=g['index'],pid=child.pid),flush=True)
            d.write(d.OUT/'queue_status.json',dict(started_utc=started,updated_utc=d.now(),manager_pid=os.getpid(),phase=a.phase,active=[r['info'] for r in active.values()],completed=sorted([list(j) for j in completed],key=str),failed=failed,pending=pending,dynamic_idle_policy=True,resource=rr))
            time.sleep(10)
        result=dict(complete=not failed and not pending,completed=[list(j) for j in completed],failed=failed,all_children_waited=True,ended_utc=d.now());d.write(d.OUT/f'{a.phase}_queue_complete.json',result);print(result,flush=True)
        if failed or pending:raise SystemExit(75)
if __name__=='__main__':main()
