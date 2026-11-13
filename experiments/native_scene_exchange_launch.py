"""Managed one-process GPU worker; actual process exit and wall/resource ledger."""
import argparse,os,signal,subprocess,sys,time
from experiments import native_scene_exchange_data as d

def main():
    p=argparse.ArgumentParser();p.add_argument('phase');p.add_argument('--gpu',required=True);p.add_argument('--name',required=True);p.add_argument('--arms',nargs='*',default=[]);a=p.parse_args()
    d.OUT.mkdir(parents=True,exist_ok=True);d.CACHE.mkdir(parents=True,exist_ok=True)
    r=d.resource(True);assert not r['stop_reasons'],r
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    row=next(x for x in gpu.splitlines() if a.gpu in x);items=[x.strip() for x in row.split(',')];assert int(items[3])>=30*1024 and int(items[2])<256,(a.gpu,row)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.DATA/'torch_extensions'))
    env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH']
    args=['/home/liwt/miniColmap/.venv/bin/python','-u','-m','experiments.native_scene_exchange_run',a.phase,'--arms',*a.arms]
    log=d.OUT/'logs'/f'{a.name}.log';log.parent.mkdir(exist_ok=True);start=time.monotonic();terminated=False
    with log.open('a',buffering=1) as f:
        child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=d.ROOT)
        info=dict(name=a.name,phase=a.phase,gpu_uuid=a.gpu,pid=child.pid,launcher_pid=os.getpid(),started_utc=d.now(),admission_gpu=row,arms=a.arms,log=str(log))
        while True:
            code=child.poll();info.update(elapsed_seconds=time.monotonic()-start,exit_code=code,updated_utc=d.now());d.write(d.OUT/'workers'/f'{a.name}.json',info)
            if code is not None:break
            rr=d.resource();live=subprocess.check_output(['nvidia-smi','--id='+a.gpu,'--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
            d.journal(d.OUT/'gpu_resources'/f'{a.name}.jsonl',dict(utc=d.now(),elapsed_seconds=info['elapsed_seconds'],gpu=live,pid=child.pid))
            if (rr['stop_reasons'] or int(live.split(',')[1])<6144) and not terminated:
                child.send_signal(signal.SIGTERM);terminated=True;info['stop_request']=dict(resources=rr['stop_reasons'],gpu=live)
            time.sleep(30 if child.poll() is None else .1)
        code=child.wait();info.update(exit_code=code,actual_process_wait_returned=True,ended_utc=d.now(),elapsed_seconds=time.monotonic()-start);d.write(d.OUT/'workers'/f'{a.name}.json',info)
    print(d.json.dumps(info),flush=True);raise SystemExit(code)

if __name__=='__main__':main()
