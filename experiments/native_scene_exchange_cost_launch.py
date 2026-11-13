"""Cost-only managed child, after all formal workers have really exited."""
import os,subprocess,sys,time
from experiments import native_scene_exchange_data as d

def main():
    gpu=sys.argv[1];assert all(r.get('actual_process_wait_returned') and r['exit_code']==0 for r in d.formal_worker_records())
    assert len(list((d.OUT/'fits').glob('*_complete.json')))==21 and len(list((d.OUT/'completed').glob('*.json')))==14784
    info=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip();used,free=[int(x) for x in info.split(',')];assert used<256 and free>30*1024
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.DATA/'torch_extensions'));env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH']
    start=time.monotonic();log=d.OUT/'logs/cost.log'
    with log.open('a',buffering=1) as f:
        child=subprocess.Popen(['/home/liwt/miniColmap/.venv/bin/python','-u','-m','experiments.native_scene_exchange_cost'],env=env,cwd=d.ROOT,stdout=f,stderr=subprocess.STDOUT)
        row=dict(name='cost',gpu_uuid=gpu,pid=child.pid,started_utc=d.now(),admission=info)
        while child.poll() is None:
            row.update(elapsed_seconds=time.monotonic()-start,exit_code=None);d.write(d.OUT/'workers/cost.json',row);r=d.resource()
            if r['stop_reasons']:child.terminate()
            time.sleep(30)
        code=child.wait();row.update(elapsed_seconds=time.monotonic()-start,exit_code=code,actual_process_wait_returned=True,ended_utc=d.now());d.write(d.OUT/'workers/cost.json',row)
    print(row,flush=True);raise SystemExit(code)

if __name__=='__main__':main()
