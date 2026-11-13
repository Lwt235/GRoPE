"""One admitted cost worker with actual wall and exit accounting."""
import os,signal,subprocess,sys,time
from experiments import native_task_writer_data as d
from experiments.native_task_writer_manage import gpu_rows

def main():
    gpu=sys.argv[1];name=f'cost_{time.time_ns()}';assert not (d.OUT/'cost.json').exists();g=next(g for g in gpu_rows() if str(g['index'])==gpu or g['uuid']==gpu);assert g['used']<256 and g['free']>=42*1024 and not d.resource(True)['stop_reasons']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=g['uuid'],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.DATA/'torch_extensions'));env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH'];start=time.monotonic();log=(d.OUT/'logs'/f'{name}.log').open('a');child=subprocess.Popen([sys.executable,'-u','-m','experiments.native_task_writer_cost'],env=env,cwd=d.ROOT,stdout=log,stderr=subprocess.STDOUT);record=dict(name=name,pid=child.pid,gpu_uuid=g['uuid'],GPU_index=g['index'],started_utc=d.now(),admission=g,elapsed_seconds=0.,exit_code=None)
    while child.poll() is None:
        record.update(elapsed_seconds=time.monotonic()-start,updated_utc=d.now());d.write(d.OUT/'workers'/f'{name}.json',record);rr=d.resource();gg=next(x for x in gpu_rows() if x['uuid']==g['uuid'])
        if rr['stop_reasons'] or gg['free']<6144:child.send_signal(signal.SIGTERM)
        time.sleep(5)
    record.update(elapsed_seconds=time.monotonic()-start,actual_process_wait_returned=True,exit_code=child.wait(),ended_utc=d.now());d.write(d.OUT/'workers'/f'{name}.json',record);log.close();print(record,flush=True);raise SystemExit(record['exit_code'])
if __name__=='__main__':main()
