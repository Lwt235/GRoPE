"""One finite prescribed cost worker with actual lifetime accounting and safe yield."""
import fcntl,os,signal,subprocess,sys,time
from experiments import native_read_address_data as d
from experiments.native_task_writer_manage import gpu_rows,compute_pids
def main():
    index=int(sys.argv[1]);g=next(g for g in gpu_rows() if g['index']==index);apps=compute_pids();assert not apps.get(g['uuid']) and g['used']<256 and g['util']<=2 and g['free']>=42*1024
    assert not (d.OUT/'cost.json').exists() and not (d.OUT/'calls/cost.jsonl').exists();assert not d.resource(True)['stop_reasons'];assert all(d.final(a,d.SEEDS[0]).exists() for a in d.ARMS)
    name=f'cost_{time.time_ns()}';path=d.OUT/'logs'/f'{name}.log';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=g['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.DATA/'torch_extensions'));env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH'];begin=time.monotonic()
    with path.open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m','experiments.native_read_address_cost'],cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);info=dict(name=name,phase='cost',pid=child.pid,gpu_uuid=g['uuid'],GPU_index=index,started_utc=d.now(),elapsed_seconds=0.,exit_code=None,log=str(path),admission=g);sent=False
        while child.poll() is None:
            info.update(elapsed_seconds=time.monotonic()-begin,updated_utc=d.now());d.write(d.OUT/'workers'/f'{name}.json',info);rr=d.resource();live=next(x for x in gpu_rows() if x['uuid']==g['uuid']);d.journal(d.OUT/'gpu_resources'/f'{name}.jsonl',dict(utc=d.now(),**live,compute_pids=sorted(compute_pids().get(g['uuid'],set()))))
            if (rr['stop_reasons'] or live['free']<6144) and not sent:child.send_signal(signal.SIGTERM);sent=True;info['stop_reason']=rr['stop_reasons'] or ['local_GPU_pressure']
            time.sleep(5)
        code=child.wait();info.update(elapsed_seconds=time.monotonic()-begin,exit_code=code,actual_process_wait_returned=True,ended_utc=d.now());d.write(d.OUT/'workers'/f'{name}.json',info);print(info,flush=True)
    if code:raise SystemExit(code)
if __name__=='__main__':
    with (d.OUT/'claims/cost.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
