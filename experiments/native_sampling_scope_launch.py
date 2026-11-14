"""One bounded CPU stage, measuring the complete child through actual wait."""
import datetime,json,os,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_sampling_scope1'
phase=sys.argv[1];assert phase in ('aggregate','delivery')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
cmd=[sys.executable,'-u','-m','experiments.native_sampling_scope',phase] if phase=='aggregate' else [sys.executable,'-u','-m','experiments.native_sampling_scope_delivery']
workers=OUT/'workers';workers.mkdir(exist_ok=True);name=f'{phase}_{time.time_ns()}';path=workers/f'{name}.json'
start=time.monotonic();epoch=time.time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
with (OUT/f'{name}.log').open('w') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    info=dict(phase=phase,name=name,pid=child.pid,command=cmd,started_epoch=epoch,started_utc=datetime.datetime.fromtimestamp(epoch,datetime.timezone.utc).isoformat(),actual_wait_returned=False)
    path.write_text(json.dumps(info,indent=2)+'\n')
    try:code=child.wait(timeout=3500)
    except subprocess.TimeoutExpired:
        child.terminate();code=child.wait(timeout=30)
    end=time.monotonic();usage=resource.getrusage(resource.RUSAGE_CHILDREN);info.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_wait_returned=True,exit_code=code,full_child_wall_seconds=end-start,full_child_CPU_seconds=usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime,full_child_peak_RSS_bytes=usage.ru_maxrss*1024,log=str(OUT/f'{name}.log'),scope='Full Popen-to-actual-wait wall includes Python/Torch imports, reads, compute, output and exit. One CPU child, native math libraries capped to1 thread, CUDA hidden.')
    path.write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info),flush=True)
raise SystemExit(code)
