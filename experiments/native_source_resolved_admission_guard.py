"""CPU-only guard for observed foreign-job gaps; never owns scientific jobs."""
import os,time,json,signal,resource
from pathlib import Path
from experiments.native_source_resolved_pool import OUT,gpus,write,now
STOP=False
def stop(*a):
    global STOP;STOP=True
def main():
    signal.signal(signal.SIGTERM,stop);start=time.monotonic();snapshots=0
    while not STOP:
        q=json.loads((OUT/'queue_status.json').read_text());rows,apps=gpus();own={w['pid'] for w in q['active']};path=OUT/'admission_guard_policy.json';policy=json.loads(path.read_text()) if path.exists() else dict(enabled=True,stable_idle_seconds=120,guarded_GPU_indices=[4],by_uuid={},scope='Only pre-CUDA admission; no scientific job ownership or model calls')
        for g in rows:
            if g['index'] not in policy['guarded_GPU_indices']:continue
            foreign=apps.get(g['uuid'],set())-own
            if foreign or g['used_MiB']>256 or g['util']>2:
                # Do not reset a known own normal worker's stage transition.
                if foreign or not apps.get(g['uuid'],set())&own:policy['by_uuid'][g['uuid']]=dict(GPU_index=g['index'],last_foreign_or_busy_seen_unix=time.time(),last_seen_utc=now(),external_PIDs=sorted(foreign))
        policy.update(updated_utc=now(),observer_PID=os.getpid());write(path,policy);snapshots+=1
        completed=OUT/'pool_fits_complete.json'
        if completed.exists() and json.loads(completed.read_text()).get('complete'):break
        for _ in range(10):
            if STOP:break
            time.sleep(.5)
    write(OUT/'CPU_admission_guard_runtime.json',dict(complete=not STOP,actual_exit_intended=0,PID=os.getpid(),wall_seconds=time.monotonic()-start,user_CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,system_CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,snapshots=snapshots,CUDA_initialized=False,GPU_model_calls=0,scientific_jobs_dispatched=0));print('CPU-only admission observer complete',flush=True)
if __name__=='__main__':main()
