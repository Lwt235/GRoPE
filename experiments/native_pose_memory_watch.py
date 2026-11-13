"""Watch only our recorded child PIDs; never operate on foreign processes."""
import os,sys,time,signal,subprocess,json
from pathlib import Path
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_pose_memory_pe1';CACHE=Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/native_pose_memory_pe_v1')
children=[(110439,'GPU-0a17e6c1-0ac8-cbed-cea7-d34fbb759d92'),(110440,'GPU-bed502d0-6c7a-c66d-42b6-8aa08676eec7')]
def alive(pid):
    try:return 'experiments.native_pose_memory_run' in Path(f'/proc/{pid}/cmdline').read_text()
    except OSError:return False
while any(alive(pid) for pid,gpu in children):
    rows=[json.loads(p.read_text()) for p in (OUT/'workers').glob('*.json')];seconds=sum(r['seconds'] for r in rows)
    status=[]
    for pid,gpu in children:
        if not alive(pid):continue
        free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());status.append(dict(pid=pid,gpu=gpu,free_MiB=free))
        fs=os.statvfs(CACHE);diskfree=fs.f_bavail*fs.f_frsize
        if seconds>8*3600-120 or free<6144 or diskfree<8*2**30:
            # Fits are separately durable every100; a signal requests graceful checkpoint in supported runtime.
            os.kill(pid,signal.SIGTERM)
            status[-1]['stop_sent']=True
    with (OUT/'watchdog.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=time.time(),worker_seconds=seconds,status=status))+'\n')
    time.sleep(30)
with (OUT/'watchdog_complete.json').open('w') as f:json.dump(dict(complete=True,epoch=time.time()),f)
