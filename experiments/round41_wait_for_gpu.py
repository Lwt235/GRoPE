"""Queue the already frozen component diagnostic on a genuinely idle GPU.

No process is stopped, no occupied GPU is shared, and no training is launched.
The bounded queue exits after one hour if indices4..7 never become idle.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/round41_shape_components'


def write(value):
    (DEST/'queue_status.json').write_text(json.dumps(value,indent=2)+'\n')


def occupancy():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    return [dict(zip(('index','memory_mib','utilization'),map(int,line.split(','))))
            for line in result.stdout.strip().splitlines()]


def main():
    if (DEST/'queue_status.json').exists():raise ValueError('queue already exists; inspect its live handle')
    started=time.time();previous=None
    info=dict(started_unix=started,pid=os.getpid(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print('Waiting for an unoccupied GPU among4..7; maximum3600seconds.',flush=True)
    while time.time()-started<3600:
        rows=occupancy()
        available=[r['index'] for r in rows if r['index'] in (4,5,6,7) and r['memory_mib']<16 and r['utilization']==0]
        gpu=available[0] if available else None
        write(dict(**info,state='waiting',observed_unix=time.time(),occupancy=rows))
        if gpu is not None and gpu==previous:
            env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),
                CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            write(dict(**info,state='running',gpu=gpu,launched_unix=time.time()))
            for stage in ('forward','metrics'):
                log=ROOT/'results'/f'round41_shape_components_{stage}.log'
                with log.open('x') as stream:
                    result=subprocess.run([sys.executable,'-m','experiments.round41_shape_components',stage],
                        cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:
                    write(dict(**info,state='failed',gpu=gpu,stage=stage,exit_code=result.returncode))
                    raise SystemExit(result.returncode)
            write(dict(**info,state='complete',gpu=gpu,completed_unix=time.time()))
            print('Component diagnostic complete.',flush=True);return
        previous=gpu;time.sleep(15)
    write(dict(**info,state='queue_timeout',completed_unix=time.time(),experiment_started=False))
    raise SystemExit(75)


if __name__=='__main__':main()
