"""Sequential bounded efficiency work; never starts scientific training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/efficiency_sidecar/deployment1'


def save(value):
    path=OUT/'queue_status.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'queue_status.json').exists():raise FileExistsError('Preserve prior queue')
    r=dict(state='running',pid=os.getpid(),started_unix=time.time(),completed=[],gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    save(r)
    for stage in ('evaluation','combination'):
        for role in ('relative5','distance_mlp'):
            env=os.environ.copy();cache=OUT/'cache'/stage/role
            assert not cache.exists();cache.mkdir(parents=True)
            env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),
                TORCHINDUCTOR_COMPILE_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8')
            log=OUT/f'{stage}_{role}.log';started=time.time()
            with log.open('x') as stream:
                proc=subprocess.Popen([sys.executable,'-u','-m','experiments.efficiency_sidecar.deployment_gpu',stage,role],
                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                r['active']=dict(stage=stage,role=role,pid=proc.pid,log=str(log));save(r)
                try:code=proc.wait(timeout=1260)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    code=124
            r['completed'].append(dict(stage=stage,role=role,exit_code=code,process_seconds=time.time()-started))
            if code:
                r.update(state='failed',active=None);save(r);sys.exit(code)
            save(r)
    r.update(state='complete',active=None,elapsed_seconds=time.time()-r['started_unix']);save(r)
