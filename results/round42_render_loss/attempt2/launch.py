"""Execute disjoint frozen scene/seed shards on idle GPUs; no model selection."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'results/round42_render_loss/attempt2'
GROUPS={0:[('teatime',11),('waldo_kitchen',44)],
        1:[('teatime',22),('waldo_kitchen',55)],
        2:[('teatime',33)],3:[('teatime',44)],4:[('teatime',55)],
        5:[('waldo_kitchen',11),('waldo_kitchen',22),('waldo_kitchen',33)]}

def save(path,record):
    path.write_text(json.dumps(record,indent=2)+'\n')

def worker(gpu,jobs):
    rows=[]
    for scene,seed in jobs:
        deadline=time.time()+3600;idle=0
        while idle<2:
            value=subprocess.check_output(['nvidia-smi','-i',str(gpu),
                '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            memory,util=[int(x.strip()) for x in value.split(',')]
            idle=idle+1 if memory<16 and util==0 else 0
            if time.time()>deadline:raise TimeoutError(f'GPU{gpu} did not become idle')
            if idle<2:time.sleep(3)
        env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),
            CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        command=[sys.executable,'-m','experiments.round42_render_train','train',
            '--scene',scene,'--seeds',str(seed),'--out',str(OUT)]
        log=OUT/f'{scene}_{seed}.log';record=dict(scene=scene,seed=seed,gpu=gpu,
            command=command,log=str(log),started_unix=time.time())
        receipt=OUT/f'{scene}_{seed}_process.json'
        with log.open('x') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            record['pid']=process.pid;save(receipt,record)
            print('START',gpu,scene,seed,process.pid,flush=True)
            record['exit_code']=process.wait()
        record['completed_unix']=time.time();save(receipt,record);rows.append(record)
        print('EXIT',gpu,scene,seed,record['exit_code'],flush=True)
        if record['exit_code']!=0:break
    return rows

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(worker,gpu,jobs) for gpu,jobs in GROUPS.items()]
        rows=[];errors=[]
        for future in futures:
            try:rows.extend(future.result())
            except BaseException as error:errors.append(repr(error))
    passed=len(rows)==10 and not errors and all(r['exit_code']==0 for r in rows)
    save(OUT/'launcher_completion.json',dict(passed=passed,records=rows,errors=errors,unix=time.time()))
    raise SystemExit(0 if passed else 1)
