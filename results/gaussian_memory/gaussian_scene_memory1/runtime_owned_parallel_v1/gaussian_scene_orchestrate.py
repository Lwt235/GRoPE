"""Finite resource-aware queue; never launch on a GPU occupied by another owned worker."""
import fcntl
import json
import os
import subprocess
import sys
import time

from experiments.gaussian_scene_data import OUT,ROOT,KINDS,SEEDS,read
from experiments.gaussian_scene_config import freeze
from experiments.gaussian_memory_incremental_data import write_json


def occupancy():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True,timeout=15)
    return [tuple(map(int,line.split(', '))) for line in rows.strip().splitlines()]


def done(job):
    if job['name']=='references':return (OUT/'references/native.json').exists() and (OUT/'references/renderer.json').exists()
    kind,seed=job['kind'],job['seed']
    return ((OUT/f'{kind}_{seed}'/'status.json').exists() and (OUT/'evaluation'/f'{kind}_{seed}.json').exists()
            and (kind!='G' or (OUT/'interventions'/f'G_{seed}.json').exists()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=freeze()
        assert all((OUT/'query_cache'/f'{m["id"]}.json').exists() for m in p['training_catalog']+p['catalog'])
        jobs=[dict(name='references',module='experiments.gaussian_scene_reference',args=[])]
        jobs += [dict(name=f'{kind}_{seed}',module='experiments.gaussian_scene_train',args=['--kind',kind,'--seed',str(seed)],kind=kind,seed=seed)
                 for seed in SEEDS for kind in KINDS]
        pending=[job for job in jobs if not done(job)];running=[];failed=[];complete=[j['name'] for j in jobs if done(j)]
        (OUT/'orchestrator.pid').write_text(str(os.getpid())+'\n')
        while pending or running:
            for task in list(running):
                rc=task['process'].poll()
                if rc is None:continue
                task['log'].close();running.remove(task)
                if rc==0 and done(task['job']):complete.append(task['job']['name'])
                elif 'no job launched' in open(task['logpath']).read()[-5000:]:
                    pending.insert(0,task['job'])
                    print(json.dumps(dict(resource_race_no_worker_started=task['job']['name'],requeued=True)),flush=True)
                else:
                    failed.append(dict(name=task['job']['name'],returncode=rc,log=task['logpath']))
                    # Existing workers finish; do not launch further jobs until the actual error is understood.
            if not failed:
                occupied={task['gpu'] for task in running};shared_active=any(task['shared'] for task in running)
                states=occupancy()
                candidates=[(gpu,False) for gpu,used,free,util in states if gpu not in occupied and used<=512 and free>=40000 and util<=5]
                if not shared_active:
                    shared=[(gpu,True) for gpu,used,free,util in sorted(states,key=lambda r:r[3]) if gpu not in occupied and used>512 and free>=24000]
                    # At most one shared worker, with the existing launcher enforcing a12GiB cap.
                    candidates+=shared[:1]
                for gpu,shared in candidates:
                    if not pending or len(running)>=4:break
                    job=pending.pop(0);logpath=OUT/'logs'/f'{job["name"]}.log';logpath.parent.mkdir(exist_ok=True)
                    stream=logpath.open('a',buffering=1)
                    launcher='experiments.gaussian_scene_launch' if shared else 'experiments.gaussian_memory_launch'
                    command=[sys.executable,'-u','-m',launcher,'--gpu',str(gpu)]
                    command += [job['module'],*job['args']]
                    process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    running.append(dict(job=job,process=process,gpu=gpu,shared=shared,log=stream,logpath=str(logpath)))
                    print(json.dumps(dict(start=job['name'],gpu=gpu,shared=shared,pid=process.pid)),flush=True)
            write_json(OUT/'queue_status.json',dict(pid=os.getpid(),plan='plan.json',complete=complete,
                running=[dict(name=t['job']['name'],pid=t['process'].pid,gpu=t['gpu'],shared=t['shared'],log=t['logpath']) for t in running],
                pending=[j['name'] for j in pending],failed=failed,
                state='failed_requires_diagnosis' if failed else 'running',updated_epoch=time.time()))
            if failed and not running:raise SystemExit(1)
            if pending or running:time.sleep(20)
        write_json(OUT/'queue_status.json',dict(complete=complete,running=[],pending=[],failed=[],state='all_fits_references_interventions_exported',updated_epoch=time.time()))


if __name__=='__main__':main()
