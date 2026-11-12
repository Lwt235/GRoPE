"""Finite resource-aware queue; never launch on a GPU occupied by another owned worker."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from experiments.gaussian_scene_data import OUT,ROOT,KINDS,SEEDS,read
from experiments.gaussian_scene_config import freeze
from experiments.gaussian_memory_incremental_data import write_json


def occupancy():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True,timeout=15)
    return [tuple(map(int,line.split(', '))) for line in rows.strip().splitlines()]


def process_state(pid):
    path=Path('/proc')/str(pid)
    if not path.exists():return None
    try:return path.joinpath('stat').read_text().split(') ',1)[1].split()[0]
    except FileNotFoundError:return None


def descendants(roots):
    parents={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:parents[int(path.name)]=int(path.joinpath('stat').read_text().split(') ',1)[1].split()[1])
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    result=set(roots)
    while True:
        expanded=result|{pid for pid,parent in parents.items() if parent in result}
        if expanded==result:return result
        result=expanded


def device_processes():
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=15)
    index={line.split(', ')[1]:int(line.split(', ')[0]) for line in gpu_rows.strip().splitlines()}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    result={gpu:set() for gpu in index.values()}
    for line in apps.splitlines():
        parts=line.split(', ')
        if len(parts)==2 and parts[0] in index:result[index[parts[0]]].add(int(parts[1]))
    return result


def yield_worker(task,free_MiB,foreign_pids):
    family=descendants({task['process'].pid});stopped=[]
    for pid in sorted(family,reverse=True):
        path=Path('/proc')/str(pid)
        if not path.exists():continue
        if path.stat().st_uid!=os.getuid():raise RuntimeError('Worker descendant ownership changed')
        try:os.kill(pid,signal.SIGTERM);stopped.append(pid)
        except ProcessLookupError:pass
    task['resource_yield']=True
    folder=OUT/'resource_yields';folder.mkdir(exist_ok=True)
    write_json(folder/f'{task["job"]["name"]}_{int(time.time())}.json',dict(
        name=task['job']['name'],gpu=task['gpu'],free_MiB=free_MiB,foreign_pids=sorted(foreign_pids),
        stopped_only_owned_pids=stopped,reason='External allocations appeared; yield our worker below12GiB free margin',
        checkpoint_preserved=True,scientific_configuration_unchanged=True))


class AdoptedWorker:
    """Observe a live original worker without restarting it or claiming child exit status."""
    def __init__(self,pid):
        self.pid=pid
        self.start=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[19]
    def poll(self):
        if process_state(self.pid) in (None,'Z'):return 0
        try:current=(Path('/proc')/str(self.pid)/'stat').read_text().split(') ',1)[1].split()[19]
        except FileNotFoundError:return 0
        return None if current==self.start else 0


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
        # A dispatcher-only restart may adopt its still-live original workers.
        if (OUT/'queue_status.json').exists():
            previous=read(OUT/'queue_status.json')
            for item in previous.get('running',[]):
                pid=item['pid'];path=Path('/proc')/str(pid)
                matching=[job for job in pending if job['name']==item['name']]
                if not matching or process_state(pid) in (None,'Z'):continue
                command=path.joinpath('cmdline').read_bytes()
                if path.stat().st_uid!=os.getuid() or b'experiments.gaussian_scene_' not in command or path.joinpath('cwd').resolve()!=ROOT:
                    raise RuntimeError('Previous worker identity is ambiguous; refusing duplicate dispatch')
                job=matching[0];pending.remove(job)
                running.append(dict(job=job,process=AdoptedWorker(pid),gpu=item['gpu'],shared=item['shared'],
                                    mode=item.get('mode','shared'),log=None,logpath=item['log']))
                print(json.dumps(dict(adopted_live_original_worker=job['name'],pid=pid)),flush=True)
        (OUT/'orchestrator.pid').write_text(str(os.getpid())+'\n')
        idle_since={}
        while pending or running:
            for task in list(running):
                rc=task['process'].poll()
                if rc is None:continue
                if task['log'] is not None:task['log'].close()
                running.remove(task)
                if task.get('resource_yield'):
                    pending.insert(0,task['job'])
                    print(json.dumps(dict(resource_yield_requeued=task['job']['name'])),flush=True)
                elif rc==0 and done(task['job']):complete.append(task['job']['name'])
                elif 'no job launched' in open(task['logpath']).read()[-5000:]:
                    pending.insert(0,task['job'])
                    print(json.dumps(dict(resource_race_no_worker_started=task['job']['name'],requeued=True)),flush=True)
                else:
                    failed.append(dict(name=task['job']['name'],returncode=rc,log=task['logpath']))
                    # Existing workers finish; do not launch further jobs until the actual error is understood.
            if not failed:
                occupied={task['gpu'] for task in running}
                states=occupancy()
                apps=device_processes();own=descendants({t['process'].pid for t in running})
                for gpu,used,free,util in states:
                    if used<=512 and free>=40000 and util<=5:idle_since.setdefault(gpu,time.monotonic())
                    else:idle_since.pop(gpu,None)
                    affected=[t for t in running if t['gpu']==gpu and not t.get('resource_yield')]
                    foreign=apps[gpu]-own
                    if affected and foreign and free<12288:
                        # Release the extra peer first; reassess after CUDA cleanup on the next poll.
                        task=next((t for t in affected if t['mode']=='owned_peer'),affected[-1])
                        yield_worker(task,free,foreign)
                foreign_shared_active=any(bool(apps[t['gpu']]-own) for t in running)
                candidates=[(gpu,'idle',None) for gpu,used,free,util in states if gpu not in occupied and gpu in idle_since and time.monotonic()-idle_since[gpu]>=30]
                for gpu,used,free,util in states:
                    peers=[t for t in running if t['gpu']==gpu]
                    if len(peers)==1 and peers[0]['job']['module']=='experiments.gaussian_scene_train' and free>=24000 and len(apps[gpu])==1 and apps[gpu].issubset(own):
                        candidates.append((gpu,'owned_peer',next(iter(apps[gpu]))))
                if not foreign_shared_active:
                    shared=[(gpu,'shared',None) for gpu,used,free,util in sorted(states,key=lambda r:r[3]) if gpu not in occupied and used>512 and free>=24000]
                    # At most one shared worker, with the existing launcher enforcing a12GiB cap.
                    candidates+=shared[:1]
                for gpu,mode,peer in candidates:
                    if not pending or len(running)>=4:break
                    job=pending.pop(0);logpath=OUT/'logs'/f'{job["name"]}.log';logpath.parent.mkdir(exist_ok=True)
                    stream=logpath.open('a',buffering=1)
                    shared=mode!='idle'
                    launcher='experiments.gaussian_scene_launch'
                    command=[sys.executable,'-u','-m',launcher,'--gpu',str(gpu)]
                    if mode=='idle':command+=['--require-idle']
                    if peer is not None:command+=['--owned-peer',str(peer)]
                    command += [job['module'],*job['args']]
                    process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    running.append(dict(job=job,process=process,gpu=gpu,shared=shared,mode=mode,log=stream,logpath=str(logpath)))
                    print(json.dumps(dict(start=job['name'],gpu=gpu,shared=shared,mode=mode,pid=process.pid)),flush=True)
            write_json(OUT/'queue_status.json',dict(pid=os.getpid(),plan='plan.json',complete=complete,
                running=[dict(name=t['job']['name'],pid=t['process'].pid,gpu=t['gpu'],shared=t['shared'],mode=t['mode'],log=t['logpath']) for t in running],
                pending=[j['name'] for j in pending],failed=failed,
                state='failed_requires_diagnosis' if failed else 'running',updated_epoch=time.time()))
            if failed and not running:raise SystemExit(1)
            if pending or running:time.sleep(20)
        write_json(OUT/'queue_status.json',dict(complete=complete,running=[],pending=[],failed=[],state='all_fits_references_interventions_exported',updated_epoch=time.time()))


if __name__=='__main__':main()
