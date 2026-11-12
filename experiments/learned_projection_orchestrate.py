"""Nine finite fits under the established spare-memory GPU envelope."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.learned_projection_data import *
from experiments import gaussian_scene_orchestrate as resource

resource.OUT = OUT


def done(job):
    a=OUT/job['name']/'status.json'; b=OUT/'evaluation'/f"{job['name']}.json"
    return all(x.exists() and read(x).get('complete',False) for x in (a,b))


def main():
    with (OUT/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(OUT/'geometry_complete.json')['complete'] and read(OUT/'preflight.json')['complete']
        jobs=[dict(name=f'{kind}_{seed}',kind=kind,seed=seed) for seed in SEEDS for kind in KINDS]
        complete=[j['name'] for j in jobs if done(j)];pending=[j for j in jobs if not done(j)]
        running=[];failed=[]
        if (OUT/'queue_status.json').exists():
            for item in read(OUT/'queue_status.json').get('running',[]):
                pid=item['pid'];path=Path('/proc')/str(pid);matching=[j for j in pending if j['name']==item['name']]
                if not matching or resource.process_state(pid) in (None,'Z'):continue
                assert path.stat().st_uid==os.getuid() and b'experiments.learned_projection_' in (path/'cmdline').read_bytes()
                assert (path/'cwd').resolve()==ROOT
                job=matching[0];pending.remove(job)
                running.append(dict(job=job,process=resource.AdoptedWorker(pid),gpu=item['gpu'],shared=item['shared'],
                    mode=item.get('mode','shared' if item['shared'] else 'idle'),staged=item.get('staged',False),log=None,logpath=item['log']))
        (OUT/'orchestrator.pid').write_text(str(os.getpid())+'\n')
        while pending or running:
            for task in list(running):
                rc=task['process'].poll()
                if rc is None:continue
                if task['log'] is not None:task['log'].close()
                running.remove(task)
                if task.get('resource_yield') or rc==75:pending.insert(0,task['job'])
                elif rc==0 and done(task['job']):complete.append(task['job']['name'])
                elif 'no job launched' in Path(task['logpath']).read_text()[-5000:]:pending.insert(0,task['job'])
                else:failed.append(dict(name=task['job']['name'],returncode=rc,log=task['logpath']))
            if not failed:
                profile_path=OUT/'memory_staging/dispatch.json'
                profile=read(profile_path) if profile_path.exists() else {}
                staged=profile.get('validated',False)
                if staged:assert read(OUT/'memory_staging/validation.json')['complete']
                admission=13312 if staged else 22528
                states=resource.occupancy();apps=resource.device_processes()
                own=resource.descendants({t['process'].pid for t in running});occupied={t['gpu'] for t in running}
                for gpu,used,free,util in states:
                    # Staged launchers enforce their measured6GiB margin every2seconds.
                    affected=[t for t in running if t['gpu']==gpu and not t.get('resource_yield') and not t.get('staged')]
                    foreign=apps[gpu]-own
                    if affected and foreign and free<12288:
                        resource.yield_worker(next((t for t in affected if t.get('mode')=='owned_peer'),affected[-1]),free,foreign)
                candidates=[(g,'idle',None) for g,u,f,v in states if g not in occupied and u<=512 and f>=40000 and v<=5]
                for gpu,used,free,util in states:
                    peers=[t for t in running if t['gpu']==gpu and not t.get('resource_yield')]
                    if len(peers)==1 and free>=admission and len(apps[gpu])==1 and apps[gpu].issubset(own):
                        candidates.append((gpu,'owned_peer',next(iter(apps[gpu]))))
                shared={t['gpu'] for t in running if apps[t['gpu']]-own}
                if len(shared)<2:
                    candidates += [(g,'shared',None) for g,u,f,v in sorted(states,key=lambda x:x[3])
                        if g not in occupied and u>512 and f>=admission][:2-len(shared)]
                if profile.get('validating',False):candidates=[]
                for gpu,mode,peer in candidates:
                    if not pending or len(running)>=4:break
                    job=pending.pop(0);logpath=OUT/'logs'/f"{job['name']}.log";logpath.parent.mkdir(exist_ok=True)
                    stream=logpath.open('a',buffering=1)
                    launcher='experiments.learned_projection_stream_launch' if staged else 'experiments.learned_projection_launch'
                    module='experiments.learned_projection_stream_train' if staged else 'experiments.learned_projection_train'
                    command=[sys.executable,'-u','-m',launcher,'--gpu',str(gpu)]
                    if mode=='idle' and not staged:command+=['--require-idle']
                    if peer is not None and not staged:command+=['--owned-peer',str(peer)]
                    command+=[module,'--kind',job['kind'],'--seed',str(job['seed'])]
                    proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    running.append(dict(job=job,process=proc,gpu=gpu,shared=mode!='idle',mode=mode,staged=staged,log=stream,logpath=str(logpath)))
                    print(dict(start=job['name'],gpu=gpu,mode=mode,staged=staged,pid=proc.pid),flush=True)
            write_json(OUT/'queue_status.json',dict(pid=os.getpid(),complete=complete,
                running=[dict(name=t['job']['name'],pid=t['process'].pid,gpu=t['gpu'],shared=t['shared'],mode=t.get('mode'),staged=t.get('staged',False),log=t['logpath']) for t in running],
                pending=[j['name'] for j in pending],failed=failed,
                state='failed_requires_diagnosis' if failed else 'running',updated_epoch=time.time()))
            if failed and not running:raise SystemExit(1)
            if pending or running:time.sleep(20)
        write_json(OUT/'queue_status.json',dict(complete=complete,running=[],pending=[],failed=[],
            state='all_nine_fits_and_predictions_complete',updated_epoch=time.time()))


if __name__=='__main__':main()
