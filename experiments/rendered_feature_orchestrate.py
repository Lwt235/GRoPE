"""Finite dependency-aware GPU queue; at most two foreign-shared cards."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.rendered_feature_data import *
from experiments import gaussian_scene_orchestrate as resource


def jobs():
    result=[dict(name='preflight',module='experiments.rendered_feature_prepare',args=['--pilot'],
                 marker='preflight.json',requires=[]),
            dict(name='features',module='experiments.rendered_feature_prepare',args=[],
                 marker='features_complete.json',requires=['preflight'])]
    result += [dict(name=f'{kind}_{seed}',kind=kind,seed=seed,module='experiments.rendered_feature_train',
        args=['--kind',kind,'--seed',str(seed)],requires=['features']) for seed in SEEDS for kind in KINDS]
    result.append(dict(name='query_cost',module='experiments.rendered_feature_cost',args=[],marker='cost.json',
        requires=[f'{kind}_{SEEDS[0]}' for kind in KINDS]))
    return result


def done(job):
    if 'marker' in job:
        path=OUT/job['marker'];return path.exists() and read(path).get('complete',False)
    paths=[OUT/job['name']/'status.json',OUT/'evaluation'/f"{job['name']}.json"]
    if job['kind']=='Render':paths.append(OUT/'interventions'/f"{job['name']}.json")
    return all(p.exists() and read(p).get('complete',False) for p in paths)


def main():
    initialize_plan();(OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        definitions=jobs();complete=[j['name'] for j in definitions if done(j)]
        pending=[j for j in definitions if not done(j)];running=[];failed=[]
        state_path=OUT/'queue_status.json'
        if state_path.exists():
            for item in read(state_path).get('running',[]):
                pid=item['pid'];path=Path('/proc')/str(pid)
                matches=[j for j in pending if j['name']==item['name']]
                if not matches or resource.process_state(pid) in (None,'Z'):continue
                assert path.stat().st_uid==os.getuid() and b'experiments.rendered_feature_launch' in (path/'cmdline').read_bytes()
                assert (path/'cwd').resolve()==ROOT
                job=matches[0];pending.remove(job)
                running.append(dict(job=job,process=resource.AdoptedWorker(pid),gpu=item['gpu'],
                    log=None,logpath=item['log']))
        (OUT/'orchestrator.pid').write_text(str(os.getpid())+'\n')
        while pending or running:
            for task in list(running):
                rc=task['process'].poll()
                if rc is None:continue
                if task['log'] is not None:task['log'].close()
                running.remove(task)
                if done(task['job']):complete.append(task['job']['name'])
                elif rc==75 or 'no job launched' in Path(task['logpath']).read_text()[-3000:]:
                    pending.insert(0,task['job'])
                else:failed.append(dict(name=task['job']['name'],returncode=rc,log=task['logpath']))
            if not failed:
                states=resource.occupancy();apps=resource.device_processes()
                own=resource.descendants({t['process'].pid for t in running})
                occupied={t['gpu'] for t in running}
                shared={t['gpu'] for t in running if apps[t['gpu']]-own}
                idle=[r for r in states if r[0] not in occupied and r[1]<=512 and r[2]>=40000 and r[3]<=5]
                foreign=[r for r in sorted(states,key=lambda x:(x[3],-x[2]))
                    if r[0] not in occupied and r[1]>512 and r[2]>=13312]
                candidates=idle+foreign[:max(0,2-len(shared))]
                for gpu,used,free,util in candidates:
                    if len(running)>=4:break
                    eligible=[j for j in pending if set(j['requires']).issubset(complete)]
                    if not eligible:break
                    # Cost code is installed before the first paired seed completes.
                    job=eligible[0];pending.remove(job)
                    logfile=OUT/'logs'/f"{job['name']}.log";stream=logfile.open('a',buffering=1)
                    command=[sys.executable,'-u','-m','experiments.rendered_feature_launch',
                        '--gpu',str(gpu),job['module'],*job['args']]
                    process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    running.append(dict(job=job,process=process,gpu=gpu,log=stream,logpath=str(logfile)))
                    print(dict(start=job['name'],gpu=gpu,pid=process.pid),flush=True)
            write_json(state_path,dict(pid=os.getpid(),complete=complete,
                running=[dict(name=t['job']['name'],pid=t['process'].pid,gpu=t['gpu'],log=t['logpath']) for t in running],
                pending=[j['name'] for j in pending],failed=failed,
                state='failed_requires_diagnosis' if failed else ('running' if running else 'waiting_for_spare_memory'),
                updated_epoch=time.time()))
            if failed and not running:raise SystemExit(1)
            if pending or running:time.sleep(15)
        write_json(state_path,dict(complete=complete,running=[],pending=[],failed=[],
            state='all_fits_exports_interventions_and_cost_complete',updated_epoch=time.time()))


if __name__=='__main__':main()
