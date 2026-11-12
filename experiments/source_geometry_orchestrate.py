"""Finite four-arm queue with the existing12GiB resource envelope and pressure yield."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.source_geometry_data import *
from experiments import gaussian_scene_orchestrate as resource
resource.OUT=OUT

def done(job):
    if 'marker' in job:
        f=OUT/job['marker'];return f.exists() and read(f).get('complete',False)
    folder=branch_out(job['mode']);kind=job['kind'];seed=job['seed']
    paths=[folder/f'{kind}_{seed}'/'status.json',folder/'evaluation'/f'{kind}_{seed}.json']
    if kind=='G':paths.append(folder/'interventions'/f'G_{seed}.json')
    return all(p.exists() and read(p).get('complete',False) for p in paths)

def ready(job):
    if job['name']!='cost':return True
    return all((branch_out(mode)/f'{kind}_{SEEDS[0]}'/'status.json').exists() for mode in MODES for kind in KINDS)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(OUT/'source_state_identity.json')['complete'] and read(OUT/'parent_input_model_identity.json')['complete']
        jobs=[dict(name=f'{mode}_{kind}_{seed}',module='experiments.source_geometry_train',mode=mode,kind=kind,seed=seed,
                   args=['--mode',mode,'--kind',kind,'--seed',str(seed)]) for seed in SEEDS for mode in MODES for kind in KINDS]
        jobs[4:4]=[dict(name='diagnostics',module='experiments.source_geometry_diagnostics',args=[],marker='diagnostics_complete.json'),
                   dict(name='cost',module='experiments.source_geometry_cost',args=[],marker='cost/complete.json')]
        complete=[j['name'] for j in jobs if done(j)];pending=[j for j in jobs if not done(j)];running=[];failed=[]
        if (OUT/'queue_status.json').exists():
            for item in read(OUT/'queue_status.json').get('running',[]):
                pid=item['pid'];path=Path('/proc')/str(pid);matches=[j for j in pending if j['name']==item['name']]
                if not matches or resource.process_state(pid) in (None,'Z'):continue
                if path.stat().st_uid!=os.getuid() or b'experiments.source_geometry_' not in (path/'cmdline').read_bytes() or (path/'cwd').resolve()!=ROOT:raise RuntimeError('Ambiguous existing worker; refusing duplicate')
                job=matches[0];pending.remove(job);running.append(dict(job=job,process=resource.AdoptedWorker(pid),gpu=item['gpu'],shared=item['shared'],mode=item.get('mode','shared'),log=None,logpath=item['log']))
        (OUT/'orchestrator.pid').write_text(str(os.getpid())+'\n');idle_since={}
        while pending or running:
            for task in list(running):
                rc=task['process'].poll()
                if rc is None:continue
                if task['log'] is not None:task['log'].close()
                running.remove(task)
                if task.get('resource_yield'):pending.insert(0,task['job'])
                elif rc==0 and done(task['job']):complete.append(task['job']['name'])
                elif 'no job launched' in Path(task['logpath']).read_text()[-5000:]:pending.insert(0,task['job'])
                else:failed.append(dict(name=task['job']['name'],returncode=rc,log=task['logpath']))
            if not failed:
                states=resource.occupancy();apps=resource.device_processes();own=resource.descendants({t['process'].pid for t in running})
                occupied={t['gpu'] for t in running}
                for gpu,used,free,util in states:
                    if used<=512 and free>=40000 and util<=5:idle_since.setdefault(gpu,time.monotonic())
                    else:idle_since.pop(gpu,None)
                    affected=[t for t in running if t['gpu']==gpu and not t.get('resource_yield')];foreign=apps[gpu]-own
                    if affected and foreign and free<12288:resource.yield_worker(affected[-1],free,foreign)
                shared_cards={t['gpu'] for t in running if apps[t['gpu']]-own}
                candidates=[(g,'idle',None) for g,u,f,v in states if g not in occupied and g in idle_since and time.monotonic()-idle_since[g]>=30]
                # Share only with a single verified same-goal worker on an otherwise-owned GPU.
                for g,u,f,v in states:
                    peers=[t for t in running if t['gpu']==g]
                    if len(peers)==1 and peers[0]['job']['module']=='experiments.source_geometry_train' and f>=22528 and len(apps[g])==1 and apps[g].issubset(own):candidates.append((g,'owned_peer',next(iter(apps[g]))))
                if len(shared_cards)<2:
                    shared=[(g,'shared',None) for g,u,f,v in sorted(states,key=lambda r:r[3]) if g not in occupied and u>512 and f>=22528]
                    candidates+=shared[:2-len(shared_cards)]
                for gpu,mode,peer in candidates:
                    if len(running)>=4:break
                    available=[j for j in pending if ready(j)]
                    if not available:break
                    job=available[0];pending.remove(job);logpath=OUT/'logs'/f'{job["name"]}.log';logpath.parent.mkdir(exist_ok=True);stream=logpath.open('a',buffering=1)
                    command=[sys.executable,'-u','-m','experiments.source_geometry_launch','--gpu',str(gpu)]
                    if mode=='idle':command+=['--require-idle']
                    if peer is not None:command+=['--owned-peer',str(peer)]
                    command +=[job['module'],*job['args']]
                    proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    running.append(dict(job=job,process=proc,gpu=gpu,shared=mode!='idle',mode=mode,log=stream,logpath=str(logpath)))
                    print(json.dumps(dict(start=job['name'],gpu=gpu,mode=mode,pid=proc.pid)),flush=True)
            write_json(OUT/'queue_status.json',dict(pid=os.getpid(),complete=complete,
                running=[dict(name=t['job']['name'],pid=t['process'].pid,gpu=t['gpu'],shared=t['shared'],mode=t['mode'],log=t['logpath']) for t in running],
                pending=[j['name'] for j in pending],failed=failed,state='failed_requires_diagnosis' if failed else 'running',updated_epoch=time.time()))
            if failed and not running:raise SystemExit(1)
            if pending or running:time.sleep(20)
        write_json(OUT/'queue_status.json',dict(complete=complete,running=[],pending=[],failed=[],state='all_fits_and_diagnostics_complete',updated_epoch=time.time()))

if __name__=='__main__':main()
