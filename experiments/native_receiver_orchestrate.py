"""Finite queue with conservative measured VRAM admission and exact process adoption."""
import fcntl
import math
import os
import subprocess
import sys

from experiments.native_receiver_data import *
from experiments import gaussian_scene_orchestrate as resource


SCIENCE_FILES=['models/gaussian_scene_rendered.py','models/gaussian_scene_native.py',
    'models/gaussian_scene_memory.py','experiments/native_receiver_data.py',
    'experiments/native_receiver_runtime.py','experiments/native_receiver_prepare.py',
    'experiments/native_receiver_train.py','experiments/gaussian_scene_config.py',
    'experiments/gaussian_scene_preflight.py','experiments/rendered_feature_data.py']


def freeze():
    p=plan();pre=read(OUT/'preflight.json');assert pre['complete']
    if not (OUT/'configuration_lock.json').exists():
        write_json(OUT/'configuration_lock.json',dict(plan_sha256=sha256(OUT/'plan.json'),
            preflight_sha256=sha256(OUT/'preflight.json'),pretrained_sha256=sha256(p['model_path']),
            source_sha256={n:sha256(ROOT/n) for n in SCIENCE_FILES},
            fixed_before_any_formal_training=True))
    if not (OUT/'resource_limits.json').exists():
        peak=max(r['peak_allocated_GiB'] for r in pre['records'])
        cap=max(p['resources']['pilot_cap_GiB'],math.ceil((peak+.35)*2)/2)
        write_json(OUT/'resource_limits.json',dict(complete=True,measured_pilot_peak_GiB=peak,
            memory_cap_GiB=cap,margin_MiB=6144,admission_MiB=int(cap*1024+6144),
            scope='Runtime headroom only; no change in trainable scope, precision, optimizer or scientific budget.'))


def jobs():
    j=[dict(name='preflight',module='experiments.native_receiver_prepare',args=[],marker=OUT/'preflight.json',requires=[])]
    for s in SEEDS:
        for k in KINDS:j.append(dict(name=f'{k}_{s}',kind=k,seed=s,module='experiments.native_receiver_train',
            args=['--kind',k,'--seed',str(s)],requires=['preflight']))
    # Cost runner is installed during preflight, well before the first seed completes.
    j.append(dict(name='query_cost',module='experiments.native_receiver_cost',args=[],marker=OUT/'cost.json',
        requires=[f'{k}_{SEEDS[0]}' for k in KINDS]))
    return j


def done(job):
    if 'marker' in job:return job['marker'].exists() and read(job['marker']).get('complete',False)
    k,s=job['kind'],job['seed']
    paths=[OUT/job['name']/'status.json']
    paths += [OUT/'evaluation'/f'{regime}_{cohort}_{k}_{s}.json'
        for regime,cohort in [('adapted','development'),('adapted','train'),('frozen','train')]]
    if k=='Render':paths.append(OUT/'evaluation'/f'adapted_development_{k}_{s}_shuffle.json')
    return all(p.exists() and read(p).get('complete',False) for p in paths)


def main():
    p=initialize();(OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        definitions=jobs();complete=[j['name'] for j in definitions if done(j)]
        pending=[j for j in definitions if not done(j)];running=[];failed=[]
        old=read(OUT/'queue_status.json') if (OUT/'queue_status.json').exists() else {}
        for item in old.get('running',[]):
            pid=item['pid'];path=Path('/proc')/str(pid)
            if resource.process_state(pid) in (None,'Z'):continue
            argv=path.joinpath('cmdline').read_bytes().split(b'\0')
            assert b'experiments.native_receiver_launch' in argv and path.stat().st_uid==os.getuid()
            assert path.joinpath('cwd').resolve()==ROOT
            matches=[j for j in pending if j['name']==item['name']];assert len(matches)==1
            job=matches[0];pending.remove(job)
            running.append(dict(job=job,process=resource.AdoptedWorker(pid),gpu=item['gpu'],log=None,logpath=item['log']))
        (OUT/'orchestrator.pid').write_text(str(os.getpid())+'\n')
        while pending or running:
            for task in list(running):
                rc=task['process'].poll()
                if rc is None:continue
                actual_exit=OUT/'worker_exits'/f"{task['process'].pid}.json"
                if actual_exit.exists():rc=read(actual_exit)['returncode']
                if task['log'] is not None:task['log'].close()
                running.remove(task)
                if done(task['job']):complete.append(task['job']['name'])
                elif rc==75:pending.insert(0,task['job'])
                else:failed.append(dict(name=task['job']['name'],returncode=rc,log=task['logpath']))
            if 'preflight' in complete:freeze()
            limits=read(OUT/'resource_limits.json') if (OUT/'resource_limits.json').exists() else dict(
                memory_cap_GiB=p['resources']['pilot_cap_GiB'],admission_MiB=p['resources']['pilot_admission_MiB'])
            if not failed:
                states=resource.occupancy();apps=resource.device_processes()
                own=resource.descendants({t['process'].pid for t in running});occupied={t['gpu'] for t in running}
                shared={t['gpu'] for t in running if apps[t['gpu']]-own}
                idle=[r for r in states if r[0] not in occupied and r[1]<=512 and r[2]>=40000 and r[3]<=5]
                spare=[r for r in sorted(states,key=lambda r:(r[3],-r[2])) if r[0] not in occupied and r[1]>512 and r[2]>=limits['admission_MiB']]
                candidates=idle+spare[:max(0,2-len(shared))]
                for gpu,used,free,util in candidates:
                    if len(running)>=p['resources']['max_workers']:break
                    ready=[j for j in pending if set(j['requires']).issubset(complete)]
                    if not ready:break
                    job=ready[0];pending.remove(job);logpath=OUT/'logs'/f"{job['name']}.log";stream=logpath.open('a',buffering=1)
                    cmd=[sys.executable,'-u','-m','experiments.native_receiver_launch','--gpu',str(gpu),
                        '--cap',str(limits['memory_cap_GiB']),'--margin',str(limits.get('margin_MiB',6144)),
                        '--admission',str(limits['admission_MiB']),job['module'],*job['args']]
                    process=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    running.append(dict(job=job,process=process,gpu=gpu,log=stream,logpath=str(logpath)))
                    print(dict(start=job['name'],gpu=gpu,pid=process.pid),flush=True)
            write_json(OUT/'queue_status.json',dict(pid=os.getpid(),complete=complete,
                running=[dict(name=t['job']['name'],pid=t['process'].pid,gpu=t['gpu'],log=t['logpath']) for t in running],
                pending=[j['name'] for j in pending],failed=failed,limits=limits,
                state='failed_requires_diagnosis' if failed else ('running' if running else 'waiting_for_safe_headroom'),
                updated_epoch=time.time()))
            if failed and not running:raise SystemExit(1)
            if pending or running:time.sleep(15)
        write_json(OUT/'queue_status.json',dict(complete=complete,running=[],pending=[],failed=[],
            state='all_fits_exports_and_cost_complete',updated_epoch=time.time()))


if __name__=='__main__':main()
