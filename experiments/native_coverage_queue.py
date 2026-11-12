"""Finite six-fit queue; dynamic idle/spare GPU admission and exact resumes."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from experiments.native_coverage_data import *
from experiments.gaussian_scene_config import schedule
from experiments import gaussian_scene_orchestrate as resource

SCIENCE_FILES=['experiments/native_coverage_data.py','experiments/native_coverage_source.py',
 'experiments/native_coverage_prepare.py','experiments/native_coverage_preflight.py',
 'experiments/native_coverage_train.py','models/gaussian_scene_rendered.py',
 'models/gaussian_scene_native.py','models/gaussian_scene_memory.py',
 'experiments/native_receiver_data.py','experiments/native_receiver_runtime.py',
 'experiments/gaussian_scene_config.py','experiments/gaussian_scene_preflight.py',
 'experiments/source_geometry_build.py','experiments/source_geometry_data.py',
 'experiments/gaussian_scene_data.py','experiments/gaussian_scene_build.py',
 'experiments/gaussian_scene_anysplat.py']


def freeze():
    p=plan();assert read(OUT/'preflight.json')['complete']
    if (OUT/'configuration_lock.json').exists():return
    sources=read(OUT/'input_identity.json')['existing_scenes']
    for rid,record in sources.items():assert sha256(record['path'])==record['sha256']
    caches={}
    for meta in p['added_training_catalog']:
        r=read(OUT/'query_cache'/f'{meta["id"]}.json');assert r['complete']
        assert all(sha256(path)==h for path,h in r['files'].items());caches.update(r['files'])
    # Existing current inputs/rendered values are identical inherited cache files.
    from experiments.rendered_feature_data import OUT as OLD_RENDER
    from experiments.native_receiver_view_data import OUT as OLD_VIEW
    for meta in p['original_training_catalog']+p['catalog']:
        for record in [read(OLD_RENDER/'features'/f'{meta["id"]}.json'),read(OLD_VIEW/'query_cache'/f'{meta["id"]}.json')]:
            merged=dict(record['files']);merged.update(record.get('inputs',{}))
            assert all(sha256(path)==h for path,h in merged.items());caches.update(merged)
    pretrained=sha256(p['model_path']);assert pretrained==p['model_sha256']
    write_json(OUT/'configuration_lock.json',dict(plan_sha256=sha256(OUT/'plan.json'),
        preflight_sha256=sha256(OUT/'preflight.json'),pretrained_sha256=pretrained,
        source_sha256={n:sha256(ROOT/n) for n in SCIENCE_FILES},
        schedule_sha256={str(s):hashlib.sha256(json.dumps(schedule(p,s)).encode()).hexdigest() for s in SEEDS},
        fixed_before_formal_training=True,unchanged_existing_sources=sources,
        input_cache_sha256=caches,source_recipe_sha256=sha256(OUT/'source_recipe_equivalence.json')))
    print('Formal configuration and data identities locked.',flush=True)


def jobs():return [dict(name=f'{kind}_{seed}',kind=kind,seed=seed) for seed in SEEDS for kind in KINDS]


def done(job):
    paths=[OUT/job['name']/'status.json']+[OUT/'evaluation'/f'{c}_{job["kind"]}_{job["seed"]}.json' for c in ['train','development']]
    return all(p.exists() and read(p).get('complete',False) for p in paths)


def main():
    p=plan();(OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze()
        definitions=jobs();pending=[j for j in definitions if not done(j)];running=[];failed=[]
        previous=read(OUT/'queue_status.json') if (OUT/'queue_status.json').exists() else {}
        for r in previous.get('running',[]):
            if resource.process_state(r['pid']) in (None,'Z'):continue
            proc=Path('/proc')/str(r['pid']);args=proc.joinpath('cmdline').read_bytes().split(b'\0')
            assert proc.stat().st_uid==os.getuid() and b'experiments.native_coverage_launch' in args
            matches=[j for j in pending if j['name']==r['name']];assert len(matches)==1
            job=matches[0];pending.remove(job);running.append(dict(job=job,process=resource.AdoptedWorker(r['pid']),gpu=r['gpu'],stream=None,log=r['log']))
        while pending or running:
            for r in list(running):
                rc=r['process'].poll()
                if rc is None:continue
                exit_path=OUT/'worker_exits'/f'{r["process"].pid}.json'
                if exit_path.exists():rc=read(exit_path)['returncode']
                if r['stream'] is not None:r['stream'].close()
                running.remove(r)
                if done(r['job']):print(dict(completed=r['job']['name']),flush=True)
                elif rc==75:pending.insert(0,r['job'])
                else:failed.append(dict(name=r['job']['name'],returncode=rc,log=r['log']))
            if not failed:
                states=resource.occupancy();apps=resource.device_processes();own=resource.descendants({r['process'].pid for r in running})
                occupied={r['gpu'] for r in running};shared={r['gpu'] for r in running if apps[r['gpu']]-own}
                idle=[r for r in states if r[0] not in occupied and r[1]<=512 and r[2]>=40000 and r[3]<=5]
                spare=[r for r in sorted(states,key=lambda r:(r[3],-r[2])) if r[0] not in occupied and r[1]>512 and r[2]>=17408]
                candidates=idle+spare[:max(0,2-len(shared))]
                for gpu,used,free,util in candidates:
                    if not pending or len(running)>=3:break
                    job=pending.pop(0);log=OUT/'logs'/f'{job["name"]}.log';stream=log.open('a',buffering=1)
                    cmd=[sys.executable,'-u','-m','experiments.native_coverage_launch','--gpu',str(gpu),'--cap','11',
                        '--margin','6144','--admission','17408','experiments.native_coverage_train','--kind',job['kind'],'--seed',str(job['seed'])]
                    process=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    running.append(dict(job=job,process=process,gpu=gpu,stream=stream,log=str(log)))
                    print(dict(start=job['name'],gpu=gpu,pid=process.pid),flush=True)
            write_json(OUT/'queue_status.json',dict(complete=[j['name'] for j in definitions if done(j)],
                running=[dict(name=r['job']['name'],pid=r['process'].pid,gpu=r['gpu'],log=r['log']) for r in running],
                pending=[j['name'] for j in pending],failed=failed,updated_epoch=time.time(),
                state='failed_requires_diagnosis' if failed else ('running' if running else ('waiting_for_headroom' if pending else 'complete'))))
            if failed and not running:raise SystemExit(1)
            if pending or running:time.sleep(15)


if __name__=='__main__':main()
