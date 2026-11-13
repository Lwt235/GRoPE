"""Fixed nine-fit queue; at most two workers and one external shared card."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import shutil
import signal
from experiments.native_render_parameter_data import *
from experiments.gaussian_scene_config import schedule
from experiments import gaussian_scene_orchestrate as resource
from experiments.native_render_parameter_launch import active_seconds

SCIENCE_FILES=['experiments/native_render_parameter_data.py','experiments/native_render_parameter_prepare.py','experiments/native_render_parameter_preflight.py','experiments/native_render_parameter_train.py','experiments/native_render_parameter_launch.py','experiments/native_render_parameter_queue.py','experiments/native_render_parameter_score.py','experiments/native_render_parameter_cost.py','models/gaussian_scene_parameter_pe.py','models/gaussian_scene_rendered.py','models/gaussian_scene_memory.py','models/gaussian_scene_native.py','models/gaussian_memory_native_pe.py','experiments/native_receiver_data.py','experiments/native_receiver_runtime.py','experiments/native_coverage_data.py','experiments/rendered_feature_data.py','experiments/gaussian_scene_config.py','experiments/gaussian_scene_preflight.py','third_party/streamvggt/src/streamvggt/layers/attention.py','third_party/streamvggt/src/streamvggt/layers/block.py']


def freeze():
    p=plan();assert read(OUT/'preflight.json')['complete']
    if (OUT/'configuration_lock.json').exists():return
    prior=read(parent.OUT/'configuration_lock.json');sources={};inputs={};geometry={}
    for meta in p['training_catalog']+p['catalog']:
        rid=meta['id'];r=read(OUT/'geometry'/f'{rid}.json');assert r['complete']
        for path,h in r['inputs'].items():
            assert sha256(path)==h
            expected=prior['input_cache_sha256'].get(path)
            if expected is None and path==str(scene_path(rid)):expected=prior['unchanged_existing_sources'][rid]['sha256']
            assert expected==h,(rid,path,expected,h)
            inputs[path]=h
        for path,h in r['files'].items():assert sha256(path)==h;geometry[path]=h
        sources[rid]=dict(path=r['source_path'],sha256=r['source_sha256'])
    pretrained=sha256(p['model_path']);assert pretrained==p['model_sha256']
    assert shutil.disk_usage(DATA).free>=36*2**30 and shutil.disk_usage(ROOT).free>=10*2**30 and shutil.disk_usage(RAM).free>=20*2**30
    write(OUT/'configuration_lock.json',dict(plan_sha256=sha256(OUT/'plan.json'),scientific_plan_sha256=sha256(SCIENCE_PLAN),preflight_sha256=sha256(OUT/'preflight.json'),pretrained_sha256=pretrained,source_sha256={n:sha256(ROOT/n) for n in SCIENCE_FILES},schedule_sha256={str(s):hashlib.sha256(json.dumps(schedule(p,s)).encode()).hexdigest() for s in SEEDS},input_cache_sha256=inputs,geometry_cache_sha256=geometry,source_scenes=sources,parent_configuration_sha256=sha256(parent.OUT/'configuration_lock.json'),fixed_before_formal_training=True,identity_verified_utc=now(),model_config=dict(task_path='/root/research_implementation_sol',creation_config=dict(model='gpt-6.1-sol',reasoning_effort='high',source='parent-reported explicit accepted spawn request'),followup_parameter_support=False,effective_model=None,effective_reasoning_effort=None,independently_verified_effective=False)))
    write(OUT/'geometry_complete.json',dict(complete=True,fields=44,queries=352,bytes=sum(Path(k).stat().st_size for k in geometry),source_and_content_unchanged=True,raw_labels_not_in_fields=True))
    print('Fixed parameter-PE inputs, scientific code, initialization and schedules locked.',flush=True)


def definitions():return [dict(name=f'{kind}_{seed}',kind=kind,seed=seed) for seed in SEEDS for kind in KINDS]
def done(j):return all(path.exists() and read(path).get('complete') for path in [OUT/j['name']/'status.json']+[OUT/'evaluation'/f'{cohort}_{j["kind"]}_{j["seed"]}.json' for cohort in ['train','development']])


def main():
    p=plan();(OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze();jobs=definitions();pending=[j for j in jobs if not done(j)];running=[];failed=[]
        previous=read(OUT/'queue_status.json') if (OUT/'queue_status.json').exists() else {}
        for r in previous.get('running',[]):
            if resource.process_state(r['pid']) in (None,'Z'):continue
            args=(Path('/proc')/str(r['pid'])/'cmdline').read_bytes();assert b'experiments.native_render_parameter_launch' in args
            j=next(j for j in pending if j['name']==r['name']);pending.remove(j);running.append(dict(job=j,process=resource.AdoptedWorker(r['pid']),gpu=r['gpu'],stream=None,log=r['log']))
        while pending or running:
            for r in list(running):
                rc=r['process'].poll()
                if rc is None:continue
                if r['stream'] is not None:r['stream'].close()
                running.remove(r)
                if done(r['job']):print(dict(completed=r['job']['name']),flush=True)
                elif rc==75 and active_seconds()<8*3600:pending.insert(0,r['job'])
                else:failed.append(dict(name=r['job']['name'],returncode=rc,log=r['log']))
            if not failed and active_seconds()<8*3600:
                states=resource.occupancy();apps=resource.device_processes();own=resource.descendants({r['process'].pid for r in running});occupied={r['gpu'] for r in running};shared={r['gpu'] for r in running if apps[r['gpu']]-own}
                if len(shared)>1:
                    victim=next(r for r in reversed(running) if r['gpu'] in shared)
                    sessions=[read(path) for path in (OUT/'runtime').glob('*.json') if read(path).get('parent_pid')==victim['process'].pid]
                    assert len(sessions)==1
                    pid=sessions[0]['pid'];proc=Path('/proc')/str(pid);assert proc.stat().st_uid==os.getuid()
                    os.kill(pid,signal.SIGTERM)
                    write(OUT/'resource_yields'/f'{time.time_ns()}.json',dict(reason='At most one external shared card',gpu=victim['gpu'],stopped_only_owned_pid=pid,complete_checkpoint_requested=True))
                idle=[(v,14,20480) for v in states if v[0] not in occupied and v[1]<=512 and v[2]>=20480 and v[3]<=5]
                spare=[(v,12,18432) for v in sorted(states,key=lambda v:(v[3],-v[2])) if v[0] not in occupied and v[1]>512 and v[2]>=18432]
                candidates=idle+spare[:max(0,1-len(shared))]
                for (gpu,used,free,util),cap,admission in candidates:
                    if not pending or len(running)>=2:break
                    j=pending.pop(0);log=OUT/'logs'/f'{j["name"]}.log';stream=log.open('a',buffering=1);cmd=[sys.executable,'-u','-m','experiments.native_render_parameter_launch','--gpu',str(gpu),'--cap',str(cap),'--admission',str(admission),'experiments.native_render_parameter_train','--kind',j['kind'],'--seed',str(j['seed'])];proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);running.append(dict(job=j,process=proc,gpu=gpu,stream=stream,log=str(log)));print(dict(start=j['name'],gpu=gpu,pid=proc.pid,cap_GiB=cap),flush=True)
            write(OUT/'queue_status.json',dict(complete=[j['name'] for j in jobs if done(j)],running=[dict(name=r['job']['name'],pid=r['process'].pid,gpu=r['gpu'],log=r['log']) for r in running],pending=[j['name'] for j in pending],failed=failed,GPU_worker_seconds=active_seconds(),updated_epoch=time.time(),state='failed' if failed else 'running' if running else 'complete' if not pending else 'waiting'))
            if failed and not running:raise SystemExit(1)
            if pending and not running and active_seconds()>=8*3600:raise RuntimeError('Hard GPU worker budget reached; preserve pending fits')
            if pending or running:time.sleep(15)


if __name__=='__main__':main()
