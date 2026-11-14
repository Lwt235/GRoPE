"""Derived engineering configuration for the sole round015 scientific contract."""
import copy,datetime,functools,hashlib,json,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from experiments import native_coverage_data as old
from experiments import camera_only_feature_data as io
ROOT=old.ROOT;DATA=old.DATA;OUT=ROOT/'results/gaussian_memory/native_task_writer_pe1';CACHE=DATA/'native_task_writer_pe_v1'
RAM=Path('/dev/shm')/f'liwt_native_task_writer_pe1_{os.getuid()}'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_015/science_plan.md';SCIENCE_SHA='a0b11609c6feca7ae910016a2233f7d364eec034c2eb0de9dcc61b8ad9a752fd'
ARMS=tuple(f'{r}_{k}' for r in ('C','J') for k in ('P','S','G'));SEEDS=old.SEEDS
load=io.load;write=io.write;read=io.read;sha=io.sha;save=io.tensor_save;journal=io.journal;now=io.now
old.plan=functools.lru_cache(None)(old.plan)
def plan():return old.plan()
def metas():return plan()['training_catalog']+plan()['catalog']
def cpu(x):
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
    return x.detach().cpu() if torch.is_tensor(x) else x

def device(x):
    if isinstance(x,dict):return {k:device(v) for k,v in x.items()}
    return x.cuda() if torch.is_tensor(x) else x

def final(arm,seed):return CACHE/'finals'/f'{arm}_{seed}.pt'
def latest(arm,seed):return RAM/'resumes'/f'{arm}_{seed}.pt'
def fieldpath(arm,seed,rid):return CACHE/'fields'/f'{arm}_{seed}'/f'{rid}.pt'
def bytes_used(root):return sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file()) if root.exists() else 0

def resource(admission=False):
    disk_used=bytes_used(CACHE);ram_used=bytes_used(RAM);used=disk_used+ram_used;project=bytes_used(OUT)+sum(p.stat().st_size for p in (ROOT/'experiments').glob('native_task_writer*'))+(ROOT/'models/native_task_writer.py').stat().st_size
    free=shutil.disk_usage(DATA).free;pfree=shutil.disk_usage(ROOT).free;mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    seconds=sum(read(p).get('elapsed_seconds',0.) for p in (OUT/'workers').glob('*.json')) if (OUT/'workers').exists() else 0.;reasons=[]
    if used>80*2**30:reasons.append('DATA_peak80GiB')
    if project>2**30 or pfree<2*2**30:reasons.append('project')
    if free<max(10*2**30,80*2**30-used+8*2**30):reasons.append('DATA_completion_headroom')
    if mem['MemAvailable']<(30 if admission else 12)*2**30:reasons.append('CPU_RAM')
    if RAM.exists() and shutil.disk_usage(RAM).free<10*2**30:reasons.append('RAM_checkpoint_filesystem_headroom')
    if seconds>18*3600-120:reasons.append('GPU18hours_reserved_shutdown')
    row=dict(utc=now(),DATA_bytes=used,DATA_disk_bytes=disk_used,owned_RAM_checkpoint_bytes=ram_used,DATA_free_bytes=free,project_bytes=project,project_free_bytes=pfree,RAM_available_bytes=mem['MemAvailable'],GPU_worker_seconds=seconds,stop_reasons=reasons)
    journal(OUT/'resources.jsonl',row);return row

def charge(worker,phase,**counts):journal(OUT/'calls'/f'{worker}.jsonl',dict(utc=now(),phase=phase,**counts))

@functools.lru_cache(6)
def basefield(rid):return load(old.scene_path(rid))
@functools.lru_cache(24)
def sample(rid,q,c):return dict(patches=load(old.patch_path(rid,q,c))['patches'],query=load(old.query_path(rid,q)),hw=(448,448),condition=c)

def initialize():
    assert sha(SCIENCE)==SCIENCE_SHA;OUT.mkdir(exist_ok=True,parents=True)
    files=['models/native_task_writer.py','experiments/native_task_writer_data.py','experiments/native_task_writer_run.py','experiments/native_task_writer_manage.py']
    p=copy.deepcopy(plan());p.update(experiment='native_task_writer_pe1',arms=ARMS,seeds=SEEDS,steps=2000,effective_batch=2,scientific_contract=dict(path=str(SCIENCE),sha256=SCIENCE_SHA),new_confirmation=False,requested_model='gpt-6.1-sol',requested_effort='high',Fast_requested=False,backend_effective='not independently exposed',budget=dict(GPU_worker_hours=18,DATA_peak_GiB=80,project_GiB=1),counts=dict(formal_fits=18,updates=36000,training_samples=72000,evaluation_predictions=12672,permanent_fields=792,preflight_updates_max=6,cost_native_queries=120,cost_formations=60),source_sha256={p:sha(ROOT/p) for p in files})
    write(OUT/'configuration.json',p)
