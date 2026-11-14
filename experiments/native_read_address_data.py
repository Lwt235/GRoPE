"""Engineering identities/resources derived from the sole round019 contract."""
import functools,hashlib,json,os,shutil
from pathlib import Path
from experiments import native_task_writer_data as parent
from experiments.native_task_writer_data import torch,np
ROOT=parent.ROOT;DATA=parent.DATA;OUT=ROOT/'results/gaussian_memory/native_read_address_pe1';CACHE=DATA/'native_read_address_pe_v1'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_019/science_plan.md';SCIENCE_SHA='92e0880cbf45c0a8d566c605fb979a36145e4603d6449a8d6cc21761653dec3e'
ARMS=('A','P','S');SEEDS=parent.SEEDS
load=parent.load;save=parent.save;write=parent.write;read=parent.read;sha=parent.sha;journal=parent.journal;now=parent.now;cpu=parent.cpu;device=parent.device;plan=parent.plan;metas=parent.metas
def viewpath(seed,rid,q):return CACHE/'views'/str(seed)/f'{rid}_{q}.pt'
def predpath(arm,seed,rid):return CACHE/'predictions'/f'{arm}_{seed}'/f'{rid}.pt'
def final(arm,seed):return CACHE/'finals'/f'{arm}_{seed}.pt'
def latest(arm,seed):return CACHE/'resumes'/f'{arm}_{seed}.pt'
def charge(worker,phase,**counts):journal(OUT/'calls'/f'{worker}.jsonl',dict(utc=now(),phase=phase,**counts))
def schedule(seed):
    from experiments.gaussian_scene_config import schedule as original
    return original(plan(),seed)[:2000]
def schedule_sha(seed):return hashlib.sha256(json.dumps(schedule(seed)).encode()).hexdigest()
@functools.lru_cache(24)
def view(seed,rid,q):return load(viewpath(seed,rid,q))
@functools.lru_cache(256)
def sample(rid,q,c):return dict(patches=load(parent.old.patch_path(rid,q,c))['patches'],query=load(parent.old.query_path(rid,q)),hw=(448,448),condition=c)
def prepared(key,seed,arm):
    rid,q,c=key;s=dict(sample(rid,q,c));v=view(seed,rid,q);s.update({k:v[k] for k in ('extra','position','alpha')});s['kernel']=v['kernels'][arm];return device(s)
def bytes_used(root):return parent.bytes_used(root)
def resource(admission=False):
    used=bytes_used(CACHE);project=bytes_used(OUT)+sum(p.stat().st_size for p in (ROOT/'experiments').glob('native_read_address*.py'))+(ROOT/'models/native_read_address.py').stat().st_size
    mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()};seconds=sum(read(p).get('elapsed_seconds',0) for p in (OUT/'workers').glob('*.json'));free=shutil.disk_usage(DATA).free;pfree=shutil.disk_usage(ROOT).free;reasons=[]
    if used>24*2**30:reasons.append('DATA24GiB')
    if project>2**30 or pfree<2*2**30:reasons.append('project')
    if free<max(5*2**30,24*2**30-used+2*2**30):reasons.append('DATA_completion_headroom')
    if mem['MemAvailable']<(30 if admission else 12)*2**30:reasons.append('CPU_RAM')
    if seconds>6*3600-90:reasons.append('GPU6h_shutdown_reserve')
    r=dict(utc=now(),DATA_bytes=used,DATA_free_bytes=free,project_bytes=project,project_free_bytes=pfree,RAM_available_bytes=mem['MemAvailable'],GPU_worker_seconds=seconds,owned_RAM_checkpoint_bytes=0,stop_reasons=reasons);journal(OUT/'resources.jsonl',r);return r
def initialize():
    assert sha(SCIENCE)==SCIENCE_SHA;OUT.mkdir(parents=True,exist_ok=True)
    for f in ('workers','calls','preparation','evaluation','fits','logs','claims','engineering_revisions'): (OUT/f).mkdir(exist_ok=True)
    if not (OUT/'configuration.json').exists():
        files=['models/native_read_address.py','experiments/native_read_address_data.py','experiments/native_read_address_run.py','experiments/native_read_address_pool.py']
        write(OUT/'configuration.json',dict(experiment='native_read_address_pe1',science_contract=dict(path=str(SCIENCE),sha256=SCIENCE_SHA),arms=ARMS,seeds=SEEDS,steps=1000,batch=2,optimizer='Adam',learning_rate=.01,weight_decay=0,clip=5,learned_parameters=24,precision='FP32 gamma/Adam; nativeBF16; FP32 cached kernels/readouts; inherited FP64 descriptor algebra',sampling='First2000 samples of original round015 paired seed schedule',fixed_base_models=[dict(seed=s,path=str(parent.final('C_P',s)),sha256=read(parent.OUT/'fits'/f'C_P_{s}'/'complete.json')['sha256']) for s in SEEDS],source_sha256={f:sha(ROOT/f) for f in files},budget=dict(GPU_worker_hours=6,DATA_peak_GiB=24,project_GiB=1),counts=dict(formal_fits=9,formal_updates=9000,formal_native=18000,new_evaluation_native=6336,unique_cached_seed_views=1056,preparation_raster=2112,preflight_native_max=7,discarded_preflight_updates_max=1,cost_complete_native=70,cost_actual_raster=60),cache_layout='1056views x(commonextra1030FP32 +position52FP32 +alpha +three1024squareFP32kernels), W/WS/Cmu transient; nine scalar-only finals; original weights/132fields referenced. Upper22GiB plus at most8 concurrent atomic temporaries of<32MiB each, <=24GiB.',all_data_used_development=True,Fast_requested=False,requested_creation='gpt-6.1-sol/high',backend_effective='not independently exposed'))
        write(OUT/'living_docs_before.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [ROOT/'docs/research_current.md',ROOT/'docs/research_outline_alignment.md',ROOT/'docs/gaussian_memory_evidence_scope.md']})
