"""Finite round012 engineering configuration, identities and resource ledger."""
import datetime,functools,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments import camera_only_feature_data as base
from experiments import native_coverage_data as old
from experiments.native_render_parameter_prepare import svec
from models.camera_only_feature_reader import view_fields

ROOT=old.ROOT;DATA=old.DATA
OUT=ROOT/'results/gaussian_memory/native_scene_exchange_pe1'
CACHE=DATA/'native_scene_exchange_pe_v1'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_012/science_plan.md'
SCIENCE_SHA='0ed7de647b3754adde545b9effcb966f5e98a54a39f8baf3c8357d214d22b46d'
ARMS=('R_P','R_S','R_G','U_P','U_S','U_G','L');SEEDS=(20261110,20261111,20261112)
load=base.load;write=base.write;read=base.read;sha=base.sha;save=base.tensor_save;nsave=base.numpy_save;journal=base.journal;now=base.now
old.plan=functools.lru_cache(None)(old.plan)
def plan():return old.plan()
def metas():return plan()['training_catalog']+plan()['catalog']
def gpath(rid):return base.fields_path(rid)
def vpath(rid,q):return CACHE/'views'/f'{rid}_{q}.pt'
def final(arm,seed):return CACHE/'checkpoints'/f'{arm}_{seed}'/'final.pt'
def active_formal_workers():return read(OUT/'parallel_execution.json')['active_workers']
def formal_worker_records():
    return [read(OUT/'workers'/f'{name}.json') if (OUT/'workers'/f'{name}.json').exists() else dict(name=name,actual_process_wait_returned=False) for name in active_formal_workers()]
def cpu(obj):return {k:cpu(v) for k,v in obj.items()} if isinstance(obj,dict) else [cpu(x) for x in obj] if isinstance(obj,list) else tuple(cpu(x) for x in obj) if isinstance(obj,tuple) else obj.detach().cpu() if torch.is_tensor(obj) else obj
def device(obj):return {k:device(v) for k,v in obj.items()} if isinstance(obj,dict) else obj.cuda(non_blocking=True) if torch.is_tensor(obj) else obj
def bytes_used(root):return sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file()) if root.exists() else 0
def resource(admission=False):
    used=bytes_used(CACHE);project=bytes_used(OUT)+sum(p.stat().st_size for p in (ROOT/'experiments').glob('native_scene_exchange*'))+(ROOT/'models/native_scene_exchange.py').stat().st_size
    s=os.statvfs(DATA);free=s.f_bavail*s.f_frsize;ps=os.statvfs(ROOT);pfree=ps.f_bavail*ps.f_frsize
    # Complete 14,784 depth arrays=2.429GiB; full exact checkpoints + atomic
    # transients + compact geometry/views/metadata bounded by 5GiB.
    transient=int(.35*2**30) # seven concurrent exact-resume atomic writes
    remaining=max(0,5*2**30-used);H=free-remaining-transient;reasons=[]
    if used+transient>12*2**30:reasons.append('new_DATA_12GiB')
    if project>2**30 or pfree-max(0,2**30-project)<2*2**30:reasons.append('project')
    if H<(8 if admission else 6)*2**30 or free<5*2**30:reasons.append('headroom')
    elapsed=sum(read(p).get('elapsed_seconds',0) for p in (OUT/'workers').glob('*.json')) if (OUT/'workers').exists() else 0
    if elapsed>=12*3600-300:reasons.append('GPU_12workerhours_safety_reserve')
    row=dict(utc=now(),epoch=time.time(),used_bytes=used,project_bytes=project,free_bytes=free,remaining_bytes=remaining,transient_bytes=transient,H_bytes=H,GPU_seconds=elapsed,stop_reasons=reasons)
    journal(OUT/'resources.jsonl',row);return row
def logcov(A):
    e,U=torch.linalg.eigh(A.double());return svec((U*e.clamp_min(1e-15).log()[:,None,:])@U.transpose(-1,-2)).float()

@torch.no_grad()
def scene_view(field,g,query,visibility):
    cam=query['c2w'].double();R=cam[:3,:3];u=g['unit'].double();cp=(g['means'].double()-cam[:3,3])@R
    B=R.T[None]@g['B_world'].double()@R[None]/u.square();C=R.T[None]@g['cov_world'].double()@R[None]/u.square()
    ordinary=torch.cat((torch.asinh(cp/u).float(),logcov(C),field['opacities'].float()[:,None],logcov(B),g['density4'].float(),torch.log1p(visibility).float()[:,None]),-1)
    v=view_fields(g,query['c2w'],query['K']);uv=v['common25'][:,3:5]
    # Patch center pixel i+.5 maps to official native integer index i+1.
    # Signed fractional projection is not clipped to the image frustum.
    sp=(uv+1)*16+.5;sp=sp[:,[1,0]]
    return dict(ordinary=ordinary,scene_pos=sp,key_mask=v['key_mask'],common25=v['common25'],P=v['P'],S=v['S'],G=v['G'])

@torch.no_grad()
def local_view(query):
    K=query['K'].double().clone();K[:2]/=14;u=load(gpath(query['id']))['unit'].double()
    yy,xx=torch.meshgrid(torch.arange(32,dtype=torch.float64)+.5,torch.arange(32,dtype=torch.float64)+.5,indexing='ij')
    pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3)
    p=pixels@torch.linalg.inv(K).T # z=1 unit plane, not GT/predicted geometry.
    p=p.repeat_interleave(2,0);physical=p*u;world=physical@query['c2w'][:3,:3].double().T+query['c2w'][:3,3].double()
    fake=dict(means=world.float(),unit=u.float(),source_c2w=query['c2w'][None],has_content=torch.ones(2048,dtype=torch.bool),rotations=torch.tensor([1.,0.,0.,0.]).repeat(2048,1),scales=u.float().expand(2048,3).clone(),opacities=torch.zeros(2048))
    g,_=base.source_geometry(fake);v=scene_view(dict(fake,features=torch.zeros(2048,1024)),g,query,torch.zeros(2048))
    # Fixed zero log-shape and opacity; unit Σ/B gives logΣ=0. Center/B/density
    # still obtained solely from rays. No real GS/source data enters L.
    v['ordinary'][:,3:9]=0.;v['ordinary'][:,9]=0.;v['ordinary'][:,-1]=0.
    from models.gaussian_memory_native_pe import grid
    v['scene_pos']=grid(448,448,'cpu')[0,5:].repeat_interleave(2,0).float()
    return v

def sample(rid,q,condition,arm):
    query=load(old.query_path(rid,q));v=load(vpath(rid,q));p=load(old.patch_path(rid,q,condition))['patches']
    if arm=='L':
        v=v['local'];ordinary=torch.cat((p[0].repeat_interleave(2,0),v['ordinary']),-1)
    else:ordinary=torch.cat((load(old.scene_path(rid))['features'].float(),v['ordinary']),-1)
    return dict(patches=p,ordinary=ordinary[None],scene_pos=v['scene_pos'][None],key_mask=v['key_mask'][None],position=torch.cat((v['common25'],v['P' if arm=='L' else arm[-1]]),-1)[None],target=query['target'],valid=query['valid'])

def schedule(seed):
    keys=[(m['id'],q,c) for m in plan()['training_catalog'] for q in old.QUERIES for c in old.CONDITIONS]
    rng=np.random.default_rng(seed);order=[]
    while len(order)<4000:order.extend(keys[int(i)] for i in rng.permutation(len(keys)))
    return order[:4000]

def identity_config():
    assert sha(SCIENCE)==SCIENCE_SHA
    OUT.mkdir(parents=True,exist_ok=True)
    files=['models/native_scene_exchange.py','experiments/native_scene_exchange_data.py','experiments/native_scene_exchange_run.py','experiments/native_scene_exchange_launch.py']
    write(OUT/'configuration.json',dict(science_contract=dict(path=str(SCIENCE),sha256=SCIENCE_SHA),CLI_UUID='01a1d350-d1ec-70f2-9e37-8960b2dfe7a8',requested=dict(model='gpt-6.1-sol',effort='high',service_tier='fast'),backend_effective=dict(model='unknown',effort='unknown',service_tier='unknown'),arms=ARMS,seeds=SEEDS,steps=2000,batch=2,optimizer=dict(name='AdamW',lr=1e-4,weight_decay=.01,clip=5),source_manifest_reference=str(base.OUT/'input_manifest.json'),input_manifest_sha256=sha(base.OUT/'input_manifest.json'),runtime_source_sha256={p:sha(ROOT/p) for p in files},counts=dict(fits=21,formal_updates=42000,formal_samples=84000,new_predictions=14784,inherited_Native=704),ordinary_channels=1045,source_input='raw f1024; query-coordinate asinh center3, logΣ6, opacity1, logB6, density4, log1p(actual per-slot sum of32x32 alpha contributions)1',rope='official image unchanged; analytical same-frequency signed fractional scene-center extension',data_directory=str(CACHE),budgets=dict(GPU_worker_hours=12,DATA_peak_GiB=12,project_GiB=1)))
