"""Round009 identities, source-only geometry, durable atomic I/O and budget ledger."""
import os,json,time,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F
from experiments import native_coverage_data as old
from experiments.camera_only_feature_data import source_geometry,ordinary
from models.camera_only_feature_reader import view_fields
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_pose_memory_pe1';CACHE=old.DATA/'native_pose_memory_pe_v1'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_009/science_plan.md'
SCIENCE_SHA='8ad01108a9f96068b749d818772108e023cc596f35be1b595e5b68e6d4ef96ee'
KINDS=('L','P','S','G');SEEDS=(20261110,20261111,20261112);QUERIES=(2,5,8,11,1,4,7,10)
now=lambda:__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,p)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)
def load(p,device='cpu'):return torch.load(p,map_location=device,weights_only=True)
def log(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(dict(utc=now(),**x),ensure_ascii=False,allow_nan=False)+'\n')
@lru_cache(None)
def plan():return old.plan()
old.plan=plan.__wrapped__ if False else old.plan
@lru_cache(None)
def metas():return plan()['training_catalog']+plan()['catalog']
@lru_cache(None)
def added():return {m['id'] for m in plan()['added_training_catalog']}
def scene(rid):return old.CACHE/'scenes'/rid/'scene_memory.pt' if rid in added() else old.SOURCE_CACHE/rid/'scene_memory.pt'
def query(rid,q):return old.CACHE/rid/f'query_{q}.pt' if rid in added() else (old.base_query(rid,q) if q in (2,5,8,11) else old.view_query(rid,q))
def source(rid):
    from experiments.gaussian_scene_data import source_path
    return old.source_rgb_path(rid) if rid in added() else source_path(rid)
def pair(rid):
    from experiments.source_geometry_data import source_path
    return old.source_pair_path(rid) if rid in added() else source_path(rid)
def casepath(rid,q):return CACHE/'cases'/rid/f'{q}.pt'
def sceneout(rid):return CACHE/'scenes'/f'{rid}.pt'
def bytes_tree(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file()) if Path(p).exists() else 0
def resources():
    used=bytes_tree(CACHE);project=bytes_tree(OUT)+sum((ROOT/p).stat().st_size for p in ('models/native_pose_memory_reader.py','experiments/native_pose_memory_data.py','experiments/native_pose_memory_run.py') if (ROOT/p).exists())
    free=os.statvfs(CACHE if CACHE.exists() else CACHE.parent).f_bavail*os.statvfs(CACHE.parent).f_frsize
    pf=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
    rows=[read(p) for p in (OUT/'workers').glob('*.json')] if (OUT/'workers').exists() else []
    h=sum(x['seconds'] for x in rows)/3600
    reasons=[]
    if used>24*2**30 or project>2*2**30:reasons.append('written_byte_cap')
    # Remaining fixed DATA allowance and 1GiB atomic/checkpoint peak, without double counting written files.
    if free-max(0,24*2**30-used)-2**30<8*2**30 or pf-max(0,2*2**30-project)<2*2**30:reasons.append('space_headroom')
    if h>=8:reasons.append('GPU_hours')
    x=dict(DATA_bytes=used,project_bytes=project,free=free,project_free=pf,GPU_worker_hours=h,stop_reasons=reasons);log('resources.jsonl',x);return x

def geometry_from_points(means,cov,unit,sourcecam,valid):
    n=len(means);e,U=torch.linalg.eigh(cov.double());e=e.clamp_min(float(unit)**2*1e-6)
    # Reuse exact source-only neighbor/density/canonical geometry recipe; own moment covariance replaces temporary eig axes.
    from experiments.native_render_parameter_prepare import svec
    R=sourcecam[:3,:3].double();o=sourcecam[:3,3].double();p=(means.double()-o)@R/float(unit)
    dist=torch.cdist(p,p).square();dist[:,~valid]=torch.inf;dist.fill_diagonal_(torch.inf)
    ix=torch.argsort(dist,dim=-1,stable=True)[:,:31];ds=dist.gather(1,ix);ok=torch.isfinite(ds);cnt=ok.sum(-1)
    delta=torch.where(ok[...,None],p[ix]-p[:,None],0.);B=torch.einsum('nki,nkj->nij',delta,delta)/cnt.clamp_min(1)[:,None,None]+torch.eye(3,device=p.device)*1e-6
    be,bu=torch.linalg.eigh(B);logB=(bu*be.log()[:,None,:])@bu.transpose(-1,-2)
    cov=(U*e[:,None,:])@U.transpose(-1,-2);c=R.T[None]@cov@R[None]/float(unit)**2;ce,cu=torch.linalg.eigh(c);logC=(cu*ce.clamp_min(1e-12).log()[:,None,:])@cu.transpose(-1,-2)
    density=[]
    for rank in (1,8,16,31):
        v=ds.gather(1,(cnt.clamp_max(rank)-1).clamp_min(0)[:,None])[:,0];density.append(torch.where(cnt>0,torch.log1p(v.sqrt()),0.))
    density=torch.stack(density,-1)
    return dict(means=means.float(),unit=torch.as_tensor(unit,device=p.device).float(),source_camera=sourcecam.float(),valid=valid,cov_world=cov,B_world=R[None]@(B*float(unit)**2)@R.T[None],density4=density.float(),source20=torch.cat((torch.asinh(p),svec(logC),torch.ones(n,1,device=p.device),svec(logB),density),-1).float())
def mappose(pose,align):
    from streamvggt.utils.rotation import quat_to_mat
    R=quat_to_mat(pose[...,3:7]);c=-(R.transpose(-1,-2)@pose[...,:3,None])[...,0]
    A=align['rotation'];s=align['scale'];t=align['translation']
    out=torch.eye(4,device=pose.device).expand(*pose.shape[:-1],4,4).clone();out[...,:3,:3]=A@R.transpose(-1,-2);out[...,:3,3]=(A@c[...,None])[...,0]*s[...,None]+t
    return out
