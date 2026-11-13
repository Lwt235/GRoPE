"""Round006 immutable input references, geometry, persistence and bounded accounting."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from experiments import native_coverage_data as old
from experiments.native_render_parameter_prepare import rotation,svec
from models.camera_only_feature_reader import anchor_from_composite,view_fields

ROOT=old.ROOT
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_006'
PLAN=SCIENCE/'plan.json'
OUT=ROOT/'results/gaussian_memory/camera_only_feature_pe1'
CACHE=Path(json.loads(PLAN.read_text())['data_directory'])
ARMS=('P','S','G');SEEDS=(20261110,20261111,20261112)
STOP=False
LIMITS=dict(formal_updates=18000,pilot_updates=6,training_reader=36000,training_source=36000,pilot_reader=12,pilot_source=12,native=4650,reader=3500,source=600,raster=400)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return old.sha256(p)
def load(p,device='cpu'):return torch.load(p,map_location=device,weights_only=True)
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def journal(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',buffering=1) as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
def tensor_save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    torch.save(obj,t)
    with t.open('rb') as f:os.fsync(f.fileno())
    t.replace(p)
def numpy_save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with t.open('wb') as f:np.save(f,obj,allow_pickle=False);f.flush();os.fsync(f.fileno())
    t.replace(p)
def contract():return read(SCIENCE/'data_contract.json')
def metas():return contract()['training_catalog']+contract()['development_catalog']
def fields_path(rid):return CACHE/'geometry'/f'{rid}.pt'
def view_path(rid,q):return CACHE/'views'/f'{rid}_{q}.pt'
def final_path(arm,seed):return CACHE/'finals'/f'{arm}_{seed}.pt'
def memory_path(arm,seed,rid,permuted=False):return CACHE/'states'/f'{arm}_{seed}'/f'{rid}{"_permuted" if permuted else ""}.pt'
def to_device(obj,device='cuda'):
    if torch.is_tensor(obj):return obj.to(device)
    if isinstance(obj,dict):return {k:to_device(v,device) for k,v in obj.items()}
    return obj

def charge(purpose,**amounts):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'budget.lock').open('a') as g:
        fcntl.flock(g,fcntl.LOCK_EX);p=OUT/'budget.json';b=read(p) if p.exists() else dict(counts={k:0 for k in LIMITS},by_purpose={})
        for k,n in amounts.items():assert b['counts'][k]+n<=LIMITS[k],(k,b['counts'],n)
        for k,n in amounts.items():b['counts'][k]+=n;b['by_purpose'].setdefault(purpose,{})[k]=b['by_purpose'].get(purpose,{}).get(k,0)+n
        b['updated_utc']=now();write(p,b)

def diskbytes(root):
    total=0
    for p in root.rglob('*') if root.exists() else []:
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:pass
    return total

def resources(previous=None,admission=False):
    used=diskbytes(CACHE);project=diskbytes(OUT)
    s=os.statvfs(CACHE if CACHE.exists() else CACHE.parent);free=s.f_bavail*s.f_frsize
    ps=os.statvfs(ROOT);pfree=ps.f_bavail*ps.f_frsize
    # Credit each already written permanent category once; reserve fixed finals+metadata
    # and a bounded two-worker atomic checkpoint/prediction peak. Never credit temp files.
    estimates={'features':3312*1024*1024*4,'depths':4368*210*210*4,'states':int(1.1*2**30),'finals':int(.18*2**30),'geometry':int(.05*2**30),'views':int(.3*2**30)}
    remaining=0;cats={}
    for key,size in estimates.items():
        materialized=sum(p.stat().st_size for p in (CACHE/key).rglob('*') if p.is_file() and '.tmp' not in p.name) if (CACHE/key).exists() else 0
        need=max(0,size-materialized);remaining+=need;cats[key]=dict(materialized=materialized,remaining=need)
    transient=int(.2*2**30);H=free-remaining-transient;reasons=[]
    if H<(8 if admission else 6)*2**30 or free<5*2**30:reasons.append('DATA_headroom')
    if pfree-max(0,2**30-project)<2*2**30:reasons.append('project_headroom')
    if used+transient>18*2**30:reasons.append('DATA_peak_budget')
    if project>2**30:reasons.append('project_budget')
    unexplained=0
    if previous and time.time()-previous['epoch']>=29:
        unexplained=max(0,previous['free']-free-max(0,used-previous['used']))
        if H<8*2**30 and unexplained>=.5*2**30:reasons.append('unexplained_drop_tight_band')
    workers=[]
    for p in (OUT/'workers').glob('*.json') if (OUT/'workers').exists() else []:
        r=read(p);workers.append(r.get('elapsed_seconds',0))
    if sum(workers)>6*3600:reasons.append('GPU_worker_hours')
    row=dict(utc=now(),epoch=time.time(),free=free,H=H,used=used,remaining=remaining,transient=transient,categories=cats,project_bytes=project,project_free=pfree,unexplained_drop=unexplained,stop_reasons=reasons)
    journal(OUT/'resources/ledger.jsonl',row);write(OUT/'resources'/f'current_{os.getpid()}.json',row)
    return row

def source_geometry(field):
    start=time.monotonic();R0=field['source_c2w'][0,:3,:3].double();o0=field['source_c2w'][0,:3,3].double();u=field['unit'].double()
    p=(field['means'].double()-o0)@R0/u;valid=field['has_content'];N=len(p)
    rot=rotation(field['rotations']);cov=(rot*field['scales'].double().square()[:,None,:])@rot.transpose(-1,-2)
    canonical=R0.T[None]@cov@R0[None]/u.square()
    eig,U=torch.linalg.eigh(canonical);logS=(U*eig.log()[:,None,:])@U.transpose(-1,-2)
    dist=(p[:,None]-p[None]).square().sum(-1);dist[:,~valid]=torch.inf;dist.fill_diagonal_(torch.inf)
    indices=torch.argsort(dist,dim=-1,stable=True)[:,:31];ranks=torch.gather(dist,1,indices);ok=torch.isfinite(ranks)
    delta=p[indices]-p[:,None];delta=torch.where(ok[:,:,None],delta,0.)
    B=torch.einsum('nki,nkj->nij',delta,delta)/ok.sum(-1).clamp_min(1)[:,None,None]+1e-6*torch.eye(3,dtype=torch.float64)
    be,bU=torch.linalg.eigh(B);logB=(bU*be.log()[:,None,:])@bU.transpose(-1,-2)
    density=torch.zeros(N,4,dtype=torch.float64)
    count=ok.sum(-1)
    for col,rank in enumerate((1,8,16,31)):
        index=(count.clamp_max(rank)-1).clamp_min(0)
        vals=torch.gather(ranks,1,index[:,None])[:,0]
        density[:,col]=torch.where(count>0,torch.log1p(vals.sqrt()),0.)
    result=dict(means=field['means'],unit=field['unit'],source_camera=field['source_c2w'][0],valid=valid,cov_world=cov,B_world=R0[None]@(B*u.square())@R0.T[None],density4=density.float(),source20=torch.cat((torch.asinh(p),svec(logS),field['opacities'].double()[:,None],svec(logB),density),-1).float())
    assert result['source20'].shape==(N,20) and all(torch.isfinite(v).all() for v in result.values())
    return result,dict(seconds=time.monotonic()-start,valid_centers=int(valid.sum()),neighbor_counts=[int(count.min()),int(count.max())],unit=float(u),source_only=True,minimum_center_B_eigenvalue=float(be.min()),extra_center_information_shared=True)

def ordinary(field,g):return torch.cat((F.layer_norm(field['features'].float(),(1024,)),g['source20'].float()),-1)
def cached_anchor(rid,q):
    r=load(old.feature_path(rid,q));return anchor_from_composite(r['render'],r['effective_alpha'])

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);assert not resources(admission=True)['stop_reasons']
    expected={'analysis_handoff.json':'789f034c0823bc7ba61d37865f29b4d6389411048f3e0f3d6a8145313af00ffc','plan.json':'61ed3c9402d342e4aeacaff69c2961a76082528948fdfdf9ff129f78f53ec730','data_contract.json':'903242ba522ce316ed2752ef505e8f76f9df6f117c63fcd76892940d358fec01'}
    for name,h in expected.items():assert sha(SCIENCE/name)==h
    assert sha(ROOT/'docs/camera_only_feature_pe_execution_plan.md')=='8de9f1dcd22c29a911d305edd502226a92a4c578d27cc2246f285fadd49f34ba'
    inherited={}
    for marker in (ROOT/'results/gaussian_memory/native_memory_context_pe1/preparation').glob('*.json'):
        prior=read(marker);inherited.update(prior.get('inputs',{}))
    manifest={};alpha_stats=[]
    for meta in metas():
        rid=meta['id'];marker=OUT/'preparation'/f'{rid}.json'
        if marker.exists():manifest.update(read(marker)['inputs']);continue
        field=load(old.scene_path(rid));g,stats=source_geometry(field);tensor_save(fields_path(rid),g)
        paths=[old.scene_path(rid)]
        for q in contract()['evaluation_queries']:
            query=load(old.query_path(rid,q));v=view_fields(g,query['c2w'],query['K']);tensor_save(view_path(rid,q),v)
            b,a=cached_anchor(rid,q);assert torch.isfinite(b).all()
            alpha_stats.append(dict(id=rid,query=q,zero_fraction=float((a<=1e-8).float().mean()),zero_all=bool((a<=1e-8).all()),attention_valid_keys=int(v['key_mask'].sum())))
            paths.extend([old.query_path(rid,q),old.feature_path(rid,q),old.patch_path(rid,q,'full'),old.patch_path(rid,q,'missing')])
        ids={}
        for p in paths:
            p=Path(p);stat=p.stat();h=inherited.get(str(p))
            assert h is not None,('missing_authoritative_input',str(p))
            ids[str(p)]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256_inherited=h,authority='native_memory_context_pe1/preparation inputs; no redundant full cache hash')
        manifest.update(ids);write(marker,dict(complete=True,id=rid,geometry_path=str(fields_path(rid)),geometry_sha256=sha(fields_path(rid)),statistics=stats,inputs=ids))
        print(dict(prepared=rid,seconds=stats['seconds']),flush=True)
    write(OUT/'input_manifest.json',dict(inputs=manifest,normal_query_cases=352,source_ids=[0,3,6,9],all_used_development=True,pretrained_model=dict(path=old.plan()['model_path'],bytes=Path(old.plan()['model_path']).stat().st_size,identity_policy='existing native_memory_context_pe1/configuration_lock pretrained_sha256 inherited; no large hash',sha256_inherited=read(ROOT/'results/gaussian_memory/native_memory_context_pe1/configuration_lock.json')['pretrained_sha256'])))
    if alpha_stats:write(OUT/'coverage_statistics.json',dict(cases=alpha_stats,all_zero_queries=[x for x in alpha_stats if x['zero_all']]))
    write(OUT/'configuration_lock.json',dict(utc=now(),science_inputs=expected,execution_sha256=sha(ROOT/'docs/camera_only_feature_pe_execution_plan.md'),runtime_sources={n:sha(ROOT/n) for n in ['models/camera_only_feature_reader.py','experiments/camera_only_feature_data.py','experiments/camera_only_feature_run.py']},input_manifest_sha256=sha(OUT/'input_manifest.json'),ordinary_shape_shared=True,query_RGB_reader_input=False,requested_model='gpt-6.1-sol',requested_effort='high',requested_service_tier='fast',CLI_UUID='01a1d350-d1ec-70f2-9e37-8960b2dfe7a8',backend_effective_tier='unknown'))
    write(OUT/'preparation_complete.json',dict(complete=True,scenes=44,views=352,geometry_source_only=True,zero_new_source_or_raster_forwards=True))

if __name__=='__main__':
    torch.set_num_threads(4);prepare()
