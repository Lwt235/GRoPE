"""Round026 fixed identities and engineering I/O; no science-plan mutation."""
import os,json,hashlib,time,datetime,functools
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from experiments import native_coverage_data as old
from experiments.native_pose_memory_data import source,pair
from experiments.camera_only_feature_data import fields_path

ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_track_memory_pe1';CACHE=old.DATA/'native_track_memory_pe_v1'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_026/science_plan.md';SCIENCE_SHA='b57eecaf79e94b756aa4541d15310cbaac97c76f0b09edb61ebffb985812a518'
KINDS=('P','S','G','L');SEEDS=(20261110,20261111,20261112);QUERIES=(2,5,8,11,1,4,7,10);SOURCES=(0,3,6,9)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');torch.save(x,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    tmp.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def cpu(x):return x.detach().cpu() if torch.is_tensor(x) else {k:cpu(v) for k,v in x.items()} if isinstance(x,dict) else [cpu(v) for v in x] if isinstance(x,list) else tuple(cpu(v) for v in x) if isinstance(x,tuple) else x
def device(x):return x.cuda() if torch.is_tensor(x) else {k:device(v) for k,v in x.items()} if isinstance(x,dict) else [device(v) for v in x] if isinstance(x,list) else tuple(device(v) for v in x) if isinstance(x,tuple) else x
def journal(name,**kw):
    p=OUT/'calls'/f'{name}.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(dict(utc=now(),**kw))+'\n');f.flush()
plan=functools.lru_cache(None)(old.plan)
def metas():return plan()['training_catalog']+plan()['catalog']
def case(rid,a,q):return CACHE/'cases'/rid/f'{a}_{q}.pt'
def anchorfile(rid,a):return CACHE/'anchors'/rid/f'{a}.pt'
def point_grid():
    y,x=torch.meshgrid((torch.arange(8).float()+.5)*56-.5,(torch.arange(8).float()+.5)*56-.5,indexing='ij');return torch.stack((x,y),-1).reshape(64,2)
def source210(payload,ix):
    def resize(t):return torch.from_numpy(np.asarray(Image.fromarray(t.numpy()).resize((210,210),Image.Resampling.NEAREST)).copy())
    K=payload['K'][ix].double().clone();K[:2]*=210/448
    return dict(depth=resize(payload['depth_sensor'][ix]).double(),valid=resize(payload['common_valid'][ix]).bool(),K=K,cam=payload['c2w'][ix].double())
def qgeom(payload):return dict(depth=payload['target'].double(),valid=payload['valid'],K=payload['K_metric'].double(),cam=payload['c2w'].double())
def labels(sourcegeom,querygeom):
    from experiments.native_crossview_content import project,unproject,sample
    a,b=device(sourcegeom),device(querygeom);p=(point_grid().cuda().double()+.5)*210/448;da,va,_,_=sample(p,a);world=unproject(p,da,a);pb,zb=project(world,b);db,vb,qb,inb=sample(pb,b);back=unproject(qb,db,b);pba,zba=project(back,a);dar,vr,qa,ina=sample(pba,a);pbr,zbr=project(unproject(qa,dar,a),b)
    fd=(zb-db).abs();rd=(zba-dar).abs();ra=(pba-p).norm(dim=-1);rb=(pbr-qb).norm(dim=-1)
    flags=[('source_valid',va),('query_front',zb>0),('query_in_image',inb),('query_valid',vb),('forward_depth',fd<=torch.maximum(db*.02,db.new_tensor(.02))),('reverse_anchor_in_image',ina),('reverse_anchor_valid',vr),('reverse_depth',rd<=torch.maximum(dar*.02,dar.new_tensor(.02))),('reprojection_anchor',ra<=1),('cycle_query_front_in_image',(zbr>0)&torch.isfinite(pbr).all(-1)&(pbr>=0).all(-1)&(pbr<210).all(-1)),('reprojection_query',rb<=1)]
    valid=torch.ones(64,device='cuda',dtype=torch.bool);stages=[]
    for name,flag in flags:before=int(valid.sum());valid&=flag;stages.append(dict(name=name,remaining=int(valid.sum()),rejected=before-int(valid.sum())))
    return cpu(dict(query_points=point_grid().cuda(),target=pb*448/210-.5,valid=valid,source_world=world,residuals=torch.stack((fd,rd,ra,rb),-1),stage_counts=stages))
def source_patches(rid):
    # Reuse raw source DINO targets, not the newly fitted C/J field itself.
    p=old.DATA/'native_feature_field_pe_v1/source_targets'/f'{rid}.pt' if old.is_new(rid) else old.DATA/'source_geometry_control_v1/inputs'/rid/'source_features.pt'
    return load(p)['features'].reshape(4,1024,1024),p
def init_config():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);assert sha(SCIENCE)==SCIENCE_SHA
    if (OUT/'configuration.json').exists():return
    write(OUT/'configuration.json',dict(science_sha256=SCIENCE_SHA,fields='Original44alpha-bound2048x1024, no C/J field',training_ids=[x['id'] for x in plan()['training_catalog']],development_ids=[x['id'] for x in plan()['catalog']],kinds=KINDS,seeds=SEEDS,steps=1000,batch=1,source_ids=SOURCES,query_ids=QUERIES,point_grid=point_grid().tolist(),point_coordinates='Native tracker0-index pixels, centers27.5+56i; sensor edge-coordinate=(point+.5)*210/448, target projection448edge−.5',labels='OriginalPILsource sensor/common_valid210, bilateral max(.02m,.02depth), bothcycle<=1pixel; no region/feature pairing selection; all64predicted',optimizer=dict(name='AdamW',lr=1e-4,weight_decay=.01,clip=5),loss='Four nativequerycoord outputs equal SmoothL1 beta8 on visibleGT points; no visibility/pose/feature loss',limits=dict(GPU_worker_hours=12,DATA_GiB=12,project_GiB=1),requested_model='gpt-6.1-sol/high',Fast_requested_or_verified=False,backend_effective_unknown=True))
