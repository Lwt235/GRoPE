"""Round023 frozen DINO input-Jacobian support; each image/probe committed once."""
import os,sys,json,time,hashlib,datetime,fcntl,signal,gc
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_content_support1'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_023/science_plan.md';SCIENCE_SHA='a145317509d7f3cbaace3a1004d9dc03d1faf805a99033e7dee44534892fdca9'
BASE=ROOT/'results/gaussian_memory/native_feature_field_pe1';J=BASE/'J';DATA=Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory')
TOKENS=[r*32+c for r in (4,12,20,28) for c in (4,12,20,28)]
STOP=False
def stop(*a):
    global STOP;STOP=True
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');torch.save(x,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    tmp.replace(p)
def load(p,device='cpu'):return torch.load(p,map_location=device,weights_only=True,mmap=True)
def log(worker,kind,**kw):
    p=OUT/'calls'/f'{worker}.jsonl';p.parent.mkdir(exist_ok=True)
    with p.open('a',buffering=1) as f:f.write(json.dumps(dict(utc=now(),kind=kind,**kw))+'\n')
def config():return read(OUT/'configuration.json')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);assert sha(SCIENCE)==SCIENCE_SHA
    if (OUT/'configuration.json').exists():return
    c=read(ROOT/'results/gaussian_memory/autonomous_research_loop/round_006/data_contract.json');manifest=read(ROOT/'results/gaussian_memory/camera_only_feature_pe1/input_manifest.json')['inputs'];plan=read(ROOT/'results/gaussian_memory/native_training_coverage1/plan.json');images=[]
    for m in c['development_catalog']:
        rid=m['id'];formation=read(J/'formation'/f'{rid}.json');source=DATA/'gaussian_scene_memory_v1'/rid/'source_input.pt'
        for cohort,indices in [('query',c['fit_queries']),('source',c['sources'])]:
            for ix,q in enumerate(indices):
                if cohort=='query':
                    paths=[p for p in manifest if p.endswith(f'/{rid}/query_{q}.pt')];patch=[p for p in manifest if p.endswith(f'/{rid}/patch_{q}_full.pt')];assert len(paths)==len(patch)==1;path=paths[0];cache=patch[0]
                else:path=str(source);cache=formation['source_target_path']
                images.append(dict(key=f'{rid}_{cohort}_{q}',id=rid,place=m['place'],cohort=cohort,frame=q,source_index=ix if cohort=='source' else None,image_path=path,cached_features=cache,field_path=formation['field_path'],published_field_sha256=formation['field_sha256']))
        field=load(formation['field_path']);valid=field['has_content'];p=field['means'].double();dist=torch.cdist(p,p,compute_mode='donot_use_mm_for_euclid_dist').square();dist[:,~valid]=torch.inf;dist.fill_diagonal_(torch.inf);ix=torch.argsort(dist,dim=-1,stable=True)[:,:31];ok=torch.isfinite(dist.gather(1,ix));delta=torch.where(ok[...,None],p[ix]-p[:,None],0.);B=[]
        for k in (1,8,16,31):
            count=ok[:,:k].sum(-1).clamp_min(1);B.append(torch.einsum('nki,nkj->nij',delta[:,:k],delta[:,:k])/count[:,None,None]+torch.eye(3,dtype=torch.float64)*float(field['unit'])**2*1e-6)
        save(OUT/'geometry'/f'{rid}.pt',dict(B4=torch.stack(B),field_path=formation['field_path'],published_field_sha256=formation['field_sha256'],neighbor_scales=[1,8,16,31],world_center_second_moments=True,unit_scaled_ridge=1e-6));del field,dist,delta,B
    rng=np.random.default_rng(20261114);probes=(2*rng.integers(0,2,(4,1024))-1).astype(np.float32)/32;save(OUT/'probes.pt',torch.from_numpy(probes))
    write(OUT/'configuration.json',dict(science_sha256=SCIENCE_SHA,images=images,model_path=plan['model_path'],published_model_sha256=plan['model_sha256'],tokens=TOKENS,token_coordinates=[[u//32,u%32] for u in TOKENS],grid_choice='Centers4,12,20,28 of four8patch bins; lower/right member of each even pair, fixed before measurements.',probes=dict(count=4,seed=20261114,normalized_Rademacher=True,sha256=sha(OUT/'probes.pt'),Frobenius_energy_scale_factor=1024,probe_variation_not_independent_places=True),projection='Original J reader perspective Jacobian in patch units, common actual gsplat read/near.01/far100; +0.3I2. No own covariance-dependent read or region-dependent re-render.',region=dict(G_mass=.95,mandatory_self=True,S_P_same_patch_count_as_G=True,tie_order='stable flat index'),precision='Original native BF16 autocast; inputFP32 RGB0..1, gradient-square/patch sumFP64; densityFP64.',limits=dict(GPU_worker_hours=2,CPU_worker_hours=4,host_RAM_GiB=32,new_storage_GiB=3),training=0,new_native_depth=0,all_data_used=True))
    print(dict(prepared_images=len(images),formal_VJP=96*16*4,raster=96),flush=True)

def encoder(worker):
    sys.path.insert(0,str(ROOT/'third_party/native_history_python'));sys.path.insert(0,str(ROOT/'third_party/streamvggt/src'))
    from streamvggt.layers.vision_transformer import vit_large
    t=time.monotonic();model=vit_large(img_size=518,patch_size=14,num_register_tokens=4,interpolate_antialias=True,interpolate_offset=0.,block_chunks=0,init_values=1.)
    weights=load(config()['model_path']);weights=weights.get('model',weights);prefix='aggregator.patch_embed.';state={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)};assert state
    model.load_state_dict(state,strict=True);del state,weights;model=model.eval().requires_grad_(False).cuda();gc.collect();torch.cuda.synchronize();log(worker,'encoder_load',seconds=time.monotonic()-t,parameters=sum(p.numel() for p in model.parameters()),old_model_rehashed=False)
    return model
def encode(model,image):
    mean=image.new_tensor([.485,.456,.406])[None,:,None,None];std=image.new_tensor([.229,.224,.225])[None,:,None,None]
    with torch.autocast('cuda',dtype=torch.bfloat16):return model((image-mean)/std)['x_norm_patchtokens']
def quaternion(q):
    q=q.double();q=q/q.norm(dim=-1,keepdim=True);w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)
def regions(density,u,area=None):
    others=torch.arange(1024,device=density.device);others=others[others!=u];order=others[torch.argsort(density[others],descending=True,stable=True)]
    if area is None:
        target=.95-density[u];n=0 if target<=0 else min(1023,int(torch.searchsorted(density[order].cumsum(0),target))+1);area=1+n
    mask=torch.zeros(1024,device=density.device,dtype=torch.bool);mask[u]=True;mask[order[:area-1]]=True;return mask
@torch.no_grad()
def layouts(case,cam,K,worker):
    from models.gaussian_scene_memory import contribution_weights
    t=time.monotonic();field={k:v.cuda() for k,v in load(case['field_path']).items()};Kp=K.float().clone();Kp[:2]/=14
    torch.cuda.synchronize();t0=time.monotonic();W,_=contribution_weights(field,cam.float()[None],Kp[None],32,32);torch.cuda.synchronize();log(worker,'actual_raster',image=case['key'],seconds=time.monotonic()-t0,raster=1)
    w=W[0].reshape(1024,-1)[TOKENS].double()*field['has_content'][None];den=w.sum(-1);w=w/torch.where(den>0,den,1)[:,None]
    cp=(field['means'].double()-cam[:3,3].double())@cam[:3,:3].double();valid=field['has_content']&(cp[:,2]>.01)&(cp[:,2]<100);assert not (~valid).any() or w[:,~valid].abs().max()==0
    cp=cp[valid];weights=w[:,valid];R=cam[:3,:3].double();rot=quaternion(field['rotations']);G=(rot*field['scales'].double().square()[:,None,:])@rot.transpose(-1,-2);S=torch.diagonal(G,dim1=-2,dim2=-1).sum(-1)[:,None,None]/3*torch.eye(3,device=G.device);B=load(OUT/'geometry'/f'{case["id"]}.pt')['B4'].cuda();covs={'G':G[None],'S':S[None],'P':B}
    fx,fy=Kp[0,0].double(),Kp[1,1].double();x,y,z=cp.unbind(-1);zero=torch.zeros_like(z);jac=torch.stack((torch.stack((fx/z,zero,-fx*x/z.square()),-1),torch.stack((zero,fy/z,-fy*y/z.square()),-1)),-2);uv=(cp@Kp.double().T)[:,:2]/z[:,None]
    yy,xx=torch.meshgrid(torch.arange(32,device=cp.device,dtype=torch.float64)+.5,torch.arange(32,device=cp.device,dtype=torch.float64)+.5,indexing='ij');pix=torch.stack((xx,yy),-1).reshape(1024,2);diff=pix[None]-uv[:,None];densities={}
    for arm,covs_arm in covs.items():
        pdf=[]
        for covariance in covs_arm:
            C3=R.T[None]@covariance[valid]@R[None];C=jac@C3@jac.transpose(-1,-2)+.3*torch.eye(2,device=cp.device);inverse=torch.linalg.inv(C);lp=-.5*torch.einsum('npi,nij,npj->np',diff,inverse,diff)-.5*torch.linalg.det(C).log()[:,None]-np.log(2*np.pi);pdf.append(lp)
        lp=torch.stack(pdf);ds=[]
        for i in range(16):
            if den[i]<=0:ds.append(torch.zeros(1024,device=cp.device,dtype=torch.float64));continue
            logw=weights[i].log()[None,:,None]-np.log(len(pdf));mixture=torch.logsumexp((lp+logw).reshape(-1,1024),0);ds.append(torch.softmax(mixture,0))
        densities[arm]=torch.stack(ds)
    masks={arm:torch.zeros(16,1024,device=cp.device,dtype=torch.bool) for arm in ('G','S','P')};areas=[]
    for i,u in enumerate(TOKENS):
        if den[i]>0:
            masks['G'][i]=regions(densities['G'][i],u);n=int(masks['G'][i].sum());masks['P'][i]=regions(densities['P'][i],u,n);masks['S'][i]=regions(densities['S'][i],u,n);areas.append(n)
            assert all(int(masks[a][i].sum())==n and masks[a][i,u] for a in masks) and densities['G'][i,masks['G'][i]].sum()>=.95-1e-12
        else:areas.append(0)
    torch.cuda.synchronize();log(worker,'density_layout',image=case['key'],seconds=time.monotonic()-t)
    return dict(densities={a:x.cpu().float() for a,x in densities.items()},regions={a:x.cpu() for a,x in masks.items()},areas=areas,read_rows=w.cpu().float(),read_mass=den.cpu(),read_defined=(den>0).cpu(),field_sha256_reference=case['published_field_sha256'],same_read_for_all_arms=True)

def image_input(case):
    x=load(case['image_path']);idx=case['source_index'];rgb=x['rgb'] if idx is None else x['rgb'][idx];cam=x['c2w'] if idx is None else x['c2w'][idx];K=x['K'] if idx is None else x['K'][idx]
    return rgb,cam.cuda(),K.cuda()
def measure(case,model,worker,preflight=False):
    folder=OUT/'images'/case['key'];folder.mkdir(parents=True,exist_ok=True)
    with (folder/'image.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'complete.json').exists():return
        rgb,cam,K=image_input(case);layout_path=folder/'layout.pt'
        if not layout_path.exists():save(layout_path,layouts(case,cam,K,worker))
        layout=load(layout_path);image=(rgb.cuda().float()/255)[None].requires_grad_(True);versions=[p._version for p in model.parameters()];assert not any(p.requires_grad for p in model.parameters())
        if preflight and not (OUT/'preflight_cache.json').exists():
            with torch.no_grad():
                t=time.monotonic();reference=encode(model,image);torch.cuda.synchronize();log(worker,'preflight_cached_reference_forward',encoder=1,seconds=time.monotonic()-t);cache=load(case['cached_features']);cached=cache['patches'] if case['cohort']=='query' else cache['features'][case['source_index']].reshape(1,1024,1024);cached=cached.cuda();diff=(reference-cached).float();write(OUT/'preflight_cache.json',dict(image=case['key'],cache_path=case['cached_features'],maximum_feature_difference=float(diff.abs().max()),RMS_feature_difference=float(diff.square().mean().sqrt()),cache_dtype=str(cached.dtype),actual_dtype=str(reference.dtype),all_old_parameters_frozen=True));save(OUT/'preflight_reference.pt',reference.cpu());del cached,cache
        if preflight and 'reference' not in locals():reference=load(OUT/'preflight_reference.pt').cuda()
        t=time.monotonic();y=encode(model,image);torch.cuda.synchronize();log(worker,'input_gradient_forward',image=case['key'],encoder=1,seconds=time.monotonic()-t)
        if preflight:
            ref=read(OUT/'preflight_cache.json');ref['grad_vs_no_grad_max_abs']=float((y-reference).abs().max());write(OUT/'preflight_cache.json',ref);del reference
        probes=load(OUT/'probes.pt').cuda();path=folder/'energy.npy';energy=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float64,shape=(16,4,1024))
        done_path=folder/'probe_commits.jsonl';done={}
        if done_path.exists():
            for line in done_path.read_text().splitlines():
                r=json.loads(line);done[r['token_index'],r['probe']]=r;assert hashlib.sha256(np.asarray(energy[r['token_index'],r['probe']]).tobytes()).hexdigest()==r['energy_sha256']
        for ti,u in enumerate(TOKENS):
            for probe in range(4):
                if (ti,probe) in done:continue
                t=time.monotonic();log(worker,'VJP_started',image=case['key'],token=u,probe=probe,VJP_attempt=1)
                grad=torch.autograd.grad((y[0,u]*probes[probe]).sum(),image,retain_graph=True)[0];e=grad.double().square().reshape(1,3,32,14,32,14).sum((0,1,3,5)).reshape(1024)*1024
                assert torch.isfinite(e).all();energy[ti,probe]=e.cpu().numpy();energy.flush()
                with path.open('rb') as sync:os.fdatasync(sync.fileno())
                record=dict(token_index=ti,token=u,probe=probe,energy_sha256=hashlib.sha256(np.asarray(energy[ti,probe]).tobytes()).hexdigest());done[ti,probe]=record
                with done_path.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
                torch.cuda.synchronize();log(worker,'VJP_complete',image=case['key'],token=u,probe=probe,VJP=1,seconds=time.monotonic()-t);del grad,e
                if STOP:del y,image,energy;torch.cuda.empty_cache();return
        assert len(done)==64 and versions==[p._version for p in model.parameters()] and all(p.grad is None for p in model.parameters());assert np.isfinite(energy).all()
        write(folder/'complete.json',dict(complete=True,image=case,VJP=64,encoder_forward=1,energy_sha256=sha(path),layout_sha256=sha(layout_path),parameter_versions_unchanged=True,parameter_grads_none=True,zero_derivative_probes=int((energy.sum(-1)==0).sum()),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),completed_utc=now()));print(dict(image_complete=case['key'],VJP=64),flush=True);del y,image,energy,layout;torch.cuda.empty_cache()

def worker(index,preflight,name):
    torch.set_num_threads(2);signal.signal(signal.SIGTERM,stop);torch.cuda.init();assert torch.cuda.is_available();log(name,'CUDA_access',device=torch.cuda.get_device_name(0));model=encoder(name);images=config()['images'];todo=[images[index]] if preflight else images[index::3]
    for case in todo:
        if STOP:break
        measure(case,model,name,preflight)
    if preflight and not STOP:write(OUT/'preflight_complete.json',dict(complete=True,formal_first_image_completed=todo[0]['key'],cache_comparison=read(OUT/'preflight_cache.json'),full_actual64VJP=True,only_input_gradients=True,old_training_not_resumed=True))
    del model;gc.collect();torch.cuda.empty_cache()
    if STOP:raise SystemExit(75)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','worker']);p.add_argument('--index',type=int,default=0);p.add_argument('--preflight',action='store_true');p.add_argument('--name',default='manual');a=p.parse_args()
    if a.phase=='prepare':torch.set_num_threads(2);prepare()
    else:worker(a.index,a.preflight,a.name)
