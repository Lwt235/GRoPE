"""Finite round009 preparation, paired fits and actual native-camera evaluation."""
import argparse,os,sys,time,traceback,random
import numpy as np
import torch
from torch.nn import functional as F
from experiments.native_pose_memory_data import *
from models.native_pose_memory_reader import NativePoseMemoryReader
from models.gaussian_scene_memory import rasterize_field,field_bytes

class Worker:
    def __init__(self,name):
        self.name=name;self.start=time.monotonic();self.count={};self.seconds={};self.file=OUT/'workers'/f'{name}.json';self.prior=read(self.file)['seconds'] if self.file.exists() else 0
        
        if self.file.exists():
            prior=read(self.file);self.count=prior['counts'];self.seconds=prior['timing_seconds']
        self.before={};torch.cuda.init();torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GROPE_POSE_MEMORY_GPU_FRACTION','.65')));self.tick()
    def tick(self):
        write(self.file,dict(name=self.name,pid=os.getpid(),seconds=self.prior+time.monotonic()-self.start,counts=self.count,timing_seconds=self.seconds,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),status='active'))
    def event(self,key,n=1,seconds=None):
        self.count[key]=self.count.get(key,0)+n
        if seconds is not None:self.seconds[key]=self.seconds.get(key,0)+seconds
        self.tick()
    def close(self,status):
        self.tick();x=read(self.file);x['status']=status;x['ended_utc']=now();write(self.file,x)

def to(x,dev='cuda'):
    if torch.is_tensor(x):return x.to(dev)
    if isinstance(x,dict):return {k:to(v,dev) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(to(v,dev) for v in x)
    return x

def official():
    from experiments.gaussian_memory_native_run import model_load
    m=model_load(plan());
    for name in ('point_head','track_head'):
        module=getattr(m,name,None)
        if module is not None:module.cpu()
    return m

def camrun(head,token,prefix):
    # A fresh list is essential: the frozen official head appends four internal iterations.
    return head([token[:,None,None,:]],past_key_values_camera=list(prefix),use_cache=True)[0][-1][:,0]

def native_step(m,image,state,need_depth=False):
    image=image[None,None].cuda().float()/255
    with torch.autocast('cuda',dtype=torch.bfloat16):tokens,start,kv=m.aggregator(image,past_key_values=list(state['kv']),use_cache=True,past_frame_idx=state['n'])
    pose,cam=m.camera_head(tokens,past_key_values_camera=list(state['cam']),use_cache=True)
    dep=m.depth_head(tokens,images=image,patch_start_idx=start)[0][0,0,...,0] if need_depth else None
    return dict(kv=kv,cam=cam,n=state['n']+1),pose[-1][0,0],tokens[-1][0,0,0].float(),tokens[-1][0,0,start:,:1024].float(),dep

def alignment(poses,sourcepayload,firstdepth,pairpayload):
    from streamvggt.utils.rotation import quat_to_mat
    R=quat_to_mat(poses[:,3:7]);c=-(R.transpose(-1,-2)@poses[:,:3,None])[...,0]
    truth=sourcepayload['c2w'].cuda();A=truth[0,:3,:3]@R[0];ratios=[]
    for i in range(4):
        for j in range(i):
            a=(c[i]-c[j]).norm();b=(truth[i,:3,3]-truth[j,:3,3]).norm()
            if a>1e-6 and b>1e-6 and torch.isfinite(a+b):ratios.append(b/a)
    fallback=False
    if ratios:s=torch.stack(ratios).double().quantile(.5).float()
    else:
        fallback=True;sensor=pairpayload['depth_sensor'][0].cuda();valid=pairpayload['common_valid'][0].cuda()&torch.isfinite(firstdepth)&(firstdepth>1e-6)
        s=(sensor[valid]/firstdepth[valid]).double().quantile(.5).float() if valid.any() else torch.tensor(float('nan'),device='cuda')
    valid=bool(torch.isfinite(s)&(s>0));s=s if valid else torch.tensor(1.,device='cuda')
    return dict(rotation=A,translation=truth[0,:3,3]-s*(A@c[0]),scale=s,valid=valid),dict(valid=valid,scale=float(s),baseline_ratios=[float(x) for x in ratios],fallback_first_source_depth=fallback,query_truth_used=False)

def patchgeometry(depth,K,cam,unit,sourcecam):
    yy,xx=torch.meshgrid(torch.arange(448,device='cuda')+.5,torch.arange(448,device='cuda')+.5,indexing='ij')
    rays=torch.stack((xx,yy,torch.ones_like(xx)),-1)@torch.linalg.inv(K).T
    p=rays*depth[...,None];p=p@cam[:3,:3].T+cam[:3,3]
    p=p.reshape(32,14,32,14,3).permute(0,2,1,3,4).reshape(1024,196,3).double()
    ok=torch.isfinite(p).all(-1);p=torch.nan_to_num(p);count=ok.sum(-1).clamp_min(1);mean=(p*ok[...,None]).sum(1)/count[:,None];d=torch.where(ok[...,None],p-mean[:,None],0.)
    cov=d.transpose(-1,-2)@d/count[:,None,None]
    return geometry_from_points(mean,cov,unit,sourcecam,ok.any(-1))

@torch.no_grad()
def prepare(w,startindex,endindex):
    assert sha(SCIENCE)==SCIENCE_SHA;CACHE.mkdir(parents=True,exist_ok=True);assert not resources()['stop_reasons']
    torch.cuda.synchronize();tm=time.monotonic();m=official();torch.cuda.synchronize();w.event('model_load',seconds=time.monotonic()-tm)
    for meta in metas()[startindex:endindex]:
        rid=meta['id'];mark=OUT/'preparation'/f'{rid}.json'
        if mark.exists():continue
        assert not resources()['stop_reasons'];begin=time.monotonic();sp=load(source(rid));pp=load(pair(rid));field=load(scene(rid));g,gs=source_geometry(field);g=to(g)
        state=dict(kv=[None]*m.aggregator.depth,cam=[None]*m.camera_head.trunk_depth,n=0);poses=[];firstdep=None;timings={}
        torch.cuda.synchronize();t=time.monotonic()
        for i,rgb in enumerate(sp['rgb']):
            state,pose,*rest=native_step(m,rgb,state,i==0);poses.append(pose)
            if i==0:firstdep=rest[-1]
            w.event('source_native',1)
        torch.cuda.synchronize();timings['source_prefix']=time.monotonic()-t
        align,adiag=alignment(torch.stack(poses),sp,firstdep,pp)
        prefix=to(state['cam'],'cpu');prefixbytes=field_bytes(state)
        save(sceneout(rid),dict(prefix=prefix,alignment=to(align,'cpu'),geometry=to(g,'cpu'),ordinary=ordinary(field,to(g,'cpu')),unit=field['unit'],source_predictions=torch.stack(poses).cpu()))
        field=to(field);colors=torch.cat((field['features'].float()*field['has_content'][:,None],field['has_content'][:,None].float()),-1)
        records=[]
        for q in QUERIES:
            p=casepath(rid,q)
            if p.exists():continue
            payload=load(query(rid,q));torch.cuda.synchronize();t=time.monotonic()
            _,pose,token,features,depth=native_step(m,payload['rgb'],state,True);torch.cuda.synchronize();native_seconds=time.monotonic()-t;w.event('query_native_initial',seconds=native_seconds)
            from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
            _,K=pose_encoding_to_extri_intri(pose[None,None],image_size_hw=(448,448));K=K[0,0];camera=mappose(pose,align)
            # Map the native predicted depths by the source-only common scale.
            lg=patchgeometry(depth*align['scale'],K,camera,field['unit'],field['source_c2w'][0]);lv=view_fields(lg,camera,K);v=view_fields(g,camera,K)
            torch.cuda.synchronize();t=time.monotonic();render,alpha,_=rasterize_field(field,camera[None],(K/torch.tensor([[14,14,14],[14,14,14],[1,1,1]],device='cuda'))[None],32,32,colors)
            torch.cuda.synchronize();render_seconds=time.monotonic()-t;w.event('gsplat_query',seconds=render_seconds)
            r=render[0].reshape(1024,1025);a=r[:,-1:];anchor=torch.where(a>0,(r[:,:1024].double()/a.double().clamp_min(1e-300)).float(),0.)
            qordinary=torch.cat((F.layer_norm(features,(1024,)),lg['source20']),-1);lo=qordinary
            xy=lv['common25'][:,3:5]/2
            normK=K.clone();normK[:2]/=448
            item=dict(qordinary=qordinary,anchor=anchor,alpha=a,local_ordinary=lo,local_view=lv,view=v,K=normK,xy=xy,token=token,initial_pose=pose,initial_camera=camera,alignment=align,target=payload['c2w'].cuda(),unit=field['unit'],alignment_valid=align['valid'])
            assert all(torch.isfinite(x).all() for x in (qordinary,anchor,a,token,pose,camera))
            save(p,to(item,'cpu'));row=dict(query=q,coverage=float((a>0).float().mean()),alpha_mean=float(a.mean()),zero_support=not bool((a>0).any()),valid_keys=int(v['key_mask'].sum()),native_seconds=native_seconds,render_seconds=render_seconds,bytes=p.stat().st_size)
            records.append(row);log('preparation_queries.jsonl',dict(id=rid,**row));del item,render,anchor,lg,lv
        from experiments.gaussian_memory_native_run import state_bytes
        write(mark,dict(complete=True,id=rid,place=meta['place'],alignment=adiag,source_prefix_bytes=state_bytes(dict(kv=state['kv'],cam=state['cam'])),source_seconds=timings['source_prefix'],seconds=time.monotonic()-begin,source_path=str(source(rid)),GS_path=str(scene(rid)),queries=records,scene_bytes=sceneout(rid).stat().st_size,scene_sha256=sha(sceneout(rid))))
        print(dict(prepared=rid,seconds=time.monotonic()-begin,alignment=adiag['valid']),flush=True);del state,g,field,colors;torch.cuda.empty_cache()
    del m

class Samples:
    def __init__(self,train=False):self.scenes={};self.cases={};self.train=train
    def get(self,rid,q):
        if rid not in self.scenes:self.scenes[rid]=load(sceneout(rid))
        key=(rid,q)
        if key not in self.cases:self.cases[key]=load(casepath(rid,q))
        return self.scenes[rid],self.cases[key]

def batch(samples,identities,kind):
    sc,ca=zip(*(samples.get(r,q) for r,q in identities))
    stack=lambda xs:to(torch.stack(xs))
    b={k:stack([c[k] for c in ca]) for k in ('qordinary','anchor','alpha','K','xy','token','target','unit','initial_pose')}
    b['alignment']={k:stack([s['alignment'][k] if torch.is_tensor(s['alignment'][k]) else torch.tensor(s['alignment'][k]) for s in sc]) for k in ('rotation','translation','scale','valid')}
    b['prefix']=[tuple(torch.cat([s['prefix'][i][j] for s in sc],0).cuda() for j in range(2)) for i in range(4)]
    b['ordinary']=stack([c['local_ordinary'] if kind=='L' else s['ordinary'] for s,c in zip(sc,ca)])
    key='local_view' if kind=='L' else 'view';b['view']={k:stack([c[key][k] for c in ca]) for k in ca[0][key]}
    if kind=='L':b['anchor'].zero_();b['alpha'].zero_()
    return b

def predict(reader,head,b,kind,encoded=None):
    residual=reader(b['qordinary'],b['anchor'],b['alpha'],b['ordinary'],b['view'],b['K'],b['xy'],kind,encoded)
    pose=camrun(head,b['token']+residual,b['prefix']);camera=mappose(pose,b['alignment']);return pose,camera,residual

def pose_loss(camera,target,unit):
    delta=camera[:,:3,:3].transpose(-1,-2)@target[:,:3,:3];cos=(delta.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    # atan2 geodesic is stable near zero and pi and preserves the squared-radian objective.
    skew=torch.stack((delta[:,2,1]-delta[:,1,2],delta[:,0,2]-delta[:,2,0],delta[:,1,0]-delta[:,0,1]),-1)/2
    angle=torch.atan2(skew.norm(dim=-1),cos)
    return ((camera[:,:3,3]-target[:,:3,3]).square().sum(-1)/unit.square()+angle.square()).mean()

def head_load(w):
    t=time.monotonic();m=official();head=m.camera_head;del m;torch.cuda.empty_cache();w.event('model_load_training',seconds=time.monotonic()-t);return head

def fit(w,head,kind,seed):
    folder=CACHE/'fits'/f'{kind}_{seed}';final=folder/'final.pt';resume=folder/'resume.pt'
    if final.exists():return
    assert not resources()['stop_reasons'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.cuda.manual_seed_all(seed)
    reader=NativePoseMemoryReader().cuda();opt=torch.optim.AdamW(reader.parameters(),lr=3e-4,weight_decay=.01);sampler=np.random.default_rng(seed);step=0;losses=[]
    train=[(m['id'],q) for m in plan()['training_catalog'] for q in QUERIES[:4]];samples=Samples(True)
    if resume.exists():
        st=load(resume);reader.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);sampler.bit_generator.state=st['sampler'];step=st['step'];losses=st['losses'];torch.set_rng_state(st['torch_rng']);torch.cuda.set_rng_state(st['cuda_rng'])
    begin=time.monotonic();reader.train()
    def checkpoint(path):save(path,dict(step=step,model=to(reader.state_dict(),'cpu'),optimizer=to(opt.state_dict(),'cpu'),sampler=sampler.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),losses=losses,kind=kind,seed=seed,science_sha256=SCIENCE_SHA,worker_counts=w.count,worker_seconds=w.prior+time.monotonic()-w.start))
    for i in range(step,2000):
        ids=[train[x] for x in sampler.integers(0,len(train),2)];b=batch(samples,ids,kind);opt.zero_grad(set_to_none=True)
        w.event('training_reader_samples',2);w.event('training_camera_head_samples',2)
        pose,cam,residual=predict(reader,head,b,kind);loss=pose_loss(cam,b['target'],b['unit']);assert torch.isfinite(loss)
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(reader.parameters(),5);assert torch.isfinite(gn);opt.step();step=i+1;losses.append(float(loss.detach()))
        if step%100==0:
            checkpoint(resume);w.event('formal_updates',100);r=resources();print(dict(fit=f'{kind}_{seed}',step=step,loss=losses[-1],seconds=time.monotonic()-begin),flush=True)
            if r['stop_reasons']:raise RuntimeError(str(r['stop_reasons']))
    checkpoint(final);write(OUT/'fits'/f'{kind}_{seed}.json',dict(complete=True,steps=step,samples=4000,seconds=time.monotonic()-begin,final_path=str(final),sha256=sha(final),parameters=sum(p.numel() for p in reader.parameters()),loss_first=losses[0],loss_last=losses[-1]));del reader,opt,samples;torch.cuda.empty_cache()

@torch.no_grad()
def evaluate(w,head,kind,seed):
    mark=OUT/'evaluation'/f'{kind}_{seed}.json'
    if mark.exists():return
    torch.manual_seed(seed);reader=NativePoseMemoryReader().cuda();reader.load_state_dict(load(CACHE/'fits'/f'{kind}_{seed}'/'final.pt')['model']);reader.eval();samples=Samples();rows=[]
    for meta in metas():
        rid=meta['id'];sc=load(sceneout(rid));enc=reader.encode_source(sc['ordinary'][None].cuda()) if kind!='L' else None
        statepath=CACHE/'public_states'/f'{kind}_{seed}'/f'{rid}.pt'
        if kind!='L':save(statepath,dict(encoded=to(enc,'cpu'),GS_reference=str(scene(rid)),ordinary_reference=str(sceneout(rid)),kind=kind,seed=seed))
        for q in QUERIES:
            path=CACHE/'predictions'/f'{kind}_{seed}'/rid/f'{q}.pt'
            if path.exists():continue
            t=time.monotonic();b=batch(samples,[(rid,q)],kind);torch.cuda.synchronize();start=time.monotonic();w.event('evaluation_reader_samples');w.event('evaluation_camera_head_samples');pose,cam,res=predict(reader,head,b,kind,enc);torch.cuda.synchronize();seconds=time.monotonic()-start
            save(path,dict(pose=pose[0].cpu(),camera=cam[0].cpu(),residual=res[0].cpu(),initial_intrinsics_unchanged=True));log('evaluation_queries.jsonl',dict(kind=kind,seed=seed,id=rid,query=q,seconds=seconds,load_and_compute_seconds=time.monotonic()-t,bytes=path.stat().st_size))
        samples=Samples();torch.cuda.empty_cache()
    write(mark,dict(complete=True,queries=352,public_states=44 if kind!='L' else 0));del reader,samples;torch.cuda.empty_cache()

def preflight(w):
    m=official();head=m.camera_head;del m;reader=NativePoseMemoryReader().cuda();s=Samples();ids=[(plan()['training_catalog'][i]['id'],2) for i in (0,1)];records=[]
    for kind in KINDS:
        b=batch(s,ids,kind);pose,cam,res=predict(reader,head,b,kind);w.event('preflight_reader_samples',2);w.event('preflight_camera_head_samples',2)
        diff=float((pose-b['initial_pose']).abs().max());assert torch.count_nonzero(res)==0
        loss=pose_loss(cam,b['target'],b['unit']);loss.backward();g=reader.residual[-1].weight.grad;assert g is not None and torch.isfinite(g).all() and g.abs().max()>0;reader.zero_grad()
        records.append(dict(kind=kind,zero_residual=True,initial_pose_max_abs=diff,loss=float(loss.detach())))
    write(OUT/'preflight.json',dict(complete=True,records=records,prefix_query_not_appended=True,official_camera_frozen=not any(p.requires_grad for p in head.parameters()),query_truth_only_loss=True));print('preflight passed',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preflight','run']);p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=44);p.add_argument('--slot',type=int,default=0);a=p.parse_args();torch.set_num_threads(4);w=Worker(f'{a.mode}_{a.slot}')
    try:
        if a.mode=='prepare':prepare(w,a.start,a.end)
        elif a.mode=='preflight':preflight(w)
        else:
            head=head_load(w)
            for ix,(kind,seed) in enumerate((k,s) for s in SEEDS for k in KINDS):
                if ix%2==a.slot:fit(w,head,kind,seed);evaluate(w,head,kind,seed)
        w.close('complete')
    except BaseException as e:
        log('failures.jsonl',dict(worker=w.name,error=repr(e),traceback=traceback.format_exc(),counts=w.count));w.close('failed');raise
if __name__=='__main__':main()
