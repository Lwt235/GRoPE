"""Original compositing transports FP64 center and intrinsic-parameter fields."""
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from experiments.native_render_parameter_data import *
from models.gaussian_scene_memory import contribution_weights


def rotation(quat):
    q=quat.double();q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)


def svec(a):return torch.stack((a[...,0,0],a[...,1,1],a[...,2,2],2**.5*a[...,0,1],2**.5*a[...,0,2],2**.5*a[...,1,2]),-1)


@torch.no_grad()
def fields(field,camera,K,weights,original_alpha=None,kind=None):
    device=field['means'].device;unit=torch.as_tensor(field['unit'],device=device).double();scales=field['scales'].double()
    assert bool(torch.isfinite(unit)) and float(unit)>0 and bool(torch.isfinite(scales).all()) and bool((scales>0).all())
    cam=camera.double();rot=cam[:3,:3];p=((field['means'].double()-cam[:3,3])@rot)/unit
    w=weights.reshape(1024,-1).double()*field['has_content'][None];a=w.sum(-1,keepdim=True);nonzero=a[:,0]>0
    denominator=torch.where(nonzero[:,None],a,torch.ones_like(a))
    def avg(value):return (w@value)/denominator
    mean=avg(p);fourier=[]
    for k in range(5):fourier.extend((avg(torch.sin(np.pi*2**k*p)),avg(torch.cos(np.pi*2**k*p))))
    delta=p[None]-mean[:,None];scatter=torch.einsum('ij,ijc,ijd->icd',w,delta,delta)/denominator[:,:,None]
    yy,xx=torch.meshgrid(torch.arange(32,device=device,dtype=torch.float64)+.5,torch.arange(32,device=device,dtype=torch.float64)+.5,indexing='ij')
    pix=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3)
    rays=F.normalize(pix@torch.linalg.inv(K.double()).T@rot.T,dim=-1)
    d=(field['means'].double()-cam[:3,3])/unit;along=rays@d.T;radial2=(d.square().sum(-1)[None]-along.square()).clamp_min(0)
    old=torch.cat((torch.log1p(radial2[...,None]/p.new_tensor([.02,.05,.1,.2,.5]).square()),torch.asinh(along)[...,None],torch.log1p(d.norm(dim=-1))[None,:,None].expand(1024,-1,1)),-1)
    center=torch.cat((mean,*fourier,torch.asinh(svec(scatter)),(w[:,:,None]*old).sum(1)/denominator),-1)
    full=torch.zeros((1024,6),device=device,dtype=torch.float64);sph=torch.zeros_like(full)
    if kind in (None,'G'):
        basis=rot.T[None]@rotation(field['rotations']);logs=2*torch.log(scales/unit);logcov=(basis*logs[:,None,:])@basis.transpose(-1,-2);full=avg(svec(logcov))
    if kind in (None,'S'):
        sphere=torch.log(scales.square().sum(-1)/(3*unit.square()))[:,None,None]*torch.eye(3,device=device,dtype=torch.float64);sph=avg(svec(sphere))
    for value in (center,full,sph):value[~nonzero]=0
    alpha=a.float().reshape(-1) if original_alpha is None else original_alpha.float().reshape(-1)
    assert torch.equal(alpha==0,~nonzero) and center.shape==(1024,46)
    result=dict(a=alpha,center46=center.float(),sphere6=sph.float(),full6=full.float())
    assert all(bool(torch.isfinite(v).all()) for v in result.values())
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1);a=parser.parse_args();p=initialize();torch.set_num_threads(4);begin=time.monotonic();files={}
    for index,meta in enumerate(p['training_catalog']+p['catalog']):
        if index%a.shards!=a.shard:continue
        rid=meta['id'];marker=OUT/'geometry'/f'{rid}.json'
        if marker.exists():
            old=read(marker);assert old['complete'] and all(sha256(k)==v for k,v in old['files'].items());continue
        t=time.monotonic();field=load(scene_path(rid),'cuda');inputs={str(scene_path(rid)):sha256(scene_path(rid))};outfiles={};diagnostics=[]
        for q in EVAL_QUERIES:
            query=load(query_path(rid,q),'cuda');K=query['K'].clone();K[:2]/=14
            weights,_=contribution_weights(field,query['c2w'][None],K[None],32,32)
            original=load(feature_path(rid,q),'cuda');payload=fields(field,query['c2w'],K,weights[0],original['effective_alpha'])
            payload['identity']=dict(id=rid,query=q,source_sha256=inputs[str(scene_path(rid))],query_sha256=sha256(query_path(rid,q)),original_feature_sha256=sha256(feature_path(rid,q)))
            path=geometry_path(rid,q);path.parent.mkdir(parents=True,exist_ok=True);save_tensor(path,payload);outfiles[str(path)]=sha256(path)
            for path in [query_path(rid,q),feature_path(rid,q)]+[patch_path(rid,q,c) for c in CONDITIONS]:inputs[str(path)]=sha256(path)
            effective=weights[0].reshape(1024,-1).float()*field['has_content'][None];alpha_error=float((effective.sum(-1)-payload['a']).abs().max())
            diagnostics.append(dict(query=q,alpha_original_max_abs=alpha_error,zero_coverage=int((payload['a']==0).sum())))
        torch.cuda.synchronize();write(marker,dict(complete=True,id=rid,source_path=str(scene_path(rid)),source_sha256=inputs[str(scene_path(rid))],files=outfiles,inputs=inputs,diagnostics=diagnostics,seconds=time.monotonic()-t,source_only_given_camera=True,raw_source_content_unchanged=True));print(dict(fields_ready=rid,seconds=time.monotonic()-t),flush=True)
    write(OUT/'preparation'/f'{a.shard}.json',dict(complete=True,seconds=time.monotonic()-begin,peak_GiB=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':main()
