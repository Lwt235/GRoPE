"""Evaluation-only source visibility and fixed native/rendered-depth references."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.gaussian_scene_data import OUT,CACHE,source_path,query_path,query_image,transformed_rgb,transformed_target,read
from experiments.gaussian_scene_bind import load_tensor
from experiments.gaussian_scene_config import freeze
from experiments.gaussian_scene_preflight import metric_depth,official_current
from experiments.gaussian_memory_incremental_data import depth_source,save_tensor,write_json
from experiments.gaussian_memory_native_run import model_load,state_new,state_fork,step,state_bytes
from experiments.gaussian_memory_setup import sha256
from models.gaussian_scene_memory import rasterize_field


@torch.no_grad()
def support(meta,p):
    rid=meta['id'];path=CACHE/'evaluation_support'/f'{rid}.pt'
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    source=load_tensor(source_path(rid),'cuda');depths=[];valids=[];K=source['K'].clone();K[:,:2]*=210/448
    identities={}
    for index in p['sources']:
        frame=meta['row']['frames'][index];dep=depth_source(frame)
        _,transform=transformed_rgb(frame['rgb_path'])
        depth,valid=transformed_target(dict(frame,depth_path=str(dep)),transform)
        depths.append(depth.cuda());valids.append(valid.cuda());identities[str(dep)]=sha256(dep)
    depths=torch.stack(depths);valids=torch.stack(valids);labels={}
    yy,xx=torch.meshgrid(torch.arange(210,device='cuda').float(),torch.arange(210,device='cuda').float(),indexing='ij')
    pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3)
    for q in p['queries']:
        query=load_tensor(query_path(rid,q),'cuda')
        camera=(pixels@torch.linalg.inv(query['K_metric']).T)*query['target'].reshape(-1,1)
        world=camera@query['c2w'][:3,:3].T+query['c2w'][:3,3]
        observed=torch.zeros(210*210,device='cuda',dtype=torch.bool)
        for v in range(4):
            point=(world-source['c2w'][v,:3,3])@source['c2w'][v,:3,:3]
            projection=point@K[v].T;uv=projection[:,:2]/projection[:,2:].clamp_min(1e-8)
            xy=uv.round().long();inside=(point[:,2]>.01)&(xy[:,0]>=0)&(xy[:,0]<210)&(xy[:,1]>=0)&(xy[:,1]<210)
            x,y=xy[:,0].clamp(0,209),xy[:,1].clamp(0,209)
            depth=depths[v,y,x];tol=p['support']['depth_tolerance_m']+p['support']['depth_tolerance_fraction']*depth
            observed|=inside&valids[v,y,x]&((point[:,2]-depth).abs()<=tol)
        labels[str(q)]=observed.reshape(210,210).cpu()&query['valid'].cpu()
    save_tensor(path,dict(labels=labels,source_depth_sha256=identities,evaluation_only=True))
    write_json(OUT/'support'/f'{rid}.json',dict(complete=True,id=rid,path=str(path),sha256=sha256(path),
        source_depth_sha256=identities,evaluation_only=True,rule=p['support']))


@torch.no_grad()
def native_references(model,p):
    marker=OUT/'references/native.json'
    if marker.exists():
        old=read(marker);assert old['prediction_sha256']==sha256(old['prediction_path']);return
    predictions=[];keys=[];times=[]
    for meta in p['catalog']:
        rid=meta['id'];source=load_tensor(source_path(rid));prefix=state_new(model)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
        for v,index in enumerate(p['sources']):step(model,source['rgb'][v].float()/255,index,prefix,False)
        torch.cuda.synchronize();source_seconds=time.monotonic()-begin
        query_times=[]
        for q in p['queries']:
            payload=load_tensor(query_path(rid,q),'cuda')
            for c in p['conditions']:
                image=query_image(payload,c)
                torch.cuda.synchronize();begin=time.monotonic()
                current=official_current(model,image)
                torch.cuda.synchronize();current_seconds=time.monotonic()-begin
                torch.cuda.synchronize();begin=time.monotonic()
                depth,_=step(model,image,q,state_fork(prefix),True)
                torch.cuda.synchronize();history_seconds=time.monotonic()-begin
                for name,value in [('native_current',current),('official_history',depth.cuda())]:
                    predictions.append(metric_depth(value).float().cpu());keys.append((rid,q,c,name))
                query_times.append(dict(query=q,condition=c,current_seconds=current_seconds,history_seconds=history_seconds))
        times.append(dict(id=rid,source_seconds=source_seconds,prefix_tensor_bytes=state_bytes(prefix),
                          query_times=query_times,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3))
        print(dict(native_references_ready=rid),flush=True)
        del prefix,source;torch.cuda.empty_cache()
    path=CACHE/'evaluation/native_references.pt';path.parent.mkdir(parents=True,exist_ok=True)
    save_tensor(path,dict(keys=keys,depth_metric=torch.stack(predictions)))
    write_json(marker,dict(complete=True,predictions=len(keys),prediction_path=str(path),prediction_sha256=sha256(path),
        plan_sha256=sha256(OUT/'plan.json'),costs=times,scope='Unadapted background references, not matched budget/parameters'))


@torch.no_grad()
def renderer_references(p):
    marker=OUT/'references/renderer.json'
    if marker.exists():
        old=read(marker);assert old['prediction_sha256']==sha256(old['prediction_path']);return
    keys=[];depths=[];alphas=[];images=[];records=[]
    for meta in p['catalog']:
        rid=meta['id'];field=load_tensor(CACHE/rid/'scene_memory.pt','cuda')
        for q in p['queries']:
            query=load_tensor(query_path(rid,q),'cuda')
            torch.cuda.synchronize();begin=time.monotonic()
            rendered,alpha,_=rasterize_field(field,query['c2w'][None],query['K_metric'][None],210,210,mode='RGB+ED')
            torch.cuda.synchronize();seconds=time.monotonic()-begin
            d=rendered[0,...,3];covered=alpha[0,...,0]>p['renderer_reference']['coverage_threshold']
            d=torch.where(covered,d,field['unit'])
            assert torch.isfinite(d).all() and (d>0).all()
            for c in p['conditions']:
                keys.append((rid,q,c,'render_depth'));depths.append(d.cpu());alphas.append(alpha[0,...,0].cpu());images.append(rendered[0,...,:3].cpu())
            records.append(dict(id=rid,query=q,seconds=seconds,coverage_fraction=float(covered.float().mean()),
                                source_state_sha256=sha256(CACHE/rid/'scene_memory.pt')))
        print(dict(renderer_reference_ready=rid),flush=True)
    path=CACHE/'evaluation/renderer_reference.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    save_tensor(path,dict(keys=keys,depth_metric=torch.stack(depths),alpha=torch.stack(alphas),rgb=torch.stack(images)))
    write_json(marker,dict(complete=True,predictions=len(keys),prediction_path=str(path),prediction_sha256=sha256(path),
        plan_sha256=sha256(OUT/'plan.json'),records=records,scope='Source-only saved field; no query RGB used for rendered depth'))


def main():
    torch.set_num_threads(4);p=freeze()
    for meta in p['catalog']:support(meta,p)
    renderer_references(p);model=model_load(p);native_references(model,p)


if __name__=='__main__':main()
