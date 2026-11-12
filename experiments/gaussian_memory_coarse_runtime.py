"""Unadapted references and resident-source full two-pass query timing."""
import argparse
import json
import time
import numpy as np
import torch
from experiments.gaussian_memory_coarse_data import *
from experiments.gaussian_memory_coarse_cache import to_device,dense_processed
from experiments.gaussian_memory_native_run import model_load,state_new,step,metric_depth,scores,state_bytes
from models.gaussian_memory_coarse_pe import CoarseReader,forward_query,patch_tokens,query_geometry,descriptors,group_geometry,START


@torch.inference_mode()
def source_geometry(depths,poses,hw):
    """Same population moments as the used source cache, computed from RGB predictions."""
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    depth=torch.stack(depths).double();pose=torch.stack(poses)[None]
    ext,intr=pose_encoding_to_extri_intri(pose,hw);ext=ext[0].double();intr=intr[0].double();h,w=hw
    y,x=torch.meshgrid(torch.arange(h,device=depth.device),torch.arange(w,device=depth.device),indexing='ij')
    camera=torch.stack(((x[None]-intr[:,0,2,None,None])/intr[:,0,0,None,None]*depth,(y[None]-intr[:,1,2,None,None])/intr[:,1,1,None,None]*depth,depth),-1)
    points=torch.einsum('shwi,sij->shwj',camera-ext[:,None,None,:3,3],ext[:,:3,:3])
    points=(points-points[0].mean((0,1)))/depth[0].median()
    patches=points.reshape(len(depths),h//14,14,w//14,14,3).permute(0,1,3,2,4,5).reshape(len(depths),-1,196,3)
    mu=patches.mean(-2);delta=patches-mu[...,None,:]
    cov=delta.transpose(-1,-2)@delta/196+1e-6*torch.eye(3,device=depth.device)
    return mu.flatten(0,1).float(),cov.flatten(0,1).float()


@torch.inference_mode()
def full_source_and_dense_cost(model,p):
    """No reused geometry: source backbone/KV, camera, depth, moments, FPS, and pooling."""
    records=[]
    for rid in p['training_ids'][:2]:
        for condition in p['conditions']:
            inp=load(input_path(rid,11,condition));hw=list(inp['images'].shape[-2:]);state=state_new(model)
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            depths=[];poses=[];encoder_camera=0.;depth_seconds=0.
            for index,source_image in enumerate(inp['images'][:-1]):
                torch.cuda.synchronize();t0=time.perf_counter();images=source_image[None,None].cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):tokens,_,state['kv']=model.aggregator(images,past_key_values=state['kv'],use_cache=True,past_frame_idx=index)
                with torch.autocast('cuda',enabled=False):enc,state['cam']=model.camera_head(tokens,past_key_values_camera=state['cam'],use_cache=True)
                torch.cuda.synchronize();t1=time.perf_counter()
                with torch.autocast('cuda',enabled=False):depth,_=model.depth_head(tokens,images,START)
                torch.cuda.synchronize();t2=time.perf_counter();encoder_camera+=t1-t0;depth_seconds+=t2-t1
                depths.append(depth[0,0,...,0]);poses.append(enc[-1][0,0])
            del tokens,enc,depth,images
            with torch.autocast('cuda',enabled=False):mu,cov=source_geometry(depths,poses,hw)
            torch.cuda.synchronize();after_geometry=time.perf_counter();geometry_seconds=after_geometry-start-encoder_camera-depth_seconds
            official_bytes=state_bytes(state)
            with torch.autocast('cuda',dtype=torch.bfloat16):keys,values=dense_processed(model,state,hw)
            torch.cuda.synchronize();after_keys=time.perf_counter();common_seconds=after_keys-start
            ntokens=1041;special=torch.arange(keys[0].shape[-2],device='cuda').remainder(ntokens)<START
            pools={}
            for n in CAPACITIES:
                torch.cuda.synchronize();t0=time.perf_counter();membership,mass,mean,sigma,_=group_geometry(mu,cov,n)
                weights=torch.nn.functional.one_hot(membership,n).float().T/mass[:,None];pk=[];pv=[]
                for k,v in zip(keys,values):
                    def pool(tensor):
                        with torch.autocast('cuda',enabled=False):average=torch.einsum('mn,bhnd->bhmd',weights,tensor[...,~special,:].float())
                        return torch.cat((tensor[...,special,:],average.to(tensor.dtype)),-2)
                    pk.append(pool(k));pv.append(pool(v))
                packed_k=torch.stack(pk);packed_v=torch.stack(pv)
                torch.cuda.synchronize();duration=time.perf_counter()-t0
                pools[str(n)]=dict(group_and_pool_seconds=duration,full_source_construction_seconds=common_seconds+duration,KV_bytes=packed_k.numel()*packed_k.element_size()+packed_v.numel()*packed_v.element_size())
                del weights,pk,pv,packed_k,packed_v,membership,mass,mean,sigma
            construction_peak=torch.cuda.max_memory_allocated()/2**30
            cached=torch.load(sample_path(rid,11,condition,256),map_location='cpu',mmap=True,weights_only=False)
            replay_mu=float((mu.cpu()-cached['source_patch_means']).abs().max());replay_cov=float((cov.cpu()-cached['source_patch_covariances']).abs().max())
            del cached,mu,cov,keys,values,depths,poses,k,v
            # Original one-pass official query, same pre-query native state each time.
            query_times=[]
            for repeat in range(4):
                kv=list(state['kv']);cam=list(state['cam']);torch.cuda.synchronize();t0=time.perf_counter();images=inp['images'][-1:][None].cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):tokens,_,_=model.aggregator(images,past_key_values=kv,use_cache=True,past_frame_idx=len(inp['frame_ids'])-1)
                with torch.autocast('cuda',enabled=False):
                    enc,_=model.camera_head(tokens,past_key_values_camera=cam,use_cache=True);depth,_=model.depth_head(tokens,images,START)
                torch.cuda.synchronize();duration=time.perf_counter()-t0
                if repeat:query_times.append(duration)
                del kv,cam,tokens,enc,depth,images
            records.append(dict(id=rid,condition=condition,source_ids=inp['frame_ids'][:-1],official_persistent_bytes=sum(official_bytes.values()),official_aggregator_bytes=official_bytes['kv'],official_camera_bytes=official_bytes['cam'],official_source_encoder_camera_seconds=encoder_camera,source_depth_head_seconds=depth_seconds,source_geometry_seconds=geometry_seconds,key_preparation_seconds=after_keys-after_geometry,pooled=pools,construction_peak_GiB=construction_peak,source_geometry_replay_mu_max_abs=replay_mu,source_geometry_replay_cov_max_abs=replay_cov,official_query_seconds=query_times))
            del state,inp;torch.cuda.empty_cache()
    return records


@torch.inference_mode()
def references(model,p):
    rows=[];pred=[];keys=[]
    for meta in p['catalog']:
        rid=meta['id']
        for c in p['conditions']:
            inp=load(input_path(rid,11,c))
            d,_=step(model,inp['images'][-1],11,state_new(model),True)
            rows.append(dict(id=rid,place=meta['place'],condition=c,capacity=0,kind='single_view',seed=None,**scores(d,inp)))
            pred.append(metric_depth(d));keys.append([rid,c,0,'single_view'])
            for n in CAPACITIES:
                s=load(sample_path(rid,11,c,n));d=s['first_pass_depth']
                rows.append(dict(id=rid,place=meta['place'],condition=c,capacity=n,kind='coarse_frozen',seed=None,**scores(d,inp)))
                pred.append(metric_depth(d));keys.append([rid,c,n,'coarse_frozen'])
    path=CACHE/'evaluation/references.pt';path.parent.mkdir(parents=True,exist_ok=True);save_tensor(path,dict(keys=keys,depth_metric=torch.stack(pred)))
    write_json(OUT/'evaluation/references.json',dict(complete=True,rows=rows,prediction_path=str(path),prediction_sha256=sha256(path),plan_sha256=sha256(OUT/'plan.json'),single_view='Unadapted official model, current RGB only; not a matched trained no-memory ablation'))


@torch.inference_mode()
def timing(model,p):
    seed=SEEDS[0];records=[];replay_max=0.
    for kind in KINDS:
        adapter=CoarseReader().cuda().eval();checkpoint=OUT/f'{kind}_{seed}'/'step_2000.pt';adapter.load_state_dict(load(checkpoint)['model'])
        for rid in p['training_ids'][:2]:
            for c in p['conditions']:
                inp=load(input_path(rid,11,c))
                for n in CAPACITIES:
                    cpu=load(sample_path(rid,11,c,n));s=to_device(cpu);camera=[tuple(t.cuda() for t in pair) for pair in cpu['camera_state']]
                    # Drop offline audit tensors from the resident query view.
                    s={k:v for k,v in s.items() if k not in ('source_patch_means','source_patch_covariances','membership','fps_indices','camera_state')}
                    times=[];parts=[]
                    for repeat in range(4):
                        torch.cuda.synchronize();begin=time.perf_counter()
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            image=inp['images'][-1].cuda();s['patch_tokens']=patch_tokens(model,image)
                            d,out=forward_query(model,s,checkpoint_blocks=False,return_tokens=True)
                        with torch.autocast('cuda',enabled=False):poses,_=model.camera_head(out,past_key_values_camera=list(camera),use_cache=True)
                        mu,cov=query_geometry(d,poses[-1][0,0],s['native_hw'],s['geometry_unit'],s['geometry_origin'])
                        s['query_means']=mu;s['query_covariances']=cov;z=descriptors(s,kind)
                        torch.cuda.synchronize();mid=time.perf_counter()
                        with torch.autocast('cuda',dtype=torch.bfloat16):result=forward_query(model,s,adapter,kind,z,False)
                        torch.cuda.synchronize();end=time.perf_counter()
                        # Audit transfer/check is outside the measured query interval.
                        replay_max=max(replay_max,float((d.cpu()-cpu['first_pass_depth']).abs().max()))
                        if repeat:times.append(end-begin);parts.append([mid-begin,end-mid])
                        del d,out,poses,mu,cov,z,result
                    records.append(dict(kind=kind,id=rid,condition=c,capacity=n,seconds=times,median_seconds=float(np.median(times)),first_pass_median_seconds=float(np.median(parts,axis=0)[0]),second_pass_median_seconds=float(np.median(parts,axis=0)[1])))
                    del cpu,s,camera;torch.cuda.empty_cache()
        del adapter;torch.cuda.empty_cache()
    del image
    construction=full_source_and_dense_cost(model,p)
    write_json(OUT/'runtime.json',dict(complete=True,gpu=torch.cuda.get_device_name(),records=records,median_seconds={str(n):{k:float(np.median([v for r in records if r['kind']==k and r['capacity']==n for v in r['seconds']])) for k in KINDS} for n in CAPACITIES},first_pass_cache_replay_max_abs=replay_max,full_source_construction_records=construction,official_query_median_seconds=float(np.median([v for r in construction for v in r['official_query_seconds']])),official_persistent_bytes=construction[0]['official_persistent_bytes'],full_source_construction_median_seconds={str(n):float(np.median([r['pooled'][str(n)]['full_source_construction_seconds'] for r in construction])) for n in CAPACITIES},scope='Current image transfer/backbone, frozen coarse depth+camera+query geometry/descriptor, adapted second query/depth. Sources resident; model/disk loading excluded. Full source cost additionally measured from RGB, including source depth and geometry; source images/model resident. Not bounded-memory construction or all-task VGGT timing.',runtime_source_sha256=sha256(ROOT/'experiments/gaussian_memory_coarse_runtime.py'),plan_sha256=sha256(OUT/'plan.json'),checkpoint_sha256={k:sha256(OUT/f'{k}_{seed}'/'step_2000.pt') for k in KINDS}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['references','timing']);a=parser.parse_args();torch.set_num_threads(4);p=plan();verify(p);model=model_load(p)
    references(model,p) if a.action=='references' else timing(model,p)


if __name__=='__main__':main()
