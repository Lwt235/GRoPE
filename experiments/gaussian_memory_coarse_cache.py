"""Build all-layer pooled K/V, then obtain query geometry using only pooled state."""
import argparse
import json
import time
import numpy as np
import torch
from models.gaussian_memory_coarse_pe import group_geometry, patch_tokens, forward_query, query_geometry, grid, START
from experiments.gaussian_memory_coarse_data import ROOT, OUT, CACHE, PARENT_OUT, PARENT_CACHE, CAPACITIES, plan, verify, sample_path, keyname, example_keys, load, read, write_json, save_tensor, sha256, input_path
from experiments.gaussian_memory_native_run import model_load, state_new, scores, metric_depth


def to_device(cpu):
    names=('patch_tokens','keys','values','source_means','source_covariances','source_special','log_mass','query_means','query_covariances','target','valid','hidden')
    return {k:(v.cuda(non_blocking=True) if k in names and isinstance(v,torch.Tensor) else v) for k,v in cpu.items()}


@torch.inference_mode()
def dense_processed(model, state, hw):
    """Native cache is BEFORE QK normalization/RoPE; transform before pooling."""
    pos=grid(*hw,'cuda'); keys=[]; values=[]
    for i,(k,v) in enumerate(state['kv']):
        attn=model.aggregator.global_blocks[i].attn
        k=k.flatten(2,3); v=v.flatten(2,3)
        k=attn.rope(attn.k_norm(k),pos.repeat(1,k.shape[-2]//pos.shape[1],1))
        keys.append(k);values.append(v)
    return keys,values


@torch.inference_mode()
def build_one(model, rid,q,condition,p):
    marks=[OUT/'cache'/(keyname(rid,q,condition,n)+'.json') for n in CAPACITIES]
    if all(m.exists() for m in marks):
        for m in marks:
            r=read(m); assert r['plan_sha256']==sha256(OUT/'plan.json') and sha256(r['path'])==r['sha256']
        return
    started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    inpfile=input_path(rid,q,condition); assert sha256(inpfile)==p['input_records'][rid]['files'][str(inpfile)]
    inp=load(inpfile); oldfile=PARENT_CACHE/rid/f'q{q}_{condition}.pt'
    oldrec=read(PARENT_OUT/'cache'/(f'{rid}_q{q}_{condition}.json'))
    assert sha256(oldfile)==oldrec['sha256']
    old=load(oldfile); hw=list(inp['images'].shape[-2:]); state=state_new(model)
    for index,image in enumerate(inp['images'][:-1]):
        image=image[None,None].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16): tokens,_,state['kv']=model.aggregator(image,past_key_values=state['kv'],use_cache=True,past_frame_idx=index)
        with torch.autocast('cuda',enabled=False): _,state['cam']=model.camera_head(tokens,past_key_values_camera=state['cam'],use_cache=True)
    source_seconds=time.monotonic()-started
    with torch.autocast('cuda',dtype=torch.bfloat16):
        keys,values=dense_processed(model,state,hw)
        patches=patch_tokens(model,inp['images'][-1].cuda())
    nframe=len(inp['frame_ids'])-1; ntokens=keys[0].shape[-2]//nframe
    special=(torch.arange(nframe*ntokens,device='cuda')%ntokens)<START
    means=old['means'][:-ntokens][~special.cpu()].cuda(); cov=old['covariances'][:-ntokens][~special.cpu()].cuda()
    camera=[tuple(v.cpu() for v in pair) for pair in state['cam']]
    camera_bytes=sum(v.numel()*v.element_size() for pair in camera for v in pair)
    sources=[]
    for capacity in CAPACITIES:
        member,mass,mu,sigma,chosen=group_geometry(means,cov,capacity)
        # Matrix contraction is deterministic, with FP32 accumulation then BF16 storage.
        weights=torch.nn.functional.one_hot(member,capacity).float().T/mass[:,None]
        pooled_k=[];pooled_v=[]
        for k,v in zip(keys,values):
            def pool(value):
                selected=value[...,~special,:].float()
                with torch.autocast('cuda',enabled=False): avg=torch.einsum('mn,bhnd->bhmd',weights,selected)
                return torch.cat((value[...,special,:],avg.to(value.dtype)),-2).cpu()
            pooled_k.append(pool(k));pooled_v.append(pool(v))
        nsp=int(special.sum())
        source={'keys':torch.stack(pooled_k),'values':torch.stack(pooled_v),
            'source_means':torch.cat((torch.zeros(nsp,3,device='cuda'),mu)).cpu(),
            'source_covariances':torch.cat((torch.eye(3,device='cuda').expand(nsp,3,3),sigma)).cpu(),
            'source_special':torch.arange(nsp+capacity)<nsp,
            'log_mass':torch.cat((torch.zeros(nsp,device='cuda'),mass.float().log())).cpu(),
            'member_counts':mass.cpu(),'membership':member.cpu(),'fps_indices':chosen,
            'source_patch_means':means.cpu(),'source_patch_covariances':cov.cpu(),
            'capacity':capacity,'native_hw':hw,'patch_tokens':patches.cpu(),
            'id':rid,'query':q,'condition':condition,'place':inp['place'],'split':inp['split'],
            'source_ids':inp['frame_ids'][:-1],'target':inp['target'],'valid':inp['valid'],'hidden':inp['hidden'],
            'geometry_unit':old['geometry_unit'],'geometry_origin':old['geometry_origin']}
        sources.append(source)
    del state,keys,values,tokens,old,means,cov,weights,k,v,patches
    torch.cuda.empty_cache()  # No dense visual memory remains in either query pass.
    for source,mark in zip(sources,marks):
        sample=to_device(source)
        with torch.autocast('cuda',dtype=torch.bfloat16): d,outputs=forward_query(model,sample,return_tokens=True,checkpoint_blocks=False)
        cam=[tuple(v.cuda() for v in pair) for pair in camera]
        with torch.autocast('cuda',enabled=False): poses,_=model.camera_head(outputs,past_key_values_camera=cam,use_cache=True)
        query_mu,query_cov=query_geometry(d,poses[-1][0,0],hw,source['geometry_unit'],source['geometry_origin'])
        source.update(query_means=query_mu.cpu(),query_covariances=query_cov.cpu(),first_pass_depth=d.cpu(),
            first_pass_pose=poses[-1][0,0].cpu(),camera_state=camera)
        persistent=sum(source[k].numel()*source[k].element_size() for k in ('keys','values','source_means','source_covariances','source_special','log_mass','member_counts'))+camera_bytes
        path=sample_path(rid,q,condition,source['capacity']);path.parent.mkdir(parents=True,exist_ok=True);save_tensor(path,source)
        write_json(mark,{'complete':True,'path':str(path),'sha256':sha256(path),'plan_sha256':sha256(OUT/'plan.json'),
            'key':[rid,q,condition,source['capacity']],'input_sha256':sha256(inpfile),'parent_cache_sha256':oldrec['sha256'],
            'source_ids':source['source_ids'],'persistent_bytes':persistent,'camera_bytes':camera_bytes,
            'source_build_seconds':source_seconds,'total_build_seconds':time.monotonic()-started,
            'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'no_dense_query_prefix':True,'no_query_truth_geometry':True,
            'first_pass_scores':scores(d.cpu(),source)})
        del sample,outputs,d,poses,cam,query_mu,query_cov
    print(json.dumps({'cache_complete':[rid,q,condition],'seconds':time.monotonic()-started}),flush=True)
    del sources;torch.cuda.empty_cache()


def main():
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1);a.add_argument('--pilot',action='store_true');args=a.parse_args()
    torch.set_num_threads(4);p=plan();verify(p);model=model_load(p)
    keys=example_keys(p,'train')+example_keys(p,'development')
    keys=list(dict.fromkeys(tuple(k[:3]) for k in keys))
    if args.pilot:keys=[(rid,11,'persistent') for rid in p['training_ids'][:2]]
    for key in keys[args.shard::args.shards]:build_one(model,*key,p)
    write_json(OUT/f'cache_worker_{args.shard}_{args.shards}_{args.pilot}.json',dict(complete=True,keys=keys[args.shard::args.shards]))


if __name__=='__main__':main()
