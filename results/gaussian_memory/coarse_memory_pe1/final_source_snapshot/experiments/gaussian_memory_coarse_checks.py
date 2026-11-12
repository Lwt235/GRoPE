"""Independent moment arithmetic and official dense-query replay on old inputs."""
import json
import numpy as np
import torch
from experiments.gaussian_memory_coarse_data import *
from experiments.gaussian_memory_coarse_cache import dense_processed,to_device
from experiments.gaussian_memory_native_run import model_load,state_new
from models.gaussian_memory_coarse_pe import forward_query,patch_tokens,START


@torch.inference_mode()
def main():
    torch.set_num_threads(4);p=plan();verify(p);model=model_load(p);checks=[]
    for rid in p['training_ids'][:2]:
        for capacity in CAPACITIES:
            s=load(sample_path(rid,11,'persistent',capacity));member=s['membership'].numpy();mu=s['source_patch_means'].numpy().astype(np.float64);cov=s['source_patch_covariances'].numpy().astype(np.float64)
            mass=s['member_counts'].numpy();nsp=len(s['source_means'])-capacity
            gm=np.zeros((capacity,3));np.add.at(gm,member,mu);gm/=mass[:,None]
            delta=mu-gm[member];gc=np.zeros((capacity,3,3));np.add.at(gc,member,cov+delta[:,:,None]*delta[:,None,:]);gc/=mass[:,None,None]
            em=float(np.max(np.abs(gm-s['source_means'][nsp:].numpy())));ec=float(np.max(np.abs(gc-s['source_covariances'][nsp:].numpy())))
            assert em<2e-5 and ec<2e-5 and (mass>0).all() and mass.sum()==len(member)
            assert s['keys'].shape==(24,1,16,capacity+nsp,64) and not any(k in s for k in ('prefix','early','memory'))
            assert s['source_ids']==[0,1,2,3,4,6,8,10]
            checks.append(dict(id=rid,capacity=capacity,mu_max_abs=em,cov_max_abs=ec,members=int(mass.sum()),special_tokens=nsp))
    # Validate the all-layer query implementation against untouched native attention.
    inp=load(input_path(p['training_ids'][0],11,'persistent'));state=state_new(model);hw=list(inp['images'].shape[-2:])
    for index,image in enumerate(inp['images'][:2]):
        with torch.autocast('cuda',dtype=torch.bfloat16):_,_,state['kv']=model.aggregator(image[None,None].cuda(),past_key_values=state['kv'],use_cache=True,past_frame_idx=index)
    image=inp['images'][2].cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        k,v=dense_processed(model,state,hw);patch=patch_tokens(model,image)
        n=k[0].shape[-2];s={'keys':torch.stack(k),'values':torch.stack(v),'patch_tokens':patch,'log_mass':torch.zeros(n,device='cuda'),'native_hw':hw}
        replay=forward_query(model,s,checkpoint_blocks=False)
        tokens,_,_=model.aggregator(image[None,None],past_key_values=state['kv'],use_cache=True,past_frame_idx=2)
    with torch.autocast('cuda',enabled=False):official,_=model.depth_head(tokens,image[None,None],START)
    delta=replay-official[0,0,...,0];rms=float(delta.square().mean().sqrt()/official.square().mean().sqrt());assert rms<.001,rms
    # Diagnose the small BF16 replay gap without changing the scientific operator:
    # dense native SDPA has no mask, while the coarse count operator has one.
    # Give untouched native global attention an identically zero mask as a control.
    original_sdpa=torch.nn.functional.scaled_dot_product_attention
    def zero_mask_sdpa(query,key,value,attn_mask=None,**kwargs):
        if attn_mask is None and key.shape[-2]>query.shape[-2]:
            attn_mask=torch.zeros(query.shape[-2],key.shape[-2],device=query.device,dtype=query.dtype)
        return original_sdpa(query,key,value,attn_mask=attn_mask,**kwargs)
    # The previous official query appended to its local list; use its original two-frame prefix.
    old_prefix=[tuple(t[:,:,:2].contiguous() for t in pair) for pair in state['kv']]
    try:
        torch.nn.functional.scaled_dot_product_attention=zero_mask_sdpa
        with torch.autocast('cuda',dtype=torch.bfloat16): tokens_z,_,_=model.aggregator(image[None,None],past_key_values=old_prefix,use_cache=True,past_frame_idx=2)
    finally:torch.nn.functional.scaled_dot_product_attention=original_sdpa
    with torch.autocast('cuda',enabled=False):official_z,_=model.depth_head(tokens_z,image[None,None],START)
    zero_delta=replay-official_z[0,0,...,0]
    zero_rms=float(zero_delta.square().mean().sqrt()/official_z.square().mean().sqrt())
    # One ordinary implementation diagnostic, independent of task quality.
    torch.manual_seed(20261111);q=torch.randn(1,2,3,8,device='cuda');k=torch.randn(1,2,5,8,device='cuda');v=torch.randn_like(k)
    zero=torch.zeros(1,2,3,5,device='cuda');bias=zero.clone();bias[:,:,:,0]=4
    a=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=zero);b=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=bias)
    influence=float((a-b).abs().max());assert influence>.01
    write_json(OUT/'independent_input_checks.json',dict(complete=True,moments=checks,official_dense_query_relative_rms=rms,official_dense_query_max_abs=float(delta.abs().max()),official_with_zero_global_mask_relative_rms=zero_rms,official_with_zero_global_mask_max_abs=float(zero_delta.abs().max()),nonzero_bias_changes_attention_max_abs=influence,scope='Implementation only; old training cases; no task gain requirement',plan_sha256=sha256(OUT/'plan.json')))
    print(json.dumps(read(OUT/'independent_input_checks.json')),flush=True)


if __name__=='__main__':main()
