"""One shared parameter PE in existing global Q/K; Render content and V retained."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.gaussian_memory_native_pe import grid,START
from models.gaussian_scene_native import DEPTH_LAYERS


class ParameterPE(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(52,64),nn.GELU(),nn.Linear(64,1024,bias=False))
        nn.init.zeros_(self.network[-1].weight)

    def forward(self,position,alpha):
        return self.network(position.float())*alpha.reshape(-1,1).float()


def global_block(block,x,pos,mask,e,trace=None,index=0):
    attention=block.attn;inp=block.norm1(x);b,n,c=inp.shape
    qkv=attention.qkv(inp).reshape(b,n,3,attention.num_heads,attention.head_dim).permute(2,0,3,1,4)
    q,k,v=qkv.unbind(0)
    added=F.pad(e,(0,0,START,0))[None].reshape(b,n,attention.num_heads,attention.head_dim).transpose(1,2).to(q.dtype)
    q=q+added;k=k+added
    if trace is not None and index==0:
        trace.update(qk_increment_norm=float(added.detach().float().norm()),special_increment_max=float(added[:,:,:START].detach().abs().max()),V_direct_max_abs=0.,insertion='after qkv, before q_norm/k_norm and original rope')
    q,k=attention.q_norm(q),attention.k_norm(k)
    if attention.rope is not None:q=attention.rope(q,pos);k=attention.rope(k,pos)
    assert attention.fused_attn and not attention.training
    result=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
    result=attention.proj_drop(attention.proj(result.transpose(1,2).reshape(b,n,c)))
    x=x+block.ls1(result)
    return x+block.ls2(block.mlp(block.norm2(x)))


def forward_read(model,sample,fusion,pe,checkpoint_blocks=True,trace=None):
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        patches=sample['patches'];extra=sample['extra'][None] if sample['extra'].ndim==2 else sample['extra']
        patches=patches+fusion(patches,extra);e=pe(sample['position'],sample['alpha'])
        if trace is not None and e.requires_grad:e.register_hook(lambda g:trace.__setitem__('PE_output_gradient',float(g.float().norm())))
        agg=model.aggregator;x=torch.cat((agg.camera_token[:,0],agg.register_token[:,0],patches),1);pos=grid(*sample['hw'],x.device);mask=x.new_zeros(x.shape[1],x.shape[1]);outputs=[None]*24
        for index in range(24):
            def pair(inp,i=index):
                frame=agg.frame_blocks[i](inp,pos=pos);result=global_block(agg.global_blocks[i],frame,pos,mask,e,trace,i)
                return (result,torch.cat((frame,result),-1)[:,None].float()) if i in DEPTH_LAYERS else result
            result=checkpoint(pair,x,use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else pair(x)
            if index in DEPTH_LAYERS:x,outputs[index]=result
            else:x=result
    with torch.autocast('cuda',enabled=False):
        dummy=torch.empty(1,1,3,*sample['hw'],device=x.device);depth,_=model.depth_head(outputs,dummy,START)
    return depth[0,0,...,0]
