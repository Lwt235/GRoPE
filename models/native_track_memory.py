"""One real query global block reads frozen physical GS; official two-frame tracker."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.native_scene_exchange import LowRank,continuous_rope,block
from models.gaussian_memory_native_pe import grid

LAYERS=(4,11,17,23)
class TrackMemory(nn.Module):
    def __init__(self):
        super().__init__();self.source=nn.Linear(1045,1024);self.position=nn.Sequential(nn.Linear(29,64),nn.GELU(),nn.Linear(64,1024,bias=False));self.qkv=nn.ModuleList([LowRank(1024,3072) for _ in range(2)]);self.proj=nn.ModuleList([LowRank(1024,1024) for _ in range(2)])
        with torch.no_grad():self.source.weight.zero_();self.source.bias.zero_();self.source.weight[:,:1024].copy_(torch.eye(1024));self.position[-1].weight.zero_()

def firstglobal(blk,x,prefix,adapter,sample,ip):
    att=blk.attn;z=blk.norm1(x);b,n,c=z.shape;h=att.num_heads;hd=att.head_dim
    q,k,v=(att.qkv(z)+adapter.qkv[1](z)).reshape(b,n,3,h,hd).permute(2,0,3,1,4).unbind(0)
    pk,pv=prefix;k=torch.cat((pk.flatten(2,3),k),-2);v=torch.cat((pv.flatten(2,3),v),-2)
    q,k=att.q_norm(q),att.k_norm(k);q=att.rope(q,ip);k=att.rope(k,ip.repeat(1,2,1))
    scene=blk.norm1(adapter.source(sample['ordinary']));skv=(att.qkv(scene)+adapter.qkv[1](scene)).reshape(b,-1,3,h,hd).permute(2,0,3,1,4);sk,sv=skv[1],skv[2]
    sk=att.k_norm(sk+adapter.position(sample['position']).reshape(b,-1,h,hd).transpose(1,2));sk=continuous_rope(sk,sample['scene_pos'],att.rope)
    k=torch.cat((k,sk,k.new_zeros(b,h,1,hd)),-2);v=torch.cat((v,sv,v.new_zeros(b,h,1,hd)),-2)
    mask=torch.cat((torch.ones(b,2*n,dtype=torch.bool,device=x.device),sample['key_mask'],torch.ones(b,1,dtype=torch.bool,device=x.device)),1)[:,None,None]
    y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.).transpose(1,2).reshape(b,n,c);x=x+blk.ls1(att.proj_drop(att.proj(y)+adapter.proj[1](y)));return x+blk.ls2(blk.mlp(blk.norm2(x)))

def query_tokens(model,patches,prefix,adapter=None,sample=None,checkpoint_blocks=False):
    agg=model.aggregator;x=torch.cat((agg.camera_token[:,1],agg.register_token[:,1],patches),1);ip=grid(448,448,x.device);out=[None]*24
    for i in range(24):
        def pair(a,i=i):
            frame=block(agg.frame_blocks[i],a,adapter,0,ip) if i==0 and adapter is not None else agg.frame_blocks[i](a,pos=ip)
            if i==0 and adapter is not None:current=firstglobal(agg.global_blocks[i],frame,prefix[i],adapter,sample,ip)
            else:current,_=agg.global_blocks[i](frame,pos=ip,past_key_values=prefix[i],use_cache=True)
            return (current,torch.cat((frame,current),-1)[:,None].float()) if i in LAYERS else current
        r=checkpoint(pair,x,use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else pair(x)
        if i in LAYERS:x,out[i]=r
        else:x=r
    return out

@torch.no_grad()
def native_tokens(model,patches,prefix=None,index=0):
    agg=model.aggregator;slot=0 if index==0 else 1;x=torch.cat((agg.camera_token[:,slot],agg.register_token[:,slot],patches),1);ip=grid(448,448,x.device);out=[];new=[]
    if prefix is None:prefix=[None]*24
    for i in range(24):
        f=agg.frame_blocks[i](x,pos=ip);x,kv=agg.global_blocks[i](f,pos=ip,past_key_values=prefix[i],use_cache=True);out.append(torch.cat((f,x),-1)[:,None].float());new.append(kv)
    return out,new

def fmap(model,tokens):
    with torch.autocast('cuda',enabled=False):return model.track_head.feature_extractor(tokens,torch.empty(1,1,3,448,448,device='cuda'),5)
def track(model,anchor_fmap,query_output,points):
    q=fmap(model,query_output)
    with torch.autocast('cuda',enabled=False):return model.track_head.tracker(query_points=points,fmaps=torch.cat((anchor_fmap,q),1),iters=4)
