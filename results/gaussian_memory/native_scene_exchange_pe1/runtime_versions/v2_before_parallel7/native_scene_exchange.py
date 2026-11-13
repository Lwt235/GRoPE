"""Round012: existing native blocks, ephemeral scene state, matched LoRA/masks."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.gaussian_memory_native_pe import grid


class LowRank(nn.Module):
    def __init__(self, ni, no):
        super().__init__();self.a=nn.Linear(ni,8,bias=False);self.b=nn.Linear(8,no,bias=False)
        nn.init.zeros_(self.b.weight)
    def forward(self,x):return self.b(self.a(x))


class SceneExchange(nn.Module):
    def __init__(self):
        super().__init__()
        # f1024 + query-camera geometry20 + actual per-slot visible mass1.
        self.source=nn.Linear(1045,1024)
        with torch.no_grad():
            self.source.weight.zero_();self.source.bias.zero_();self.source.weight[:,:1024].copy_(torch.eye(1024))
        self.position=nn.Sequential(nn.Linear(29,64),nn.GELU(),nn.Linear(64,1024,bias=False))
        nn.init.zeros_(self.position[-1].weight)
        self.qkv=nn.ModuleList([LowRank(1024,3072) for _ in range(4)])
        self.proj=nn.ModuleList([LowRank(1024,1024) for _ in range(4)])


def continuous_rope(tokens,positions,rope):
    # Exactly the official frequency/half-rotation, analytically extended to
    # signed, fractional projected scene coordinates. Image uses official API.
    pieces=[]
    for axis,x in enumerate(tokens.chunk(2,-1)):
        inv=rope.base_frequency**(-torch.arange(0,x.shape[-1],2,device=x.device).float()/x.shape[-1])
        angle=positions[...,axis,None].float()*inv
        angle=torch.cat((angle,angle),-1)[:,None].to(x.dtype)
        rotated=torch.cat((-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),-1)
        pieces.append(x*angle.cos()+rotated*angle.sin())
    return torch.cat(pieces,-1)


def attend(attn,z,adapter,index,image_pos,scene_pos=None,mask=None,pe=None,background=False):
    b,n,c=z.shape;h=attn.num_heads;hd=attn.head_dim
    q,k,v=(attn.qkv(z)+adapter.qkv[index](z)).reshape(b,n,3,h,hd).permute(2,0,3,1,4).unbind(0)
    ni=0 if image_pos is None else image_pos.shape[1]
    if pe is not None:
        delta=pe.reshape(b,-1,h,hd).transpose(1,2)
        k=torch.cat((k[:,:,:ni],k[:,:,ni:]+delta),-2)
    q,k=attn.q_norm(q),attn.k_norm(k)
    if ni:
        qi,ki=attn.rope(q[:,:,:ni],image_pos),attn.rope(k[:,:,:ni],image_pos)
    if scene_pos is not None:
        qs,ks=continuous_rope(q[:,:,ni:],scene_pos,attn.rope),continuous_rope(k[:,:,ni:],scene_pos,attn.rope)
        q=torch.cat((qi,qs),-2) if ni else qs;k=torch.cat((ki,ks),-2) if ni else ks
    else:q,k=qi,ki
    if background:
        k=torch.cat((k,k.new_zeros(b,h,1,hd)),-2);v=torch.cat((v,v.new_zeros(b,h,1,hd)),-2)
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.).transpose(1,2).reshape(b,n,c)
    return attn.proj_drop(attn.proj(out)+adapter.proj[index](out))


def block(blk,x,adapter,index,ip,sp=None,mask=None,pe=None,bg=False):
    x=x+blk.ls1(attend(blk.attn,blk.norm1(x),adapter,index,ip,sp,mask,pe,bg))
    return x+blk.ls2(blk.mlp(blk.norm2(x)))


def forward(model,sample,adapter,arm,checkpoint_blocks=True,trace=None):
    agg=model.aggregator;p=sample['patches'];b=p.shape[0]
    x=torch.cat((agg.camera_token[:,0].expand(b,-1,-1),agg.register_token[:,0].expand(b,-1,-1),p),1)
    ip=grid(448,448,x.device).expand(b,-1,-1);ni=x.shape[1]
    scene=adapter.source(sample['ordinary']);sp=sample['scene_pos'];valid=sample['key_mask']
    pe=adapter.position(sample['position']);ns=scene.shape[1]
    # Validity applies to keys, not the validity of native image tokens. Fixed
    # zero K/V background is always available; no trainable extra scene slot.
    sm=F.pad(valid,(0,1),value=True)[:,None,None,:]
    global_mask=torch.ones(b,1,ni+ns,ni+ns+1,dtype=torch.bool,device=x.device)
    global_mask[:,:,:,ni:ni+ns]=valid[:,None,None,:]
    if arm.startswith('R'):global_mask[:,:,ni:,:ni]=False
    outputs=[None]*24
    for i in range(24):
        if i<2:
            def pair(a,s,i=i):
                frame=block(agg.frame_blocks[i],a,adapter,2*i,ip)
                sf=block(agg.frame_blocks[i],s,adapter,2*i,None,sp,sm,bg=True)
                combined=torch.cat((frame,sf),1)
                gm=global_mask if i==0 else torch.cat((torch.ones_like(global_mask[:,:,:,:ni]),global_mask[:,:,:,ni:]),-1)
                out=block(agg.global_blocks[i],combined,adapter,2*i+1,ip,sp,gm,pe,True)
                if trace is not None:trace[f'scene_after_global_{i}']=out[:,ni:].detach().float().cpu()
                return out[:,:ni],out[:,ni:]
            x,scene=checkpoint(pair,x,scene,use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else pair(x,scene)
        else:
            def pair(a,i=i):
                frame=agg.frame_blocks[i](a,pos=ip);out=agg.global_blocks[i](frame,pos=ip)
                return (out,torch.cat((frame,out),-1)[:,None].float()) if i in (4,11,17,23) else out
            out=checkpoint(pair,x,use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else pair(x)
            if i in (4,11,17,23):x,outputs[i]=out
            else:x=out
    with torch.autocast('cuda',enabled=False):
        dummy=torch.empty(b,1,3,448,448,device=x.device);depth,_=model.depth_head(outputs,dummy,5)
    return depth[:,0,...,0]
