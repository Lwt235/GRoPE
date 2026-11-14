"""Frozen round015 C/P reads with a learned scalar on a prescribed address kernel."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.gaussian_scene_memory import rasterize_field
from models.native_task_writer import log6
from experiments.native_render_parameter_prepare import svec
from models.gaussian_scene_parameter_pe import global_block
from models.gaussian_memory_native_pe import grid,START
from models.gaussian_scene_native import DEPTH_LAYERS

class AddressBias(nn.Module):
    def __init__(self):
        super().__init__();self.gamma=nn.Parameter(torch.zeros(24,dtype=torch.float32))

def weights(field,query):
    K=query['K'].float().clone();K[:2]/=14
    eye=torch.eye(len(field['means']),device=K.device,dtype=torch.float32)
    image,_,_=rasterize_field(field,query['c2w'].float()[None],K[None],32,32,eye)
    return image[0].reshape(1024,-1)*field['has_content'][None]

def center_kernel(field):
    valid=field['has_content'];p=field['means'][valid].double()
    distances=torch.cdist(p,p,compute_mode='donot_use_mm_for_euclid_dist')
    nearest=distances.masked_fill(distances<=0,float('inf')).min(-1).values
    finite=nearest[torch.isfinite(nearest)]
    if finite.numel():
        d=torch.quantile(finite,.5);C=sum(torch.exp(-.5*(distances/(d*s)).square()) for s in (.5,1.,2.))/3
        median=float(d)
    else:C=torch.ones_like(distances);median=None
    full=torch.zeros((len(valid),len(valid)),device=p.device,dtype=torch.float32)
    ii=valid.nonzero().flatten();full[ii[:,None],ii[None,:]]=C.float()
    return full,dict(nearest_positive_distance_median=median,valid_centers=int(valid.sum()),all_valid_centers_coincident=not bool(finite.numel()),RBF_scales=[.5,1.,2.],RBF_convention='exp(-distance_squared/(2*scale_squared)); scales times d; three equally weighted')

def normalize(H):
    diagonal=H.diagonal();inverse=torch.where(diagonal>0,diagonal.clamp_min(torch.finfo(H.dtype).tiny).rsqrt(),torch.zeros_like(diagonal))
    return H*inverse[:,None]*inverse[None,:]

def centered_kernel(W,kind,C=None,shadow=None):
    if kind=='A':H=W@W.T
    elif kind=='P':H=(W@C)@W.T
    elif kind=='S':H=shadow@shadow.T
    else:raise ValueError(kind)
    K=normalize(H);return K-K.mean(-1,keepdim=True)

def content_view(field,query,w):
    # Same round015 unnormalized content/full logSigma and 46 center quantities.
    K=query['K'].float().clone();K[:2]/=14;cam=query['c2w'].float();unit=field['unit'].double()
    wd=w.double();alpha=wd.sum(-1,keepdim=True);den=torch.where(alpha>0,alpha,torch.ones_like(alpha))
    def avg(x):return wd@x/den
    p=(field['means'].double()-cam[:3,3].double())@cam[:3,:3].double()/unit
    mean=avg(p);fourier=[]
    for k in range(5):fourier.extend((avg(torch.sin(math.pi*2**k*p)),avg(torch.cos(math.pi*2**k*p))))
    scatter=avg(svec(p[:,:,None]*p[:,None,:]))-svec(mean[:,:,None]*mean[:,None,:])
    yy,xx=torch.meshgrid(torch.arange(32,device=w.device,dtype=torch.float64)+.5,torch.arange(32,device=w.device,dtype=torch.float64)+.5,indexing='ij')
    pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3)
    rays=F.normalize(pixels@torch.linalg.inv(K.double()).T@cam[:3,:3].double().T,dim=-1)
    delta=(field['means'].double()-cam[:3,3].double())/unit;along=rays@delta.T;radius=(delta.square().sum(-1)[None]-along.square()).clamp_min(0)
    ray=[(wd*torch.log1p(radius/(s*s))).sum(-1,keepdim=True)/den for s in (.02,.05,.1,.2,.5)]
    ray.extend(((wd*torch.asinh(along)).sum(-1,keepdim=True)/den,avg(torch.log1p(delta.norm(dim=-1))[:,None])))
    center=torch.cat((mean,*fourier,torch.asinh(scatter),*ray),-1)
    ordinary=wd@log6(field,cam)
    return dict(extra=torch.cat((w@field['features'].float(),ordinary.float()),-1),position=torch.cat((center,torch.zeros_like(ordinary)),-1).float(),alpha=alpha[:,0].float())

def sphere_field(field):
    return dict(field,scales=field['scales'].square().mean(-1,keepdim=True).sqrt().expand(-1,3).contiguous())

def forward_read(model,sample,fusion,pe,address,checkpoint_blocks=True):
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        patches=sample['patches'];extra=sample['extra'][None]
        patches=patches+fusion(patches,extra);e=pe(sample['position'],sample['alpha'])
        agg=model.aggregator;x=torch.cat((agg.camera_token[:,0],agg.register_token[:,0],patches),1);pos=grid(*sample['hw'],x.device);outputs=[None]*24
        kernel=sample['kernel'].float()
        for index in range(24):
            def pair(inp,i=index):
                frame=agg.frame_blocks[i](inp,pos=pos)
                bias=F.pad(address.gamma[i]*kernel,(START,0,START,0))[None,None]
                result=global_block(agg.global_blocks[i],frame,pos,bias,e,index=i)
                return (result,torch.cat((frame,result),-1)[:,None].float()) if i in DEPTH_LAYERS else result
            result=checkpoint(pair,x,use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else pair(x)
            if index in DEPTH_LAYERS:x,outputs[index]=result
            else:x=result
    with torch.autocast('cuda',enabled=False):
        dummy=torch.empty(1,1,3,*sample['hw'],device=x.device);depth,_=model.depth_head(outputs,dummy,START)
    return depth[0,0,...,0]
