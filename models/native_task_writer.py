"""Source-only shared physical writer and current differentiable gsplat readout."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.gaussian_scene_memory import rasterize_field
from models.gaussian_scene_rendered import RenderedFeatureFusion
from experiments.native_render_parameter_prepare import rotation,svec


def log6(field, camera):
    unit=field['unit'].double();basis=camera[:3,:3].double().T[None]@rotation(field['rotations'])
    logcov=(basis*(2*torch.log(field['scales'].double()/unit))[:,None,:])@basis.transpose(-1,-2)
    return svec(logcov)


def qmul(a,b):
    w,x,y,z=a.unbind(-1);v,i,j,k=b.unbind(-1)
    return torch.stack((w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v),-1)


class TaskWriter(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(1034,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,1034))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,base,row):
        # API contains only a source-built field. No query or target arguments.
        cam=base['source_c2w'][0];unit=base['unit'].float();valid=base['has_content'];R=cam[:3,:3].float()
        center=(base['means'].float()-cam[:3,3])@R/unit
        x=torch.cat((base['features'].float(),center,log6(base,cam).float(),base['opacities'].float()[:,None]),-1)
        y=self.net(x);active=valid[:,None].to(y.dtype);y=y*active
        out=dict(base,features=(base['features'].float()+y[:,:1024]).half())
        # Original valid identity and slots persist; invalid opacity stays exact.
        opacity=torch.sigmoid(torch.logit(base['opacities'].float().clamp(1e-6,1-1e-6))+y[:,1024])
        out['opacities']=torch.where(valid,opacity,base['opacities'])
        if row=='J':
            out['means']=base['means'].float()+(.25*unit*torch.tanh(y[:,1025:1028]))@R.T
            out['scales']=base['scales'].float()*torch.exp(.7*torch.tanh(y[:,1028:1031]))
            v=y[:,1031:1034];norm=(v.square().sum(-1,keepdim=True)+1e-12).sqrt();v=v*((math.pi/6)*torch.tanh(norm)/norm)
            angle=(v.square().sum(-1,keepdim=True)+1e-12).sqrt();dq=torch.cat((torch.cos(angle/2),.5*torch.sinc(angle/(2*math.pi))*v),-1)
            out['rotations']=F.normalize(qmul(base['rotations'].float(),dq),dim=-1)
        return out


class ShapeFusion(RenderedFeatureFusion):
    def __init__(self):
        super().__init__();old=self.input;self.input=nn.Linear(2054,256)
        with torch.no_grad():self.input.weight.zero_();self.input.weight[:,:2048].copy_(old.weight);self.input.bias.copy_(old.bias)


def read_current(field,query,kind,trace=None):
    """Actual live basis alpha rendering; no stored or detached contribution matrix."""
    device=field['means'].device;K=query['K'].float().clone();K[:2]/=14;cam=query['c2w'].float();unit=field['unit'].double()
    identity=torch.eye(len(field['means']),device=device,dtype=torch.float32)
    image,_,_=rasterize_field(field,cam[None],K[None],32,32,identity)
    w=image[0].reshape(1024,-1)*field['has_content'][None];wd=w.double();alpha=wd.sum(-1,keepdim=True);den=torch.where(alpha>0,alpha,torch.ones_like(alpha))
    def avg(x):return (wd@x)/den
    p=(field['means'].double()-cam[:3,3].double())@cam[:3,:3].double()/unit
    mean=avg(p);fourier=[]
    for k in range(5):fourier.extend((avg(torch.sin(math.pi*2**k*p)),avg(torch.cos(math.pi*2**k*p))))
    # Equivalent centered mixture moment, without allocating pixel x slot x 3x3.
    moment=avg(svec(p[:,:,None]*p[:,None,:]));scatter=moment-svec(mean[:,:,None]*mean[:,None,:])
    yy,xx=torch.meshgrid(torch.arange(32,device=device,dtype=torch.float64)+.5,torch.arange(32,device=device,dtype=torch.float64)+.5,indexing='ij')
    pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3)
    rays=F.normalize(pixels@torch.linalg.inv(K.double()).T@cam[:3,:3].double().T,dim=-1)
    delta=(field['means'].double()-cam[:3,3].double())/unit;along=rays@delta.T;radius=(delta.square().sum(-1)[None]-along.square()).clamp_min(0)
    ray=[(wd*torch.log1p(radius/(s*s))).sum(-1,keepdim=True)/den for s in (.02,.05,.1,.2,.5)]
    ray.extend(((wd*torch.asinh(along)).sum(-1,keepdim=True)/den,avg(torch.log1p(delta.norm(dim=-1))[:,None])))
    center46=torch.cat((mean,*fourier,torch.asinh(scatter),*ray),-1)
    ordinary6=wd@log6(field,cam)
    if kind=='P':extra6=torch.zeros_like(ordinary6)
    elif kind=='G':extra6=ordinary6/den
    elif kind=='S':
        sphere=torch.log(field['scales'].double().square().sum(-1)/(3*unit.square()))[:,None,None]*torch.eye(3,device=device,dtype=torch.float64)
        extra6=avg(svec(sphere))
    else:raise ValueError(kind)
    position=torch.cat((center46,extra6),-1).float();render=w@field['features'].float()
    result=dict(extra=torch.cat((render,ordinary6.float()),-1),position=position,alpha=alpha[:,0].float())
    assert position.shape==(1024,52) and result['extra'].shape==(1024,1030)
    if trace is not None:
        for name,t in [('means',field['means']),('scales',field['scales']),('rotations',field['rotations']),('opacities',field['opacities']),('features',field['features']),('ordinary6',ordinary6),('position',position)]:
            if t.requires_grad:t.register_hook(lambda g,n=name:trace.__setitem__(n,float(g.float().norm())))
    return result
