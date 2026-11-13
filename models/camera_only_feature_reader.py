"""Round006 camera-only reader. No image, teacher, depth, mask or ID input API."""
import torch
from torch import nn
import torch.nn.functional as F


class CameraOnlyFeatureReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = nn.Sequential(nn.Linear(1044,128),nn.GELU(),nn.LayerNorm(128))
        self.key = nn.Linear(128,128)
        self.value = nn.Linear(128,128)
        self.query = nn.Sequential(nn.Linear(1045,128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,128))
        self.position = nn.Sequential(nn.Linear(29,64),nn.GELU(),nn.Linear(64,128,bias=False))
        self.decoder = nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,1024))
        nn.init.zeros_(self.position[-1].weight)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode_source(self, ordinary):
        h=self.source(ordinary)
        return self.key(h),self.value(h)

    def read(self, anchor, alpha, camera20, common25, extra4, key_mask, key, value):
        single=anchor.ndim==2
        if single:
            anchor,alpha,camera20,common25,extra4,key_mask,key,value=[x[None] for x in (anchor,alpha,camera20,common25,extra4,key_mask,key,value)]
        q=self.query(torch.cat((F.layer_norm(anchor,(1024,)),camera20,torch.log1p(alpha)),dim=-1))
        k=key+self.position(torch.cat((common25,extra4),dim=-1))
        # Always-present zero background prevents undefined all-masked softmax.
        k=torch.cat((k,k.new_zeros(k.shape[0],1,128)),1)
        v=torch.cat((value,value.new_zeros(value.shape[0],1,128)),1)
        mask=torch.cat((key_mask,torch.ones(key_mask.shape[0],1,dtype=torch.bool,device=key_mask.device)),1)
        b=q.shape[0]
        heads=lambda x:x.reshape(b,-1,4,32).transpose(1,2)
        out=F.scaled_dot_product_attention(heads(q),heads(k),heads(v),attn_mask=mask[:,None,None,:],dropout_p=0.)
        out=out.transpose(1,2).reshape(b,1024,128)
        prediction=anchor+self.decoder(torch.cat((q,out),-1))
        return prediction[0] if single else prediction


def anchor_from_composite(render, alpha):
    alpha=alpha.reshape(-1,1).float()
    return torch.where(alpha>1e-8,render.float()/alpha.clamp_min(1e-8),0.),alpha


def harmonics(uv):
    terms=[]
    for frequency in (1,2,4):
        terms.extend((torch.sin(torch.pi*frequency*uv),torch.cos(torch.pi*frequency*uv)))
    return torch.cat(terms,-1)


def footprint(A, physical_camera_center, intrinsic):
    x,y,z=physical_camera_center.unbind(-1)
    zero=torch.zeros_like(z)
    fx,fy=intrinsic[0,0],intrinsic[1,1]
    J=torch.stack((torch.stack((fx/z,zero,-fx*x/z.square()),-1),torch.stack((zero,fy/z,-fy*y/z.square()),-1)),-2)
    C=J@A@J.transpose(-1,-2)+.3*torch.eye(2,device=A.device,dtype=A.dtype)
    a,b,c=C[...,0,0],C[...,0,1],C[...,1,1]
    norm=torch.hypot(a-c,2*b)
    major=(a+c+norm)/2
    minor=((a+c-norm)/2).clamp_min(torch.finfo(A.dtype).tiny)
    isotropic=norm<1e-12
    co=torch.where(isotropic,torch.ones_like(norm),(a-c)/norm.clamp_min(1e-12))
    si=torch.where(isotropic,torch.zeros_like(norm),2*b/norm.clamp_min(1e-12))
    return torch.stack((.5*major.log(),.5*minor.log(),co,si),-1)


def view_fields(geometry, camera, intrinsic):
    """Only saved source geometry and a query camera are accepted."""
    cam=camera.double();K=intrinsic.double().clone();K[:2]/=14
    R=cam[:3,:3];u=geometry['unit'].double()
    cp=(geometry['means'].double()-cam[:3,3])@R
    valid=geometry['valid']&(cp[:,2]>.01)
    placeholder=cp.new_tensor([0.,0.,1.])
    safe=torch.where(valid[:,None],cp,placeholder)
    t=safe/u
    pix=safe@K.T;uv=2*(pix[:,:2]/pix[:,2:])/32-1
    B=R.T[None]@geometry['B_world'].double()@R[None]
    cov=R.T[None]@geometry['cov_world'].double()@R[None]
    sphere=torch.diagonal(cov,dim1=-2,dim2=-1).sum(-1)[:,None,None]/3*torch.eye(3,device=cp.device,dtype=cp.dtype)
    common=torch.cat((torch.asinh(t),uv,torch.log1p(t.norm(dim=-1))[:,None],F.normalize(t,dim=-1),harmonics(uv),footprint(B,safe,K)),-1)
    yy,xx=torch.meshgrid(torch.arange(32,device=cp.device,dtype=cp.dtype)+.5,torch.arange(32,device=cp.device,dtype=cp.dtype)+.5,indexing='ij')
    pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3)
    R0=geometry['source_camera'].double()[:3,:3];o0=geometry['source_camera'].double()[:3,3]
    origin=torch.asinh((cam[:3,3]-o0)@R0/u).expand(1024,3)
    rays=F.normalize(pixels@torch.linalg.inv(K).T@R.T@R0,dim=-1)
    xy=2*pixels[:,:2]/32-1
    camera20=torch.cat((origin,rays,xy,harmonics(xy)),-1)
    result=dict(camera20=camera20.float(),common25=common.float(),P=geometry['density4'].float(),S=footprint(sphere,safe,K).float(),G=footprint(cov,safe,K).float(),key_mask=valid)
    assert result['common25'].shape[-1]==25 and result['camera20'].shape[-1]==20
    assert all(torch.isfinite(x).all() for x in result.values())
    return result
