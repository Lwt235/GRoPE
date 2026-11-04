"""R69's compact relative PE in an official-LVSM-derived image Transformer.

No target image or target depth in this interface. Additional relation applies
only to reference tokens; all query tokens retain public camera geometry.
"""
import math
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'third_party/round70_prope'))
from nvs.lvsm import LVSMDecoderOnlyModel, LVSMDecoderOnlyModelConfig
from prope.utils.functional import Camera, patchify, unpatchify
from prope.utils.transformer import TransformerEncoderConfig, TransformerEncoderLayerConfig
from prope.torch import _rope_apply_coeffs
from models.round62_rgb_relations import gaussian_from_raw
from models.round50_local_frame import rotation6

ROLES = ('rgb_raymap', 'rgb_prope', 'rgb_distance', 'rgb_direct', 'rgb_relative5',
         'rgb_hrope', 'observed_prope', 'observed_distance', 'observed_direct', 'observed_relative5')
HEIGHT, WIDTH, PATCH, DIM, HEADS, HD = 96, 128, 8, 192, 4, 48
TOKENS = (HEIGHT//PATCH)*(WIDTH//PATCH)
BASE_FIELDS = ('ref_rgb', 'K', 'c2w')
OBSERVED_FIELDS = ('ref_mean', 'ref_cov', 'ref_coverage')


class _SymmetricLog(torch.autograd.Function):
    """Spectral matrix log with its divided-difference derivative.

Unlike differentiating eigenvectors, the matrix-function derivative stays
defined at repeated eigenvalues (e.g. isotropic covariance).
    """
    @staticmethod
    def forward(ctx, covariance):
        values, vectors = torch.linalg.eigh(covariance.double())
        values = values.clamp_min(1e-15)
        ctx.save_for_backward(values, vectors)
        ctx.dtype = covariance.dtype
        return ((vectors*(.5*(values/.04).log()).unsqueeze(-2))@vectors.transpose(-1,-2)).to(covariance.dtype)

    @staticmethod
    def backward(ctx, gradient):
        values, vectors = ctx.saved_tensors
        left, right = values.unsqueeze(-1), values.unsqueeze(-2)
        diff = left-right
        close = diff.abs() <= 1e-7*(left+right)
        denominator = torch.where(close, torch.ones_like(diff), diff)
        divided = .5*torch.log1p(diff/right)/denominator
        divided = torch.where(close, 1/(left+right), divided)
        g = gradient.double(); g = (g+g.transpose(-1,-2))*.5
        inner = vectors.transpose(-1,-2)@g@vectors
        return (vectors@(inner*divided)@vectors.transpose(-1,-2)).to(ctx.dtype)


def shape_features(covariance):
    matrix = _SymmetricLog.apply(covariance)
    ij = torch.triu_indices(3,3,device=covariance.device)
    weights = torch.where(ij[0]==ij[1], 1., math.sqrt(2.)).to(matrix)
    return matrix[...,ij[0],ij[1]]*weights


def descriptors(mean, covariance, kind):
    delta = mean.unsqueeze(-2)-mean.unsqueeze(-3)
    distance = delta.norm(dim=-1,keepdim=True)/.2
    if kind == 'distance': return distance.contiguous()
    if kind == 'direct':
        sh = shape_features(covariance)
        n = mean.shape[1]
        return torch.cat((delta/.2,distance,sh[:,:,None].expand(-1,-1,n,-1),
                          sh[:,None].expand(-1,n,-1,-1)), -1)
    if kind != 'relative5': raise ValueError(kind)
    d = delta.double(); s = covariance.double(); p = torch.linalg.inv(s)
    mi = torch.einsum('bijd,bide,bije->bij', d,p,d)
    mj = torch.einsum('bijd,bjde,bije->bij', d,p,d)
    ti = torch.einsum('bide,bjed->bij', p,s)
    tj = torch.einsum('bide,bjed->bij', s,p)
    terms = torch.stack((mi,mj,ti,tj),-1).clamp_min(0).log1p().to(mean.dtype)
    return torch.cat((distance,terms),-1)


def hrope_matrices(raw, intrinsics):
    """Paper A.5 and Sherman-Morrison; no principal-point omission."""
    r = rotation6(raw[...,:6]); rt = r.transpose(-1,-2)
    t = raw[...,6:9].clamp(-.5,.5)
    normal = F.normalize(raw[...,9:12],dim=-1)
    kappa = raw[...,12].clamp(-.5,.5)
    a = r+kappa[...,None,None]*t[..., :,None]*normal[...,None,:]
    u = (rt@t[...,None])[...,0]
    v = (normal[...,None,:]@rt)[...,0,:]
    denom = (1+kappa*(normal*u).sum(-1)).clamp_min(.1)
    ai = rt-kappa[...,None,None]*u[..., :,None]*v[...,None,:]/denom[...,None,None]
    return intrinsics@a.clamp(-5,5), ai.clamp(-5,5)@torch.linalg.inv(intrinsics)


class CompactNVS(LVSMDecoderOnlyModel):
    def __init__(self, role):
        if role not in ROLES: raise ValueError(role)
        cfg = LVSMDecoderOnlyModelConfig(ref_views=2, tar_views=1,
            img_shape=(HEIGHT,WIDTH,3),cam_shape=(HEIGHT,WIDTH,6),patch_size=PATCH,
            ray_encoding='plucker', pos_enc='prope',
            encoder=TransformerEncoderConfig(layer=TransformerEncoderLayerConfig(
                d_model=DIM,nhead=HEADS,dim_feedforward=768,dropout=0.,activation=F.relu,
                batch_first=True,norm_first=True,bias=False,qk_norm=True),
                num_layers=6,input_norm=True,output_norm=True,checkpointing=False))
        super().__init__(cfg)
        self.role = role; self.kind = role.split('_',1)[1]
        self.observed = role.startswith('observed_')
        self.geometry_embed = nn.Linear(10,DIM,bias=False)
        nn.init.normal_(self.geometry_embed.weight,std=.02)
        # All common random draws precede role-specific modules.
        if not self.observed:
            self.geometry_head = nn.Linear(DIM,7)
            nn.init.zeros_(self.geometry_head.weight); nn.init.zeros_(self.geometry_head.bias)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed()+70000)
            if self.kind in ('distance','direct','relative5'):
                self.relative = nn.Sequential(nn.Linear({'distance':1,'direct':16,'relative5':5}[self.kind],96),
                                              nn.GELU(),nn.Linear(96,DIM,bias=False))
            if self.kind == 'hrope':
                self.hfield = nn.Linear(DIM,13)
                nn.init.zeros_(self.hfield.weight); nn.init.zeros_(self.hfield.bias)
                with torch.no_grad():
                    self.hfield.bias[:6].copy_(torch.tensor([1.,0,0,0,1.,0]))
                    self.hfield.bias[11] = 1.; self.hfield.bias[12] = .25

    def geometry(self, x, data):
        if self.observed:
            mean,cov,coverage = [data[key].float() for key in OBSERVED_FIELDS]
        else:
            raw = self.geometry_head(x).float()
            with torch.autocast(device_type=x.device.type,enabled=False):
                y,xx = torch.meshgrid((torch.arange(HEIGHT//PATCH,device=x.device)+.5)*PATCH,
                                     (torch.arange(WIDTH//PATCH,device=x.device)+.5)*PATCH,indexing='ij')
                pix = torch.stack((xx.flatten(),y.flatten(),torch.ones(TOKENS,device=x.device)),-1)
                rays = torch.einsum('bvij,nj->bvni', torch.linalg.inv(data['K'][:,:2].float()),pix)
                mean,cov = gaussian_from_raw(raw,rays)
                coverage = mean.new_ones(mean.shape[:-1])
        with torch.autocast(device_type=x.device.type,enabled=False):
            rotation = data['c2w'][:,:2,:3,:3].float()
            translation = data['c2w'][:,:2,:3,3].float()
            mean = torch.einsum('bvij,bvnj->bvni',rotation,mean)+translation[:,:,None]
            cov = rotation[:,:,None]@cov@rotation[:,:,None].transpose(-1,-2)
            return mean.flatten(1,2),cov.flatten(1,2),coverage.flatten(1,2)

    def _hrope_transport(self,value,matrix,inverse_rope):
        geom = value[...,:24].reshape(*value.shape[:-1],8,3)
        geom = torch.einsum('btij,bhtkj->bhtki',matrix.to(value),geom).flatten(-2)
        ax = self.attention
        rx = _rope_apply_coeffs(value[...,24:36],(ax.coeffs_x_0,ax.coeffs_x_1),inverse=inverse_rope)
        ry = _rope_apply_coeffs(value[...,36:48],(ax.coeffs_y_0,ax.coeffs_y_1),inverse=inverse_rope)
        return torch.cat((geom,rx,ry),-1)

    def forward(self, data, disable_relation=False):
        expected = BASE_FIELDS+(OBSERVED_FIELDS if self.observed else ())
        if set(data) != set(expected):
            raise ValueError(f'inference inputs must be exactly {expected}, received {tuple(data)}')
        rgb,K,c2w = data['ref_rgb'],data['K'],data['c2w']
        assert rgb.shape[1:] == (2,HEIGHT,WIDTH,3)
        cams = Camera(K,c2w,WIDTH,HEIGHT)
        with torch.autocast(device_type=rgb.device.type,enabled=False):
            rays = self.create_rays(cams)
        rays = patchify(rays,PATCH)
        x = self.input_tokenizer(torch.cat((patchify(rgb,PATCH),rays[:,:2]),-1))
        mean,cov,coverage = self.geometry(x,data)
        with torch.autocast(device_type=rgb.device.type,enabled=False):
            content = torch.cat((mean,shape_features(cov),coverage[...,None]),-1)
        x = x.flatten(1,2)+self.geometry_embed(content)
        qtokens = self.query_tokenizer(rays[:,2])
        tokens = torch.cat((x,qtokens),1)
        relative = None
        if self.kind in ('distance','direct','relative5') and not disable_relation:
            with torch.autocast(device_type=rgb.device.type,enabled=False):
                desc = descriptors(mean,cov,self.kind)
            relative = self.relative(desc).reshape(len(rgb),2*TOKENS,2*TOKENS,HEADS,HD)
        if self.kind not in ('raymap','hrope'):
            with torch.autocast(device_type=rgb.device.type,enabled=False):
                self.attention._precompute_and_cache_apply_fns(torch.linalg.inv(c2w),K)
        if self.kind == 'hrope':
            raw = self.hfield(x).float()
            with torch.autocast(device_type=rgb.device.type,enabled=False):
                kn = K.clone().float()
                kn[...,0,:] /= WIDTH; kn[...,1,:] /= HEIGHT
                kn[...,:2,2] -= .5
                kref = kn[:,:2,None].expand(-1,-1,TOKENS,-1,-1).flatten(1,2)
                hr,hi = hrope_matrices(raw,kref)
                ht = kn[:,2,None].expand(-1,TOKENS,-1,-1)
                h = torch.cat((hr,ht),1)
                hinv = torch.cat((hi,torch.linalg.inv(ht)),1)

        def attention(q,k,v,**kwargs):
            if self.kind == 'raymap': return F.scaled_dot_product_attention(q,k,v,**kwargs)
            if self.kind == 'hrope':
                # Paper diag(H,Phi): transpose/inverse use negative phase.
                out = F.scaled_dot_product_attention(
                    self._hrope_transport(q,h.transpose(-1,-2),True),
                    self._hrope_transport(k,hinv,True),
                    self._hrope_transport(v,hinv,True),**kwargs)
                return self._hrope_transport(out,h,False)
            mask = None
            if relative is not None:
                bias = torch.einsum('bhid,bijhd->bhij',q[:,:,:2*TOKENS],relative)/math.sqrt(HD)
                mask = F.pad(bias,(0,TOKENS,0,TOKENS))
            out = F.scaled_dot_product_attention(self.attention._apply_to_q(q),
                self.attention._apply_to_kv(k),self.attention._apply_to_kv(v),attn_mask=mask,**kwargs)
            return self.attention._apply_to_o(out)

        output = self.encoder(tokens,sdpa_fn=attention)
        return unpatchify(self.output_layer(output[:,-TOKENS:]),HEIGHT,WIDTH,PATCH)
