"""Exact-form FP32 two-head relation contraction; private fixed-geometry models."""
import math
from types import MethodType
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from models.factored_relation_runtime import ProjectedRelation
from models.round58_support_pe import HEADS,HD,DIM
from experiments.efficiency_sidecar.fused_hidden_grad import LayerRelations


@triton.jit
def forward_kernel(Q,Z,O, I:tl.constexpr,J:tl.constexpr,K:tl.constexpr,
                   QB:tl.constexpr,QH:tl.constexpr,QI:tl.constexpr,QK:tl.constexpr,
                   ZB:tl.constexpr,ZI:tl.constexpr,ZJ:tl.constexpr,ZK:tl.constexpr,
                   BK:tl.constexpr,BJ:tl.constexpr):
    row=tl.program_id(0);b=row//I;i=row%I
    j=tl.program_id(1)*BJ+tl.arange(0,BJ)
    k=tl.arange(0,BK)
    z=tl.load(Z+b*ZB+i*ZI+j[:,None]*ZJ+k[None,:]*ZK,(j[:,None]<J)&(k[None,:]<K),0)
    q0=tl.load(Q+b*QB+i*QI+k*QK,k<K,0)
    q1=tl.load(Q+b*QB+QH+i*QI+k*QK,k<K,0)
    s0=tl.sum(z*q0[None,:],1);s1=tl.sum(z*q1[None,:],1)
    tl.store(O+row*2*J+j,s0,j<J)
    tl.store(O+row*2*J+J+j,s1,j<J)


@triton.jit
def hidden_backward_kernel(Q,G,D, I:tl.constexpr,J:tl.constexpr,K:tl.constexpr,
                   QB:tl.constexpr,QH:tl.constexpr,QI:tl.constexpr,QK:tl.constexpr,
                   GB:tl.constexpr,GH:tl.constexpr,GI:tl.constexpr,GJ:tl.constexpr,
                   BK:tl.constexpr,BJ:tl.constexpr):
    row=tl.program_id(0);b=row//I;i=row%I
    j=tl.program_id(1)*BJ+tl.arange(0,BJ);k=tl.arange(0,BK)
    q0=tl.load(Q+b*QB+i*QI+k*QK,k<K,0)
    q1=tl.load(Q+b*QB+QH+i*QI+k*QK,k<K,0)
    g0=tl.load(G+b*GB+i*GI+j*GJ,j<J,0)
    g1=tl.load(G+b*GB+GH+i*GI+j*GJ,j<J,0)
    value=g0[:,None]*q0[None,:]+g1[:,None]*q1[None,:]
    tl.store(D+row*J*K+j[:,None]*K+k[None,:],value,(j[:,None]<J)&(k[None,:]<K))


class Contract(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,z):
        assert q.ndim==z.ndim==4 and q.shape[1]==2 and q.shape[0]==z.shape[0] and q.shape[2]==z.shape[1] and q.shape[3]==z.shape[3]
        ctx.save_for_backward(q,z)
        if not q.is_cuda:return torch.einsum('bhik,bijk->bhij',q,z)
        assert q.dtype==z.dtype==torch.float32
        b,i,j,k=z.shape
        physical=torch.empty((b,i,2,j),device=z.device,dtype=z.dtype)
        forward_kernel[(b*i,triton.cdiv(j,16))](q,z,physical,i,j,k,*q.stride(),*z.stride(),triton.next_power_of_2(k),16,enable_fp_fusion=False)
        return physical.permute(0,2,1,3)

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        q,z=ctx.saved_tensors
        dq=torch.einsum('bhij,bijk->bhik',g,z) if ctx.needs_input_grad[0] else None
        dz=None
        if ctx.needs_input_grad[1]:
            if not q.is_cuda:dz=torch.einsum('bhij,bhik->bijk',g,q)
            else:
                b,i,j,k=z.shape;dz=torch.empty(z.shape,device=z.device,dtype=z.dtype)
                hidden_backward_kernel[(b*i,triton.cdiv(j,16))](q,g,dz,i,j,k,*q.stride(),*g.stride(),triton.next_power_of_2(k),16,enable_fp_fusion=False)
        return dq,dz


def block(self,x,relative=None,centers=None,frequency=None):
    if isinstance(relative,LayerRelations):
        relative=ProjectedRelation(relative.hidden[self._merge_slot],relative.weight)
    assert frequency is None and isinstance(relative,ProjectedRelation)
    batch,tokens,_=x.shape
    q,k,v=self.qkv(self.norm(x)).reshape(batch,tokens,3,HEADS,HD).permute(2,0,3,1,4).unbind(0)
    w=relative.weight.unflatten(0,(HEADS,HD))
    pq=torch.einsum('bhid,hdk->bhik',q,w)
    score=q@k.transpose(-1,-2)+Contract.apply(pq,relative.hidden)
    probability=(score/math.sqrt(HD)).softmax(-1)
    aggregate=(probability@v).transpose(1,2).reshape(batch,tokens,DIM)
    x=x+self.output(aggregate)
    return x+self.ffn(self.ffn_norm(x))


def adapt_contraction(model):
    assert getattr(model,'_efficiency_sidecar',False) and not getattr(model,'_relation_contraction',False)
    assert model.role in ('relative5','distance_mlp') and HEADS==2
    for layer in model.layers:layer.forward=MethodType(block,layer)
    model._relation_contraction=True
    return model
