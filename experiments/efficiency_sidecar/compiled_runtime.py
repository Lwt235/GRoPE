"""Opt-in local Inductor compilation; geometry remains eager FP64."""
from types import MethodType
import torch
import torch.nn.functional as F
from models.factored_relation_runtime import native_block
from experiments.efficiency_sidecar.adapters import quadratic_descriptors

OPTIONS={'triton.cudagraphs':False,'max_autotune':False,'compile_threads':1}


def hidden(value,weight,bias):
    return F.gelu(F.linear(value,weight,bias),approximate='none')


# Lazy compile: creation itself does not compile or initialize CUDA.
compiled_hidden=torch.compile(hidden,fullgraph=True,dynamic=False,options=OPTIONS)
compiled_block=torch.compile(native_block,fullgraph=True,dynamic=False,options=OPTIONS)


def position_forward(self,centers,covariance):
    if centers.requires_grad or covariance.requires_grad:raise ValueError('Fixed geometry only')
    value=(self.descriptors(centers,covariance) if self._efficiency_role=='distance_mlp'
           else quadratic_descriptors(centers,covariance))
    z=compiled_hidden(value,self.mlp[0].weight,self.mlp[0].bias)
    return self.mlp[2](z)


def adapt_compiled(model):
    assert getattr(model,'_efficiency_sidecar',False) and not getattr(model,'_compiled_runtime',False)
    assert model.role in ('relative5','distance_mlp') and not model.position._recompute
    assert model.position.mlp[-1].forward.__func__.__name__=='carry_projection'
    assert model.position.mlp[1].approximate=='none'
    model.position.forward=MethodType(position_forward,model.position)
    for layer in model.layers:layer.forward=MethodType(compiled_block,layer)
    model._compiled_runtime=True
    return model
