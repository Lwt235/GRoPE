"""Optional evaluation adapter for the existing bias-free relation projection.

The registered model state and mathematical function are unchanged. This file
does not alter any frozen model or automatically enable the adapter in training.
"""
import math
from typing import NamedTuple
from types import MethodType

import torch
from models.round58_support_pe import DIM, HEADS, HD, AttentionBlock
from models.round65_actual_gs import Block


class ProjectedRelation(NamedTuple):
    hidden: torch.Tensor
    weight: torch.Tensor


def carry_projection(self, hidden):
    assert self.bias is None
    return ProjectedRelation(hidden, self.weight)


def factored_bias(query, relation):
    """q_h dot W_h z_ij = (W_h.T q_h) dot z_ij, with optional view axis."""
    weight = relation.weight.unflatten(0, (HEADS, HD))
    projected_query = torch.einsum('...hid,hdk->...hik', query, weight)
    return torch.einsum('...hik,...ijk->...hij', projected_query, relation.hidden)


def native_block(self, x, relative=None, centers=None, frequency=None):
    if relative is None:
        return Block.forward(self, x, relative, centers, frequency)
    assert frequency is None and isinstance(relative, ProjectedRelation)
    batch, tokens, _ = x.shape
    q, k, v = self.qkv(self.norm(x)).reshape(batch, tokens, 3, HEADS, HD).permute(2, 0, 3, 1, 4).unbind(0)
    score = q @ k.transpose(-1, -2) + factored_bias(q, relative)
    probability = (score / math.sqrt(HD)).softmax(-1)
    aggregate = (probability @ v).transpose(1, 2).reshape(batch, tokens, DIM)
    x = x + self.output(aggregate)
    return x + self.ffn(self.ffn_norm(x))


def region_block(self, x, relative=None, cross=False):
    if relative is None:
        return AttentionBlock.forward(self, x, relative, cross)
    assert not cross and isinstance(relative, ProjectedRelation)
    batch, views, tokens, _ = x.shape
    q, k, v = self.qkv(self.norm(x)).reshape(batch, views, tokens, 3, HEADS, HD).permute(3, 0, 1, 4, 2, 5).unbind(0)
    score = q @ k.transpose(-1, -2) + factored_bias(q, relative)
    probability = (score / math.sqrt(HD)).softmax(-1)
    aggregate = (probability @ v).transpose(-3, -2).reshape(batch, views, tokens, DIM)
    x = x + self.output(aggregate)
    return x + self.ffn(self.ffn_norm(x))


def factorize(model, task):
    """Apply only to a private model instance, preserving all parameter names."""
    assert task in ('native', 'region') and not getattr(model, '_factored_relation_runtime', False)
    assert model.role in ('distance_mlp', 'direct_mlp', 'gaussian_relation', 'relative5')
    last = model.position.mlp[-1]
    assert isinstance(last, torch.nn.Linear) and last.bias is None and last.out_features == DIM
    expected = Block if task == 'native' else AttentionBlock
    assert all(isinstance(layer, expected) for layer in model.layers)
    last.forward = MethodType(carry_projection, last)
    for layer in model.layers:
        layer.forward = MethodType(native_block if task == 'native' else region_block, layer)
    model._factored_relation_runtime = True
    return model
