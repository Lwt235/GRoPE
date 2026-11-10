"""Private four-layer hidden-gradient merge. FP32 CUDA, first-order only.

Preserves forward values/state. Trades additional live branch gradients for
one memory pass. Geometry inputs remain fixed as in the existing sidecar.
"""
from types import MethodType
from typing import NamedTuple
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from models.factored_relation_runtime import ProjectedRelation, native_block


@triton.jit
def merge_kernel(G0, G1, G2, G3, OUT, STRIDES: tl.constexpr,
                 B: tl.constexpr, I: tl.constexpr, J: tl.constexpr,
                 K: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < B*I*J*K
    j = index % J
    k = index // J % K
    i = index // (J*K) % I
    b = index // (J*K*I)
    pointers = (G0, G1, G2, G3)
    total = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(COUNT):
        strides = STRIDES[slot]
        value = tl.load(pointers[slot] + b*strides[0] + i*strides[1] + j*strides[2] + k*strides[3], mask, 0)
        total = total + value
    tl.store(OUT + index, total, mask)


class FanoutHidden(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.set_materialize_grads(False)
        return tuple(value.view_as(value) for _ in range(4))

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        grads = tuple(g for g in gradients if g is not None)
        if not grads:
            return None
        if len(grads) == 1:
            return grads[0]
        if not grads[0].is_cuda:
            return sum(grads[1:], grads[0])
        assert all(g.dtype == torch.float32 and g.ndim == 4 and g.shape == grads[0].shape for g in grads)
        b, i, j, k = grads[0].shape
        physical = torch.empty((b, i, k, j), dtype=torch.float32, device=grads[0].device)
        padded = grads + (grads[0],)*(4-len(grads))
        merge_kernel[(triton.cdiv(physical.numel(), 1024),)](
            *padded, physical, tuple(g.stride() for g in padded), b, i, j, k, len(grads), 1024)
        return physical.transpose(-1, -2)


class LayerRelations(NamedTuple):
    hidden: tuple
    weight: torch.Tensor


def projection(self, hidden):
    return LayerRelations(FanoutHidden.apply(hidden), self.weight)


def layer_forward(self, x, relative=None, centers=None, frequency=None):
    if isinstance(relative, LayerRelations):
        relative = ProjectedRelation(relative.hidden[self._merge_slot], relative.weight)
    return native_block(self, x, relative, centers, frequency)


def fuse_hidden_gradient(model):
    assert getattr(model, '_efficiency_sidecar', False)
    assert not getattr(model, '_fused_hidden_gradient', False)
    assert len(model.layers) == 4 and not model.position._recompute
    assert model.position.mlp[-1].forward.__func__.__name__ == 'carry_projection'
    model.position.mlp[-1].forward = MethodType(projection, model.position.mlp[-1])
    for slot, layer in enumerate(model.layers):
        layer._merge_slot = slot
        layer.forward = MethodType(layer_forward, layer)
    model._fused_hidden_gradient = True
    return model
