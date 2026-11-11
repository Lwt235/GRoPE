"""Matched low-rank native query readers; geometry does not change V content size."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

START, DIM, HEADS, HD = 5, 1024, 16, 64
LAYERS = (20, 21, 22, 23)
KINDS = ('native', 'center', 'sphere', 'gaussian')


class _SplitKLinear(torch.autograd.Function):
    """Same linear forward; parallelize the very long small weight-gradient reduction."""
    @staticmethod
    def forward(ctx, x, weight, bias):
        out = F.linear(x, weight, bias)
        ctx.save_for_backward(x.to(out.dtype), weight.to(out.dtype))
        ctx.input_dtype, ctx.weight_dtype = x.dtype, weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return out

    @staticmethod
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        dx = F.linear(gradient, weight.T).to(ctx.input_dtype) if ctx.needs_input_grad[0] else None
        flat_x = x.reshape(-1, x.shape[-1]); flat_g = gradient.reshape(-1, gradient.shape[-1])
        block = 4096; padding = (-len(flat_x)) % block
        if padding:
            flat_x = F.pad(flat_x, (0, 0, 0, padding)); flat_g = F.pad(flat_g, (0, 0, 0, padding))
        # Independent FP32 partial sums expose enough GPU blocks for 5/16-wide matrices.
        xb = flat_x.float().reshape(-1, block, x.shape[-1])
        gb = flat_g.float().reshape(-1, block, gradient.shape[-1])
        with torch.autocast('cuda', enabled=False):
            dw = torch.bmm(gb.transpose(1, 2), xb).sum(0).to(gradient.dtype).to(ctx.weight_dtype)
            db = gradient.float().reshape(-1, gradient.shape[-1]).sum(0).to(gradient.dtype).to(ctx.bias_dtype) if ctx.bias_dtype is not None else None
        return dx, dw, db


class SplitKLinear(nn.Linear):
    def forward(self, x): return _SplitKLinear.apply(x, self.weight, self.bias)


class LowRank(nn.Module):
    def __init__(self, ni, no, rank=8):
        super().__init__()
        self.a = nn.Linear(ni, rank, bias=False)
        self.b = nn.Linear(rank, no, bias=False)
        nn.init.zeros_(self.b.weight)

    def forward(self, x): return self.b(self.a(x))


class NativePEReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.ModuleList([LowRank(DIM, DIM * 3) for _ in LAYERS])
        self.proj = nn.ModuleList([LowRank(DIM, DIM) for _ in LAYERS])
        self.relation = nn.Sequential(SplitKLinear(5, 16), nn.GELU(), SplitKLinear(16, HEADS, bias=False))
        nn.init.zeros_(self.relation[-1].weight)


def grid(height, width, device):
    y, x = torch.meshgrid(torch.arange(height // 14, device=device), torch.arange(width // 14, device=device), indexing='ij')
    p = torch.stack((y, x), -1).reshape(1, -1, 2) + 1
    return F.pad(p, (0, 0, START, 0))


def rope3d(tokens, centers):
    """Official Point3R multi-base mechanism on 3 even 20-d blocks; 4 untouched."""
    chunks = []
    for axis in range(3):
        x = tokens[..., axis * 20:(axis + 1) * 20]
        rotated = torch.cat((-x[..., 10:], x[..., :10]), -1)
        out = []
        for base in (10., 100., 1000., 10000.):
            inv = base ** (-torch.arange(0, 20, 2, device=tokens.device).float() / 20)
            angle = (centers[..., axis, None].float() * inv).to(tokens.dtype)
            angle = torch.cat((angle, angle), -1).unsqueeze(1)
            out.append(x * angle.cos() + rotated * angle.sin())
        chunks.append(torch.stack(out).mean(0))
    return torch.cat((*chunks, tokens[..., 60:]), -1)


@torch.no_grad()
def relations(sample, kind, device='cuda'):
    """Compute the five fixed descriptors, no learned or label-based normalization."""
    means, cov = sample['means'].to(device).float(), sample['covariances'].to(device).float()
    n = sample['tokens_per_frame']; q = means[-n:]
    special = (torch.arange(len(means), device=device) % n) < START
    if kind == 'native':
        p = grid(*sample['native_hw'], device)[0].float() / max(sample['native_hw']) * 14
        distance2 = torch.cdist(p, p.repeat(len(means) // n, 1)).square()
    else:
        delta = q[:, None] - means[None]
        distance2 = delta.square().sum(-1)
    if kind in ('native', 'center'):
        z = torch.stack([torch.log1p(distance2 / scale**2) for scale in (.02, .05, .1, .2, .5)], -1)
    else:
        if kind == 'sphere':
            cov = torch.diag_embed(cov.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None].expand(-1, 3))
        inv = torch.linalg.inv(cov.double()).float(); pq = inv[-n:]; cq = cov[-n:]
        mq = torch.einsum('qki,qij,qkj->qk', delta, pq, delta)
        mk = torch.einsum('qki,kij,qkj->qk', delta, inv, delta)
        tq = pq.flatten(1) @ cov.transpose(-1, -2).flatten(1).T
        tk = cq.transpose(-1, -2).flatten(1) @ inv.flatten(1).T
        z = torch.stack((distance2.clamp_min(0).sqrt() / .2,
                         mq.clamp_min(0).log1p(), mk.clamp_min(0).log1p(),
                         tq.clamp_min(0).log1p(), tk.clamp_min(0).log1p()), -1)
    z[:START] = 0; z[:, special] = 0
    assert torch.isfinite(z).all()
    return z.to(torch.bfloat16)


def attention(attn, x, saved, adapter, layer, pos, centers, descriptor, kind, chunk=1041):
    b, nq, _ = x.shape
    current = attn.qkv(x) + adapter.qkv[layer](x)
    q, k, v = current.reshape(b, nq, 3, HEADS, HD).permute(2, 0, 3, 1, 4).unbind(0)
    source_delta = adapter.qkv[layer](saved['norm'])
    _, dk, dv = source_delta.reshape(b, -1, 3, HEADS, HD).permute(2, 0, 3, 1, 4).unbind(0)
    k = torch.cat((saved['k'] + dk, k), -2); v = torch.cat((saved['v'] + dv, v), -2)
    q, k = attn.q_norm(q), attn.k_norm(k)
    q, k = attn.rope(q, pos), attn.rope(k, pos.repeat(1, k.shape[-2] // nq, 1))
    if kind != 'native':
        q = rope3d(q, centers[:, -nq:]); k = rope3d(k, centers)
    outputs = []
    for begin in range(0, nq, chunk):
        end = min(begin + chunk, nq)
        bias = adapter.relation(descriptor[begin:end]).permute(2, 0, 1)[None]
        active_q = torch.arange(begin, end, device=x.device) >= START
        active_k = torch.arange(k.shape[-2], device=x.device).remainder(nq) >= START
        bias = bias.masked_fill(~(active_q[:, None] & active_k[None])[None, None], 0.)
        outputs.append(F.scaled_dot_product_attention(q[:, :, begin:end], k, v, attn_mask=bias.to(q.dtype), dropout_p=0.))
    result = torch.cat(outputs, -2).transpose(1, 2).reshape(b, nq, DIM)
    return attn.proj_drop(attn.proj(result) + adapter.proj[layer](result))


def native_tail(model, sample, adapter, kind, descriptor=None, checkpoint_blocks=True):
    h, w = sample['native_hw']; n = sample['tokens_per_frame']; pos = grid(h, w, sample['prefix'].device)
    centers = sample['means'][None].float(); x = sample['prefix']
    outputs = [None] * 24
    for index in (4, 11, 17): outputs[index] = sample['early'][index].float()
    for layer, index in enumerate(LAYERS):
        frame = x if index == 20 else model.aggregator.frame_blocks[index](x, pos=pos)
        block = model.aggregator.global_blocks[index]
        def apply(inp, blk=block, layer_id=layer, memory=sample['memory'][layer]):
            out = attention(blk.attn, blk.norm1(inp), memory, adapter, layer_id, pos, centers, descriptor, kind)
            z = inp + blk.ls1(out)
            return z + blk.ls2(blk.mlp(blk.norm2(z)))
        if checkpoint_blocks and torch.is_grad_enabled():
            x = checkpoint(apply, frame, use_reentrant=False)
        else: x = apply(frame)
        if index == 23: outputs[23] = torch.cat((frame, x), -1)[:, None].float()
    with torch.autocast('cuda', enabled=False):
        dummy = torch.empty(1, 1, 3, h, w, device=x.device)
        depth, _ = model.depth_head(outputs, dummy, START)
    return depth[0, 0, ..., 0]


def loss_for(depth, sample):
    pred = F.interpolate(depth[None, None], (210, 280), mode='bilinear', align_corners=False)[0, 0]
    residual = (pred.clamp_min(1e-6) / sample['target'])[sample['valid']].log()
    return residual.square().mean() + residual.var(unbiased=False)
