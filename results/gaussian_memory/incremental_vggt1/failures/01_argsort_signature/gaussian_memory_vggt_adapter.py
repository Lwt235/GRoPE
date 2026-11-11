"""A single internal VGGT memory insertion, with frozen native downstream layers."""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HEIGHT, WIDTH, PATCH, START, INSERT = 210, 280, 14, 5, 11


def position_grid(device, frames=4):
    y, x = torch.meshgrid(torch.arange(15, device=device), torch.arange(20, device=device), indexing='ij')
    grid = torch.stack((y, x), -1).reshape(1, 300, 2) + 1
    return torch.cat((torch.zeros(1, START, 2, device=device, dtype=grid.dtype), grid), 1).expand(frames, -1, -1)


def local_position(device, frames=4):
    y, x = torch.meshgrid((torch.arange(15, device=device) + .5) / 15 * 2 - 1,
                         (torch.arange(20, device=device) + .5) / 20 * 2 - 1, indexing='ij')
    uv = torch.stack((x, y), -1).flatten(0, 1)
    p = torch.cat([uv] + [f(uv * (math.pi * k)) for k in (1, 2) for f in (torch.sin, torch.cos)], -1)
    return torch.cat((torch.zeros(START, 10, device=device), p), 0)[None].expand(frames, -1, -1)


def project_memory(memory, c2w, K, center_variance=None):
    """Float32, front-to-back alpha compositing on the patch grid.

    Memory and cameras already share the *fixed* initial normalized unit.
    Shape covariance describes extent; it is not an uncertainty estimate.
    """
    from gsplat.cuda._torch_impl import _persp_proj
    mean, cov = memory['means'].float(), memory['covars'].float()
    if center_variance is not None:
        cov = torch.eye(3, device=mean.device)[None].expand(len(mean), -1, -1) * center_variance
    R = c2w[:, :3, :3].transpose(-1, -2).float()
    center = torch.einsum('vij,vnj->vni', R, mean[None] - c2w[:, None, :3, 3].float())
    camera_cov = R[:, None] @ cov[None] @ R[:, None].transpose(-1, -2)
    projected, footprint = _persp_proj(center, camera_cov, K.float(), 20, 15)
    footprint = footprint + .1 * torch.eye(2, device=mean.device)
    y, x = torch.meshgrid(torch.arange(15, device=mean.device) + .5,
                         torch.arange(20, device=mean.device) + .5, indexing='ij')
    pixel = torch.stack((x, y), -1).reshape(-1, 2)
    delta = pixel[None, :, None] - projected[:, None]
    d2 = torch.einsum('vpni,vnij,vpnj->vpn', delta, torch.linalg.inv(footprint), delta).clamp_min(0)
    valid = (center[..., 2] > 1e-4) & torch.isfinite(center).all(-1)
    alpha = memory['opacity'].float().reshape(1, 1, -1) * torch.exp(-.5 * d2)
    alpha = torch.where(valid[:, None] & (d2 <= 9), alpha, 0.).clamp(0, .99)
    order = center[..., 2].argsort(-1, stable=True)
    alpha = alpha.gather(2, order[:, None].expand(-1, 300, -1))
    trans = torch.cat((torch.ones_like(alpha[..., :1]), (1 - alpha).cumprod(-1)[..., :-1]), -1)
    weights = alpha * trans
    coverage = weights.sum(-1)
    tri = torch.tril_indices(3, 3, device=mean.device)
    geom = torch.cat((center, camera_cov[..., tri[0], tri[1]]), -1)
    geom = geom.gather(1, order[..., None].expand(-1, -1, 9))
    summed = weights @ geom
    pe = torch.cat((summed, coverage[..., None]), -1)
    z = center[..., 2].gather(1, order)
    depth = (weights * z[:, None]).sum(-1) / coverage.clamp_min(1e-8)
    return {'pe': pe, 'coverage': coverage, 'depth': depth, 'center': center,
            'footprint': footprint, 'projected': projected, 'valid': valid}


def padded_geometry(memory, cameras, intrinsic, rms=None, center_variance=None):
    p = project_memory(memory, cameras, intrinsic, center_variance)['pe']
    if rms is not None:
        p = p / rms.to(p).clamp_min(1e-4)
    # No centering/bias: empty support must remain exactly zero.
    p = p.clamp(-20, 20)
    return F.pad(p, (0, 0, START, 0))


class InternalMemoryAdapter(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.pe_q = nn.Linear(10, 1024, bias=False)
        self.pe_k = nn.Linear(10, 1024, bias=False)
        self.norm = nn.LayerNorm(1024)
        self.content_norm = nn.LayerNorm(32)
        self.q = nn.Linear(1024, width)
        self.k = nn.Linear(32, width)
        self.v = nn.Linear(32, width)
        self.out = nn.Linear(width, 1024)
        for layer in (self.pe_q, self.pe_k, self.out):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def attention(self, attention, x, pos, pe):
        B, N, C = x.shape
        qkv = attention.qkv(x).reshape(B, N, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attention.q_norm(q), attention.k_norm(k)
        if attention.rope is not None:
            q, k = attention.rope(q, pos), attention.rope(k, pos)
        dq = self.pe_q(pe).reshape(B, N, attention.num_heads, attention.head_dim).transpose(1, 2)
        dk = self.pe_k(pe).reshape(B, N, attention.num_heads, attention.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q + dq, k + dk, v, dropout_p=0.)
        return attention.proj_drop(attention.proj(out.transpose(1, 2).reshape(B, N, C)))

    def content_residual(self, tokens, content):
        # tokens [1,4,305,1024], shared content [256,32].
        patches = tokens[:, :, START:]
        q = self.q(self.norm(patches)).reshape(1, -1, 4, 16).transpose(1, 2)
        c = self.content_norm(content)
        k = self.k(c).reshape(1, -1, 4, 16).transpose(1, 2)
        v = self.v(c).reshape(1, -1, 4, 16).transpose(1, 2)
        r = F.scaled_dot_product_attention(q, k, v, dropout_p=0.).transpose(1, 2)
        r = self.out(r.reshape(1, 4, 300, 64))
        return tokens + F.pad(r, (0, 0, START, 0))


def native_tail(backbone, cached, adapter=None, pe=None, content=None, checkpoint_blocks=False):
    """Resume immediately before global[11]; no hooks or mutation of VGGT source."""
    a = backbone.aggregator
    frame11 = cached['prefix']  # [4,305,1024], frozen frame[11] output
    pos = position_grid(frame11.device)
    global_pos = pos.reshape(1, -1, 2)
    tokens = frame11.reshape(1, -1, 1024)
    block = a.global_blocks[INSERT]
    if adapter is None:
        tokens = block(tokens, pos=global_pos)
    else:
        pe = pe.reshape(1, -1, 10)
        tokens = tokens + block.ls1(adapter.attention(block.attn, block.norm1(tokens), global_pos, pe))
        tokens = tokens + block.ls2(block.mlp(block.norm2(tokens)))
        tokens = adapter.content_residual(tokens.reshape(1, 4, 305, 1024), content).reshape(1, -1, 1024)
    outputs = [None] * 24
    outputs[4] = cached['early4']
    outputs[11] = torch.cat((frame11.reshape(1, 4, 305, 1024), tokens.reshape(1, 4, 305, 1024)), -1)
    for index in range(12, 24):
        f = a.frame_blocks[index]
        g = a.global_blocks[index]
        x = tokens.reshape(4, 305, 1024)
        if checkpoint_blocks and torch.is_grad_enabled():
            frame = checkpoint(f, x, pos, use_reentrant=False)
            tokens = checkpoint(g, frame.reshape(1, -1, 1024), global_pos, use_reentrant=False)
        else:
            frame = f(x, pos=pos)
            tokens = g(frame.reshape(1, -1, 1024), pos=global_pos)
        if index in (17, 23):
            outputs[index] = torch.cat((frame.reshape(1, 4, 305, 1024), tokens.reshape(1, 4, 305, 1024)), -1)
    # DPT is per view. Decode only the current (last) frame, preserving its native head.
    outputs = [x[:, -1:].float() if x is not None else None for x in outputs]
    with torch.autocast('cuda', enabled=False):
        dummy = torch.empty(1, 1, 3, HEIGHT, WIDTH, device=tokens.device)
        depth, confidence = backbone.depth_head(outputs, dummy, START)
    return depth[0, 0, ..., 0], confidence[0, 0]
