"""All-layer read-only coarse native memory. No historically fused query prefix."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.gaussian_memory_native_pe import LowRank, SplitKLinear, rope3d, grid, START, DIM, HEADS, HD

DEPTH_LAYERS = (4, 11, 17, 23)


class CoarseReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.ModuleList([LowRank(DIM, DIM) for _ in range(24)])
        self.proj = nn.ModuleList([LowRank(DIM, DIM) for _ in range(24)])
        self.relation = nn.Sequential(SplitKLinear(5, 16), nn.GELU(), SplitKLinear(16, HEADS, bias=False))
        nn.init.zeros_(self.relation[-1].weight)


@torch.no_grad()
def group_geometry(means, cov, count):
    """FPS seed index 0, smallest-index argmax/argmin ties, uniform patch mass."""
    distance = torch.full((len(means),), float('inf'), device=means.device)
    chosen = []; index = 0
    for _ in range(count):
        chosen.append(index)
        torch.minimum(distance, (means - means[index]).square().sum(-1), out=distance)
        # A selected center must not be selected again, including duplicate points.
        distance[torch.tensor(chosen, device=means.device)] = -1
        index = int(distance.argmax())
    centers = means[chosen]
    membership = torch.cdist(means.double(), centers.double()).argmin(-1)
    # Identical centers can leave empty Voronoi cells; deterministically own seeds.
    membership[torch.tensor(chosen, device=means.device)] = torch.arange(count, device=means.device)
    mass = torch.bincount(membership, minlength=count)
    def average(value):
        result = torch.zeros((count, *value.shape[1:]), device=value.device, dtype=torch.float64)
        result.index_add_(0, membership, value.double())
        return result / mass.reshape(count, *([1] * (value.ndim - 1)))
    mu = average(means); delta = means.double() - mu[membership]
    sigma = average(cov.double() + delta[..., :, None] * delta[..., None, :])
    return membership, mass, mu.float(), sigma.float(), chosen


@torch.no_grad()
def query_geometry(depth, pose, hw, unit, origin):
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    ext, intr = pose_encoding_to_extri_intri(pose[None, None], hw)
    ext, intr, d = ext[0, 0].double(), intr[0, 0].double(), depth.double()
    h, w = hw
    y, x = torch.meshgrid(torch.arange(h, device=d.device), torch.arange(w, device=d.device), indexing='ij')
    camera = torch.stack(((x - intr[0, 2]) / intr[0, 0] * d, (y - intr[1, 2]) / intr[1, 1] * d, d), -1)
    world = ((camera - ext[:3, 3]) @ ext[:3, :3] - origin.to(d.device).double()) / unit
    patches = world.reshape(h // 14, 14, w // 14, 14, 3).permute(0, 2, 1, 3, 4).reshape(-1, 196, 3)
    mu = patches.mean(1); delta = patches - mu[:, None]
    cov = delta.transpose(-1, -2) @ delta / 196 + 1e-6 * torch.eye(3, device=d.device)
    return F.pad(mu.float(), (0, 0, START, 0)), torch.cat((torch.eye(3, device=d.device).expand(START, 3, 3), cov.float()))


@torch.no_grad()
def descriptors(sample, kind, device='cuda'):
    mu = torch.cat((sample['source_means'], sample['query_means'])).to(device).float()
    cov = torch.cat((sample['source_covariances'], sample['query_covariances'])).to(device).float()
    n = len(sample['query_means']); delta = mu[-n:, None] - mu[None]
    d2 = delta.square().sum(-1)
    if kind == 'center':
        z = torch.stack([torch.log1p(d2 / s**2) for s in (.02, .05, .1, .2, .5)], -1)
    else:
        if kind == 'sphere': cov = torch.diag_embed(cov.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None].expand(-1, 3))
        inv = torch.linalg.inv(cov.double()).float(); pq = inv[-n:]; cq = cov[-n:]
        mq = torch.einsum('qki,qij,qkj->qk', delta, pq, delta)
        mk = torch.einsum('qki,kij,qkj->qk', delta, inv, delta)
        tq = pq.flatten(1) @ cov.transpose(-1, -2).flatten(1).T
        tk = cq.transpose(-1, -2).flatten(1) @ inv.flatten(1).T
        z = torch.stack((d2.clamp_min(0).sqrt() / .2, mq.clamp_min(0).log1p(), mk.clamp_min(0).log1p(), tq.clamp_min(0).log1p(), tk.clamp_min(0).log1p()), -1)
    z[:START] = 0
    special = torch.cat((sample['source_special'].to(device), torch.arange(n, device=device) < START))
    z[:, special] = 0
    assert torch.isfinite(z).all()
    return z.to(torch.bfloat16)


def read_attention(attn, x, sample, adapter, index, pos, kind, z):
    b, nq, _ = x.shape
    q, k, v = attn.qkv(x).reshape(b, nq, 3, HEADS, HD).permute(2, 0, 3, 1, 4).unbind(0)
    if adapter is not None: q = q + adapter.q[index](x).reshape(b, nq, HEADS, HD).transpose(1, 2)
    q = attn.rope(attn.q_norm(q), pos); k = attn.rope(attn.k_norm(k), pos)
    k = torch.cat((sample['keys'][index], k), -2); v = torch.cat((sample['values'][index], v), -2)
    bias = F.pad(sample['log_mass'], (0, nq))[None, None, None]
    if kind != 'frozen':
        mu = torch.cat((sample['source_means'], sample['query_means']))[None]
        q = rope3d(q, mu[:, -nq:]); k = rope3d(k, mu)
        active_k = ~torch.cat((sample['source_special'], torch.arange(nq, device=x.device) < START))
        active_q = torch.arange(nq, device=x.device) >= START
        # Count correction stays active for all queries. Only learned geometry is masked.
        relation = adapter.relation(z).permute(2, 0, 1)[None]
        relation = relation.masked_fill(~(active_q[:, None] & active_k[None])[None, None], 0.)
        bias = F.pad(sample['log_mass'], (0, nq))[None, None, None] + relation
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=0.)
    result = result.transpose(1, 2).reshape(b, nq, DIM)
    output = attn.proj(result)
    if adapter is not None: output = output + adapter.proj[index](result)
    return attn.proj_drop(output)


def forward_query(model, sample, adapter=None, kind='frozen', z=None, checkpoint_blocks=True, return_tokens=False):
    agg = model.aggregator; hw = sample['native_hw']
    # Only current-image backbone output is cached. No historical attention has run.
    x = torch.cat((agg.camera_token[:, 1], agg.register_token[:, 1], sample['patch_tokens']), 1)
    pos = grid(*hw, x.device); outputs = [None] * 24
    for index in range(24):
        def pair(inp, i=index):
            frame = agg.frame_blocks[i](inp, pos=pos)
            block = agg.global_blocks[i]
            out = read_attention(block.attn, block.norm1(frame), sample, adapter, i, pos, kind, z)
            result = frame + block.ls1(out)
            result = result + block.ls2(block.mlp(block.norm2(result)))
            return (result, torch.cat((frame, result), -1)[:, None].float()) if i in DEPTH_LAYERS else result
        result = checkpoint(pair, x, use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else pair(x)
        if index in DEPTH_LAYERS: x, outputs[index] = result
        else: x = result
    with torch.autocast('cuda', enabled=False):
        dummy = torch.empty(1, 1, 3, *hw, device=x.device)
        depth, _ = model.depth_head(outputs, dummy, START)
    return (depth[0, 0, ..., 0], outputs) if return_tokens else depth[0, 0, ..., 0]


@torch.no_grad()
def patch_tokens(model, image):
    agg = model.aggregator
    image = (image[None, None] - agg._resnet_mean) / agg._resnet_std
    result = agg.patch_embed(image.flatten(0, 1))
    return result['x_norm_patchtokens'] if isinstance(result, dict) else result
