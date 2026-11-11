"""Matched content and spatial reads, with common camera-valid reference slots."""
import hashlib

import torch

from models.gaussian_memory_direct_reference import DirectReferenceAdapter


class ContextAdapter(DirectReferenceAdapter):
    def __init__(self, method, center_variance=.7146816253662109):
        assert method in ('wide', 'content', 'spatial')
        super().__init__('wide' if method == 'wide' else 'direct', center_variance)
        self.method = method

    def forward(self, tokens, geometry, condition='normal'):
        if self.method == 'content':
            geometry = dict(geometry,
                            key=torch.cat([geometry['key'][..., :2048],
                                           torch.zeros_like(geometry['key'][..., 2048:])], -1),
                            distance2=torch.zeros_like(geometry['distance2']))
        return super().forward(tokens, geometry, condition)


def permute_visible_content(geometry, episode, queries):
    """Permute features within each image's query-valid slots, keeping geometry.

    Binary opacity means all retained log weights are equal. Content-only reads
    are therefore permutation-invariant; the spatial read loses its bindings.
    """
    valid = torch.isfinite(geometry['log_weight']).detach().cpu()
    assert torch.all(geometry['log_weight'][valid.to(geometry['log_weight'].device)] == 0)
    permutations = []
    for b, query in enumerate(queries):
        seed = int.from_bytes(hashlib.sha256(f'context-binding-v1:{episode}:{int(query)}'.encode()).digest()[:8], 'little')
        generator = torch.Generator().manual_seed(seed)
        p = torch.arange(1200)
        for view in range(4):
            ids = torch.arange(view * 300, (view + 1) * 300)[valid[b, view * 300:(view + 1) * 300]]
            p[ids] = ids[torch.randperm(len(ids), generator=generator)]
        permutations.append(p)
    permutations = torch.stack(permutations).to(geometry['key'].device)
    batch = torch.arange(len(permutations), device=permutations.device)[:, None]
    key = geometry['key'].clone()
    key[..., :2048] = geometry['key'][batch, permutations][..., :2048]
    return dict(geometry, key=key, value=geometry['value'][batch, permutations]), permutations
