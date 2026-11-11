"""A fixed-weight check of the adapter's slot/content binding boundary."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import OUT, CACHE, write_json
from models.gaussian_memory_vggt_adapter import InternalMemoryAdapter


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    checkpoint = OUT / 'combined_20261112/step_2000.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    adapter = InternalMemoryAdapter().eval().requires_grad_(False)
    adapter.load_state_dict(state['model'])
    rid = 'transfer_00'
    initial = torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False)
    sample = torch.load(CACHE / rid / 'q11_c0.pt', map_location='cpu', weights_only=False)
    # Invariance holds for any query tokens; use actual cached tokens as a numerical witness.
    tokens = sample['prefix'].float()[None]
    features = initial['memory']['features'].float()
    torch.manual_seed(20261110); permutation = torch.randperm(256)
    original = adapter.content_residual(tokens, features)
    permuted = adapter.content_residual(tokens, features[permutation])
    zero = adapter.content_residual(tokens, torch.zeros_like(features))
    value = {'complete': True, 'scope': 'content branch algebra and numerical witness, not downstream utility',
        'algebra': 'softmax(Q(PK)^T)PV = softmax(QK^T P^T)PV = softmax(QK^T)V',
        'geometry_held_fixed': True, 'learned_weights': True,
        'content_residual_mean_abs': float((original - tokens).abs().mean()),
        'content_permutation_max_abs_difference': float((original - permuted).abs().max()),
        'zero_content_mean_abs_difference': float((original - zero).abs().mean()),
        'conclusion': 'Historical content is used as an unordered set. This adapter does not explicitly bind each content slot to its Gaussian position.',
        'no_model_or_training_change': True, 'checkpoint_sha256': sha256(checkpoint),
        'cache_identity_sha256': sha256(CACHE / rid / 'identity.json'), 'source_sha256': sha256(__file__)}
    write_json(OUT / 'content_binding_check.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
