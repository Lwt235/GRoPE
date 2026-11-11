"""Standalone actual-image -> VGGT -> memory adapter -> existing depth head."""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_view_data import load_backbone
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_view_token import ViewTokenAdapter, inject_last_layer


@torch.inference_mode()
def run(input_path, memory_path, checkpoint_path, output_path):
    # NPZ has only an observed image and known calibration, never teacher data.
    with np.load(input_path) as a:
        assert set(a.files) == {'image', 'K', 'c2w'}
        image = torch.from_numpy(a['image']).float().cuda()[None, None]
        K = torch.from_numpy(a['K']).float().cuda()[None]
        c2w = torch.from_numpy(a['c2w']).float().cuda()[None]
    memory = torch.load(memory_path, map_location='cuda', weights_only=False)['memory']
    ckpt = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    net = ViewTokenAdapter(ckpt['config']['variant'], ckpt['config']['center_variance']).cuda().eval()
    net.load_state_dict(ckpt['model'])
    model = load_backbone()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        tokens, start = model.aggregator(image)
    geometry = projected_inputs(memory, c2w, K)
    modified = inject_last_layer(tokens, start, net, geometry)
    # DPT takes the actual four cached VGGT layer outputs.
    original_float = [None if x is None else x.float() for x in tokens]
    modified_float = [None if x is None else x.float() for x in modified]
    depth_base, _ = model.depth_head(original_float, images=image, patch_start_idx=start)
    depth_modified, _ = model.depth_head(modified_float, images=image, patch_start_idx=start)
    output_path = Path(output_path)
    assert not output_path.exists()
    np.savez_compressed(output_path, tokens=modified[-1][0, 0, start:].cpu().numpy(),
                        original_tokens=tokens[-1][0, 0, start:].float().cpu().numpy(),
                        depth_base=depth_base.float().cpu().numpy(),
                        depth_modified=depth_modified.float().cpu().numpy())
    print({'standalone_output': str(output_path), 'finite': bool(torch.isfinite(depth_modified).all()),
           'depth_absolute_change_mean': float((depth_modified - depth_base).abs().mean())}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('input', 'memory', 'checkpoint', 'output'): p.add_argument('--' + name, required=True)
    a = p.parse_args(); run(a.input, a.memory, a.checkpoint, a.output)
