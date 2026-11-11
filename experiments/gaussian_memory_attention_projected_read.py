"""Standalone reading using the same projection as the frozen writer renderer."""
import argparse

import numpy as np
import torch

from models.gaussian_memory_attention import GaussianAttentionReader
from models.gaussian_memory_projection import projected_inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--memory', required=True)
    p.add_argument('--cameras', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    ckpt = torch.load(a.checkpoint, weights_only=False, map_location=a.device)
    config = ckpt['config']
    assert config.get('projection') == 'gsplat'
    model = GaussianAttentionReader(config['variant'], config['center_variance']).to(a.device).eval()
    model.load_state_dict(ckpt['model'])
    memory = torch.load(a.memory, weights_only=False, map_location=a.device)['memory']
    with np.load(a.cameras) as cameras:
        c2w = torch.from_numpy(cameras['c2w']).to(a.device)
        K = torch.from_numpy(cameras['K']).to(a.device)
    with torch.no_grad():
        prediction = model(projected_inputs(memory, c2w, K)).reshape(len(K), 15, 20, 32).cpu().numpy()
    np.savez(a.output, prediction=prediction)
    print({'views': len(K), 'output': a.output}, flush=True)


if __name__ == '__main__':
    main()
