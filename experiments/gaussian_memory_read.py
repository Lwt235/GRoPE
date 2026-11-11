"""Read a serialized memory with cameras only, without images or the writer."""
import argparse

import numpy as np
import torch

from models.gaussian_memory import read_memory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--memory', required=True)
    p.add_argument('--cameras', required=True, help='npz with c2w and K already expressed in memory coordinates')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(a.memory, weights_only=False, map_location='cuda')
    with np.load(a.cameras) as z:
        camera = torch.from_numpy(z['c2w']).cuda().float()
        K = torch.from_numpy(z['K']).cuda().float()
    with torch.no_grad():
        features, alpha = read_memory(saved['memory'], camera, K)
    np.savez(a.output, features=features.cpu().numpy(), alpha=alpha.cpu().numpy())
    print(f'Read {len(camera)} views from one saved memory: {a.output}', flush=True)


if __name__ == '__main__': main()
