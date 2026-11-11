"""Independent inference with only reader weights, serialized memory and cameras."""
import argparse

import numpy as np
import torch

from models.gaussian_memory_attention import GaussianAttentionReader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--memory', required=True)
    p.add_argument('--cameras', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=args.device)
    config = ckpt['config']
    model = GaussianAttentionReader(config['variant'], config['center_variance']).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    memory = torch.load(args.memory, weights_only=False, map_location=args.device)['memory']
    with np.load(args.cameras) as a:
        c2w = torch.from_numpy(a['c2w']).to(args.device)
        K = torch.from_numpy(a['K']).to(args.device)
    with torch.no_grad():
        prediction = model.read_memory(memory, c2w, K).cpu().numpy()
    np.savez(args.output, prediction=prediction)
    print({'views': len(prediction), 'output': args.output}, flush=True)


if __name__ == '__main__':
    main()
