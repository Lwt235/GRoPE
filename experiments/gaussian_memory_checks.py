"""Small correctness checks tied to the spatial-memory claim, not numeric gates."""
import io
import json

import numpy as np
import torch

from experiments.gaussian_memory_data import REPORT, REF, TRAIN_QUERY, HOLD_QUERY, episodes
from models.gaussian_memory import GaussianWriter, read_memory, detached_memory


def main():
    torch.set_num_threads(4)
    torch.manual_seed(81)
    device = 'cuda'
    n = 64
    seeds = {'seed_mean': torch.randn(n, 3, device=device) * .2 + torch.tensor([0., 0., 2.], device=device),
             'seed_chol': torch.eye(3, device=device)[None].repeat(n, 1, 1) * .09,
             'seed_context': torch.randn(n, 2048, device=device),
             'seed_teacher': torch.randn(n, 32, device=device)}
    model = GaussianWriter().to(device)
    memory = model(seeds)
    cams = torch.eye(4, device=device)[None].repeat(2, 1, 1)
    cams[1, 0, 3] = .2
    K = torch.tensor([[20., 0., 10.], [0., 20., 7.5], [0., 0., 1.]], device=device)[None].repeat(2, 1, 1)
    output, alpha = read_memory(memory, cams, K)
    loss = output.square().mean()
    loss.backward()
    grad = model.net[-1].weight.grad
    grads = {'center': grad[:3].norm().item(), 'shape': grad[3:9].norm().item(),
             'opacity': grad[9].norm().item(), 'feature': grad[10:].norm().item()}
    assert all(np.isfinite(v) and v > 0 for v in grads.values()), grads
    # An independently specified rigid transform must preserve projection.
    angle = .37
    T = torch.tensor([[np.cos(angle), 0, np.sin(angle), .31], [0, 1, 0, -.17],
                      [-np.sin(angle), 0, np.cos(angle), .20], [0, 0, 0, 1]], device=device).float()
    moved = detached_memory(memory, device)
    moved['means'] = moved['means'] @ T[:3, :3].T + T[:3, 3]
    moved['covars'] = T[:3, :3] @ moved['covars'] @ T[:3, :3].T
    with torch.no_grad():
        after, _ = read_memory(moved, T[None] @ cams, K)
        transform_error = (after - output).abs().max().item()
        assert transform_error < 1e-4, transform_error
        buf = io.BytesIO()
        torch.save(detached_memory(memory), buf)
        buf.seek(0)
        restored = torch.load(buf, weights_only=True, map_location=device)
        replay, _ = read_memory(restored, cams, K)
        replay_error = (replay - output).abs().max().item()
        assert replay_error == 0, replay_error
        blank = detached_memory(memory, device)
        blank['opacities'].zero_()
        removed, _ = read_memory(blank, cams, K)
        assert removed.abs().max().item() == 0
    rows, _ = episodes()
    assert not (set(REF) & (set(TRAIN_QUERY) | set(HOLD_QUERY)))
    assert not (set(TRAIN_QUERY) & set(HOLD_QUERY))
    for row in rows:
        assert len({f['rgb_path'] for f in row['frames']}) == 13
    result = {'passed': True, 'gradient_norms': grads, 'joint_transform_max_error': transform_error,
              'save_load_max_error': replay_error, 'removed_memory_output_max': 0.,
              'query_split_disjoint': True, 'episodes': len(rows),
              'claims': 'implementation and gradients only; no learning or utility evidence'}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
