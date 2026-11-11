"""Frozen-weight compatibility check against the writer's gsplat projection."""
import argparse
import inspect
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import OUT, SEEDS, load_inputs, write_json, assert_identity
from experiments.gaussian_memory_attention_evaluate import aggregate
from experiments.gaussian_memory_setup import sha256
from models.gaussian_memory_attention import GaussianAttentionReader, geometry_inputs

DEST = OUT / 'projection_diagnostic1'


def compatible_inputs(memory, c2w, K):
    from gsplat.cuda._torch_impl import _persp_proj
    data = geometry_inputs(memory, c2w, K)
    R = c2w[:, :3, :3].transpose(-1, -2)
    center = torch.einsum('bij,bnj->bni', R, memory['means'][None] - c2w[:, None, :3, 3])
    cov = R[:, None] @ memory['covars'][None] @ R[:, None].transpose(-1, -2)
    projected, covariance = _persp_proj(center, cov, K, 20, 15)
    covariance = covariance + .1 * torch.eye(2, device=K.device)
    y, x = torch.meshgrid(torch.arange(15, device=K.device) + .5, torch.arange(20, device=K.device) + .5, indexing='ij')
    pixels = torch.stack([x, y], -1).reshape(-1, 2)
    delta = pixels[None, :, None] - projected[:, None]
    quadratic = torch.einsum('bpni,bnij,bpnj->bpn', delta, torch.linalg.inv(covariance), delta).clamp_min(0)
    # Invalid centers are masked identically in all variants. Their projection is irrelevant.
    data['mahalanobis'] = torch.where(data['valid'][:, None], quadratic, data['mahalanobis'])
    return data


def plan():
    from gsplat.cuda._torch_impl import _persp_proj
    DEST.mkdir(exist_ok=True)
    assert not (DEST / 'plan.json').exists()
    source = inspect.getsourcefile(_persp_proj)
    write_json(DEST / 'plan.json', {
        'stage': 'post-result compatibility diagnostic; weights and original results frozen',
        'observed_difference': 'Existing writer renderer uses gsplat perspective covariance with off-axis Jacobian clipping; original attention used the unbounded analytic Jacobian',
        'hypothesis': 'Very large footprints from off-axis primitives explain part of the scene-dependent attention failure',
        'competitor': 'Learned features or non-projection aspects of reading are responsible',
        'intervention': 'Use installed gsplat 1.5.3 _persp_proj for only the explicit Gaussian quadratic term; original full key geometry, opacity, Q/K/V, learned lambda, background slot, data and checkpoint stay fixed',
        'fixed_comparison': 'All three final Gaussian checkpoints; rerun Center unchanged as negative implementation control',
        'decision': 'If Gaussian improves, report projection compatibility contribution; retain original negative result. A frozen intervention is not a trained, independently confirmed method.',
        'gsplat_source': source, 'gsplat_source_sha256': sha256(source),
        'source_sha256': sha256(__file__), 'original_scores_sha256': sha256(OUT / 'evaluation/scores.json')})


def evaluate():
    torch.set_num_threads(4)
    assert_identity()
    plan = json.loads((DEST / 'plan.json').read_text())
    assert sha256(__file__) == plan['source_sha256']
    assert sha256(plan['gsplat_source']) == plan['gsplat_source_sha256']
    assert not (DEST / 'scores.json').exists()
    models = {}
    for seed in SEEDS:
        ckpt = torch.load(OUT / f'gaussian_{seed}/step_2000.pt', weights_only=False, map_location='cuda')
        model = GaussianAttentionReader('gaussian', ckpt['config']['center_variance']).cuda().eval()
        model.load_state_dict(ckpt['model']); models[seed] = model
    ckpt = torch.load(OUT / f'center_{SEEDS[0]}/step_2000.pt', weights_only=False, map_location='cuda')
    control = GaussianAttentionReader('center', ckpt['config']['center_variance']).cuda().eval()
    control.load_state_dict(ckpt['model'])
    scores, figures, control_error = [], {}, 0.
    with torch.no_grad():
        for item in load_inputs('cuda'):
            row, raw, memory = item['row'], item['raw'], item['memory']
            qids = sorted(row['training_query'] + row['heldout_query'])
            c2w, K = raw['c2w'][qids].cuda(), raw['K'][qids].cuda()
            target = raw['teacher'][qids].cuda().flatten(1, 2)
            data = compatible_inputs(memory, c2w, K)
            original = geometry_inputs(memory, c2w, K)
            control_error = max(control_error, (control(data) - control(original)).abs().max().item())
            for seed, model in models.items():
                prediction, attention = model(data, return_attention=True)
                mse = (prediction - target).square().mean((1, 2))
                cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean(1)
                ratio = prediction.norm(dim=-1).mean(1) / target.norm(dim=-1).mean(1).clamp_min(1e-8)
                background = attention[..., -1].mean((1, 2))
                if seed == SEEDS[0]:
                    figures[row['id']] = prediction.cpu().reshape(-1, 15, 20, 32)
                for i, q in enumerate(qids):
                    role = ('train_query' if q in row['training_query'] else
                            'heldout_view' if row['split'] == 'fit' else 'heldout_scene')
                    scores.append({'episode': row['id'], 'scene': row['scene'], 'query_ordinal': q,
                        'frame': row['frames'][q]['index'], 'role': role, 'variant': 'gaussian', 'seed': seed,
                        'condition': 'gsplat_projection', 'mse': mse[i].item(), 'cosine': cosine[i].item(),
                        'norm_ratio': ratio[i].item(), 'background_mass': background[i].item()})
    assert control_error == 0
    fields = ['role', 'variant', 'condition', 'seed']
    result = {'plan_sha256': sha256(DEST / 'plan.json'), 'scores': scores, 'summary': aggregate(scores, fields),
              'per_scene': aggregate(scores, ['scene', *fields]), 'center_control_max_change': control_error}
    write_json(DEST / 'scores.json', result)
    torch.save(figures, DEST / 'visual_data.pt')
    assert_identity()
    print(json.dumps({'diagnostic_complete': True, 'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['plan', 'evaluate'])
    a = p.parse_args()
    plan() if a.command == 'plan' else evaluate()
