"""Post-result diagnostic: visibility vs reverse depth, with every weight frozen."""
import argparse
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import OUT, SEEDS, load_inputs, write_json, assert_identity
from experiments.gaussian_memory_attention_evaluate import aggregate
from experiments.gaussian_memory_setup import sha256
from models.gaussian_memory_attention import GaussianAttentionReader, geometry_inputs

DEST = OUT / 'visibility_diagnostic1'


def transmittance(alpha, depth, reverse=False):
    order = depth.argsort(dim=-1, descending=reverse)[:, None].expand_as(alpha)
    a = alpha.gather(-1, order)
    cumulative = torch.log1p(-a.clamp(max=.999)).cumsum(-1)
    exclusive = torch.cat([torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]], -1)
    result = torch.zeros_like(alpha).scatter(-1, order, exclusive).exp()
    return torch.cat([result, cumulative[..., -1:].exp()], -1)


def plan():
    DEST.mkdir(exist_ok=True)
    assert not (DEST / 'plan.json').exists()
    write_json(DEST / 'plan.json', {
        'stage': 'post-result development diagnostic, not a new trained method or fresh confirmation',
        'hypothesis': 'Some Gaussian-bias errors arise because its broad projected regions attend through occluders in memory trained by alpha compositing',
        'competitor': 'Incorrect geometry or generic sharpening, rather than correct front-to-back visibility, explains the errors',
        'fixed_models': 'All three Gaussian final-2000-step models; no tuning or optimization',
        'inputs': 'Same frozen memory, query camera, original learned attention probabilities; targets only for scoring',
        'operator': 'alpha_j=opacity_j*exp(-mahalanobis/2), invalid entries zero, capped .999; T_j=product of (1-alpha) in front of j, T_background=product of all (1-alpha). Multiply original attention probabilities by T and renormalize; leave V/output weights unchanged.',
        'conditions': ['front_to_back', 'reverse_depth'],
        'limits': 'Analytic projected attenuation; not exact gsplat tile/truncation/early-termination behavior. Post-training intervention may also create mismatch.',
        'decision': 'Front-only improvement supports a visibility contribution; equal/reverse advantage weakens that explanation. No outcome promotes an architecture or activates the failed original confirmation.',
        'original_scores_sha256': sha256(OUT / 'evaluation/scores.json'),
        'source_sha256': sha256(__file__)})


def evaluate():
    torch.set_num_threads(4)
    assert_identity()
    p = json.loads((DEST / 'plan.json').read_text())
    assert sha256(__file__) == p['source_sha256']
    assert sha256(OUT / 'evaluation/scores.json') == p['original_scores_sha256']
    assert not (DEST / 'scores.json').exists()
    alpha = torch.tensor([[[.5, .25]]])
    depth = torch.tensor([[1., 2.]])
    torch.testing.assert_close(transmittance(alpha, depth), torch.tensor([[[1., .5, .375]]]))
    torch.testing.assert_close(transmittance(alpha, depth, True), torch.tensor([[[.75, 1., .375]]]))
    models = {}
    for seed in SEEDS:
        ckpt = torch.load(OUT / f'gaussian_{seed}/step_2000.pt', weights_only=False, map_location='cuda')
        model = GaussianAttentionReader('gaussian', ckpt['config']['center_variance']).cuda().eval()
        model.load_state_dict(ckpt['model']); models[seed] = model
    scores, figures = [], {}
    with torch.no_grad():
        for item in load_inputs('cuda'):
            row, raw, memory = item['row'], item['raw'], item['memory']
            qids = sorted(row['training_query'] + row['heldout_query'])
            c2w, K = raw['c2w'][qids].cuda(), raw['K'][qids].cuda()
            target = raw['teacher'][qids].cuda().flatten(1, 2)
            inputs = geometry_inputs(memory, c2w, K)
            alpha = inputs['mahalanobis'].mul(-.5).exp() * memory['opacities'][None, None]
            alpha = alpha * inputs['valid'][:, None]
            R = c2w[:, :3, :3].transpose(-1, -2)
            center = torch.einsum('bij,bnj->bni', R, memory['means'][None] - c2w[:, None, :3, 3])
            B, P, N = alpha.shape
            figures[row['id']] = {}
            for seed, model in models.items():
                normal, attention = model(inputs, return_attention=True)
                H, D = model.heads, model.width // model.heads
                values = model.value(inputs['value']).view(B, N, H, D).transpose(1, 2)
                for condition in ('front_to_back', 'reverse_depth'):
                    T = transmittance(alpha, center[..., 2], reverse=condition == 'reverse_depth')
                    weights = attention * T[:, None]
                    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)
                    read = (weights[..., :-1] @ values).transpose(1, 2).reshape(B, P, model.width)
                    prediction = model.output(read)
                    assert torch.isfinite(prediction).all()
                    mse = (prediction - target).square().mean((1, 2))
                    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean(1)
                    ratio = prediction.norm(dim=-1).mean(1) / target.norm(dim=-1).mean(1).clamp_min(1e-8)
                    background = weights[..., -1].mean((1, 2))
                    if seed == SEEDS[0]:
                        figures[row['id']][condition] = prediction.cpu().reshape(-1, 15, 20, 32)
                    for i, q in enumerate(qids):
                        role = ('train_query' if q in row['training_query'] else
                                'heldout_view' if row['split'] == 'fit' else 'heldout_scene')
                        scores.append({'episode': row['id'], 'scene': row['scene'], 'query_ordinal': q,
                                       'frame': row['frames'][q]['index'], 'role': role, 'variant': 'gaussian',
                                       'condition': condition, 'seed': seed, 'mse': mse[i].item(),
                                       'cosine': cosine[i].item(), 'norm_ratio': ratio[i].item(),
                                       'background_mass': background[i].item()})
    fields = ['role', 'variant', 'condition', 'seed']
    result = {'plan_sha256': sha256(DEST / 'plan.json'), 'scores': scores,
              'summary': aggregate(scores, fields), 'per_scene': aggregate(scores, ['scene', *fields])}
    write_json(DEST / 'scores.json', result)
    torch.save(figures, DEST / 'visual_data.pt')
    assert_identity()
    print(json.dumps({'diagnostic_complete': True, 'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['plan', 'evaluate'])
    args = parser.parse_args()
    plan() if args.command == 'plan' else evaluate()
