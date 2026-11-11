"""Post-result, no-training diagnosis of 2D priors seen in feature visualizations."""
import collections
import json

import numpy as np
import torch

from experiments.gaussian_memory_data import CACHE, REPORT, REF, TRAIN_QUERY


def main():
    torch.set_num_threads(4)
    output = REPORT / 'template_diagnostic1'
    output.mkdir(exist_ok=True)
    plan = {'timing': 'after primary scores and three view-grid figures, before computing these diagnostics',
            'observation': 'first three teacher PCA channels show broad spatial color bands across scenes',
            'hypothesis': 'view-specific Gaussian reading contains information beyond a fixed 2D feature prior or reference copying',
            'competing_explanation': 'the teacher target mostly rewards a shared 2D position template or copying one nearby view',
            'fixed_controls': ['training-query pixelwise mean template', 'same-episode reference pixelwise mean',
                              'nearest reference by mean squared world unit-depth ray-endpoint distance; no target image'],
            'decision': 'If simple templates or copying match or exceed the model, limit the claim accordingly; do not retune models or select a new checkpoint.',
            'new_training': False, 'fresh_confirmation': False,
            'memory_scope': 'nearest-reference control retains all four 2D maps; it is not matched-memory compression'}
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    rows = json.loads((REPORT / 'data_manifest.json').read_text())['episodes']
    data = {r['id']: torch.load(CACHE / (r['id'] + '.pt'), weights_only=False, map_location='cpu') for r in rows}
    global_template = torch.cat([data[r['id']]['teacher'][TRAIN_QUERY] for r in rows if r['split'] == 'fit']).mean(0)
    y, x = torch.meshgrid(torch.arange(15) + .5, torch.arange(20) + .5, indexing='ij')
    uv = torch.stack([x, y, torch.ones_like(x)], -1)
    scores = []
    for row in rows:
        d = data[row['id']]
        rays = torch.einsum('hwj,vij->vhwi', uv, torch.linalg.inv(d['K']))
        points = torch.einsum('vhwi,vji->vhwj', rays, d['c2w'][:, :3, :3]) + d['c2w'][:, None, None, :3, 3]
        template = d['teacher'][REF].mean(0)
        for q in sorted(row['training_query'] + row['heldout_query']):
            nearest = REF[int((points[REF] - points[q]).square().mean((1, 2, 3)).argmin())]
            predictions = {'global_pixel_template': global_template, 'reference_pixel_mean': template,
                           'nearest_reference_copy': d['teacher'][nearest]}
            role = 'train_query' if q in row['training_query'] else 'heldout_view' if row['split'] == 'fit' else 'heldout_scene'
            for name, predicted in predictions.items():
                scores.append({'episode': row['id'], 'scene': row['scene'], 'role': role, 'control': name,
                    'query_ordinal': q, 'nearest_reference_ordinal': nearest,
                    'mse': (predicted - d['teacher'][q]).square().mean().item()})
    primary = json.loads((REPORT / 'evaluation/scores.json').read_text())
    all_scores = scores + [{'episode': s['episode'], 'scene': s['scene'], 'role': s['role'],
                            'control': 'learned_geometry', 'mse': s['mse']} for s in primary['scores']
                           if s['variant'] == 'learned_geometry' and s['condition'] == 'normal']
    grouped, scene = collections.defaultdict(list), collections.defaultdict(list)
    for s in all_scores:
        grouped[s['role'], s['control']].append(s['mse'])
        scene[s['scene'], s['role'], s['control']].append(s['mse'])
    summary = [{'role': k[0], 'control': k[1], 'mse': float(np.mean(v)), 'queries': len(v)} for k, v in grouped.items()]
    per_scene = [{'scene': k[0], 'role': k[1], 'control': k[2], 'mse': float(np.mean(v))} for k, v in scene.items()]
    (output / 'scores.json').write_text(json.dumps({'plan': plan, 'summary': summary, 'per_scene': per_scene, 'scores': scores}, indent=2) + '\n')
    print(json.dumps({'summary': summary}), flush=True)


if __name__ == '__main__': main()
