"""Paired evaluation and bounded memory interventions, no checkpoint selection."""
import argparse
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import load_inputs, write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_view_data import OUT, TOKEN_CACHE, CONF_CACHE, SEEDS, VARIANTS, checked
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_view_token import ViewTokenAdapter, normalized_error


def models():
    result = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            ckpt = torch.load(OUT / f'{variant}_{seed}/step_2000.pt', weights_only=False, map_location='cuda')
            net = ViewTokenAdapter(variant, ckpt['config']['center_variance']).cuda().eval()
            net.load_state_dict(ckpt['model']); result[(variant, seed)] = net
    return result


def items(transfer=False):
    if not transfer:
        return load_inputs('cuda')
    identity = json.loads((OUT / 'transfer_cache_identity.json').read_text())
    for path, expected in identity['sha256'].items(): assert sha256(path) == expected
    return [{'row': r, 'raw': torch.load(CONF_CACHE / (r['id'] + '_raw.pt'), weights_only=False),
             'memory': torch.load(OUT / 'transfer_memories' / (r['id'] + '.pt'), weights_only=False,
                                  map_location='cuda')['memory']} for r in identity['episodes']]


def error_row(error, mask, support, threshold):
    low, high = mask & (support < threshold), mask & (support >= threshold)
    return {'hidden_mse': error[mask].mean().item(), 'visible_mse': error[~mask].mean().item(),
            'support_low_mse': error[low].mean().item() if low.any() else None,
            'support_high_mse': error[high].mean().item() if high.any() else None,
            'support_low_count': int(low.sum()), 'support_high_count': int(high.sum())}


def summary(rows):
    result = {}
    for split in sorted({r['split'] for r in rows}):
        result[split] = {}
        for variant in sorted({r['variant'] for r in rows}):
            for condition in sorted({r['condition'] for r in rows}):
                selected = [r for r in rows if r['split'] == split and r['variant'] == variant and r['condition'] == condition]
                if selected:
                    result[split][variant + ':' + condition] = {k: float(np.mean([r[k] for r in selected if r[k] is not None]))
                         for k in ('hidden_mse', 'visible_mse', 'support_low_mse', 'support_high_mse')
                         if any(r[k] is not None for r in selected)}
    return result


@torch.inference_mode()
def evaluate(transfer=False):
    checked(); torch.set_num_threads(4)
    nets, data = models(), items(transfer)
    threshold = json.loads((OUT / 'support_threshold.json').read_text())['threshold']
    dest = OUT / ('transfer_evaluation' if transfer else 'development_evaluation')
    assert not dest.exists(); dest.mkdir()
    cache = CONF_CACHE if transfer else TOKEN_CACHE
    identity = json.loads((OUT / ('transfer_cache_identity.json' if transfer else 'cache_identity.json')).read_text())
    rows, preservation = [], []
    for idx, item in enumerate(data):
        rid, raw, memory = item['row']['id'], item['raw'], item['memory']
        path = cache / (rid + '.pt'); assert sha256(path) == identity['sha256'][str(path)]
        rec = torch.load(path, weights_only=False)
        keep = torch.tensor([s == 'evaluation' for s in rec['roles']])
        query = rec['query'][keep]; masks = rec['mask'][keep].cuda()
        x, y = rec['input_tokens'][keep].cuda(), rec['teacher_tokens'][query].cuda()
        K, c2w = raw['K'][query].cuda(), raw['c2w'][query].cuda()
        geometry = projected_inputs(memory, c2w, K)
        support = (geometry['log_weight'][:, None] - .5 * geometry['distance2'] / .7146816253662109).exp().amax(-1)
        split = 'transfer' if transfer else ('heldout_view' if item['row']['split'] == 'fit' else 'development_scene')
        base = {'episode': rid, 'scene': item['row']['scene'], 'split': split}
        errors = {'backbone': normalized_error(x, y).cpu()}
        rows.append(dict(base, variant='backbone', seed=None, condition='normal',
                         **error_row(errors['backbone'].cuda(), masks, support, threshold)))
        # Feature-only spatial shuffle leaves all geometry fixed.
        permutation = torch.randperm(256, generator=torch.Generator().manual_seed(371 + idx)).cuda()
        shuffled = {k: v.clone() for k, v in memory.items()}; shuffled['features'] = memory['features'][permutation]
        wrong = next(d['memory'] for d in data[idx + 1:] + data[:idx + 1] if d['row']['scene'] != item['row']['scene'])
        first_cam = raw['c2w'][0].cuda().expand_as(c2w)
        geometries = {'normal': geometry, 'removed': geometry,
                      'shuffled': projected_inputs(shuffled, c2w, K),
                      'swapped': projected_inputs(wrong, c2w, K),
                      'wrong_camera': projected_inputs(memory, first_cam, K)}
        for (variant, seed), net in nets.items():
            for condition in (['normal'] if variant == 'local' else list(geometries)):
                prediction = net(x, geometries[condition], condition)
                error = normalized_error(prediction, y)
                rows.append(dict(base, variant=variant, seed=seed, condition=condition,
                                 **error_row(error, masks, support, threshold)))
                if condition == 'normal' and seed == SEEDS[0]: errors[variant] = error.cpu()
            clean_error = normalized_error(net(y, geometry), y)
            preservation.append(dict(base, variant=variant, seed=seed, clean_input_mse=clean_error.mean().item()))
        torch.save({'query': query, 'mask': masks.cpu(), 'support': support.cpu(), 'errors': errors,
                    'rgb': raw['rgb'], 'episode': item['row']}, dest / (rid + '_maps.pt'))
        print(json.dumps({'evaluated': rid, 'split': split}), flush=True)
    result = {'rows': rows, 'summary': summary(rows), 'preservation': preservation,
              'model_sha256': {str(p): sha256(p) for p in OUT.glob('*/step_2000.pt')}}
    write_json(dest / 'scores.json', result)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['development', 'transfer'])
    a = p.parse_args(); evaluate(a.command == 'transfer')
