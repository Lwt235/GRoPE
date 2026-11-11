"""All predeclared loss groups, paired inputs and memory interventions."""
import argparse
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import load_inputs, write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_fidelity_data import (OUT, OLD_OUT, TOKEN_CACHE, NEW_CACHE,
    SEEDS, VARIANTS, REGIMES, checked, assert_lock, model_path)
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_view_token import ViewTokenAdapter, normalized_error


def models():
    result = {}
    for regime in REGIMES:
        for seed in SEEDS:
            for variant in VARIANTS:
                ckpt = torch.load(model_path(regime, variant, seed), weights_only=False, map_location='cuda')
                net = ViewTokenAdapter(variant, ckpt['config']['center_variance']).cuda().eval()
                net.load_state_dict(ckpt['model']); result[(regime, variant, seed)] = net
    return result


def items(transfer=False):
    if not transfer: return load_inputs('cuda')
    identity = json.loads((OUT / 'transfer_cache_identity.json').read_text())
    for path, expected in identity['sha256'].items(): assert sha256(path) == expected
    return [{'row': r, 'raw': torch.load(NEW_CACHE / (r['id'] + '_raw.pt'), weights_only=False),
             'memory': torch.load(OUT / 'memories' / (r['id'] + '.pt'), weights_only=False,
                                  map_location='cuda')['memory']} for r in identity['episodes']]


def error_row(error, mask):
    return {'hidden_mse': error[mask].mean().item(), 'visible_mse': error[~mask].mean().item(),
            'all_patch_mse': error.mean().item()}


@torch.inference_mode()
def evaluate(transfer=False):
    assert_lock() if transfer else checked()
    torch.set_num_threads(4)
    nets, data = models(), items(transfer)
    dest = OUT / ('transfer_evaluation' if transfer else 'development_evaluation')
    assert not dest.exists(); dest.mkdir()
    cache = NEW_CACHE if transfer else TOKEN_CACHE
    identity = json.loads(((OUT / 'transfer_cache_identity.json') if transfer else (OLD_OUT / 'cache_identity.json')).read_text())
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
        split = 'transfer' if transfer else ('heldout_view' if item['row']['split'] == 'fit' else 'development_scene')
        base = {'episode': rid, 'scene': item['row']['scene'], 'split': split}
        errors = {'backbone': normalized_error(x, y).cpu()}
        rows.append(dict(base, regime='backbone', variant='backbone', seed=None, condition='normal',
                         **error_row(errors['backbone'].cuda(), masks)))
        permutation = torch.randperm(256, generator=torch.Generator().manual_seed(371 + idx)).cuda()
        shuffled = {k: v.clone() for k, v in memory.items()}; shuffled['features'] = memory['features'][permutation]
        wrong = next(d['memory'] for d in data[idx + 1:] + data[:idx + 1] if d['row']['scene'] != item['row']['scene'])
        geometries = {'normal': geometry, 'removed': geometry,
                      'shuffled': projected_inputs(shuffled, c2w, K),
                      'swapped': projected_inputs(wrong, c2w, K),
                      'wrong_camera': projected_inputs(memory, raw['c2w'][0].cuda().expand_as(c2w), K)}
        for (regime, variant, seed), net in nets.items():
            for condition in (['normal'] if variant == 'local' else list(geometries)):
                error = normalized_error(net(x, geometries[condition], condition), y)
                rows.append(dict(base, regime=regime, variant=variant, seed=seed, condition=condition,
                                 **error_row(error, masks)))
                if condition == 'normal' and seed == SEEDS[0]: errors[regime + '_' + variant] = error.cpu()
            clean_error = normalized_error(net(y, geometry), y)
            preservation.append(dict(base, regime=regime, variant=variant, seed=seed,
                                     clean_input_mse=clean_error.mean().item()))
        torch.save({'query': query, 'mask': masks.cpu(), 'errors': errors,
                    'rgb': raw['rgb'], 'episode': item['row']}, dest / (rid + '_maps.pt'))
        print(json.dumps({'evaluated': rid, 'split': split}), flush=True)
    result = {'rows': rows, 'preservation': preservation,
              'model_sha256': {str(model_path(r,v,s)): sha256(model_path(r,v,s))
                               for r in REGIMES for v in VARIANTS for s in SEEDS}}
    write_json(dest / 'scores.json', result)
    print(json.dumps({'complete': True, 'episodes': len(data), 'rows': len(rows)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['development', 'transfer'])
    a = p.parse_args(); evaluate(a.command == 'transfer')
