"""Fixed narrow/broad models on the same clips; old development training is explicit."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_coverage_data import OUT, SEEDS, VARIANTS, model_path, checked_cache, load_record
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_direct_reference import DirectReferenceAdapter
from models.gaussian_memory_view_token import ViewTokenAdapter, normalized_error
from experiments.gaussian_memory_direct_evaluate import error_row


def load_models():
    models = {}
    for regime in ('narrow', 'broad'):
        for variant in VARIANTS:
            for seed in SEEDS:
                ckpt = torch.load(model_path(regime, variant, seed), map_location='cuda', weights_only=False)
                cls = ViewTokenAdapter if variant in ('local', 'center') else DirectReferenceAdapter
                net = cls(variant, ckpt['config']['center_variance']).cuda().eval()
                net.load_state_dict(ckpt['model']); models[(regime, variant, seed)] = net
    return models


@torch.inference_mode()
def evaluate():
    plan = checked_cache(); torch.set_num_threads(4); nets = load_models()
    dest = OUT / 'evaluation'; assert not dest.exists(); dest.mkdir()
    paths = [model_path(r, v, s) for r in ('narrow', 'broad') for v in VARIANTS for s in SEEDS]
    lock = {'models': {str(p): sha256(p) for p in paths}, 'model_selection': False,
            'confirmation_claim': False, 'plan_sha256': sha256(OUT / 'plan.json')}
    write_json(OUT / 'evaluation_lock.json', lock)
    rows, preservation = [], []
    for index, meta in enumerate(plan['catalog']):
        d = load_record(meta, 'cuda'); raw, rec = d['raw'], d['rec']
        keep = torch.tensor([r == 'evaluation' for r in rec['roles']]); query = rec['query'][keep]
        x, y = rec['input_tokens'][keep].cuda(), rec['teacher_tokens'][query].cuda()
        mask = rec['mask'][keep].cuda(); c2w, K = raw['c2w'][query].cuda(), raw['K'][query].cuda()
        base = {'episode': meta['id'], 'scene': meta['row']['scene'], 'split': meta['eval_split'],
                'cohort': meta['cohort'], 'query_count': len(query),
                'broad_training_scene': meta['eval_split'] in ('common_train_heldout', 'added_train_heldout', 'old_fit_other_clip')}
        errors = {'backbone': normalized_error(x, y).cpu()}
        rows.append(dict(base, regime='frozen', variant='backbone', seed=None, condition='normal',
                         **error_row(errors['backbone'].cuda(), mask)))
        # Match the original swap source inside each historical cohort. The
        # additional single-cohort records fall back to the full catalog.
        cohort = [r for r in plan['catalog'] if r['cohort'] == meta['cohort']]
        offset = next(i for i, r in enumerate(cohort) if r['id'] == meta['id'])
        candidates = cohort[offset + 1:] + cohort[:offset + 1] + plan['catalog']
        wrong_meta = next(r for r in candidates if r['row']['scene'] != meta['row']['scene'])
        wrong = load_record(wrong_meta, 'cuda')
        gs_g = projected_inputs(d['memory'], c2w, K)
        for variant in ('local', 'wide'):
            for regime in ('narrow', 'broad'):
                for seed in SEEDS:
                    net = nets[(regime, variant, seed)]
                    error = normalized_error(net(x, gs_g), y)
                    rows.append(dict(base, regime=regime, variant=variant, seed=seed, condition='normal', **error_row(error, mask)))
                    preservation.append(dict(base, regime=regime, variant=variant, seed=seed,
                                             clean_input_mse=normalized_error(net(y, gs_g), y).mean().item()))
                    if seed == SEEDS[0]:
                        errors[regime + '_' + variant] = error.cpu()
        for variant, key in [('center', 'memory'), ('direct', 'direct_memory')]:
            for condition in ('normal', 'removed', 'swapped'):
                g = projected_inputs(wrong[key] if condition == 'swapped' else d[key], c2w, K)
                for regime in ('narrow', 'broad'):
                    for seed in SEEDS:
                        net = nets[(regime, variant, seed)]
                        error = normalized_error(net(x, g, condition), y)
                        rows.append(dict(base, regime=regime, variant=variant, seed=seed, condition=condition, **error_row(error, mask)))
                        if condition == 'normal':
                            preservation.append(dict(base, regime=regime, variant=variant, seed=seed,
                                                     clean_input_mse=normalized_error(net(y, g), y).mean().item()))
                            if seed == SEEDS[0]:
                                errors[regime + '_' + variant] = error.cpu()
        torch.save({'episode': meta['row'], 'query': query, 'mask': mask.cpu(), 'errors': errors,
                    'rgb': raw['rgb'], 'split': meta['eval_split']}, dest / (meta['id'] + '_maps.pt'))
        print(json.dumps({'evaluated': meta['id'], 'split': meta['eval_split']}), flush=True)
        del d, wrong, gs_g
    write_json(dest / 'scores.json', {'rows': rows, 'preservation': preservation,
               'model_sha256': lock['models'], 'confirmation_claim': False, 'episodes': len(plan['catalog'])})


if __name__ == '__main__':
    evaluate()
