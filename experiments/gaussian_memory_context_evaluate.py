"""Matched context-target evaluation, with reference and binding interventions."""
import json

import numpy as np
import torch
from torch.nn import functional as F

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_context_data import OUT, SEEDS, VARIANTS, model_path, checked_cache, load_record
from experiments.gaussian_memory_direct_evaluate import error_row
from models.gaussian_memory_context import ContextAdapter, permute_visible_content
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_view_token import normalized_error


@torch.inference_mode()
def evaluate():
    plan = checked_cache(); torch.set_num_threads(4); nets = {}
    specifications = [('context', v, s) for v in VARIANTS for s in SEEDS]
    specifications += [('single', v, s) for v in ('wide', 'spatial') for s in SEEDS]
    for regime, variant, seed in specifications:
        ckpt = torch.load(model_path(variant, seed, regime), map_location='cuda', weights_only=False)
        model = ContextAdapter(variant, ckpt['config']['center_variance']).cuda().eval()
        model.load_state_dict(ckpt['model']); nets[(regime, variant, seed)] = model
    dest = OUT / 'evaluation'; assert not dest.exists(); dest.mkdir()
    lock = {'models': {str(model_path(v, s, r)): sha256(model_path(v, s, r)) for r, v, s in specifications},
            'model_selection': False, 'confirmation_claim': False, 'plan_sha256': sha256(OUT / 'plan.json')}
    write_json(OUT / 'evaluation_lock.json', lock)
    rows, full_query, teacher_diagnostics, invariance = [], [], [], []
    for meta in plan['catalog']:
        d = load_record(meta, 'cuda'); raw, rec, target = d['raw'], d['rec'], d['target']
        keep = torch.tensor([r == 'evaluation' for r in rec['roles']]); query = rec['query'][keep]
        assert set(query.tolist()) <= set(target['query_ids'])
        x, y = rec['input_tokens'][keep].cuda(), target['context_targets'][query].cuda()
        single = rec['teacher_tokens'][query].cuda(); repeated = target['repeat_query_targets'][query].cuda()
        mask = rec['mask'][keep].cuda()
        base = {'episode': meta['id'], 'scene': meta['row']['scene'], 'split': meta['eval_split'],
                'query_count': len(query), 'cohort': meta['cohort']}
        errors = {'backbone': normalized_error(x, y).cpu()}
        rows.append(dict(base, regime='none', variant='backbone', seed=None, condition='normal',
                         **error_row(errors['backbone'].cuda(), mask), original_s1_all_mse=normalized_error(x, single).mean().item()))
        change = F.layer_norm(y.float(), (2048,)) - F.layer_norm(repeated.float(), (2048,))
        global_change = change.mean(1, keepdim=True)
        teacher_diagnostics.append(dict(base,
            real_reference_vs_repeat_query_mse=change.square().mean().item(),
            spatially_centered_reference_change_mse=(change - global_change).square().mean().item(),
            global_reference_change_mse=global_change.square().mean().item(),
            s1_vs_context_target_mse=normalized_error(single, y).mean().item(),
            s1_vs_repeat_query_target_mse=normalized_error(single, repeated).mean().item()))
        g = projected_inputs(d['s1'], raw['c2w'][query].cuda(), raw['K'][query].cuda())
        shuffled, permutations = permute_visible_content(g, meta['id'], query.tolist())
        for regime, variant, seed in specifications:
            net = nets[(regime, variant, seed)]; normal = None
            conditions = ('normal', 'removed', 'shuffled') if regime == 'context' and variant != 'wide' else ('normal',)
            for condition in conditions:
                geometry = shuffled if condition == 'shuffled' else g
                prediction = net(x, geometry, condition)
                error = normalized_error(prediction, y)
                rows.append(dict(base, regime=regime, variant=variant, seed=seed, condition=condition,
                                 **error_row(error, mask), original_s1_all_mse=normalized_error(prediction, single).mean().item()))
                if condition == 'normal':
                    normal = prediction
                    full_prediction = net(single, geometry)
                    full_query.append(dict(base, regime=regime, variant=variant, seed=seed,
                        full_query_context_mse=normalized_error(full_prediction, y).mean().item(),
                        full_query_s1_change_mse=normalized_error(full_prediction, single).mean().item()))
                if variant == 'content' and condition == 'shuffled':
                    invariance.append(dict(base, seed=seed, max_output_difference=(prediction - normal).abs().max().item(),
                                           max_normalized_difference=(F.layer_norm(prediction, (2048,)) - F.layer_norm(normal, (2048,))).abs().max().item()))
                if seed == SEEDS[0]:
                    name = regime + '_' + variant + ('' if condition == 'normal' else '_' + condition)
                    errors[name] = error.cpu()
        torch.save({'episode': meta['row'], 'query': query, 'mask': mask.cpu(), 'errors': errors,
                    'rgb': raw['rgb'], 'split': meta['eval_split'], 'permutations': permutations.cpu(),
                    'valid_slots': torch.isfinite(g['log_weight']).cpu(),
                    'teacher_reference_change': change.square().mean(-1).cpu()},
                   dest / (meta['id'] + '_maps.pt'))
        print(json.dumps({'evaluated': meta['id'], 'split': meta['eval_split']}), flush=True)
    write_json(dest / 'scores.json', {'rows': rows, 'full_query': full_query,
               'teacher_diagnostics': teacher_diagnostics, 'content_permutation_invariance': invariance,
               'model_sha256': lock['models'], 'confirmation_claim': False, 'episodes': len(plan['catalog'])})


if __name__ == '__main__':
    evaluate()
