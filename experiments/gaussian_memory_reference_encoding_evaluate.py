"""Matched S1/S4/Wide evaluation and within-view correspondence interventions."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_reference_encoding_data import OUT, SEEDS, VARIANTS, model_path, checked_cache, load_record, spatial_permutation
from experiments.gaussian_memory_direct_evaluate import error_row
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_direct_reference import DirectReferenceAdapter
from models.gaussian_memory_view_token import normalized_error


@torch.inference_mode()
def evaluate():
    plan = checked_cache(); torch.set_num_threads(4); nets = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            ckpt = torch.load(model_path(variant, seed), map_location='cuda', weights_only=False)
            model = DirectReferenceAdapter('wide' if variant == 'wide' else 'direct', ckpt['config']['center_variance']).cuda().eval()
            model.load_state_dict(ckpt['model']); nets[(variant, seed)] = model
    dest = OUT / 'evaluation'; assert not dest.exists(); dest.mkdir()
    paths = [model_path(v, s) for v in VARIANTS for s in SEEDS]
    lock = {'models': {str(p): sha256(p) for p in paths}, 'model_selection': False,
            'confirmation_claim': False, 'plan_sha256': sha256(OUT / 'plan.json')}
    write_json(OUT / 'evaluation_lock.json', lock)
    rows, preservation = [], []
    for meta in plan['catalog']:
        d = load_record(meta, 'cuda'); raw, rec = d['raw'], d['rec']
        keep = torch.tensor([r == 'evaluation' for r in rec['roles']]); query = rec['query'][keep]
        x, y = rec['input_tokens'][keep].cuda(), rec['teacher_tokens'][query].cuda()
        mask = rec['mask'][keep].cuda(); c2w, K = raw['c2w'][query].cuda(), raw['K'][query].cuda()
        base = {'episode': meta['id'], 'scene': meta['row']['scene'], 'split': meta['eval_split'],
                'query_count': len(query), 'cohort': meta['cohort']}
        errors = {'backbone': normalized_error(x, y).cpu()}
        rows.append(dict(base, variant='backbone', seed=None, condition='normal', **error_row(errors['backbone'].cuda(), mask)))
        permutation = spatial_permutation(d['s4'], meta['id'])
        for variant in VARIANTS:
            memory = d['s4'] if variant == 'wide' else d[variant]
            for condition in (('normal',) if variant == 'wide' else ('normal', 'removed', 'shuffled')):
                m = memory
                if condition == 'shuffled':
                    m = dict(memory, features=memory['features'][permutation])
                g = projected_inputs(m, c2w, K)
                for seed in SEEDS:
                    net = nets[(variant, seed)]
                    error = normalized_error(net(x, g, condition), y)
                    rows.append(dict(base, variant=variant, seed=seed, condition=condition, **error_row(error, mask)))
                    if condition == 'normal':
                        preservation.append(dict(base, variant=variant, seed=seed,
                                                 clean_input_mse=normalized_error(net(y, g), y).mean().item()))
                    if seed == SEEDS[0]:
                        errors[variant if condition == 'normal' else variant + '_' + condition] = error.cpu()
        torch.save({'episode': meta['row'], 'query': query, 'mask': mask.cpu(), 'errors': errors,
                    'rgb': raw['rgb'], 'split': meta['eval_split'], 'permutation': permutation.cpu()},
                   dest / (meta['id'] + '_maps.pt'))
        print(json.dumps({'evaluated': meta['id'], 'split': meta['eval_split']}), flush=True)
    write_json(dest / 'scores.json', {'rows': rows, 'preservation': preservation,
               'model_sha256': lock['models'], 'confirmation_claim': False, 'episodes': len(plan['catalog'])})


if __name__ == '__main__':
    evaluate()
