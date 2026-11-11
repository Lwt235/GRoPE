"""Nine fixed-budget fits differing only in reference access and target choice."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_context_data import OUT, SEEDS, VARIANTS, model_path, initial_source, checked_cache, load_record
from models.gaussian_memory_context import ContextAdapter
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_view_token import normalized_error


def train():
    plan = checked_cache(); torch.set_num_threads(4); data = []
    for slot, rid in enumerate(plan['training_ids']):
        meta = next(r for r in plan['catalog'] if r['id'] == rid)
        d = load_record(meta, 'cuda'); raw, rec = d['raw'], d['rec']
        keep = torch.tensor([r == 'train' for r in rec['roles']]); query = rec['query'][keep]
        assert int(keep.sum()) == 24 and set(query.tolist()) <= set(d['target']['query_ids'])
        g = projected_inputs(d['s1'], raw['c2w'].cuda(), raw['K'].cuda())
        data.append({'id': rid, 'slot': slot, 'query': query,
                     'x': rec['input_tokens'][keep].cuda(), 'y': d['target']['context_targets'][query].cuda(),
                     'mask': rec['mask'][keep].cuda(),
                     'geometry': {k: g[k] for k in ('query', 'key', 'value', 'distance2', 'log_weight')}})
        del d, g
    for seed in SEEDS:
        for variant in VARIANTS:
            dest = model_path(variant, seed).parent; assert not dest.exists(); dest.mkdir()
            initial = initial_source(variant, seed)
            ckpt = torch.load(initial, map_location='cuda', weights_only=False)
            torch.manual_seed(seed)
            model = ContextAdapter(variant, plan['center_variance']).cuda()
            model.load_state_dict(ckpt['model'])
            assert all(torch.equal(v, ckpt['model'][k]) for k, v in model.state_dict().items())
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            config = {'variant': variant, 'target': 'query_first_S5_context', 'seed': seed,
                      'steps': 2000, 'batch': 2, 'loss': 'uniform_all_patches', 'lr': 3e-4,
                      'weight_decay': .01, 'clip': 5., 'center_variance': plan['center_variance'],
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'inactive_parameters': 644 if variant == 'content' else 0,
                      'initial_sha256': sha256(initial), 'plan_sha256': sha256(OUT / 'plan.json'),
                      'target_identity_sha256': sha256(OUT / 'target_cache_identity.json'),
                      'training_ids': plan['training_ids']}
            assert config['parameters'] == (530548 if variant == 'wide' else 531652)
            write_json(dest / 'config.json', config)
            torch.save({'model': model.state_dict(), 'config': config}, dest / 'initial.pt')
            rng = np.random.default_rng(seed); started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            with (dest / 'training.jsonl').open('w', buffering=1) as log:
                for step in range(1, 2001):
                    d = data[int(rng.integers(len(data)))]; ids = rng.choice(len(d['x']), 2, replace=False).tolist()
                    q = d['query'][ids].tolist(); g = {k: v[q] for k, v in d['geometry'].items()}
                    prediction = model(d['x'][ids], g)
                    error = normalized_error(prediction, d['y'][ids]); mask = d['mask'][ids]
                    loss = error.mean()
                    opt.zero_grad(set_to_none=True); loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    assert torch.isfinite(loss) and torch.isfinite(grad), (variant, seed, step)
                    opt.step()
                    row = {'step': step, 'episode': d['id'], 'slot': d['slot'], 'sample_ids': ids,
                           'loss': loss.item(), 'hidden': error[mask].mean().item(),
                           'visible': error[~mask].mean().item(), 'gradient_norm': grad.item(),
                           'seconds': time.monotonic() - started}
                    log.write(json.dumps(row) + '\n')
                    if step == 1 or step % 500 == 0:
                        print(json.dumps({'run': dest.name, **row}), flush=True)
                    if step % 500 == 0:
                        torch.save({'model': model.state_dict(), 'config': config, 'step': step}, dest / f'step_{step:04d}.pt')
            torch.cuda.synchronize()
            write_json(dest / 'status.json', {'complete': True, 'steps': 2000, 'seconds': time.monotonic() - started,
                       'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3})
            print(json.dumps({'finished': dest.name}), flush=True)
            del model, opt, ckpt
    write_json(OUT / 'training_status.json', {'complete': True, 'new_runs': 9, 'reused_models': 6})


if __name__ == '__main__':
    train()
