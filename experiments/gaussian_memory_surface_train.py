"""Matched fixed-budget fits for a shared surface memory."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_surface_data import OUT, SEEDS, VARIANTS, checked, load_record, model_path
from models.gaussian_memory import seeds_to_device
from models.gaussian_memory_surface import SurfaceMemory


def loss_function(prediction, target, valid):
    return ((prediction - target).square() * valid).sum(-1).div(valid.sum(-1).clamp_min(1)).mean()


def train():
    plan = checked(True); torch.set_num_threads(4); data = []
    for rid in plan['training_ids']:
        meta = next(r for r in plan['catalog'] if r['id'] == rid); d = load_record(meta)
        raw, rec, target = d['raw'], d['rec'], d['target']
        keep = torch.tensor([r == 'train' for r in rec['roles']]); query = rec['query'][keep]
        assert int(keep.sum()) == 24 and not set(query.tolist()) & {0, 2, 4, 6, 8, 10, 12}
        data.append({'id': rid, 'x': rec['input_tokens'][keep].cuda(), 'query': query.cuda(),
                     'y': target['log_depth_m'][query].cuda(), 'valid': target['valid'][query].cuda(),
                     'K': raw['K'].cuda(), 'c2w': raw['c2w'].cuda(), 'unit': raw['diagnostics']['unit_meters'],
                     'seeds': seeds_to_device(raw, 'cuda')})
    for seed in SEEDS:
        torch.manual_seed(seed)
        common = SurfaceMemory('gaussian', plan['local_width'], plan['center_variance'])
        initial = {k: v.detach().clone() for k, v in common.state_dict().items()}; del common
        for variant in VARIANTS:
            dest = model_path(variant, seed).parent; assert not dest.exists(); dest.mkdir()
            torch.manual_seed(seed)
            model = SurfaceMemory(variant, plan['local_width'], plan['center_variance']).cuda()
            if variant != 'local': model.load_state_dict(initial)
            opt = torch.optim.AdamW(model.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
            config = {'variant': variant, 'seed': seed, 'steps': 2000, 'batch': 2,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'inactive_writer_output_parameters': 9 * 129 if variant == 'fixed' else 0,
                      'training_ids': plan['training_ids'], 'plan_sha256': sha256(OUT / 'plan.json')}
            torch.save({'model': model.state_dict(), 'config': config}, model_path(variant, seed, True))
            write_json(dest / 'config.json', config)
            rng = np.random.default_rng(seed + 73); losses = []; gradients = []
            torch.cuda.synchronize(); started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            with (dest / 'training.jsonl').open('w') as log:
                for step in range(1, 2001):
                    slot = int(rng.integers(len(data))); d = data[slot]
                    ids = rng.choice(len(d['x']), size=2, replace=False).tolist(); q = d['query'][ids]
                    opt.zero_grad(set_to_none=True)
                    pred = model(d['x'][ids], d['seeds'], d['c2w'][q], d['K'][q], d['unit'])
                    loss = loss_function(pred, d['y'][ids], d['valid'][ids]); assert torch.isfinite(loss)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan['gradient_clip'])
                    assert torch.isfinite(norm)
                    if variant != 'local' and step in (2, 100, 2000):
                        grad = model.writer.net[-1].weight.grad
                        gradients.append({'step': step, 'center': float(grad[:3].norm()),
                                          'shape': float(grad[3:9].norm()), 'opacity': float(grad[9].norm()),
                                          'content': float(grad[10:].norm())})
                    opt.step(); losses.append(float(loss))
                    row = {'step': step, 'slot': slot, 'episode': d['id'], 'sample_ids': ids,
                           'loss': float(loss), 'gradient_norm': float(norm)}
                    log.write(json.dumps(row) + '\n')
                    if step % 500 == 0:
                        log.flush(); print(json.dumps({'training': variant, 'seed': seed, 'step': step,
                                                      'loss_last100': float(np.mean(losses[-100:]))}), flush=True)
            torch.cuda.synchronize(); seconds = time.monotonic() - started
            torch.save({'model': model.state_dict(), 'config': config}, model_path(variant, seed))
            status = {'complete': True, 'steps': 2000, 'seconds': seconds,
                      'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3,
                      'first100': float(np.mean(losses[:100])), 'last100': float(np.mean(losses[-100:])),
                      'writer_gradients': gradients, 'model_sha256': sha256(model_path(variant, seed))}
            write_json(dest / 'status.json', status)
            print(json.dumps({'fit_complete': variant, 'seed': seed, **status}), flush=True)
            del model, opt; torch.cuda.empty_cache()
    write_json(OUT / 'evaluation_lock.json', {'plan_sha256': sha256(OUT / 'plan.json'),
               'models': {str(model_path(v, s)): sha256(model_path(v, s)) for v in VARIANTS for s in SEEDS}})


if __name__ == '__main__': train()
