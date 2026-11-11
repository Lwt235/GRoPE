"""Fixed loss controls on the unchanged actual-view-token adapter."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_view_train import load_training
from experiments.gaussian_memory_fidelity_data import OUT, SEEDS, VARIANTS, model_path, checked
from models.gaussian_memory_view_token import ViewTokenAdapter, normalized_error


def train():
    checked(); torch.set_num_threads(4)
    data, _ = load_training()
    for regime in ('B', 'C'):
        for seed in SEEDS:
            for variant in VARIANTS:
                dest = model_path(regime, variant, seed).parent
                assert not dest.exists(), 'preserve prior attempts'
                dest.mkdir()
                initial = model_path('A', variant, seed, initial=True)
                ckpt = torch.load(initial, weights_only=False, map_location='cuda')
                torch.manual_seed(seed)
                model = ViewTokenAdapter(variant, ckpt['config']['center_variance']).cuda()
                model.load_state_dict(ckpt['model'])
                assert all(torch.equal(v, ckpt['model'][k]) for k, v in model.state_dict().items())
                opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
                config = dict(ckpt['config'], regime=regime, hidden_visible_weight=None,
                              masked_patch_weight='uniform', clean_identity_weight=int(regime == 'C'),
                              plan_sha256=sha256(OUT / 'plan.json'), initial_sha256=sha256(initial),
                              masked_examples=4000, clean_examples=4000 if regime == 'C' else 0,
                              adapter_forward_batches=4000 if regime == 'C' else 2000)
                write_json(dest / 'config.json', config)
                torch.save({'config': config, 'model': model.state_dict()}, dest / 'initial.pt')
                rng, started = np.random.default_rng(seed), time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                with (dest / 'training.jsonl').open('w', buffering=1) as log:
                    for step in range(1, 2001):
                        d = data[int(rng.integers(len(data)))]
                        ids = rng.choice(len(d['x']), 2, replace=False).tolist()
                        geometry = {k: v[ids] for k, v in d['geometry'].items()}
                        # The masked forward has no clean-token argument or shared state.
                        pred = model(d['x'][ids], geometry)
                        error = normalized_error(pred, d['y'][ids]); mask = d['mask'][ids]
                        loss = error.mean()
                        clean = None
                        if regime == 'C':
                            clean_tokens = d['y'][ids].detach()
                            clean = normalized_error(model(clean_tokens, geometry), clean_tokens).mean()
                            if step == 1: assert float(clean) < 1e-12
                            loss = loss + clean
                        opt.zero_grad(set_to_none=True); loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        assert torch.isfinite(loss) and torch.isfinite(norm), (regime, variant, seed, step)
                        opt.step()
                        row = {'step': step, 'episode': d['id'], 'sample_ids': ids, 'loss': loss.item(),
                               'all_patch': error.mean().item(), 'hidden': error[mask].mean().item(),
                               'visible': error[~mask].mean().item(),
                               'clean_identity': clean.item() if clean is not None else None,
                               'gradient_norm': norm.item(), 'seconds': time.monotonic() - started}
                        log.write(json.dumps(row) + '\n')
                        if step == 1 or step % 500 == 0:
                            print(json.dumps({'run': dest.name, **row}), flush=True)
                        if step % 500 == 0:
                            torch.save({'model': model.state_dict(), 'config': config, 'step': step}, dest / f'step_{step:04d}.pt')
                torch.cuda.synchronize()
                write_json(dest / 'status.json', {'complete': True, 'steps': 2000,
                           'seconds': time.monotonic() - started,
                           'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3})
                print(json.dumps({'finished': dest.name, 'seconds': time.monotonic() - started}), flush=True)
                del model, opt, ckpt
    write_json(OUT / 'training_status.json', {'complete': True, 'new_runs': 12, 'reused_A_runs': 6})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['train'])
    p.parse_args(); train()
