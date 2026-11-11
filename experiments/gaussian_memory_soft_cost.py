"""Bounded descriptive costs under shared GPU occupancy."""
import json
import os
import time

import numpy as np
import torch

from experiments.gaussian_memory_soft_data import OUT, SEEDS, load_priors, write_json, device_tree
from experiments.gaussian_memory_soft_evaluate import clip_samples, load_adapter
from experiments.gaussian_memory_view_data import load_backbone
from models.gaussian_memory_vggt_adapter import native_tail
from models.gaussian_memory_bound_read import bound_native_tail, projected_slot_weights, attention_prior
from models.gaussian_memory_soft_read import soft_attention_prior


@torch.no_grad()
def run():
    backbone = load_backbone(); adapters = {k: load_adapter(k, SEEDS[0])[0] for k in ('content', 'soft')}
    rows = []; torch.cuda.reset_peak_memory_stats(); rng = np.random.default_rng(20261111)
    for rid in ('transfer_00', 'confirm_00', 'fresh_00'):
        samples, initial, sources = clip_samples(rid, (0,)); sample = device_tree(samples[11, 0])
        content = initial['memory']['features']; projected = load_priors([rid])[rid, 11]
        priors = {'content': attention_prior(projected, 'content'), 'soft': soft_attention_prior(projected)}
        calls = {'native': lambda: native_tail(backbone, sample)}
        calls.update({k: (lambda k=k: bound_native_tail(backbone, sample, adapters[k], content, priors[k])) for k in adapters})
        times = {k: [] for k in calls}; orders = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for call in calls.values():
                for _ in range(2): call()
            for _ in range(5):
                order = rng.permutation(list(calls)).tolist(); orders.append(order)
                for key in order:
                    torch.cuda.synchronize(); start = time.perf_counter(); calls[key](); torch.cuda.synchronize()
                    times[key].append(time.perf_counter() - start)
        projection_times = []
        for iteration in range(7):
            torch.cuda.synchronize(); start = time.perf_counter()
            soft_attention_prior(projected_slot_weights(initial['memory'], sample['c2w'], sample['K']))
            torch.cuda.synchronize()
            if iteration >= 2: projection_times.append(time.perf_counter() - start)
        times['projection_and_soft_prior'] = projection_times
        rows.append({'id': rid, 'query': 11, 'condition': 0,
                     'timing': {k: {'median_seconds': float(np.median(v)), 'samples_seconds': v} for k, v in times.items()},
                     'interleaved_orders': orders, 'sources': sources,
                     'single_window_log_prior_bytes': priors['soft'].numel() * priors['soft'].element_size()})
    write_json(OUT / 'native_costs.json', {'rows': rows, 'parameters': 137472,
        'device': torch.cuda.get_device_name(), 'visible_device': os.getenv('CUDA_VISIBLE_DEVICES'),
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
        'scope': 'shared GPU, fixed-seed interleaved timing; cached prefix through native DPT; excludes first half, calibration, I/O; projection separate; no exclusive speed claim'})
    print(json.dumps({'native_costs_complete': True}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); run()
