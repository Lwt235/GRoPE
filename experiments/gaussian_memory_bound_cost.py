"""Descriptive shared-GPU costs; no exclusive-performance claim."""
import json
import os
import time

import numpy as np
import torch

from experiments.gaussian_memory_bound_data import OUT, SEEDS, load_priors, write_json, device_tree
from experiments.gaussian_memory_bound_evaluate import clip_samples, load_adapter
from experiments.gaussian_memory_view_data import load_backbone
from models.gaussian_memory_vggt_adapter import native_tail
from models.gaussian_memory_bound_read import bound_native_tail, projected_slot_weights, attention_prior


@torch.no_grad()
def run():
    backbone = load_backbone()
    adapters = {k: load_adapter(k, SEEDS[0])[0] for k in ('coverage', 'bound')}
    rows = []; torch.cuda.reset_peak_memory_stats()
    for rid in ('transfer_00', 'confirm_00', 'fresh_00'):
        samples, initial, sources = clip_samples(rid, (0,))
        sample = device_tree(samples[11, 0]); content = initial['memory']['features']
        projected = load_priors([rid])[rid, 11]
        priors = {k: attention_prior(projected, k) for k in adapters}
        calls = {'native': lambda: native_tail(backbone, sample)}
        calls.update({k: (lambda k=k: bound_native_tail(backbone, sample, adapters[k], content, priors[k])) for k in adapters})
        calls['projection_and_bound_prior'] = lambda: attention_prior(
            projected_slot_weights(initial['memory'], sample['c2w'], sample['K']), 'bound')
        timing = {}
        for key, call in calls.items():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=key != 'projection_and_bound_prior'):
                for _ in range(2): call()
                elapsed = []
                for _ in range(5):
                    torch.cuda.synchronize(); start = time.perf_counter(); call(); torch.cuda.synchronize()
                    elapsed.append(time.perf_counter() - start)
            timing[key] = {'median_seconds': float(np.median(elapsed)), 'samples_seconds': elapsed}
        rows.append({'id': rid, 'query': 11, 'condition': 0, 'timing': timing, 'sources': sources,
                     'single_window_log_prior_bytes': priors['bound'].numel() * priors['bound'].element_size()})
    result = {'rows': rows, 'device': torch.cuda.get_device_name(), 'visible_device': os.getenv('CUDA_VISIBLE_DEVICES'),
              'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
              'scope': 'shared GPU, cached prefix through native DPT; excludes first half of network, camera calibration branch and input I/O; projection separately timed; not an exclusive speed benchmark',
              'parameters': 137472, 'four_view_patch_slots': [4, 300, 257]}
    write_json(OUT / 'native_costs.json', result)
    print(json.dumps({'native_costs_complete': True}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); run()
