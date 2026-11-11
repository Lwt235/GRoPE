"""Descriptive native-read cost and one fixed attention-path diagnostic."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_incremental_data import OUT, CACHE, SEEDS, device_tree, write_json
from experiments.gaussian_memory_incremental_train import inputs_for
from experiments.gaussian_memory_incremental_evaluate import load_adapter
from experiments.gaussian_memory_view_data import load_backbone
from models.gaussian_memory_vggt_adapter import native_tail, position_grid, padded_geometry


@torch.no_grad()
def run():
    torch.set_num_threads(4); backbone = load_backbone()
    adapter, statistics, _ = load_adapter('combined', SEEDS[0]); rows = []
    for rid in ('transfer_00', 'confirm_00', 'fresh_00'):
        s = device_tree(torch.load(CACHE / rid / 'q11_c0.pt', map_location='cpu', weights_only=False))
        initial = device_tree(torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False))
        pe, content = inputs_for(s, initial, 'combined', statistics)
        durations = {}; outputs = {}
        for name, use in [('native', False), ('combined', True)]:
            times = []
            torch.cuda.reset_peak_memory_stats()
            for rep in range(12):
                torch.cuda.synchronize(); start = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result, _ = native_tail(backbone, s, adapter if use else None, pe if use else None, content if use else None)
                torch.cuda.synchronize()
                if rep >= 2: times.append(time.perf_counter() - start)
            durations[name] = {'median_seconds': float(np.median(times)), 'min_seconds': min(times),
                               'max_seconds': max(times), 'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3}
            outputs[name] = result
        torch.cuda.synchronize(); start = time.perf_counter()
        for _ in range(20): padded_geometry(initial['memory'], s['c2w'], s['K'])
        torch.cuda.synchronize(); projection_s = (time.perf_counter() - start) / 20
        block = backbone.aggregator.global_blocks[11]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = block.norm1(s['prefix'].reshape(1, -1, 1024)); attention = block.attn
            qkv = attention.qkv(x).reshape(1, 1220, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0); q, k = attention.q_norm(q), attention.k_norm(k)
            pos = position_grid(x.device).reshape(1, -1, 2)
            q, k = attention.rope(q, pos), attention.rope(k, pos)
            dq = adapter.pe_q(pe.reshape(1, -1, 10)).reshape(1, 1220, attention.num_heads, attention.head_dim).transpose(1, 2)
            dk = adapter.pe_k(pe.reshape(1, -1, 10)).reshape(1, 1220, attention.num_heads, attention.head_dim).transpose(1, 2)
            indices = torch.arange(920, 1220, 30, device=x.device)
            before = (q[:, :, indices].float() @ k.float().transpose(-1, -2)) / attention.head_dim**.5
            after = ((q + dq)[:, :, indices].float() @ (k + dk).float().transpose(-1, -2)) / attention.head_dim**.5
        rows.append({'id': rid, 'timing': durations, 'projection_four_views_s': projection_s,
                     'native_output_mean_abs_change': float((outputs['combined'] - outputs['native']).abs().mean()),
                     'q_delta_rms': float(dq.float().square().mean().sqrt()),
                     'k_delta_rms': float(dk.float().square().mean().sqrt()),
                     'ten_fixed_query_logits_mean_abs_change': float((after - before).abs().mean()),
                     'ten_fixed_query_attention_mean_abs_change': float((after.softmax(-1) - before.softmax(-1)).abs().mean())})
    write_json(OUT / 'native_costs.json', {'complete': True, 'rows': rows,
               'shared_gpu': True, 'exclusive_benchmark': False, 'cache_io_excluded': True,
               'scope': 'resume from cached frame11 through native DPT, not full online pipeline or training throughput',
               'seed': SEEDS[0], 'warmup': 2, 'timed_repetitions': 10})
    print(json.dumps({'native_costs_complete': True, 'rows': rows}), flush=True)


if __name__ == '__main__': run()
