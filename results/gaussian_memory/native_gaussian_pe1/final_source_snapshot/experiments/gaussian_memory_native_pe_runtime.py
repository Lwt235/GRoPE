"""Actual two-pass and direct-native query costs on fixed already-used inputs."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_native_pe_data import (
    ROOT, OUT, load, read, write_json, load_plan, sha256, sample_path, input_path, KINDS)
from experiments.gaussian_memory_native_pe_cache import patch_geometry
from experiments.gaussian_memory_native_run import model_load, state_new, state_fork, step, state_bytes
from models.gaussian_memory_native_pe import NativePEReader, native_tail, relations, LAYERS


class PrefixReady(Exception): pass


@torch.inference_mode()
def query_first(model, image, source, full=True):
    captured = {}; frame = {}; hooks = []
    for index in (4, 11, 17):
        def before(module, args, i=index): frame[i] = args[0]
        def after(module, args, output, i=index):
            x = output[0] if isinstance(output, tuple) else output
            captured[i] = torch.cat((frame[i], x), -1)[:, None].float()
        hooks.append(model.aggregator.global_blocks[index].register_forward_pre_hook(before))
        hooks.append(model.aggregator.global_blocks[index].register_forward_hook(after))
    def prefix(module, args):
        captured['prefix'] = args[0]
        if not full: raise PrefixReady()
    hooks.append(model.aggregator.global_blocks[20].register_forward_pre_hook(prefix))
    state = state_fork(source); begin = time.monotonic()
    try:
        if full:
            depth, pose = step(model, image, 11, state, True)
        else:
            try:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    model.aggregator(image[None, None].cuda(), past_key_values=state['kv'], use_cache=True, past_frame_idx=source['n_seen'])
            except PrefixReady: pass
            depth = pose = None
    finally:
        for hook in hooks: hook.remove()
    torch.cuda.synchronize()
    return captured, depth, pose, time.monotonic() - begin


@torch.inference_mode()
def main():
    torch.set_num_threads(4); plan = load_plan()
    checkpoints = {kind: sha256(OUT / f'{kind}_{plan["seeds"][0]}/step_{plan["steps"]}.pt') for kind in KINDS}
    if (OUT / 'runtime.json').exists():
        saved = read(OUT / 'runtime.json')
        assert saved['complete'] and saved['source_sha256'] == sha256(ROOT / 'experiments/gaussian_memory_native_pe_runtime.py')
        assert saved['checkpoint_sha256'] == checkpoints and saved['plan_sha256'] == sha256(OUT / 'plan.json')
        print(json.dumps({'reused_completed_runtime': True}), flush=True); return
    model = model_load(plan); adapters = {}
    for kind in KINDS:
        adapter = NativePEReader().cuda().eval()
        adapter.load_state_dict(load(OUT / f'{kind}_{plan["seeds"][0]}/step_{plan["steps"]}.pt')['model'])
        adapters[kind] = adapter
    rows = []; resource = []; direct_diffs = []
    for rid in plan['training_ids'][:2]:
        for condition in plan['conditions']:
            cpu = load(sample_path(rid, 11, condition)); inp = load(input_path(rid, 11, condition))
            source = state_new(model); torch.cuda.synchronize(); begin = time.monotonic()
            for j, f in enumerate(cpu['source_ids']): step(model, inp['images'][j], f, source, True)
            torch.cuda.synchronize(); cold_seconds = time.monotonic() - begin
            # Alias native source K/V; preserve extra normalized source inputs for LoRA.
            memory = [{'k': source['kv'][i][0].flatten(2, 3), 'v': source['kv'][i][1].flatten(2, 3),
                       'norm': cpu['memory'][j]['norm'].cuda()} for j, i in enumerate(LAYERS)]
            norm_bytes = sum(m['norm'].numel() * m['norm'].element_size() for m in memory)
            source_geometry_bytes = len(cpu['source_ids']) * cpu['tokens_per_frame'] * 12 * 4
            resource.append({'id': rid, 'condition': condition, 'cold_history_seconds': cold_seconds,
                             'source_cache_bytes': state_bytes(source), 'extra_source_norm_bytes': norm_bytes,
                             'source_geometry_bytes': source_geometry_bytes})
            sample_base = {k: cpu[k] for k in ('native_hw', 'tokens_per_frame')}; sample_base['memory'] = memory
            current_image = inp['images'][-1]
            # Warm the full query and decoder once before timed repeated paths.
            capture, depth, pose, _ = query_first(model, current_image, source)
            g = patch_geometry(list(cpu['geometry_depths'][:-1]) + [depth], list(cpu['geometry_poses'][0, :-1]) + [pose], cpu['native_hw'])
            sample = dict(sample_base, prefix=capture['prefix'], early={i: capture[i] for i in (4, 11, 17)},
                          means=g['means'].cuda(), covariances=g['covariances'].cuda())
            for kind in KINDS:
                z = relations(sample, kind)
                with torch.autocast('cuda', dtype=torch.bfloat16): native_tail(model, sample, adapters[kind], kind, z, False)
            del sample, capture, g, z
            for repeat in range(3):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                capture, depth, pose, first_seconds = query_first(model, current_image, source)
                rows.append({'id': rid, 'condition': condition, 'kind': 'official', 'repeat': repeat,
                             'query_total_seconds': first_seconds, 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30})
                begin = time.monotonic()
                g = patch_geometry(list(cpu['geometry_depths'][:-1]) + [depth], list(cpu['geometry_poses'][0, :-1]) + [pose], cpu['native_hw'])
                sample = dict(sample_base, prefix=capture['prefix'], early={i: capture[i] for i in (4, 11, 17)},
                              means=g['means'].cuda(), covariances=g['covariances'].cuda())
                torch.cuda.synchronize(); geometry_seconds = time.monotonic() - begin
                for kind in KINDS:
                    torch.cuda.reset_peak_memory_stats(); begin = time.monotonic(); z = relations(sample, kind)
                    torch.cuda.synchronize(); relation_seconds = time.monotonic() - begin; begin = time.monotonic()
                    with torch.autocast('cuda', dtype=torch.bfloat16): result = native_tail(model, sample, adapters[kind], kind, z, False)
                    torch.cuda.synchronize(); tail_seconds = time.monotonic() - begin
                    rows.append({'id': rid, 'condition': condition, 'kind': kind, 'repeat': repeat,
                                 'first_seconds': first_seconds, 'geometry_seconds': geometry_seconds,
                                 'relations_seconds': relation_seconds, 'tail_seconds': tail_seconds,
                                 'query_total_seconds': first_seconds + geometry_seconds + relation_seconds + tail_seconds,
                                 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30})
                    if kind == 'native': native_two_pass = result
                # A needs no current depth/camera geometry: measure its cheaper one-pass realization.
                torch.cuda.reset_peak_memory_stats(); capture_a, _, _, prefix_seconds = query_first(model, current_image, source, False)
                s_a = dict(sample, prefix=capture_a['prefix'], early={i: capture_a[i] for i in (4, 11, 17)})
                begin = time.monotonic(); z_a = relations(s_a, 'native'); torch.cuda.synchronize()
                relation_seconds = time.monotonic() - begin; begin = time.monotonic()
                with torch.autocast('cuda', dtype=torch.bfloat16): direct = native_tail(model, s_a, adapters['native'], 'native', z_a, False)
                torch.cuda.synchronize(); tail_seconds = time.monotonic() - begin
                difference = float((direct - native_two_pass).abs().max()); direct_diffs.append(difference)
                assert difference < 1e-5 * max(1., float(direct.mean())), difference
                rows.append({'id': rid, 'condition': condition, 'kind': 'native_direct', 'repeat': repeat,
                             'prefix_seconds': prefix_seconds, 'relations_seconds': relation_seconds, 'tail_seconds': tail_seconds,
                             'query_total_seconds': prefix_seconds + relation_seconds + tail_seconds,
                             'peak_GiB': torch.cuda.max_memory_allocated() / 2**30})
                del capture, capture_a, sample, s_a, g, z, z_a, result, direct, native_two_pass
            del source, memory; torch.cuda.empty_cache()
    summary = {}
    for kind in ('official', *KINDS, 'native_direct'):
        selected = [r for r in rows if r['kind'] == kind]
        summary[kind] = {'mean_query_seconds': float(np.mean([r['query_total_seconds'] for r in selected])),
                         'median_query_seconds': float(np.median([r['query_total_seconds'] for r in selected])),
                         'peak_GiB': max(r['peak_GiB'] for r in selected)}
    write_json(OUT / 'runtime.json', {'complete': True, 'summary': summary, 'rows': rows, 'resources': resource,
        'checkpoint_sha256': checkpoints, 'plan_sha256': sha256(OUT / 'plan.json'),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'repeats': 3, 'cases': 'first two fixed training clips, full and persistent, query11',
        'source_features_gpu_resident': True, 'disk_io_and_model_load_excluded': True,
        'current_geometry_recomputes_cached_source_moments': True,
        'native_direct_vs_matched_two_pass_max_abs': max(direct_diffs),
        'note': 'A can skip query geometry; report its direct one-pass cost as the cheaper fair reference. Cold source construction includes native source depth/camera heads. No compact-storage claim.',
        'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_native_pe_runtime.py')})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
