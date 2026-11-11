"""Per-query causal visibility interventions on the original training places."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_visibility_training_data import (
    ROOT, DATA, OUT, CACHE, EARLY, LATE, QUERIES, CONDITIONS, POLICIES, SEEDS,
    load_plan, verify_sources, checked_base, sha256, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_incremental_data import (
    load_rgb, encode_window, sensor_target, observations, build_memory)
from experiments.gaussian_memory_view_data import load_backbone, rectangle
from experiments.gaussian_memory_visibility_run import pixel_mask, project_indices
from experiments.gaussian_memory_raw_history_run import unproject
from models.gaussian_memory_incremental import concatenate_observations
from models.gaussian_memory_view_token import apply_rgb_mask


def masks_for_query(meta, query):
    row = meta['row']; rid = meta['id']
    K = torch.tensor(row['K'], device='cuda', dtype=torch.float64)
    cameras = torch.tensor(row['c2w_metric'], device='cuda', dtype=torch.float64)
    roi = pixel_mask(rectangle(rid, query, 0).cuda())
    clouds = []
    for f in EARLY:
        depth, valid = sensor_target(row['frames'][f])
        clouds.append(unproject(depth.to(device='cuda', dtype=torch.float64), valid.cuda(), cameras[f], K[f]))
    early = torch.cat(clouds)
    inside, uv = project_indices(early, cameras[query], K[query])
    selected = early[torch.where(inside)[0][roi[uv[:, 1], uv[:, 0]]]]
    masks = torch.zeros(12, 300, device='cuda', dtype=torch.bool)
    for f in list(EARLY) + [f for f in LATE if f < query]:
        _, uv = project_indices(selected, cameras[f], K[f])
        masks[f, uv[:, 1] // 14 * 20 + uv[:, 0] // 14] = True
    masks[query] = rectangle(rid, query, 0).cuda()
    return masks, len(selected)


@torch.no_grad()
def build_clip(meta, backbone, pca, plan):
    rid = meta['id']; folder = CACHE / rid; folder.mkdir(parents=True, exist_ok=True)
    marker = OUT / 'inputs' / f'{rid}.json'
    if marker.exists():
        rec = json.loads(marker.read_text()); assert rec['complete'] and rec['plan_sha256'] == sha256(OUT / 'plan.json')
        for n, h in rec['files'].items(): assert sha256(folder / n) == h
        return rec
    start = time.monotonic(); source = plan['inputs'][rid]
    images = load_rgb(meta['row'])
    K = torch.tensor(meta['row']['K'], device='cuda', dtype=torch.float32)
    cameras = torch.tensor(meta['row']['c2w_metric'], device='cuda', dtype=torch.float32)
    initial = device_tree(checked_base(rid, 'initial.pt', plan))
    old_updates = device_tree(checked_base(rid, 'updates.pt', plan))
    records = []; windows = []; replays = []; inferences = 0
    for q in QUERIES:
        late = [f for f in LATE if f < q]
        masks, selected_count = masks_for_query(meta, q)
        masked = apply_rgb_mask(images, masks); new_updates = {}
        for f in late:
            spec = source['write_scales'][str(f)]; ids = spec['window']
            assert ids[-1] == f and all(i <= f and i in list(EARLY) + late for i in ids)
            enc = encode_window(backbone, masked[ids], pca); inferences += 1
            new_updates[f] = observations(enc, ids, K, cameras, spec['scale'], source['unit_meters'], True)
            windows.append({'query': q, 'kind': 'write', 'frames': ids, 'scale': spec['scale'],
                            'mask_patch_counts': masks[ids].sum(-1).tolist()})
            del enc
        states = {}; state_sources = {}
        for condition, updates in [('full', old_updates), ('persistent', new_updates)]:
            obs = [updates[f] for f in late]; assert all(o is not None for o in obs)
            for policy, parts in [('recent', obs), ('history', [initial['observation']] + obs)]:
                m = build_memory(concatenate_observations(parts))
                used = m['sources'].any(0).nonzero().flatten().tolist()
                assert set(used).issubset(set(list(EARLY) + late if policy == 'history' else late))
                assert all(f < q for f in used) and m['features'].shape == (256, 32)
                assert all(torch.isfinite(m[k]).all() for k in ('means', 'covars', 'features'))
                key = condition + '/' + policy; states[key] = m; state_sources[key] = used
        spec = source['queries'][str(q)]; ids = spec['window']; samples = {}
        for c in (0, 1):
            sample = checked_base(rid, f'q{q}_c{c}.pt', plan)
            assert sample['input_frames'] == ids
            sample = dict(sample, hidden=pixel_mask(masks[q]).cpu(), metric_scale=spec['scale'],
                          visibility_condition=CONDITIONS[c])
            samples[c] = sample
        enc = encode_window(backbone, masked[ids], pca); inferences += 1
        sample = {k: enc[k] for k in ('prefix', 'early4', 'local_content', 'query_depth_native')}
        sample.update({k: samples[0][k] for k in ('id', 'query', 'input_frames', 'c2w', 'K', 'target', 'valid', 'hidden')})
        sample.update(condition=2, visibility_condition='persistent', metric_scale=spec['scale'],
                      calibration_valid=True, rgb=(masked[q].permute(1, 2, 0) * 255).round().byte())
        samples[2] = sample; del enc
        windows.append({'query': q, 'kind': 'query', 'frames': ids, 'scale': spec['scale'],
                        'mask_patch_counts': masks[ids].sum(-1).tolist()})
        if rid == plan['training_ids'][0] and q in (5, 11):
            for c in (0, 1):
                inp = images[ids].clone()
                if c == 1: inp[-1:] = apply_rgb_mask(inp[-1:], masks[q:q + 1])
                replay = encode_window(backbone, inp, pca); inferences += 1; diffs = {}
                for key in ('prefix', 'early4', 'local_content', 'query_depth_native'):
                    torch.testing.assert_close(replay[key].cpu(), samples[c][key], rtol=1e-4, atol=2e-5)
                    diffs[key] = float((replay[key].float().cpu() - samples[c][key].float().cpu()).abs().max())
                replays.append({'query': q, 'condition': c, 'max_abs': diffs}); del replay
        save_tensor(folder / f'q{q}_states.pt', states)
        save_tensor(folder / f'q{q}_masks.pt', {'patch_masks': masks, 'selected_early_points': selected_count})
        save_tensor(folder / f'q{q}_observations_persistent.pt', new_updates)
        for c in range(3): save_tensor(folder / f'q{q}_c{c}.pt', samples[c])
        records.append({'query': q, 'selected_early_points': selected_count,
                        'mask_patches': masks.sum(-1).tolist(), 'state_sources': state_sources})
    rec = {'complete': True, 'id': rid, 'place': meta['place'],
        'plan_sha256': sha256(OUT / 'plan.json'), 'base_identity_sha256': source['base_identity_sha256'],
        'files': {p.name: sha256(p) for p in folder.glob('*.pt')}, 'queries': records,
        'windows': windows, 'replays': replays, 'new_backbone_inferences': inferences,
        'seconds': time.monotonic() - start, 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'all_context_copies_masked': True, 'early_memory_unmasked': True,
        'query_frames_never_written': True, 'sensor_in_state': False}
    write_json(marker, rec)
    print(json.dumps({'complete': rid, 'windows': inferences, 'seconds': rec['seconds']}), flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(); p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    args = p.parse_args(); torch.set_num_threads(4); torch.manual_seed(SEEDS[0])
    plan = load_plan(); verify_sources(plan); assert torch.cuda.is_available()
    backbone = load_backbone(); pca = torch.load(DATA / 'cache_v1/pca.pt', map_location='cuda', weights_only=False)
    start = time.monotonic(); records = []
    for meta in plan['training_catalog'][args.shard::args.shards]:
        records.append(build_clip(meta, backbone, pca, plan)['id'])
    write_json(OUT / f'extract_worker_{args.shard}.json', {'complete': True, 'ids': records,
        'seconds': time.monotonic() - start, 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30})


if __name__ == '__main__': main()
