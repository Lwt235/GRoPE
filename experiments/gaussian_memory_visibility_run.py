"""Matched visibility interventions, prefix memories, and frozen native inference."""
import argparse
import json
import math
import time

import numpy as np
import torch

from experiments.gaussian_memory_visibility_data import (
    ROOT, DATA, OUT, CACHE, EARLY, LATE, CURRENT, CONDITIONS, POLICIES, FAMILIES,
    SEEDS, load_plan, checked_base, reader_kind, sha256, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_incremental_data import (
    load_rgb, encode_window, observations, build_memory, sensor_target)
from experiments.gaussian_memory_view_data import load_backbone, rectangle
from experiments.gaussian_memory_raw_history_run import unproject, zbuffer, merge_depth, history_gain, score as support_score
from experiments.gaussian_memory_incremental_evaluate import score
from experiments.gaussian_memory_bound_evaluate import predict, load_adapter as load_local
from experiments.gaussian_memory_history_evaluate import load_adapter
from experiments.gaussian_memory_state_data import memory_prior
from models.gaussian_memory_incremental import concatenate_observations
from models.gaussian_memory_view_token import apply_rgb_mask


def pixel_mask(patches):
    return patches.reshape(15, 20).repeat_interleave(14, 0).repeat_interleave(14, 1)


def project_indices(points, camera, K):
    local = (points - camera[:3, 3]) @ camera[:3, :3]
    uvw = local @ K.T
    uv = uvw[:, :2] / uvw[:, 2:].clamp_min(1e-12)
    inside = (torch.isfinite(local).all(-1) & (local[:, 2] > 1e-4)
              & (uv[:, 0] >= 0) & (uv[:, 0] < 280) & (uv[:, 1] >= 0) & (uv[:, 1] < 210))
    return inside, uv[inside].floor().long()


def masks_and_information(meta):
    """Masks depend only on early source depths, camera poses, and the fixed ROI."""
    row = meta['row']; rid = meta['id']; device = 'cuda'
    K = torch.tensor(row['K'], device=device, dtype=torch.float64)
    cameras = torch.tensor(row['c2w_metric'], device=device, dtype=torch.float64)
    roi = pixel_mask(rectangle(rid, 11, 0).to(device))
    clouds = {}; valid_masks = {}; depths = {}
    # Only early depth is read before constructing the masks.
    for f in EARLY:
        d, valid = sensor_target(row['frames'][f]); d = d.to(device=device, dtype=torch.float64); valid = valid.to(device)
        depths[f], valid_masks[f] = d, valid
        clouds[f] = unproject(d, valid, cameras[f], K[f])
    early = torch.cat([clouds[f] for f in EARLY])
    inside, uv = project_indices(early, cameras[11], K[11])
    eligible = torch.where(inside)[0][roi[uv[:, 1], uv[:, 0]]]
    selected = early[eligible]
    masks = torch.zeros(12, 300, device=device, dtype=torch.bool)
    for f in EARLY + LATE:
        _, uv = project_indices(selected, cameras[f], K[f])
        masks[f, (uv[:, 1] // 14) * 20 + uv[:, 0] // 14] = True
    masks[11] = rectangle(rid, 11, 0).to(device)
    # Subsequent target labels only measure the task construction, never select it.
    recent = []; unmasked_recent = []
    for f in LATE:
        d, valid = sensor_target(row['frames'][f]); d = d.to(device=device, dtype=torch.float64); valid = valid.to(device)
        recent.append(unproject(d, valid, cameras[f], K[f]))
        unmasked_recent.append(unproject(d, valid & ~pixel_mask(masks[f]), cameras[f], K[f]))
    ep = zbuffer(early, cameras[11], K[11])
    rp = zbuffer(torch.cat(recent), cameras[11], K[11])
    mp = zbuffer(torch.cat(unmasked_recent), cameras[11], K[11])
    maps = {'early': ep, 'recent_full': rp, 'recent_persistent': mp,
            'history_full': merge_depth(ep, rp), 'history_persistent': merge_depth(ep, mp)}
    target, valid = sensor_target(row['frames'][11]); roi_cpu = roi.cpu().numpy()
    target, valid = target.numpy(), valid.numpy()
    rows = []
    for region_name, region in [('roi', roi_cpu), ('all', np.ones_like(roi_cpu))]:
        for name, d in maps.items():
            rows.append({'map': name, 'region': region_name, **support_score(d.cpu().numpy(), target, valid, region)})
        for cond, r, h in [('full', rp, maps['history_full']), ('persistent', mp, maps['history_persistent'])]:
            rows.append({'map': 'gain_' + cond, 'region': region_name,
                         **history_gain(r.cpu().numpy(), h.cpu().numpy(), target, valid, region)})
    return masks, maps, {'id': rid, 'place': meta['place'], 'selected_early_points': len(selected),
        'mask_patches_per_frame': masks.sum(-1).cpu().tolist(), 'rows': rows,
        'source_selection_uses_query_depth': False, 'visibility_pruning': False}


@torch.no_grad()
def build_clip(meta, backbone, pca, plan):
    rid = meta['id']; folder = CACHE / rid; folder.mkdir(parents=True, exist_ok=True)
    marker = OUT / 'inputs' / f'{rid}.json'
    if marker.exists():
        rec = json.loads(marker.read_text()); assert rec['complete'] and rec['plan_sha256'] == sha256(OUT / 'plan.json')
        for name, digest in rec['files'].items(): assert sha256(folder / name) == digest
        return rec
    start = time.monotonic(); masks, maps, diagnostic = masks_and_information(meta)
    images = load_rgb(meta['row']); K = torch.tensor(meta['row']['K'], device='cuda', dtype=torch.float32)
    cameras = torch.tensor(meta['row']['c2w_metric'], device='cuda', dtype=torch.float32)
    source = plan['inputs'][rid]; initial = device_tree(checked_base(rid, 'initial.pt', plan))
    old_updates = device_tree(checked_base(rid, 'updates.pt', plan))
    masked_images = apply_rgb_mask(images, masks)
    windows = []; new_updates = {}; replay_checks = []; inferences = 0
    for f in LATE:
        spec = source['write_scales'][str(f)]; ids = spec['window']
        assert ids[-1] == f and all(i in EARLY + LATE and i <= f for i in ids)
        encoded = encode_window(backbone, masked_images[ids], pca); inferences += 1
        new_updates[f] = observations(encoded, ids, K, cameras, spec['scale'], source['unit_meters'], True)
        windows.append({'kind': 'write', 'frames': ids, 'scale': spec['scale'],
                        'mask_patch_counts': masks[ids].sum(-1).tolist(), 'all_context_copies_masked': True})
        del encoded
    states = {}
    for condition, updates in [('full', old_updates), ('persistent', new_updates)]:
        late = [updates[f] for f in LATE]
        assert all(o is not None for o in late)
        for policy, obs in [('recent', late), ('history', [initial['observation']] + late)]:
            state = build_memory(concatenate_observations(obs))
            used = state['sources'].any(0).nonzero().flatten().tolist()
            assert set(used).issubset(set(EARLY + LATE if policy == 'history' else LATE))
            assert state['features'].shape == (256, 32)
            assert all(torch.isfinite(state[k]).all() for k in ['means', 'covars', 'features'])
            states[f'{condition}/{policy}'] = state
    samples = {}
    for c in (0, 1):
        sample = checked_base(rid, f'q11_c{c}.pt', plan)
        assert sample['input_frames'] == list(CURRENT)
        sample = dict(sample, condition=c, hidden=pixel_mask(masks[11]).cpu(), metric_scale=source['query_scale_full'])
        sample['visibility_condition'] = CONDITIONS[c]
        samples[c] = sample
    encoded = encode_window(backbone, masked_images[list(CURRENT)], pca); inferences += 1
    sample = {k: encoded[k] for k in ('prefix', 'early4', 'local_content', 'query_depth_native')}
    sample.update({k: samples[0][k] for k in ('id', 'query', 'input_frames', 'c2w', 'K', 'target', 'valid', 'hidden')})
    sample.update(condition=2, visibility_condition='persistent', metric_scale=source['query_scale_full'],
                  calibration_valid=True, rgb=(masked_images[11].permute(1, 2, 0) * 255).round().byte())
    samples[2] = sample; del encoded
    windows.append({'kind': 'query', 'frames': list(CURRENT), 'scale': source['query_scale_full'],
                    'mask_patch_counts': masks[list(CURRENT)].sum(-1).tolist(), 'all_context_copies_masked': True})
    # Fixed first clip verifies that reused caches encode the exact same input rule.
    if rid == 'fidelity_00':
        for c in (0, 1):
            inp = images[list(CURRENT)].clone()
            if c == 1: inp[-1:] = apply_rgb_mask(inp[-1:], masks[11:12])
            replay = encode_window(backbone, inp, pca); inferences += 1
            diffs = {}
            for key in ('prefix', 'early4', 'local_content', 'query_depth_native'):
                difference = float((replay[key].float().cpu() - samples[c][key].float().cpu()).abs().max())
                torch.testing.assert_close(replay[key].cpu(), samples[c][key], rtol=1e-4, atol=2e-5)
                diffs[key] = difference
            replay_checks.append({'condition': c, 'max_absolute_difference': diffs})
            del replay
    save_tensor(folder / 'states.pt', states)
    save_tensor(folder / 'masks.pt', {'patch_masks': masks, 'sensor_support_maps': maps})
    save_tensor(folder / 'observations_persistent.pt', new_updates)
    for c in range(3): save_tensor(folder / f'q11_c{c}.pt', samples[c])
    files = {p.name: sha256(p) for p in folder.glob('*.pt')}
    record = {'complete': True, 'id': rid, 'place': meta['place'], 'files': files,
        'plan_sha256': sha256(OUT / 'plan.json'), 'base_identity_sha256': source['base_identity_sha256'],
        'metric_scale_shared': source['query_scale_full'], 'windows': windows,
        'diagnostic': diagnostic, 'replay_checks': replay_checks, 'new_backbone_inferences': inferences,
        'seconds': time.monotonic() - start, 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'early_memory_unmasked': True, 'sensor_geometry_in_candidate_states': False,
        'query_frame_never_written': True, 'new_training_steps': 0, 'new_download_bytes': 0}
    write_json(marker, record)
    print(json.dumps({'inputs_complete': rid, 'new_backbone_inferences': inferences,
                      'seconds': record['seconds'], 'mask_patches': diagnostic['mask_patches_per_frame']}), flush=True)
    return record


@torch.no_grad()
def evaluate_clip(meta, backbone, adapters, plan):
    rid = meta['id']; folder = CACHE / rid; start = time.monotonic()
    input_record = json.loads((OUT / 'inputs' / f'{rid}.json').read_text())
    marker = OUT / 'evaluation' / f'{rid}.json'
    if marker.exists():
        rec = json.loads(marker.read_text()); assert rec['complete'] and rec['plan_sha256'] == sha256(OUT / 'plan.json')
        assert sha256(rec['prediction_path']) == rec['prediction_sha256']
        return rec
    samples = {c: torch.load(folder / f'q11_c{c}.pt', map_location='cpu', weights_only=False) for c in range(3)}
    states = device_tree(torch.load(folder / 'states.pt', map_location='cpu', weights_only=False))
    priors = {(condition, policy, kind): memory_prior(states[f'{condition}/{policy}'], samples[0], kind, plan['center_variance'])
              for condition in ('full', 'persistent') for policy in POLICIES for kind in ('content', 'soft', 'soft_center')}
    rows = []; predictions = []
    def add(pred, sample, family, seed=None, policy=None):
        r = score(pred, sample)
        r.update(family=family, seed=seed, policy=policy, place=meta['place'], visibility_condition=CONDITIONS[sample['condition']])
        rows.append(r); predictions.append(pred.float().cpu())
    for c, sample in samples.items():
        add(sample['query_depth_native'], sample, 'native')
        uniform = priors['full', 'recent', 'content']
        for seed in SEEDS:
            prediction = predict(backbone, adapters[f'local_{seed}'], sample, sample['local_content'].cuda(), uniform)
            add(prediction, sample, 'local', seed)
        condition = 'persistent' if c == 2 else 'full'
        for policy in POLICIES:
            memory = states[f'{condition}/{policy}']
            for family in FAMILIES:
                prior = priors[condition, policy, reader_kind(family)]
                for seed in SEEDS:
                    prediction = predict(backbone, adapters[f'{family}_{seed}'], sample, memory['features'], prior)
                    add(prediction, sample, family, seed, policy)
    assert len(rows) == 102
    target = folder / 'predictions.pt'; save_tensor(target, {'prediction_native': torch.stack(predictions),
        'keys': [[r['condition'], r['family'], r['seed'], r['policy']] for r in rows]})
    rec = {'complete': True, 'id': rid, 'place': meta['place'], 'rows': rows,
        'prediction_path': str(target), 'prediction_sha256': sha256(target),
        'plan_sha256': sha256(OUT / 'plan.json'), 'input_record_sha256': sha256(OUT / 'inputs' / f'{rid}.json'),
        'checkpoint_identities': plan['checkpoints'], 'seconds': time.monotonic() - start,
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30}
    write_json(marker, rec)
    print(json.dumps({'predictions_complete': rid, 'predictions': len(rows), 'seconds': rec['seconds']}), flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(); p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    args = p.parse_args(); torch.set_num_threads(4); torch.manual_seed(SEEDS[0])
    assert torch.cuda.is_available()
    plan = load_plan()
    for name, digest in plan['source_sha256'].items(): assert sha256(ROOT / name) == digest, name
    backbone = load_backbone(); pca = torch.load(DATA / 'cache_v1/pca.pt', map_location='cuda', weights_only=False)
    adapters = {}
    for family in list(FAMILIES) + ['local']:
        for seed in SEEDS:
            adapter, digest = (load_local('local', seed) if family == 'local' else load_adapter(family, seed))
            assert digest == plan['checkpoints'][f'{family}_{seed}']['sha256']
            adapters[f'{family}_{seed}'] = adapter
    start = time.monotonic(); records = []
    for meta in plan['catalog'][args.shard::args.shards]:
        build_clip(meta, backbone, pca, plan)
        rec = evaluate_clip(meta, backbone, adapters, plan); records.append(meta['id'])
        write_json(OUT / f'progress_shard_{args.shard}.json', {'complete_clips': records, 'seconds': time.monotonic() - start})
    write_json(OUT / f'worker_{args.shard}_complete.json', {'complete': True, 'clips': records,
        'seconds': time.monotonic() - start, 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'new_training_steps': 0, 'new_download_bytes': 0})


if __name__ == '__main__': main()
