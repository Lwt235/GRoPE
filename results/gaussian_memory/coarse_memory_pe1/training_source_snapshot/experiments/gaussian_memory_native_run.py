"""Unmodified pretrained native attention with controlled historical inputs."""
import argparse
import copy
import json
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.gaussian_memory_native_data import (
    ROOT, OUT, CACHE, DEST, STORE, PARENT, RECENT, HISTORY, EXTRA, CONDITIONS, ARMS,
    load, read, sha256, write_json, save_tensor)
from experiments.gaussian_memory_incremental_data import sensor_target
from experiments.gaussian_memory_raw_history_run import unproject, zbuffer, merge_depth, score as support_score, history_gain


def model_load(plan):
    sys.path.insert(0, str(ROOT / 'third_party/native_history_python'))
    sys.path.insert(0, str(DEST / 'src'))
    from streamvggt.models.streamvggt import StreamVGGT
    from safetensors.torch import load_file
    model = StreamVGGT()
    if plan['model_path'].endswith('.safetensors'):
        weights = load_file(plan['model_path'])
    else:
        weights = torch.load(plan['model_path'], map_location='cpu', weights_only=True)
        if 'model' in weights: weights = weights['model']
    model.load_state_dict(weights, strict=True); del weights
    return model.eval().requires_grad_(False).cuda()


def state_new(model):
    return {'kv': [None] * model.aggregator.depth, 'cam': [None] * model.camera_head.trunk_depth,
            'ids': [], 'cam_ids': [], 'n_seen': 0}


def state_fork(state):
    # Official attention appends by concatenation and replaces list entries.
    return {k: list(v) if isinstance(v, list) else v for k, v in state.items()}


def state_bytes(state):
    return {name: sum(t.numel() * t.element_size() for pair in state[name] if pair is not None for t in pair)
            for name in ('kv', 'cam')}


def drop_early(state):
    result = state_fork(state)
    before = state_bytes(state)
    for name, identity_name in [('kv', 'ids'), ('cam', 'cam_ids')]:
        keep = [i for i, source in enumerate(state[identity_name]) if source not in EXTRA]
        result[name] = [tuple(t.index_select(2, torch.tensor(keep, device=t.device)) for t in pair)
                        for pair in state[name]]
        result[identity_name] = [state[identity_name][i] for i in keep]
        assert len(result[identity_name]) == (len(RECENT) if name == 'kv' else 4 * len(RECENT))
    assert result['ids'] == list(RECENT) and result['n_seen'] == len(HISTORY)
    return result, {'removed_frame_ids': list(EXTRA), 'aggregator_removed_entries_per_layer': 3,
                    'camera_removed_entries_per_layer': 12, 'bytes_before': before,
                    'bytes_after': state_bytes(result), 'kept_frame_ids': list(result['ids']),
                    'kept_camera_source_ids': list(result['cam_ids']), 'n_seen_preserved': result['n_seen']}


@torch.inference_mode()
def step(model, image, frame_id, state, depth_needed=False):
    images = image[None, None].cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        tokens, start, state['kv'] = model.aggregator(images, past_key_values=state['kv'],
            use_cache=True, past_frame_idx=state['n_seen'])
    with torch.autocast('cuda', enabled=False):
        poses, state['cam'] = model.camera_head(tokens, past_key_values_camera=state['cam'], use_cache=True)
        if depth_needed:
            depth, confidence = model.depth_head(tokens, images=images, patch_start_idx=start)
            depth = depth[0, 0, ..., 0].float().cpu()
        else: depth = None
    state['ids'].append(frame_id); state['cam_ids'].extend([frame_id] * 4); state['n_seen'] += 1
    for pairs, ids in [(state['kv'], state['ids']), (state['cam'], state['cam_ids'])]:
        assert all(k.shape[2] == v.shape[2] == len(ids) for k, v in pairs)
    assert all(torch.isfinite(p).all() for p in poses)
    return depth, poses[-1][0, 0].float().cpu()


def images_for(sample, condition, mask_early=False):
    images = sample['rgb_uint8'].float() / 255.
    identities = sample['frame_ids']; masks = sample['source_masks']
    fill = torch.tensor([.485, .456, .406])[:, None, None]
    mapping = {}
    for i, source in enumerate(identities):
        value = images[i]
        masked = (condition == 'persistent' and source in (4, 6, 8, 10, 11)) or (mask_early and source in EXTRA)
        mapping[source] = torch.where(masks[source][None], fill, value) if masked else value
    assert torch.equal(mapping[0], images[0])
    return mapping


def metric_depth(depth):
    return F.interpolate(depth[None, None], (210, 280), mode='bilinear', align_corners=False)[0, 0]


def scores(depth, sample):
    pred = metric_depth(depth).double(); target = sample['target'].double()
    assert torch.isfinite(depth).all() and (depth > 0).all()
    valid, hidden = sample['valid'], sample['hidden']; residual = (pred.clamp_min(1e-6) / target).log()
    def mse(region): return float(residual[valid & region].square().mean())
    selected = residual[valid]; error = pred[valid] - target[valid]
    hidden_residual = residual[valid & hidden]; hidden_error = pred[valid & hidden] - target[valid & hidden]
    return {'log_mse': float(selected.square().mean()), 'hidden_log_mse': mse(hidden), 'visible_log_mse': mse(~hidden),
            'centered_log_mse': float(selected.var(unbiased=False)), 'bias_squared': float(selected.mean().square()),
            'log_bias': float(selected.mean()), 'absrel': float((error.abs() / target[valid]).mean()),
            'hidden_centered_log_mse': float(hidden_residual.var(unbiased=False)),
            'hidden_bias_squared': float(hidden_residual.mean().square()),
            'hidden_absrel': float((hidden_error.abs() / target[valid & hidden]).mean()),
            'rmse_m': float(error.square().mean().sqrt()), 'valid_pixels': int(valid.sum()),
            'hidden_valid_pixels': int((valid & hidden).sum())}


@torch.inference_mode()
def support(meta, sample):
    row = meta['row']; cameras = torch.tensor(row['c2w_metric'], device='cuda', dtype=torch.float64)
    K = torch.tensor(row['K'], device='cuda', dtype=torch.float64)
    clouds = {}; unmasked = {}
    for source in HISTORY:
        d, valid = sensor_target(row['frames'][source]); d = d.cuda().double(); valid = valid.cuda()
        clouds[source] = unproject(d, valid, cameras[source], K[source])
        unmasked[source] = unproject(d, valid & ~sample['metric_masks'][source].cuda(), cameras[source], K[source])
    rows = []; maps = {}
    target, valid = sample['target'].numpy(), sample['valid'].numpy()
    for condition in CONDITIONS:
        common = torch.cat([unmasked[f] if condition == 'persistent' and f != 0 else clouds[f] for f in RECENT])
        recent = zbuffer(common, cameras[11], K[11])
        extra = zbuffer(torch.cat([clouds[f] for f in EXTRA]), cameras[11], K[11])
        extra_mask = zbuffer(torch.cat([unmasked[f] for f in EXTRA]), cameras[11], K[11])
        history, history_mask = merge_depth(recent, extra), merge_depth(recent, extra_mask)
        maps[condition] = {name: value.cpu() for name, value in [('recent', recent), ('extra', extra), ('history', history), ('history_mask', history_mask)]}
        for region_name, region in [('roi', sample['hidden'].numpy()), ('all', np.ones_like(valid))]:
            r, h, hm = (v.cpu().numpy() for v in (recent, history, history_mask))
            rows.append({'condition': condition, 'region': region_name,
                         'history_gain': history_gain(r, h, target, valid, region),
                         'history_mask_gain': history_gain(r, hm, target, valid, region),
                         'recent': support_score(r, target, valid, region), 'history': support_score(h, target, valid, region)})
    return rows, maps


@torch.inference_mode()
def preflight(model, meta):
    sample = load(CACHE / meta['id'] / 'inputs.pt'); images = images_for(sample, 'full')
    ids = [0, 1, 2]
    from streamvggt.utils.load_fn import load_and_preprocess_images
    official_images = load_and_preprocess_images([meta['row']['frames'][i]['rgb_path'] for i in ids])
    own_images = torch.stack([images[f] for f in ids])
    assert torch.equal(own_images, official_images), float((own_images - official_images).abs().max())
    state = state_new(model); predictions = []
    for f in ids: predictions.append(step(model, images[f], f, state, True)[0])
    frames = [{'img': images[f][None].cuda()} for f in ids]
    with torch.autocast('cuda', dtype=torch.bfloat16): official = model.inference(frames)
    replay = max(float((predictions[i] - official.ress[i]['depth'][0, ..., 0].float().cpu()).abs().max()) for i in range(3))
    assert replay < 1e-5, replay
    del official, frames
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_tokens, start = model.aggregator(own_images[None].cuda())
    with torch.autocast('cuda', enabled=False):
        batch_depth, _ = model.depth_head(batch_tokens, images=own_images[None].cuda(), patch_start_idx=start)
    batch_depth = batch_depth[0, -1, ..., 0].float().cpu()
    difference = batch_depth - predictions[-1]
    relative_rms = float(difference.square().mean().sqrt() / predictions[-1].square().mean().sqrt())
    assert relative_rms < .01, relative_rms
    dropped, drop_record = drop_early_for_preflight(model)
    write_json(OUT / 'preflight.json', {'complete': True, 'rgb_preprocessing_exact': True,
        'official_inference_depth_max_abs': replay, 'causal_batch_vs_cached_relative_rms': relative_rms,
        'causal_batch_vs_cached_max_abs': float(difference.abs().max()),
        'camera_cache_entries_per_frame': 4, 'cache_drop_synthetic_check': drop_record,
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'scope': 'Interface and resource check; not evidence of memory utility; camera batch equivalence not assumed.'})


def drop_early_for_preflight(model):
    state = {'ids': list(HISTORY), 'cam_ids': [f for f in HISTORY for _ in range(4)], 'n_seen': 8,
             'kv': [(torch.arange(8, device='cuda').reshape(1, 1, 8, 1, 1).float(), torch.zeros(1, 1, 8, 1, 1, device='cuda'))] * model.aggregator.depth,
             'cam': [(torch.arange(32, device='cuda').reshape(1, 1, 32, 1, 1).float(), torch.zeros(1, 1, 32, 1, 1, device='cuda'))] * model.camera_head.trunk_depth}
    result, record = drop_early(state)
    assert result['kv'][0][0].flatten().tolist() == [0, 4, 5, 6, 7]
    assert result['cam'][0][0].flatten().tolist() == [i for i in range(32) if i // 4 not in (1, 2, 3)]
    return result, record


@torch.inference_mode()
def run_clip(model, meta, plan):
    rid = meta['id']; marker = OUT / 'evaluation' / f'{rid}.json'
    if marker.exists():
        rec = read(marker); assert rec['complete'] and rec['plan_sha256'] == sha256(OUT / 'plan.json')
        assert sha256(rec['prediction_path']) == rec['prediction_sha256']; return rec
    sample = load(CACHE / rid / 'inputs.pt'); started = time.monotonic()
    rows, predictions, keys, cameras, cache_records = [], [], [], [], []
    for condition in CONDITIONS:
        images = images_for(sample, condition); masked = images_for(sample, condition, True)
        before_cache = {}
        for arm, sequence, mapping in [('recent', RECENT, images), ('history', HISTORY, images), ('history_mask', HISTORY, masked)]:
            state = state_new(model)
            for f in sequence: step(model, mapping[f], f, state)
            if arm == 'history': history_state = state_fork(state)
            before_cache[arm] = state_bytes(state)
            prediction, pose = step(model, mapping[11], 11, state, True)
            rows.append({'condition': condition, 'arm': arm, 'place': meta['place'], **scores(prediction, sample)})
            predictions.append(prediction); cameras.append(pose); keys.append([condition, arm])
            cache_records.append({'condition': condition, 'arm': arm, 'before_query_bytes': before_cache[arm],
                'after_query_bytes': state_bytes(state), 'source_ids_before_query': list(sequence)})
            del state
        state, removal = drop_early(history_state); del history_state
        before_cache['history_drop'] = state_bytes(state)
        prediction, pose = step(model, images[11], 11, state, True)
        rows.append({'condition': condition, 'arm': 'history_drop', 'place': meta['place'], **scores(prediction, sample)})
        predictions.append(prediction); cameras.append(pose); keys.append([condition, 'history_drop'])
        cache_records.append({'condition': condition, 'arm': 'history_drop', 'before_query_bytes': before_cache['history_drop'],
            'after_query_bytes': state_bytes(state), 'removal': removal})
        del state
    source_record = read(OUT / 'sources' / f'{rid}.json')
    assert source_record['input_sha256'] == sha256(CACHE / rid / 'inputs.pt')
    assert source_record['sha256'] == sha256(source_record['path'])
    source = load(source_record['path']); support_rows, support_maps = source['rows'], source['maps']
    path = CACHE / rid / 'predictions.pt'
    save_tensor(path, {'depth_native': torch.stack(predictions), 'depth_metric': torch.stack([metric_depth(d) for d in predictions]),
                      'camera_pose': torch.stack(cameras), 'keys': keys, 'support_maps': support_maps})
    rec = {'complete': True, 'id': rid, 'place': meta['place'], 'rows': rows, 'support': support_rows,
           'prediction_path': str(path), 'prediction_sha256': sha256(path), 'cache_records': cache_records,
           'input_sha256': sha256(CACHE / rid / 'inputs.pt'), 'plan_sha256': sha256(OUT / 'plan.json'),
           'seconds': time.monotonic() - started, 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30}
    write_json(marker, rec); print(json.dumps({'complete': rid, 'outputs': len(rows), 'seconds': rec['seconds'],
        'peak_GiB': rec['peak_allocated_GiB']}), flush=True)
    return rec


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['sources', 'preflight', 'run']); parser.add_argument('--limit', type=int)
    args = parser.parse_args(); torch.set_num_threads(4); torch.manual_seed(20261111)
    assert torch.cuda.is_available()
    if args.action == 'sources':
        for meta in read(PARENT / 'plan.json')['catalog']:
            rid = meta['id']; sample = load(CACHE / rid / 'inputs.pt'); rows, maps = support(meta, sample)
            path = CACHE / rid / 'source_support.pt'; save_tensor(path, {'rows': rows, 'maps': maps})
            write_json(OUT / 'sources' / f'{rid}.json', {'complete': True, 'path': str(path), 'sha256': sha256(path),
                'input_sha256': sha256(CACHE / rid / 'inputs.pt'), 'rows': rows})
            fraction = next(r['history_gain']['new_consistent_support_fraction'] for r in rows if r['condition'] == 'persistent' and r['region'] == 'roi')
            print(json.dumps({'sources_complete': rid, 'persistent_roi_new_consistent_fraction': fraction}), flush=True)
        return
    plan = read(OUT / 'plan.json')
    for source, digest in plan['source_sha256'].items(): assert sha256(ROOT / source) == digest
    assert sha256(plan['model_path']) == plan['model_sha256']
    model = model_load(plan); started = time.monotonic()
    if not (OUT / 'preflight.json').exists(): preflight(model, plan['catalog'][0])
    if args.action == 'preflight': return
    catalog = plan['catalog'][:args.limit] if args.limit else plan['catalog']; completed = []
    for meta in catalog:
        run_clip(model, meta, plan); completed.append(meta['id'])
        write_json(OUT / 'progress.json', {'ids': completed, 'seconds': time.monotonic() - started})
    write_json(OUT / ('pilot_worker.json' if args.limit else 'worker.json'), {'complete': True, 'ids': completed,
        'seconds': time.monotonic() - started, 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'gpu': torch.cuda.get_device_name(), 'new_training_steps': 0})


if __name__ == '__main__': main()
