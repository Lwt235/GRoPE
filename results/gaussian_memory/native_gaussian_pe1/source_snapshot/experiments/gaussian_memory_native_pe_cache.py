"""Frozen causal first pass; historical contents and predicted geometry shared by arms."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_native_pe_data import (
    ROOT, OUT, CACHE, load, read, write_json, save_tensor, sha256, load_plan, verify,
    examples, input_path, sample_path, sample_key)
from experiments.gaussian_memory_native_run import model_load, state_new
from models.gaussian_memory_native_pe import LAYERS, START


@torch.inference_mode()
def patch_geometry(depths, poses, hw):
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    depth = torch.stack(depths).cuda().double()
    pose = torch.stack(poses).cuda()[None]
    ext, intr = pose_encoding_to_extri_intri(pose, hw)
    ext, intr = ext[0].double(), intr[0].double()
    h, w = hw
    y, x = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
    camera = torch.stack(((x[None] - intr[:, 0, 2, None, None]) / intr[:, 0, 0, None, None] * depth,
                          (y[None] - intr[:, 1, 2, None, None]) / intr[:, 1, 1, None, None] * depth, depth), -1)
    points = torch.einsum('shwi,sij->shwj', camera - ext[:, None, None, :3, 3], ext[:, :3, :3])
    unit = depth[0].median(); origin = points[0].mean((0, 1))
    points = (points - origin) / unit
    patches = points.reshape(len(depths), h // 14, 14, w // 14, 14, 3).permute(0, 1, 3, 2, 4, 5).reshape(len(depths), -1, 196, 3)
    means = patches.mean(-2); delta = patches - means[..., None, :]
    cov = delta.transpose(-1, -2) @ delta / 196 + 1e-6 * torch.eye(3, device='cuda')
    mean_pad = torch.cat((means.new_zeros(len(depths), START, 3), means), 1)
    cov_pad = torch.cat((torch.eye(3, device='cuda').expand(len(depths), START, 3, 3), cov), 1)
    eigen = torch.linalg.eigvalsh(cov[-1]); radius = cov[-1].diagonal(dim1=-2, dim2=-1).sum(-1).sqrt()
    nearest = torch.cdist(means[-1].float(), means[:-1].flatten(0, 1).float()).min(-1).values
    stats = {'extent_spread': float(radius.log().std(unbiased=False)),
             'anisotropy': float((eigen[:, -1] / eigen[:, 0]).median()),
             'overlap': float((nearest < .1).float().mean())}
    assert all(torch.isfinite(t).all() for t in (means, cov)) and unit > 0
    return {'means': mean_pad.flatten(0, 1).float().cpu(), 'covariances': cov_pad.flatten(0, 1).float().cpu(),
            'geometry_statistics': stats, 'geometry_unit': float(unit), 'geometry_origin': origin.cpu(),
            'predicted_extrinsics': ext.cpu(), 'predicted_intrinsics': intr.cpu(),
            'geometry_depths': depth.float().cpu(), 'geometry_poses': pose.cpu()}


@torch.inference_mode()
def build_one(model, key, plan):
    rid, q, condition = key; name = sample_key(*key); marker = OUT / 'cache' / (name.replace('/', '_') + '.json')
    if marker.exists():
        record = read(marker); assert record['plan_sha256'] == sha256(OUT / 'plan.json')
        assert sha256(record['path']) == record['sha256']; return record
    inp = load(input_path(*key)); assert sha256(input_path(*key)) == plan['records'][rid]['files'][str(input_path(*key))]
    started = time.monotonic(); state = state_new(model); captures = {}; prefix = {}; hooks = []
    for index in LAYERS:
        def capture(module, args, i=index): captures[i] = args[0].to(torch.bfloat16).cpu()
        hooks.append(model.aggregator.global_blocks[index].attn.qkv.register_forward_pre_hook(capture))
    def prefix_hook(module, args): prefix['value'] = args[0].cpu()
    hooks.append(model.aggregator.global_blocks[20].register_forward_pre_hook(prefix_hook))
    norms = {i: [] for i in LAYERS}; depths, poses = [], []; memory = None
    try:
        for j, frame_id in enumerate(inp['frame_ids']):
            if frame_id == q:
                memory = []
                for i in LAYERS:
                    k, v = state['kv'][i]
                    memory.append({'k': k.flatten(2, 3).cpu(), 'v': v.flatten(2, 3).cpu(),
                                   'norm': torch.cat(norms[i], 1)})
            images = inp['images'][j:j + 1, None].cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tokens, start, state['kv'] = model.aggregator(images, past_key_values=state['kv'], use_cache=True, past_frame_idx=j)
            with torch.autocast('cuda', enabled=False):
                enc, state['cam'] = model.camera_head(tokens, past_key_values_camera=state['cam'], use_cache=True)
                depth, _ = model.depth_head(tokens, images, start)
            depths.append(depth[0, 0, ..., 0].float().cpu()); poses.append(enc[-1][0, 0].float().cpu())
            if frame_id != q:
                for i in LAYERS: norms[i].append(captures[i])
            else:
                early = {i: tokens[i].float().cpu() for i in (4, 11, 17)}
        geometry = patch_geometry(depths, poses, list(inp['images'].shape[-2:]))
        sample = {'id': rid, 'query': q, 'condition': condition, 'place': inp['place'], 'split': inp['split'],
                  'source_ids': inp['frame_ids'][:-1], 'memory': memory, 'prefix': prefix['value'], 'early': early,
                  'baseline_depth': depths[-1], 'target': inp['target'], 'valid': inp['valid'], 'hidden': inp['hidden'],
                  'native_hw': list(inp['images'].shape[-2:]), 'tokens_per_frame': prefix['value'].shape[1], **geometry}
        path = sample_path(*key); save_tensor(path, sample)
    finally:
        for hook in hooks: hook.remove()
    record = {'complete': True, 'key': list(key), 'path': str(path), 'sha256': sha256(path),
              'plan_sha256': sha256(OUT / 'plan.json'), 'input_sha256': sha256(input_path(*key)),
              'source_ids': inp['frame_ids'][:-1], 'geometry_statistics': sample['geometry_statistics'],
              'memory_bytes': sum(t.numel() * t.element_size() for mem in memory for t in mem.values()),
              'seconds': time.monotonic() - started, 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30,
              'no_query_truth_in_geometry': True, 'future_frames_used': False}
    write_json(marker, record); print(json.dumps({'cache_complete': name, 'seconds': record['seconds']}), flush=True)
    del state, tokens, depth, memory, sample, norms, captures, prefix
    torch.cuda.empty_cache()
    return record


def statistics(plan):
    rows = [read(OUT / 'cache' / (sample_key(*key).replace('/', '_') + '.json')) for key in examples(plan, 'train')]
    thresholds = {name: float(np.median([r['geometry_statistics'][name] for r in rows])) for name in ('extent_spread', 'anisotropy', 'overlap')}
    write_json(OUT / 'training_statistics.json', {'thresholds': thresholds, 'examples': len(rows),
               'training_keys': [r['key'] for r in rows], 'evaluation_used': False, 'plan_sha256': sha256(OUT / 'plan.json')})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--split', choices=['train', 'development', 'all'], default='all')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--statistics', action='store_true'); a = p.parse_args()
    torch.set_num_threads(4); plan = load_plan(); verify(plan)
    if a.statistics: statistics(plan); return
    model = model_load(plan); begin = time.monotonic()
    keys = examples(plan, 'train') + examples(plan, 'development') if a.split == 'all' else examples(plan, a.split)
    records = [build_one(model, key, plan) for key in keys[a.shard::a.shards]]
    write_json(OUT / f'cache_worker_{a.split}_{a.shard}.json', {'complete': True, 'keys': [r['key'] for r in records],
        'seconds': time.monotonic() - begin, 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30})


if __name__ == '__main__': main()
