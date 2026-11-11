"""Replay only the original frozen, causal source windows to retain dense depth."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_raw_history_data import (
    ROOT, DATA, OUT, CACHE, BASE_CACHE, SOURCE_FRAMES, load_plan, sha256, write_json, save_tensor)
from experiments.gaussian_memory_incremental_data import load_rgb, encode_window, observations
from experiments.gaussian_memory_view_data import load_backbone


@torch.no_grad()
def main(shard, shards):
    plan = load_plan(); start = time.monotonic(); torch.set_num_threads(4)
    assert torch.cuda.is_available()
    for name, digest in plan['source_sha256'].items(): assert sha256(ROOT / name) == digest
    backbone = load_backbone(); pca = torch.load(DATA / 'cache_v1/pca.pt', map_location='cuda', weights_only=False)
    total = 0; max_difference = 0.; completed = []
    for meta in plan['catalog'][shard::shards]:
        rid = meta['id']; marker = OUT / 'sources' / f'{rid}.json'
        if marker.exists():
            old = json.loads(marker.read_text()); assert old['complete'] and sha256(old['path']) == old['sha256']
            completed.append(rid); continue
        tick = time.monotonic(); identity_path = BASE_CACHE / rid / 'identity.json'
        assert sha256(identity_path) == plan['input_identities'][rid]['sha256']
        identity = json.loads(identity_path.read_text())
        initial = torch.load(BASE_CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False)
        updates = torch.load(BASE_CACHE / rid / 'updates.pt', map_location='cpu', weights_only=False)
        images = load_rgb(meta['row']); unit = initial['unit_meters']
        K = torch.tensor(meta['row']['K'], device='cuda', dtype=torch.float32)
        cameras = torch.tensor(meta['row']['c2w_metric'], device='cuda', dtype=torch.float32)
        depths = {}; rows = []; patch_error = 0.; valid_mismatch = 0
        for d in [x for x in identity['diagnostics'] if x['kind'] in ('initial', 'write')]:
            ids = d['window']; assert len(ids) == 4 and all(i in SOURCE_FRAMES for i in ids)
            assert d['valid'] and d.get('scale') is not None, (rid, ids, 'old scale unavailable')
            encoded = encode_window(backbone, images[ids], pca)
            current_only = d['kind'] == 'write'; chosen = [-1] if current_only else list(range(4))
            old = updates[ids[-1]] if current_only else initial['observation']
            replay = observations(encoded, ids, K, cameras, d['scale'], unit, current_only)
            err = float((replay['depth_patch'].cpu() - old['depth_patch']).abs().max())
            patch_error = max(patch_error, err)
            torch.testing.assert_close(replay['depth_patch'].cpu(), old['depth_patch'],
                rtol=plan['replay_patch_depth_rtol'], atol=plan['replay_patch_depth_atol'])
            valid_mismatch += int((replay['valid'].cpu() != old['valid']).sum())
            for j in chosen:
                frame = ids[j]; assert frame not in depths
                depths[frame] = (encoded['depth'][j] * d['scale']).cpu()
                assert torch.isfinite(depths[frame]).all() and (depths[frame] > 0).all()
            rows.append({'input_frames': ids, 'output_frames': [ids[j] for j in chosen],
                         'old_metric_scale': d['scale'], 'patch_depth_max_absolute_difference': err})
            total += 1; del encoded, replay
        assert sorted(depths) == list(SOURCE_FRAMES)
        target = CACHE / 'sources' / f'{rid}.pt'; target.parent.mkdir(parents=True, exist_ok=True)
        save_tensor(target, {'source_depth_m': depths, 'source_frames': list(SOURCE_FRAMES)})
        result = {'complete': True, 'id': rid, 'path': str(target), 'sha256': sha256(target),
            'plan_sha256': sha256(OUT / 'plan.json'), 'source_identity_sha256': sha256(identity_path),
            'replayed_windows': rows, 'patch_depth_max_absolute_difference': patch_error,
            'patch_confidence_admission_mismatches': valid_mismatch,
            'seconds': time.monotonic() - tick, 'new_training_steps': 0}
        write_json(marker, result); completed.append(rid); max_difference = max(max_difference, patch_error)
        print(json.dumps({'source_complete': rid, 'windows': total, 'patch_depth_max_difference': patch_error}), flush=True)
        del images, initial, updates
    result = {'complete': True, 'shard': shard, 'shards': shards, 'clips': completed, 'windows_this_run': total,
        'seconds': time.monotonic() - start, 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'max_patch_depth_difference_this_run': max_difference, 'no_new_training': True}
    write_json(OUT / f'extract_shard_{shard}.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args(); main(a.shard, a.shards)
