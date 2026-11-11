"""Raw-source z-buffer information and existing-state support, without fitting."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_raw_history_data import (
    ROOT, OUT, CACHE, BASE_CACHE, SOURCE_FRAMES, VARIANTS, QUERIES, POLICIES,
    load_plan, sha256, write_json, save_tensor)
from experiments.gaussian_memory_incremental_data import sensor_target
from experiments.gaussian_memory_view_data import rectangle
from models.gaussian_memory_vggt_adapter import project_memory


def unproject(depth, valid, camera, intrinsic):
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype) + .5,
                          torch.arange(w, device=depth.device, dtype=depth.dtype) + .5, indexing='ij')
    uv = torch.stack((x, y, torch.ones_like(x)), -1)
    points = (uv @ torch.linalg.inv(intrinsic).T) * depth[..., None]
    return points[valid] @ camera[:3, :3].T + camera[:3, 3]


def zbuffer(points, camera, intrinsic, height=210, width=280):
    local = (points - camera[:3, 3]) @ camera[:3, :3]
    keep = torch.isfinite(local).all(-1) & (local[:, 2] > 1e-4)
    local = local[keep]; uvw = local @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    pixel = uv[inside].floor().long(); z = local[inside, 2]
    result = points.new_full((height * width,), float('inf'))
    result.scatter_reduce_(0, pixel[:, 1] * width + pixel[:, 0], z, reduce='amin', include_self=True)
    return torch.where(torch.isfinite(result), result, 0.).reshape(height, width)


def merge_depth(a, b):
    return torch.where((a > 0) & (b > 0), torch.minimum(a, b), torch.maximum(a, b))


def conditions(rid, q):
    all_pixels = np.ones((210, 280), bool)
    yield 0, 'all', all_pixels
    for c in (1, 2):
        mask = rectangle(rid, q, c - 1).numpy().reshape(15, 20).repeat(14, 0).repeat(14, 1)
        yield c, 'all', all_pixels
        yield c, 'hidden', mask
        yield c, 'visible', ~mask


def ratio(n, d): return float(n / d) if d else None


def score(depth, target, valid, region):
    support = depth > 0; known = valid & region; selected = support & known
    relative = np.abs(depth - target) / target
    good = selected & (relative <= .1)
    n = int(selected.sum()); denom = int(known.sum())
    return {'region_pixels': int(region.sum()), 'valid_pixels': denom, 'supported_valid_pixels': n,
        'support_fraction': ratio((support & region).sum(), region.sum()),
        'valid_support_fraction': ratio(n, denom), 'consistent_fraction': ratio(good.sum(), denom),
        'consistency_given_support': ratio(good.sum(), n),
        'foreground_conflict_fraction': ratio((selected & (depth < target * .9)).sum(), denom),
        'background_fraction': ratio((selected & (depth > target * 1.1)).sum(), denom),
        'supported_absrel': float(relative[selected].mean()) if n else None,
        'supported_log_mse': float(np.mean(np.log(depth[selected] / target[selected])**2)) if n else None}


def history_gain(recent, history, target, valid, region):
    known = valid & region; rs = recent > 0; hs = history > 0; new = hs & ~rs
    rg = rs & (np.abs(recent - target) / target <= .1)
    hg = hs & (np.abs(history - target) / target <= .1)
    new_valid = new & known; denom = int(known.sum()); n = int(new_valid.sum())
    return {'new_support_fraction': ratio((new & region).sum(), region.sum()),
        'overlap_support_fraction': ratio((rs & hs & region).sum(), region.sum()),
        'new_valid_support_pixels': n, 'new_valid_support_fraction': ratio(n, denom),
        'new_consistent_support_fraction': ratio((new_valid & hg).sum(), denom),
        'new_support_consistency': ratio((new_valid & hg).sum(), n),
        'new_support_absrel': float((np.abs(history - target) / target)[new_valid].mean()) if n else None,
        'consistency_net_gain': ratio(int((hg & known).sum()) - int((rg & known).sum()), denom),
        'consistency_added_fraction': ratio((hg & ~rg & known).sum(), denom),
        'consistency_lost_fraction': ratio((rg & ~hg & known).sum(), denom)}


def source_clouds(meta, common_from_local, device):
    rid = meta['id']; identity_path = OUT / 'sources' / f'{rid}.json'
    info = json.loads(identity_path.read_text()); assert info['complete'] and sha256(info['path']) == info['sha256']
    saved = torch.load(info['path'], map_location='cpu', weights_only=False)['source_depth_m']
    cameras = torch.tensor(meta['row']['c2w_metric'], device=device, dtype=torch.float64)
    cameras = common_from_local @ cameras
    intrinsic = torch.tensor(meta['row']['K'], device=device, dtype=torch.float64)
    clouds = {v: {} for v in VARIANTS}; counts = []
    for frame in SOURCE_FRAMES:
        f = meta['row']['frames'][frame]; assert sha256(f['depth_path']) == f['depth_sha256']
        sensor, sv = sensor_target(f); sensor = sensor.to(device=device, dtype=torch.float64); sv = sv.to(device)
        pred = saved[frame].to(device=device, dtype=torch.float64)
        pv = torch.isfinite(pred) & (pred > .1) & (pred < 10.)
        common = sv & pv
        for v, d, mask in [('sensor', sensor, sv), ('prediction', pred, pv),
                           ('sensor_matched', sensor, common), ('prediction_matched', pred, common)]:
            clouds[v][frame] = unproject(d, mask, cameras[frame], intrinsic[frame])
        counts.append({'id': rid, 'frame': frame, 'sensor_valid': int(sv.sum()),
                       'prediction_valid': int(pv.sum()), 'common_valid': int(common.sum()), 'pixels': 58800})
    return clouds, counts


def preflight():
    k = torch.tensor([[3., 0., 2.], [0., 3., 1.5], [0., 0., 1.]], dtype=torch.float64, device='cuda')
    c = torch.eye(4, dtype=torch.float64, device='cuda')
    d = torch.full((3, 4), 2., dtype=torch.float64, device='cuda'); v = torch.ones_like(d, dtype=torch.bool)
    points = unproject(d, v, c, k)
    identity = zbuffer(points, c, k, 3, 4)
    torch.testing.assert_close(identity, d, rtol=0, atol=1e-12)
    front = unproject(d / 2, v, c, k)
    merged = zbuffer(torch.cat((points, front, -points)), c, k, 3, 4)
    torch.testing.assert_close(merged, d / 2, rtol=0, atol=1e-12)
    moved = c.clone(); moved[:3, 3] = torch.tensor([.3, -.2, .7], device='cuda')
    torch.testing.assert_close(zbuffer(unproject(d, v, moved, k), moved, k, 3, 4), d, rtol=0, atol=1e-12)
    return {'identity_projection': True, 'nearest_front_depth_and_behind_camera_rejection': True,
            'rigid_translation_invariance': True, 'scope': 'Small implementation check; no task labels or tuning.'}


@torch.no_grad()
def main():
    plan = load_plan(); start = time.monotonic(); assert torch.cuda.is_available()
    for name, digest in plan['source_sha256'].items(): assert sha256(ROOT / name) == digest
    write_json(OUT / 'projection_preflight.json', preflight())
    catalog = {m['id']: m for m in plan['catalog']}; all_counts = []; cpu_rows = 0
    for pair in plan['pairs']:
        tick = time.monotonic(); rid = pair['id']; a = catalog[pair['early_id']]; b = catalog[rid]
        eye = torch.eye(4, device='cuda', dtype=torch.float64)
        transform = torch.tensor(pair['common_from_late_metric'], device='cuda', dtype=torch.float64)
        early, ec = source_clouds(a, eye, 'cuda'); late, lc = source_clouds(b, transform, 'cuda'); all_counts += ec + lc
        source_info = pair['state_identity']; assert sha256(source_info['trajectories_path']) == source_info['trajectories_sha256']
        tracks = torch.load(source_info['trajectories_path'], map_location='cpu', weights_only=False)
        cameras = transform @ torch.tensor(b['row']['c2w_metric'], device='cuda', dtype=torch.float64)
        intrinsics = torch.tensor(b['row']['K'], device='cuda', dtype=torch.float64)
        raw_rows = []; delta_rows = []; state_rows = []; retained_rows = []; maps = {}
        for ti, q in enumerate(QUERIES, 1):
            target, valid = sensor_target(b['row']['frames'][q]); target = target.numpy().astype('float64'); valid = valid.numpy()
            raw_maps = {}; selected = [f for f in SOURCE_FRAMES if f < q]
            assert all(f < q and f not in QUERIES for f in selected)
            for variant in VARIANTS:
                ed = zbuffer(torch.cat(list(early[variant].values())), cameras[q], intrinsics[q])
                rd = zbuffer(torch.cat([late[variant][f] for f in selected]), cameras[q], intrinsics[q])
                hd = merge_depth(ed, rd)
                raw_maps[variant] = {name: d.cpu().numpy() for name, d in [('early', ed), ('recent', rd), ('history', hd)]}
                for c, region_name, region in conditions(rid, q):
                    base = {'id': rid, 'place': pair['place'], 'query': q, 'condition': c, 'region': region_name, 'variant': variant}
                    for name, depth in raw_maps[variant].items():
                        raw_rows.append({**base, 'range': name, **score(depth, target, valid, region)})
                    delta_rows.append({**base, **history_gain(raw_maps[variant]['recent'], raw_maps[variant]['history'], target, valid, region)})
            state_maps = {}
            cam = cameras[q].float().clone(); cam[:3, 3] /= pair['common_unit_meters']
            intrinsic = intrinsics[q].float() / intrinsics.new_tensor([14, 14, 1]).float()[:, None]
            for policy in POLICIES:
                state = {k: v.cuda() for k, v in tracks[policy]['states'][ti].items()}
                p = project_memory(state, cam[None], intrinsic[None])
                coverage = p['coverage'][0].reshape(15, 20).repeat_interleave(14, 0).repeat_interleave(14, 1)
                depth = p['depth'][0].reshape(15, 20).repeat_interleave(14, 0).repeat_interleave(14, 1) * pair['common_unit_meters']
                depth = torch.where(coverage > .1, depth, 0.).cpu().numpy().astype('float64')
                state_maps[policy] = {'depth_m': depth, 'coverage': coverage.cpu().numpy()}
                for c, region_name, region in conditions(rid, q):
                    base = {'id': rid, 'place': pair['place'], 'query': q, 'condition': c, 'region': region_name, 'policy': policy}
                    state_rows.append({**base, **score(depth, target, valid, region)})
                    for variant in VARIANTS:
                        raw = raw_maps[variant]; new = (raw['history'] > 0) & (raw['recent'] == 0) & valid & region
                        useful = new & (np.abs(raw['history'] - target) / target <= .1)
                        present = depth > 0; consistent = present & (np.abs(depth - target) / target <= .1)
                        denom = int((valid & region).sum()); n = int(useful.sum())
                        retained_rows.append({**base, 'variant': variant, 'raw_new_consistent_pixels': n,
                            'state_covers_raw_new_fraction': ratio((present & useful).sum(), n),
                            'state_consistent_on_raw_new_fraction': ratio((consistent & useful).sum(), n),
                            'raw_new_consistent_fraction': ratio(n, denom),
                            'retained_consistent_fraction_all_valid': ratio((consistent & useful).sum(), denom),
                            'state_covers_any_raw_new_fraction': ratio((present & new).sum(), int(new.sum()))})
            maps[q] = {'raw': raw_maps, 'state': state_maps, 'recent_sources': selected,
                       'early_sources': list(SOURCE_FRAMES), 'target_depth_sha256': b['row']['frames'][q]['depth_sha256']}
        path = CACHE / 'maps' / f'{rid}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
        save_tensor(path, maps)
        result = {'complete': True, 'id': rid, 'place': pair['place'], 'raw_rows': raw_rows, 'history_rows': delta_rows,
            'state_rows': state_rows, 'retention_rows': retained_rows, 'map_path': str(path), 'map_sha256': sha256(path),
            'plan_sha256': sha256(OUT / 'plan.json'), 'seconds': time.monotonic() - tick}
        write_json(OUT / 'evaluation' / f'{rid}.json', result)
        cpu_rows += len(raw_rows) + len(delta_rows) + len(state_rows) + len(retained_rows)
        print(json.dumps({'pair_complete': rid, 'seconds': result['seconds'], 'all_score_rows': cpu_rows}), flush=True)
        del early, late, tracks, maps
    write_json(OUT / 'source_validity.json', {'complete': True, 'frames': all_counts, 'total_sources': len(all_counts)})
    result = {'complete': True, 'pairs': 12, 'query_cameras': 48, 'query_conditions': 144,
        'variants': list(VARIANTS), 'score_rows': cpu_rows, 'seconds': time.monotonic() - start,
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'new_training_steps': 0, 'new_download_bytes': 0,
        'scope': 'Projection information diagnostic, not new native predictions or learned utility.'}
    write_json(OUT / 'projection_complete.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); main()
