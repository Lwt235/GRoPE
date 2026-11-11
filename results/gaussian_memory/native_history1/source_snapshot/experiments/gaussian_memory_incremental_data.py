"""Causal, reference-only observations and resumable VGGT prefix caches."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_view_data import load_backbone, rectangle
from experiments.gaussian_memory_data import merge_seeds
from experiments.round5_rgbd_data import _listing, sun3d_timestamp, load_sensor_depth
from models.gaussian_memory_view_token import apply_rgb_mask
from models.gaussian_memory_vggt_adapter import HEIGHT, WIDTH, PATCH, START, INSERT, padded_geometry

OUT = ROOT / 'results/gaussian_memory/incremental_vggt1'
CACHE = DATA / 'incremental_vggt_v1'
QUERIES = (5, 7, 9, 11)
WRITES = (4, 6, 8, 10)
SEEDS = (20261110, 20261111, 20261112)
KINDS = ('local', 'content', 'geometry', 'combined')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def cpu_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [cpu_tree(v) for v in x]
    return x


def device_tree(x, device='cuda'):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: device_tree(v, device) for k, v in x.items()}
    return x


def save_tensor(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(cpu_tree(value), temp)
    temp.replace(path)


def depth_source(frame):
    if 'depth_path' in frame:
        path = Path(frame['depth_path'])
    else:
        rgb = Path(frame['rgb_path'])
        folder = rgb.parent.parent
        names = _listing(folder / 'depth_index.html', 'png')
        name = min(names, key=lambda n: abs(sun3d_timestamp(n) - sun3d_timestamp(rgb.name)))
        path = folder / 'depth' / name
    assert path.is_file(), path
    return path


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'plan.json'
    if destination.exists():
        return json.loads(destination.read_text())
    previous = ROOT / 'results/gaussian_memory/content_center1/plan.json'
    old = json.loads(previous.read_text())
    rows, identities = [], {}
    for m in old['catalog']:
        row = dict(m['row'])
        if 'K' not in row or 'c2w_metric' not in row:
            raw = torch.load(m['raw'], map_location='cpu', weights_only=False)
            K = raw['K'].clone().float(); K[:, :2] *= PATCH
            pose = raw['c2w'].clone().double()
            pose[:, :3, 3] *= raw['diagnostics']['unit_meters']
            row.update(K=K.tolist(), c2w_metric=pose.tolist())
            identities[m['raw']] = sha256(m['raw'])
        frames = []
        for index, original in enumerate(row['frames'][:12]):
            f = dict(original)
            rgb = f['rgb_path']; digest = sha256(rgb)
            if 'rgb_sha256' in f:
                assert digest == f['rgb_sha256'], rgb
            f['rgb_sha256'] = digest; identities[rgb] = digest
            if index in QUERIES:
                source = depth_source(f); digest = sha256(source)
                if 'depth_sha256' in f:
                    assert digest == f['depth_sha256'], source
                f.update(depth_path=str(source), depth_sha256=digest)
                identities[str(source)] = digest
            frames.append(f)
        assert all(frames[i]['index'] < frames[i + 1]['index'] for i in range(11))
        row.update(frames=frames, reference=[0, 1, 2, 3], writable=list(WRITES), readonly=list(QUERIES))
        rows.append({'id': m['id'], 'row': row, 'eval_split': m['eval_split'],
                     'place': row['scene'].split('/')[0], 'training': m['id'] in old['training_ids']})
    identities[str(DATA / 'cache_v1/pca.pt')] = sha256(DATA / 'cache_v1/pca.pt')
    identities[str(DATA / 'source_identity.json')] = sha256(DATA / 'source_identity.json')
    plan = {'schema': 1, 'catalog': rows, 'training_ids': old['training_ids'],
            'primary_places': old['primary_places'], 'seeds': SEEDS, 'kinds': KINDS,
            'steps': 2000, 'effective_batch': 2, 'microbatch': 1, 'lr': 3e-4,
            'weight_decay': .01, 'gradient_clip': 5., 'insert_global_index': INSERT,
            'memory_slots': 256, 'content_dim': 32, 'known_camera': True,
            'reference': [0, 1, 2, 3], 'write_frames': WRITES, 'query_frames': QUERIES,
            'short_window_frames': 4, 'conditions': ['full', 'mask0', 'mask1'],
            'source_plan_sha256': sha256(previous), 'input_sha256': identities,
            'document_sha256': sha256(ROOT / 'docs/gaussian_memory_incremental_vggt_plan.md'),
            'backbone_identity': json.loads((DATA / 'source_identity.json').read_text()),
            'confirmation_claim': False, 'all_data_previously_used': True,
            'scale_rule': 'median known/predicted camera-baseline ratio, >=3 valid pairs; otherwise explicitly unavailable; no target alignment',
            'pixel_valid_range_m': [.1, 10.], 'shape_extent_floor_normalized': .005,
            'opacity': .7, 'projection_sigma_cutoff': 3., 'projection_blur_patch_variance': .1,
            'geometry_standardization': 'training-M0 projected PE RMS, no centering, clamp at 20',
            'history_weight_cap': 4., 'conflict_frames': 2, 'depth_agreement': .1,
            'updates_are_memory_free': True, 'read_before_write': True,
            'error_fraction': .2, 'error_amplitudes': [.2, .05],
            'cache_directory': str(CACHE)}
    write_json(destination, plan)
    print(json.dumps({'prepared': len(rows), 'training': len(old['training_ids']),
                      'query_labels': len(rows) * 4, 'new_downloads': 0}), flush=True)
    return plan


def load_plan():
    return json.loads((OUT / 'plan.json').read_text())


def select_rows(plan, subset):
    if subset == 'smoke':
        ids = plan['training_ids'][:2]
        return [next(m for m in plan['catalog'] if m['id'] == rid) for rid in ids]
    return [m for m in plan['catalog'] if subset == 'all' or m['training']]


def load_rgb(row, device='cuda'):
    images = []
    for f in row['frames']:
        with Image.open(f['rgb_path']) as im:
            a = np.asarray(im.convert('RGB').resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)).copy()
        images.append(torch.from_numpy(a).permute(2, 0, 1))
    return torch.stack(images).to(device).float() / 255.


def sensor_target(frame):
    d = load_sensor_depth(frame['depth_path'])
    d = np.asarray(Image.fromarray(d).resize((WIDTH, HEIGHT), Image.Resampling.NEAREST)).copy()
    valid = np.isfinite(d) & (d > .1) & (d < 10.)
    return torch.from_numpy(np.where(valid, d, 1).astype('float32')), torch.from_numpy(valid)


def calibrate(pose, known):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    ext, _ = pose_encoding_to_extri_intri(pose.float(), (HEIGHT, WIDTH))
    ext = ext[0].float()
    centers = -(ext[:, :3, :3].transpose(-1, -2) @ ext[:, :3, 3, None]).squeeze(-1)
    predicted = torch.pdist(centers)
    measured = torch.pdist(known[:, :3, 3])
    valid = (predicted > 1e-5) & (measured > 1e-4) & torch.isfinite(predicted)
    ratios = measured[valid] / predicted[valid]
    ready = len(ratios) >= 3 and torch.isfinite(ratios).all().item()
    diagnostic = {'valid': bool(ready), 'usable_pairs': len(ratios),
                  'ratios': ratios.tolist(), 'known_baselines_m': measured.tolist()}
    if not ready:
        return None, diagnostic
    scale = ratios.median().item()
    diagnostic.update(scale=scale, ratio_max_over_min=(ratios.max() / ratios.min()).item())
    return scale, diagnostic


@torch.no_grad()
def encode_window(backbone, images, pca):
    held = {}
    def prefix(module, args):
        held['prefix'] = args[0].detach().reshape(4, 305, 1024).clone()
    def patch(module, args, result):
        held['patch'] = result['x_norm_patchtokens'].detach()
    h1 = backbone.aggregator.global_blocks[INSERT].register_forward_pre_hook(prefix)
    h2 = backbone.aggregator.patch_embed.register_forward_hook(patch)
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tokens, start = backbone.aggregator(images[None])
    finally:
        h1.remove(); h2.remove()
    assert start == START
    with torch.autocast('cuda', enabled=False):
        pose = backbone.camera_head(tokens)[-1]
        depth, confidence = backbone.depth_head(tokens, images[None], start)
        # Same one-frame DPT call as training; other views are independent in DPT.
        one = [t[:, -1:] if t is not None else None for t in tokens]
        query_depth, _ = backbone.depth_head(one, images[None, -1:], start)
        feature = (held['patch'].float() - pca['mean']) @ pca['basis'] / pca['scale']
        local = F.adaptive_avg_pool2d(feature.reshape(4, 15, 20, 32).permute(0, 3, 1, 2), (8, 8))
        local = local.permute(0, 2, 3, 1).reshape(256, 32)
    return {'prefix': held['prefix'], 'early4': tokens[4].detach(), 'patch_feature': feature,
            'local_content': local, 'depth': depth[0, ..., 0].float(), 'confidence': confidence[0].float(),
            'query_depth_native': query_depth[0, 0, ..., 0].float(), 'pose': pose}


def observations(encoded, ids, K, c2w_metric, scale, unit, current_only=False):
    chosen = [-1] if current_only else list(range(len(ids)))
    actual = [ids[i] for i in chosen]
    camera = c2w_metric[actual].clone()
    camera[:, :3, 3] /= unit
    depth = encoded['depth'][chosen] * (scale / unit)
    y, x = torch.meshgrid(torch.arange(HEIGHT, device=depth.device) + .5,
                         torch.arange(WIDTH, device=depth.device) + .5, indexing='ij')
    uv = torch.stack((x, y, torch.ones_like(x)), -1)
    rays = torch.einsum('hwj,vij->vhwi', uv, torch.linalg.inv(K[actual]))
    points = rays * depth[..., None]
    points = torch.einsum('vhwi,vji->vhwj', points, camera[:, :3, :3]) + camera[:, None, None, :3, 3]
    grid = points.reshape(-1, 15, 14, 20, 14, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 196, 3)
    means = grid.mean(1); delta = grid - means[:, None]
    cov = delta.transpose(1, 2) @ delta / 196 + torch.eye(3, device=depth.device) * .005**2
    confidence = encoded['confidence'][chosen].reshape(-1, 15, 14, 20, 14)
    confidence = confidence.permute(0, 1, 3, 2, 4).reshape(-1, 196).mean(1)
    valid = torch.isfinite(means).all(-1) & torch.isfinite(cov).all((-1, -2))
    valid &= torch.isfinite(confidence) & (confidence >= confidence.quantile(.1))
    return {'means': means, 'covars': cov, 'features': encoded['patch_feature'][chosen].reshape(-1, 32),
            'confidence': confidence, 'valid': valid,
            'source_frame': torch.tensor(actual, device=depth.device).repeat_interleave(300),
            'camera': camera, 'K': K[actual] / K.new_tensor([14, 14, 1])[:, None],
            'depth_patch': depth.reshape(-1, 15, 14, 20, 14).permute(0, 1, 3, 2, 4).reshape(-1, 196).median(-1).values}


def build_memory(obs):
    mean, chol, near, weights = merge_seeds(obs['means'], obs['covars'], obs['valid'])
    feature = (weights[..., None] * obs['features'][near]).sum(1)
    sources = F.one_hot(obs['source_frame'][near], 12).bool().any(1)
    return {'means': mean, 'covars': chol @ chol.transpose(-1, -2), 'features': feature,
            'opacity': mean.new_full((256,), .7), 'support': sources.sum(-1).float(),
            'sources': sources, 'conflicts': torch.zeros(256, 12, device=mean.device, dtype=torch.bool),
            'lineage': torch.eye(256, device=mean.device, dtype=torch.bool)}


def extract(subset='all'):
    plan = load_plan(); rows = select_rows(plan, subset)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(SEEDS[0])
    backbone = load_backbone()
    pca = torch.load(DATA / 'cache_v1/pca.pt', map_location='cuda', weights_only=False)
    start_time = time.monotonic(); completed = []; failures = []
    sources = {p: sha256(ROOT / p) for p in ['experiments/gaussian_memory_incremental_data.py',
                                           'models/gaussian_memory_vggt_adapter.py']}
    for meta in rows:
        rid = meta['id']; folder = CACHE / rid; folder.mkdir(parents=True, exist_ok=True)
        identity_path = folder / 'identity.json'
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            assert identity['schema'] == 1 and identity['source_sha256'] == sources, rid
            completed.append(rid); continue
        row = meta['row']; images = load_rgb(row)
        K = torch.tensor(row['K'], device='cuda', dtype=torch.float32)
        cameras = torch.tensor(row['c2w_metric'], device='cuda', dtype=torch.float32)
        diagnostics = []; hashes = {}
        initial = encode_window(backbone, images[:4], pca)
        scale, diag = calibrate(initial['pose'], cameras[:4]); diag.update(window=[0, 1, 2, 3], kind='initial')
        diagnostics.append(diag)
        if scale is None:
            failures.append({'id': rid, 'reason': 'initial scale unavailable', 'diagnostic': diag})
            write_json(folder / 'unavailable.json', failures[-1]); continue
        unit = scale * initial['depth'].median().item()
        obs0 = observations(initial, [0, 1, 2, 3], K, cameras, scale, unit)
        memory = build_memory(obs0)
        normalized_camera = cameras.clone(); normalized_camera[:, :3, 3] /= unit
        patch_K = K / K.new_tensor([14, 14, 1])[:, None]
        save_tensor(folder / 'initial.pt', {'memory': memory, 'observation': obs0, 'unit_meters': unit,
                     'c2w': normalized_camera, 'K': patch_K, 'source_frames': [0, 1, 2, 3]})
        del initial
        past = [0, 1, 2, 3]; samples = []; updates = {}
        for frame in range(4, 12):
            ids = past[-3:] + [frame]
            assert all(i < frame and i not in QUERIES for i in ids[:-1])
            if frame in WRITES:
                encoded = encode_window(backbone, images[ids], pca)
                scale, diag = calibrate(encoded['pose'], cameras[ids])
                diag.update(window=ids, kind='write'); diagnostics.append(diag)
                if scale is not None:
                    updates[frame] = observations(encoded, ids, K, cameras, scale, unit, True)
                else:
                    updates[frame] = None  # Explicit missing observation; no incompatible scale fallback.
                past.append(frame); del encoded
            else:
                target, valid = sensor_target(row['frames'][frame])
                pe = padded_geometry(memory, normalized_camera[ids], patch_K[ids])
                for condition in range(3):
                    mask = torch.zeros(300, dtype=torch.bool) if condition == 0 else rectangle(rid, frame, condition - 1)
                    inputs = images[ids].clone()
                    inputs[-1:] = apply_rgb_mask(inputs[-1:], mask[None].cuda())
                    encoded = encode_window(backbone, inputs, pca)
                    scale, diag = calibrate(encoded['pose'], cameras[ids])
                    diag.update(window=ids, kind='query', condition=condition); diagnostics.append(diag)
                    sample = {k: encoded[k] for k in ('prefix', 'early4', 'local_content', 'query_depth_native')}
                    sample.update(id=rid, query=frame, condition=condition, input_frames=ids,
                                  pe_raw=pe, metric_scale=scale, calibration_valid=scale is not None,
                                  c2w=normalized_camera[ids], K=patch_K[ids],
                                  target=target, valid=valid, hidden=mask.reshape(15, 20).repeat_interleave(14, 0).repeat_interleave(14, 1),
                                  rgb=(inputs[-1].permute(1, 2, 0) * 255).round().byte(),
                                  memory_source_frames=[0, 1, 2, 3])
                    name = f'q{frame}_c{condition}.pt'
                    save_tensor(folder / name, sample); samples.append(name)
                    del sample, encoded
        save_tensor(folder / 'updates.pt', updates)
        for path in [folder / 'initial.pt', folder / 'updates.pt'] + [folder / name for name in samples]:
            hashes[path.name] = sha256(path)
        identity = {'schema': 1, 'id': rid, 'source_sha256': sources, 'sha256': hashes,
                    'samples': samples, 'unit_meters': unit, 'diagnostics': diagnostics,
                    'query_frames_never_written': True, 'sensor_in_state': False,
                    'calibration_failures': sum(not d['valid'] for d in diagnostics)}
        write_json(identity_path, identity); completed.append(rid)
        elapsed = time.monotonic() - start_time
        write_json(OUT / 'extract_progress.json', {'completed': completed, 'subset': subset, 'seconds': elapsed, 'failures': failures})
        print(json.dumps({'extracted': rid, 'completed': len(completed), 'planned': len(rows),
                          'seconds': round(elapsed, 2), 'scale_unavailable': identity['calibration_failures'],
                          'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
    write_json(OUT / f'extract_{subset}_complete.json', {'completed': completed, 'failures': failures,
               'seconds': time.monotonic() - start_time, 'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3})


def training_statistics():
    plan = load_plan(); squares = torch.zeros(10, dtype=torch.float64); count = 0; variances = []
    for rid in plan['training_ids']:
        initial = torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False)
        variances.extend(initial['memory']['covars'].diagonal(dim1=-2, dim2=-1).mean(-1).tolist())
        for q in QUERIES:
            s = torch.load(CACHE / rid / f'q{q}_c0.pt', map_location='cpu', weights_only=False)
            p = s['pe_raw'][:, START:].double().reshape(-1, 10)
            squares += p.square().sum(0); count += len(p)
    result = {'geometry_rms': (squares / count).sqrt().clamp_min(1e-4).tolist(),
              'center_variance': float(np.median(variances)), 'training_ids': plan['training_ids'],
              'pixels_seen': count, 'evaluation_data_used': False}
    write_json(OUT / 'training_statistics.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'extract', 'statistics'))
    parser.add_argument('--subset', choices=('smoke', 'train', 'all'), default='all')
    args = parser.parse_args()
    if args.stage == 'extract': extract(args.subset)
    elif args.stage == 'statistics': training_statistics()
    else: prepare()
