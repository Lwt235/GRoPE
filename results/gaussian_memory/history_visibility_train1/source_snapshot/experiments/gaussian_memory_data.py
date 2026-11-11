"""Reference-only VGGT extraction and explicitly separated teacher supervision."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np  # Import before torch on this server.
import torch
from PIL import Image

from experiments.gaussian_memory_setup import DATA, ROOT, COMMIT, sha256

REF = [0, 4, 8, 12]
TRAIN_QUERY = [1, 3, 5, 7, 9, 11]
HOLD_QUERY = [2, 6, 10]
HEIGHT, WIDTH, PATCH, MEMORY = 210, 280, 14, 256
CACHE = DATA / 'cache_v1'
REPORT = ROOT / 'results/gaussian_memory/prototype1'


def episodes():
    source = ROOT / 'results/round70_nvs/attempt1/data'
    manifest = json.loads((source / 'manifest.json').read_text())
    result = []
    for split, scenes in manifest['scenes'].items():
        with np.load(source / f'{split}.npz') as a:
            for scene in scenes:
                frames = sorted((f for f in manifest['frames'] if f['split'] == split and f['scene'] == scene),
                                key=lambda f: f['index'])
                for start in ([0, 40, 80, 120] if split == 'fit' else [0]):
                    chosen = frames[start:start + 13]
                    assert len(chosen) == 13
                    ids = [f['bank_index'] for f in chosen]
                    K = a['K'][ids].copy()
                    K[:, :2] *= HEIGHT / 96
                    poses = a['c2w'][ids].astype('float64')
                    poses = np.linalg.inv(poses[0])[None] @ poses
                    result.append({'id': f'{split}_{len(result):02d}', 'split': split, 'scene': scene,
                                   'start': start, 'frames': chosen, 'K': K.tolist(),
                                   'c2w_metric': poses.tolist(), 'reference': REF,
                                   'training_query': TRAIN_QUERY if split == 'fit' else [],
                                   'heldout_query': HOLD_QUERY if split == 'fit' else sorted(TRAIN_QUERY + HOLD_QUERY)})
    return result, source / 'manifest.json'


def make_manifest():
    REPORT.mkdir(parents=True, exist_ok=True)
    rows, source = episodes()
    data = {'source_manifest': str(source), 'source_manifest_sha256': sha256(source),
            'history': 'all scenes previously used; heldout means not optimized by this prototype',
            'reference_rgb_only': True, 'sensor_depth_read': False, 'given_camera': True,
            'image_hw': [HEIGHT, WIDTH], 'feature_hw': [HEIGHT // PATCH, WIDTH // PATCH],
            'memory_budget': MEMORY, 'episodes': rows,
            'source_identity': json.loads((DATA / 'source_identity.json').read_text())}
    dest = REPORT / 'data_manifest.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == data, 'do not silently change existing selection'
    else:
        dest.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'manifest': str(dest), 'episodes': len(rows),
                      'fit': sum(r['split'] == 'fit' for r in rows)}), flush=True)
    return data


def load_images(row, device):
    out = []
    for f in row['frames']:
        assert sha256(f['rgb_path']) == f['rgb_sha256'], f['rgb_path']
        with Image.open(f['rgb_path']) as im:
            image = np.asarray(im.convert('RGB').resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)).copy()
        out.append(torch.from_numpy(image).permute(2, 0, 1))
    return torch.stack(out).to(device).float() / 255


def source_geometry(depth, confidence, pose_enc, K, c2w):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    ext, _ = pose_encoding_to_extri_intri(pose_enc, (HEIGHT, WIDTH))
    ext = ext[0].float()
    pred_centers = -(ext[:, :3, :3].transpose(1, 2) @ ext[:, :3, 3, None]).squeeze(-1)
    dist_pred = torch.pdist(pred_centers)
    dist_known = torch.pdist(c2w[REF, :3, 3])
    usable = (dist_pred > 1e-5) & (dist_known > 1e-4)
    if usable.sum() < 3:
        raise RuntimeError('Insufficient nonzero reference baselines for scale calibration')
    ratios = dist_known[usable] / dist_pred[usable]
    metric_scale = ratios.median()
    depth = depth[0, ..., 0].float()
    if not torch.isfinite(depth).all() or not (depth > 0).all():
        raise RuntimeError('nonfinite/nonpositive VGGT reference depth')
    depth_unit = depth.median()
    unit_meters = metric_scale * depth_unit
    c2w = c2w.clone()
    c2w[:, :3, 3] /= unit_meters
    y, x = torch.meshgrid(torch.arange(HEIGHT, device=depth.device) + .5,
                         torch.arange(WIDTH, device=depth.device) + .5, indexing='ij')
    uv = torch.stack([x, y, torch.ones_like(x)], -1)
    rays = torch.einsum('hwj,vij->vhwi', uv, torch.linalg.inv(K[REF]))
    points = rays * (depth / depth_unit)[..., None]
    points = torch.einsum('vhwi,vji->vhwj', points, c2w[REF, :3, :3]) + c2w[REF, None, None, :3, 3]
    grid = points.reshape(4, HEIGHT // PATCH, PATCH, WIDTH // PATCH, PATCH, 3)
    grid = grid.permute(0, 1, 3, 2, 4, 5).reshape(-1, PATCH * PATCH, 3)
    means = grid.mean(1)
    delta = grid - means[:, None]
    cov = delta.transpose(1, 2) @ delta / (PATCH * PATCH)
    cov += torch.eye(3, device=depth.device) * .005 ** 2
    conf = confidence[0].float().reshape(4, HEIGHT // PATCH, PATCH, WIDTH // PATCH, PATCH)
    conf = conf.permute(0, 1, 3, 2, 4).reshape(-1, PATCH * PATCH).mean(1)
    valid = torch.isfinite(means).all(-1) & torch.isfinite(conf) & (conf >= conf.quantile(.10))
    diagnostics = {'unit_meters': unit_meters.item(), 'metric_scale': metric_scale.item(),
                   'predicted_depth_median': depth_unit.item(), 'reference_baselines_m': dist_known.tolist(),
                   'scale_ratios': ratios.tolist(), 'valid_reference_patches': int(valid.sum()),
                   'reference_scale_ratio_max_over_min': float(ratios.max() / ratios.min())}
    return means, cov, valid, c2w, diagnostics


def merge_seeds(means, cov, valid):
    ids = valid.nonzero().flatten()
    if len(ids) < MEMORY:
        raise RuntimeError('too few finite reference patches')
    pts = means[ids]
    # Deterministic FPS, then local moment-preserving merging across all views.
    current = (pts - pts.mean(0)).square().sum(-1).argmin()
    minimum = torch.full((len(pts),), float('inf'), device=pts.device)
    selected = []
    for _ in range(MEMORY):
        selected.append(current)
        minimum = torch.minimum(minimum, (pts - pts[current]).square().sum(-1))
        current = minimum.argmax()
    centers = pts[torch.stack(selected)]
    distance, nearest = torch.cdist(centers, pts).topk(8, largest=False)
    weights = (-distance.square() / distance[:, -1:].square().clamp_min(1e-6)).softmax(-1)
    near_ids = ids[nearest]
    seed_mean = (weights[..., None] * means[near_ids]).sum(1)
    delta = means[near_ids] - seed_mean[:, None]
    seed_cov = (weights[..., None, None] * (cov[near_ids] + delta[..., :, None] * delta[..., None, :])).sum(1)
    seed_cov += torch.eye(3, device=pts.device) * 1e-6
    return seed_mean, torch.linalg.cholesky(seed_cov), near_ids, weights


def extract(limit=None):
    from safetensors.torch import load_file
    sys.path.insert(0, str(ROOT / 'third_party/vggt'))
    from vggt.models.vggt import VGGT
    assert (ROOT / 'third_party/vggt/PINNED_COMMIT').read_text().strip() == COMMIT
    manifest = make_manifest()
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261110)
    device = torch.device('cuda')
    model = VGGT(enable_point=False, enable_track=False)
    state = load_file(str(DATA / 'model.safetensors'))
    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.missing_keys, incompatible.missing_keys
    assert all(k.startswith(('point_head.', 'track_head.')) for k in incompatible.unexpected_keys)
    del state
    model.eval().requires_grad_(False).to(device)
    rows = manifest['episodes'][:limit]
    for row in rows:
        dest = CACHE / (row['id'] + '_raw.pt')
        if dest.exists():
            print(json.dumps({'cached': row['id']}), flush=True)
            continue
        images = load_images(row, device)
        # The query teacher does image-wise self-attention only. It is never
        # concatenated with references for VGGT alternating/global attention.
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            teacher = []
            for chunk in images.split(4):
                norm = (chunk - model.aggregator._resnet_mean[0]) / model.aggregator._resnet_std[0]
                teacher.append(model.aggregator.patch_embed(norm)['x_norm_patchtokens'])
            teacher = torch.cat(teacher).float()
            ref_images = images[REF][None]
            tokens, start = model.aggregator(ref_images)
            context = tokens[-1][0, :, start:].float().reshape(-1, 2048)
            with torch.autocast('cuda', enabled=False):
                pose = model.camera_head(tokens)[-1]
                depth, conf = model.depth_head(tokens, images=ref_images, patch_start_idx=start)
            with torch.autocast('cuda', enabled=False):
                K = torch.tensor(row['K'], device=device).float()
                c2w = torch.tensor(row['c2w_metric'], device=device).float()
                means, cov, valid, c2w, diagnostics = source_geometry(depth, conf, pose, K, c2w)
                seed_mean, seed_chol, near_ids, weights = merge_seeds(means, cov, valid)
                seed_context = (context[near_ids] * weights[..., None]).sum(1)
                seed_teacher = (teacher[REF].reshape(-1, 1024)[near_ids] * weights[..., None]).sum(1)
            raw = {'seed_mean': seed_mean.cpu(), 'seed_chol': seed_chol.cpu(),
                   'seed_context': seed_context.half().cpu(), 'seed_teacher': seed_teacher.cpu(),
                   'teacher': teacher.half().cpu(), 'K': (K / K.new_tensor([PATCH, PATCH, 1])[:, None]).cpu(),
                   'c2w': c2w.cpu(), 'diagnostics': diagnostics, 'episode': row,
                   'rgb': (images.permute(0, 2, 3, 1) * 255).round().byte().cpu()}
            torch.save(raw, dest)
        print(json.dumps({'extracted': row['id'], **diagnostics, 'max_memory_GiB':
                          torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
        del tokens, context, images, teacher, ref_images, raw
    print(json.dumps({'extraction_complete': len(rows)}), flush=True)


def compress():
    manifest = make_manifest()
    torch.set_num_threads(4)
    torch.manual_seed(20261110)
    training = []
    for row in manifest['episodes']:
        if row['split'] == 'fit':
            raw = torch.load(CACHE / (row['id'] + '_raw.pt'), weights_only=False, map_location='cpu')
            training.append(raw['teacher'][REF].float().flatten(0, 1))
    X = torch.cat(training).to('cuda')
    mean = X.mean(0)
    _, singular, vectors = torch.pca_lowrank(X - mean, q=40, center=False, niter=4)
    basis = vectors[:, :32]
    scale = (singular[:32] / (len(X) - 1)**.5).clamp_min(.25)
    transform = {'mean': mean.cpu(), 'basis': basis.cpu(), 'scale': scale.cpu(),
                 'fit_source': 'training-reference-only', 'input_features': 1024, 'output_features': 32,
                 'explained_variance_fraction': float(singular[:32].square().sum() / (X - mean).square().sum())}
    torch.save(transform, CACHE / 'pca.pt')
    hashes = {}
    for row in manifest['episodes']:
        raw = torch.load(CACHE / (row['id'] + '_raw.pt'), weights_only=False, map_location='cpu')
        for key in ('teacher', 'seed_teacher'):
            raw[key] = (((raw[key].float().to('cuda') - mean) @ basis) / scale).cpu()
        raw['teacher'] = raw['teacher'].reshape(13, HEIGHT // PATCH, WIDTH // PATCH, 32)
        dest = CACHE / (row['id'] + '.pt')
        torch.save(raw, dest)
        hashes[str(dest)] = sha256(dest)
    (REPORT / 'cache_identity.json').write_text(json.dumps({'cache_sha256': hashes,
        'pca_sha256': sha256(CACHE / 'pca.pt'), 'pca_explained_variance': transform['explained_variance_fraction'],
        'manifest_sha256': sha256(REPORT / 'data_manifest.json'), 'extractor_sha256': sha256(__file__)}, indent=2) + '\n')
    print(json.dumps({'compressed_episodes': len(hashes), 'pca_variance': transform['explained_variance_fraction']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['manifest', 'extract', 'compress'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'manifest': make_manifest()
    elif args.command == 'extract': extract(args.limit)
    else: compress()
