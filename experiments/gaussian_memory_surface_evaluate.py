"""Saved-state reads, fixed interventions and cross-view surface diagnostics."""
import hashlib
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_surface_data import OUT, SEEDS, VARIANTS, model_path, checked, load_record
from models.gaussian_memory import seeds_to_device, detached_memory
from models.gaussian_memory_surface import SurfaceMemory
from models.gaussian_memory_projection import projected_inputs


def metrics(pred, log_target, valid, hidden):
    error = (pred - log_target).square()
    def avg(value, mask):
        per_query = (value * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        return float(per_query[mask.sum(-1) > 0].mean()) if (mask.sum(-1) > 0).any() else None
    depth, target = pred.exp(), log_target.exp()
    assert torch.isfinite(depth).all()
    return {'log_mse': avg(error, valid), 'hidden_log_mse': avg(error, valid & hidden),
            'visible_log_mse': avg(error, valid & ~hidden),
            'rmse_m': float(torch.sqrt(((depth - target).square() * valid).sum(-1) / valid.sum(-1)).mean()),
            'absrel': avg((depth - target).abs() / target, valid),
            'valid_patches': int(valid.sum()), 'samples': len(pred)}


def world_points(depth_m, c2w, K, unit):
    y, x = torch.meshgrid(torch.arange(15, device=K.device) + .5,
                          torch.arange(20, device=K.device) + .5, indexing='ij')
    pixels = torch.stack([x, y, torch.ones_like(x)], -1).reshape(300, 3)
    ray = torch.einsum('bij,pj->bpi', torch.linalg.inv(K), pixels)
    camera = ray * depth_m[..., None]
    return torch.einsum('bij,bpj->bpi', c2w[:, :3, :3], camera) + c2w[:, None, :3, 3] * unit


def correspondence(target, valid, camera, K, unit):
    world = world_points(target.exp(), camera, K, unit); pairs = []; distance = []
    for a in range(len(target) - 1):
        b = a + 1
        cost = torch.cdist(world[a], world[b]).masked_fill(~(valid[a, :, None] & valid[b, None]), torch.inf)
        nearest = cost.argmin(-1); reverse = cost.argmin(0); ids = torch.arange(300, device=target.device)
        dist = cost[ids, nearest]
        tolerance = .1 + .03 * torch.minimum(target[a].exp(), target[b, nearest].exp())
        good = valid[a] & valid[b, nearest] & (reverse[nearest] == ids) & (dist < tolerance)
        for i in ids[good].tolist():
            pairs.append([a, i, b, int(nearest[i])]); distance.append(float(dist[i]))
    return world, pairs, distance


def consistency(pred, gt_world, pairs, camera, K, unit):
    if not pairs: return {'pairs': 0, 'residual_xyz_mse': None, 'raw_pair_distance_m': None}
    world = world_points(pred.exp(), camera, K, unit)
    p = torch.tensor(pairs, device=pred.device); a, i, b, j = p.T
    residual = world - gt_world
    return {'pairs': len(pairs), 'residual_xyz_mse': float((residual[a, i] - residual[b, j]).square().sum(-1).mean()),
            'raw_pair_distance_m': float((world[a, i] - world[b, j]).norm(dim=-1).mean())}


@torch.inference_mode()
def baseline(seeds, camera, K, unit, variant, variance):
    memory = {'means': seeds['seed_mean'], 'covars': seeds['seed_chol'] @ seeds['seed_chol'].transpose(-1, -2),
              'opacities': seeds['seed_mean'].new_full((256,), .7), 'features': seeds['seed_teacher']}
    g = projected_inputs(memory, camera, K)
    dist = g['distance2'] / variance if variant == 'seed_center' else g['mahalanobis']
    score = -.5 * dist + g['log_weight'][:, None]
    center = torch.einsum('bij,bnj->bni', camera[:, :3, :3].transpose(-1, -2), memory['means'][None] - camera[:, None, :3, 3])
    z = (center[..., 2].clamp_min(.01) * unit).log()
    weights = score.softmax(-1)
    out = (weights * z[:, None]).sum(-1)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


@torch.inference_mode()
def evaluate():
    plan = checked(); torch.set_num_threads(4); nets = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            net = SurfaceMemory(variant, plan['local_width'], plan['center_variance']).cuda().eval()
            net.load_state_dict(torch.load(model_path(variant, seed), map_location='cuda', weights_only=False)['model'])
            nets[(variant, seed)] = net
    dest = OUT / 'evaluation'; assert not dest.exists(); dest.mkdir()
    memory_dir = OUT / 'memories'; memory_dir.mkdir()
    # Independent state per scene; the reader receives no raw reference or sensor tensors.
    identities = {}; changes = []; swapped_sources = {}
    for index, meta in enumerate(plan['catalog']):
        d = load_record(meta); raw = d['raw']; seeds = seeds_to_device(raw, 'cuda')
        for (variant, seed), net in nets.items():
            if variant == 'local': continue
            memory = detached_memory(net.write(seeds))
            path = memory_dir / f'{meta["id"]}_{variant}_{seed}.pt'
            torch.save({'memory': memory, 'unit': raw['diagnostics']['unit_meters'], 'episode': meta['id'],
                        'model_sha256': sha256(model_path(variant, seed)), 'query_RGB_used': False,
                        'sensor_used': False, 'reference_frames': [0, 4, 8, 12]}, path)
            identities[str(path)] = sha256(path)
            cov0 = raw['seed_chol'] @ raw['seed_chol'].transpose(-1, -2)
            changes.append({'episode': meta['id'], 'variant': variant, 'seed': seed,
                            'center_mean_shift': float((memory['means'] - raw['seed_mean']).norm(dim=-1).mean()),
                            'covariance_trace_ratio': float((memory['covars'].diagonal(dim1=-2, dim2=-1).sum(-1) / cov0.diagonal(dim1=-2, dim2=-1).sum(-1)).mean()),
                            'opacity_mean': float(memory['opacities'].mean()),
                            'content_change_rms': float((memory['features'] - raw['seed_teacher']).square().mean().sqrt())})
        other = next(plan['catalog'][(index + k) % len(plan['catalog'])] for k in range(1, len(plan['catalog']))
                     if plan['catalog'][(index + k) % len(plan['catalog'])]['row']['scene'].split('/')[0] != meta['row']['scene'].split('/')[0])
        swapped_sources[meta['id']] = other['id']
    write_json(OUT / 'memory_identity.json', {'sha256': identities, 'changes': changes, 'swapped_sources': swapped_sources})
    rows, full_rows, consistency_rows, replay = [], [], [], []
    for meta in plan['catalog']:
        d = load_record(meta); raw, rec, target = d['raw'], d['rec'], d['target']
        keep = torch.tensor([r == 'evaluation' for r in rec['roles']]); query = rec['query'][keep]
        x = rec['input_tokens'][keep].cuda(); hidden = rec['mask'][keep].cuda()
        y = target['log_depth_m'][query].cuda(); valid = target['valid'][query].cuda()
        unit = raw['diagnostics']['unit_meters']; c2w = raw['c2w'][query].cuda(); K = raw['K'][query].cuda()
        uq = sorted(set(query.tolist())); clean_x = rec['teacher_tokens'][uq].cuda()
        clean_y = target['log_depth_m'][uq].cuda(); clean_v = target['valid'][uq].cuda()
        clean_c2w, clean_K = raw['c2w'][uq].cuda(), raw['K'][uq].cuda()
        gt_world, pairs, distances = correspondence(clean_y, clean_v, clean_c2w, clean_K, unit)
        base = {'episode': meta['id'], 'scene': meta['row']['scene'], 'split': meta['eval_split']}
        predictions = {}; full_predictions = {}; seeds = seeds_to_device(raw, 'cuda')
        for v in ('seed_center', 'seed_gaussian', 'train_mean'):
            pred = (torch.full_like(y, plan['training_log_depth_mean']) if v == 'train_mean' else
                    baseline(seeds, c2w, K, unit, v, plan['center_variance']))
            rows.append(dict(base, variant=v, seed=None, condition='normal', **metrics(pred, y, valid, hidden)))
            predictions[v] = pred.cpu()
        generator = torch.Generator(device='cpu').manual_seed(int(hashlib.sha256(meta['id'].encode()).hexdigest()[:8], 16))
        permutation = torch.randperm(256, generator=generator).cuda()
        for (variant, seed), net in nets.items():
            memory = None
            if variant != 'local':
                path = memory_dir / f'{meta["id"]}_{variant}_{seed}.pt'
                stored = torch.load(path, map_location='cuda', weights_only=False); memory = stored['memory']
                live = net.write(seeds)
                replay.append(dict(base, variant=variant, seed=seed,
                                   state_max_diff=max(float((memory[k] - live[k]).abs().max()) for k in memory)))
            conditions = ('normal',) if variant == 'local' else ('normal', 'removed', 'swapped', 'shuffled', 'initial_geometry', 'wrong_camera')
            for condition in conditions:
                current = memory; wrong_camera = None
                if condition == 'swapped':
                    path = memory_dir / f'{swapped_sources[meta["id"]]}_{variant}_{seed}.pt'
                    other = torch.load(path, map_location='cuda', weights_only=False)
                    ratio = other['unit'] / unit
                    current = dict(other['memory'], means=other['memory']['means'] * ratio, covars=other['memory']['covars'] * ratio**2)
                elif condition == 'shuffled':
                    current = dict(memory, features=memory['features'][permutation])
                elif condition == 'initial_geometry':
                    current = dict(memory, means=seeds['seed_mean'], covars=seeds['seed_chol'] @ seeds['seed_chol'].transpose(-1, -2))
                elif condition == 'wrong_camera':
                    wrong_camera = raw['c2w'][(query + 4) % 13].cuda()
                pred = net.read(x, current, c2w, K, unit, condition, memory_c2w=wrong_camera)
                rows.append(dict(base, variant=variant, seed=seed, condition=condition, **metrics(pred, y, valid, hidden)))
                name = f'{variant}_{seed}_{condition}'; predictions[name] = pred.cpu()
                if condition == 'normal':
                    full = net.read(clean_x, memory, clean_c2w, clean_K, unit)
                    full_rows.append(dict(base, variant=variant, seed=seed, condition='normal',
                                          **metrics(full, clean_y, clean_v, torch.zeros_like(clean_v))))
                    full_predictions[f'{variant}_{seed}'] = full.cpu()
                    consistency_rows.append(dict(base, variant=variant, seed=seed,
                        gt_pair_distance_mean_m=float(np.mean(distances)) if distances else None,
                        **consistency(full, gt_world, pairs, clean_c2w, clean_K, unit)))
        torch.save({'episode': meta['row'], 'split': meta['eval_split'], 'query': query, 'full_query': uq,
                    'mask': hidden.cpu(), 'valid': valid.cpu(), 'target': y.cpu(),
                    'predictions': predictions, 'full_predictions': full_predictions,
                    'correspondences': pairs, 'gt_correspondence_distances': distances,
                    'rgb': raw['rgb'], 'permutation': permutation.cpu()}, dest / (meta['id'] + '_maps.pt'))
        print(json.dumps({'evaluated': meta['id'], 'valid_patches': int(valid.sum()), 'surface_correspondences': len(pairs)}), flush=True)
    write_json(dest / 'scores.json', {'rows': rows, 'full_query': full_rows, 'consistency': consistency_rows,
                                    'memory_replay': replay, 'episodes': len(plan['catalog']), 'confirmation_claim': False})


if __name__ == '__main__': evaluate()
