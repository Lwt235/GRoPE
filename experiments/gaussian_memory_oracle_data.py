"""Reference-only sensor moments with frozen slot membership and content."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_surface_data import OUT as OLD, CACHE as TARGETS, SEEDS, checked as old_checked, load_record as old_load, sensor_path, model_path as old_model_path
from experiments.round5_rgbd_data import load_sensor_depth
from models.gaussian_memory import seeds_to_device

OUT = ROOT / 'results/gaussian_memory/reference_geometry1'
CACHE = DATA / 'reference_geometry_v1'
VARIANTS = ('fixed', 'center', 'gaussian')
REF = [0, 4, 8, 12]


def model_path(variant, seed, regime='sensor', initial=False):
    if regime == 'predicted' or variant == 'local': return old_model_path(variant, seed, initial)
    return OUT / f'{variant}_{seed}' / ('initial.pt' if initial else 'step_2000.pt')


def reference_moments(depths, c2w, K, unit):
    """Only four reference depth maps and their calibration are accepted."""
    assert depths.shape == (4, 210, 280) and c2w.shape == (4, 4, 4)
    depths, c2w, K = depths.double(), c2w.double(), K.double()
    valid = torch.isfinite(depths) & (depths > .1) & (depths < 10.)
    y, x = torch.meshgrid(torch.arange(210, device=depths.device, dtype=torch.float64)+.5,
                          torch.arange(280, device=depths.device, dtype=torch.float64)+.5, indexing='ij')
    pixels = torch.stack([x, y, torch.ones_like(x)], -1)
    ray = torch.einsum('vij,hwj->vhwi', torch.linalg.inv(K), pixels)
    points = torch.einsum('vij,vhwj->vhwi', c2w[:, :3, :3], ray * (depths/unit)[..., None]) + c2w[:, None, None, :3, 3]
    points = points.reshape(4, 15, 14, 20, 14, 3).permute(0, 1, 3, 2, 4, 5).reshape(1200, 196, 3)
    weights = valid.reshape(4, 15, 14, 20, 14).permute(0, 1, 3, 2, 4).reshape(1200, 196).double()
    count = weights.sum(-1); weights = weights / count[:, None].clamp_min(1)
    points = torch.where(torch.isfinite(points), points, torch.zeros_like(points))
    mean = (points * weights[..., None]).sum(1)
    delta = points - mean[:, None]
    cov = torch.einsum('npi,npj,np->nij', delta, delta, weights) + torch.eye(3, device=depths.device) * .005**2
    return mean.float(), cov.float(), count >= 98, (count / 196).float()


def aggregate(means, covars, near, weights):
    means, covars, weights = means.double(), covars.double(), weights.double()
    center = (weights[..., None] * means[near]).sum(1)
    delta = means[near] - center[:, None]
    cov = (weights[..., None, None] * (covars[near] + delta[..., :, None]*delta[..., None, :])).sum(1)
    cov = cov + torch.eye(3, device=means.device) * 1e-6
    return center.float(), torch.linalg.cholesky(cov).float()


def prepare():
    old = old_checked(True); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'plan.json').exists(); identities = dict(old['input_sha256']); rows = []
    for meta in old['catalog']:
        raw = torch.load(meta['raw'], map_location='cpu', weights_only=False)
        direct = torch.load(meta['direct'], map_location='cuda', weights_only=False)['memory']
        binding_path = OUT/'bindings'/(meta['id']+'.pt'); binding = torch.load(binding_path, map_location='cuda', weights_only=False)
        assert binding['diagnostics']['mean_max_diff'] == 0 and binding['diagnostics']['chol_max_diff'] == 0
        assert binding['raw_sha256'] == sha256(meta['raw']) and binding['direct_sha256'] == sha256(meta['direct'])
        sources = []; depths = []
        for q in REF:
            path, gap = sensor_path(meta['row']['frames'][q]); depth = load_sensor_depth(path)
            depths.append(np.asarray(Image.fromarray(depth).resize((280, 210), Image.Resampling.NEAREST)).copy())
            sources.append({'frame': q, 'path': str(path), 'sha256': sha256(path), 'time_gap_raw': gap})
            identities[str(path)] = sha256(path)
        c2w = raw['c2w'][REF].cuda(); K = (raw['K'][REF] * torch.tensor([14.,14.,1.])[:, None]).cuda()
        unit = raw['diagnostics']['unit_meters']; depths = torch.tensor(np.stack(depths), device='cuda')
        mean, cov, replace, fraction = reference_moments(depths, c2w, K, unit)
        hybrid_mean = torch.where(replace[:, None], mean, direct['means'])
        hybrid_cov = torch.where(replace[:, None, None], cov, direct['covars'])
        near, weights = binding['near'], binding['weights']
        new_mean, new_chol = aggregate(hybrid_mean, hybrid_cov, near, weights)
        seed_fraction = (weights * replace[near]).sum(-1)
        diag = {'id': meta['id'], 'scene': meta['row']['scene'], 'valid_sensor_pixel_fraction': float(fraction.mean()),
                'replaced_patch_fraction': float(replace.float().mean()),
                'seed_replaced_weight_mean': float(seed_fraction.mean()), 'seed_replaced_weight_min': float(seed_fraction.min()),
                'seed_any_replacement_fraction': float((seed_fraction > 0).float().mean()),
                'seed_center_displacement_m_mean': float((new_mean-raw['seed_mean'].cuda()).norm(dim=-1).mean()*unit),
                'seed_cov_trace_ratio': float(((new_chol @ new_chol.transpose(-1,-2)).diagonal(dim1=-2,dim2=-1).sum(-1) /
                                               (raw['seed_chol'].cuda() @ raw['seed_chol'].cuda().transpose(-1,-2)).diagonal(dim1=-2,dim2=-1).sum(-1)).mean()),
                'sensor_sources': sources}
        path = CACHE/(meta['id']+'.pt'); assert not path.exists()
        torch.save({'seed_mean': new_mean.cpu(), 'seed_chol': new_chol.cpu(), 'patch_sensor_mean': mean.cpu(),
                    'patch_sensor_cov': cov.cpu(), 'replace': replace.cpu(), 'valid_pixel_fraction': fraction.cpu(),
                    'seed_replaced_weight': seed_fraction.cpu(), 'reference_frames': REF, 'sources': sources,
                    'raw_sha256': sha256(meta['raw']), 'binding_sha256': sha256(binding_path),
                    'query_RGB_used': False, 'query_depth_used': False, 'reference_sensor_used': True,
                    'slot_content_changed': False, 'diagnostics': diag}, path)
        for path0 in (path, binding_path, Path(meta['direct'])): identities[str(path0)] = sha256(path0)
        rows.append(diag); print(json.dumps({'geometry_cached': meta['id'], 'coverage': diag['seed_replaced_weight_mean']}), flush=True)
    for v in ('local',)+VARIANTS:
        for s in SEEDS:
            for initial in (False, True):
                path = old_model_path(v, s, initial); identities[str(path)] = sha256(path)
    sources = dict(old['source_sha256'])
    sources.update({str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob('experiments/gaussian_memory_oracle_*.py') if p.stem.rsplit('_',1)[-1] in ('data','train','evaluate','run','probe')})
    plan = {**{k: old[k] for k in ('catalog','training_ids','primary_places','steps','batch','lr','weight_decay','center_variance','local_width','gradient_clip','training_log_depth_mean')},
            'seeds': SEEDS, 'variants': VARIANTS, 'new_fits': 9, 'reused_fits': 12,
            'source_sha256': sources, 'input_sha256': identities, 'old_plan_sha256': sha256(OLD/'plan.json'),
            'document_sha256': sha256(ROOT/'docs/gaussian_memory_reference_geometry_plan.md'),
            'reference_sensor_input': True, 'query_sensor_input': False, 'query_RGB_in_memory': False,
            'confirmation_claim': False, 'strict_upper_bound_claim': False, 'fixed_compression_membership': True}
    write_json(OUT/'plan.json', plan); write_json(OUT/'geometry_identity.json', {'rows': rows, 'episodes': len(rows)})


def checked(inputs=False):
    plan = json.loads((OUT/'plan.json').read_text())
    for p, digest in plan['source_sha256'].items(): assert sha256(ROOT/p) == digest, p
    assert sha256(ROOT/'docs/gaussian_memory_reference_geometry_plan.md') == plan['document_sha256']
    if inputs:
        for p, digest in plan['input_sha256'].items(): assert sha256(p) == digest, p
    return plan


def load_record(meta):
    d = old_load(meta)
    d['sensor'] = torch.load(CACHE/(meta['id']+'.pt'), map_location='cpu', weights_only=False)
    return d


def input_seeds(record, regime, device='cuda'):
    result = seeds_to_device(record['raw'], device)
    if regime == 'sensor':
        for k in ('seed_mean','seed_chol'): result[k] = record['sensor'][k].to(device).float()
    return result


if __name__ == '__main__': prepare()
