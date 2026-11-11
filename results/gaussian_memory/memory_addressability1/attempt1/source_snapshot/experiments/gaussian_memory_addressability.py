"""Frozen memory provenance and query-slot routing; diagnostic labels only."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_visibility_data import (
    OUT as PARENT, CACHE, BASE_CACHE, EARLY, LATE, CONDITIONS, POLICIES, SEEDS)
from experiments.gaussian_memory_incremental_data import (
    sensor_target, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_data import merge_seeds
from experiments.gaussian_memory_state_data import memory_prior
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_bound_evaluate import predict
from models.gaussian_memory_incremental import concatenate_observations
from models.gaussian_memory_bound_read import BoundMemoryAdapter
from models.gaussian_memory_vggt_adapter import position_grid

OUT = ROOT / 'results/gaussian_memory/memory_addressability1'
TRAIN = ROOT / 'results/gaussian_memory/history_visibility_train1'
KINDS = ('content', 'soft', 'soft_center')
OLD = ('content_static', 'content_dynamic', 'soft_dynamic', 'center_dynamic')
SOURCE_FRAMES = EARLY + LATE


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def prepare():
    assert read(TRAIN / 'phase_completion.json')['experiment_complete']
    parent = read(PARENT / 'plan.json'); trained = read(TRAIN / 'plan.json')
    inputs = {}; records = {}; models = []
    for meta in parent['catalog']:
        rid = meta['id']; marker = PARENT / 'inputs' / f'{rid}.json'
        rec = read(marker); records[str(marker)] = sha256(marker)
        for name, digest in rec['files'].items():
            inputs[str(CACHE / rid / name)] = digest
        for name in ('initial.pt', 'updates.pt'):
            inputs[str(BASE_CACHE / rid / name)] = parent['inputs'][rid]['files'][name]
        for f in SOURCE_FRAMES + (11,):
            frame = meta['row']['frames'][f]
            inputs[frame['depth_path']] = frame['depth_sha256']
    for family in OLD:
        kind = 'soft_center' if family == 'center_dynamic' else family.split('_')[0]
        for seed in SEEDS:
            models.append(dict(parent['checkpoints'][f'{family}_{seed}'], family=family, kind=kind, seed=seed))
    for kind, family in zip(KINDS, ('new_content', 'new_soft', 'new_center')):
        for seed in SEEDS:
            path = TRAIN / f'{kind}_{seed}/step_2000.pt'
            status = read(path.parent / 'status.json'); assert status['complete'] and status['steps'] == 2000
            models.append({'path': str(path), 'sha256': status['checkpoint_sha256'],
                           'family': family, 'kind': kind, 'seed': seed})
    for path, digest in inputs.items(): assert sha256(path) == digest, path
    for model in models: assert sha256(model['path']) == model['sha256']
    names = ['experiments/gaussian_memory_addressability.py',
             'experiments/gaussian_memory_data.py', 'experiments/gaussian_memory_incremental_data.py',
             'experiments/gaussian_memory_state_data.py', 'experiments/gaussian_memory_view_data.py',
             'experiments/gaussian_memory_bound_evaluate.py', 'models/gaussian_memory_incremental.py',
             'models/gaussian_memory_bound_read.py', 'models/gaussian_memory_soft_read.py',
             'models/gaussian_memory_vggt_adapter.py']
    plan = {'experiment': 'memory_addressability1', 'catalog': parent['catalog'], 'models': models,
            'inputs': inputs, 'input_records': records, 'center_variance': parent['center_variance'],
            'backbone_identity': trained['backbone_identity'],
            'parents': {str(p): sha256(p) for p in [PARENT / 'plan.json', TRAIN / 'plan.json',
                       PARENT / 'analysis.json', TRAIN / 'analysis.json', TRAIN / 'method_decision.json']},
            'source_sha256': {n: sha256(ROOT / n) for n in names},
            'document_sha256': sha256(ROOT / 'docs/gaussian_memory_addressability_plan.md'),
            'seeds': list(SEEDS), 'conditions': list(CONDITIONS), 'policies': list(POLICIES),
            'source_frames': list(SOURCE_FRAMES), 'query': 11, 'relative_depth_tolerance': .1,
            'primary': 'persistent/history, fixed query ROI: provenance retention and same-QK soft vs content/center routing',
            'label': 'Weighted member fractions of valid query pixels with an unmasked source depth correspondence; diagnostic only',
            'attention': 'actual post-global[11] Q/K, four heads; query 11 direct read only',
            'comparisons': 'all 21 frozen readers with all three priors, plus prior-only controls; no output intervention',
            'inference': '5000 paired place/seed bootstrap, seed 20261111; no fresh confirmation',
            'expected_states': 48, 'expected_qk_records': 1512,
            'new_training_steps': 0, 'new_download_bytes': 0, 'fresh_confirmation': False}
    path = OUT / 'plan.json'
    if path.exists(): assert read(path) == plan
    else: write_json(path, plan)
    for n, digest in plan['source_sha256'].items():
        dst = OUT / 'source_snapshot' / n; dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists(): assert sha256(dst) == digest
        else: shutil.copy2(ROOT / n, dst)
    write_json(OUT / 'input_checks.json', {'complete': True, 'files': len(inputs), 'models': len(models),
        'parent_records': records, 'source_files': plan['source_sha256']})
    return plan


def patch_grid(device):
    y, x = torch.meshgrid(torch.arange(210, device=device), torch.arange(280, device=device), indexing='ij')
    return y, x, ((y // 14) * 20 + x // 14).flatten()


@torch.no_grad()
def correspondences(meta, masks, persistent):
    """Unique (query pixel, source patch) labels using source and query sensor depths."""
    row = meta['row']; device = 'cuda'; n = 2400
    cameras = torch.tensor(row['c2w_metric'], dtype=torch.float64, device=device)
    K = torch.tensor(row['K'], dtype=torch.float64, device=device)
    target, valid = sensor_target(row['frames'][11]); target = target.to(device).double().flatten(); valid = valid.to(device).flatten()
    y, x, patch = patch_grid(device)
    uvh = torch.stack((x + .5, y + .5, torch.ones_like(x)), -1).double().reshape(-1, 3)
    pair_codes = []
    for fi, f in enumerate(SOURCE_FRAMES):
        d, ok = sensor_target(row['frames'][f]); d, ok = d.to(device).double().flatten(), ok.to(device).flatten()
        if persistent and f in LATE: ok = ok & ~masks[f][patch]
        xyz = ((uvh[ok] @ torch.linalg.inv(K[f]).T) * d[ok, None])
        world = xyz @ cameras[f, :3, :3].T + cameras[f, :3, 3]
        local = (world - cameras[11, :3, 3]) @ cameras[11, :3, :3]
        projected = local @ K[11].T; uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-12)
        inside = (torch.isfinite(local).all(-1) & (local[:, 2] > 1e-4)
                  & (uv[:, 0] >= 0) & (uv[:, 0] < 280) & (uv[:, 1] >= 0) & (uv[:, 1] < 210))
        ij = uv[inside].floor().long(); qp = ij[:, 1] * 280 + ij[:, 0]
        sp = patch[ok][inside] + fi * 300
        matched = valid[qp] & ((local[inside, 2] - target[qp]).abs() <= .1 * target[qp])
        pair_codes.append(qp[matched] * n + sp[matched])
    codes = torch.unique(torch.cat(pair_codes)); qp, sp = codes // n, codes % n
    early = torch.zeros(58800, dtype=torch.bool, device=device); late = early.clone()
    early[qp[sp < 1200]] = True; late[qp[sp >= 1200]] = True
    exclusive = early & ~late
    counts = torch.bincount(patch[valid], minlength=300).float()
    matrices = {}
    for label, keep in [('all', torch.ones_like(qp, dtype=torch.bool)), ('early_exclusive', exclusive[qp])]:
        matrix = torch.bincount(patch[qp[keep]] * n + sp[keep], minlength=300 * n).reshape(300, n).float()
        matrices[label] = matrix / counts.clamp_min(1)[:, None]
    return {'pairs': torch.stack((qp, sp)).to(torch.int32), 'matrix': matrices, 'valid_counts': counts,
            'exclusive': exclusive, 'query_valid': valid, 'query_patch': patch}


def slot_labels(obs, memory, labels, policy):
    means, chol, near, weights = merge_seeds(obs['means'], obs['covars'], obs['valid'])
    feature = (weights[..., None] * obs['features'][near]).sum(1)
    errors = {k: float((value - memory[k]).abs().max())
              for k, value in [('means', means), ('covars', chol @ chol.transpose(-1, -2)), ('features', feature)]}
    for key, value in errors.items():
        scale = max(1., float(memory[key].abs().max()))
        assert value <= 2e-5 * scale, (key, value, scale)
    offset = 0 if policy == 'history' else 1200
    assert len(obs['means']) == 2400 - offset
    expected_frame = torch.tensor(SOURCE_FRAMES[0 if policy == 'history' else 4:], device='cuda').repeat_interleave(300)
    assert torch.equal(obs['source_frame'], expected_frame)
    global_near = near + offset
    admitted = torch.zeros(2400, dtype=torch.bool, device='cuda'); admitted[offset:] = obs['valid']
    retained = torch.zeros_like(admitted); retained[global_near.flatten()] = True
    assert not (retained & ~admitted).any()
    universe = torch.zeros_like(admitted); universe[offset:] = True
    qp, sp = labels['pairs'].long(); coverage = {}
    for stage, keep in [('raw', universe), ('admitted', admitted), ('retained', retained)]:
        hit = torch.zeros(58800, dtype=torch.bool, device='cuda'); hit[qp[keep[sp]]] = True
        coverage[stage] = hit
    utility = {label: (matrix[:, global_near] * weights[None]).sum(-1) for label, matrix in labels['matrix'].items()}
    return {'near': global_near, 'weights': weights, 'admitted': admitted, 'retained': retained,
            'utility': utility, 'coverage': coverage, 'reconstruction_max_abs': errors}


def route_scores(attention, utility, counts, roi, label_name, identity):
    """attention [heads, query patches, slots]; unconditional pixel-weighted scores."""
    hit = utility > 0
    value = (attention * utility).sum(-1)
    mass = (attention * hit).sum(-1)
    best = utility.max(-1).values; uniform = utility.mean(-1)
    rows = []
    for region, selected in [('roi', roi), ('all', torch.ones_like(roi))]:
        weight = counts * selected; denom = float(weight.sum()); assert denom > 0
        supported = (best > 0) & selected & (counts > 0)
        support_weight = counts * supported; support_denom = float(support_weight.sum())
        rows.append(dict(identity, label=label_name, region=region,
            score=float((value.mean(0) * weight).sum() / denom),
            score_by_head=((value * weight).sum(-1) / denom).tolist(),
            uniform_score=float((uniform * weight).sum() / denom),
            best_slot_score=float((best * weight).sum() / denom),
            support_patch_pixels=support_denom, total_valid_pixels=denom,
            supported_patch_count=int(supported.sum()),
            hit_mass_on_supported=float((mass.mean(0) * support_weight).sum() / support_denom) if support_denom else None,
            uniform_hit_mass_on_supported=float((hit.float().mean(-1) * support_weight).sum() / support_denom) if support_denom else None))
    return rows


@torch.no_grad()
def run_clip(meta, backbone, adapters, plan):
    rid = meta['id']; marker = OUT / 'clips' / f'{rid}.json'
    if marker.exists():
        rec = read(marker); assert rec['complete'] and rec['plan_sha256'] == sha256(OUT / 'plan.json')
        assert sha256(rec['artifact_path']) == rec['artifact_sha256']; return rec
    start = time.monotonic()
    initial = device_tree(load(BASE_CACHE / rid / 'initial.pt'))
    updates = {'full': device_tree(load(BASE_CACHE / rid / 'updates.pt')),
               'persistent': device_tree(load(CACHE / rid / 'observations_persistent.pt'))}
    memories = device_tree(load(CACHE / rid / 'states.pt'))
    masks = load(CACHE / rid / 'masks.pt')['patch_masks'].cuda()
    samples = {c: device_tree(load(CACHE / rid / f'q11_c{c}.pt')) for c in range(3)}
    roi = masks[11]; labels = {c: correspondences(meta, masks, c == 'persistent') for c in updates}
    bindings = {}; provenance = []
    for state_key, memory in memories.items():
        condition, policy = state_key.split('/')
        parts = ([initial['observation']] if policy == 'history' else []) + [updates[condition][f] for f in LATE]
        obs = concatenate_observations(parts)
        bindings[state_key] = slot_labels(obs, memory, labels[condition], policy)
        binding = bindings[state_key]; label = labels[condition]
        for region, region_mask in [('roi', roi[label['query_patch']]), ('all', torch.ones(58800, dtype=torch.bool, device='cuda'))]:
            valid = label['query_valid'] & region_mask
            for subset, mask in [('all', valid), ('early_exclusive', valid & label['exclusive'])]:
                den = int(valid.sum()); pop = int(mask.sum())
                counts = {stage: int((covered & mask).sum()) for stage, covered in binding['coverage'].items()}
                provenance.append({'id': rid, 'place': meta['place'], 'state': state_key, 'region': region, 'label': subset,
                    'valid_pixels': den, 'subset_pixels': pop, **{k + '_pixels': v for k, v in counts.items()},
                    **{k + '_coverage': v / den for k, v in counts.items()},
                    'admitted_over_raw': counts['admitted'] / counts['raw'] if counts['raw'] else None,
                    'retained_over_admitted': counts['retained'] / counts['admitted'] if counts['admitted'] else None,
                    'retained_over_raw': counts['retained'] / counts['raw'] if counts['raw'] else None,
                    'admitted_patch_count': int(binding['admitted'].sum()), 'retained_patch_count': int(binding['retained'].sum()),
                    'early_slots': int((binding['near'] < 1200).any(-1).sum()),
                    'early_mixture_weight': float((binding['weights'] * (binding['near'] < 1200)).sum(-1).mean())})
    tokens = {}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for c, sample in samples.items():
            tokens[c] = backbone.aggregator.global_blocks[11](sample['prefix'].reshape(1, -1, 1024),
                           pos=position_grid('cuda').reshape(1, -1, 2)).reshape(1, 4, 305, 1024)
    rows = []; saved_qk = []; checks = []; priors_saved = {}
    for c, sample in samples.items():
        condition = 'persistent' if c == 2 else 'full'; label = labels[condition]
        for policy in POLICIES:
            state_key = condition + '/' + policy; memory = memories[state_key]; binding = bindings[state_key]
            priors = {kind: memory_prior(memory, sample, kind, plan['center_variance']) for kind in KINDS}
            priors_saved[state_key] = {k: v[-1, :, :256].cpu() for k, v in priors.items()}
            identity = {'id': rid, 'place': meta['place'], 'condition': CONDITIONS[c], 'policy': policy}
            for kind, prior in priors.items():
                attn = prior[-1, :, :256].softmax(-1)[None]
                for name, utility in binding['utility'].items():
                    rows += route_scores(attn, utility, label['valid_counts'], roi, name,
                        dict(identity, family='prior_only', seed=None, prior=kind, actual=True))
            for spec, adapter in adapters:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    query = adapter.q(adapter.norm(tokens[c][:, :, 5:])).reshape(1, -1, 4, 16).transpose(1, 2)
                    content = adapter.content_norm(memory['features'])
                    key = adapter.k(content).reshape(1, 256, 4, 16).transpose(1, 2)
                    value = adapter.v(content).reshape(1, 256, 4, 16).transpose(1, 2)
                q = query[0, :, -300:]; k = key[0]; logits = q.float() @ k.float().transpose(-1, -2) / 4
                saved_qk.append(dict(identity, family=spec['family'], seed=spec['seed'], kind=spec['kind'],
                    q=q.cpu(), k=k.cpu()))
                for kind, prior in priors.items():
                    attention = (logits + prior[-1, :, :256]).softmax(-1)
                    for name, utility in binding['utility'].items():
                        rows += route_scores(attention, utility, label['valid_counts'], roi, name,
                            dict(identity, family=spec['family'], seed=spec['seed'], prior=kind, actual=(kind == spec['kind'])))
                    if kind == spec['kind'] and rid == 'fidelity_00':
                        k_null = torch.cat((key, key.new_zeros(1, 4, 1, 16)), 2)
                        v_null = torch.cat((value, value.new_zeros(1, 4, 1, 16)), 2)
                        reference = F.scaled_dot_product_attention(query[:, :, -300:], k_null, v_null,
                            attn_mask=prior[-1].reshape(1, 1, 300, 257))[0].float()
                        manual = attention @ value[0].float()
                        rms = float((manual - reference).square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-8))
                        assert rms < .03, rms
                        checks.append(dict(identity, family=spec['family'], seed=spec['seed'],
                                           sdpa_relative_rms=rms, sdpa_max_abs=float((manual - reference).abs().max())))
    sentinel = None
    if rid == 'fidelity_00':
        spec, adapter = next(x for x in adapters if x[0]['family'] == 'soft_dynamic' and x[0]['seed'] == SEEDS[0])
        old_method = adapter.content_residual; captured = {}
        def capture(x, payload):
            captured['tokens'] = x.detach().clone()
            return old_method(x, payload)
        adapter.content_residual = capture
        mem = memories['persistent/history']; prior = memory_prior(mem, samples[2], 'soft', plan['center_variance'])
        output = predict(backbone, adapter, samples[2], mem['features'], prior)
        adapter.content_residual = old_method
        parent = read(PARENT / 'evaluation' / f'{rid}.json'); saved = load(parent['prediction_path'])
        index = saved['keys'].index([2, 'soft_dynamic', SEEDS[0], 'history'])
        output_error = float((output.cpu() - saved['prediction_native'][index]).abs().max())
        token_error = float((captured['tokens'] - tokens[2]).abs().max())
        assert token_error == 0 and output_error < 1e-4, (token_error, output_error)
        sentinel = {'post_global_token_max_abs': token_error, 'native_output_max_abs': output_error,
                    'parent_prediction_sha256': parent['prediction_sha256']}
    artifact = OUT / 'artifacts' / f'{rid}.pt'
    def cpu_tree(obj):
        if torch.is_tensor(obj): return obj.cpu()
        if isinstance(obj, dict): return {k: cpu_tree(v) for k, v in obj.items()}
        if isinstance(obj, list): return [cpu_tree(v) for v in obj]
        return obj
    save_tensor(artifact, cpu_tree({'labels': labels, 'bindings': bindings, 'qk': saved_qk,
                                   'priors': priors_saved, 'roi': roi}))
    rec = {'complete': True, 'id': rid, 'place': meta['place'], 'rows': rows, 'provenance': provenance,
           'reconstruction': {k: v['reconstruction_max_abs'] for k, v in bindings.items()},
           'attention_checks': checks, 'sentinel': sentinel, 'qk_records': len(saved_qk),
           'artifact_path': str(artifact), 'artifact_sha256': sha256(artifact),
           'plan_sha256': sha256(OUT / 'plan.json'), 'seconds': time.monotonic() - start}
    write_json(marker, rec)
    print(json.dumps({'completed': rid, 'qk_records': len(saved_qk), 'rows': len(rows), 'seconds': rec['seconds']}), flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'run'))
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args(); torch.set_num_threads(4)
    if a.stage == 'prepare':
        plan = prepare(); print(json.dumps({'prepared': plan['experiment'], 'models': len(plan['models'])})); return
    plan = read(OUT / 'plan.json')
    for name, digest in plan['source_sha256'].items(): assert sha256(ROOT / name) == digest
    backbone = load_backbone(); adapters = []
    for spec in plan['models']:
        assert sha256(spec['path']) == spec['sha256']
        adapter = BoundMemoryAdapter().cuda().eval(); adapter.load_state_dict(load(spec['path'])['model'])
        adapters.append((spec, adapter))
    start = time.monotonic(); ids = []
    for meta in plan['catalog'][a.shard::a.shards]: ids.append(run_clip(meta, backbone, adapters, plan)['id'])
    write_json(OUT / f'worker_{a.shard}.json', {'complete': True, 'ids': ids, 'seconds': time.monotonic() - start,
               'peak_GiB': torch.cuda.max_memory_allocated() / 2**30, 'gpu': torch.cuda.get_device_name(0)})


if __name__ == '__main__': main()
