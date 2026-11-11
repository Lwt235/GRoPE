"""Half original read, half correspondence-addressed values; frozen diagnostic."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.gaussian_memory_addressability import OUT as ADDRESS, PARENT, TRAIN, read, load
from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_visibility_data import CACHE as INPUT_CACHE, SEEDS
from experiments.gaussian_memory_incremental_data import write_json, save_tensor, device_tree
from experiments.gaussian_memory_state_data import memory_prior
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_bound_evaluate import predict
from experiments.gaussian_memory_incremental_evaluate import score
from models.gaussian_memory_bound_read import BoundMemoryAdapter

OUT = ROOT / 'results/gaussian_memory/address_intervention1'
CACHE = DATA / 'address_intervention_v1'
CONDITIONS = {0: 'full', 2: 'persistent'}
FAMILIES = ('soft_dynamic', 'new_soft')
POLICIES = ('recent', 'history')
REGIONS = ('supported_roi', 'history_only_roi', 'unrepresented_roi')
ALPHA = .5


def permutation(rid):
    seed = int.from_bytes(hashlib.sha256((rid + '/address_intervention_v1').encode()).digest()[:8], 'little')
    order = np.random.default_rng(seed).permutation(256); perm = np.empty(256, dtype=np.int64)
    perm[order] = np.roll(order, 1)
    assert np.unique(perm).size == 256 and (perm != np.arange(256)).all()
    return perm


def prepare():
    parent = read(ADDRESS / 'plan.json'); assert read(ADDRESS / 'phase_completion.json')['experiment_complete']
    records = {}; inputs = {}; references = {}
    for meta in parent['catalog']:
        rid = meta['id']; record = ADDRESS / 'clips' / f'{rid}.json'; r = read(record)
        records[str(record)] = sha256(record); inputs[r['artifact_path']] = r['artifact_sha256']
        old_inputs = read(PARENT / 'inputs' / f'{rid}.json')
        for name in ('states.pt', 'q11_c0.pt', 'q11_c2.pt'):
            inputs[str(INPUT_CACHE / rid / name)] = old_inputs['files'][name]
        references[rid] = {}
        for role, folder in [('old', PARENT), ('new', TRAIN)]:
            path = folder / 'evaluation' / f'{rid}.json'; record = read(path)
            records[str(path)] = sha256(path)
            inputs[record['prediction_path']] = record['prediction_sha256']
            references[rid][role] = {'record': str(path), 'predictions': record['prediction_path']}
    models = [m for m in parent['models'] if m['family'] in FAMILIES]
    assert len(models) == 6
    for path, digest in inputs.items(): assert sha256(path) == digest, path
    for spec in models: assert sha256(spec['path']) == spec['sha256']
    sources = ['experiments/gaussian_memory_address_intervention.py',
               'models/gaussian_memory_bound_read.py', 'models/gaussian_memory_vggt_adapter.py',
               'models/gaussian_memory_soft_read.py', 'experiments/gaussian_memory_state_data.py',
               'experiments/gaussian_memory_view_data.py', 'experiments/gaussian_memory_incremental_evaluate.py',
               'experiments/gaussian_memory_bound_evaluate.py', 'experiments/gaussian_memory_incremental_data.py']
    plan = {'experiment': 'address_intervention1', 'catalog': parent['catalog'], 'models': models,
            'inputs': inputs, 'records': records, 'references': references, 'center_variance': parent['center_variance'],
            'backbone_identity': parent['backbone_identity'], 'conditions': CONDITIONS,
            'seeds': list(SEEDS), 'families': list(FAMILIES), 'alpha': ALPHA,
            'permutations': {m['id']: permutation(m['id']).tolist() for m in parent['catalog']},
            'scope': 'query11 ROI direct memory read only; original three contexts, special tokens and other patches unchanged at insertion',
            'targets': ['all', 'early_exclusive'], 'teacher': 'existing source-correspondence labels, normalized per supported patch',
            'control': 'same support gate, target weight multiset and half mixture; permuted slots, not guaranteed equal residual norm',
            'expected_outputs': 1152, 'expected_original_replays': 288,
            'primary': 'persistent/history ROI log-MSE, correct_all vs original and sham_all, old Soft primary, new Soft sensitivity',
            'auxiliary_regions': 'defined by History retained support for both policies; all primary places retained',
            'source_sha256': {n: sha256(ROOT / n) for n in sources},
            'plan_document_sha256': sha256(ROOT / 'docs/gaussian_memory_address_intervention_plan.md'),
            'parent_plan_sha256': sha256(ADDRESS / 'plan.json'),
            'all_data_previously_used': True, 'query_truth_in_intervention': True,
            'new_training_steps': 0, 'new_download_bytes': 0, 'fresh_confirmation': False}
    path = OUT / 'plan.json'
    if path.exists(): assert read(path) == json.loads(json.dumps(plan))
    else: write_json(path, plan)
    for name, digest in plan['source_sha256'].items():
        dst = OUT / 'source_snapshot' / name; dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists(): assert sha256(dst) == digest
        else: shutil.copy2(ROOT / name, dst)
    write_json(OUT / 'input_checks.json', {'complete': True, 'files': len(inputs), 'records': len(records),
        'models': len(models), 'input_sha256': inputs, 'plan_sha256': sha256(path)})
    return plan


def arms(policy):
    return ['original', 'correct_all', 'sham_all'] + (['correct_history', 'sham_history'] if policy == 'history' else [])


def targets(artifact, condition, policy, perm):
    binding = artifact['bindings'][condition + '/' + policy]
    roi = artifact['roi']; result = {}
    for label, suffix in [('all', 'all'), ('early_exclusive', 'history')]:
        u = binding['utility'][label]; active = roi & (u.sum(-1) > 0)
        t = u / u.sum(-1, keepdim=True).clamp_min(1e-12)
        t = torch.where(active[:, None], t, 0.)
        wrong = t[:, perm]
        result['correct_' + suffix] = {'target': t, 'active': active, 'utility': u}
        result['sham_' + suffix] = {'target': wrong, 'active': active, 'utility': u}
    return result


def common_regions(artifact, condition):
    labels = artifact['labels'][condition]; history = artifact['bindings'][condition + '/history']
    roi = artifact['roi'][labels['query_patch']].bool()
    retained = history['coverage']['retained'].bool()
    return {'supported_roi': (roi & retained).reshape(210, 280),
            'history_only_roi': (roi & retained & labels['exclusive']).reshape(210, 280),
            'unrepresented_roi': (roi & ~retained).reshape(210, 280)}


def evaluate_score(prediction, sample, regions):
    row = score(prediction, sample)
    residual = prediction.float().cpu().clamp_min(1e-6).log() - sample['target'].float().cpu().log() + np.log(sample['metric_scale'])
    valid = sample['valid'].cpu()
    for name, region in regions.items():
        selected = region.cpu() & valid; row[name + '_pixels'] = int(selected.sum())
        row[name + '_log_mse'] = float(residual[selected].square().mean()) if selected.any() else None
    return row


class InterventionAdapter(BoundMemoryAdapter):
    def __init__(self):
        super().__init__(); self.spec = None; self.diagnostic = None; self.capture = False; self.captured = None

    def content_residual(self, tokens, payload):
        if self.spec is None:
            self.diagnostic = None
            return super().content_residual(tokens, payload)
        q = self.q(self.norm(tokens[:, :, 5:])).reshape(1, -1, 4, 16).transpose(1, 2)
        content = self.content_norm(payload['content'])
        k = self.k(content).reshape(1, 256, 4, 16).transpose(1, 2)
        v = self.v(content).reshape(1, 256, 4, 16).transpose(1, 2)
        k_null = torch.cat((k, k.new_zeros(1, 4, 1, 16)), 2)
        v_null = torch.cat((v, v.new_zeros(1, 4, 1, 16)), 2)
        prior = payload['log_prior'].reshape(1, 1, 1200, 257)
        value = F.scaled_dot_product_attention(q, k_null, v_null, attn_mask=prior, dropout_p=0.)
        target, active, utility = (self.spec[n] for n in ('target', 'active', 'utility'))
        original = value[:, :, -300:]
        with torch.autocast('cuda', enabled=False):
            target_value = torch.einsum('ps,bhsd->bhpd', target.float(), v.float())
            blend = (1 - ALPHA) * original.float() + ALPHA * target_value
            mixed = torch.where(active[None, None, :, None], blend, original.float()).to(value.dtype)
            logits = q[0, :, -300:].float() @ k[0].float().transpose(-1, -2) / 4
            # Native SDPA autocast also rounds the floating mask to BF16.
            attention = (logits + payload['log_prior'][-1, :, :256].to(torch.bfloat16).float()).softmax(-1)
            after = torch.where(active[None, :, None], (1 - ALPHA) * attention + ALPHA * target[None], attention)
            before_score = (attention * utility[None]).sum(-1).mean(0)
            after_score = (after * utility[None]).sum(-1).mean(0)
            delta = mixed.float() - original.float()
            self.diagnostic = {'active_patches': int(active.sum()),
                'route_before': before_score.cpu(), 'route_after': after_score.cpu(),
                'attention_L1': (after - attention).abs().sum(-1).mean(0).cpu(),
                'value_delta_rms': delta.square().mean((0, 1, 3)).sqrt().cpu(),
                'original_value_rms': original.float().square().mean((0, 1, 3)).sqrt().cpu(),
                'target_entropy': (-(target * target.clamp_min(1e-30).log()).sum(-1)).cpu(),
                'inactive_value_max_abs': float(delta[:, :, ~active].abs().max()) if (~active).any() else 0.}
        assert self.diagnostic['inactive_value_max_abs'] == 0
        if self.capture:
            self.captured = {'original_value': original.float().cpu(), 'values': v.float().cpu(),
                             'target': target.cpu(), 'active': active.cpu(), 'mixed': mixed.float().cpu()}
        updated = value.clone(); updated[:, :, -300:] = mixed
        assert torch.equal(updated[:, :, :-300], value[:, :, :-300])
        read = updated.transpose(1, 2).reshape(1, 4, 300, 64)
        return tokens + F.pad(self.out(read), (0, 0, 5, 0))


@torch.no_grad()
def run_clip(meta, backbone, adapters, plan):
    rid = meta['id']; marker = OUT / 'evaluation' / f'{rid}.json'
    if marker.exists():
        rec = read(marker); assert rec['complete'] and rec['plan_sha256'] == sha256(OUT / 'plan.json')
        assert sha256(rec['prediction_path']) == rec['prediction_sha256']; return rec
    start = time.monotonic(); address = read(ADDRESS / 'clips' / f'{rid}.json')
    artifact = device_tree(load(address['artifact_path'])); memories = device_tree(load(INPUT_CACHE / rid / 'states.pt'))
    refs = {key: load(value['predictions']) for key, value in plan['references'][rid].items()}
    rows = []; predictions = []; keys = []; diagnostics = []; captures = {}; replay = []
    target_records = {}; common_masks = {}; control_rows = []
    perm = torch.tensor(plan['permutations'][rid], device='cuda')
    for ci, condition in CONDITIONS.items():
        sample = load(INPUT_CACHE / rid / f'q11_c{ci}.pt'); regions = common_regions(artifact, condition)
        common_masks[condition] = {k: v.cpu() for k, v in regions.items()}
        # Background references are rescored on common masks, never called new predictions.
        for role in ('old', 'new'):
            for index, key in enumerate(refs[role]['keys']):
                c, family, seed, policy = key
                include = family in ('native', 'local', 'content_static', 'center_dynamic') if role == 'old' else family in ('new_content', 'new_center')
                if c != ci or not include: continue
                row = evaluate_score(refs[role]['prediction_native'][index], sample, regions)
                row.update(family=family, seed=seed, policy=policy, condition_name=condition,
                           place=meta['place'], arm='reference', reused=True, reference_role=role)
                control_rows.append(row)
        for policy in POLICIES:
            state = condition + '/' + policy; memory = memories[state]
            options = targets(artifact, condition, policy, perm)
            target_records[state] = {k: {n: v.cpu() for n, v in value.items()} for k, value in options.items()}
            prior = memory_prior(memory, sample, 'soft', plan['center_variance'])
            for spec, adapter in adapters:
                family, seed = spec['family'], spec['seed']; original_prediction = None
                for arm in arms(policy):
                    adapter.spec = None if arm == 'original' else options[arm]
                    adapter.capture = rid == 'fidelity_00' and seed == SEEDS[0] and family == FAMILIES[0]
                    prediction = predict(backbone, adapter, sample, memory['features'], prior)
                    key = [ci, family, seed, policy, arm]
                    if arm == 'original':
                        role = 'old' if family == 'soft_dynamic' else 'new'
                        index = refs[role]['keys'].index(key[:4]); baseline = refs[role]['prediction_native'][index]
                        difference = float((prediction - baseline).abs().max())
                        assert difference < 1e-4, (rid, key, difference)
                        replay.append({'key': key, 'max_abs': difference}); original_prediction = prediction
                    else:
                        d = adapter.diagnostic; diagnostics.append({'key': key, **d})
                        if not d['active_patches']: assert torch.equal(prediction, original_prediction), (key, 'empty support changed output')
                        if adapter.capture: captures['/'.join(map(str, key))] = adapter.captured
                    row = evaluate_score(prediction, sample, regions)
                    row.update(family=family, seed=seed, policy=policy, arm=arm,
                               condition_name=condition, place=meta['place'], reused=False)
                    rows.append(row); predictions.append(prediction); keys.append(key)
    assert len(rows) == 96 and len(replay) == 24
    folder = CACHE / rid; folder.mkdir(parents=True, exist_ok=True)
    pred_path = folder / 'predictions.pt'; diag_path = folder / 'diagnostics.pt'
    save_tensor(pred_path, {'prediction_native': torch.stack(predictions), 'keys': keys})
    save_tensor(diag_path, {'diagnostics': diagnostics, 'targets': target_records, 'regions': common_masks, 'captures': captures})
    rec = {'complete': True, 'id': rid, 'place': meta['place'], 'rows': rows, 'reference_rows': control_rows,
           'replays': replay, 'prediction_path': str(pred_path), 'prediction_sha256': sha256(pred_path),
           'diagnostic_path': str(diag_path), 'diagnostic_sha256': sha256(diag_path),
           'plan_sha256': sha256(OUT / 'plan.json'), 'seconds': time.monotonic() - start}
    write_json(marker, rec)
    print(json.dumps({'completed': rid, 'outputs': len(rows), 'replays': len(replay), 'seconds': rec['seconds']}), flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'run'))
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'prepare':
        plan = prepare(); print(json.dumps({'prepared': plan['experiment'], 'outputs': plan['expected_outputs']})); return
    plan = read(OUT / 'plan.json')
    for name, digest in plan['source_sha256'].items(): assert sha256(ROOT / name) == digest
    backbone = load_backbone(); adapters = []
    for spec in plan['models']:
        assert sha256(spec['path']) == spec['sha256']
        adapter = InterventionAdapter().cuda().eval().requires_grad_(False)
        adapter.load_state_dict(load(spec['path'])['model']); adapters.append((spec, adapter))
    start = time.monotonic(); ids = []
    for meta in plan['catalog'][args.shard::args.shards]: ids.append(run_clip(meta, backbone, adapters, plan)['id'])
    write_json(OUT / f'worker_{args.shard}.json', {'complete': True, 'ids': ids,
        'seconds': time.monotonic() - start, 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'gpu': torch.cuda.get_device_name(0)})


if __name__ == '__main__': main()
