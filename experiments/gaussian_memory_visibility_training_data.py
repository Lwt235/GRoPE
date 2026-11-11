"""Fixed identities and matched examples for task-matched visibility training."""
import copy
import json
import shutil

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_incremental_data import (
    CACHE as BASE_CACHE, SEEDS, QUERIES, depth_source, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_visibility_data import (
    OUT as PARENT, CACHE as EVAL_CACHE, EARLY, LATE, CONDITIONS, POLICIES)
from experiments.gaussian_memory_state_data import memory_prior

OUT = ROOT / 'results/gaussian_memory/history_visibility_train1'
CACHE = DATA / 'history_visibility_training_v1'
KINDS = ('content', 'soft', 'soft_center')
NEW_SOURCES = ('experiments/gaussian_memory_visibility_training_data.py',
               'experiments/gaussian_memory_visibility_training_cache.py',
               'experiments/gaussian_memory_visibility_training_fit.py')


def load_plan():
    return json.loads((OUT / 'plan.json').read_text())


def verify_sources(plan):
    for name, digest in plan['source_sha256'].items():
        assert sha256(ROOT / name) == digest, name


def prepare():
    parent = json.loads((PARENT / 'plan.json').read_text())
    assert json.loads((PARENT / 'phase_completion.json').read_text())['experiment_complete']
    old = json.loads((ROOT / 'results/gaussian_memory/state_read1/plan.json').read_text())
    lookup = {m['id']: m for m in old['catalog']}
    catalog = [copy.deepcopy(lookup[rid]) for rid in old['training_ids']]
    assert len(catalog) == 12
    assert not {m['place'] for m in catalog} & {m['place'] for m in parent['catalog']}
    inputs = {}; source_files = {}
    for meta in catalog:
        rid = meta['id']; path = BASE_CACHE / rid / 'identity.json'
        identity = json.loads(path.read_text()); files = {}
        for name in ['initial.pt', 'updates.pt'] + [f'q{q}_c{c}.pt' for q in QUERIES for c in (0, 1)]:
            files[name] = sha256(BASE_CACHE / rid / name)
            assert files[name] == identity['sha256'][name], (rid, name)
        scales = {str(d['window'][-1]): {'window': d['window'], 'scale': d['scale']}
                  for d in identity['diagnostics'] if d['kind'] == 'write'}
        assert set(scales) == set(map(str, LATE)) and all(v['scale'] is not None for v in scales.values())
        queries = {}
        for q in QUERIES:
            sample = torch.load(BASE_CACHE / rid / f'q{q}_c0.pt', map_location='cpu', weights_only=False)
            assert sample['metric_scale'] is not None
            queries[str(q)] = {'window': sample['input_frames'], 'scale': sample['metric_scale']}
            assert sample['input_frames'] == (list(EARLY) + [f for f in LATE if f < q])[-3:] + [q]
        for i, frame in enumerate(meta['row']['frames']):
            assert sha256(frame['rgb_path']) == frame['rgb_sha256']
            source_files[frame['rgb_path']] = frame['rgb_sha256']
            if i in EARLY:
                p = depth_source(frame)
                frame['depth_path'] = str(p)
                digest = sha256(p)
                if frame.get('depth_sha256'): assert frame['depth_sha256'] == digest
                frame['depth_sha256'] = digest; source_files[str(p)] = digest
        inputs[rid] = {'base_identity_sha256': sha256(path), 'files': files,
                       'write_scales': scales, 'queries': queries, 'unit_meters': identity['unit_meters']}
    reused = {}
    for meta in parent['catalog']:
        rid = meta['id']
        for name in ('inputs', 'evaluation'):
            p = PARENT / name / f'{rid}.json'; reused[str(p)] = sha256(p)
        record = json.loads((PARENT / 'inputs' / f'{rid}.json').read_text())
        for name, digest in record['files'].items():
            assert sha256(EVAL_CACHE / rid / name) == digest
        record = json.loads((PARENT / 'evaluation' / f'{rid}.json').read_text())
        assert sha256(record['prediction_path']) == record['prediction_sha256']
    pca = DATA / 'cache_v1/pca.pt'; source_files[str(pca)] = sha256(pca)
    sources = set(parent['source_sha256']) | set(NEW_SOURCES) | {
        'experiments/gaussian_memory_state_data.py', 'experiments/gaussian_memory_incremental_train.py'}
    plan = {'experiment': 'history_visibility_train1', 'training_catalog': catalog,
        'training_ids': old['training_ids'], 'catalog': parent['catalog'], 'inputs': inputs,
        'input_sha256': source_files, 'reused_record_sha256': reused,
        'seeds': list(SEEDS), 'kinds': list(KINDS), 'queries': list(QUERIES),
        'conditions': list(CONDITIONS), 'policies': list(POLICIES),
        'steps': 2000, 'effective_batch': 2, 'microbatch': 1, 'lr': 3e-4,
        'weight_decay': .01, 'gradient_clip': 5., 'parameters': 137472,
        'memory_slots': 256, 'insert_global_index': 11,
        'center_variance': parent['center_variance'], 'backbone_identity': old['backbone_identity'],
        'training_examples': 288, 'sample_order': 'training_ids, ascending query, condition, Recent/History',
        'loss': 'valid full-image native metric log-MSE; original full-input scale shared across conditions',
        'sampling': 'seeded random permutations repeated; two examples per step; matched across all three kinds',
        'geometry_frozen': True, 'backbone_frozen': True, 'final_checkpoint_only': True,
        'mask': 'same fixed mask0 ROI; each query independently selects early sensor points; all later input copies masked',
        'sensor_in_model_state': False, 'query_depth_selects_mask': False,
        'primary': ['persistent ROI new Soft History vs Recent', 'persistent ROI new Soft History vs new C History',
                    'persistent ROI new Soft History vs strong old C-static History',
                    'persistent ROI new Soft History vs new Center History'],
        'secondary': ['all/visible log-MSE', 'centered log-MSE', 'bias squared', 'absrel', 'rmse_m',
                      'history increments for all readers', 'history gain interaction across visibility conditions'],
        'expected_new_predictions': 648, 'reused_predictions': 1224,
        'expected_training_cache_windows': 172, 'new_download_bytes': 0,
        'fixed_figure_ids': ['fidelity_00', 'fidelity_01', 'fidelity_02'],
        'counterexample': 'Largest persistent History new Soft minus strong old C-static ROI error; seed mean; alphabetical tie break',
        'all_data_previously_used': True, 'new_confirmation': False,
        'parent_plan_sha256': sha256(PARENT / 'plan.json'),
        'parent_completion_sha256': sha256(PARENT / 'phase_completion.json'),
        'document_sha256': sha256(ROOT / 'docs/gaussian_memory_visibility_training_plan.md'),
        'source_sha256': {n: sha256(ROOT / n) for n in sorted(sources)}}
    p = OUT / 'plan.json'
    if p.exists(): assert json.loads(p.read_text()) == plan
    else: write_json(p, plan)
    for name, digest in plan['source_sha256'].items():
        dest = OUT / 'source_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): assert sha256(dest) == digest
        else: shutil.copy2(ROOT / name, dest)
    return plan


def checked_base(rid, name, plan):
    p = BASE_CACHE / rid / name
    assert sha256(p) == plan['inputs'][rid]['files'][name], (rid, name)
    return torch.load(p, map_location='cpu', weights_only=False)


def read_samples(plan):
    samples = []
    for rid in plan['training_ids']:
        marker = OUT / 'inputs' / f'{rid}.json'; rec = json.loads(marker.read_text())
        assert rec['complete'] and rec['plan_sha256'] == sha256(OUT / 'plan.json')
        for name, digest in rec['files'].items(): assert sha256(CACHE / rid / name) == digest
        for q in QUERIES:
            for c in range(3):
                s = torch.load(CACHE / rid / f'q{q}_c{c}.pt', map_location='cpu', weights_only=False)
                for policy in POLICIES: samples.append(dict(s, policy=policy))
    assert len(samples) == plan['training_examples']
    return samples


@torch.no_grad()
def training_inputs(plan, samples, kind):
    inputs = {}; lookup = {(s['id'], s['query']): s for s in samples}
    for rid in plan['training_ids']:
        for q in QUERIES:
            states = torch.load(CACHE / rid / f'q{q}_states.pt', map_location='cpu', weights_only=False)
            for state_key, state in states.items():
                m = device_tree(state)
                assert m['features'].shape == (256, 32)
                used = m['sources'].any(0).nonzero().flatten().tolist()
                allowed = ([f for f in LATE if f < q] + (list(EARLY) if state_key.endswith('/history') else []))
                assert set(used).issubset(allowed) and all(f < q for f in used)
                prior = memory_prior(m, lookup[rid, q], kind, plan['center_variance'])
                assert torch.isfinite(prior[..., :-1]).all() and torch.isneginf(prior[..., -1]).all()
                inputs[rid, q, state_key] = (m['features'].detach(), prior.detach())
    return inputs


def inputs_for(sample, inputs):
    state = 'persistent' if sample['condition'] == 2 else 'full'
    return inputs[sample['id'], sample['query'], state + '/' + sample['policy']]


if __name__ == '__main__':
    p = prepare(); print(json.dumps({'prepared': p['experiment'], 'training_examples': p['training_examples']}))
