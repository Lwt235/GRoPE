"""Fixed inputs and controls for the finite historical-visibility experiment."""
import json
import shutil

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_incremental_data import (
    CACHE as BASE_CACHE, SEEDS, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_history_data import FAMILIES, checkpoint_path, reader_kind

OUT = ROOT / 'results/gaussian_memory/history_visibility_control1'
CACHE = DATA / 'history_visibility_v1'
RAW = ROOT / 'results/gaussian_memory/raw_history_information1'
EARLY = (0, 1, 2, 3)
LATE = (4, 6, 8, 10)
CURRENT = (6, 8, 10, 11)
CONDITIONS = ('full', 'query_only', 'persistent')
POLICIES = ('recent', 'history')
NEW_SOURCES = ('experiments/gaussian_memory_visibility_data.py',
               'experiments/gaussian_memory_visibility_run.py')


def load_plan():
    return json.loads((OUT / 'plan.json').read_text())


def prepare():
    assert json.loads((RAW / 'phase_completion.json').read_text())['experiment_complete']
    raw = json.loads((RAW / 'plan.json').read_text())
    parent = json.loads((ROOT / 'results/gaussian_memory/history_read1/plan.json').read_text())
    catalog = [m for m in raw['catalog'] if m['id'].startswith('fidelity_')]
    assert len(catalog) == 12 and len({m['place'] for m in catalog}) == 12
    identities = {}; inputs = {}
    for meta in catalog:
        rid = meta['id']; identity_path = BASE_CACHE / rid / 'identity.json'
        identity = json.loads(identity_path.read_text())
        files = {}
        for name in ('initial.pt', 'updates.pt', 'q11_c0.pt', 'q11_c1.pt'):
            p = BASE_CACHE / rid / name
            assert sha256(p) == identity['sha256'][name]
            files[name] = identity['sha256'][name]
        write_scales = {str(d['window'][-1]): {'window': d['window'], 'scale': d.get('scale')}
                        for d in identity['diagnostics'] if d['kind'] == 'write'}
        assert set(write_scales) == set(map(str, LATE))
        assert all(d['scale'] is not None for d in write_scales.values())
        sample = torch.load(BASE_CACHE / rid / 'q11_c0.pt', map_location='cpu', weights_only=False)
        assert sample['metric_scale'] is not None and sample['input_frames'] == list(CURRENT)
        inputs[rid] = {'base_identity_sha256': sha256(identity_path), 'files': files,
                       'write_scales': write_scales, 'query_scale_full': sample['metric_scale'],
                       'unit_meters': identity['unit_meters']}
        for frame in EARLY + LATE + (11,):
            f = meta['row']['frames'][frame]
            for kind in ('rgb', 'depth'):
                p = f[kind + '_path']; digest = sha256(p)
                assert digest == f[kind + '_sha256']
                identities[p] = digest
        for kind in ('intrinsics', 'poses'):
            p = meta['row'][kind]; identities[p] = sha256(p)
            assert identities[p] == meta['row'][kind + '_sha256']
    checkpoints = {}
    for fam in list(FAMILIES) + ['local']:
        for seed in SEEDS:
            p = (ROOT / f'results/gaussian_memory/bound_read1/local_{seed}/step_2000.pt'
                 if fam == 'local' else checkpoint_path(fam, seed))
            status = json.loads((p.parent / 'status.json').read_text())
            assert status['complete'] and sha256(p) == status['checkpoint_sha256']
            checkpoints[f'{fam}_{seed}'] = {'path': str(p), 'sha256': sha256(p)}
    sources = set(NEW_SOURCES) | set(parent['source_sha256']) | {
        'experiments/gaussian_memory_raw_history_run.py',
        'experiments/gaussian_memory_view_data.py', 'models/gaussian_memory_view_token.py',
        'experiments/gaussian_memory_data.py', 'models/gaussian_memory_incremental.py'}
    identities[str(DATA / 'cache_v1/pca.pt')] = sha256(DATA / 'cache_v1/pca.pt')
    plan = {'experiment': 'history_visibility_control1', 'catalog': catalog, 'inputs': inputs,
        'input_identities': identities, 'checkpoints': checkpoints, 'seeds': list(SEEDS),
        'families': list(FAMILIES), 'conditions': list(CONDITIONS), 'policies': list(POLICIES),
        'early': list(EARLY), 'late': list(LATE), 'current': list(CURRENT), 'query': 11,
        'center_variance': parent['center_variance'], 'memory_slots': 256,
        'backbone_identity': parent['backbone_identity'],
        'mask_rule': 'Existing query11 mask0; all early sensor points projected inside it, then every touched source 14x14 patch. No target-depth selection or visibility pruning.',
        'mask_fill': [.485, .456, .406], 'pixel_centers': .5,
        'source_depth_range_m': [.1, 10.], 'camera_z_min_m': 1e-4,
        'metric_scale': 'Original full-input query scale shared by all three conditions; original full-input write scales fixed.',
        'hidden_metric_region': 'Same existing query11 mask0 ROI for all conditions, including full input.',
        'state_admission': 'Original predicted confidence and finite checks; masked patches not selectively removed, so inferred geometry may remain.',
        'information_diagnostic': 'Sensor raw support after excluding directly masked source pixels; not coherent-input predicted geometry or an upper bound on the network.',
        'primary': 'persistent fixed-ROI log-MSE: Soft-dynamic History vs Recent; History Soft-dynamic vs both C-static and C-dynamic and Center-dynamic',
        'secondary': ['whole-image log-MSE', 'visible log-MSE', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m', 'history-gain interaction across visibility conditions'],
        'bootstrap': {'draws': 5000, 'seed': 20261111, 'unit': 'paired place and model seed'},
        'fixed_figure_ids': ['fidelity_00', 'fidelity_01', 'fidelity_02'],
        'counterexample': 'Largest persistent History Soft-dynamic minus C-static ROI log-MSE, seed mean; alphabetical place tie break; descriptive, no exclusions.',
        'expected_predictions': 1224, 'expected_new_backbone_windows': 62,
        'new_training_steps': 0, 'new_download_bytes': 0, 'new_confirmation': False,
        'all_data_previously_used': True,
        'document_sha256': sha256(ROOT / 'docs/gaussian_memory_history_visibility_plan.md'),
        'source_sha256': {n: sha256(ROOT / n) for n in sorted(sources)},
        'parent_completion_sha256': sha256(RAW / 'phase_completion.json')}
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        write_json(p, plan)
    for name, digest in plan['source_sha256'].items():
        p = OUT / 'source_snapshot' / name; p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists(): assert sha256(p) == digest
        else: shutil.copy2(ROOT / name, p)
    return plan


def checked_base(rid, name, plan):
    p = BASE_CACHE / rid / name
    assert sha256(p) == plan['inputs'][rid]['files'][name]
    return torch.load(p, map_location='cpu', weights_only=False)


if __name__ == '__main__':
    p = prepare()
    print(json.dumps({'prepared': p['experiment'], 'places': len(p['catalog']),
                      'frozen_models': len(p['checkpoints']), 'planned_predictions': p['expected_predictions']}))
