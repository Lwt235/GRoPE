"""Fixed source inventory for the finite raw-history information diagnostic."""
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import torch

from experiments.gaussian_memory_history_data import (
    ROOT, DATA, OUT as HISTORY, CACHE as HISTORY_CACHE, BASE_CACHE,
    QUERIES, POLICIES, sha256, write_json, save_tensor)
from experiments.gaussian_memory_incremental_data import depth_source

OUT = ROOT / 'results/gaussian_memory/raw_history_information1'
CACHE = DATA / 'raw_history_information_v1'
SOURCE_FRAMES = (0, 1, 2, 3, 4, 6, 8, 10)
VARIANTS = ('sensor', 'prediction', 'sensor_matched', 'prediction_matched')
NEW_SOURCES = ('experiments/gaussian_memory_raw_history_data.py',
               'experiments/gaussian_memory_raw_history_extract.py',
               'experiments/gaussian_memory_raw_history_run.py')


def load_plan(): return json.loads((OUT / 'plan.json').read_text())


def prepare():
    if (OUT / 'plan.json').exists(): return load_plan()
    parent = json.loads((HISTORY / 'plan.json').read_text())
    previous = ROOT / 'results/gaussian_memory/native_anchor_probe1'
    assert json.loads((previous / 'phase_completion.json').read_text())['experiment_complete']
    catalog = parent['early_catalog'] + parent['catalog']
    for meta in catalog:
        for i, frame in enumerate(meta['row']['frames'][:12]):
            if i in SOURCE_FRAMES or (meta['id'].startswith('fidelity_') and i in QUERIES):
                path = depth_source(frame); digest = sha256(path)
                if 'depth_sha256' in frame: assert digest == frame['depth_sha256']
                frame.update(depth_path=str(path), depth_sha256=digest)
            assert sha256(frame['rgb_path']) == frame['rgb_sha256']
    identities = {}
    for pair in parent['pairs']:
        for rid in (pair['early_id'], pair['id']):
            ip = BASE_CACHE / rid / 'identity.json'
            assert sha256(ip) == pair['inputs'][rid]['identity_sha256']
            identities[rid] = {'path': str(ip), 'sha256': sha256(ip)}
            identity = json.loads(ip.read_text())
            for name in ('initial.pt', 'updates.pt'):
                assert sha256(BASE_CACHE / rid / name) == identity['sha256'][name]
        path = HISTORY / 'states' / pair['id'] / 'complete.json'
        state = json.loads(path.read_text())
        assert state['complete'] and sha256(state['trajectories_path']) == state['trajectories_sha256']
        pair['state_identity'] = {'path': str(path), 'sha256': sha256(path),
                                 'trajectories_path': state['trajectories_path'],
                                 'trajectories_sha256': state['trajectories_sha256']}
    sources = (*NEW_SOURCES, 'experiments/gaussian_memory_incremental_data.py',
               'experiments/gaussian_memory_view_data.py', 'models/gaussian_memory_vggt_adapter.py')
    plan = {'experiment': 'raw_history_information1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'document_sha256': sha256(ROOT / 'docs/gaussian_memory_raw_history_information_plan.md'),
        'parent_history_plan_sha256': sha256(HISTORY / 'plan.json'),
        'parent_anchor_completion_sha256': sha256(previous / 'phase_completion.json'),
        'catalog': catalog, 'late_ids': [m['id'] for m in parent['catalog']], 'pairs': parent['pairs'],
        'input_identities': identities, 'source_frames': list(SOURCE_FRAMES), 'queries': list(QUERIES),
        'policies': list(POLICIES), 'variants': list(VARIANTS),
        'height': 210, 'width': 280, 'patch_size': 14,
        'pixel_centers': 'index+0.5', 'raster': 'floor(uv), minimum positive z, no hole filling',
        'depth_valid_m': [.1, 10.], 'camera_z_min_m': 1e-4,
        'depth_agreement_relative': .1, 'state_coverage_threshold': .1,
        'state_projection': 'unchanged project_memory at 15x20, repeated14x14 for pixel-region diagnostics',
        'recent_source_rule': 'B0/1/2/3 plus B4/6/8/10 strictly before query',
        'history_source_rule': 'All eight A sources plus the same Recent B sources',
        'matched_source_rule': 'Both source sensor and source predicted depths valid at identical pixels, diagnostic only',
        'replay_windows': 120, 'source_frames_total': 192, 'query_cameras': 48, 'query_conditions': 144,
        'frozen_backbone': parent['backbone_identity'],
        'cached_original_calibration': True, 'replay_patch_depth_rtol': 1e-4, 'replay_patch_depth_atol': 2e-5,
        'source_sha256': {name: sha256(ROOT / name) for name in sources},
        'new_training_steps': 0, 'new_download_bytes': 0, 'all_data_previously_used': True,
        'new_confirmation': False, 'bootstrap': {'draws': 5000, 'unit': 'paired place', 'seed': 20261111}}
    write_json(OUT / 'plan.json', plan)
    for name, digest in plan['source_sha256'].items():
        dest = OUT / 'source_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    return plan


if __name__ == '__main__':
    p = prepare(); print(json.dumps({'prepared': p['experiment'], 'source_frames': 192, 'pairs': len(p['pairs'])}), flush=True)
