"""Fixed identities and common-coordinate inputs for history_read1."""
import argparse
import json
import shutil

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_state_data import (
    OUT as STATE_READ, PARENT_OUT as SOFT_READ, MATCH_OUT as BOUND_READ,
    BASE_CACHE, SEEDS, QUERIES, STATE_OUT, source_identity as inherited_sources,
    write_json, save_tensor, device_tree)
from experiments.gaussian_memory_history_geometry import camera_to_common

OUT = ROOT / 'results/gaussian_memory/history_read1'
CACHE = DATA / 'history_read_v1'
FAMILIES = ('content_static', 'soft_static', 'content_dynamic', 'soft_dynamic', 'center_dynamic')
POLICIES = ('recent_corrective', 'history_corrective', 'recent_rebuild', 'history_rebuild')
WRITES = (4, 6, 8, 10)
ALLOWED = (0, 1, 2, 3, 4, 6, 8, 10, 12, 13, 14, 15, 16, 18, 20, 22)
NEW_SOURCES = ('models/gaussian_memory_history.py',
               'experiments/gaussian_memory_history_geometry.py',
               'experiments/gaussian_memory_history_data.py',
               'experiments/gaussian_memory_history_states.py',
               'experiments/gaussian_memory_history_evaluate.py')


def load_plan():
    return json.loads((OUT / 'plan.json').read_text())


def source_identity():
    result = inherited_sources()
    for name in NEW_SOURCES + ('models/gaussian_memory_incremental.py',
            'experiments/gaussian_memory_bound_evaluate.py',
            'experiments/gaussian_memory_soft_evaluate.py',
            'experiments/gaussian_memory_incremental_evaluate.py'):
        result[name] = sha256(ROOT / name)
    return result


def checkpoint_path(family, seed):
    kind = reader_kind(family)
    folder = STATE_READ if family.endswith('_dynamic') else (BOUND_READ if kind == 'content' else SOFT_READ)
    return folder / f'{kind}_{seed}/step_2000.pt'


def reader_kind(family):
    assert family in FAMILIES
    return 'soft_center' if family == 'center_dynamic' else family.split('_')[0]


def prepare():
    parent = json.loads((STATE_READ / 'plan.json').read_text())
    assert json.loads((STATE_READ / 'phase_completion.json').read_text())['experiment_complete']
    inventory_path = STATE_READ / 'history_common_gauge_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    lookup = {m['id']: m for m in parent['catalog']}
    pairs = []
    for index, scene in enumerate(inventory['scenes']):
        a, b = scene['clips']
        assert a['id'] == f'transfer_{index:02d}' and b['id'] == f'fidelity_{index:02d}'
        assert max(a['frame_indices']) < min(b['frame_indices'])
        inputs = {}
        for clip in (a, b):
            rid = clip['id']; identity = BASE_CACHE / rid / 'identity.json'
            identities = json.loads(identity.read_text())
            inputs[rid] = {'identity_sha256': sha256(identity), 'files': {}}
            for name in ('initial.pt', 'updates.pt'):
                digest = sha256(BASE_CACHE / rid / name)
                assert digest == identities['sha256'][name]
                inputs[rid]['files'][name] = digest
        initial_a = torch.load(BASE_CACHE / a['id'] / 'initial.pt', map_location='cpu', weights_only=False)
        initial_b = torch.load(BASE_CACHE / b['id'] / 'initial.pt', map_location='cpu', weights_only=False)
        track = STATE_OUT / a['id'] / 'trajectories.pt'
        assert sha256(track) == json.loads((track.parent / 'complete.json').read_text())['sha256']['trajectories.pt']
        pairs.append({'id': b['id'], 'early_id': a['id'], 'place': scene['place'], 'scene': scene['scene'],
                      'common_from_late_metric': b['common_from_clip_metric'],
                      'common_unit_meters': float(initial_a['unit_meters']),
                      'late_unit_meters': float(initial_b['unit_meters']),
                      'frames_early': a['frame_indices'], 'frames_late': b['frame_indices'],
                      'pose_path': scene['pose_path'], 'pose_sha256': scene['pose_sha256'],
                      'inputs': inputs, 'early_trajectories_sha256': sha256(track)})
    checkpoints = {}
    for family in FAMILIES:
        for seed in SEEDS:
            path = checkpoint_path(family, seed)
            status = json.loads((path.parent / 'status.json').read_text())
            assert status['complete'] and status['steps'] == 2000
            digest = sha256(path); assert digest == status['checkpoint_sha256']
            checkpoints[f'{family}_{seed}'] = {'path': str(path), 'sha256': digest}
    plan = {'experiment': 'history_read1', 'schema': 1,
            'document_sha256': sha256(ROOT / 'docs/gaussian_memory_history_read_plan.md'),
            'parent_completion_sha256': sha256(STATE_READ / 'phase_completion.json'),
            'inventory_sha256': sha256(inventory_path), 'pairs': pairs,
            'catalog': [lookup[p['id']] for p in pairs],
            'early_catalog': [lookup[p['early_id']] for p in pairs],
            'primary_places': [p['place'] for p in pairs], 'seeds': list(SEEDS),
            'families': list(FAMILIES), 'policies': list(POLICIES), 'checkpoints': checkpoints,
            'center_variance': parent['center_variance'], 'backbone_identity': parent['backbone_identity'],
            'allowed_source_frames': list(ALLOWED), 'source_columns': 24,
            'query_frames': list(QUERIES), 'memory_slots': 256, 'recent_frames': 4,
            'late_initialization_available_at': 15, 'old_data_only': True,
            'new_training': False, 'new_downloads': False, 'new_confirmation': False,
            'expected_new_predictions': 9108, 'expected_reused_rows': 756,
            'source_sha256': source_identity()}
    path = OUT / 'plan.json'
    if path.exists(): assert json.loads(path.read_text()) == plan, 'History plan changed.'
    else: write_json(path, plan)
    for name, digest in plan['source_sha256'].items():
        dest = OUT / 'source_snapshot' / name
        if dest.exists(): assert sha256(dest) == digest
        else: dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    return plan


def checked_load(rid, name, pair):
    path = BASE_CACHE / rid / name
    assert sha256(path) == pair['inputs'][rid]['files'][name]
    return torch.load(path, map_location='cpu', weights_only=False)


def common_sample(sample, pair):
    # Only the memory projection reads these poses. Native prefix, RGB,
    # backbone tail, labels and metric calibration retain their old identity.
    result = dict(sample)
    result['c2w'] = camera_to_common(sample['c2w'], pair['common_from_late_metric'],
                                   pair['late_unit_meters'], pair['common_unit_meters'])
    return result


def load_tracks(rid):
    path = CACHE / 'states' / rid / 'trajectories.pt'
    identity_path = OUT / 'states' / rid / 'complete.json'
    identity = json.loads(identity_path.read_text())
    assert identity['complete'] and identity['plan_sha256'] == sha256(OUT / 'plan.json')
    assert sha256(path) == identity['trajectories_sha256']
    return torch.load(path, map_location='cpu', weights_only=False), identity


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    torch.set_num_threads(4)
    p = prepare()
    print(json.dumps({'prepared': p['experiment'], 'pairs': len(p['pairs']), 'models': len(p['checkpoints'])}))
