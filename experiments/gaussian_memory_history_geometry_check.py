"""Input-only preparation checks; no history experiment or label evaluation."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_incremental_data import CACHE, build_memory, write_json
from experiments.gaussian_memory_history_geometry import (
    observation_to_common, build_common_memory, widen_source_columns)

OUT = ROOT / 'results/gaussian_memory/state_read1'


def point_moment_check():
    generator = torch.Generator().manual_seed(20261111)
    points = torch.randn(5, 196, 3, generator=generator, dtype=torch.float64)
    points *= torch.tensor([.3, .1, .02]); points += torch.tensor([.2, -.4, 2.])
    mean = points.mean(1); delta = points - mean[:, None]; eye = torch.eye(3, dtype=points.dtype)
    cov = delta.transpose(-1, -2) @ delta / points.shape[1] + eye * .005**2
    transform = torch.eye(4, dtype=points.dtype)
    angle = .7
    transform[:3, :3] = torch.tensor([[np.cos(angle), 0., np.sin(angle)],
                                    [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    transform[:3, 3] = torch.tensor([.5, -.1, .2])
    observation = {'means': mean, 'covars': cov, 'features': torch.zeros(5, 32),
        'confidence': torch.ones(5), 'valid': torch.ones(5, dtype=torch.bool),
        'source_frame': torch.zeros(5, dtype=torch.long), 'depth_patch': mean[:, 2],
        'camera': torch.eye(4, dtype=points.dtype)[None], 'K': eye[None]}
    result = observation_to_common(observation, transform, 1.7, .8, 12)
    moved = (1.7 * points @ transform[:3, :3].T + transform[:3, 3]) / .8
    expected_mean = moved.mean(1); delta = moved - expected_mean[:, None]
    expected_cov = delta.transpose(-1, -2) @ delta / points.shape[1] + eye * .005**2
    torch.testing.assert_close(result['means'], expected_mean, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(result['covars'], expected_cov, rtol=1e-10, atol=1e-10)
    return {'mean_max_abs_difference': float((result['means'] - expected_mean).abs().max()),
            'covariance_max_abs_difference': float((result['covars'] - expected_cov).abs().max())}


def main():
    torch.set_num_threads(4)
    inventory_path = OUT / 'history_common_gauge_inventory.json'
    inventory = json.loads(inventory_path.read_text()); records = []
    parent_equal = None
    for scene in inventory['scenes']:
        early, later = scene['clips']
        initial = {}
        input_hashes = {}
        for clip in (early, later):
            folder = CACHE / clip['id']
            identity = json.loads((folder / 'identity.json').read_text())
            for name in ('initial.pt', 'updates.pt'):
                digest = sha256(folder / name)
                assert digest == identity['sha256'][name]
                input_hashes[f"{clip['id']}/{name}"] = digest
            initial[clip['id']] = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=False)
        a, b = initial[early['id']], initial[later['id']]
        if parent_equal is None:
            old = build_memory(a['observation'])
            same = build_common_memory(a['observation'], source_columns=12)
            assert old.keys() == same.keys() and all(torch.equal(old[k], same[k]) for k in old)
            wider = build_common_memory(a['observation'], source_columns=24)
            padded = widen_source_columns(old, 24)
            assert all(torch.equal(wider[k], padded[k]) for k in wider)
            parent_equal = {'clip': early['id'], 'all_tensors_equal_same_cpu_execution': True,
                            '24_source_columns_only_add_false_provenance_bits': True}
        updates = torch.load(CACHE / later['id'] / 'updates.pt', map_location='cpu', weights_only=False)
        observed = [b['observation']] + [v for v in updates.values() if v is not None]
        transform = torch.tensor(later['common_from_clip_metric'], dtype=torch.float32)
        inverse = torch.linalg.inv(transform); largest = {'means': 0., 'covars': 0., 'camera': 0., 'depth_patch': 0.}
        minimum_eigenvalue = float('inf'); camera_local_error = 0.
        for observation in observed:
            common = observation_to_common(observation, transform, b['unit_meters'], a['unit_meters'], 12)
            assert torch.equal(common['source_frame'], observation['source_frame'] + 12)
            assert all(torch.equal(common[k], observation[k]) for k in ('features', 'valid', 'confidence', 'K'))
            for j, frame in enumerate(observation['source_frame'].unique(sorted=True)):
                keep = observation['source_frame'] == frame
                old_camera = observation['camera'][j]; new_camera = common['camera'][j]
                old_xyz = (observation['means'][keep] - old_camera[:3, 3]) @ old_camera[:3, :3]
                new_xyz = (common['means'][keep] - new_camera[:3, 3]) @ new_camera[:3, :3]
                expected = old_xyz * (b['unit_meters'] / a['unit_meters'])
                torch.testing.assert_close(new_xyz, expected, rtol=1e-4, atol=2e-5)
                camera_local_error = max(camera_local_error, float((new_xyz - expected).abs().max()))
            minimum_eigenvalue = min(minimum_eigenvalue, float(torch.linalg.eigvalsh(common['covars']).min()))
            common['source_frame'] -= 12
            restored = observation_to_common(common, inverse, a['unit_meters'], b['unit_meters'], 0)
            for key in largest:
                error = float((restored[key] - observation[key]).abs().max())
                largest[key] = max(largest[key], error)
                torch.testing.assert_close(restored[key], observation[key], rtol=1e-4, atol=2e-5)
        assert minimum_eigenvalue > 0
        records.append({'early': early['id'], 'later': later['id'],
            'input_sha256': input_hashes, 'observation_batches': len(observed),
            'later_over_common_unit': b['unit_meters'] / a['unit_meters'],
            'roundtrip_max_abs_difference': largest, 'camera_local_max_abs_difference': camera_local_error,
            'minimum_common_covariance_eigenvalue': minimum_eigenvalue})
    result = {'complete': True, 'scope': 'Input geometry preparation only, not method validation.',
        'next_experiment_adopted': False, 'target_labels_read': False, 'model_predictions_run': False,
        'new_downloads': 0, 'point_moment_check': point_moment_check(),
        'parent_build_check': parent_equal, 'pairs': records,
        'inventory_sha256': sha256(inventory_path),
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in (
            ROOT / 'experiments/gaussian_memory_history_geometry.py', __file__ and ROOT / 'experiments/gaussian_memory_history_geometry_check.py')}}
    write_json(OUT / 'history_geometry_preparation.json', result)
    print(json.dumps({'complete': True, 'pairs': len(records),
        'batches': sum(r['observation_batches'] for r in records),
        'maximum_roundtrip_error': max(v for r in records for v in r['roundtrip_max_abs_difference'].values()),
        'minimum_common_eigenvalue': min(r['minimum_common_covariance_eigenvalue'] for r in records),
        'parent_build_equal': True, 'no_new_method_results': True}), flush=True)


if __name__ == '__main__': main()
