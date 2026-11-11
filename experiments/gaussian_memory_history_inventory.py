"""Read-only camera-gauge inventory for a possible later history study."""
import json
from pathlib import Path

import numpy as np

from experiments.gaussian_memory_setup import ROOT, sha256


def main():
    current = ROOT / 'results/gaussian_memory/state_read1'
    plan = json.loads((current / 'plan.json').read_text())
    metadata = {m['id']: m for m in plan['catalog']}; scenes = []
    for index in range(12):
        clips = [metadata[f'{prefix}_{index:02d}'] for prefix in ('transfer', 'fidelity')]
        rows = [m['row'] for m in clips]
        assert len({r['scene'] for r in rows}) == 1 and len({r['poses'] for r in rows}) == 1
        source = Path(rows[0]['poses']); digest = sha256(source)
        assert all(digest == r['poses_sha256'] for r in rows)
        data = np.loadtxt(source).reshape(-1, 3, 4)
        poses = np.broadcast_to(np.eye(4), (len(data), 4, 4)).copy(); poses[:, :3] = data
        first = rows[0]['frames'][0]['index']; common_from_world = np.linalg.inv(poses[first])
        ids = [[f['index'] for f in row['frames']] for row in rows]
        assert len(set(ids[0]) & set(ids[1])) == 0 and max(ids[0]) < min(ids[1])
        clip_records = []
        for meta, row, indices in zip(clips, rows, ids):
            old = np.asarray(row['c2w_metric'])[:len(indices)]
            original_local = np.linalg.inv(poses[indices[0]])[None] @ poses[indices]
            common = common_from_world[None] @ poses[indices]
            common_from_local = common_from_world @ poses[indices[0]]
            reconstructed = common_from_local[None] @ old
            error = float(np.max(np.abs(reconstructed - common)))
            local_error = float(np.max(np.abs(original_local - old)))
            # Float32-derived cached poses are only being identified, not tuned.
            assert max(error, local_error) < 1e-4, (meta['id'], error, local_error)
            clip_records.append({'id': meta['id'], 'frame_indices': indices,
                'rgb_sha256': [f['rgb_sha256'] for f in row['frames']],
                'common_from_clip_metric': common_from_local.tolist(),
                'cached_local_pose_max_abs_difference': local_error,
                'reconstructed_common_pose_max_abs_difference': error})
        scenes.append({'scene': rows[0]['scene'], 'place': clips[0]['place'],
            'pose_path': str(source), 'pose_sha256': digest, 'clips': clip_records,
            'gap_frame_indices': min(ids[1]) - max(ids[0]),
            'total_unique_rgb': len(set(ids[0] + ids[1]))})
    value = {'scope': 'Planning feasibility only; no evaluation, fitting, re-extraction or method choice.',
        'all_data_previously_used': True, 'nontraining_sources': len(scenes), 'scenes': scenes,
        'current_plan_sha256': sha256(current / 'plan.json'),
        'source_sha256': sha256(__file__),
        'coordinate_rule': 'One world-to-first-camera rigid gauge for both clips; use one initial metric scene unit if extracting a joint memory.',
        'writer_boundary': 'Existing 256-slot update operations accept variable source width, but old build_memory fixes 12 source columns; a future extractor must explicitly allocate the full timeline without changing frozen parent code.',
        'limits': ['Windows are separated sparse observations, not continuous dense video.',
                   'Known camera poses and previously used places; not new confirmation.',
                   'Combining poses does not combine or recalibrate per-window cached memory units.',
                   'Future protocol must distinguish information outside the native window from a changed training distribution.']}
    path = current / 'history_common_gauge_inventory.json'
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'sources': len(scenes), 'rgb': sum(s['total_unique_rgb'] for s in scenes),
        'max_pose_reconstruction_difference': max(c['reconstructed_common_pose_max_abs_difference'] for s in scenes for c in s['clips']),
        'frame_gaps': [s['gap_frame_indices'] for s in scenes], 'no_new_experiment': True}), flush=True)


if __name__ == '__main__': main()
