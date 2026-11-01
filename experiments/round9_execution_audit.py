"""Independent Round 9 artifact/trajectory/readout audit; never opens target RGB.

Before-evaluation mode must finish before calibration is created. Optional GPU
replay calls gsplat directly, without importing the experiment/model modules.
After-evaluation mode verifies that calibration froze those same predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SCENES = ('teatime', 'waldo_kitchen')
ROLES = ('full', 'isotropic', 'fixed', 'point')
HS = (.25, .5, 1., 2.)
METHODS = [f'{role}_h{h:g}' for role in ROLES for h in HS] + ['bilinear', 'original_gs']
EXPECTED_LOCK = '7d7f53de51c4c179efad6b571f172bb966acf991342d24ebe0dc476dbdc4ffb2'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    actual = digest(path)
    require(actual == expected, f'Hash mismatch: {path}')
    return actual


def document(path):
    return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def covariance_np(state):
    q = state['quaternions'].astype(np.float64)
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    scales = np.exp(state['log_scales'].astype(np.float64))
    return (rotation * scales[:, None, :]) @ np.swapaxes(rotation * scales[:, None, :], 1, 2)


def resize_K(K, old_w, old_h):
    sx, sy = 192 / old_w, 144 / old_h
    return np.array([[sx, 0., (sx - 1) / 2], [0., sy, (sy - 1) / 2], [0., 0., 1.]]) @ K


def check_scene(scene, runs, data_root, lock):
    root, data_path = runs / scene, data_root / scene
    train, run = document(root / 'training.json'), document(root / 'run.json')
    manifest, targets = document(data_path / 'manifest.json'), document(data_path / 'target_cameras.json')
    artifacts = {}
    for filename, expected in (
        ('state.pt', train['state_sha256']), ('gaussians.npz', train['gaussians_sha256']),
        ('training.json', run['training_sha256']), ('colors.npz', run['colors_sha256']),
        ('predictions.npz', run['prediction_sha256']), ('operator_samples.json', run['operator_samples_sha256'])):
        p = root / filename
        artifacts[str(p.resolve())] = checked(p, expected)
    artifacts[str((root / 'run.json').resolve())] = digest(root / 'run.json')
    for record in (train, run):
        require(record['source_hashes'] == lock['source_hashes'], f'{scene}: source table')
        require(record['data_hashes'] == lock['data_hashes'], f'{scene}: data table')
        checked(runs / 'lock.json', record['lock_sha256'])
        require(record['scene'] == scene and record['target_rgb_read'] is False, f'{scene}: task identity')
    for name, key in (('manifest.json', 'data_manifest_sha256'), ('target_cameras.json', 'target_cameras_sha256'),
                      ('context.npz', 'context_archive_sha256')):
        checked(data_path / name, run[key])
    checked(data_path / 'context.npz', manifest['context_archive_sha256'])
    checked(data_path / 'target_cameras.json', manifest['target_cameras_sha256'])
    checked(ROOT / 'experiments/round9_context_data.py', manifest['exporter_sha256'])
    checked(manifest['source_dataset'], manifest['source_dataset_sha256'])
    source_dataset = document(manifest['source_dataset'])
    require(set(source_dataset) == {'cameras', 'images', 'poses', 'pose_convention'}, f'{scene}: camera-only input')
    image_rows = {r['id']: r for r in source_dataset['images']}
    pose_rows = {r['image_id']: r for r in source_dataset['poses']}
    camera_rows = {r['id']: r for r in source_dataset['cameras']}
    ordered = sorted(pose_rows, key=lambda i: image_rows[i]['name'])
    blocks = np.array_split(ordered, 3)
    development = np.concatenate(blocks[:2])
    selected = development[np.linspace(0, len(development) - 1, 32, dtype=np.int64)]
    expected_targets = selected[3::4].tolist()
    expected_context = [int(v) for j, v in enumerate(selected) if j % 4 != 3]
    require(manifest['split']['context_ids'] == expected_context, f'{scene}: context split')
    require(manifest['split']['target_ids'] == expected_targets, f'{scene}: target split')
    require(not set(selected) & set(blocks[2]), f'{scene}: old sealed image membership')
    require(manifest['context_rgb_reads'] == 24 and manifest['target_rgb_reads'] == 0 and
            manifest['old_geometry_keypoints_tracks_read'] is False, f'{scene}: exporter access contract')
    context = archive(data_path / 'context.npz')
    require(context['ids'].tolist() == expected_context, f'{scene}: array identity')
    require(all(np.isfinite(a).all() for a in context.values()), f'{scene}: finite inputs')
    require(context['rgb'].shape == (24, 144, 192, 3), f'{scene}: context image shape')
    require(targets['target_rgb_read'] is False, f'{scene}: target camera-only contract')
    require([f['id'] for f in targets['frames']] == expected_targets, f'{scene}: target order')
    require([f['split'] for f in targets['frames']] == ['calibration', 'gate'] * 4, f'{scene}: gate order')
    original_R = np.array([pose_rows[i]['R'] for i in expected_context])
    original_t = np.array([pose_rows[i]['t'] for i in expected_context])
    centers = -np.einsum('nji,nj->ni', original_R, original_t)
    center = np.mean(centers, axis=0)
    scale = 1.1 * np.linalg.norm(centers - center, axis=1).max()
    np.testing.assert_allclose(manifest['normalization']['center'], center, atol=1e-12, rtol=0)
    np.testing.assert_allclose(manifest['normalization']['scale'], scale, atol=1e-12, rtol=0)
    context_paths = set()
    for i, identity in enumerate(expected_context):
        img, pose = image_rows[identity], pose_rows[identity]
        cam = camera_rows[img['camera_id']]
        path = (Path(manifest['source_dataset']).parent / img['path']).resolve()
        context_paths.add(str(path))
        checked(path, manifest['source_image_sha256'][str(path)])
        np.testing.assert_allclose(context['K'][i], resize_K(cam['K'], cam['width'], cam['height']), atol=1e-12, rtol=0)
        np.testing.assert_allclose(context['R'][i], pose['R'], atol=1e-12, rtol=0)
        nt = (np.array(pose['R']) @ center + np.array(pose['t'])) / scale
        np.testing.assert_allclose(context['t'][i], nt, atol=1e-12, rtol=0)
    require(context_paths == set(manifest['source_image_sha256']), f'{scene}: RGB hash allowlist')
    for frame in targets['frames']:
        img, pose = image_rows[frame['id']], pose_rows[frame['id']]
        cam = camera_rows[img['camera_id']]
        expected_path = str((Path(manifest['source_dataset']).parent / img['path']).resolve())
        require(frame['name'] == img['name'] and frame['image_path'] == expected_path, f'{scene}: target source identity')
        require(expected_path not in context_paths, f'{scene}: target/context path overlap')
        np.testing.assert_allclose(frame['K'], resize_K(cam['K'], cam['width'], cam['height']), atol=1e-12, rtol=0)
        np.testing.assert_allclose(frame['R'], pose['R'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(frame['t'], (np.array(pose['R']) @ center + pose['t']) / scale, atol=1e-12, rtol=0)
    require(train['steps'] == 3000 and train['seed'] == 11, f'{scene}: training budget')
    require(train['checkpoint_selection'] == 'fixed final step, no validation', f'{scene}: checkpoint selection')
    rng = np.random.default_rng(11)
    indices = rng.integers(24, size=3000)
    batch_sha = hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()
    require(train['batch_sequence_sha256'] == batch_sha, f'{scene}: independently replayed batch sequence')
    expected_steps = [1] + list(range(300, 3001, 300))
    require([r['step'] for r in train['history']] == expected_steps, f'{scene}: training history steps')
    for row in train['history']:
        require(row['context_image_id'] == expected_context[indices[row['step'] - 1]], f'{scene}: context batch identity')
    require([r['id'] for r in train['context_metrics']] == expected_context, f'{scene}: training metric population')
    n = len(context['points'])
    require(train['initial_point_count'] == n, f'{scene}: initialization count')
    for step, entry in zip((500, 1000, 1500, 2000), train['topology'], strict=True):
        split_count = min(n, 12000 - n)
        n += split_count
        require(entry['step'] == step and entry['split_parents'] == split_count and entry['primitives'] == n,
                f'{scene}: topology schedule')
    state = {k: v.cpu().numpy() for k, v in torch.load(root / 'state.pt', map_location='cpu', weights_only=True).items()}
    require(set(state) == {'means', 'log_scales', 'quaternions', 'opacity_logits', 'color_logits'}, f'{scene}: state fields')
    require(n == train['point_count'] == run['point_count'] == len(state['means']), f'{scene}: final topology')
    require(all(np.isfinite(v).all() for v in state.values()), f'{scene}: finite learned state')
    np.testing.assert_allclose(np.linalg.norm(state['quaternions'], axis=1), 1., atol=2e-6, rtol=0)
    gaussians, colors, predictions = (archive(root / name) for name in ('gaussians.npz', 'colors.npz', 'predictions.npz'))
    np.testing.assert_array_equal(gaussians['means'], state['means'])
    np.testing.assert_array_equal(gaussians['quaternions'], state['quaternions'])
    cov = covariance_np(state)
    covariance_error = float(np.max(np.abs(cov - gaussians['covariances'])))
    np.testing.assert_allclose(cov, gaussians['covariances'], atol=2e-8, rtol=2e-5)
    for name in ('colors', 'opacities'):
        logits = state['color_logits' if name == 'colors' else 'opacity_logits'].astype(np.float64)
        np.testing.assert_allclose(gaussians[name], 1 / (1 + np.exp(-logits)), atol=1e-7, rtol=1e-7)
    require(colors['methods'].tolist() == METHODS == predictions['methods'].tolist() == run['methods'], f'{scene}: methods')
    require(colors['colors'].shape == (18, n, 3), f'{scene}: equal primitive RGB capacity')
    require(np.isfinite(colors['colors']).all(), f'{scene}: finite readout RGB')
    np.testing.assert_array_equal(colors['colors'][-1], gaussians['colors'])
    counts = colors['source_view_counts']
    require(counts.shape == (n,) and counts.dtype == np.int64 and counts.min() >= 0 and counts.max() <= 24,
            f'{scene}: source counts')
    fallback = counts == 0
    require(int(fallback.sum()) == run['fallback_primitive_count'], f'{scene}: fallback count')
    require(int((~fallback).sum()) == run['readable_primitive_count'], f'{scene}: readable count')
    require(sum(v['visible_primitive_count'] for v in run['views']) == int(counts.sum()), f'{scene}: source count sum')
    require([v['id'] for v in run['views']] == expected_context, f'{scene}: source view identities')
    np.testing.assert_array_equal(colors['colors'][:, fallback],
                                  np.broadcast_to(gaussians['colors'][fallback], (18, int(fallback.sum()), 3)))
    require(predictions['predictions'].shape == (18, 8, 144, 192, 3), f'{scene}: all prediction shapes')
    require(predictions['target_ids'].tolist() == expected_targets, f'{scene}: prediction targets')
    require(predictions['target_splits'].tolist() == ['calibration', 'gate'] * 4, f'{scene}: prediction target split')
    require(np.isfinite(predictions['predictions']).all(), f'{scene}: finite predictions')
    require(predictions['predictions'].min() >= -1e-6 and predictions['predictions'].max() <= 1.000001,
            f'{scene}: bounded predictions')
    contribution, alpha = predictions['fallback_contribution'], predictions['alpha']
    require(contribution.shape == alpha.shape == (8, 144, 192), f'{scene}: full-pixel alpha/fallback')
    require(np.isfinite(contribution).all() and np.isfinite(alpha).all(), f'{scene}: finite contribution')
    require(contribution.min() >= -1e-6 and np.max(contribution - alpha) <= 1e-5, f'{scene}: fallback is alpha submass')
    return dict(scene=scene, passed=True, artifacts=artifacts, batch_sequence_sha256=batch_sha,
                context_images=24, target_cameras=8, target_rgb_reads=0, final_primitives=n,
                initial_primitives=len(context['points']), fallback_primitives=int(fallback.sum()),
                source_primitive_view_memberships=int(counts.sum()), covariance_max_absolute_error=covariance_error,
                context_metrics=train['context_metrics']), (state, context, targets, colors, predictions, cov)


def bilinear(image, xy):
    x, y = xy.T
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    dx, dy = x - x0, y - y0
    xx = np.stack((x0, x0 + 1, x0, x0 + 1), 1).clip(0, 191)
    yy = np.stack((y0, y0, y0 + 1, y0 + 1), 1).clip(0, 143)
    weights = np.stack(((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy), 1)
    return np.sum(image[yy, xx] * weights[..., None], axis=1)


def replay(payload, device):
    """Independent direct-render and NumPy geometry/averaging reconstruction."""
    state, context, targets, colors, predictions, cov = payload
    sys.path.insert(0, '/home/liwt/miniColmap/src')
    from minicolmap.rendering.gsplat_runtime import load_project_extension
    load_project_extension()
    from gsplat import rasterization
    tensors = {k: torch.tensor(v, device=device) for k, v in state.items()}
    n = len(state['means'])

    def render(K, R, t, rgb, depth=False):
        k = torch.as_tensor(np.array(K), dtype=torch.float32, device=device)
        k[0, 2] += .5
        k[1, 2] += .5
        view = torch.eye(4, dtype=torch.float32, device=device)
        view[:3, :3] = torch.as_tensor(np.array(R), dtype=torch.float32, device=device)
        view[:3, 3] = torch.as_tensor(np.array(t), dtype=torch.float32, device=device)
        result, alpha, _ = rasterization(
            means=tensors['means'], quats=tensors['quaternions'], scales=tensors['log_scales'].exp(),
            opacities=tensors['opacity_logits'].sigmoid(), colors=torch.as_tensor(np.ascontiguousarray(rgb), dtype=torch.float32, device=device),
            viewmats=view[None], Ks=k[None], width=192, height=144, backgrounds=k.new_zeros((1, 3)),
            packed=False, sh_degree=None, eps2d=.3, near_plane=1e-4, far_plane=1e6,
            rasterize_mode='classic', render_mode='RGB+ED' if depth else 'RGB')
        return result[0].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()

    counts = np.zeros(n, np.int64)
    sums = np.zeros((n, 3), np.float64)
    readable = np.flatnonzero(colors['source_view_counts'] > 0)
    sample_ids = readable[np.linspace(0, len(readable) - 1, min(8, len(readable)), dtype=int)]
    sampled_sums = np.zeros((16, len(sample_ids), 3), np.float64)
    yy, xx = np.mgrid[:144, :192]
    pixels = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    with torch.no_grad():
        for i in range(24):
            K, R, t = [context[k][i].astype(np.float32).astype(np.float64) for k in ('K', 'R', 't')]
            xyz = state['means'].astype(np.float64) @ R.T + t
            z = xyz[:, 2]
            safe = np.maximum(z, 1e-6)
            xy = (xyz @ K.T)[:, :2] / safe[:, None]
            j = np.zeros((n, 2, 3))
            j[:, 0, 0], j[:, 1, 1] = K[0, 0] / safe, K[1, 1] / safe
            j[:, 0, 2], j[:, 1, 2] = -K[0, 0] * xyz[:, 0] / safe ** 2, -K[1, 1] * xyz[:, 1] / safe ** 2
            world_j = j @ R
            screen_cov = world_j @ cov @ np.swapaxes(world_j, 1, 2)
            screen_cov = (screen_cov + np.swapaxes(screen_cov, 1, 2)) / 2 + np.eye(2) * 1e-10
            rendered, alpha = render(K, R, t, colors['colors'][-1], depth=True)
            inside = (z > 1e-4) & (xy[:, 0] >= 0) & (xy[:, 0] <= 191) & (xy[:, 1] >= 0) & (xy[:, 1] <= 143)
            sampled = bilinear(np.stack((alpha, rendered[..., 3], rendered[..., 3]), axis=-1).astype(np.float64), xy)
            visible = inside & (sampled[:, 0] >= .5) & (np.abs(z - sampled[:, 1]) <= .02 + .03 * sampled[:, 1])
            counts += visible
            sums[visible] += bilinear(context['rgb'][i].astype(np.float64), xy[visible])
            sample_valid = visible[sample_ids]
            chosen = sample_ids[sample_valid]
            if len(chosen):
                c = screen_cov[chosen]
                delta = pixels[None] - xy[chosen, None]
                matrix_options = (c, np.trace(c, axis1=1, axis2=2)[:, None, None] * np.eye(2) / 2,
                                  np.broadcast_to(np.eye(2), c.shape), np.zeros_like(c))
                method = 0
                for control in matrix_options:
                    for h in HS:
                        precision = np.linalg.inv(control + np.eye(2) * (1 / 12 + h * h))
                        exponent = -.5 * np.einsum('npi,nij,npj->np', delta, precision, delta)
                        weight = np.exp(exponent - exponent.max(axis=1, keepdims=True))
                        weight /= weight.sum(axis=1, keepdims=True)
                        sampled_sums[method, sample_valid] += weight @ context['rgb'][i].reshape(-1, 3).astype(np.float64)
                        method += 1
        np.testing.assert_array_equal(counts, colors['source_view_counts'])
        b = colors['colors'][-1].astype(np.float64).copy()
        b[counts > 0] = sums[counts > 0] / counts[counts > 0, None]
        bilinear_error = float(np.abs(b - colors['colors'][-2]).max())
        np.testing.assert_allclose(b, colors['colors'][-2], atol=2e-7, rtol=0)
        sampled_result = sampled_sums / counts[sample_ids][None, :, None]
        sampled_error = float(np.max(np.abs(sampled_result - colors['colors'][:16, sample_ids])))
        np.testing.assert_allclose(sampled_result, colors['colors'][:16, sample_ids], atol=3e-6, rtol=0)
        prediction_error = alpha_error = contribution_error = 0.
        for j, frame in enumerate(targets['frames']):
            for m, rgb in enumerate(colors['colors']):
                generated, generated_alpha = render(frame['K'], frame['R'], frame['t'], rgb)
                prediction_error = max(prediction_error, float(np.abs(generated - predictions['predictions'][m, j]).max()))
                alpha_error = max(alpha_error, float(np.abs(generated_alpha - predictions['alpha'][j]).max()))
            fallback_rgb = np.repeat((counts == 0)[:, None], 3, axis=1).astype(np.float32)
            generated, _ = render(frame['K'], frame['R'], frame['t'], fallback_rgb)
            contribution_error = max(contribution_error, float(np.abs(generated[..., 0] - predictions['fallback_contribution'][j]).max()))
        require(prediction_error <= 2e-6 and alpha_error <= 2e-6 and contribution_error <= 2e-6, 'render replay discrepancy')
    return dict(passed=True, source_view_counts_exact=True, bilinear_all_primitive_max_error=bilinear_error,
                directly_recomputed_sample_ids=sample_ids.tolist(), direct_mahalanobis_16_configurations_max_error=sampled_error,
                target_predictions_replayed=144, target_prediction_max_error=prediction_error,
                alpha_max_error=alpha_error, fallback_contribution_max_error=contribution_error,
                target_rgb_reads=0, local_model_or_runner_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('before-evaluation', 'after-evaluation'), default='before-evaluation')
    parser.add_argument('--runs', type=Path, default=ROOT / 'results/round9_gs_readout')
    parser.add_argument('--data', type=Path, default=ROOT / 'results/round9_context_data')
    parser.add_argument('--analysis', type=Path, default=ROOT / 'results/round9_analysis')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/round9_execution_audit')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    start = time.time()
    if args.phase == 'before-evaluation':
        require(not (args.analysis / 'selection.json').exists(), 'Calibration already exists; cannot claim before-evaluation audit')
    checked(args.runs / 'lock.json', EXPECTED_LOCK)
    lock = document(args.runs / 'lock.json')
    for key in ('source_hashes', 'data_hashes'):
        for path, expected in lock[key].items():
            checked(path, expected)
    require(lock['scenes'] == list(SCENES) and lock['methods'] == METHODS, 'Locked experiment contract')
    rows = []
    for scene in SCENES:
        row, payload = check_scene(scene, args.runs, args.data, lock)
        if args.replay:
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            row['replay'] = replay(payload, args.device)
        rows.append(row)
        print('AUDITED', scene, 'primitives', row['final_primitives'], 'replay', args.replay, flush=True)
    artifacts = {path: value for row in rows for path, value in row['artifacts'].items()}
    chronology = {'all_predictions_present_before_this_audit': True}
    if args.phase == 'after-evaluation':
        before_path = args.out / 'verification_before_evaluation.json'
        if before_path.exists():
            before = document(before_path)
            require(before['passed'] and before['artifact_hashes'] == artifacts, 'Before-evaluation predictions/artifacts changed')
            prior_auditor = 'this independent execution audit'
        else:
            before_path = ROOT / 'results/round9_pre_evaluation_integrity.json'
            before = document(before_path)
            require(before['passed'] and before['target_rgb_read'] is False, 'Prior root pre-evaluation audit failed')
            require(before['lock_sha256'] == EXPECTED_LOCK, 'Prior root audit experiment lock')
            for path, expected in artifacts.items():
                require(before['checked_artifacts'].get(path) == expected, 'Pre-evaluation root artifact hash changed')
            prior_auditor = 'root pre-evaluation integrity audit; this independent audit executed afterwards'
        selection_path = args.analysis / 'selection.json'
        checked(selection_path, (args.analysis / 'selection.sha256').read_text().strip())
        selection = document(selection_path)
        for scene in SCENES:
            for path, expected in selection['provenance'][scene]['artifact_hashes'].items():
                if path in artifacts:
                    require(artifacts[path] == expected, 'Calibration used different prediction/run artifact')
        require(before_path.stat().st_mtime_ns < selection_path.stat().st_mtime_ns, 'Before-evaluation audit timing')
        for scene in SCENES:
            require((args.runs / scene / 'predictions.npz').stat().st_mtime_ns < before_path.stat().st_mtime_ns,
                    'Prediction completion timing')
        chronology.update(calibration_uses_same_prediction_hashes=True, audit_precedes_selection_file=True,
                          before_evaluation_evidence=str(before_path), before_evaluation_auditor=prior_auditor,
                          timing_evidence='immutable hashes plus observed local file chronology; not an external timestamp authority')
    report = dict(passed=True, phase=args.phase, lock_sha256=EXPECTED_LOCK, scenes=rows,
                  artifact_hashes=artifacts, chronology=chronology, audit_source_sha256=digest(__file__),
                  target_rgb_reads=0, any_historic_test_data_read=False, seconds=time.time() - start,
                  limitations=['Full-image SfM intrinsics/poses are supplied calibration',
                               'This checks immutable execution artifacts; batch identity is independently replayed, not an OS file-access trace'])
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f'verification_{args.phase.replace("-", "_")}.json'
    require(not output.exists(), f'Refusing to overwrite {output}')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': True, 'phase': args.phase, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
