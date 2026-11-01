"""Independent NumPy/SciPy audit of real-GS projected RGB reading.

Reads context RGB, frozen GS checkpoints, and exported operator samples only.
No target image, target prediction, training code, or projection implementation
is imported. Extra numerical checks import the PE readout under test, but the
primary comparison is against the stored experiment outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch


ROOT = Path(__file__).resolve().parents[1]
SCENES = ('teatime', 'waldo_kitchen')
ROLES = ('full', 'isotropic', 'fixed', 'point')
BANDWIDTHS = (.25, .5, 1., 2.)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def assert_close(actual, expected, label, *, atol, rtol=0.):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError(f'{label}: shape or finite-value failure')
    error = float(np.max(np.abs(actual-expected), initial=0.))
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f'{label}: maximum error {error:g} exceeds atol={atol:g}, rtol={rtol:g}')
    return error


def covariance_reference(log_scales, scalar_first_quaternions):
    # SciPy independently normalizes x,y,z,w quaternions and builds rotations.
    rotation = Rotation.from_quat(scalar_first_quaternions[:, [1, 2, 3, 0]]).as_matrix()
    variances = np.exp(2 * log_scales)
    return np.einsum('nij,nj,nkj->nik', rotation, variances, rotation)


def projection_reference(means, covariance, K, R, t):
    camera = means @ R.T + t
    homogeneous = camera @ K.T
    xy = homogeneous[:, :2] / homogeneous[:, 2, None]
    # Differentiate the full homogeneous camera map, then map world variations.
    derivative = ((K[None, :2, :] * homogeneous[:, 2, None, None]
                   - homogeneous[:, :2, None] * K[None, 2:3, :])
                  / homogeneous[:, 2, None, None]**2)
    jacobian = derivative @ R
    projected = jacobian @ covariance @ np.swapaxes(jacobian, -1, -2)
    projected = (projected + np.swapaxes(projected, -1, -2)) / 2
    return xy, projected + np.eye(2)[None] * 1e-10, camera[:, 2]


def mahalanobis_reference(image, means, covariance, role='full', bandwidth=.25):
    """Dense all-pixel direct Gaussian average; no lift or truncation."""
    height, width, _ = image.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    colors = []
    for mean, cov in zip(means, covariance):
        if role == 'point':
            selected_cov = np.zeros((2, 2))
        elif role == 'fixed':
            selected_cov = np.eye(2)
        elif role == 'isotropic':
            selected_cov = np.eye(2) * np.trace(cov) / 2
        elif role == 'full':
            selected_cov = cov
        else:
            raise ValueError(role)
        total = selected_cov + np.eye(2) * (1/12 + bandwidth**2)
        displacement = pixels - mean
        energy = -.5 * np.sum(displacement * np.linalg.solve(total, displacement.T).T, axis=1)
        weight = np.exp(energy - energy.max())
        weight /= weight.sum()
        colors.append(weight @ image.reshape(-1, 3))
    return np.asarray(colors).reshape(-1, 3)


def bilinear_reference(image, means):
    output = []
    height, width, _ = image.shape
    for x, y in means:
        if not 0 <= x <= width-1 or not 0 <= y <= height-1:
            raise ValueError('An audit query lies outside bilinear support')
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0+1, width-1), min(y0+1, height-1)
        dx, dy = x-x0, y-y0
        output.append((1-dx)*(1-dy)*image[y0, x0] + dx*(1-dy)*image[y0, x1]
                      + (1-dx)*dy*image[y1, x0] + dx*dy*image[y1, x1])
    return np.asarray(output).reshape(-1, 3)


def audit_scene(scene, experiment, data_root):
    directory = experiment / scene
    paths = {name: directory/name for name in ('state.pt', 'gaussians.npz', 'training.json',
                                              'operator_samples.json', 'run.json')}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Readout must finish before audit: '+', '.join(missing))
    training, record = read_json(paths['training.json']), read_json(paths['run.json'])
    for filename, source, key in [('state.pt', training, 'state_sha256'),
                                   ('gaussians.npz', training, 'gaussians_sha256'),
                                   ('training.json', record, 'training_sha256'),
                                   ('operator_samples.json', record, 'operator_samples_sha256')]:
        if sha(paths[filename]) != source[key]:
            raise ValueError(f'{scene}: {filename} hash mismatch')
    context_path = data_root/scene/'context.npz'
    expected_hash = record['data_hashes'][str(context_path)]
    if sha(context_path) != expected_hash:
        raise ValueError('Context archive hash mismatch')
    with np.load(context_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if data['rgb'].shape != (24, 144, 192, 3):
        raise ValueError('Unexpected context image shape')
    state = {key: value.numpy().astype(np.float64)
             for key, value in torch.load(paths['state.pt'], weights_only=True, map_location='cpu').items()}
    world_covariance = covariance_reference(state['log_scales'], state['quaternions'])
    with np.load(paths['gaussians.npz'], allow_pickle=False) as archive:
        archive_errors = {
            'means': assert_close(archive['means'], state['means'], 'archived means', atol=0),
            'quaternions': assert_close(archive['quaternions'], state['quaternions'], 'archived quaternion', atol=0),
            'scales': assert_close(archive['scales'], np.exp(state['log_scales']), 'archived scale', atol=3e-8, rtol=3e-7),
            'covariance_float32': assert_close(archive['covariances'], world_covariance, 'archived covariance', atol=3e-8, rtol=2e-5),
        }
    samples = read_json(paths['operator_samples.json'])
    index_by_id = {int(v): i for i, v in enumerate(data['ids'])}
    view_by_id = {int(row['id']): row for row in record['views']}
    expected_sources = {key for key, value in view_by_id.items() if value['visible_primitive_count'] > 0}
    sampled_sources = [int(row['source_id']) for row in samples]
    if len(set(sampled_sources)) != len(sampled_sources) or set(sampled_sources) != expected_sources:
        raise ValueError('Missing or repeated source audit records')
    checks = []
    extras = []
    # Main replay uses only stored outputs and independent NumPy formulas.
    for row in samples:
        source_id = int(row['source_id'])
        source_index = index_by_id[source_id]
        ids = np.asarray(row['primitive_ids'], dtype=np.int64)
        if len(ids) != min(8, view_by_id[source_id]['visible_primitive_count']) or len(np.unique(ids)) != len(ids):
            raise ValueError('Unexpected number of primitive samples')
        # The fixed runner uses float32 calibrated matrices for rasterization
        # and casts those same values to double for readout projection.
        K, R, t = (data[k][source_index].astype(np.float32).astype(np.float64) for k in ('K', 'R', 't'))
        xy, covariance, depth = projection_reference(state['means'][ids], world_covariance[ids], K, R, t)
        mean_error = assert_close(np.asarray(row['mean']), xy, 'projected mean', atol=2e-11, rtol=2e-13)
        covariance_error = assert_close(np.asarray(row['covariance']), covariance, 'projected covariance', atol=2e-9, rtol=2e-11)
        if not np.all(depth > 1e-4) or not np.all((xy >= 0) & (xy <= [191, 143])):
            raise ValueError('Selected audit primitive violates geometric visibility')
        image = data['rgb'][source_index].astype(np.float64)
        reference = mahalanobis_reference(image, xy, covariance)
        color_error = assert_close(np.asarray(row['full_h025']), reference, 'stored full h=.25 RGB', atol=2e-8)
        checks.append(dict(source_id=source_id,primitive_ids=ids.tolist(),sample_count=len(ids),
                           maximum_mean_error=mean_error,maximum_covariance_error=covariance_error,
                           maximum_rgb_error=color_error))
        if source_id in (sampled_sources[0], sampled_sources[-1]):
            selection = np.unique(np.array([0, len(ids)//2, len(ids)-1]))
            extras.append((source_id, ids[selection], image, xy[selection], covariance[selection]))
    # Additional tests compare all methods and h values on six actual points
    # per scene; they are tests of the operator, not fresh model predictions.
    from models.round9_gaussian_readout import gaussian_readout, bilinear_readout
    extra_results = []
    for source_id, ids, image, xy, covariance in extras:
        ti, tx, tc = (torch.from_numpy(a) for a in (image, xy, covariance))
        for role in ROLES:
            for bandwidth in BANDWIDTHS:
                actual = gaussian_readout(ti, tx, tc, method=role, bandwidth=bandwidth,
                                          query_chunk_size=2, key_chunk_size=1237, compute_dtype=torch.float64)
                expected = mahalanobis_reference(image, xy, covariance, role, bandwidth)
                if not actual.valid.all():
                    raise ValueError('Unexpected invalid control audit query')
                error = assert_close(actual.rgb.numpy(), expected, 'extra '+role, atol=2e-8)
                extra_results.append(dict(source_id=source_id,primitive_ids=ids.tolist(),role=role,
                                          bandwidth=bandwidth,maximum_rgb_error=error))
        actual = bilinear_readout(ti, tx)
        expected = bilinear_reference(image, xy)
        if not actual.valid.all():
            raise ValueError('Unexpected invalid bilinear audit query')
        error = assert_close(actual.rgb.numpy(), expected, 'extra bilinear', atol=2e-14)
        extra_results.append(dict(source_id=source_id,primitive_ids=ids.tolist(),role='bilinear',
                                  maximum_rgb_error=error))
    return dict(scene=scene,passed=True,source_views=len(checks),
                primary_primitive_source_pairs=sum(c['sample_count'] for c in checks),
                maximum_mean_error=max(c['maximum_mean_error'] for c in checks),
                maximum_covariance_error=max(c['maximum_covariance_error'] for c in checks),
                maximum_rgb_error=max(c['maximum_rgb_error'] for c in checks),
                archive_errors=archive_errors,primary_checks=checks,extra_checks=extra_results,
                maximum_extra_rgb_error=max(c['maximum_rgb_error'] for c in extra_results),
                input_sha256={str(path):sha(path) for path in [*paths.values(),context_path]},
                target_rgb_read=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, default=ROOT/'results/round9_gs_readout')
    parser.add_argument('--data', type=Path, default=ROOT/'results/round9_context_data')
    parser.add_argument('--out', type=Path, default=ROOT/'results/round9_operator_audit')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Refuse overwrite '+str(args.out))
    torch.set_num_threads(1)
    start = time.monotonic()
    lock_path = args.experiment/'lock.json'
    lock = read_json(lock_path)
    for path, expected in lock['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: '+path)
    scenes = [audit_scene(scene, args.experiment, args.data) for scene in SCENES]
    result = dict(passed=True,scenes=scenes,seconds=time.monotonic()-start,
                  audit_source_sha256=sha(__file__),lock_sha256=sha(lock_path),
                  primary_primitive_source_pairs=sum(s['primary_primitive_source_pairs'] for s in scenes),
                  maximum_stored_rgb_error=max(s['maximum_rgb_error'] for s in scenes),
                  maximum_projection_mean_error=max(s['maximum_mean_error'] for s in scenes),
                  maximum_projection_covariance_error=max(s['maximum_covariance_error'] for s in scenes),
                  maximum_extra_control_rgb_error=max(s['maximum_extra_rgb_error'] for s in scenes),
                  reference='Independent NumPy float64 direct Mahalanobis all-pixel averaging; SciPy quaternion rotation; full homogeneous Jacobian',
                  source_mask='All 192x144 source pixels; sampled queries passed fixed original-run visibility',
                  visibility_rerendered=False,target_rgb_read=False,target_predictions_read=False,
                  trainer_or_projection_implementation_imported=False,
                  production_readout_imported_for_extra_checks_only=True,
                  limitations=['Audits exported representative primitive/source pairs, not every primitive.',
                               'Does not independently rerender alpha/depth visibility or final target rendering.'])
    args.out.mkdir(parents=True)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='scenes'},indent=2),flush=True)


if __name__ == '__main__':
    main()
