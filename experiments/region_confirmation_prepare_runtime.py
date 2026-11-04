"""Equivalent float64 CPU tensor arithmetic for large raw-input arrays."""
import argparse
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from experiments import round68_fixed_region_data as regional
from experiments import round60_data as sensor
from experiments import round68_fixed_region_run as engine
from experiments import region_confirmation_data as data
from experiments.region_confirmation_common import ROOT, OUT, PARENT, read, write, sha, preserved, check

RUNTIME = OUT / 'prepare_runtime_update_01'
def tensor_backproject(depth, intrinsics):
    values = torch.as_tensor(depth, dtype=torch.float64)
    ids = torch.nonzero((torch.isfinite(values) & (values > 0)).flatten(), as_tuple=False).flatten()
    y = torch.div(ids, values.shape[1], rounding_mode='floor')
    x = ids % values.shape[1]
    rays = torch.stack((x.double(), y.double(), torch.ones_like(x, dtype=torch.float64)), -1)
    rays = rays @ torch.linalg.inv(torch.as_tensor(intrinsics, dtype=torch.float64)).T
    points = rays * values.flatten()[ids, None]
    return dict(points=points.numpy(), pixel_ids=ids.numpy())


def tensor_sample(clouds, identity, realization, count=512, augment=False):
    assert count == 512 and not augment and all(len(c['points']) >= 512 for c in clouds)
    points = [torch.as_tensor(c['points'], dtype=torch.float64) for c in clouds]
    centers = [p.mean(0) for p in points]
    radius2 = [float((p - c).square().sum(-1).mean()) for p, c in zip(points, centers)]
    scale = max(float(np.sqrt(np.mean(radius2))), 1e-6)
    digest = hashlib.sha256(f"{sensor.CONFIG['data_salt']}:{identity}:{realization}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    raw, local, ids = [], [], []
    for cloud, p, center in zip(clouds, points, centers):
        chosen = rng.choice(len(p), 512, replace=False)
        selected = p[torch.as_tensor(chosen)]
        raw.append(selected.numpy())
        local.append(((selected - center) / scale).float().numpy())
        ids.append(cloud['pixel_ids'][chosen])
    return dict(points=np.stack(local), sensor=np.stack(raw), source_ids=np.stack(ids),
        origins=torch.stack(centers).numpy(), scale=np.array(scale),
        augment_rotations=np.stack((np.eye(3), np.eye(3))), augment_translations=np.zeros((2, 3)))


def bind():
    sensor.backproject = tensor_backproject
    regional.sample_inputs = tensor_sample


def validate():
    check()
    engine.configure()
    bind()
    assert not (RUNTIME / 'old_validation_check.json').exists()
    old = read(ROOT / 'results/round60_real_scene/attempt1/lock.json')
    entries = [copy.deepcopy(e) for e in old['entries'] if e['split'] == 'val']
    for entry in entries:
        entry['split'] = 'old_eval'
    started = time.time()
    arrays, sparse, sample, meta = data.build_arrays(dict(old, entries=entries),
        read(PARENT / 'data/strata_thresholds.json')['thresholds'])
    errors = {}
    with np.load(PARENT / 'data/val.npz', allow_pickle=False) as saved:
        assert len(arrays['ids']) == len(saved['ids']) == 64
        for key in saved.files:
            if key in ('ids', 'scenes', 'overlap', 'strata', 'permutation', 'frame_gaps'):
                np.testing.assert_array_equal(arrays[key], saved[key])
                errors[key] = 0.
            else:
                np.testing.assert_allclose(arrays[key], saved[key], atol=2e-5, rtol=2e-5)
                errors[key] = float(abs(arrays[key] - saved[key]).max())
    assert not meta['excluded']
    preserved(RUNTIME / 'old_validation_check.json', dict(passed=True, pairs=64,
        maximum_error_by_field=errors, seconds=time.time() - started, source_sha256=sha(__file__),
        original_source_sha256=sha(ROOT / 'experiments/round60_data.py'),
        baseline_sha256=sha(PARENT / 'data/val.npz'), new_model_scores=False,
        scope='The same float64 backprojection and input-centering/radius formulas use CPU tensors for large raw arrays; NumPy RNG, point identities, all regional construction and label definitions are unchanged.'))
    print('TENSOR_INPUTS_VALIDATED', 64, 'pairs', errors, flush=True)


def prepare():
    check()
    validation = read(RUNTIME / 'old_validation_check.json')
    assert validation['passed'] and validation['source_sha256'] == sha(__file__)
    engine.configure()
    bind()
    data.prepare()
    preserved(RUNTIME / 'data_runtime_identity.json', dict(source_sha256=sha(__file__),
        old_validation_check_sha256=sha(RUNTIME / 'old_validation_check.json'),
        data_manifest_sha256=sha(OUT / 'data_manifest.json'),
        evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
        change='Equivalent CPU float64 tensor backprojection and large-cloud means/radii; unchanged point RNG, region construction, thinning, label definitions, models and metrics.',
        completed_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('validate', 'prepare'))
    globals()[parser.parse_args().action]()
