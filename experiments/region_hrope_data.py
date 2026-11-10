"""Recover camera information for existing regions using old observations only."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image

from experiments.round65_common import ROOT, read, write, sha
from experiments.round58_data import tokenize
from experiments.round60_data import scene_reader, sample_inputs, attach_supervision
from experiments.round5_rgbd_data import read_sun3d_intrinsics

BASE = ROOT/'results/region_hrope_control'
PREP = BASE/'preparation'
CAMERA = BASE/'camera_data'
OLD = ROOT/'results/round68_fixed_region/attempt1'
SCENE_LOCK = ROOT/'results/round60_real_scene/attempt1/lock.json'
PLAN = ROOT/'docs/region_hrope_control_plan.md'
SPECS = (('fit0', 'fit', 0), ('fit1', 'fit', 1), ('val', 'val', 0), ('eval_base', 'old_eval', 0))


def camera_fields(source_ids, neighbors, intrinsics, sizes):
    """No poses, world centers, or correspondence labels accepted."""
    batch, views, _, _ = neighbors.shape
    member_pixels = source_ids[np.arange(batch)[:, None, None, None],
                               np.arange(views)[None, :, None, None], neighbors]
    height, width = sizes[..., 0, None, None], sizes[..., 1, None, None]
    x, y = member_pixels % width, member_pixels // width
    positions = np.stack((x/width-.5, y/height-.5), -1).mean(-2)
    k = intrinsics.astype(np.float64).copy()
    k[..., 0, :] /= sizes[..., 1, None]
    k[..., 1, :] /= sizes[..., 0, None]
    k[..., :2, 2] -= .5
    return k.astype(np.float32), positions.astype(np.float32)


def freeze():
    assert not (PREP/'data_lock.json').exists()
    manifest = read(OLD/'data/manifest.json')
    caches = {n: manifest['caches'][n] for n, _, _ in SPECS}
    for r in caches.values():
        assert sha(r['path']) == r['sha256']
    sources = [Path(__file__), PLAN, SCENE_LOCK, OLD/'data/manifest.json',
               ROOT/'experiments/round58_data.py', ROOT/'experiments/round58_common.py',
               ROOT/'experiments/round60_data.py', ROOT/'experiments/round60_common.py',
               ROOT/'experiments/round5_rgbd_data.py', ROOT/'experiments/round65_common.py']
    write(PREP/'data_lock.json', dict(created_unix=time.time(), caches=caches,
        source_sha256={str(p): sha(p) for p in sources},
        scene_lock_path=str(SCENE_LOCK), old_data_only=True, new_pixel_access=False,
        camera_input='mean original member pixel normalized to [-.5,.5]; normalized full intrinsics',
        goal_complete=False))
    (PREP/'data_lock.sha256').write_text(sha(PREP/'data_lock.json')+'\n')
    print('CAMERA_DATA_FROZEN', sha(PREP/'data_lock.json'), flush=True)


def check():
    assert sha(PREP/'data_lock.json') == (PREP/'data_lock.sha256').read_text().strip()
    lock = read(PREP/'data_lock.json')
    for path, digest in lock['source_sha256'].items():
        assert sha(path) == digest, path
    return lock


def prepare():
    lock = check()
    assert not CAMERA.exists(), 'Preserve previous camera reconstruction.'
    CAMERA.mkdir(parents=True)
    torch.set_num_threads(4)
    raw = read(SCENE_LOCK)
    manifest = {}
    for name, split, realization in SPECS:
        before = time.time()
        with np.load(lock['caches'][name]['path'], allow_pickle=False) as z:
            old = {k: z[k] for k in ('ids', 'features', 'centers', 'covariance', 'overlap')}
        ids, pending, matrices, positions = [], [], [], []
        errors = {k: 0. for k in ('features', 'centers', 'covariance')}
        pixel_error = 0.
        offset = 0

        def flush():
            nonlocal offset, pixel_error
            if not pending:
                return
            batch = {k: np.stack([x[k] for x in pending]) for k in pending[0]}
            with torch.no_grad():
                reference = tokenize({k: torch.from_numpy(batch[k]) for k in ('points', 'world', 'clean')})
            n = len(pending)
            for k in errors:
                actual = reference[k].numpy()
                errors[k] = max(errors[k], float(np.abs(actual-old[k][offset:offset+n]).max()))
            np.testing.assert_array_equal(reference['overlap'].numpy(), old['overlap'][offset:offset+n])
            assert max(errors.values()) < 2e-4, (name, offset, errors)
            sensor = batch['sensor']
            projected = np.einsum('bvij,bvnj->bvni', batch['camera_intrinsics'], sensor)
            projected = projected[..., :2]/projected[..., 2:]
            width = batch['image_sizes'][..., 1, None]
            expected = np.stack((batch['source_ids'] % width, batch['source_ids'] // width), -1)
            pixel_error = max(pixel_error, float(np.abs(projected-expected).max()))
            assert pixel_error < 1e-7, pixel_error
            k, xy = camera_fields(batch['source_ids'], reference['neighbors'].numpy(),
                                  batch['camera_intrinsics'], batch['image_sizes'])
            matrices.append(k)
            positions.append(xy)
            pending.clear()
            offset += n

        for entry in raw['entries']:
            if entry['split'] != split:
                continue
            reader, poses = scene_reader(entry, raw)
            intrinsic = read_sun3d_intrinsics(entry['intrinsics'])
            sizes = {}
            for frame in entry['frames']:
                with Image.open(frame['path']) as im:
                    sizes[frame['index']] = (im.height, im.width)
            for q, k in entry['pairs']:
                identity = f"{entry['scene']}:{q}:{k}"
                clouds = [reader(q), reader(k)]
                if any(len(c['points']) < 512 for c in clouds):
                    continue
                observed = sample_inputs(clouds, identity, realization, count=512, augment=split == 'fit')
                # Poses are used only by the independent old-label consistency check.
                item = attach_supervision(observed, poses[[q, k]])
                item.update(camera_intrinsics=np.stack((intrinsic, intrinsic)),
                            image_sizes=np.asarray((sizes[q], sizes[k])))
                pending.append(item)
                ids.append(identity)
                if len(pending) == 16:
                    flush()
            reader.cache_clear()
            print('CAMERA_RECONSTRUCTED', name, entry['scene'], len(ids), flush=True)
        flush()
        np.testing.assert_array_equal(np.asarray(ids), old['ids'])
        assert offset == len(old['ids'])
        target = CAMERA/(name+'.npz')
        np.savez_compressed(target, ids=old['ids'], intrinsics=np.concatenate(matrices),
                            image_positions=np.concatenate(positions))
        manifest[name] = dict(path=str(target), sha256=sha(target), pairs=len(ids),
            original_cache_sha256=lock['caches'][name]['sha256'],
            old_region_max_errors=errors, all_overlap_labels_identical=True,
            maximum_original_pixel_reprojection_error=pixel_error,
            seconds=time.time()-before)
        print('CAMERA_CACHE_COMPLETE', name, manifest[name], flush=True)
    write(CAMERA/'manifest.json', dict(caches=manifest, data_lock_sha256=sha(PREP/'data_lock.json'),
        completed_unix=time.time(), new_pixel_access=False, fresh_confirmation=False))


def load_cache(name):
    lock = check()
    manifest = read(CAMERA/'manifest.json')
    assert manifest['data_lock_sha256'] == sha(PREP/'data_lock.json')
    old, camera = lock['caches'][name], manifest['caches'][name]
    assert sha(old['path']) == old['sha256'] and sha(camera['path']) == camera['sha256']
    with np.load(old['path'], allow_pickle=False) as z:
        result = {k: z[k] for k in z.files}
    with np.load(camera['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(result['ids'], z['ids'])
        result.update(intrinsics=z['intrinsics'], image_positions=z['image_positions'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'prepare'))
    globals()[parser.parse_args().action]()
