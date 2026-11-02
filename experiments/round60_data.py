"""Pose-independent observed inputs; sensor points and poses define labels only."""
import argparse
from functools import lru_cache
import hashlib
import time
import numpy as np
import torch
from experiments.round5_rgbd_data import load_sensor_depth, read_sun3d_intrinsics, read_sun3d_extrinsics
from experiments.round58_data import tokenize, random_rotation
from experiments.round60_common import OUT, CONFIG, CACHE_NAMES, checked, read, write, sha


def backproject(depth, intrinsics):
    valid = np.isfinite(depth) & (depth > 0)
    ids = np.flatnonzero(valid)
    y, x = np.unravel_index(ids, depth.shape)
    ray = np.column_stack((x, y, np.ones(len(ids))))@np.linalg.inv(intrinsics).T
    points = ray*depth.ravel()[ids, None]
    return dict(points=points, pixel_ids=ids.astype(np.int64))


def sample_inputs(clouds, identity, realization, count=512, augment=False):
    """No camera poses, masks from another view, or correspondences accepted."""
    if count not in (256, 512) or any(len(c['points']) < 512 for c in clouds):
        raise ValueError('input availability/count')
    centers = np.stack([c['points'].mean(0) for c in clouds])
    radius2 = [np.square(c['points']-center).sum(-1).mean() for c, center in zip(clouds, centers)]
    scale = max(float(np.sqrt(np.mean(radius2))), 1e-6)
    seed = hashlib.sha256(f"{CONFIG['data_salt']}:{identity}:{realization}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], 'little'))
    raw, local, ids, rotations, translations = [], [], [], [], []
    for c, center in zip(clouds, centers):
        selected = rng.choice(len(c['points']), 512, replace=False)[:count]
        rotation = random_rotation(rng) if augment else np.eye(3)
        translation = rng.uniform(-.5, .5, 3) if augment else np.zeros(3)
        sensor = c['points'][selected]
        raw.append(sensor);ids.append(c['pixel_ids'][selected])
        local.append(((sensor-center)/scale)@rotation.T+translation)
        rotations.append(rotation);translations.append(translation)
    return dict(points=np.stack(local).astype(np.float32), sensor=np.stack(raw),
        source_ids=np.stack(ids), origins=centers, scale=np.array(scale),
        augment_rotations=np.stack(rotations), augment_translations=np.stack(translations))


def attach_supervision(inputs, poses):
    # Preserve the already-materialized forward input. Pose only supplies labels.
    world = (np.einsum('vni,vji->vnj', inputs['sensor'], poses[:, :3, :3])+
             poses[:, None, :3, 3]-poses[0, :3, 3])/inputs['scale']
    return dict(inputs, world=world.astype(np.float32), clean=world.astype(np.float32), poses=poses)


def load_cache(name):
    if name not in CACHE_NAMES:
        raise ValueError('R60 cache scope')
    if name.startswith('eval_') and not (OUT/'selected.json').is_file():
        raise ValueError('target selections precede old_eval')
    rec = read(OUT/'data'/f'{name}.json');path = OUT/'data'/f'{name}.npz'
    if sha(path) != rec['sha256']:
        raise ValueError('R60 cache identity')
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def scene_reader(entry, lock):
    for name in ('intrinsics', 'poses'):
        if sha(entry[name]) != lock['raw_hashes'][entry[name]]:
            raise ValueError('camera input changed')
    k = read_sun3d_intrinsics(entry['intrinsics'])
    poses = read_sun3d_extrinsics(entry['poses'])
    frames = {f['index']: f for f in entry['frames']}
    @lru_cache(maxsize=8)
    def load(index):
        f = frames[index]
        if sha(f['path']) != f['sha256']:
            raise ValueError('depth input changed')
        return backproject(load_sensor_depth(f['path']), k)
    return load, poses


def prepare(stage):
    lock = checked()
    if stage == 'eval' and not (OUT/'selected.json').is_file():
        raise ValueError('all target selections required')
    specs = [('fit0','fit',0,512,True), ('fit1','fit',1,512,True), ('val','val',0,512,False)]
    if stage == 'eval':
        specs = [('eval_base','old_eval',0,512,False), ('eval_sparse256','old_eval',0,256,False)]
    dest = OUT/'data';dest.mkdir(exist_ok=True)
    for name, split, realization, count, augment in specs:
        path = dest/f'{name}.npz'
        if path.exists() or path.with_suffix('.json').exists():
            raise ValueError('preserve existing cache '+name)
        collected, identities, scene_names, scales, gaps, excluded = {}, [], [], [], [], []
        pending, sample_saved, started = [], False, time.time()
        def flush():
            nonlocal sample_saved
            if not pending:
                return
            batch = {k:torch.as_tensor(np.stack([p[k] for p in pending]),device='cuda') for k in pending[0]}
            result = tokenize(batch)
            for key,value in result.items():
                if key not in ('neighbors','anchors'):
                    collected.setdefault(key,[]).append(value.cpu().numpy())
            if not sample_saved:
                np.savez_compressed(dest/f'{name}_sample.npz',
                    **{k:v[:2].cpu().numpy() for k,v in batch.items()},
                    **{'token_'+k:v[:2].cpu().numpy() for k,v in result.items()})
                sample_saved = True
            pending.clear()
        for entry in lock['entries']:
            if entry['split'] != split:
                continue
            load, poses = scene_reader(entry, lock)
            accepted = 0
            for q,k in entry['pairs']:
                identity = f"{entry['scene']}:{q}:{k}"
                clouds = [load(q),load(k)]
                if any(len(c['points']) < 512 for c in clouds):
                    excluded.append(dict(id=identity,reason='fewer_than_512_observed_points',
                                         counts=[len(c['points']) for c in clouds]));continue
                inputs = sample_inputs(clouds,identity,realization,count,augment)
                pending.append(attach_supervision(inputs,poses[[q,k]]))
                identities.append(identity);scene_names.append(entry['scene']);scales.append(float(inputs['scale']))
                gaps.append(k-q);accepted += 1
                if len(pending) == CONFIG['cache_batch_size']:
                    flush()
            load.cache_clear()
            print('DATA',name,entry['scene'],accepted,'/',len(entry['pairs']),flush=True)
        flush()
        if not identities:
            raise ValueError('no usable observation pairs')
        arrays = {k:np.concatenate(v) for k,v in collected.items()}
        arrays.update(ids=np.asarray(identities),scenes=np.asarray(scene_names),scales=np.asarray(scales),
                      frame_gaps=np.asarray(gaps))
        if any(not np.isfinite(v).all() for k,v in arrays.items() if k not in ('ids','scenes')):
            raise ValueError('nonfinite native cache')
        np.savez(path,**arrays)
        overlap = arrays['overlap'];counts = [(overlap.max(axis=ax) >= .3).sum(axis=1) for ax in (2,1)]
        write(path.with_suffix('.json'),dict(name=name,split=split,realization=realization,points=count,
            sha256=sha(path),count=len(identities),excluded=excluded,valid_query_counts=[int(x.sum()) for x in counts],
            zero_valid_pair_count=int(((counts[0]+counts[1])==0).sum()),seconds=time.time()-started,
            units='pair-normalized scene units',sensor_measurement_noise_in_labels=True,
            pose_in_forward=False,fresh_confirmation=False,lock_sha256=sha(OUT/'lock.json')))
        print('CACHED',name,len(identities),'valid',[int(x.sum()) for x in counts],flush=True)
    names = [x[0] for x in specs]
    write(dest/f'{stage}_manifest.json',dict(caches={n:sha(dest/f'{n}.npz') for n in names},
        records={n:sha(dest/f'{n}.json') for n in names},completed_unix=time.time()))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('fit','eval'))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    prepare(parser.parse_args().stage)
