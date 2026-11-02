"""Official-train-only data preparation; no official-test HDF5 is opened."""
import argparse
import hashlib
import io
from pathlib import Path
import re
import zipfile
import numpy as np
import h5py

from experiments.round48_acquire import ARCHIVE, EXPECTED_SHA
from experiments.round48_common import OUT, CONFIG, checked, sha, write, read


def split_indices(labels, ids):
    result = {name: [] for name in ('fit', 'val', 'dev_eval')}
    for label in range(40):
        rows = np.flatnonzero(labels == label).tolist()
        rows.sort(key=lambda i: hashlib.sha256(
            (CONFIG['split_salt']+':'+ids[i]).encode()).hexdigest())
        n = max(1, len(rows)//10)
        if len(rows) < 3*n:
            raise ValueError('class too small for prespecified stratified split')
        result['val'] += rows[:n]
        result['dev_eval'] += rows[n:2*n]
        result['fit'] += rows[2*n:]
    return {k: np.asarray(sorted(v), dtype=np.int64) for k, v in result.items()}


def canonical_indices(identity):
    digest = hashlib.sha256((CONFIG['sample_salt']+':'+identity).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little')).permutation(2048)


def prepare(out=OUT):
    checked(out)
    if sha(ARCHIVE) != EXPECTED_SHA:
        raise ValueError('archive identity')
    dest = out/'data'
    if dest.exists():
        raise ValueError('preserve prior data preparation')
    dest.mkdir()
    points, labels, ids, members = [], [], [], []
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = sorted(n for n in archive.namelist() if re.fullmatch(
            r'modelnet40_ply_hdf5_2048/ply_data_train[0-4]\.h5', n))
        if len(names) != 5:
            raise ValueError('expected five official training HDF5 members')
        for name in names:
            payload = archive.read(name)
            with h5py.File(io.BytesIO(payload), 'r') as handle:
                x = np.asarray(handle['data'], dtype=np.float32)
                y = np.asarray(handle['label'], dtype=np.int64).reshape(-1)
            if x.shape != (len(y), 2048, 3) or not np.isfinite(x).all():
                raise ValueError('point array shape/finiteness')
            if y.min() < 0 or y.max() > 39:
                raise ValueError('class range')
            points.append(x)
            labels.append(y)
            ids.extend(name+':'+str(i) for i in range(len(y)))
            members.append(dict(name=name, sha256=hashlib.sha256(payload).hexdigest(), count=len(y)))
    points, labels = np.concatenate(points), np.concatenate(labels)
    if len(points) != 9840 or len(set(ids)) != len(ids):
        raise ValueError('official-train population identity')
    split = split_indices(labels, ids)
    cloud_hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in points]
    assigned = {}
    for name, indices in split.items():
        for i in indices:
            previous = assigned.setdefault(cloud_hashes[i], name)
            if previous != name:
                raise ValueError('byte-identical clouds cross development partitions')
    records = []
    for name, indices in split.items():
        path = dest/(name+'.npz')
        identities = np.asarray([ids[i] for i in indices])
        orders = np.stack([canonical_indices(identity) for identity in identities])
        np.savez(path, points=points[indices], labels=labels[indices], ids=identities,
                 canonical=orders.astype(np.int16), source_rows=indices)
        records.append(dict(split=name, path=str(path), sha256=sha(path), count=len(indices),
                            class_counts=np.bincount(labels[indices], minlength=40).tolist(),
                            ids=identities.tolist()))
    manifest = dict(archive_sha256=EXPECTED_SHA, members_opened=members, partitions=records,
                    total=9840, official_test_members_opened=[], all_40_classes=True,
                    raw_development_arrays_partitioned=True,
                    training_and_selection_use_only=['fit', 'val'],
                    model_dev_eval_requires_all_checkpoints_sealed=True,
                    cross_partition_identical_clouds=False)
    write(dest/'manifest.json', manifest)
    (dest/'manifest.sha256').write_text(sha(dest/'manifest.json')+'\n')
    print('PREPARED', {x['split']:x['count'] for x in records}, flush=True)


def load(split, out=OUT):
    checked(out)
    if split not in ('fit', 'val', 'dev_eval'):
        raise ValueError('only declared official-train development partitions are supported')
    if split == 'dev_eval' and not (out/'audit_training/completion.json').is_file():
        raise ValueError('seal all selected models before development evaluation')
    path = out/'data/manifest.json'
    if sha(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('data manifest identity')
    record = next(r for r in read(path)['partitions'] if r['split'] == split)
    if sha(record['path']) != record['sha256']:
        raise ValueError('data identity')
    with np.load(record['path'], allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files}
    if result['ids'].tolist() != record['ids']:
        raise ValueError('data identity ordering')
    return result


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def evaluation_points(bundle, variant):
    count = 256 if variant == 'points256' else 1024 if variant == 'points1024' else 512
    points = np.take_along_axis(bundle['points'], bundle['canonical'][:, :count, None], axis=1).copy()
    if variant == 'y90':
        points = points@rotation_y(np.pi/2).T
    elif variant == 'so3':
        a, b = np.pi/6, np.pi/4
        rx = np.asarray([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        rz = np.asarray([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
        points = points@(rz@rx).astype(np.float32).T
    elif variant == 'jitter02':
        for i, identity in enumerate(bundle['ids']):
            digest = hashlib.sha256(('R48-jitter:'+identity).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
            points[i] += rng.normal(0, .02, points[i].shape).clip(-.1, .1).astype(np.float32)
    elif variant == 'permuted':
        points = points[:, ::-1].copy()
    elif variant not in ('base', 'points256', 'points1024'):
        raise ValueError(variant)
    return np.ascontiguousarray(points, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    prepare(parser.parse_args().out)
