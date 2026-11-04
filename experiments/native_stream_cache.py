"""One-object raw staging into the unchanged native-GS cache representation.

This module has no acquisition CLI. Its caller supplies an explicit population;
current validation uses only old TRAIN objects. Original raw files are read-only.
"""
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

import numpy as np

from experiments.round65_common import CONFIG, sha
from experiments.round65_data import native_arrays, read_vertices


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def schema(count, tokens):
    return dict(centers=(np.float32, (count, tokens, 3)),
                covariance=(np.float64, (count, tokens, 3, 3)),
                appearance=(np.float32, (count, tokens, 4)),
                indices=(np.int32, (count, tokens)), labels=(np.int64, (count,)))


def row_hashes(arrays, index):
    return {name:hashlib.sha256(np.asarray(value[index]).tobytes()).hexdigest()
            for name, value in arrays.items()}


class LocalProvider:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def __call__(self, row):
        path = (self.root / row['file_path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('source outside raw root')
        return path.open('rb')


class MirrorProvider:
    def __init__(self, rows, transport_directory):
        from experiments.round65_mirror_acquire import DirectMirror
        from experiments.round65_acquire import REPO, REVISION
        transport_directory.mkdir(parents=True, exist_ok=False)
        self.transport = DirectMirror([r['file_path'] for r in rows], transport_directory)
        self.prefix = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'

    def __call__(self, row):
        return self.transport.urlopen(urllib.request.Request(self.prefix + row['file_path']), timeout=45)


def stage_object(row, directory, provider, minimum_free_bytes):
    # directory is a private child of the identity-marked cache directory.
    pending, verified = directory / 'raw.pending', directory / 'raw.verified.ply'
    if shutil.disk_usage(directory).free < minimum_free_bytes + row['size']:
        raise OSError('insufficient space for one staged object plus declared reserve')
    for path in (pending, verified):
        if path.is_symlink():
            raise ValueError('unexpected staging symlink')
        if path.exists():
            path.unlink()  # Only our previous, bounded temporary object.
    atomic_json(directory / 'identity.json', dict(file_path=row['file_path'], sha256=row['sha256']))
    errors = []
    for attempt in range(3):
        try:
            digest, size = hashlib.sha256(), 0
            with provider(row) as source, pending.open('wb') as sink:
                while True:
                    chunk = source.read(2**20)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > row['size']:
                        raise ValueError('source exceeds declared size')
                    sink.write(chunk)
                    digest.update(chunk)
                sink.flush()
                os.fsync(sink.fileno())
            if size < row['size']:
                raise EOFError('source shorter than declared size')
            if digest.hexdigest() != row['sha256']:
                raise ValueError('source SHA256 mismatch')
            pending.replace(verified)
            return verified, errors
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead, EOFError) as error:
            errors.append(dict(attempt=attempt, error_type=type(error).__name__))
            atomic_json(directory / 'transport_errors.json', errors)
            if attempt == 2:
                raise
            time.sleep(min(8, 2**attempt))


def build_cache(rows, destination, provider, *, tokens=2048,
                minimum_free_bytes=50*2**30, after_row=None):
    """Resume committed rows; preserve a bounded failed object for diagnosis."""
    destination = Path(destination)
    rows = list(rows)
    if not rows or len({r['file_path'] for r in rows}) != len(rows):
        raise ValueError('population must be nonempty and unique')
    for row in rows:
        path = Path(row['file_path'])
        if (len(path.parts) != 2 or path.is_absolute() or '..' in path.parts
                or path.suffix != '.ply' or int(path.parts[0]) != row['taxonomy_id']):
            raise ValueError('invalid catalog path')
        if row['size'] <= 0 or len(row['sha256']) != 64:
            raise ValueError('invalid catalog identity')
    identity = dict(rows=rows, tokens=tokens, covariance_floor=CONFIG['covariance_floor'],
                    sample_salt=CONFIG['sample_salt'], converter_sha256=sha(Path(__file__).with_name('round65_data.py')),
                    implementation_sha256=sha(__file__), minimum_free_bytes=minimum_free_bytes)
    encoded = json.dumps(identity, sort_keys=True).encode()
    identity_sha = hashlib.sha256(encoded).hexdigest()
    marker = destination / 'identity.json'
    if destination.exists():
        if not marker.exists() or json.loads(marker.read_text()) != dict(sha256=identity_sha, **identity):
            raise ValueError('existing cache identity differs or is unowned')
    else:
        destination.mkdir(parents=True)
        atomic_json(marker, dict(sha256=identity_sha, **identity))
    stage = destination / 'raw_stage'
    if stage.is_symlink():
        raise ValueError('unexpected staging directory symlink')
    stage.mkdir(exist_ok=True)
    shapes = schema(len(rows), tokens)
    required = sum(np.dtype(dtype).itemsize * int(np.prod(shape)) for dtype,shape in shapes.values())
    if not (destination / 'progress.json').exists():
        if shutil.disk_usage(destination).free < minimum_free_bytes + required + max(r['size'] for r in rows):
            raise OSError('insufficient space for complete cache, one raw object and reserve')
        for name, (dtype, shape) in shapes.items():
            path = destination / (name + '.npy')
            if not path.exists():
                array = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
                array.flush()
                del array
            with path.open('r+b') as stream:
                os.posix_fallocate(stream.fileno(), 0, path.stat().st_size)
        atomic_json(destination / 'progress.json', dict(identity_sha256=identity_sha, completed=[]))
    progress = json.loads((destination / 'progress.json').read_text())
    if progress['identity_sha256'] != identity_sha or len(progress['completed']) > len(rows):
        raise ValueError('invalid cache journal')
    arrays = {name:np.load(destination / (name + '.npy'), mmap_mode='r+') for name in shapes}
    for name, (dtype, shape) in shapes.items():
        if arrays[name].dtype != np.dtype(dtype) or arrays[name].shape != shape:
            raise ValueError('cache array schema differs')
    for index, record in enumerate(progress['completed']):
        if (record['file_path'] != rows[index]['file_path'] or record['source_sha256'] != rows[index]['sha256']
                or record['array_row_sha256'] != row_hashes(arrays, index)):
            raise ValueError('committed cache row changed')
    # A process may stop between journal commit and removal of its staged copy.
    if (stage / 'identity.json').exists():
        staged_identity = json.loads((stage / 'identity.json').read_text())
        committed = {r['file_path']:r['source_sha256'] for r in progress['completed']}
        if committed.get(staged_identity['file_path']) == staged_identity['sha256']:
            for name in ('raw.pending', 'raw.verified.ply'):
                path = stage / name
                if path.is_symlink():
                    raise ValueError('unexpected staging symlink')
                if path.exists():
                    path.unlink()
    if (destination / 'manifest.json').exists():
        manifest = json.loads((destination / 'manifest.json').read_text())
        if len(progress['completed']) != len(rows) or manifest['identity_sha256'] != identity_sha:
            raise ValueError('incomplete cache marked complete')
        for name, digest in manifest['array_sha256'].items():
            if sha(destination / name) != digest:
                raise ValueError('completed array changed')
        return manifest
    for index in range(len(progress['completed']), len(rows)):
        row = rows[index]
        staged, errors = stage_object(row, stage, provider, minimum_free_bytes)
        vertices = read_vertices(staged)
        try:
            values, stats = native_arrays(vertices, row['file_path'], tokens)
        finally:
            vertices._mmap.close()
        for name, value in values.items():
            arrays[name][index] = value
        arrays['labels'][index] = row['taxonomy_id']
        for name, value in arrays.items():
            value.flush()
            with (destination / (name + '.npy')).open('rb') as stream:
                os.fsync(stream.fileno())
        progress['completed'].append(dict(file_path=row['file_path'], source_sha256=row['sha256'],
            stats=stats, array_row_sha256=row_hashes(arrays, index), transport_errors=errors))
        atomic_json(destination / 'progress.json', progress)
        staged.unlink()
        print('CACHED', index + 1, '/', len(rows), row['file_path'], flush=True)
        if after_row is not None:
            after_row(index, progress)
    manifest = dict(complete=True, identity_sha256=identity_sha, count=len(rows), ids=[r['file_path'] for r in rows],
        array_sha256={name + '.npy':sha(destination / (name + '.npy')) for name in shapes},
        schema={name:dict(dtype=np.dtype(dtype).str, shape=shape) for name,(dtype,shape) in shapes.items()},
        raw_originals_modified=False, maximum_raw_staging_bytes=max(r['size'] for r in rows),
        cache_array_bytes=sum((destination / (name + '.npy')).stat().st_size for name in shapes),
        cache_tokens=tokens, formal_test_population=all(r['split'] == 'test' for r in rows))
    atomic_json(destination / 'manifest.json', manifest)
    return manifest
