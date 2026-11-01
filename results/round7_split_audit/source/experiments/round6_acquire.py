"""Acquire predeclared fresh SUN3D metadata/frames without scoring images.

Uses the established download/listing primitives but never calls or changes
Round 5's acquisition entry point or output directory. Selection depends only
on original frame ordinals and nearest timestamps. Metadata is acquired first;
frames are a separate explicit CLI phase. No labels or model predictions here.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np

from experiments.round5_acquire import fetch, listing, sha


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/round6_sun3d'
SCENES = {
    'val': ['brown_bm_4/brown_bm_4'],
    'test': ['brown_bm_1/brown_bm_1', 'brown_cogsci_1/brown_cogsci_1',
             'brown_cs_3/brown_cs3', 'mit_32_d507/d507_2',
             'home_at/home_at_scan1_2013_jan_1'],
}


def stamp(filename):
    return int(filename.split('-')[1].split('.')[0])


def select_pairs(rgb_count, split):
    count = {'val': 32, 'test': 48}[split]
    if rgb_count-120 < count:
        raise ValueError(f'{rgb_count} frames cannot supply {count} unique fixed query ordinals')
    queries = np.unique(np.linspace(30, rgb_count-91, count).astype(int))
    if len(queries) != count:
        raise ValueError('query ordinals unexpectedly collide')
    return [(int(q), int(q+offset)) for q in queries for offset in (15, 60)]


def one_scene(out, split, name):
    directory = out/name
    base = 'https://sun3d.cs.princeton.edu/data/'+name+'/'
    rgb = listing(base+'image/', directory/'image_index.html', 'jpg')
    depth = listing(base+'depth/', directory/'depth_index.html', 'png')
    poses = listing(base+'extrinsics/', directory/'extrinsics_index.html', 'txt')
    if not rgb or not depth or not poses:
        raise ValueError(f'incomplete native listings: {name}')
    pose = fetch(base+'extrinsics/'+poses[-1], directory/'extrinsics'/poses[-1])
    kpath = fetch(base+'intrinsics.txt', directory/'intrinsics.txt')
    pose_rows = np.loadtxt(pose, ndmin=2)
    if pose_rows.shape != (3*len(rgb), 4):
        raise ValueError(f'full RGB/pose count or layout mismatch: {name}')
    K = np.loadtxt(kpath)
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(pose_rows).all():
        raise ValueError(f'nonfinite/invalid cameras: {name}')
    pairs = select_pairs(len(rgb), split)
    depth_times = np.array([stamp(value) for value in depth], np.int64)
    frames = []
    for index in sorted({i for pair in pairs for i in pair}):
        gaps = np.abs(depth_times-stamp(rgb[index]))
        match = int(gaps.argmin())
        frames.append(dict(index=index, rgb=rgb[index], depth=depth[match], time_gap_us=int(gaps[match])))
    row = dict(scene=name, split=split, base_url=base, rgb_count=len(rgb), depth_count=len(depth),
        pose_count=len(rgb), pose_file=str(pose.relative_to(out)), pose_sha256=sha(pose),
        K_sha256=sha(kpath), pairs=pairs, frames=frames,
        listing_sha256={sub: sha(directory/f'{sub}_index.html') for sub in ['image', 'depth', 'extrinsics']})
    files = [(directory/'image_index.html', base+'image/'),
             (directory/'depth_index.html', base+'depth/'),
             (directory/'extrinsics_index.html', base+'extrinsics/'),
             (pose, base+'extrinsics/'+poses[-1]), (kpath, base+'intrinsics.txt')]
    provenance = [dict(path=str(path.relative_to(out)), url=url, bytes=path.stat().st_size,
                       sha256=sha(path)) for path, url in files]
    print('METADATA', name, split, 'RGB', len(rgb), 'frames', len(frames), 'pairs', len(pairs),
          'max_sync_us', max(frame['time_gap_us'] for frame in frames), flush=True)
    return row, provenance


def write_preserving(path, value):
    text = json.dumps(value, indent=2)+'\n'
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f'refusing to alter existing frozen selection/manifest: {path}')
    else:
        path.write_text(text)


def metadata(out=OUT, workers=4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    old_path = ROOT/'data/round5_sun3d/selection.json'
    old = json.loads(old_path.read_text())
    old_places = {row['scene'].split('/')[0] for row in old['scenes']}
    jobs = [(split, name) for split, names in SCENES.items() for name in names]
    places = [name.split('/')[0] for _, name in jobs]
    if len(places) != len(set(places)) or set(places) & old_places:
        raise ValueError('fresh place IDs collide with each other or previous Round 5 places')
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda job: one_scene(out, *job), jobs))
    result = dict(selection='linspace query ordinals 30..N-91; offsets15,60; nearest sensor-depth timestamps',
        scene_split_version=1, source_sha256=sha(__file__),
        helper_source_sha256=sha(ROOT/'experiments/round5_acquire.py'),
        previous_selection_sha256=sha(old_path), previous_place_ids=sorted(old_places),
        no_model_or_label_inspection=True, scenes=[row for row, _ in results])
    write_preserving(out/'selection.json', result)
    write_preserving(out/'metadata_manifest.json', [record for _, records in results for record in records])
    print('SELECTION', sha(out/'selection.json'), 'val_pairs', 64, 'test_pairs', 480, flush=True)
    return result


def frames(out=OUT, workers=8, splits=None):
    out = Path(out)
    selection = json.loads((out/'selection.json').read_text())
    if selection['source_sha256'] != sha(__file__):
        raise ValueError('acquisition source changed after metadata selection')
    jobs = {}
    for scene in selection['scenes']:
        if splits and scene['split'] not in splits:
            continue
        for frame in scene['frames']:
            for key, sub in [('rgb', 'image'), ('depth', 'depth')]:
                path = out/scene['scene']/sub/frame[key]
                jobs[str(path)] = (scene['base_url']+sub+'/'+frame[key], path)

    def one(job):
        url, path = job
        fetch(url, path)
        return dict(path=str(path.relative_to(out)), url=url, bytes=path.stat().st_size, sha256=sha(path))

    records = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for record in pool.map(one, jobs.values()):
            records.append(record)
            if len(records) % 100 == 0:
                print('FILES', len(records), '/', len(jobs), flush=True)
    suffix = '_'+ '_'.join(splits) if splits else ''
    write_preserving(out/f'download_manifest{suffix}.json', records)
    print('FRAMES_DONE', len(records), 'bytes', sum(row['bytes'] for row in records), flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', action='store_true')
    parser.add_argument('--frames', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--splits', nargs='+', choices=['val', 'test'])
    args = parser.parse_args()
    if not args.metadata and not args.frames:
        parser.error('choose --metadata and/or --frames')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.metadata:
        metadata(args.out, args.workers)
    if args.frames:
        frames(args.out, args.workers, args.splits)
