"""Metadata-only cohort preparation; pixel access is a separate frozen phase."""
import argparse
import concurrent.futures
from pathlib import Path
import re
import time

from experiments import round6_acquire as acquisition
from experiments.projection_confirmation_metadata import fetch_metadata, listing
from experiments.round5_rgbd_data import read_sun3d_intrinsics, read_sun3d_extrinsics
from experiments.region_confirmation_common import (
    ROOT, PREP, PLAN, RAW, INVENTORY, SALT, PREFIXES, read, write, sha, preserved, select_places, check_lock)


def freeze():
    assert not (PREP / 'cohort_lock.json').exists()
    old_path = ROOT / 'results/research_data_usage/confirmation_inventory_20261104.json'
    old = read(old_path)
    sources = {}
    for name, digest in old['source_sha256'].items():
        path = ROOT / name
        assert sha(path) == digest, str(path)
        sources[str(path)] = digest
    projection = ROOT / 'results/projection_confirmation/attempt1'
    projection_lock = read(projection / 'evaluation_lock.json')
    assert read(projection / 'review.json')['scientific_review_complete']
    excluded = sorted(set(old['excluded_sun3d_places']) | {s.split('/')[0] for s in projection_lock['scenes']})
    index = PREP / 'sun3d_root_index.html'
    places = sorted(set(re.findall(r'href="((?:brown|harvard|home|hotel|mit)[A-Za-z0-9_-]*)/"', index.read_text())))
    chosen = select_places(places, excluded)
    # Verify the current regional data lock adds no unrecorded place.
    regional = ROOT / 'results/round60_real_scene/attempt1/lock.json'
    assert {e['scene'].split('/')[0] for e in read(regional)['entries']} <= set(excluded)
    inventory = dict(checked_unix=time.time(), scope='usage records and directory names only',
        historical_inventory_sha256=sha(old_path), projection_evaluation_lock_sha256=sha(projection / 'evaluation_lock.json'),
        excluded_places=excluded, official_places=places, chosen_places=chosen,
        allocation='two per prefix, ascending SHA256 of salt:place', salt=SALT, prefixes=PREFIXES,
        root_directory_url='https://sun3d.cs.princeton.edu/data/', root_directory_sha256=sha(index),
        pixel_content_read=False, alias_limitation='Distinct place IDs do not prove physical independence.')
    preserved(INVENTORY, inventory)
    files = [Path(__file__), PLAN, INVENTORY, index, old_path, regional,
             ROOT / 'experiments/region_confirmation_common.py', ROOT / 'experiments/round6_acquire.py',
             ROOT / 'experiments/round5_acquire.py', ROOT / 'experiments/round5_rgbd_data.py',
             ROOT / 'experiments/projection_confirmation_metadata.py', ROOT / 'experiments/round65_common.py']
    files.extend(projection / n for n in ('evaluation_lock.json', 'review.json', 'data_access_started.json'))
    sources.update({str(p): sha(p) for p in files})
    preserved(PREP / 'cohort_lock.json', dict(places=chosen, excluded_places=excluded,
        scene_selection='lexicographically first child directory for each frozen place',
        sampling='48 linspace queries 30..N-91, offsets 15/60, nearest depth timestamp',
        source_sha256=sources, raw_directory=str(RAW), created_unix=time.time(), pixels_access_permitted=False))
    (PREP / 'cohort_lock.sha256').write_text(sha(PREP / 'cohort_lock.json') + '\n')
    print('REGION_COHORT_FROZEN', chosen, sha(PREP / 'cohort_lock.json'), flush=True)


def metadata():
    lock = check_lock(PREP / 'cohort_lock.json')
    assert not (PREP / 'metadata_complete.json').exists()
    RAW.mkdir(parents=True, exist_ok=True)
    scenes = []
    for place in lock['places']:
        path = RAW / place / 'scan_index.html'
        url = 'https://sun3d.cs.princeton.edu/data/' + place + '/'
        text = fetch_metadata(url, path).read_text()
        names = sorted(set(re.findall(r'href="([A-Za-z0-9_-]+)/"', text)))
        assert names, place
        scenes.append(place + '/' + names[0])
    preserved(PREP / 'scene_selection.json', dict(scenes=scenes,
        listing_sha256={str(RAW / p / 'scan_index.html'): sha(RAW / p / 'scan_index.html') for p in lock['places']},
        cohort_lock_sha256=sha(PREP / 'cohort_lock.json'), pixels_read=False))
    acquisition.fetch, acquisition.listing = fetch_metadata, listing
    rows, records, failures = [], [], []

    def one(scene):
        row, provenance = acquisition.one_scene(RAW, 'test', scene)
        read_sun3d_intrinsics(RAW / scene / 'intrinsics.txt')
        read_sun3d_extrinsics(RAW / row['pose_file'])
        return row, provenance

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        tasks = {pool.submit(one, s): s for s in scenes}
        for task in concurrent.futures.as_completed(tasks):
            try:
                row, provenance = task.result()
                rows.append(row)
                records.extend(provenance)
            except Exception as error:
                failures.append(dict(scene=tasks[task], type=type(error).__name__, message=str(error)))
            write(PREP / 'metadata_progress.json', dict(completed=[r['scene'] for r in rows], failures=failures,
                observed_unix=time.time(), pixels_read=False))
    if failures:
        write(PREP / f'metadata_failure_{time.time_ns()}.json', dict(failures=failures))
        raise RuntimeError('Fixed metadata failures preserved; retry same scenes without replacement.')
    rows.sort(key=lambda r: scenes.index(r['scene']))
    records.sort(key=lambda r: r['path'])
    preserved(RAW / 'selection.json', dict(scenes=rows, cohort_lock_sha256=sha(PREP / 'cohort_lock.json')))
    preserved(RAW / 'metadata_manifest.json', records)
    preserved(PREP / 'metadata_complete.json', dict(scenes=scenes, pairs=sum(len(r['pairs']) for r in rows),
        selection_sha256=sha(RAW / 'selection.json'), manifest_sha256=sha(RAW / 'metadata_manifest.json'),
        completed_unix=time.time(), pixels_read=False, proxy_policy='disabled'))
    print('REGION_METADATA_COMPLETE', len(scenes), 'scenes; no pixels', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'metadata'))
    globals()[parser.parse_args().action]()
