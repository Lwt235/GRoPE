"""Fixed twenty-place metadata for the unchanged rotary-relation confirmation; never fetch pixels."""
import argparse
import concurrent.futures
import hashlib
from pathlib import Path
import re
import time

import numpy as np
from experiments import round6_acquire as acquisition
from experiments.projection_confirmation_metadata import fetch_metadata, listing
from experiments.round5_rgbd_data import read_sun3d_intrinsics, read_sun3d_extrinsics
from experiments.round65_common import ROOT, read, write, sha

PREP = ROOT/'results/region_rotary_confirmation/preparation'
PLAN = ROOT/'docs/region_rotary_confirmation_plan.md'
INVENTORY = ROOT/'results/research_data_usage/sun3d_remaining_after_region_hrope_access_20261109.json'
DEVELOPMENT = ROOT/'results/region_frame_robustness/attempt1'
RAW = Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/region_rotary_confirmation_20261110/raw')
PREFIXES = ('brown', 'harvard', 'home', 'hotel', 'mit')
SALT = 'GRoPE-region-rotary-confirmation-20261110-v1'
QUOTAS = dict(brown=4, harvard=4, home=2, hotel=4, mit=6)


def freeze():
    assert not (PREP/'cohort_lock.json').exists()
    review = read(DEVELOPMENT/'review.json')
    assert review['scientific_review_complete']
    assert review['result_sha256'] == sha(DEVELOPMENT/'result.json')
    inventory = read(INVENTORY)
    sources = dict(inventory['source_sha256'])
    for name, digest in sources.items():
        assert sha(name) == digest, name
    excluded = set(inventory['excluded_union'])
    candidates = set(inventory['remaining_directory_candidates'])
    assert not excluded & candidates and len(excluded) == 56 and len(candidates) == 150
    # Inspect selection metadata only; any later place consumption must be accounted for.
    for path in RAW.parents[1].glob('*/raw/selection.json'):
        used = {r['scene'].split('/')[0] for r in read(path)['scenes']}
        assert used <= excluded, (str(path), sorted(used-excluded))
        sources[str(path)] = sha(path)
    chosen = []
    for prefix in PREFIXES:
        pool = [p for p in candidates if p.startswith(prefix+'_')]
        pool.sort(key=lambda p: (hashlib.sha256((SALT+':'+p).encode()).hexdigest(), p))
        assert len(pool) >= QUOTAS[prefix]
        chosen.extend(pool[:QUOTAS[prefix]])
    assert len(chosen) == len(set(chosen)) == 20
    models_path = DEVELOPMENT/'selected.json'
    models = read(models_path)['records']
    assert len(models) == 18
    for model in models:
        assert sha(model['checkpoint_path']) == model['checkpoint_sha256']
        sources[model['checkpoint_path']] = model['checkpoint_sha256']
    files = [Path(__file__), PLAN, INVENTORY, models_path,
             ROOT/'experiments/round6_acquire.py', ROOT/'experiments/round5_acquire.py',
             ROOT/'experiments/round5_rgbd_data.py', ROOT/'experiments/projection_confirmation_metadata.py',
             ROOT/'experiments/round65_common.py']
    files.extend(DEVELOPMENT/n for n in ('lock.json', 'result.json', 'review.json', 'completion_checks.json'))
    sources.update({str(p): sha(p) for p in files})
    write(PREP/'cohort_lock.json', dict(created_unix=time.time(), places=chosen,
        excluded_places=sorted(excluded), candidate_count=150, salt=SALT, quotas=QUOTAS,
        source_sha256=sources, models=models,
        scene_selection='lexicographically first child scan; no replacement',
        sampling='48 linspace queries 30..N-91; offsets15/60; nearest depth; no replacement',
        raw_directory=str(RAW), pixels_access_permitted=False, phase='metadata only', goal_complete=False))
    (PREP/'cohort_lock.sha256').write_text(sha(PREP/'cohort_lock.json')+'\n')
    print('COHORT_FROZEN', chosen, sha(PREP/'cohort_lock.json'), flush=True)


def checked():
    assert sha(PREP/'cohort_lock.json') == (PREP/'cohort_lock.sha256').read_text().strip()
    lock = read(PREP/'cohort_lock.json')
    for name, digest in lock['source_sha256'].items():
        assert sha(name) == digest, name
    assert not lock['pixels_access_permitted']
    return lock


def metadata():
    lock = checked()
    assert not (PREP/'metadata_accounted.json').exists()
    RAW.mkdir(parents=True, exist_ok=True)
    selections, failures = [], []
    for place in lock['places']:
        path = RAW/place/'scan_index.html'
        try:
            fetch_metadata('https://sun3d.cs.princeton.edu/data/'+place+'/', path)
            names = sorted(set(re.findall(r'href="([A-Za-z0-9_-]+)/"', path.read_text())))
            assert names, place
            selections.append(dict(place=place, scene=place+'/'+names[0], listing_sha256=sha(path)))
        except Exception as error:
            failures.append(dict(place=place, phase='scan_listing', type=type(error).__name__, message=str(error)))
    acquisition.fetch, acquisition.listing = fetch_metadata, listing
    rows, records = [], []

    def one(item):
        row, provenance = acquisition.one_scene(RAW, 'test', item['scene'])
        read_sun3d_intrinsics(RAW/item['scene']/'intrinsics.txt')
        read_sun3d_extrinsics(RAW/row['pose_file'])
        return row, provenance

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        tasks = {pool.submit(one, item): item for item in selections}
        for task in concurrent.futures.as_completed(tasks):
            item = tasks[task]
            try:
                row, provenance = task.result()
                rows.append(row)
                records.extend(provenance)
            except Exception as error:
                failures.append(dict(place=item['place'], scene=item['scene'], phase='camera_and_frames',
                                     type=type(error).__name__, message=str(error)))
            write(PREP/'metadata_progress.json', dict(completed=[r['scene'] for r in rows],
                failures=failures, observed_unix=time.time(), pixels_read=False))
    rows.sort(key=lambda r: lock['places'].index(r['scene'].split('/')[0]))
    records.sort(key=lambda r: r['path'])
    write(PREP/'scene_selection.json', dict(records=selections, failures=failures,
        cohort_lock_sha256=sha(PREP/'cohort_lock.json'), pixels_read=False))
    write(RAW/'selection.json', dict(scenes=rows, cohort_lock_sha256=sha(PREP/'cohort_lock.json')))
    write(RAW/'metadata_manifest.json', records)
    write(PREP/'metadata_accounted.json', dict(completed_unix=time.time(), planned_places=lock['places'],
        available_scenes=[r['scene'] for r in rows], failures=failures, complete_cohort=not failures,
        planned_pairs=1920, pairs=sum(len(r['pairs']) for r in rows),
        selection_sha256=sha(RAW/'selection.json'), manifest_sha256=sha(RAW/'metadata_manifest.json'),
        pixels_read=False, proxy_policy='disabled', goal_complete=False))
    print('METADATA_ACCOUNTED', len(rows), 'available', len(failures), 'missing; no pixels', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'metadata'))
    globals()[parser.parse_args().action]()
