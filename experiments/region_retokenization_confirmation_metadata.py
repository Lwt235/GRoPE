"""Prospective unused-place metadata only; pixels remain inaccessible here."""
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

PREP=ROOT/'results/region_retokenization_confirmation/preparation'
PLAN=ROOT/'docs/region_retokenization_confirmation_plan.md'
INVENTORY=ROOT/'results/research_data_usage/sun3d_remaining_metadata_20261107.json'
DEVELOPMENT=ROOT/'results/region_retokenization/attempt1'
RAW=Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/region_retokenization_confirmation_20261109/raw')
PREFIXES=('brown','harvard','home','hotel','mit')
SALT='GRoPE-region-retokenization-confirmation-20261109-v1'


def freeze():
    assert not (PREP/'cohort_lock.json').exists()
    inventory=read(INVENTORY)
    assert read(DEVELOPMENT/'review.json')['scientific_review_complete']
    assert read(DEVELOPMENT/'review.json')['decision']['development_signal_supports_fresh_scene_confirmation']
    sources={}
    for group in ('source_sha256','checked_historical_source_sha256'):
        for name,digest in inventory[group].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path)==digest,path
            sources[str(path)]=digest
    excluded=set(inventory['excluded_union'])
    candidates=set(inventory['remaining_directory_candidates'])
    assert not excluded&candidates and len(excluded)==36 and len(candidates)==170
    chosen=[]
    for prefix in PREFIXES:
        pool=[p for p in candidates if p.startswith(prefix+'_')]
        pool.sort(key=lambda p:(hashlib.sha256((SALT+':'+p).encode()).hexdigest(),p))
        assert len(pool)>=2
        chosen.extend(pool[:2])
    assert len(chosen)==len(set(chosen))==10
    for name in ('lock.json','result.json','review.json'):
        sources[str(DEVELOPMENT/name)]=sha(DEVELOPMENT/name)
    for item in read(DEVELOPMENT/'lock.json')['models']:
        assert sha(item['checkpoint_path'])==item['checkpoint_sha256']
        sources[item['checkpoint_path']]=item['checkpoint_sha256']
    for p in [Path(__file__),PLAN,INVENTORY,ROOT/'experiments/round6_acquire.py',
              ROOT/'experiments/round5_acquire.py',ROOT/'experiments/round5_rgbd_data.py',
              ROOT/'experiments/projection_confirmation_metadata.py',ROOT/'experiments/round65_common.py']:
        sources[str(p)]=sha(p)
    write(PREP/'cohort_lock.json',dict(created_unix=time.time(),places=chosen,
        excluded_places=sorted(excluded),candidate_count=170,salt=SALT,
        source_sha256=sources,models=read(DEVELOPMENT/'lock.json')['models'],
        scene_selection='lexicographically first child scan; no replacement',
        sampling='48 linspace queries 30..N-91; offsets15/60; nearest depth; no replacement',
        raw_directory=str(RAW),pixels_access_permitted=False,phase='metadata only',goal_complete=False))
    (PREP/'cohort_lock.sha256').write_text(sha(PREP/'cohort_lock.json')+'\n')
    print('COHORT_FROZEN',chosen,sha(PREP/'cohort_lock.json'),flush=True)


def checked():
    assert sha(PREP/'cohort_lock.json')==(PREP/'cohort_lock.sha256').read_text().strip()
    lock=read(PREP/'cohort_lock.json')
    for p,digest in lock['source_sha256'].items():
        assert sha(p)==digest,p
    assert not lock['pixels_access_permitted']
    return lock


def metadata():
    lock=checked()
    assert not (PREP/'metadata_accounted.json').exists()
    RAW.mkdir(parents=True,exist_ok=True)
    selections=[];failures=[]
    for place in lock['places']:
        path=RAW/place/'scan_index.html'
        try:
            fetch_metadata('https://sun3d.cs.princeton.edu/data/'+place+'/',path)
            names=sorted(set(re.findall(r'href="([A-Za-z0-9_-]+)/"',path.read_text())))
            assert names,place
            selections.append(dict(place=place,scene=place+'/'+names[0],listing_sha256=sha(path)))
        except Exception as error:
            failures.append(dict(place=place,phase='scan_listing',type=type(error).__name__,message=str(error)))
    acquisition.fetch,acquisition.listing=fetch_metadata,listing
    rows=[];records=[]
    def one(item):
        row,provenance=acquisition.one_scene(RAW,'test',item['scene'])
        read_sun3d_intrinsics(RAW/item['scene']/'intrinsics.txt')
        read_sun3d_extrinsics(RAW/row['pose_file'])
        return row,provenance
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        tasks={pool.submit(one,item):item for item in selections}
        for task in concurrent.futures.as_completed(tasks):
            item=tasks[task]
            try:
                row,provenance=task.result();rows.append(row);records.extend(provenance)
            except Exception as error:
                failures.append(dict(place=item['place'],scene=item['scene'],phase='camera_and_frames',
                                     type=type(error).__name__,message=str(error)))
            write(PREP/'metadata_progress.json',dict(completed=[r['scene'] for r in rows],failures=failures,
                  observed_unix=time.time(),pixels_read=False))
    rows.sort(key=lambda r:lock['places'].index(r['scene'].split('/')[0]))
    records.sort(key=lambda r:r['path'])
    write(PREP/'scene_selection.json',dict(records=selections,failures=failures,
          cohort_lock_sha256=sha(PREP/'cohort_lock.json'),pixels_read=False))
    write(RAW/'selection.json',dict(scenes=rows,cohort_lock_sha256=sha(PREP/'cohort_lock.json')))
    write(RAW/'metadata_manifest.json',records)
    write(PREP/'metadata_accounted.json',dict(completed_unix=time.time(),planned_places=lock['places'],
        available_scenes=[r['scene'] for r in rows],failures=failures,complete_cohort=not failures,
        planned_pairs=960,pairs=sum(len(r['pairs']) for r in rows),
        selection_sha256=sha(RAW/'selection.json'),manifest_sha256=sha(RAW/'metadata_manifest.json'),
        pixels_read=False,proxy_policy='disabled',goal_complete=False))
    print('METADATA_ACCOUNTED',len(rows),'available',len(failures),'missing; no pixels',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','metadata'))
    globals()[parser.parse_args().action]()
