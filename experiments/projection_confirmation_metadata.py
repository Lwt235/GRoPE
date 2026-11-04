"""Prospective fresh-scene metadata only. No image/depth download interface."""
import argparse
import concurrent.futures
import re
from pathlib import Path
import time
import urllib.request
import numpy as np

from experiments import round6_acquire as acquisition
from experiments.round5_rgbd_data import read_sun3d_intrinsics, read_sun3d_extrinsics
from experiments.round65_common import ROOT, read, write, sha

OUT=ROOT/'results/projection_confirmation/preparation'
RAW=Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/projection_confirmation_20261104/raw')
INVENTORY=ROOT/'results/research_data_usage/confirmation_inventory_20261104.json'
PLAN=ROOT/'docs/projection_confirmation_plan.md'
SOURCES=(Path(__file__),PLAN,INVENTORY,ROOT/'experiments/round6_acquire.py',
         ROOT/'experiments/round5_acquire.py',ROOT/'experiments/round5_rgbd_data.py',ROOT/'experiments/round65_common.py')


def freeze():
    assert not (OUT/'cohort_lock.json').exists()
    inventory=read(INVENTORY);scenes=inventory['candidate_sun3d_scenes']
    assert len(scenes)==5 and len({s.split('/')[0] for s in scenes})==5
    assert not {s.split('/')[0] for s in scenes}&set(inventory['excluded_sun3d_places'])
    write(OUT/'cohort_lock.json',dict(scenes=scenes,excluded_places=inventory['excluded_sun3d_places'],
        raw_directory=str(RAW),source_sha256={str(f):sha(f) for f in SOURCES},
        selection='48 unique linspace query ordinals 30..N-91; triple offsets 0,15,60; nearest depth timestamp',
        created_unix=time.time(),pixels_access_permitted=False,phase='metadata preparation only',
        prior_R71_result_sha256=sha(ROOT/'results/round71_target_projection/attempt1/result.json'),
        prior_R71_review_sha256=sha(ROOT/'results/round71_target_projection/attempt1/review.json')))
    (OUT/'cohort_lock.sha256').write_text(sha(OUT/'cohort_lock.json')+'\n')
    print('COHORT_FROZEN',sha(OUT/'cohort_lock.json'),flush=True)


def check():
    assert sha(OUT/'cohort_lock.json')==(OUT/'cohort_lock.sha256').read_text().strip()
    lock=read(OUT/'cohort_lock.json')
    for path,digest in lock['source_sha256'].items():assert sha(path)==digest,path
    return lock


def fetch_metadata(url,path):
    # Restrict the interface itself to public listings/cameras, never pixels.
    prefix='https://sun3d.cs.princeton.edu/data/'
    assert url.startswith(prefix) and (url.endswith('/') or url.endswith('.txt'))
    path=Path(path)
    if path.exists():return path
    path.parent.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(3):
        try:
            with opener.open(url,timeout=35) as response,path.with_suffix(path.suffix+'.part').open('wb') as stream:
                while chunk:=response.read(2**20):stream.write(chunk)
            path.with_suffix(path.suffix+'.part').replace(path)
            return path
        except Exception as error:
            print('METADATA_RETRY',url,attempt+1,type(error).__name__,str(error),flush=True)
            if attempt==2:raise
            time.sleep(attempt+1)


def listing(url,path,extension):
    return sorted(set(re.findall(r'href="([^"/]+\.'+extension+r')"',fetch_metadata(url,path).read_text())))


def metadata():
    lock=check();assert not (OUT/'metadata_complete.json').exists()
    RAW.mkdir(parents=True,exist_ok=True)
    acquisition.fetch=fetch_metadata;acquisition.listing=listing
    rows=[];provenance=[];failures=[]
    def one(scene):
        row,records=acquisition.one_scene(RAW,'test',scene)
        read_sun3d_intrinsics(RAW/scene/'intrinsics.txt')
        read_sun3d_extrinsics(RAW/row['pose_file'])
        return row,records
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        futures={pool.submit(one,scene):scene for scene in lock['scenes']}
        for future in concurrent.futures.as_completed(futures):
            scene=futures[future]
            try:
                row,records=future.result();rows.append(row);provenance.extend(records)
            except Exception as error:
                failures.append(dict(scene=scene,type=type(error).__name__,message=str(error)))
            write(OUT/'metadata_progress.json',dict(completed=[r['scene'] for r in rows],failures=failures,
                observed_unix=time.time(),pixels_read=False))
    if failures:
        write(OUT/f'metadata_failure_{time.time_ns()}.json',dict(failures=failures,completed=[r['scene'] for r in rows]))
        raise RuntimeError('Preserved metadata failures; do not replace scenes')
    rows.sort(key=lambda x:lock['scenes'].index(x['scene']))
    selection=dict(scenes=rows,selection=lock['selection'],cohort_lock_sha256=sha(OUT/'cohort_lock.json'),
        source_sha256=sha(Path(__file__)),no_model_or_pixel_inspection=True)
    acquisition.write_preserving(RAW/'selection.json',selection)
    acquisition.write_preserving(RAW/'metadata_manifest.json',provenance)
    write(OUT/'metadata_complete.json',dict(raw_directory=str(RAW),scene_count=len(rows),triples=240,
        selection_sha256=sha(RAW/'selection.json'),manifest_sha256=sha(RAW/'metadata_manifest.json'),
        completed_unix=time.time(),pixels_read=False,proxy_policy='disabled including redirects',
        evaluation_executed=False,goal_complete=False))
    print('METADATA_COMPLETE',len(rows),'scenes; no pixels acquired',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','metadata']);args=p.parse_args()
    globals()[args.action]()
