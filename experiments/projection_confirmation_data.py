"""Fixed fresh-scene pixels/cache; the inherited R70 batch interface is unchanged."""
import concurrent.futures
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image

from experiments.round65_common import ROOT, read, write, sha
from experiments import projection_confirmation_metadata as metadata
from experiments.round5_rgbd_data import load_sun3d_scene, load_sensor_depth
from experiments.round70_nvs_data import NVSData, edge_intrinsics, observed_moments, HEIGHT, WIDTH

OUT=ROOT/'results/projection_confirmation/attempt1'
RAW=metadata.RAW
CACHE=RAW.parent/'cache'


def check_sources():
    lock=read(OUT/'evaluation_lock.json')
    assert sha(OUT/'evaluation_lock.json')==(OUT/'evaluation_lock.sha256').read_text().strip()
    assert lock['pixels_access_permitted'] and lock['groups']==['observed']
    for path,digest in lock['source_sha256'].items():assert sha(path)==digest,path
    return lock


def download():
    check_sources();assert not (OUT/'download_complete.json').exists()
    if not (OUT/'data_access_started.json').exists():
        write(OUT/'data_access_started.json',dict(unix=time.time(),
            evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
            phase='First pixel acquisition for fixed confirmation; this cohort is now reserved to this evaluation.'))
    selection=read(RAW/'selection.json');jobs={}
    for row in selection['scenes']:
        for frame in row['frames']:
            for key,sub in [('rgb','image'),('depth','depth')]:
                path=RAW/row['scene']/sub/frame[key]
                jobs[str(path)]=(row['base_url']+sub+'/'+frame[key],path)
    records=[];failures=[]
    def fetch(job):
        url,path=job
        assert url.startswith('https://sun3d.cs.princeton.edu/data/')
        assert path.suffix in ('.png','.jpg')
        path.parent.mkdir(parents=True,exist_ok=True)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for attempt in range(3):
            try:
                if not path.exists():
                    partial=path.with_suffix(path.suffix+'.part')
                    with opener.open(url,timeout=35) as response,partial.open('wb') as stream:
                        while chunk:=response.read(2**20):stream.write(chunk)
                    with Image.open(partial) as im:im.verify()
                    partial.replace(path)
                with Image.open(path) as im:im.verify()
                return dict(path=str(path),url=url,bytes=path.stat().st_size,sha256=sha(path))
            except Exception as error:
                print('DOWNLOAD_RETRY',url,attempt+1,type(error).__name__,str(error),flush=True)
                if attempt==2:raise
                time.sleep(attempt+1)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        tasks={pool.submit(fetch,job):job[0] for job in jobs.values()}
        for future in concurrent.futures.as_completed(tasks):
            try:records.append(future.result())
            except Exception as error:failures.append(dict(url=tasks[future],error=str(error)))
            if (len(records)+len(failures))%40==0:
                progress=dict(completed=len(records),total=len(jobs),failures=failures,unix=time.time())
                write(OUT/'download_progress.json',progress);print('DOWNLOAD',progress,flush=True)
    if failures:
        write(OUT/f'download_failure_{time.time_ns()}.json',dict(failures=failures,completed=len(records)))
        raise RuntimeError('Fixed source downloads failed; preserve/retry these sources without replacing scenes')
    records.sort(key=lambda r:r['path'])
    write(OUT/'download_manifest.json',records)
    write(OUT/'download_complete.json',dict(files=len(records),bytes=sum(r['bytes'] for r in records),
        manifest_sha256=sha(OUT/'download_manifest.json'),evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
        proxy_policy='disabled including redirects',unix=time.time()))
    print('DOWNLOAD_COMPLETE',len(records),flush=True)


def frame_arrays(frame):
    # This is exactly R70's pixel/geometry preprocessing, checked against its cache.
    with Image.open(frame.rgb_path) as im:
        im=im.convert('RGB');h,w=im.height,im.width
        crop=(w//8,h//8,w-w//8,h-h//8)
        rgb=np.asarray(im.resize((WIDTH,HEIGHT),Image.Resampling.BILINEAR)).copy()
        zoom=np.asarray(im.crop(crop).resize((WIDTH,HEIGHT),Image.Resampling.BILINEAR)).copy()
    depth=load_sensor_depth(frame.depth_path);assert depth.shape==(h,w)
    mean,cov,coverage=observed_moments(depth,frame.K_native)
    return dict(rgb=rgb,zoom_rgb=zoom,K=edge_intrinsics(frame.K_native,(h,w)).astype('float32'),
        zoom_K=edge_intrinsics(frame.K_native,(h,w),crop).astype('float32'),c2w=frame.c2w,
        mean=mean,cov=cov,coverage=coverage)


def selected_triples(row):
    pairs=set(map(tuple,row['pairs']))
    triples=[(q,q+15,q+60) for q in sorted({q for q,k in pairs})]
    assert len(triples)==48
    assert all((q,q+15) in pairs and (q,q+60) in pairs for q,_,_ in triples)
    return triples


def prepare():
    check_sources();done=read(OUT/'download_complete.json')
    assert done['manifest_sha256']==sha(OUT/'download_manifest.json')
    for row in read(OUT/'download_manifest.json'):assert sha(row['path'])==row['sha256'],row['path']
    assert not (OUT/'data_manifest.json').exists()
    CACHE.mkdir(parents=True,exist_ok=True)
    banks={k:[] for k in ('rgb','zoom_rgb','K','zoom_K','c2w','mean','cov','coverage')}
    triples=[];scene_ids=[];scenes=[];frames_record=[]
    for row in read(RAW/'selection.json')['scenes']:
        scene=row['scene'];scenes.append(scene);frames,_=load_sun3d_scene(RAW,scene)
        chosen=selected_triples(row);lookup={}
        for i in sorted({j for triple in chosen for j in triple}):
            frame=frames[i];lookup[i]=len(banks['rgb']);values=frame_arrays(frame)
            for key,value in values.items():banks[key].append(value)
            frames_record.append(dict(scene=scene,index=i,bank_index=lookup[i],
                rgb_sha256=sha(frame.rgb_path),depth_sha256=sha(frame.depth_path),
                mean_coverage=float(values['coverage'].mean()),zero_patches=int((values['coverage']==0).sum())))
        triples.extend([[lookup[i] for i in triple] for triple in chosen]);scene_ids.extend([len(scenes)-1]*len(chosen))
        print('PREPARED',scene,len(chosen),flush=True)
    assert len(triples)==240 and len(scenes)==5
    path=CACHE/'confirmation.npz';assert not path.exists()
    np.savez(path,**{k:np.stack(v) for k,v in banks.items()},triples=np.asarray(triples,dtype=np.int64),
             scene_id=np.asarray(scene_ids,dtype=np.int64))
    write(OUT/'data_manifest.json',dict(cache_path=str(path),cache_sha256=sha(path),scenes=scenes,
        frames=frames_record,triples=240,selection_sha256=sha(RAW/'selection.json'),
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),unix=time.time(),no_quality_filter=True))


class ConfirmationData(NVSData):
    def __init__(self,split='confirmation'):
        assert split=='confirmation'
        manifest=read(OUT/'data_manifest.json');path=Path(manifest['cache_path'])
        assert sha(path)==manifest['cache_sha256']
        with np.load(path) as z:self.arrays={k:z[k] for k in z.files}
        self.split=split
