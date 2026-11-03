"""Privileged native-depth patch moments; never supplied to RGB variants."""
import argparse
from functools import lru_cache
import time
import numpy as np
from experiments.round5_rgbd_data import (load_sensor_depth,resize_intrinsics,
    resize_pixel_coordinates,sample_nearest_depth,read_sun3d_intrinsics)
from experiments.round62_common import OUT,CONFIG,checked,read,write,sha,pixels
from models.round62_rgb_relations import FLOOR,POINT_VARIANCE


def observed_geometry(depth,native_k):
    """No pair/label/visibility input; all432tokens survive including depth holes."""
    image_k=resize_intrinsics(native_k,depth.shape,(144,192))
    centers=pixels().astype(np.float64)
    y,x=np.mgrid[:144,:192]
    samples=np.stack((x,y),-1).reshape(18,8,24,8,2).transpose(0,2,1,3,4).reshape(432,64,2)
    native=resize_pixel_coordinates(samples.reshape(-1,2),(144,192),depth.shape)
    z,known=sample_nearest_depth(depth,native)
    z=np.where(known,np.clip(z,*CONFIG['observed_depth_clip_m']),0).reshape(432,64)
    known=known.reshape(432,64)
    center_native=resize_pixel_coordinates(centers,(144,192),depth.shape)
    cz,center_known=sample_nearest_depth(depth,center_native)
    cz=np.where(center_known,np.clip(cz,*CONFIG['observed_depth_clip_m']),0)
    counts=known.sum(-1)
    for i in np.flatnonzero(~center_known):
        cz[i]=np.median(z[i,known[i]]) if counts[i] else CONFIG['observed_missing_depth_m']
    inv=np.linalg.inv(image_k)
    rays=np.column_stack((centers,np.ones(432)))@inv.T
    mean=rays*cz[:,None]/3
    rays64=np.concatenate((samples,np.ones((432,64,1))),-1)@inv.T
    delta=rays64*z[...,None]/3-mean[:,None]
    covariance=np.einsum('nki,nkj,nk->nij',delta,delta,known)/np.maximum(counts,1)[:,None,None]
    covariance+=FLOOR**2*np.eye(3)
    covariance[counts==0]=POINT_VARIANCE*np.eye(3)
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError('nonfinite observed support')
    return dict(mean=mean.astype(np.float32),cov=covariance.astype(np.float32)),dict(
        tokens=432,missing_centers=int((~center_known).sum()),
        patch_median_fallbacks=int(((~center_known)&(counts>0)).sum()),
        empty_patches=int((counts==0).sum()),observed_patch_samples=int(known.sum()))


def prepare(split):
    lock=checked()
    if split not in ('fit','val','old_eval'):raise ValueError(split)
    if split=='old_eval' and not (OUT/'selected.json').exists():raise ValueError('selections required')
    path=OUT/'data'/('observed_'+split+'.npz')
    if path.exists() or path.with_suffix('.json').exists():raise ValueError('preserve observed cache')
    raw={e['scene']:e for e in lock['depth_entries']}
    arrays={};ids=[];coverage={};source_hashes={};started=time.time()
    for e in read(OUT/'data/manifest.json')['entries']:
        if e['split']!=split:continue
        entry=raw[e['scene']]
        if entry['pairs']!=e['pairs'] or entry['split']!=split:raise ValueError('pair metadata changed')
        if sha(entry['intrinsics'])!=lock['raw_hashes'][entry['intrinsics']]:raise ValueError('intrinsics changed')
        native_k=read_sun3d_intrinsics(entry['intrinsics'])
        frame_map={f['index']:f for f in entry['frames']}
        @lru_cache(maxsize=8)
        def frame(index):
            rec=frame_map[index]
            if sha(rec['path'])!=rec['sha256']:raise ValueError('native depth changed')
            result,counts=observed_geometry(load_sensor_depth(rec['path']),native_k)
            coverage[f"{e['scene']}:{index}"]=counts
            source_hashes[rec['path']]=rec['sha256']
            return result
        for qi,ki in e['pairs']:
            for prefix,index in (('q',qi),('k',ki)):
                for key,value in frame(index).items():
                    arrays.setdefault(prefix+'observed_'+key,[]).append(value)
            ids.append(f"{e['scene']}:{qi}:{ki}")
        frame.cache_clear()
        print('OBSERVED',split,e['scene'],len(e['pairs']),flush=True)
    np.savez_compressed(path,**{k:np.stack(v) for k,v in arrays.items()},ids=np.asarray(ids))
    write(path.with_suffix('.json'),dict(split=split,count=len(ids),sha256=sha(path),
        lock_sha256=sha(OUT/'lock.json'),frame_coverage=coverage,depth_hashes=source_hashes,
        seconds=time.time()-started,privileged_diagnostic=True,confirmation_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=('fit','val','old_eval'))
    prepare(parser.parse_args().split)
