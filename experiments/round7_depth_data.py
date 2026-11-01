"""Observation-only input construction and separately hashed query depth labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.round6_region_data import (prepare_frame, make_pair_inputs, read_scene_selection,
    _load_cache, make_grid, resize_pixel_coordinates, sample_nearest_depth, load_sensor_depth)
from models.round7_depth_attention import FIELDS

ROOT = Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def center_prediction(predicted_depth):
    z = np.nan_to_num(predicted_depth, nan=10., posinf=20., neginf=.1).clip(.1,20)
    return z.reshape(18,8,24,8).transpose(0,2,1,3)[:,:,3:5,3:5].mean((-1,-2)).reshape(-1).astype('float32')


def observations(qcache, kcache, qprepared=None, kprepared=None):
    q = prepare_frame(qcache,8) if qprepared is None else qprepared
    k = prepare_frame(kcache,8) if kprepared is None else kprepared
    data, diagnostics = make_pair_inputs(q,k)
    transform = np.linalg.inv(q.c2w)@k.c2w
    key_in_query = k.mean3@transform[:3,:3].T + transform[:3,3]
    data.update(q_depth=center_prediction(qcache['predicted_depth']),
                q_value_depth=q.mean3[:,2].clip(.1,20).astype('float32'),
                k_value_depth=key_in_query[:,2].clip(.1,20).astype('float32'))
    return {field:data[field] for field in FIELDS}, diagnostics


def query_labels(native_depth):
    centers = make_grid(8).centers
    pixels = resize_pixel_coordinates(centers,(144,192),native_depth.shape)
    target, valid = sample_nearest_depth(native_depth,pixels)
    target = target.reshape(18,24)
    valid = valid.reshape(18,24) & np.isfinite(target) & (target>0)
    boundary = np.zeros_like(valid)
    for axis in (0,1):
        a=[slice(None),slice(None)];b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        edge=valid[a]&valid[b]&(np.abs(target[a]-target[b])>(.1+.1*np.minimum(target[a],target[b])))
        boundary[a]|=edge;boundary[b]|=edge
    return dict(target=np.where(valid,target,0).astype('float32').reshape(-1),
                valid=valid.reshape(-1),boundary=boundary.reshape(-1))


def export(raw_root, selection, cache_root, out, splits, lock=None):
    if out.exists(): raise ValueError('refuse overwrite: '+str(out))
    if 'test' in splits:
        if lock is None: raise ValueError('fresh test requires frozen confirmation lock')
        frozen=json.loads(lock.read_text())
        if not (frozen.get('test_export_authorized') is True and frozen['selection_sha256']==sha(selection)
                and frozen['source_hashes']['experiments/round7_depth_data.py']==sha(__file__)):
            raise ValueError('test selection/exporter mismatch')
        for path,digest in frozen['source_hashes'].items():
            if sha(ROOT/path)!=digest: raise ValueError('changed locked source '+path)
    out.mkdir(parents=True); started=time.time(); manifests=[]
    for row in json.loads(selection.read_text())['scenes']:
        if row['split'] not in splits: continue
        dest=out/row['split']/row['scene'].replace('/','__');dest.mkdir(parents=True)
        frames=read_scene_selection(raw_root,row); caches={}; prepared={}; provenance=[]
        for index in sorted({i for pair in row['pairs'] for i in pair}):
            path=cache_root/row['scene']/f'{index:07d}.npz'
            caches[index],sidecar=_load_cache(path,frames[index],row)
            prepared[index]=prepare_frame(caches[index],8)
            provenance.append(dict(index=index,cache_sha256=sha(path),rgb_sha256=sidecar['producer']['rgb_sha256']))
        # No GT has been opened; serialize all forward fields first.
        xs={f:[] for f in FIELDS}
        for qi,ki in row['pairs']:
            x,_=observations(caches[qi],caches[ki],prepared[qi],prepared[ki])
            for f in FIELDS: xs[f].append(x[f])
        np.savez_compressed(dest/'inputs.npz',**{f:np.stack(v) for f,v in xs.items()})
        del xs,caches,prepared
        labels={}; gt_hashes=[]
        for qi in sorted({q for q,k in row['pairs']}):
            labels[qi]=query_labels(load_sensor_depth(frames[qi].depth_path))
            gt_hashes.append(dict(index=qi,sensor_sha256=sha(frames[qi].depth_path)))
        np.savez_compressed(dest/'labels.npz',**{f:np.stack([labels[q][f] for q,k in row['pairs']])
            for f in ('target','valid','boundary')})
        manifest=dict(scene=row['scene'],split=row['split'],pairs=row['pairs'],
            input_sha256=sha(dest/'inputs.npz'),label_sha256=sha(dest/'labels.npz'),
            selection_sha256=sha(selection),exporter_sha256=sha(__file__),
            cache_sources=provenance,query_sensor_sources=gt_hashes,
            sensor_depth_used_for_forward=False,key_sensor_read=False,
            target='query native-nearest depth at stride8 centers',
            boundary='valid 4-neighbor delta > .1+.1*min(z); no GT-hole edges',
            total_positions=len(row['pairs'])*432,
            valid_positions=int(sum(labels[q]['valid'].sum() for q,k in row['pairs'])))
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)); manifests.append(manifest)
        print('EXPORTED',row['scene'],len(row['pairs']),manifest['valid_positions'],flush=True)
    (out/'completion.json').write_text(json.dumps(dict(manifests=manifests,seconds=time.time()-started,
        lock_sha256=sha(lock) if lock else None),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--splits',nargs='+',required=True)
    p.add_argument('--lock',type=Path);a=p.parse_args()
    export(a.raw_root,a.selection,a.cache_root,a.out,a.splits,a.lock)

if __name__=='__main__':main()
