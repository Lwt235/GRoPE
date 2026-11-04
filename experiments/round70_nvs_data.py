"""Old-scene RGB NVS cache. Sensor geometry is a separate reference-only input."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.round5_rgbd_data import (load_sun3d_scene, load_sensor_depth,
    resize_intrinsics, sample_nearest_depth, sha256)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/round70_nvs/attempt1'
HEIGHT, WIDTH, PATCH = 96, 128, 8
OLD_LOCK = ROOT/'results/round46_native3d/attempt1/lock.json'


def edge_intrinsics(K_native, native_hw, crop=None):
    """Old zero-center K -> resized corner-origin K used by official raymap."""
    K = K_native.copy()
    if crop is not None:
        left, top, right, bottom = crop
        K[:2, 2] -= [left, top]
        native_hw = (bottom-top, right-left)
    K = resize_intrinsics(K, native_hw, (HEIGHT, WIDTH))
    K[:2, 2] += .5
    return K


def observed_moments(depth, K_native):
    h, w = depth.shape
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    # Pixel-center map back to the native zero-based camera convention.
    native = np.stack(((xx+.5)*w/WIDTH-.5, (yy+.5)*h/HEIGHT-.5), -1)
    values, valid = sample_nearest_depth(depth, native.reshape(-1, 2))
    rays = np.c_[native.reshape(-1, 2), np.ones(HEIGHT*WIDTH)]@np.linalg.inv(K_native).T
    points = np.where(valid[:, None], rays*np.nan_to_num(values)[:, None]/3., 0)
    points = points.reshape(HEIGHT//PATCH, PATCH, WIDTH//PATCH, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(-1, PATCH**2, 3)
    valid = valid.reshape(HEIGHT//PATCH, PATCH, WIDTH//PATCH, PATCH).transpose(0, 2, 1, 3).reshape(-1, PATCH**2)
    count = valid.sum(-1)
    mean = points.sum(1)/np.maximum(count[:, None], 1)
    delta = (points-mean[:, None])*valid[..., None]
    cov = np.einsum('nki,nkj->nij', delta, delta)/np.maximum(count[:, None, None], 1)
    cov += np.eye(3)[None]*(1/300)**2
    # Same public default as the initial RGB geometry generator; no token removal.
    px, py = np.meshgrid((np.arange(WIDTH//PATCH)+.5)*PATCH,
                        (np.arange(HEIGHT//PATCH)+.5)*PATCH)
    rays0 = np.c_[px.ravel(), py.ravel(), np.ones(px.size)]@np.linalg.inv(edge_intrinsics(K_native, (h,w))).T
    mean[count == 0] = rays0[count == 0]
    cov[count == 0] = np.diag(np.array([.025, .030, .010])**2)+(1/300)**2*np.eye(3)
    return mean.astype('float32'), cov.astype('float32'), (count/PATCH**2).astype('float32')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT/'data'
    if (dest/'manifest.json').exists():
        manifest = json.loads((dest/'manifest.json').read_text())
        for path, expected in manifest['cache_sha256'].items():
            assert sha256(path) == expected, path
        print(json.dumps({'cache_already_verified': True, 'counts':manifest['counts']}), flush=True)
        return
    dest.mkdir(exist_ok=True)
    lock = json.loads(OLD_LOCK.read_text())
    manifest = {'old_lock_sha256':sha256(OLD_LOCK), 'source_sha256':sha256(__file__),
                'old_eval_is_not_fresh':True, 'fresh_data_read':False, 'frames':[],
                'cache_sha256':{}, 'counts':{}, 'scenes':{}}
    mapping = {'train':'fit', 'val':'val', 'test':'old_eval'}
    for original, split in mapping.items():
        banks = {k:[] for k in ('rgb', 'zoom_rgb', 'K', 'zoom_K', 'c2w', 'mean', 'cov', 'coverage')}
        triples, scene_ids, scenes = [], [], []
        for entry in lock['entries']:
            if entry['split'] != original:
                continue
            scene = entry['scene']; scenes.append(scene)
            frames, metadata = load_sun3d_scene(ROOT/'data/round5_sun3d', scene)
            pairs = set(map(tuple, entry['pairs']))
            trip = [(q,q+15,q+60) for q in sorted(set(q for q,k in pairs))
                    if (q,q+15) in pairs and (q,q+60) in pairs]
            ids = sorted(set(i for row in trip for i in row))
            lookup = {}
            for i in ids:
                f = frames[i]; lookup[i] = len(banks['rgb'])
                with Image.open(f.rgb_path) as im:
                    im = im.convert('RGB'); h, w = im.height, im.width
                    crop = (w//8, h//8, w-w//8, h-h//8)
                    rgb = np.asarray(im.resize((WIDTH,HEIGHT), Image.Resampling.BILINEAR)).copy()
                    zoom = np.asarray(im.crop(crop).resize((WIDTH,HEIGHT), Image.Resampling.BILINEAR)).copy()
                depth = load_sensor_depth(f.depth_path)
                assert depth.shape == (h,w)
                mean,cov,coverage = observed_moments(depth, f.K_native)
                vals = dict(rgb=rgb,zoom_rgb=zoom,K=edge_intrinsics(f.K_native,(h,w)).astype('float32'),
                    zoom_K=edge_intrinsics(f.K_native,(h,w),crop).astype('float32'), c2w=f.c2w,
                    mean=mean,cov=cov,coverage=coverage)
                for key,val in vals.items(): banks[key].append(val)
                manifest['frames'].append({'split':split,'scene':scene,'index':i,
                    'bank_index':lookup[i], 'rgb_path':str(f.rgb_path),'rgb_sha256':sha256(f.rgb_path),
                    'depth_path':str(f.depth_path),'depth_sha256':sha256(f.depth_path),
                    'K_sha256':metadata['K_sha256'],'pose_sha256':metadata['pose_sha256'],
                    'mean_coverage':float(coverage.mean()),'zero_patches':int((coverage==0).sum())})
            triples.extend([[lookup[i] for i in row] for row in trip])
            scene_ids.extend([len(scenes)-1]*len(trip))
            print(json.dumps({'prepared_scene':scene,'split':split,'triples':len(trip)}),flush=True)
        path = dest/f'{split}.npz'
        np.savez(path, **{k:np.stack(v) for k,v in banks.items()},
                 triples=np.asarray(triples,dtype=np.int64), scene_id=np.asarray(scene_ids,dtype=np.int64))
        manifest['cache_sha256'][str(path)] = sha256(path)
        manifest['counts'][split] = len(triples); manifest['scenes'][split] = scenes
    assert manifest['counts'] == {'fit':288,'val':32,'old_eval':240}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'prepared':True,'counts':manifest['counts']}),flush=True)


class NVSData:
    def __init__(self, split):
        with np.load(OUT/'data'/f'{split}.npz') as z:
            self.arrays = {k:z[k] for k in z.files}
        self.split = split

    def __len__(self): return len(self.arrays['triples'])

    def batch(self, indices, role, device, condition='base', swaps=None):
        import torch
        a = self.arrays; rows = a['triples'][indices].copy()
        refs, target = (rows[:,[0,1]],rows[:,2]) if condition=='extrapolate' else (rows[:,[0,2]],rows[:,1])
        if swaps is not None:
            refs[swaps] = refs[swaps, ::-1]
        both = np.concatenate((refs,target[:,None]),1)
        c2w = a['c2w'][both]
        c2w = np.linalg.inv(c2w[:,0])[:,None]@c2w
        c2w[...,:3,3] /= 3
        K = a['K'][both].copy()
        if condition=='target_zoom': K[:,-1] = a['zoom_K'][target]
        inputs = {'ref_rgb':torch.from_numpy(a['rgb'][refs]).to(device).float()/255,
                  'K':torch.from_numpy(K).to(device),
                  'c2w':torch.from_numpy(c2w.astype('float32')).to(device)}
        if role.startswith('observed_'):
            for key in ('mean','cov','coverage'):
                inputs['ref_'+key] = torch.from_numpy(a[key][refs]).to(device)
        target_rgb = a['zoom_rgb' if condition=='target_zoom' else 'rgb'][target]
        labels = torch.from_numpy(target_rgb).to(device).float()/255
        return inputs, labels


if __name__ == '__main__': prepare()
