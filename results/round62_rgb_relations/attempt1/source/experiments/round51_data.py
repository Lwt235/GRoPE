"""Already-used real RGB features and cameras, with no token-depth input."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/round51_visual_alignment/attempt1'
PRIOR = ROOT/'results/round46_native3d/attempt1/lock.json'
LABELS = ('gt_valid', 'pixcontinuous', 'key_indices', 'key_weights')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def pixels():
    y, x = np.mgrid[:18, :24]
    return np.stack((x.ravel()*8+3.5, y.ravel()*8+3.5), -1).astype(np.float32)


def camera_fields(frame, relative):
    K = frame['K']
    if abs(K[0, 1])+abs(K[1, 0]) > 1e-10 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('this centered image chart requires zero-skew intrinsics')
    pos = (pixels()-K[:2, 2])/8
    kn = np.diag([K[0, 0]/192, K[1, 1]/144, 1.])
    meta = np.concatenate((relative[:3, :3].ravel(), relative[:3, 3]/3,
        np.log([K[0, 0]/192, K[1, 1]/144]), K[:2, 2]/[192, 144]))
    geom = np.concatenate((np.broadcast_to(meta, (432, 16)), pos/12), -1)
    return dict(geom=geom.astype('float32'), pos=pos.astype('float32'), K=kn.astype('float32'))


def build(out=OUT):
    if (out/'data/manifest.json').exists(): raise FileExistsError('retain data manifest')
    prior = read(PRIOR)
    if sha(PRIOR) != PRIOR.with_name('lock.sha256').read_text().strip(): raise ValueError('prior lock identity')
    entries = []; camera_hashes = {}; counters = {}
    for e in prior['entries']:
        # R46 "test" is repeatedly used development; no unused scene is opened.
        split = {'train':'fit', 'val':'val', 'test':'old_eval'}[e['split']]
        folder = Path(e['folder'])
        for path, digest in e['hashes'].items():
            if sha(path) != digest: raise ValueError('old pair identity changed')
        frames = {}
        for rec in prior['frames']:
            if rec['scene'] != e['scene']: continue
            for path, digest in rec['hashes'].items():
                if sha(path) != digest: raise ValueError('frame identity changed')
                camera_hashes[path] = digest
            # npz fields are lazy. Depth/mean/covariance arrays are NOT read.
            with np.load(rec['path'], allow_pickle=False) as z:
                frames[rec['index']] = {k:z[k] for k in ('features', 'K', 'c2w')}
        arrays = {k:[] for k in ('q', 'k', 'qgeom', 'kgeom', 'qpos', 'kpos', 'qK', 'kK')}
        with np.load(folder/'inputs.npz', allow_pickle=False) as old:
            qfeatures, kfeatures = old['q'], old['k']
        for j, (qi, ki) in enumerate(e['pairs']):
            q, k = frames[qi], frames[ki]
            if not np.array_equal(q['features'], qfeatures[j]) or not np.array_equal(k['features'], kfeatures[j]):
                raise ValueError('RGB feature identity changed')
            relative = np.linalg.inv(q['c2w'])@k['c2w']
            for prefix, frame, transform in (('q', q, np.eye(4)), ('k', k, relative)):
                arrays[prefix].append(frame['features'])
                for name, value in camera_fields(frame, transform).items(): arrays[prefix+name].append(value)
        arrays = {k:np.stack(v) for k, v in arrays.items()}
        with np.load(folder/'labels.npz', allow_pickle=False) as labels:
            arrays.update({k:labels[k] for k in LABELS})
        if len(arrays['q']) != len(e['pairs']): raise ValueError('pair count')
        valid = arrays['gt_valid']
        if not np.all(np.isfinite(arrays['pixcontinuous'][valid])): raise ValueError('valid target nonfinite')
        if not np.allclose(arrays['key_weights'].sum(-1)[valid], 1, atol=1e-6): raise ValueError('target weight sum')
        destination = out/'data'/split/(e['scene'].replace('/', '__')+'.npz')
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
        nearest = np.round((arrays['pixcontinuous']-3.5)/8)
        nearest = np.clip(nearest, [0, 0], [23, 17])*8+3.5
        best_error = np.linalg.norm(nearest-arrays['pixcontinuous'], axis=-1)[valid]
        entries.append(dict(split=split, scene=e['scene'], pairs=e['pairs'], path=str(destination),
            sha256=sha(destination), original_hashes=e['hashes'], valid_queries=int(valid.sum()),
            argmax_ceiling_auc8=float(np.clip(1-best_error/8, 0, 1).mean()),
            argmax_ceiling_epe=float(best_error.mean())))
        counters[split] = counters.get(split, 0)+len(e['pairs'])
        print('DATA', split, e['scene'], len(e['pairs']), flush=True)
    if counters != dict(fit=576, val=64, old_eval=480): raise ValueError(counters)
    write(out/'data/manifest.json', dict(entries=entries, original_lock=str(PRIOR), prior_sha256=sha(PRIOR),
        frames=camera_hashes, counts=counters, feature_identity_exact=True,
        numeric_frame_fields_read=['features', 'K', 'c2w'], confirmation_access=False))


def load(out, split, device):
    arrays = {}; identities = []
    for e in read(out/'data/manifest.json')['entries']:
        if e['split'] != split: continue
        if sha(e['path']) != e['sha256']: raise ValueError('R51 cache changed')
        with np.load(e['path'], allow_pickle=False) as z:
            for key in z.files: arrays.setdefault(key, []).append(z[key])
        identities.extend(dict(scene=e['scene'], pair_id=f'{q}:{k}', q_index=q, k_index=k) for q, k in e['pairs'])
    result = {k:torch.from_numpy(np.concatenate(v)).to(device) for k, v in arrays.items()}
    for key in result:
        result[key] = result[key].bool() if key == 'gt_valid' else result[key].long() if key == 'key_indices' else result[key].float()
    return result, identities


if __name__ == '__main__': build()
