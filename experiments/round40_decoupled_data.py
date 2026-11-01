"""R40 metadata freeze and observation-only central 3D support construction."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/round40_decoupled_gpu'
SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1',
          'harvard_c8/hv_c8_3', 'brown_bm_4/brown_bm_4')
SOURCE_ROOTS = ('ROUND40_DECOUPLED_GPU_PROTOCOL.md', 'models/round40_decoupled_pe.py',
    'experiments/round40_decoupled_data.py', 'experiments/round40_decoupled_train.py',
    'experiments/round40_decoupled_audit.py', 'tests/test_round40_decoupled.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_names():
    pending, found = list(SOURCE_ROOTS), set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        p = ROOT / name
        if not p.is_file():
            raise FileNotFoundError(p)
        found.add(name)
        if p.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(p.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.level:
                    raise ValueError('relative project import')
                modules = [node.module] + [node.module + '.' + n.name for n in node.names]
            for module in modules:
                parts = module.split('.')
                candidates = [Path(*parts).with_suffix('.py'), Path(*parts) / '__init__.py']
                candidates += [Path(*parts[:i]) / '__init__.py' for i in range(1, len(parts))]
                pending.extend(str(c) for c in candidates if (ROOT / c).is_file())
    return sorted(found)


def freeze(out=OUT):
    if out.exists():
        raise ValueError('preserve existing attempt')
    entries = []
    for i, scene in enumerate(SCENES):
        split = 'train' if i < 3 else 'val'
        root = ROOT / ('results/round7_depth_bm4' if i == 4 else 'results/round7_depth_old')
        folder = root / split / scene.replace('/', '__')
        m = read(folder / 'manifest.json')
        if m['scene'] != scene or m['split'] != split or len(m['pairs']) != (192 if i < 3 else 64):
            raise ValueError('original cohort mismatch')
        for name, key in (('inputs.npz', 'input_sha256'), ('labels.npz', 'label_sha256')):
            # Old supervised TRAIN/VAL, not confirmation data. Bytes may be
            # hashed here; no numerical array is decoded before the freeze.
            if sha(folder / name) != m[key]:
                raise ValueError('changed depth data')
        cache_root = ROOT / ('results/round6_inputs/cache' if i == 4 else 'results/round5_inputs/cache') / scene
        caches = []
        for row in m['cache_sources']:
            p = cache_root / f"{row['index']:07d}.npz"
            side = read(p.with_suffix('.json'))
            if sha(p) != row['cache_sha256'] or side['cache_sha256'] != row['cache_sha256']:
                raise ValueError('changed predictor cache')
            if side['producer']['rgb_sha256'] != row['rgb_sha256']:
                raise ValueError('RGB source mismatch')
            caches.append(dict(index=row['index'], path=str(p), sha256=row['cache_sha256']))
        if set(x['index'] for x in caches) != {x for pair in m['pairs'] for x in pair}:
            raise ValueError('frame population mismatch')
        entries.append(dict(scene=scene, split=split, folder=str(folder), pairs=m['pairs'],
            manifest_sha256=sha(folder / 'manifest.json'), input_sha256=m['input_sha256'],
            label_sha256=m['label_sha256'], caches=caches))
    hashes = {n: sha(ROOT / n) for n in source_names()}
    out.mkdir(parents=True)
    for name in hashes:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    lock = dict(created_unix=time.time(), entries=entries, source_hashes=hashes,
                data_decoded=False, primary='decoupled', seeds=[11,22,33,44,55],
                goal_complete=False, fresh_data_authorized=False)
    write(out / 'lock.json', lock)
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')
    return lock


def verify(out=OUT):
    lock = read(out / 'lock.json')
    if sha(out / 'lock.json') != (out / 'lock.sha256').read_text().strip():
        raise ValueError('changed lock')
    for name, expected in lock['source_hashes'].items():
        if sha(ROOT / name) != expected or sha(out / 'source' / name) != expected:
            raise ValueError('changed source: ' + name)
    return lock


def frame_support(depth, K):
    z = np.nan_to_num(np.asarray(depth, np.float64), nan=10., posinf=20., neginf=.1).clip(.1,20.)
    if z.shape != (144,192) or np.shape(K) != (3,3):
        raise ValueError('unexpected image/camera shape')
    inv = np.linalg.inv(np.asarray(K, np.float64))
    yy, xx = np.mgrid[:144,:192]
    points = (np.stack((xx,yy,np.ones_like(xx)), -1) @ inv.T) * z[...,None]
    patches = points.reshape(18,8,24,8,3).transpose(0,2,1,3,4).reshape(432,64,3)
    mean = patches.mean(1)
    centered = patches - mean[:,None]
    zpatch = z.reshape(18,8,24,8).transpose(0,2,1,3).reshape(432,8,8)
    basis = inv[:,:2]
    cov = np.einsum('nsi,nsj->nij',centered,centered)/64.
    cov += np.mean(zpatch*zpatch,axis=(1,2))[:,None,None] * (basis @ basis.T)/12.
    y, x = np.mgrid[:18,:24]
    rays = np.column_stack((x.ravel()*8+3.5,y.ravel()*8+3.5,np.ones(432))) @ inv.T
    native_z = np.nan_to_num(np.asarray(depth), nan=10., posinf=20., neginf=.1).clip(.1,20.)
    center_z = native_z.reshape(18,8,24,8).transpose(0,2,1,3)[:,:,3:5,3:5].mean((-1,-2)).reshape(-1)
    anchor = rays * center_z[:,None]
    if not np.isfinite(anchor).all() or not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError('invalid central support')
    return anchor, cov


def prepare(out=OUT):
    lock = verify(out)
    dest = out / 'prepared'
    if dest.exists():
        raise ValueError('preserve prepared attempt')
    dest.mkdir()
    started = time.time()
    records = []
    for entry in lock['entries']:
        frames = {}
        for cache in entry['caches']:
            if sha(cache['path']) != cache['sha256']:
                raise ValueError('changed RGB cache')
            with np.load(cache['path'], allow_pickle=False) as z:
                depth, K, c2w, features = (z[n] for n in ('predicted_depth','K','c2w','features'))
            a, cov = frame_support(depth, K)
            frames[cache['index']] = (a,cov,c2w,features)
        folder = Path(entry['folder'])
        if sha(folder/'inputs.npz') != entry['input_sha256']:
            raise ValueError('changed old input')
        packed = {k: [] for k in ('q_anchor3','k_anchor3','q_cov3','k_cov3')}
        with np.load(folder/'inputs.npz', allow_pickle=False) as old:
            qf,kf,qdepth = (old[n] for n in ('q','k','q_depth'))
            for row,(qi,ki) in enumerate(entry['pairs']):
                qa,qc,qpose,qfeat=frames[qi]; ka,kc,kpose,kfeat=frames[ki]
                if not np.array_equal(qf[row],qfeat) or not np.array_equal(kf[row],kfeat):
                    raise ValueError('feature join mismatch')
                if not np.array_equal(qa[:,2].astype(np.float32),qdepth[row]):
                    raise ValueError('center-depth join mismatch')
                T = np.linalg.inv(kpose) @ qpose
                R = T[:3,:3]
                if np.max(abs(R @ R.T-np.eye(3))) > 1e-6:
                    raise ValueError('invalid supplied camera rotation')
                values = dict(q_anchor3=qa @ R.T+T[:3,3],k_anchor3=ka,
                              q_cov3=R @ qc @ R.T,k_cov3=kc)
                for key,val in values.items():
                    packed[key].append(val.astype(np.float32))
        path=dest/(entry['scene'].replace('/','__')+'.npz')
        np.savez_compressed(path,**{k:np.stack(v) for k,v in packed.items()})
        records.append(dict(scene=entry['scene'],split=entry['split'],path=str(path),
                            sha256=sha(path),pairs=len(entry['pairs'])))
        print('PREPARED',entry['scene'],len(entry['pairs']),flush=True)
    write(dest/'completion.json',dict(records=records,seconds=time.time()-started,labels_decoded=False))
    return records
