"""Controlled multi-plane correspondence, with no Gaussian/identity-based labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/round50_local_frame/attempt1'
SIDE, PATCH, GRID = 64, 8, 8
CONDITIONS = ('base', 'slant', 'zoom', 'wide')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def centers():
    y, x = np.mgrid[:GRID, :GRID]
    return np.stack((x.ravel()*PATCH+3.5, y.ravel()*PATCH+3.5), -1)


def unit(v):
    return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def scene(seed, condition='base'):
    rng = np.random.default_rng(seed)
    c = np.array([[-.85, -.85, 3], [.85, -.85, 3], [-.85, .85, 3], [.85, .85, 3]], float)
    c += rng.uniform(-.2, .2, (4, 3))
    az = rng.uniform(-np.pi, np.pi, 4)
    u = rng.uniform(0, 1, 4)
    angle = np.deg2rad(45+20*u if condition == 'slant' else 25*u)
    n = np.stack((np.sin(angle)*np.cos(az), np.sin(angle)*np.sin(az), np.cos(angle)), -1)
    n = np.concatenate((n, [[0, 0, 1]]))
    c = np.concatenate((c, [[0, 0, 6]]))
    a = unit(np.cross(np.broadcast_to([0, 1, 0], n.shape), n))
    b = np.cross(n, a)
    extent = np.r_[rng.uniform(1., 1.15, 4), 1e6]
    baseline = rng.uniform(1., 1.4) if condition == 'wide' else rng.uniform(.2, .6)
    camera = np.array([baseline*rng.choice([-1, 1]), rng.uniform(-.18, .18), rng.uniform(-.08, .08)])
    forward = unit(np.array([0, 0, 3.3])-camera)
    right = unit(np.cross([0, 1, 0], forward))
    up = np.cross(forward, right)
    rotation = np.stack((right, up, forward), -1)
    focal = 48*(rng.uniform(1.6, 2) if condition == 'zoom' else rng.uniform(.85, 1.15))
    freq = unit(rng.normal(size=(18, 3)))*rng.uniform(.3, 3.5, (18, 1))*2*np.pi
    color = rng.normal(size=(18, 3))/np.sqrt(18)
    phase = rng.uniform(-np.pi, np.pi, 18)
    return dict(c=c, n=n, a=a, b=b, extent=extent, camera=camera, rotation=rotation,
                focal=focal, freq=freq, color=color, phase=phase)


def intersect(s, origin, rotation, focal, uv):
    rays = np.column_stack(((uv-31.5)/focal, np.ones(len(uv))))@rotation.T
    denom = rays@s['n'].T
    numer = np.sum((s['c']-origin)*s['n'], -1)
    distance = np.divide(numer, denom, out=np.full_like(denom, np.inf), where=np.abs(denom)>1e-10)
    points = origin+distance[..., None]*rays[:, None, :]
    offset = points-s['c']
    inside = (np.abs(np.sum(offset*s['a'], -1)) <= s['extent']) & (np.abs(np.sum(offset*s['b'], -1)) <= s['extent'])
    distance = np.where(inside & (distance>.01), distance, np.inf)
    index = np.argmin(distance, -1)
    depth = distance[np.arange(len(uv)), index]
    if not np.isfinite(depth).all(): raise ValueError('background must cover image')
    xyz = origin+depth[:, None]*rays
    normal = s['n'][index]
    dr = rotation[:, :2]/focal
    # Analytic derivative dX_world / d(u,v), with plane held fixed.
    nd = np.sum(normal*rays, -1)
    jac = depth[:, None, None]*(dr[None]-rays[:, :, None]*(normal@dr)[:, None, :]/nd[:, None, None])
    return xyz, normal, jac, index


def view(s, origin, rotation, focal, seed):
    y, x = np.mgrid[:SIDE, :SIDE]
    uv = np.stack((x.ravel(), y.ravel()), -1)
    xyz, _, _, _ = intersect(s, origin, rotation, focal, uv)
    # Shared continuous texture field; never use the surface identity in color.
    rgb = .5+.45*np.tanh(np.cos(xyz@s['freq'].T+s['phase'])@s['color'])
    rng = np.random.default_rng(seed)
    rgb = np.clip(rgb+rng.normal(0, .005, rgb.shape), 0, 1)
    patches = rgb.reshape(GRID, PATCH, GRID, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(64, -1)
    xyz, normal, jac, index = intersect(s, origin, rotation, focal, centers())
    cov = PATCH**2/12*(jac@jac.transpose(0, 2, 1))+.0001*normal[:, :, None]*normal[:, None, :]
    return dict(rgb=patches, xyz=xyz, normal=normal, cov=cov, jac=jac, index=index,
                origin=origin, rotation=rotation, focal=focal)


def geometry(v, rng=None):
    xyz, n, cov = v['xyz'].copy(), v['normal'].copy(), v['cov'].copy()
    if rng is not None:
        xyz += rng.normal(0, .03, xyz.shape)
        # One coherent small rotation of the normal and footprint per token.
        axis = unit(rng.normal(size=xyz.shape)); angle = rng.normal(0, .12, len(xyz))
        w = np.zeros((len(xyz), 3, 3))
        w[:, 0, 1], w[:, 0, 2], w[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
        w = w-w.transpose(0, 2, 1)
        r = np.eye(3)+np.sin(angle)[:, None, None]*w+(1-np.cos(angle))[:, None, None]*(w@w)
        n = (r@n[..., None])[..., 0]
        cov = (r@cov@r.transpose(0, 2, 1))*np.exp(rng.normal(0, .2, (len(xyz), 1, 1)))
    ii, jj = np.triu_indices(3)
    camera = np.r_[v['rotation'].ravel(), v['origin']/3,
                   np.log(v['focal']/48), np.log(v['focal']/48)]
    return np.concatenate((xyz/3, n, np.clip(cov[:, ii, jj]/.05, -20, 20),
        np.broadcast_to(camera, (64, len(camera))), (centers()-31.5)/32), -1)


def labels(s, source, target):
    cam = (source['xyz']-target['origin'])@target['rotation']
    uv = target['focal']*cam[:, :2]/cam[:, 2, None]+31.5
    hit, _, _, _ = intersect(s, target['origin'], target['rotation'], target['focal'], uv)
    visible = (np.linalg.norm(hit-source['xyz'], axis=-1)<1e-5) & (cam[:, 2]>.01)
    valid = visible & (uv.min(-1)>=3.5) & (uv.max(-1)<=59.5)
    loc = (uv-3.5)/8
    low = np.floor(loc).astype(int)
    frac = loc-low
    offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    xy = np.clip(low[:, None]+offsets, 0, 7)
    indices = xy[..., 1]*8+xy[..., 0]
    weights = np.stack(((1-frac[:, 0])*(1-frac[:, 1]), frac[:, 0]*(1-frac[:, 1]),
                        (1-frac[:, 0])*frac[:, 1], frac[:, 0]*frac[:, 1]), -1)
    jp = np.zeros((64, 2, 3))
    jp[:, 0, 0] = target['focal']/cam[:, 2]; jp[:, 1, 1] = target['focal']/cam[:, 2]
    jp[:, :, 2] = -target['focal']*cam[:, :2]/cam[:, 2, None]**2
    warp_jac = jp@target['rotation'].T@source['jac']
    sv = np.linalg.svd(warp_jac, compute_uv=False)
    return dict(valid=valid, target=uv, indices=indices, weights=weights,
                anisotropy=sv[:, 0]/np.maximum(sv[:, 1], 1e-9), warp_jac=warp_jac)


def example(seed, condition='base'):
    s = scene(seed, condition)
    q = view(s, np.zeros(3), np.eye(3), 48., seed+100000)
    k = view(s, s['camera'], s['rotation'], s['focal'], seed+200000)
    result = {}
    for name, value, other in [('q', q, k), ('k', k, q)]:
        result[name+'rgb'] = value['rgb']
        result[name+'geom'] = geometry(value)
        result[name+'noisy'] = geometry(value, np.random.default_rng(seed+(300000 if name=='q' else 400000)))
        result[name+'pos'] = centers()/8
        result[name+'K'] = np.diag([value['focal']/64, value['focal']/64, 1.])
        for key, val in labels(s, value, other).items(): result[name+key] = val
    return {k:np.asarray(v, dtype=(bool if k.endswith('valid') else np.int64 if k.endswith('indices') else np.float32)) for k,v in result.items()}


def generate(out=OUT):
    folder = out/'data'
    if folder.exists(): raise FileExistsError('retain existing generated data')
    folder.mkdir(parents=True)
    manifest = dict(generator_sha256=sha(Path(__file__)), kind='procedural_development_not_confirmation', entries=[])
    entries = [('fit', 'base', 512, 500000), ('val', 'base', 64, 600000)]
    entries += [('dev', c, 128, 700000) for c in CONDITIONS]
    for split, condition, count, start in entries:
        rows = [example(start+i, condition) for i in range(count)]
        arrays = {key:np.stack([row[key] for row in rows]) for key in rows[0]}
        path = folder/f'{split}_{condition}.npz'
        np.savez_compressed(path, **arrays)
        valid = arrays['qvalid']
        manifest['entries'].append(dict(split=split, condition=condition, count=count, seed_start=start,
            path=str(path), sha256=sha(path), valid_queries=int(valid.sum()), total_queries=int(valid.size),
            anisotropy_quantiles=np.quantile(arrays['qanisotropy'][valid], [.1, .5, .9]).tolist()))
        print('DATA', split, condition, count, 'valid', int(valid.sum()), flush=True)
    write(folder/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=OUT)
    args = p.parse_args(); generate(args.out)
