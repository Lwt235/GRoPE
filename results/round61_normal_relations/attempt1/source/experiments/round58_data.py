"""Disjoint observed supports. Pose/world points are used only to build labels.

Stored input and supervision arrays are explicitly separated at model calls.
"""
import argparse
import hashlib
import time
import numpy as np
import torch
from experiments.round58_common import OUT, OLD_DATA, CONFIG, CONDITIONS, checked, read, write, sha


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def draw_pair(points, identity, realization, condition='base'):
    seed = hashlib.sha256((CONFIG['data_salt']+':'+identity+':'+str(realization)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], 'little'))
    pools = rng.permutation(len(points)).reshape(2, -1)
    clean, observed, transforms, source_ids = [], [], [], []
    count = 256 if condition == 'sparse256' else 512
    keep = 512 if condition == 'low_overlap' else 768
    noise = .02 if condition == 'noise02' else .005
    for pool in pools:
        direction = rng.normal(size=3)
        ranked = np.argsort(points[pool]@direction, kind='stable')[::-1]
        # Fixed-length random scores keep later draws paired across conditions.
        priority = rng.uniform(size=1024)
        retained = ranked[:keep]
        selected = retained[np.argsort(priority[retained], kind='stable')[:512]][:count]
        ids = pool[selected]
        eps = rng.normal(size=(512, 3))[:count].clip(-4, 4)
        rotation, translation = random_rotation(rng), rng.uniform(-.5, .5, size=3)
        x = points[ids].astype(np.float64)
        clean.append(x)
        observed.append(x+noise*eps)
        transforms.append((rotation, translation))
        source_ids.append(ids)
    if np.intersect1d(*source_ids).size:
        raise ValueError('view sampling shares raw IDs')
    world = np.stack(observed)
    rotations = np.stack([x[0] for x in transforms])
    translations = np.stack([x[1] for x in transforms])
    local = np.einsum('vni,vji->vnj', world, rotations)+translations[:, None]
    return dict(points=local.astype(np.float32), clean=np.stack(clean).astype(np.float32),
                world=world.astype(np.float32), source_ids=np.stack(source_ids),
                rotations=rotations.astype(np.float32), translations=translations.astype(np.float32))


def fps_and_neighbors(points, tokens=64, neighbors=32):
    b, n, _ = points.shape
    selected = torch.zeros(b, dtype=torch.long, device=points.device)
    closest = torch.full((b, n), float('inf'), device=points.device)
    rows = torch.arange(b, device=points.device)
    anchors = []
    for _ in range(tokens):
        anchors.append(selected)
        distance = (points-points[rows, selected, None]).square().sum(-1)
        closest = torch.minimum(closest, distance)
        selected = closest.argmax(-1)
    anchors = torch.stack(anchors, -1)
    c = points[rows[:, None], anchors]
    distance = (c[:, :, None]-points[:, None]).square().sum(-1)
    indices = distance.topk(neighbors, largest=False, sorted=True).indices
    return anchors, indices


@torch.no_grad()
def tokenize(batch, tokens=64, neighbors=32):
    points, clean, world = batch['points'], batch['clean'], batch['world']
    b, v, n, _ = points.shape
    flat = points.reshape(b*v, n, 3)
    anchors, ids = fps_and_neighbors(flat, tokens, neighbors)
    rows = torch.arange(b*v, device=flat.device)[:, None, None]
    patch = flat[rows, ids]
    centers = patch.mean(-2)
    delta = patch-centers.unsqueeze(-2)
    covariance = torch.einsum('btki,btkj->btij', delta, delta)/neighbors
    covariance = covariance+torch.eye(3, device=flat.device)*CONFIG['resolution']**2
    eigen = torch.linalg.eigvalsh(covariance.double()).float()
    radial = delta.norm(dim=-1).sort(-1).values
    quantile_ids = ((torch.arange(1, 9, device=flat.device)*neighbors)//8-1).long()
    features = torch.cat((.5*(eigen/.04).log(), (radial[..., quantile_ids]/.2).log1p()), -1)
    world_centers = world.reshape(b*v, n, 3)[rows, ids].mean(-2)
    ids = ids.reshape(b, v, tokens, neighbors)
    membership = torch.zeros(b, v, tokens, n, device=flat.device)
    membership.scatter_(-1, ids, 1.)
    distance = (clean[:, 0, :, None]-clean[:, 1, None]).square().sum(-1)
    adjacent = (distance < CONFIG['overlap_radius']**2).float()
    ma, mb = membership.unbind(1)
    ab = ma@((adjacent@mb.transpose(-1, -2)) > 0).float()/neighbors
    ba = mb@((adjacent.transpose(-1, -2)@ma.transpose(-1, -2)) > 0).float()/neighbors
    overlap = .5*(ab+ba.transpose(-1, -2))
    return dict(features=features.reshape(b, v, tokens, 11), centers=centers.reshape(b, v, tokens, 3),
                covariance=covariance.reshape(b, v, tokens, 3, 3),
                world_centers=world_centers.reshape(b, v, tokens, 3), overlap=overlap,
                neighbors=ids, anchors=anchors.reshape(b, v, tokens))


def targets(overlap):
    weight = overlap*(overlap >= CONFIG['target_overlap'])
    valid = overlap.max(-1).values >= CONFIG['valid_overlap']
    return weight/weight.sum(-1, keepdim=True).clamp_min(1e-12), valid


def loss_function(logits, overlap):
    terms, present = [], []
    for score, truth in ((logits, overlap), (logits.transpose(-1, -2), overlap.transpose(-1, -2))):
        target, valid = targets(truth)
        value = -(target*score.log_softmax(-1)).sum(-1)
        terms.append((value*valid).sum()/valid.sum().clamp_min(1))
        present.append((valid.sum() > 0).to(value.dtype))
    return (terms[0]+terms[1])/(present[0]+present[1]).clamp_min(1)


def load_cache(name, verify=True):
    record = read(OUT/'data'/f'{name}.json')
    path = OUT/'data'/f'{name}.npz'
    if verify and sha(path) != record['sha256']:
        raise ValueError('cache identity '+name)
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def prepare(stage):
    lock = checked()
    if stage == 'eval' and not (OUT/'selected.json').is_file():
        raise ValueError('all models must be selected before evaluation cache')
    specifications = [('fit0', 'fit', 0, 'base'), ('fit1', 'fit', 1, 'base'), ('val', 'val', 0, 'base')]
    if stage == 'eval':
        specifications = [('eval_'+c, 'dev_eval', 0, c) for c in CONDITIONS]
    manifest = read(OLD_DATA/'manifest.json')
    destination = OUT/'data'
    destination.mkdir(exist_ok=True)
    prior_split, source = None, None
    for name, split, realization, condition in specifications:
        if (destination/f'{name}.json').exists() or (destination/f'{name}.npz').exists():
            raise ValueError('preserve existing cache '+name)
        if prior_split != split:
            record = next(x for x in manifest['partitions'] if x['split'] == split)
            if sha(record['path']) != lock['input_data'][split]['sha256']:
                raise ValueError('source identity')
            with np.load(record['path'], allow_pickle=False) as f:
                source = dict(points=f['points'], ids=f['ids'])
            prior_split = split
        collected = {}
        started = time.time()
        for begin in range(0, len(source['ids']), CONFIG['cache_batch_size']):
            pairs = [draw_pair(p, str(i), realization, condition) for p, i in
                     zip(source['points'][begin:begin+CONFIG['cache_batch_size']],
                         source['ids'][begin:begin+CONFIG['cache_batch_size']])]
            batch = {k: torch.as_tensor(np.stack([x[k] for x in pairs]), device='cuda') for k in pairs[0]}
            result = tokenize(batch)
            for key, value in result.items():
                if key not in ('neighbors', 'anchors'):
                    collected.setdefault(key, []).append(value.cpu().numpy())
            if begin == 0:
                np.savez_compressed(destination/f'{name}_sample.npz',
                    **{k: v[:2].cpu().numpy() for k, v in batch.items()},
                    **{'token_'+k: v[:2].cpu().numpy() for k, v in result.items()})
            if begin % (32*CONFIG['cache_batch_size']) == 0:
                print('CACHE', name, begin, len(source['ids']), flush=True)
        arrays = {k: np.concatenate(v) for k, v in collected.items()}
        arrays['ids'] = source['ids']
        if any(not np.isfinite(v).all() for k, v in arrays.items() if k != 'ids'):
            raise ValueError('nonfinite cache')
        path = destination/f'{name}.npz'
        np.savez(path, **arrays)
        overlap = arrays['overlap']
        counts = [(overlap.max(axis=ax) >= .3).sum(axis=1) for ax in (2, 1)]
        write(destination/f'{name}.json', dict(name=name, split=split, realization=realization,
              condition=condition, sha256=sha(path), count=len(source['ids']),
              source_sha256=record['sha256'], seconds=time.time()-started,
              valid_query_counts=[int(x.sum()) for x in counts],
              zero_valid_pair_count=int(((counts[0]+counts[1]) == 0).sum()),
              source_ids_disjoint=True, fresh_confirmation=False))
        print('CACHED', name, len(source['ids']), 'valid', [int(x.sum()) for x in counts], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('fit', 'eval'))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepare(parser.parse_args().stage)
