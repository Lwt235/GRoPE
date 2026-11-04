"""Read-only, fit-only description of the two covariance input regimes."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.round69_compact_relation import descriptors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/geometry_scale_diagnostic/attempt1'
QUANTILES = [0, .05, .25, .5, .75, .95, 1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rng(identity):
    salt = 'geometry-scale-fit-only-v1:' + identity
    return np.random.default_rng(int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], 'little'))


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.size and np.isfinite(values).all()
    return dict(count=int(values.size), mean=float(values.mean()),
                quantiles=np.quantile(values, QUANTILES).tolist())


def describe(identity, group, centers, covariance, pair_count, reference_length):
    centers, covariance = np.asarray(centers, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
    n = len(centers)
    eigen = np.linalg.eigvalsh(covariance)
    assert eigen.min() > 0 and np.isfinite(eigen).all()
    inverse = np.linalg.inv(covariance)
    distance2 = np.square(centers[:, None] - centers[None, :]).sum(-1)
    np.fill_diagonal(distance2, np.inf)
    nearest = np.sqrt(distance2.min(-1))
    codes = rng(identity + ':pairs').choice(n * (n - 1), pair_count, replace=False)
    i, j = codes // (n - 1), codes % (n - 1)
    j = j + (j >= i)
    delta = centers[i] - centers[j]
    d2 = np.square(delta).sum(-1)
    mi = np.einsum('nd,nde,ne->n', delta, inverse[i], delta)
    mj = np.einsum('nd,nde,ne->n', delta, inverse[j], delta)
    ti = np.einsum('nde,ned->n', inverse[i], covariance[j])
    tj = np.einsum('nde,ned->n', inverse[j], covariance[i])
    phi = np.column_stack([np.sqrt(d2) / reference_length,
                           np.log1p(np.maximum(np.column_stack([mi, mj, ti, tj]), 0))])
    trace = eigen.sum(-1)
    distinct = d2 > 0
    positive_nearest = nearest > 0
    raw = dict(axis_min_over_length=np.sqrt(eigen[:, 0]) / reference_length,
               axis_mid_over_length=np.sqrt(eigen[:, 1]) / reference_length,
               axis_max_over_length=np.sqrt(eigen[:, 2]) / reference_length,
               covariance_condition=eigen[:, 2] / eigen[:, 0],
               nearest_distance_over_length=nearest / reference_length,
               rms_support_over_nearest=np.sqrt(trace[positive_nearest] / 3) / nearest[positive_nearest],
               directed_metric_over_squared_normalized_distance=mi[distinct] / (d2[distinct] / reference_length**2),
               directed_metric_over_same_trace_isotropic=mi[distinct] / (d2[distinct] / (trace[i[distinct]] / 3)))
    raw.update({name:phi[:, k] for k, name in enumerate(
        ['descriptor_distance', 'descriptor_metric_i', 'descriptor_metric_j',
         'descriptor_trace_i', 'descriptor_trace_j'])})
    # Direct linear solves provide an independent contraction check on fixed pairs.
    sel = np.arange(min(32, pair_count))
    solved = np.linalg.solve(covariance[i[sel]], delta[sel, :, None])[..., 0]
    direct = np.einsum('nd,nd->n', delta[sel], solved)
    solve_relative_error = float(np.max(abs(direct - mi[sel]) / np.maximum(1, abs(direct))))
    h = hashlib.sha256()
    h.update(centers.tobytes()); h.update(covariance.tobytes()); h.update(codes.tobytes())
    record = dict(identity=identity, group=group, tokens=n, sampled_ordered_pairs=pair_count,
        sampled_array_sha256=h.hexdigest(), zero_distance_pairs=int((~distinct).sum()),
        zero_nearest_tokens=int((~positive_nearest).sum()), solve_relative_error=solve_relative_error,
        metrics={name:quantiles(value) for name, value in raw.items()})
    return record, raw


def run():
    torch.set_num_threads(4)
    plan = read(OUT / 'plan.json')
    assert not (OUT / 'result.json').exists()
    for path, digest in plan['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in plan['manifest_sha256'].items():
        assert sha(ROOT / path) == digest, path
    native = ROOT / 'results/round65_actual_gs/attempt1/data'
    manifest = read(native / 'manifest.json')
    fit = next(row for row in manifest['partitions'] if row['split'] == 'fit')
    verified = {}
    arrays = {}
    for name in ['centers', 'covariance', 'labels']:
        path = native / 'fit' / (name + '.npy')
        verified[str(path)] = sha(path)
        assert verified[str(path)] == fit['hashes'][path.name]
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    region_manifest = read(ROOT / 'results/round68_fixed_region/attempt1/data/manifest.json')
    cache = region_manifest['caches']['fit0']
    verified[cache['path']] = sha(cache['path'])
    assert verified[cache['path']] == cache['sha256']
    with np.load(cache['path'], allow_pickle=False) as z:
        region = {name:z[name] for name in ['centers', 'covariance', 'overlap', 'ids', 'scenes']}
    items = []
    for category in range(60):
        indices = np.flatnonzero(arrays['labels'] == category).tolist()
        indices.sort(key=lambda i:fit['ids'][i])
        for i in indices[:plan['native_objects_per_class']]:
            identity = fit['ids'][i]
            token_ids = rng(identity + ':tokens').choice(arrays['centers'].shape[1], plan['native_tokens'], replace=False)
            items.append(('native', identity, 'class_' + str(category),
                          arrays['centers'][i, token_ids], arrays['covariance'][i, token_ids]))
    eligible = (region['overlap'].max(-1) >= .3).any(-1)
    for scene in sorted(set(region['scenes'].tolist())):
        indices = np.flatnonzero((region['scenes'] == scene) & eligible).tolist()
        indices.sort(key=lambda i:str(region['ids'][i]))
        assert len(indices) >= plan['region_pairs_per_scene']
        for i in indices[:plan['region_pairs_per_scene']]:
            for view in range(2):
                items.append(('observed', str(region['ids'][i]) + ':view' + str(view), scene,
                              region['centers'][i, view], region['covariance'][i, view]))
    rows, pooled, checks = [], {}, []
    for kind, identity, group, centers, covariance in items:
        row, raw = describe(identity, group, centers, covariance, plan['ordered_pairs_per_context'], plan['reference_length'])
        row['kind'] = kind
        rows.append(row)
        for name, value in raw.items():
            pooled.setdefault(kind, {}).setdefault(name, []).append(value)
        # Check the first selected context per family against the unmodified PE implementation.
        if kind not in {r['kind'] for r in checks}:
            c, s = np.asarray(centers[:16]), np.asarray(covariance[:16])
            with torch.no_grad():
                actual = descriptors('relative5', torch.from_numpy(c[None]), torch.from_numpy(s[None])).double().numpy()[0]
            delta = c.astype(np.float64)[:, None] - c.astype(np.float64)[None]
            inv = np.linalg.inv(s.astype(np.float64))
            mi = np.einsum('ijd,ide,ije->ij', delta, inv, delta)
            tr = np.einsum('ide,jed->ij', inv, s.astype(np.float64))
            expected = np.concatenate([np.linalg.norm(delta, axis=-1)[..., None] / plan['reference_length'],
                np.log1p(np.maximum(np.stack([mi, mi.T, tr, tr.T], -1), 0))], -1)
            checks.append(dict(kind=kind, maximum_descriptor_error=float(abs(actual - expected).max())))
    summary = {kind:{name:quantiles(np.concatenate(values)) for name, values in cols.items()}
               for kind, cols in pooled.items()}
    result = dict(created_unix=time.time(), plan_sha256=sha(OUT / 'plan.json'),
        source_sha256=sha(Path(__file__)), verified_input_sha256=verified,
        quantile_probabilities=QUANTILES, context_counts={k:sum(r['kind'] == k for r in rows) for k in pooled},
        summary=summary, contexts=rows, implementation_checks=checks,
        maximum_solve_relative_error=max(r['solve_relative_error'] for r in rows),
        scope=plan['scope'], performance_evidence=False, fresh_confirmation=False,
        changes_any_model=False, goal_complete=False, interpretation_pending=True)
    write(OUT / 'result.json', result)
    print(json.dumps({k:result[k] for k in ['context_counts', 'implementation_checks', 'maximum_solve_relative_error']}), flush=True)


if __name__ == '__main__':
    run()
