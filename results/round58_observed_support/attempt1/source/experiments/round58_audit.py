"""Bounded independent NumPy label, Gaussian and metric arithmetic."""
import argparse
import time
import numpy as np
from experiments.round58_common import OUT, CONFIG, ROLES, SEEDS, CONDITIONS, checked, read, write, sha


def direct_gaussian_distance(centers, covariances):
    n = len(centers)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            delta = centers[i]-centers[j]
            kl = 0.
            for a, b in ((i, j), (j, i)):
                kl += .5*(np.trace(np.linalg.solve(covariances[b], covariances[a]))
                           + delta@np.linalg.solve(covariances[b], delta)-3
                           + np.linalg.slogdet(covariances[b])[1]-np.linalg.slogdet(covariances[a])[1])
            result[i, j] = np.sqrt(max(0, kl))  # sqrt(KLij+KLji)=sqrt(2J)
    np.fill_diagonal(result, 0)
    return result


def audit_cache_sample(path):
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    errors = dict(center=0., covariance=0., overlap=0., world_center=0.)
    for b in range(len(arrays['points'])):
        if np.intersect1d(*arrays['source_ids'][b]).size:
            raise ValueError('same source point in both views')
        supports = []
        for v in range(2):
            ids = arrays['token_neighbors'][b, v]
            points = arrays['points'][b, v].astype(np.float64)[ids]
            center = points.mean(1)
            residual = points-center[:, None]
            covariance = np.einsum('nki,nkj->nij', residual, residual)/ids.shape[-1]+np.eye(3)*.02**2
            world = arrays['world'][b, v].astype(np.float64)[ids].mean(1)
            errors['center'] = max(errors['center'], float(abs(center-arrays['token_centers'][b, v]).max()))
            errors['covariance'] = max(errors['covariance'], float(abs(covariance-arrays['token_covariance'][b, v]).max()))
            errors['world_center'] = max(errors['world_center'], float(abs(world-arrays['token_world_centers'][b, v]).max()))
            supports.append(arrays['clean'][b, v].astype(np.float64)[ids])
        for i, first in enumerate(supports[0]):
            for j, second in enumerate(supports[1]):
                adjacent = np.square(first[:, None]-second[None]).sum(-1) < .05**2
                overlap = .5*(adjacent.any(0).mean()+adjacent.any(1).mean())
                errors['overlap'] = max(errors['overlap'], abs(overlap-arrays['token_overlap'][b, i, j]))
    if max(errors.values()) > 2e-5:
        raise ValueError('substantive cache mismatch '+str(errors))
    return dict(path=str(path), errors=errors, source_ids_disjoint=True)


def metrics_numpy(logits, overlap, world):
    logits, overlap, world = [np.asarray(x, dtype=np.float64) for x in (logits, overlap, world)]
    sums, counts, direction_ce = {}, [], []
    per_object = {}
    for reverse in (False, True):
        score, truth = (logits.swapaxes(-1, -2), overlap.swapaxes(-1, -2)) if reverse else (logits, overlap)
        a, b = (world[:, 1], world[:, 0]) if reverse else (world[:, 0], world[:, 1])
        weight = truth*(truth >= .1)
        label = weight/np.maximum(weight.sum(-1, keepdims=True), 1e-12)
        valid = truth.max(-1) >= .3
        shifted = score-score.max(-1, keepdims=True)
        logp = shifted-np.log(np.exp(shifted).sum(-1, keepdims=True))
        probability = np.exp(logp)
        index = score.argmax(-1)[..., None]
        picked = np.take_along_axis(truth, index, axis=-1)[..., 0]
        distances = np.linalg.norm(a[:, :, None]-b[:, None], axis=-1)
        hard = np.take_along_axis(distances, index, axis=-1)[..., 0]
        values = dict(region_recall=(picked >= .3).astype(float), overlap=picked,
                      center_pck=(hard <= .1).astype(float), hard_epe=hard,
                      soft_epe=np.linalg.norm(probability@b-a, axis=-1), ce=-(label*logp).sum(-1),
                      oracle_center_pck=(distances.min(-1) <= .1).astype(float), oracle_overlap=truth.max(-1),
                      overlap_oracle_center_pck=(np.take_along_axis(distances, truth.argmax(-1)[..., None], -1)[..., 0] <= .1).astype(float),
                      center_oracle_region_recall=(np.take_along_axis(truth, distances.argmin(-1)[..., None], -1)[..., 0] >= .3).astype(float))
        count = int(valid.sum())
        counts.append(count)
        for key, value in values.items():
            object_sums = (value*valid).sum(-1)
            sums.setdefault(key, []).append(float(object_sums.sum()))
            per_object.setdefault(key, []).append(object_sums)
        per_object.setdefault('valid_count', []).append(valid.sum(-1))
        if count:
            direction_ce.append(float((values['ce']*valid).sum())/count)
    total = sum(counts)
    per_object = {k: np.stack(v).sum(0) for k, v in per_object.items()}
    result = dict(**{k: sum(v)/max(1, total) for k, v in sums.items()},
                  selection_ce=float(np.mean(direction_ce)) if direction_ce else 0., valid_counts=counts,
                  eligible_fraction=total/(2*len(logits)*logits.shape[1]))
    return result, per_object


def training():
    checked()
    if (OUT/'selected.json').exists():
        raise ValueError('selection already sealed')
    from experiments.round58_data import load_cache
    import torch
    from models.round58_support_pe import SupportMatcher, scalar_geometry
    torch.set_num_threads(4)
    validation = load_cache('val')
    actual_data = {n: read(OUT/'data'/f'{n}.json')['sha256'] for n in ('fit0', 'fit1', 'val')}
    for name, digest in actual_data.items():
        if sha(OUT/'data'/f'{name}.npz') != digest:
            raise ValueError('cache changed before sealing')
    small_c = validation['centers'][0, 0, :8].astype(np.float64)
    small_s = validation['covariance'][0, 0, :8].astype(np.float64)
    _, m, s = scalar_geometry(torch.from_numpy(small_c), torch.from_numpy(small_s))
    gaussian_error = float(np.abs((.5*(m+s)).sqrt().numpy()-direct_gaussian_distance(small_c, small_s)).max())
    if gaussian_error > 1e-6:
        raise ValueError('direct actual-input Gaussian discrepancy')
    inputs = tuple(torch.from_numpy(validation[k][:2]) for k in ('features', 'centers', 'covariance'))
    rotation = torch.tensor([[[0., -1, 0], [1, 0, 0], [0, 0, 1]], [[1., 0, 0], [0, 0, -1], [0, 1, 0]]])
    moved = inputs[1]@rotation.transpose(-1, -2)+torch.tensor([.7, -.3, .2])
    moved_cov = rotation[:, None]@inputs[2]@rotation[:, None].transpose(-1, -2)
    records, paired = [], {}
    for role in ROLES:
        for seed in SEEDS:
            path = OUT/'runs'/role/str(seed)
            run, history = read(path/'run.json'), read(path/'history.json')
            if (run['role'], run['seed'], run['epochs'], run['steps']) != (role, seed, 60, 14880):
                raise ValueError('fit identity/budget')
            if run['data_sha256'] != actual_data:
                raise ValueError('unmatched fit inputs')
            for name, digest in run['hashes'].items():
                if sha(path/name) != digest:
                    raise ValueError('run artifact changed')
            identity = (run['initial_shared_sha256'], run['stream_sha256'])
            if paired.setdefault(seed, identity) != identity:
                raise ValueError('unpaired stream or common initialization')
            best = min(history['validation'], key=lambda x: x['selection_ce'])
            if best['epoch'] != run['selected_epoch'] or best['selection_ce'] != run['best_val_ce']:
                raise ValueError('selection mismatch')
            with np.load(path/'validation.npz', allow_pickle=False) as f:
                if not np.array_equal(f['ids'], validation['ids']):
                    raise ValueError('validation IDs')
                metric, _ = metrics_numpy(f['logits'], validation['overlap'], validation['world_centers'])
            error = abs(metric['selection_ce']-best['selection_ce'])
            if error > 1e-6 or not run['shared_parameters_changed']:
                raise ValueError('validation arithmetic or no learned network')
            model = SupportMatcher(role).eval()
            checkpoint = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
            if (checkpoint['role'], checkpoint['seed'], checkpoint['epoch']) != (role, seed, run['selected_epoch']):
                raise ValueError('checkpoint identity')
            model.load_state_dict(checkpoint['state_dict'])
            with torch.no_grad():
                base = model(*inputs)
                rigid = model(inputs[0], moved, moved_cov)
                permuted = model(inputs[0].flip(-2), inputs[1].flip(-2), inputs[2].flip(-3)).flip(-2, -1)
                rigid_error = float((base.softmax(-1)-rigid.softmax(-1)).abs().max())
                order_error = float((base-permuted).abs().max())
                changed = int((base.argmax(-1) != rigid.argmax(-1)).sum())
            if rigid_error > .002 or order_error > .002:
                raise ValueError('substantive trained-network geometric discrepancy')
            records.append(dict(role=role, seed=seed, selected_epoch=run['selected_epoch'],
                    checkpoint_sha256=sha(path/'best.pt'), run_sha256=sha(path/'run.json'),
                    validation_ce_error=error, completed_unix=run['completed_unix'],
                    rigid_probability_error=rigid_error, permutation_logit_error=order_error,
                    rigid_winner_changes=changed))
    write(OUT/'selected.json', dict(runs=records, sealed_unix=time.time(), paired_seeds=list(paired),
                                   lock_sha256=sha(OUT/'lock.json'), fresh_confirmation=False,
                                   actual_gaussian_scalar_error=gaussian_error, data_sha256=actual_data))
    print('SEALED 18 selected models', flush=True)


def metrics():
    checked()
    from experiments.round58_data import load_cache
    selected = read(OUT/'selected.json')
    rows, errors = [], []
    for condition in CONDITIONS:
        cache = load_cache('eval_'+condition)
        for role in ROLES:
            for seed in SEEDS:
                path = OUT/'evaluation'/role/str(seed)/(condition+'.npz')
                record = read(path.with_suffix('.json'))
                if sha(path) != record['predictions_sha256'] or record['exported_unix'] < selected['sealed_unix']:
                    raise ValueError('prediction identity/timing')
                with np.load(path, allow_pickle=False) as f:
                    if not np.array_equal(f['ids'], cache['ids']):
                        raise ValueError('prediction IDs')
                    actual, per_object = metrics_numpy(f['logits'], cache['overlap'], cache['world_centers'])
                for key, value in actual.items():
                    old = record['metrics'][key]
                    if key == 'valid_counts':
                        if value != old:
                            raise ValueError('query counts disagree')
                    else:
                        error = abs(value-old)
                        errors.append(error)
                        if error > 1e-6:
                            raise ValueError('substantive metric discrepancy')
                dest = OUT/'metrics'/role/str(seed)
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest/(condition+'_objects.npz'), **per_object, ids=cache['ids'])
                rows.append(dict(role=role, seed=seed, condition=condition, **actual))
    write(OUT/'metrics/summary.json', dict(rows=rows, evaluated_models=18, groups=len(rows),
                                          max_metric_error=max(errors), fresh_confirmation=False,
                                          independent_arithmetic=True, goal_complete=False))
    print('METRICS', len(rows), 'maxerror', max(errors), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('training', 'metrics'))
    globals()[parser.parse_args().stage]()
