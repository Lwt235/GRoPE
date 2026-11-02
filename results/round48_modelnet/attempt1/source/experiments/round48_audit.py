"""Direct positional formula checks and full object-level metric replay.

These are independent formula paths by the same author, not an independent lab
replication. Small floating-point warnings are reported alongside task results.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from scipy.special import logsumexp, ndtr

from models.round48_point_transformer import ROLES, GAUSSIAN
from experiments.round48_common import OUT, SEEDS, CONFIG, VARIANTS, checked, sha, read, write


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def pair_energy(role, state, features, x, y):
    n = len(x)
    result = np.zeros((2, n, len(y)), dtype=np.float64)
    size = 4 if role == 'isotropic' else 6 if role in ('diagonal', 'fixed') else 9
    if role == 'fixed':
        raw = np.zeros((2, n, size))
    elif role == 'static':
        raw = np.broadcast_to(array(state['position_field.raw'])[:, None, :], (2, n, size))
    else:
        raw = (array(features)@array(state['position_field.predictor.weight']).T
               + array(state['position_field.predictor.bias'])).reshape(n, 2, size).transpose(1, 0, 2)
    for head in range(2):
        for i in range(n):
            mean = 4*np.tanh(raw[head, i, :3]/4)
            shape = 3*np.tanh(raw[head, i, 3:]/3)
            diagonal = np.repeat(shape[0], 3) if role == 'isotropic' else shape[:3]
            lower = np.zeros((3, 3))
            if role in ('conditional', 'static', 'free_quadratic'):
                lower[np.tril_indices(3, -1)] = shape[3:]
            delta = array(y)-array(x)[i]
            if role == 'free_quadratic':
                u = np.diag(diagonal)+lower
                precision = (np.eye(3)+u+u.T)/16
                result[head, i] = -.5*np.einsum('ni,ij,nj->n', delta, precision, delta)+delta@mean/16
            else:
                factor = np.diag(np.exp(diagonal))+lower
                precision = factor@factor.T/16
                shifted = delta-mean
                result[head, i] = -.5*np.einsum('ni,ij,nj->n', shifted, precision, shifted)
    return result


def phase(role, state, points):
    x = array(points)
    if role in ('fourier', 'mlp'):
        p = {k.removeprefix('position_field.'): array(v) for k, v in state.items()
             if k.startswith('position_field.')}
        z = x/12
        if role == 'fourier':
            angle = z@(p['base_frequency']+12*p['frequency_offset']).T+p['phase_offset']
            basis = np.concatenate((np.cos(angle), np.sin(angle)), -1)
        else:
            hidden = z@(p['base_mlp']+p['mlp_offset'])+p['bias']
            basis = hidden*ndtr(hidden)
        x = x+12*(basis/p['initial_rms'])@p['projection']
    return x@array(state['frequency']).T


def rotate(content, angles):
    value = array(content).reshape(*content.shape[:-1], -1, 2)
    a, b = value[..., 0], value[..., 1]
    c, s = np.cos(angles), np.sin(angles)
    return np.stack((a*c-b*s, a*s+b*c), -1).reshape(content.shape)


def check_sample(role, state, path):
    with np.load(path, allow_pickle=False) as archive:
        z = {k: archive[k] for k in archive.files}
    q, k = array(z['q']), array(z['k'])
    if role in ('rope', 'fourier', 'mlp'):
        q = rotate(q, phase(role, state, z['qmean']))
        k = rotate(k, phase(role, state, z['kmean']))
    temperature = np.exp(min(float(array(state['layers.0.log_temperature'])), np.log(60)))
    scores = temperature*(q@k.swapaxes(-1, -2))
    if role in GAUSSIAN:
        scores += pair_energy(role, state, z['features'], z['qmean'], z['kmean'])
    probability = np.exp(scores-logsumexp(scores, axis=-1, keepdims=True))
    error = float(np.abs(probability-z['probability']).max())
    return dict(p_max_error=error,
                winner_changes=int(np.sum(probability.argmax(-1) != z['probability'].argmax(-1))),
                warning=bool(error > 5e-5),
                scope='first layer, first selection-validation cloud, eight rows and all keys/two heads')


def object_metrics(logits, labels):
    scores = array(logits)
    correct = scores.argmax(-1) == labels
    ce = logsumexp(scores, axis=-1)-scores[np.arange(len(labels)), labels]
    by_class = [float(correct[labels == c].mean()) for c in range(40)]
    return dict(oa=float(correct.mean()), mca=float(np.mean(by_class)), ce=float(ce.mean()),
                correct=int(correct.sum()), count=len(labels), by_class=by_class)


def training(out):
    checked(out)
    dest = out/'audit_training'
    if dest.exists():
        raise ValueError('preserve training seal')
    for seed in SEEDS:
        if not read(out/f'seed_{seed}_complete.json')['completed']:
            raise ValueError('all training runs required before dev_eval')
    rows, warnings, identities = [], [], {}
    val_manifest = next(r for r in read(out/'data/manifest.json')['partitions'] if r['split'] == 'val')
    if sha(val_manifest['path']) != val_manifest['sha256']:
        raise ValueError('validation data identity')
    with np.load(val_manifest['path'], allow_pickle=False) as z:
        val_ids, val_labels = z['ids'], z['labels']
    for role in ROLES:
        for seed in SEEDS:
            path = out/'runs'/role/str(seed)
            record = read(path/'run.json')
            for name, digest in record['hashes'].items():
                if sha(path/name) != digest:
                    raise ValueError('training output changed')
            identity = [record['initial_shared_sha256'], record['batch_augmentation_sha256']]
            if identities.setdefault(str(seed), identity) != identity:
                raise ValueError('unpaired backbone initialization or stochastic data stream')
            history = read(path/'history.json')
            if [r['epoch'] for r in history['validation']] != list(range(0, 121, 5)):
                raise ValueError('selection schedule changed')
            if min(history['validation'], key=lambda r: r['ce'])['epoch'] != record['best_epoch']:
                raise ValueError('checkpoint selection differs from frozen rule')
            if len(history['training']) != 120:
                raise ValueError('incomplete training budget')
            state = torch.load(path/'best.pt', weights_only=True, map_location='cpu')['state_dict']
            numeric = check_sample(role, state, path/'score_sample.npz')
            with np.load(path/'validation.npz', allow_pickle=False) as z:
                if not np.array_equal(z['ids'], val_ids) or not np.array_equal(z['labels'], val_labels):
                    raise ValueError('selection population identity')
                metric = object_metrics(z['logits'], z['labels'])
            ce_error = abs(metric['ce']-record['best_val_ce'])
            if numeric['warning'] or ce_error > 2e-5:
                warnings.append(dict(role=role, seed=seed, numeric=numeric, val_ce_error=ce_error))
            rows.append(dict(role=role, seed=seed, path=str(path),
                checkpoint_sha256=sha(path/'best.pt'), run_sha256=sha(path/'run.json'),
                numeric=numeric, validation_metric=metric, val_ce_error=ce_error))
            print('SEALED', role, seed, numeric['p_max_error'], flush=True)
    dest.mkdir()
    write(dest/'completion.json', dict(records=rows, warnings=warnings, paired_identity=identities,
        model_dev_eval_performed=False, official_test_access=False, independently_authored=False,
        unix=time.time()))


def metrics(out):
    checked(out)
    seal = read(out/'audit_training/completion.json')
    dest = out/'metrics'
    if dest.exists():
        raise ValueError('preserve metric report')
    for seed in SEEDS:
        if not read(out/f'export_{seed}_complete.json')['completed']:
            raise ValueError('all exports required')
    data_record = next(r for r in read(out/'data/manifest.json')['partitions'] if r['split'] == 'dev_eval')
    if sha(data_record['path']) != data_record['sha256']:
        raise ValueError('development data changed')
    with np.load(data_record['path'], allow_pickle=False) as z:
        ids, labels = z['ids'], z['labels']
    table, discrepancies, sources, permutations = {}, [], {}, []
    for rec in seal['records']:
        role, seed = rec['role'], rec['seed']
        path = Path(rec['path'])
        record = read(path/'dev_eval.json')
        if record['checkpoint_sha256'] != rec['checkpoint_sha256']:
            raise ValueError('checkpoint mismatch')
        if sha(path/'dev_eval.npz') != record['predictions_sha256']:
            raise ValueError('prediction identity')
        sources[str(path/'dev_eval.npz')] = record['predictions_sha256']
        row = {}
        with np.load(path/'dev_eval.npz', allow_pickle=False) as z:
            if not np.array_equal(z['ids'], ids) or not np.array_equal(z['labels'], labels):
                raise ValueError('evaluation population changed')
            for variant in VARIANTS:
                if z[variant].shape != (len(ids), 40) or not np.isfinite(z[variant]).all():
                    raise ValueError('invalid classification predictions')
                row[variant] = object_metrics(z[variant], labels)
                old = record['metrics'][variant]
                discrepancies.append(dict(role=role, seed=seed, variant=variant,
                    correct_difference=row[variant]['correct']-old['correct'],
                    ce_error=abs(row[variant]['ce']-old['ce'])))
            permutations.append(dict(role=role, seed=seed,
                max_logit_error=float(abs(z['permuted']-z['base']).max()),
                winner_changes=int(np.sum(z['permuted'].argmax(-1) != z['base'].argmax(-1)))))
        table.setdefault(role, {})[str(seed)] = row
    means = {role: {variant: {key: float(np.mean([table[role][str(seed)][variant][key] for seed in SEEDS]))
        for key in ('oa', 'mca', 'ce')} for variant in VARIANTS} for role in ROLES}
    comparisons = []
    for control in tuple(r for r in ROLES if r != 'conditional'):
        deltas = [100*(table['conditional'][str(s)]['base']['oa']-table[control][str(s)]['base']['oa'])
                  for s in SEEDS]
        comparisons.append(dict(control=control, oa_difference_pp=float(np.mean(deltas)),
            mca_difference_pp=100*(means['conditional']['base']['mca']-means[control]['base']['mca']),
            ce_difference=means['conditional']['base']['ce']-means[control]['base']['ce'],
            paired_seed_pp=deltas, winning_seeds=sum(d > 0 for d in deltas)))
    strong = [r for r in comparisons if r['control'] in ('rope', 'fourier', 'mlp', 'nope', 'free_quadratic')]
    promising = all(r['oa_difference_pp'] >= 1 and r['winning_seeds'] >= 4
                    and r['mca_difference_pp'] >= -.5 for r in strong)
    dest.mkdir()
    write(dest/'summary.json', dict(means=means, by_role_seed=table, comparisons=comparisons,
        promising_development_only=promising, main_role='conditional', total_models=50,
        official_test_access=False, fresh_confirmation=False, goal_complete=False,
        interpretation='Another task cannot erase the R47 matching failure; no canonical Point Transformer reproduction.'))
    write(dest/'verification.json', dict(objects=len(ids), models=50, variants=len(VARIANTS),
        metric_replay=discrepancies, permutation_checks=permutations,
        sampled_PE_warnings=seal['warnings'], input_hashes=sources,
        independent_formula_paths=True, independently_authored=False))
    print('METRICS', means, 'promising_development_only', promising, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('training', 'metrics'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.stage](args.out)
