"""Frozen normalized-PE covariance interventions; descriptive mechanism audit.

No training, checkpoint replacement, or new independent-test claim. The same
normalized_shape checkpoints are evaluated after one/both covariance inputs are
changed. All means, RGB features, masks and targets stay fixed. Temperature is
either frozen, shared across interventions and selected on old validation, or
separately selected per intervention on that same validation set.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.isometric_confirmation_analysis import (
    SEEDS, archived_matcher, aligned_real_inputs, real_row_identity,
    paired_camera_seed_statistics, sha256, write_json)
from experiments.query_support import use_fitted_query
from experiments.matching_probe import digest


CONTROLS = ['original'] + [f'{side}_{mode}' for mode in ['fixed', 'trace', 'area', 'shuffled']
                           for side in ['query', 'key', 'both']]
FACTORS = np.exp(np.linspace(np.log(.25), np.log(4.), 25))
METRICS = ['cross_entropy', 'js', 'top1_target_mass', 'dominant_region_accuracy']


def covariance_control(covariance, mode):
    identity = torch.eye(2, dtype=covariance.dtype, device=covariance.device)
    if mode == 'fixed':
        return identity.expand_as(covariance)/12
    if mode == 'trace':
        scale = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)/2
        return identity*scale[..., None, None]
    if mode == 'area':
        scale = torch.linalg.det(covariance).clamp_min(1e-12).sqrt()
        return identity*scale[..., None, None]
    if mode == 'shuffled':
        return covariance.roll(13, dims=-3)
    raise ValueError(mode)


def intervention(data, name):
    if name == 'original':
        return data
    side, mode = name.split('_', 1)
    changed = dict(data)
    for prefix in (['q', 'k'] if side == 'both' else ['q' if side == 'query' else 'k']):
        field = f'projected_{prefix}cov'
        changed[field] = covariance_control(data[field], mode)
    return changed


@torch.no_grad()
def valid_scores(model, data, name, batch=32):
    transformed = intervention(data, name)
    rows = []
    for start in range(0, len(data['q']), batch):
        part = {key: value[start:start+batch] for key, value in transformed.items()}
        score = model(part)[..., :-1]
        rows.append(score[part['label_valid']].double().cpu().numpy())
    scores = np.concatenate(rows)
    mask = data['key_valid'][:, None].expand(-1, data['q'].shape[1], -1)
    valid_key = mask[data['label_valid']].cpu().numpy()
    return scores, valid_key


def metric_rows(scores, valid_keys, targets, factor=1.):
    scaled = np.where(valid_keys, scores*factor, -10000.)
    scaled -= scaled.max(-1, keepdims=True)
    probability = np.exp(scaled)
    probability /= probability.sum(-1, keepdims=True)
    midpoint = (probability+targets)/2
    ce = -(targets*np.log(np.maximum(probability, 1e-30))).sum(-1)
    js = .5*(targets*np.log(np.maximum(targets, 1e-30)/np.maximum(midpoint, 1e-30))
              + probability*np.log(np.maximum(probability, 1e-30)/np.maximum(midpoint, 1e-30))).sum(-1)
    selected = probability.argmax(-1)
    return {'cross_entropy': ce, 'js': js,
            'top1_target_mass': np.take_along_axis(targets, selected[:, None], -1)[:, 0],
            'dominant_region_accuracy': (selected == targets.argmax(-1)).astype(float)}


def chosen_temperature(scores, valid_keys, targets):
    losses = [float(metric_rows(scores, valid_keys, targets, factor)['cross_entropy'].mean())
              for factor in FACTORS]
    index = int(np.argmin(losses))
    return float(FACTORS[index]), losses


def load_data(root, scene, split):
    data, metadata, pairs, _ = aligned_real_inputs(root, scene, split)
    data, metadata = use_fitted_query(data, metadata, Path(root)/scene)
    targets = data['soft_labels'][data['label_valid']].double().numpy()
    identities = real_row_identity(pairs, data['label_valid'].numpy(), scene)
    return data, targets, identities, dict(dataset_sha256=digest(data), metadata=metadata,
                                          valid_rows=len(identities), pairs=len(pairs))


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__)
    protocol = dict(config=vars(args), code_sha256=sha256(source),
        purpose='descriptive frozen-checkpoint input intervention, not retrained ablation or independent confirmation',
        checkpoints='round3_normalized_confirmation/training_{raw,frozen}/checkpoints/{seed}-normalized_shape.pt',
        conditions=CONTROLS,
        intervention='replace query, key, or both projected covariances BEFORE the same .25^2 I resolution blur; means/RGB/labels/validity unchanged',
        definitions=dict(fixed='I/12 patch^2', trace='tr(C)/2 I', area='sqrt(det(C)) I',
                         shuffled='fixed cyclic roll of 13 regions, separately within each view'),
        temperatures=dict(frozen='original learned temperature',
            shared='single positive multiplier per checkpoint chosen by minimum mean old validation CE across all 13 interventions',
            separate='positive multiplier per checkpoint/intervention chosen by old validation CE',
            multiplier_grid=FACTORS.tolist()),
        data='old Teatime validation for temperature only; previously evaluated figurines+temple camera pairs for descriptive transfer; original GS parity only',
        inference='paired seed/query-camera bootstrap, fixed scenes stratified; 5 seeds; intervention shifts the learned input distribution, so degradation is not proof of necessary information',
        limits='conditional valid-support sparse SfM tracks; known GS reconstructions may include evaluation photos; no retraining, no fresh discovery claim, no new geometry predictor',
        cpu_threads=1)
    if (out/'protocol.json').exists():
        raise FileExistsError('use a new output directory for another audit')
    write_json(out/'protocol.json', protocol)
    (out/'source').mkdir(exist_ok=True)
    (out/'source'/source.name).write_bytes(source.read_bytes())
    started = time.time()
    records, calibration, metadata, checkpoint_hashes, implementations = [], [], {}, {}, {}
    arrays, identities = {}, {}
    archive = {}
    for kind in ['raw', 'frozen']:
        run = Path('results/round3_normalized_confirmation')/f'training_{kind}'
        manifest = json.loads((run/'manifest.json').read_text())
        matcher, implementation = archived_matcher(run, manifest)
        implementations[kind] = implementation
        if sha256('models/geometry.py') != manifest['source_hashes']['models/geometry.py']:
            raise ValueError('geometry dependency changed')
        root = manifest['config']['real_root']
        val, val_targets, _, val_meta = load_data(root, 'teatime', 'validation')
        fresh_root = Path('results')/('round3_fresh_real_data' if kind == 'raw' else 'round3_fresh_frozen_visual_data')
        tests = {}
        for scene in ['figurines', 'temple']:
            tests[scene] = load_data(fresh_root, scene, 'test')
        metadata[kind] = dict(validation=val_meta, evaluation={s: d[3] for s, d in tests.items()})
        identities[kind] = [r for scene in ['figurines', 'temple'] for r in tests[scene][2]]
        for seed in SEEDS:
            checkpoint = run/'checkpoints'/f'{seed}-normalized_shape.pt'
            checkpoint_hashes[str(checkpoint)] = sha256(checkpoint)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = matcher('normalized_shape', payload['features'], payload['resolution'])
            model.load_state_dict(payload['state_dict'], strict=True)
            model.eval()
            validation_scores, validation_curves = {}, {}
            for control in CONTROLS:
                score, mask = valid_scores(model, val, control)
                factor, losses = chosen_temperature(score, mask, val_targets)
                validation_scores[control] = (score, mask, factor)
                validation_curves[control] = losses
            shared_losses = np.array(list(validation_curves.values())).mean(0)
            shared = float(FACTORS[int(shared_losses.argmin())])
            calibration.append(dict(kind=kind, seed=seed, shared_multiplier=shared,
                                    shared_validation_curve=shared_losses.tolist(),
                                    separate_multipliers={c: v[2] for c, v in validation_scores.items()},
                                    separate_validation_curves=validation_curves))
            for control in CONTROLS:
                evaluated = {scene: valid_scores(model, test[0], control) for scene, test in tests.items()}
                for temperature, factor in [('frozen', 1.), ('shared', shared),
                                             ('separate', validation_scores[control][2])]:
                    by_scene = {}
                    for scene, (score, mask) in evaluated.items():
                        rows = metric_rows(score, mask, tests[scene][1], factor)
                        by_scene[scene] = rows
                        for metric, values in rows.items():
                            archive[f'{kind}/{seed}/{control}/{temperature}/{scene}/{metric}'] = values
                        records.append(dict(kind=kind, seed=seed, intervention=control,
                                            temperature=temperature, multiplier=factor, scene=scene,
                                            metrics={m: float(v.mean()) for m, v in rows.items()},
                                            valid_rows=len(score)))
                    for metric in METRICS:
                        key = (kind, control, temperature, metric)
                        arrays.setdefault(key, []).append(np.concatenate([by_scene[s][metric] for s in ['figurines', 'temple']]))
            print('AUDITED', kind, seed, flush=True)
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    summary, comparisons, recovery = {}, {}, {}
    for kind in ['raw', 'frozen']:
        for temperature in ['frozen', 'shared', 'separate']:
            for control in CONTROLS:
                prefix = f'{kind}/{temperature}/{control}'
                summary[prefix] = {metric: float(arrays[(kind, control, temperature, metric)].mean()) for metric in METRICS}
                summary[prefix]['seed_ce'] = {str(seed): float(value) for seed, value in zip(SEEDS,
                    arrays[(kind, control, temperature, 'cross_entropy')].mean(-1))}
                if control != 'original':
                    comparisons[prefix] = paired_camera_seed_statistics(
                        arrays[(kind, 'original', temperature, 'cross_entropy')],
                        arrays[(kind, control, temperature, 'cross_entropy')], identities[kind])
            for mode in ['fixed', 'trace', 'area', 'shuffled']:
                prefix = f'{kind}/{temperature}/{mode}'
                recovery[prefix] = paired_camera_seed_statistics(
                    arrays[(kind, 'both_'+mode, temperature, 'cross_entropy')],
                    arrays[(kind, 'key_'+mode, temperature, 'cross_entropy')], identities[kind])
    for path, expected in checkpoint_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f'checkpoint mutated: {path}')
    output = dict(protocol_sha256=sha256(out/'protocol.json'), summary=summary,
        original_minus_intervention_ce=comparisons, both_minus_key_only_ce=recovery,
        calibration=calibration, records=records, metadata=metadata,
        archived_implementations=implementations, checkpoint_sha256=checkpoint_hashes,
        elapsed_seconds=time.time()-started)
    write_json(out/'summary.json', output)
    np.savez_compressed(out/'row_metrics.npz', **archive)
    with (out/'means.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['kind','temperature','intervention',*METRICS])
        writer.writeheader()
        for name, values in summary.items():
            kind, temperature, intervention_name = name.split('/')
            writer.writerow(dict(kind=kind, temperature=temperature, intervention=intervention_name,
                                 **{metric: values[metric] for metric in METRICS}))
    lines = ['# Frozen normalized Gaussian PE: covariance mechanism audit', '',
        'Descriptive follow-up on previously evaluated camera pairs, not new confirmation. '
        'Ten frozen normalized_shape checkpoints (two visual inputs, five seeds); no training.', '',
        'All means/RGB/labels/masks are fixed. Covariances are changed before the existing .25² I blur. '
        'Temperature multipliers use old Teatime validation only. The separate-calibration column '
        'gives each intervention its own scalar; it cannot change candidate rankings.', '',
        '| Input | Intervention | Frozen CE | Shared-temp CE | Separate-temp CE | Frozen top1 target mass |',
        '|---|---|---:|---:|---:|---:|']
    for kind in ['raw','frozen']:
        for control in CONTROLS:
            means = [summary[f'{kind}/{t}/{control}'] for t in ['frozen','shared','separate']]
            lines.append(f"| {kind} | {control} | {means[0]['cross_entropy']:.6f} | {means[1]['cross_entropy']:.6f} | {means[2]['cross_entropy']:.6f} | {means[0]['top1_target_mass']:.6f} |")
    lines += ['', 'Interpretation must distinguish geometric information from matching of learned spectral filters. '
        'Removing only key shape can misalign its frequency allocation with the still-fitted query. '
        'If both-side replacement restores performance, the key-only ablation overstates a uniquely '
        'key-shape interpretation. Conversely, frozen intervention degradation alone cannot prove '
        'shape necessity: the content encoder was trained with full covariance, and retrained controls '
        'are required to separate information loss from input-distribution shift.', '',
        'See summary.json for per-scene/seed metrics, selected validation temperatures, paired camera/seed '
        'intervals, exact data/implementation/checkpoint hashes, and both-minus-key-only contrasts. '
        'Inference fixes scenes and resamples query cameras; it is not a population-of-scenes confidence claim.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE', output['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/round4_mechanism_audit')
    main(parser.parse_args())
