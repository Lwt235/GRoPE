"""All-Gaussian-scale fairness extension of the completed real scale stress.

Every scale is reported. No clean/stress evaluation result chooses a scale.
Old stress files, source snapshots and validation-selected settings stay intact.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.round4_real_analysis import completed_branch, real_inputs
from experiments.round4_real_scale_stress import FACTORS, SCENES, METRICS
from experiments.isometric_confirmation_analysis import (
    SEEDS, model_probability, real_probability_metrics, real_row_identity,
    paired_camera_seed_statistics, sha256, write_json)
from experiments.matching_probe import digest


GAUSSIANS = ['gaussian25_full', 'gaussian50_full', 'gaussian_full', 'gaussian200_full']
METHODS = GAUSSIANS+['resolvent_full']


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise FileExistsError('preserve completed tradeoff analyses; use a new directory')
    old = Path('results/round4_real_scale_stress')
    old_hashes = {str(path): sha256(path) for path in [
        old/'protocol.json', old/'summary.json', old/'verification.json',
        old/'source/round4_real_scale_stress.py']}
    protocol = dict(kind='post-hoc all-scale fairness extension on previously evaluated real pairs',
        methods=METHODS, gaussian_physical_scale=[.25, .5, 1., 2.],
        input_covariance_factors=FACTORS, scenes=SCENES, seeds=SEEDS,
        intervention='scale both projected physical covariance inputs before the unchanged .25² I model blur; means/RGB/labels/masks fixed',
        reporting='all predeclared scales, clean and all stress factors; no test-best selection or replacement of the prior validation-selected Gaussian',
        primary_comparisons='resolvent minus each Gaussian scale: CE, JS and top1 target mass, paired crossed seed/query-camera bootstrap',
        scope='known-scene sparse observed-track transfer, original GS parity, raw RGB and frozen ResNet18; no realistic geometry-error or untouched-confirmation claim',
        preserved_old_result_sha256=old_hashes, source_sha256=sha256(__file__), config=vars(args))
    write_json(out/'protocol.json', protocol)
    (out/'source').mkdir(exist_ok=True)
    (out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    roots = {kind: Path(f'results/round4_spectral_real_{kind}') for kind in ['raw', 'frozen']}
    branches = {kind: completed_branch(root) for kind, root in roots.items()}
    verification = dict(branches={}, checkpoint_sha256={}, old_stress_replay={})
    started = time.time()
    rows, identities, records = {}, {}, []
    old_summary = json.loads((old/'summary.json').read_text())
    for kind, (manifest, cls, checked) in branches.items():
        verification['branches'][kind] = checked
        fresh = Path('results')/('round3_fresh_real_data' if kind == 'raw' else 'round3_fresh_frozen_visual_data')
        datasets, identities[kind] = {}, []
        for scene in SCENES:
            data, meta, pairs = real_inputs(fresh, scene, 'test')
            datasets[scene] = data
            identities[kind].extend(real_row_identity(pairs, data['label_valid'].numpy(), scene))
            verification['branches'][kind][scene] = dict(dataset_sha256=digest(data), metadata=meta)
        for method in METHODS:
            for seed in SEEDS:
                checkpoint = roots[kind]/'checkpoints'/f'{seed}-{method}.pt'
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
                verification['checkpoint_sha256'][str(checkpoint)] = sha256(checkpoint)
                model = cls(method, payload['features'], payload['resolution']).eval()
                model.load_state_dict(payload['state_dict'], strict=True)
                for factor in FACTORS:
                    combined = {metric: [] for metric in METRICS}
                    for scene, original in datasets.items():
                        data = dict(original, projected_qcov=original['projected_qcov']*factor,
                                    projected_kcov=original['projected_kcov']*factor)
                        probability = model_probability(model, data)
                        summary, per = real_probability_metrics(probability, data)
                        valid = data['label_valid'].numpy()
                        for metric in METRICS:
                            combined[metric].append(per[metric][valid])
                        records.append(dict(kind=kind, method=method, seed=seed,
                                            factor=factor, scene=scene, summary=summary))
                    for metric in METRICS:
                        rows.setdefault((kind, method, factor, metric), []).append(np.concatenate(combined[metric]))
            print('REPLAYED_ALL_SCALE', kind, method, flush=True)
    rows = {key: np.stack(value) for key, value in rows.items()}
    means, comparisons, clean_stress_changes = {}, {}, {}
    for kind in ['raw', 'frozen']:
        for factor in FACTORS:
            for method in METHODS:
                key = f'{kind}/{factor}/{method}'
                means[key] = {metric: float(rows[(kind, method, factor, metric)].mean()) for metric in METRICS}
                if key in old_summary['means']:
                    difference = max(abs(means[key][metric]-old_summary['means'][key][metric]) for metric in METRICS)
                    if difference != 0.:
                        raise ValueError('overlapping old stress result did not replay exactly')
                    verification['old_stress_replay'][key] = difference
                if factor != 1.:
                    clean_stress_changes[key] = {metric: paired_camera_seed_statistics(
                        rows[(kind, method, factor, metric)], rows[(kind, method, 1., metric)], identities[kind])
                        for metric in METRICS}
            for control in GAUSSIANS:
                comparisons[f'{kind}/{factor}/resolvent_full_minus_{control}'] = {
                    metric: paired_camera_seed_statistics(rows[(kind, 'resolvent_full', factor, metric)],
                        rows[(kind, control, factor, metric)], identities[kind]) for metric in METRICS}
    for path, checksum in {**old_hashes, **verification['checkpoint_sha256']}.items():
        if sha256(path) != checksum:
            raise ValueError(f'preserved input changed: {path}')
    output = dict(scope=protocol['kind'], means=means, resolvent_minus_gaussian=comparisons,
                  stress_minus_clean=clean_stress_changes, records=records,
                  elapsed_seconds=time.time()-started)
    write_json(out/'summary.json', output)
    write_json(out/'verification.json', verification)
    np.savez_compressed(out/'row_metrics.npz', **{
        f'{kind}/{method}/{factor}/{metric}': value for (kind, method, factor, metric), value in rows.items()})
    lines = ['# Covariance-filter scale tradeoff: all Gaussian controls', '',
        protocol['kind'], '',
        'All four physical Gaussian scales are reported, with the same frozen model seeds, visual inputs, '
        'query/key means, labels and candidate masks. No evaluation result selects a preferred Gaussian scale. '
        'Both covariance inputs are scaled before the unchanged resolution floor. This is a controlled '
        'corruption, not a measured real estimator-error distribution.', '',
        '| Visual input | Input covariance multiplier | Gaussian×0.25 CE | Gaussian×0.5 CE | Gaussian×1 CE | Gaussian×2 CE | Resolvent CE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for kind in ['raw', 'frozen']:
        for factor in FACTORS:
            entries = [f'{means[f"{kind}/{factor}/{method}"]["cross_entropy"]:.6f}' for method in METHODS]
            lines.append(f'| {kind} | {factor} | '+' | '.join(entries)+' |')
    lines += ['', 'summary.json contains CE/JS/Top1 target mass means, all resolvent-minus-Gaussian '
        'paired seed/query-camera confidence intervals, clean-to-stress changes and every scene/seed value. '
        'Negative CE differences favor Resolvent. Prior completed stress results and their source hashes '
        'are preserved; overlapping means replay exactly. Tradeoff interpretations must consider clean '
        'performance and all stress factors rather than selecting whichever control loses on one condition.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE', output['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/round4_scale_tradeoff')
    main(parser.parse_args())
