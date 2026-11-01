"""Post-hoc physical-covariance scaling stress on already evaluated real pairs.

This is a controlled input corruption, not a real estimator-error distribution.
Means/RGB/labels and the resolution floor stay fixed; both physical covariance
inputs are multiplied by the same predefined factor. No model is retrained.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.round4_real_analysis import completed_branch, real_inputs
from experiments.isometric_confirmation_analysis import (
    SEEDS, model_probability, real_probability_metrics, real_row_identity,
    paired_camera_seed_statistics, sha256, write_json)
from experiments.matching_probe import digest


FACTORS = [.25, 1., 4., 16.]
SCENES = ['figurines', 'temple']
METRICS = ['cross_entropy', 'js', 'top1_target_mass']


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise FileExistsError('use a new stress output directory')
    protocol = dict(kind='post-hoc frozen input stress, not prospective confirmation or a realistic prediction-error guarantee',
        factors=FACTORS, intervention='multiply projected_qcov and projected_kcov before the unchanged model resolution floor .25² I; all means/RGB/labels/masks fixed',
        methods=['gaussian_full', 'resolvent_full', 'validation_selected_gaussian'],
        selection='same five-seed aggregate old Teatime validation selection as main Round4 real analysis, no stress selection',
        data='previously evaluated figurines+temple, original GS parity, raw RGB and frozen ResNet18, five paired model seeds',
        stats='paired seed/query-camera bootstrap within fixed scenes',
        main_analysis=args.main_analysis, source_sha256=sha256(__file__), config=vars(args))
    write_json(out/'protocol.json', protocol)
    deadline = time.monotonic()+args.wait_seconds
    main_summary = Path(args.main_analysis)/'summary.json'
    while not main_summary.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('main real analysis must complete before real stress')
        print('WAITING_FOR_MAIN_REAL_ANALYSIS', flush=True)
        time.sleep(min(10., max(0., deadline-time.monotonic())))
    started = time.time()
    main_result = json.loads(main_summary.read_text())
    roots = {kind: Path(f'results/round4_spectral_real_{kind}') for kind in ['raw', 'frozen']}
    # Both complete branches and validation-only selections precede stress data.
    branches = {kind: completed_branch(root) for kind, root in roots.items()}
    verification = dict(main_analysis_summary_sha256=sha256(main_summary), branches={}, checkpoint_sha256={})
    rows, identities, records = {}, {}, []
    for kind, (manifest, cls, checked) in branches.items():
        selected = checked['scale_selection']['selected']
        if selected != main_result['selected_gaussian_scale'][kind]['selected']:
            raise ValueError('main and stress validation selection differ')
        verification['branches'][kind] = checked
        fresh = Path('results')/('round3_fresh_real_data' if kind == 'raw' else 'round3_fresh_frozen_visual_data')
        datasets = {}
        identities[kind] = []
        for scene in SCENES:
            data, meta, pairs = real_inputs(fresh, scene, 'test')
            datasets[scene] = data
            identities[kind].extend(real_row_identity(pairs, data['label_valid'].numpy(), scene))
            verification['branches'][kind][scene] = dict(dataset_sha256=digest(data), metadata=meta)
        for method in list(dict.fromkeys(['gaussian_full', 'resolvent_full', selected])):
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
                        records.append(dict(kind=kind, method=method, seed=seed, factor=factor, scene=scene, summary=summary))
                    for metric in METRICS:
                        key = (kind, method, factor, metric)
                        rows.setdefault(key, []).append(np.concatenate(combined[metric]))
            print('STRESSED', kind, method, flush=True)
    rows = {key: np.stack(value) for key, value in rows.items()}
    means, comparisons = {}, {}
    for kind in ['raw', 'frozen']:
        selected = verification['branches'][kind]['scale_selection']['selected']
        for factor in FACTORS:
            for method in list(dict.fromkeys(['gaussian_full', 'resolvent_full', selected])):
                means[f'{kind}/{factor}/{method}'] = {metric: float(rows[(kind, method, factor, metric)].mean()) for metric in METRICS}
            for control in list(dict.fromkeys(['gaussian_full', selected])):
                key = f'{kind}/{factor}/resolvent_full_minus_{control}'
                comparisons[key] = {metric: paired_camera_seed_statistics(rows[(kind, 'resolvent_full', factor, metric)],
                    rows[(kind, control, factor, metric)], identities[kind]) for metric in METRICS}
    for path, checksum in verification['checkpoint_sha256'].items():
        if sha256(path) != checksum:
            raise ValueError('checkpoint changed during stress')
    output = dict(scope=protocol['kind'], means=means, comparisons=comparisons,
                  records=records, elapsed_seconds=time.time()-started)
    write_json(out/'summary.json', output)
    write_json(out/'verification.json', verification)
    np.savez_compressed(out/'row_metrics.npz', **{
        f'{kind}/{method}/{factor}/{metric}': value for (kind, method, factor, metric), value in rows.items()})
    lines = ['# Real covariance scaling stress', '', protocol['kind'], '',
        'Both covariance inputs are multiplied before the unchanged blur floor. No retraining, '
        'temperature fitting, test-scale choice, or mean perturbation. These corruptions are controlled '
        'probes and do not model the full error of predicted geometry.', '',
        '| Visual input | Physical covariance multiplier | Method | CE | JS | Top1 target mass |',
        '|---|---:|---|---:|---:|---:|']
    for key, value in means.items():
        kind, factor, method = key.split('/')
        lines.append(f"| {kind} | {factor} | {method} | {value['cross_entropy']:.6f} | {value['js']:.6f} | {value['top1_target_mass']:.6f} |")
    lines += ['', 'See summary.json for paired camera/seed differences and all per-scene/seed values. '
        'The validation-selected Gaussian is fixed across stress factors. This descriptive probe alone '
        'cannot establish general robustness or a new deployment improvement.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE', output['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/round4_real_scale_stress')
    parser.add_argument('--main-analysis', default='results/round4_spectral_real_analysis')
    parser.add_argument('--wait-seconds', type=int, default=0)
    main(parser.parse_args())
