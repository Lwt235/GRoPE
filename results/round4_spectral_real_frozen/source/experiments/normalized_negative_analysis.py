"""Fixed point-correspondence/rejection negative controls for compact PE."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from experiments.isometric_confirmation_analysis import (
    SEEDS, load_completed_manifest, analyse_point, compare_summary, sha256, write_json)


ROOT = Path('results/round3_normalized_point_confirmation')
OUT = Path('results/round3_normalized_point_analysis')


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=True)
    protocol = json.loads((ROOT/'frozen_protocol.json').read_text())
    manifest = load_completed_manifest(ROOT, 'point')
    config = manifest['config']
    for name in ['resolution','features','steps','val_every','batch','train_scenes','val_scenes','test_scenes']:
        if config[name] != protocol[name]:
            raise ValueError(f'negative-control setting changed: {name}')
    if config['methods'].split(',') != protocol['methods']:
        raise ValueError('method list differs from frozen negative controls')
    history = json.loads((ROOT/'training_history.json').read_text())
    for r in manifest['records']:
        candidates = [h for h in history if (h['seed'],h['regime'],h['method']) ==
                      (r['seed'],r['regime'],r['method'])]
        if [h['step'] for h in candidates] != [100,200,300,400,500,600]:
            raise ValueError('incomplete negative-control validation history')
        best = min(candidates,key=lambda h:h['validation']['loss'])
        if best['step'] != r['step']:
            raise ValueError('checkpoint did not use minimum validation NLL')
        compare_summary(best['validation'],r['best_validation'])
    replayed_conditions = ['oracle','mixed','depth_fit','scale_noise','rgb_pred']
    existing = OUT/'point_summary.json'
    if existing.exists():
        result = json.loads(existing.read_text())
        if result['training_manifest_sha256'] != manifest['_analysis_input_sha256']:
            raise ValueError('cached point replay belongs to a different completed run')
        for path, expected in result['checkpoint_sha256'].items():
            if sha256(path) != expected:
                raise ValueError('checkpoint changed after existing point replay')
    else:
        result = analyse_point(SimpleNamespace(point_run=ROOT,out=OUT,bootstrap=4000),
            manifest,candidate='normalized_shape', conditions_to_replay=replayed_conditions)
    metrics = ['visible_r1','loss','unmatched_auprc','ece',
               'false_match_rate','brier_unmatched']
    means = {}
    for regime in protocol['regimes']:
        for condition in manifest['records'][0]['test']:
            for method in protocol['methods']:
                records = [r for r in manifest['records'] if r['regime']==regime and r['method']==method]
                means[f'{regime}/{condition}/{method}'] = {metric:dict(
                    mean=float(np.mean([r['test'][condition][metric] for r in records])),
                    per_seed={str(r['seed']):r['test'][condition][metric] for r in records})
                    for metric in metrics}
                if condition in replayed_conditions:
                    closed = {}
                    for seed in SEEDS:
                        path = OUT/'predictions'/'point'/f'{regime}-{seed}-{method}-{condition}.npz'
                        with np.load(path) as archive:
                            visible = archive['visible']
                            closed[str(seed)] = float(archive['closed_correct'][visible].mean())
                    means[f'{regime}/{condition}/{method}']['closed_visible_r1'] = dict(
                        mean=float(np.mean(list(closed.values()))), per_seed=closed)
    write_json(OUT/'all_condition_means.json',means)
    lines = ['# Compact Gaussian PE: point/rejection negative controls', '',
        'All 50 fixed runs completed. These are repeats on previously examined synthetic namespaces, '
        'not new-data discovery evidence. The soft real-region task and this hard point-label task '
        'have different target semantics. Ordinary gradients must remain finite; no post-failure tuning is allowed.', '',
        '## Absolute means', '',
        '| Training | Condition | Method | Visible R1 | Closed R1 | NLL | Unmatched AP | ECE | FMR |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for regime in protocol['regimes']:
        for condition in ['oracle','mixed','scale_noise','rgb_pred']:
            for method in protocol['methods']:
                m=means[f'{regime}/{condition}/{method}']
                lines.append(f"| {regime} | {condition} | {method} | {m['visible_r1']['mean']:.4f} | {m['closed_visible_r1']['mean']:.4f} | {m['loss']['mean']:.4f} | {m['unmatched_auprc']['mean']:.4f} | {m['ece']['mean']:.4f} | {m['false_match_rate']['mean']:.4f} |")
    lines += ['', '## Paired differences', '',
        '| Training | Condition | Control | Full-minus-control visible R1 | 95% seed-scene CI |',
        '|---|---|---|---:|---|']
    for regime in protocol['regimes']:
        for condition in ['oracle','mixed','depth_fit','scale_noise','rgb_pred']:
            for control in ['normalized_fixed','expected_shape','projected_rope','expected_fixed']:
                r=result['differences'][f'{regime}/{condition}/normalized_shape_minus_{control}/visible_r1']
                lines.append(f"| {regime} | {condition} | {control} | {r['mean']:+.6f} | {r['ci95_seed_scene']} |")
    lines += ['', 'All ten conditions, every seed, proper losses and calibration metrics remain in '
        'all_condition_means.json. The five principal conditions are independently replayed from '
        'archived model code, checked against saved test metrics and dataset digests; probability '
        'arrays and paired scene-bootstrap results are in point_summary.json. '
        'AP/ECE are summarized by seed rather than treating individual token rows as independent. '
        'This does not establish rejection calibration on real images, for which absent-track labels '
        'are not valid occlusion ground truth.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    write_json(OUT/'analysis_source_hashes.json', {
        __file__:sha256(__file__),
        'experiments/isometric_confirmation_analysis.py':sha256('experiments/isometric_confirmation_analysis.py')})
    print('COMPLETE normalized negative controls')


if __name__ == '__main__':
    main()
