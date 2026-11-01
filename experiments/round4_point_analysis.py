"""Replay and paired analysis of the prospectively fixed fresh point experiment."""
import json
from pathlib import Path
import numpy as np
import torch

from experiments import rendered_data
from experiments.rendered_probe import concatenate
from experiments.rendered_metrics import metrics
from experiments.matching_probe import digest
from experiments.rgb_geometry_probe import RGBGeometry, observation_from_rgb
from experiments.round4_spectral_probe import factory
from experiments.isometric_confirmation_analysis import (
    SEEDS, sha256, write_json, compare_summary, paired_point_statistics,
    point_probability, load_checkpoint, checkpoint_hash_final_check)


ROOT = Path('results/round4_spectral_point')
OUT = Path('results/round4_point_analysis')
REGIMES = ['oracle_train', 'robust_train']
SCALES = ['gaussian25_full', 'gaussian50_full', 'gaussian_full', 'gaussian200_full']
KEYS = ['q', 'k', 'mq', 'mk', 'pixq', 'k_to_q', 'projected_kcov', 'labels', 'pixel_error']


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest_path = ROOT/'manifest.json'
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if 'elapsed_seconds' not in manifest:
        raise RuntimeError('Training must finish before analysis')
    config = manifest['config']
    protocol_path = ROOT/'frozen_protocol.json'
    protocol_hash = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    assert protocol['config'] == config
    assert protocol['namespace_offset'] == 10000
    methods = config['methods'].split(',')
    expected = {(r, s, m) for r in REGIMES for s in SEEDS for m in methods}
    records = manifest['records']
    actual = {(r['regime'], r['seed'], r['method']) for r in records}
    assert len(records) == len(expected) == len(actual) and actual == expected
    history = json.loads((ROOT/'training_history.json').read_text())
    used_sources = ['models/gaussian_spectral_pe.py', 'models/gaussian_pe.py', 'models/geometry.py',
        'experiments/round4_spectral_probe.py', 'experiments/gaussian_pe_probe.py',
        'experiments/rendered_data.py', 'experiments/rendered_scene.py',
        'experiments/rendered_metrics.py', 'experiments/rgb_geometry_probe.py',
        'experiments/matching_probe.py', 'experiments/rendered_probe.py',
        'experiments/isometric_confirmation_analysis.py']
    for path in used_sources:
        assert sha256(path) == manifest['source_hashes'][path]
        assert sha256(ROOT/'source'/path) == manifest['source_hashes'][path]
        assert manifest['source_hashes'][path] == protocol['source_hashes'][path]
    for r in records:
        rows = [h for h in history if (h['regime'], h['seed'], h['method']) ==
                 (r['regime'], r['seed'], r['method'])]
        assert [h['step'] for h in rows] == [100, 200, 300, 400, 500, 600]
        best = min(rows, key=lambda h: h['validation']['loss'])
        assert best['step'] == r['step']
        compare_summary(best['validation'], r['best_validation'])
    selected = {regime: min(SCALES, key=lambda method: np.mean([
        r['best_validation']['loss'] for r in records
        if r['regime'] == regime and r['method'] == method])) for regime in REGIMES}
    OUT.mkdir(parents=True, exist_ok=True)
    # Persist the validation-only family choice before opening any replayed
    # test dataset. This is provenance, not a change to the selection rule.
    selection = dict(training_manifest_sha256=manifest_hash, protocol_sha256=protocol_hash,
        criterion='minimum mean best-validation loss across the five training seeds, separately by regime',
        selected=selected, validation_losses={regime: {method: float(np.mean([
            r['best_validation']['loss'] for r in records
            if r['regime'] == regime and r['method'] == method])) for method in SCALES}
            for regime in REGIMES}, test_values_used=False)
    selection_path = OUT/'selected_gaussian_by_validation.json'
    if selection_path.exists():
        assert json.loads(selection_path.read_text()) == selection
    else:
        write_json(selection_path, selection)
    selection_hash = sha256(selection_path)
    archive = OUT/'predictions'
    archive.mkdir(exist_ok=True)
    predictor_path = Path('results/round2_rgb_geometry/checkpoints/geometry-11.pt')
    assert sha256(predictor_path) == manifest['predictor_sha256']
    predictor = RGBGeometry()
    predictor.load_state_dict(torch.load(predictor_path, map_location='cpu', weights_only=True)['state_dict'])
    predictor.eval()
    replay_errors, hashes, dataset_hashes, measurements, means = [], {}, {}, {}, {}
    archive_hashes = {}
    for seed in SEEDS:
        cache = {}
        train_parts = [rendered_data.make_dataset(12100+seed, 192, c, cache)[0]
                       for c in ['oracle', 'mixed', 'depth_fit']]
        val_parts = [rendered_data.make_dataset(12200+seed, 48, c, cache)[0]
                     for c in ['oracle', 'mixed', 'depth_fit']]
        tests = {c: rendered_data.make_dataset(12300+seed, 96, c, cache)[0]
                 for c in rendered_data.CONDITIONS}
        original = rendered_data.patch_observation
        try:
            rendered_data.patch_observation = lambda view, cx, cy, rng, c: observation_from_rgb(
                predictor, view, cx, cy, rng, c)
            tests['rgb_pred'] = rendered_data.make_dataset(12300+seed, 96, 'rgb_pred', cache)[0]
        finally:
            rendered_data.patch_observation = original
        del cache
        for regime in REGIMES:
            tr = train_parts[0] if regime == 'oracle_train' else concatenate(train_parts)
            va = val_parts[0] if regime == 'oracle_train' else concatenate(val_parts)
            ds = dict(train=digest(tr), validation=digest(va), test={c: digest(t) for c, t in tests.items()})
            assert ds == manifest['datasets'][f'{regime}/{seed}']
            dataset_hashes[f'{regime}/{seed}'] = ds
            va = {k: va[k] for k in KEYS}
            for method in methods:
                r = next(r for r in records if (r['regime'], r['seed'], r['method']) == (regime, seed, method))
                model = load_checkpoint(ROOT, r, config, hashes, factory, point=True)
                pv, lv = point_probability(model, va)
                mv, _ = metrics(pv, va['labels'].numpy(), va['pixel_error'].numpy())
                mv['loss'] = float(lv.mean())
                replay_errors.extend(compare_summary(mv, r['best_validation']).values())
                for condition, raw in tests.items():
                    data = {k: raw[k] for k in KEYS}
                    p, loss = point_probability(model, data)
                    label = data['labels'].numpy()
                    summary, _ = metrics(p, label, data['pixel_error'].numpy())
                    summary['loss'] = float(loss.mean())
                    replay_errors.extend(compare_summary(summary, r['test'][condition]).values())
                    key = f'{regime}/{condition}/{method}'
                    means.setdefault(key, {})[str(seed)] = summary
                    visible = label < p.shape[-1]-1
                    pred = p.argmax(-1)
                    n = p.shape[-1]-1
                    scene = dict(visible_r1=((pred == label)&visible).sum(-1),
                        closed_visible_r1=((p[..., :-1].argmax(-1) == label)&visible).sum(-1),
                        false_match_rate=((pred < n)&~visible).sum(-1),
                        loss=loss.sum(-1), visible=visible.sum(-1), unmatched=(~visible).sum(-1),
                        count=np.full(len(label), label.shape[1]))
                    measurements[f'{seed}/{key}'] = scene
                    if condition in ['oracle', 'mixed', 'scale_noise', 'rgb_pred', 'depth_fit']:
                        target = archive/f'{regime}-{seed}-{method}-{condition}.npz'
                        np.savez_compressed(target, probability=p, labels=label, loss=loss)
                        archive_hashes[str(target)] = sha256(target)
                print('REPLAYED', regime, seed, method, flush=True)
        del train_parts, val_parts, tests
    pairs = [('resolvent_full', m) for m in methods if m != 'resolvent_full']
    pairs += [('gaussian_full', 'gaussian_none'), ('gaussian_full', 'projected_rope')]
    differences = {}
    for regime in REGIMES:
        for condition in records[0]['test']:
            for a, b in pairs:
                for metric, denominator in [('visible_r1', 'visible'), ('closed_visible_r1', 'visible'),
                                             ('false_match_rate', 'unmatched'), ('loss', 'count')]:
                    aa = [measurements[f'{s}/{regime}/{condition}/{a}'] for s in SEEDS]
                    bb = [measurements[f'{s}/{regime}/{condition}/{b}'] for s in SEEDS]
                    for x, y in zip(aa, bb):
                        np.testing.assert_array_equal(x[denominator], y[denominator])
                    numer = np.stack([x[metric]-y[metric] for x, y in zip(aa, bb)]).astype(np.float64)
                    denom = np.stack([x[denominator] for x in aa])
                    differences[f'{regime}/{condition}/{a}_minus_{b}/{metric}'] = paired_point_statistics(numer, denom)
    aggregate = {k: {metric: float(np.mean([s[metric] for s in by_seed.values()]))
                     for metric in next(iter(by_seed.values())) if next(iter(by_seed.values()))[metric] is not None}
                 for k, by_seed in means.items()}
    for k in aggregate:
        for metric, denom in [('closed_visible_r1', 'visible')]:
            aggregate[k][metric] = float(np.mean([measurements[f'{s}/{k}'][metric].sum()/measurements[f'{s}/{k}'][denom].sum() for s in SEEDS]))
    checkpoint_hash_final_check(hashes)
    assert sha256(manifest_path) == manifest_hash
    assert sha256(protocol_path) == protocol_hash
    assert sha256(selection_path) == selection_hash
    for path in used_sources:
        assert sha256(path) == manifest['source_hashes'][path]
    np.savez_compressed(OUT/'scene_counts.npz', **{f'{k}/{m}': v for k, d in measurements.items() for m, v in d.items()})
    result = dict(training_manifest_sha256=manifest_hash, protocol_sha256=protocol_hash,
        selection_sha256=selection_hash,
        selected_gaussian_by_validation=selected, means=aggregate, per_seed=means,
        differences=differences, maximum_replay_error=max(replay_errors), checkpoint_sha256=hashes,
        prediction_sha256=archive_hashes, dataset_sha256=dataset_hashes,
        scope='Prospectively fixed new procedural datasets. Real-data evaluation is a separate post-hoc extension; no real rejection claim.')
    write_json(OUT/'summary.json', result)
    lines = ['# Round 4 fresh point validation', '',
        'Five independent training/data seeds, 70 models, both oracle and mixed-geometry training. '
        'All train/validation/test dataset digests and all validation/test metrics replayed; '
        'checkpoint and prediction hashes saved. New procedural namespaces12100/12200/12300.', '',
        f'Validation-selected Gaussian scales: {selected}.', '',
        '| Train | Condition | Method | Visible R1 | Closed R1 | NLL | ECE | FMR |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for regime in REGIMES:
        for condition in ['oracle', 'mixed', 'scale_noise', 'rgb_pred']:
            for method in methods:
                a = aggregate[f'{regime}/{condition}/{method}']
                lines.append(f"| {regime} | {condition} | {method} | {a['visible_r1']:.5f} | {a['closed_visible_r1']:.5f} | {a['loss']:.5f} | {a['ece']:.5f} | {a['false_match_rate']:.5f} |")
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE', max(replay_errors), selected, flush=True)


if __name__ == '__main__':
    main()
