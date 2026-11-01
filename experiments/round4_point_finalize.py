"""Independent metric arithmetic from completed point prediction archives.

Inference/dataset/validation replay completed before an integer-division error
in the initial bootstrap reporting call. This finalizer avoids repeating model
inference: it verifies the complete replay log and recomputes all five principal
conditions from saved probabilities. Secondary conditions retain trainer means.
"""
import json
from pathlib import Path
import re
import numpy as np

from experiments.rendered_metrics import metrics
from experiments.isometric_confirmation_analysis import (
    SEEDS, sha256, write_json, paired_point_statistics, compare_summary)


ROOT = Path('results/round4_spectral_point')
REPLAY = Path('results/round4_point_verified')
OUT = Path('results/round4_point_final_analysis')
CONDITIONS = ['oracle', 'mixed', 'depth_fit', 'scale_noise', 'rgb_pred']
REGIMES = ['oracle_train', 'robust_train']


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    protocol = json.loads((ROOT/'frozen_protocol.json').read_text())
    assert 'elapsed_seconds' in manifest and protocol['config'] == manifest['config']
    methods = protocol['methods']
    records = manifest['records']
    expected = {(r, s, m) for r in REGIMES for s in SEEDS for m in methods}
    actual = {(r['regime'], r['seed'], r['method']) for r in records}
    assert actual == expected and len(records) == len(expected)
    # Every line is emitted only after train/val/test digest checks, validation
    # prediction checks, and all eight test-condition replay checks for a model.
    logfile = Path('/tmp/grope_round4_point_verified.log')
    log = logfile.read_text()
    replayed = re.findall(r'^REPLAYED (\S+) (\d+) (\S+)$', log, re.M)
    assert len(replayed) == 70
    assert {(r, int(s), m) for r, s, m in replayed} == expected
    assert 'Cannot cast ufunc' in log and "output from dtype('float64') to dtype('int64')" in log
    # Check the model/data code against the pre-training freeze independently.
    source_names = ['models/gaussian_spectral_pe.py', 'models/gaussian_pe.py', 'models/geometry.py',
        'experiments/round4_spectral_probe.py', 'experiments/gaussian_pe_probe.py',
        'experiments/rendered_data.py', 'experiments/rendered_scene.py',
        'experiments/rendered_metrics.py', 'experiments/rgb_geometry_probe.py',
        'experiments/matching_probe.py', 'experiments/isometric_confirmation_analysis.py',
        'experiments/rendered_probe.py']
    for name in source_names:
        assert sha256(name) == manifest['source_hashes'][name] == protocol['source_hashes'][name]
        assert sha256(ROOT/'source'/name) == manifest['source_hashes'][name]
    history = json.loads((ROOT/'training_history.json').read_text())
    for r in records:
        rows = [h for h in history if (h['regime'], h['seed'], h['method']) ==
                (r['regime'], r['seed'], r['method'])]
        assert [h['step'] for h in rows] == [100, 200, 300, 400, 500, 600]
        best = min(rows, key=lambda h: h['validation']['loss'])
        assert best['step'] == r['step']
        compare_summary(best['validation'], r['best_validation'])
    scales = ['gaussian25_full', 'gaussian50_full', 'gaussian_full', 'gaussian200_full']
    selected = {r: min(scales, key=lambda m: np.mean([x['best_validation']['loss']
        for x in records if x['regime'] == r and x['method'] == m])) for r in REGIMES}
    selection = json.loads((REPLAY/'selected_gaussian_by_validation.json').read_text())
    assert selection['selected'] == selected
    arrays, means, errors, hashes = {}, {}, [], {}
    for r in records:
        regime, seed, method = r['regime'], r['seed'], r['method']
        for condition in CONDITIONS:
            path = REPLAY/'predictions'/f'{regime}-{seed}-{method}-{condition}.npz'
            hashes[str(path)] = sha256(path)
            with np.load(path) as saved:
                p, labels, losses = saved['probability'], saved['labels'], saved['loss']
            assert p.shape == (96, 48, 49)
            assert np.isfinite(p).all() and np.allclose(p.sum(-1), 1, atol=2e-6)
            target_probability = np.take_along_axis(p.astype(np.float64), labels[..., None], -1)[..., 0]
            assert (target_probability > 0).all()
            probability_loss = -np.log(target_probability)
            assert np.allclose(probability_loss, losses, atol=3e-6, rtol=3e-6)
            losses = probability_loss
            summary, _ = metrics(p, labels, np.zeros_like(p[..., :-1]))
            summary.pop('accepted_visible_pixel_error')
            summary['loss'] = float(losses.mean())
            exp = {k: v for k, v in r['test'][condition].items() if k != 'accepted_visible_pixel_error'}
            errors.extend(compare_summary(summary, exp).values())
            visible = labels < 48
            pred = p.argmax(-1)
            closed = p[..., :-1].argmax(-1)
            scene = dict(visible_r1=((pred == labels)&visible).sum(-1),
                closed_visible_r1=((closed == labels)&visible).sum(-1),
                false_match_rate=((pred < 48)&~visible).sum(-1), loss=losses.sum(-1),
                visible=visible.sum(-1), unmatched=(~visible).sum(-1), count=np.full(96, 48))
            arrays[f'{regime}/{condition}/{method}/{seed}'] = scene
            summary['closed_visible_r1'] = float(scene['closed_visible_r1'].sum()/visible.sum())
            means.setdefault(f'{regime}/{condition}/{method}', {})[str(seed)] = summary
    pairs = [('resolvent_full', m) for m in methods if m != 'resolvent_full']
    pairs += [('gaussian_full', 'gaussian_none'), ('gaussian_full', 'projected_rope')]
    comparisons = {}
    for regime in REGIMES:
        for condition in CONDITIONS:
            for a, b in pairs:
                aa = [arrays[f'{regime}/{condition}/{a}/{s}'] for s in SEEDS]
                bb = [arrays[f'{regime}/{condition}/{b}/{s}'] for s in SEEDS]
                for metric, den in [('visible_r1', 'visible'), ('closed_visible_r1', 'visible'),
                                    ('false_match_rate', 'unmatched'), ('loss', 'count')]:
                    assert all(np.array_equal(x[den], y[den]) for x, y in zip(aa, bb))
                    numer = np.stack([x[metric]-y[metric] for x, y in zip(aa, bb)]).astype(np.float64)
                    denom = np.stack([x[den] for x in aa]).astype(np.float64)
                    comparisons[f'{regime}/{condition}/{a}_minus_{b}/{metric}'] = paired_point_statistics(numer, denom)
    aggregate = {k: {m: float(np.mean([v[m] for v in byseed.values()]))
                     for m in next(iter(byseed.values()))} for k, byseed in means.items()}
    ckpts = {str(p): sha256(p) for p in sorted((ROOT/'checkpoints').glob('*.pt'))}
    assert len(ckpts) == 70
    OUT.mkdir(exist_ok=True)
    (OUT/'completed_replay.log').write_text(log)
    (OUT/'replay_source.py').write_bytes((REPLAY/'replay_source.py').read_bytes())
    secondary = {f'{r}/{c}/{m}': {metric: float(np.mean([x['test'][c][metric]
         for x in records if x['regime'] == r and x['method'] == m]))
         for metric in ['visible_r1', 'loss', 'ece', 'false_match_rate', 'unmatched_auprc']}
         for r in REGIMES for c in records[0]['test'] for m in methods}
    result = dict(means=aggregate, per_seed=means, comparisons=comparisons,
        all_condition_trainer_means=secondary, selected_gaussian_by_validation=selected,
        maximum_independent_probability_metric_error=max(errors),
        checkpoint_sha256=ckpts, prediction_sha256=hashes,
        manifest_sha256=sha256(ROOT/'manifest.json'), protocol_sha256=sha256(ROOT/'frozen_protocol.json'),
        replay_log_sha256=sha256(OUT/'completed_replay.log'), replay_source_sha256=sha256(OUT/'replay_source.py'),
        selection_sha256=sha256(REPLAY/'selected_gaussian_by_validation.json'),
        finalizer_sha256=sha256(__file__),
        provenance='All70 models completed full dataset, validation and eight-condition checkpoint replay before initial bootstrap integer-output casting failure. This finalizer independently recomputes primary metrics and float64 paired bootstrap from350 saved probability archives; no retraining or checkpoint reselection.',
        inference='Paired training-seed/scene bootstrap on five prospective procedural datasets; primary five conditions. AP/ECE described across seeds, not treated as independent token observations.')
    write_json(OUT/'summary.json', result)
    lines = ['# Round 4 point verification', '', result['provenance'], '',
        f'Validation-selected Gaussian scales: {selected}. Maximum independent metric error: {max(errors):.3g}.', '',
        '| Training | Condition | Method | Visible R1 | Closed R1 | NLL | ECE | FMR |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for regime in REGIMES:
        for condition in ['oracle', 'mixed', 'scale_noise', 'rgb_pred']:
            for method in methods:
                v = aggregate[f'{regime}/{condition}/{method}']
                lines.append(f"| {regime} | {condition} | {method} | {v['visible_r1']:.6f} | {v['closed_visible_r1']:.6f} | {v['loss']:.6f} | {v['ece']:.6f} | {v['false_match_rate']:.6f} |")
    lines += ['', 'The Gaussian quarter/half/full/double scale grid is reported in full. Point queries use known fixed covariance; '
        'the point experiment has no resolvent_none control, so its support contribution is established separately by the symmetric real-region ablations.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE', max(errors), selected, flush=True)


if __name__ == '__main__':
    main()
