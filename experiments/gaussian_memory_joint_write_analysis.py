"""Paired analysis and bounded verification of the fixed shared-write experiment."""
import argparse
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_coverage_report import estimate
from experiments.gaussian_memory_joint_write import OUT, SEEDS, NEW, KINDS, checked, model_path, make_models, previous

METRICS = ('log_mse', 'hidden_log_mse', 'visible_log_mse', 'rmse_m', 'absrel',
           'full_log_mse', 'bias_squared', 'centered_mse', 'full_bias_squared', 'full_centered_mse')
PAIRS = (('gaussian', 'full_gaussian'), ('content_gaussian', 'full_gaussian'),
         ('full_center', 'full_gaussian'), ('large_local', 'full_gaussian'),
         ('global', 'full_gaussian'), ('gaussian', 'content_gaussian'),
         ('large_local', 'content_gaussian'), ('center', 'full_center'),
         ('large_local', 'full_center'))


def matrix(rows, kind, metric, scenes, condition='normal'):
    return np.array([[np.mean([r[metric] for r in rows if r['kind'] == kind and r['seed'] == s
                               and r['scene'] == scene and r.get('condition', 'normal') == condition])
                      for s in SEEDS] for scene in scenes])


def selected(row, split):
    if split == 'primary32':
        return row['split'] in ('used_clip40', 'used_clip80', 'used_summary20')
    if split == 'old12':
        return row['split'] in ('used_clip40', 'used_clip80')
    return row['split'] == split


def analysis():
    checked(True); data = json.loads((OUT / 'evaluation/scores.json').read_text()); stats = {}
    splits = ['primary32', 'old12'] + sorted({r['split'] for r in data['rows']})
    for split in splits:
        rows = [r for r in data['rows'] if selected(r, split)]
        full = [r for r in data['full_query'] if selected(r, split)]
        updates = [r for r in data['updates'] if selected(r, split)]
        scenes = sorted({r['scene'] for r in rows}); summary = {k: {} for k in KINDS}; comparisons = {}
        for metric in METRICS:
            source, key = (full, metric.removeprefix('full_')) if metric.startswith('full_') else (rows, metric)
            values = {k: matrix(source, k, key, scenes) for k in KINDS}
            assert all(np.isfinite(v).all() for v in values.values())
            for kind, value in values.items():
                summary[kind][metric] = float(value.mean())
            comparisons[metric] = {}
            for left, right in PAIRS:
                effect = estimate(values[left] - values[right], scenes)
                effect['relative_reduction_percent'] = 100 * effect['mean_gain'] / values[left].mean()
                comparisons[metric][left + '_minus_' + right] = effect
        interventions = {}; changes = {}
        for kind in NEW:
            normal = matrix(rows, kind, 'log_mse', scenes); interventions[kind] = {}
            for condition in ('removed', 'shuffled', 'wrong_camera', 'initial_geometry', 'initial_memory'):
                effect = estimate(matrix(rows, kind, 'log_mse', scenes, condition) - normal, scenes)
                effect['relative_error_increase_percent'] = 100 * effect['mean_gain'] / normal.mean()
                interventions[kind][condition] = effect
            changes[kind] = {key: float(matrix(updates, kind, key, scenes).mean()) for key in
                             ('center_displacement', 'center_displacement_m', 'covariance_relative_change',
                              'opacity_abs_change', 'feature_delta_rms')}
        stats[split] = {'summary': summary, 'comparisons': comparisons, 'interventions': interventions,
                        'updates': changes, 'scenes': scenes, 'episodes': len({r['episode'] for r in rows}),
                        'confirmation_claim': False}
    assert len(stats['primary32']['scenes']) == 32
    write_json(OUT / 'analysis.json', stats)
    print(json.dumps({s: {'means': {k: v['log_mse'] for k, v in stats[s]['summary'].items()},
                          'comparisons': {k: {f: v[f] for f in ('relative_reduction_percent', 'paired_scene_seed_bootstrap_95',
                                                               'paired_seed_gain', 'scenes_better', 'scenes_total')}
                                          for k, v in stats[s]['comparisons']['log_mse'].items()}}
                       for s in ('primary32', 'old12', 'used_summary20')}), flush=True)


def checks():
    plan = checked(True); torch.set_num_threads(4); fit_checks = []
    for seed in SEEDS:
        old_rows = [json.loads(s) for s in (previous.path('gaussian', seed).parent / 'training.jsonl').read_text().splitlines()]
        common_writer = None
        for kind in NEW:
            directory = model_path(kind, seed).parent
            status = json.loads((directory / 'status.json').read_text()); assert status['complete'] and status['steps'] == 2000
            rows = [json.loads(s) for s in (directory / 'training.jsonl').read_text().splitlines()]
            assert len(rows) == len(old_rows) == 2000
            assert all(all(a[f] == b[f] for f in ('step', 'slot', 'episode', 'sample_ids')) for a, b in zip(rows, old_rows))
            assert all(np.isfinite([r['loss'], r['gradient_norm']]).all() for r in rows)
            initial = torch.load(model_path(kind, seed, True), map_location='cpu', weights_only=False)
            old_initial = torch.load(previous.path(kind.split('_')[-1], seed, True), map_location='cpu', weights_only=False)['model']
            assert all(torch.equal(v, old_initial[k]) for k, v in initial['model'].items())
            if common_writer is None:
                common_writer = initial['writer']
            assert all(torch.equal(v, common_writer[k]) for k, v in initial['writer'].items())
            assert status['writer_gradients'][-1]['content'] > 0
            if kind.startswith('full'):
                assert all(status['writer_gradients'][-1][k] > 0 for k in ('center', 'shape', 'opacity'))
            else:
                assert all(status['writer_gradients'][-1][k] == 0 for k in ('center', 'shape', 'opacity'))
            fit_checks.append({'kind': kind, 'seed': seed, 'sampling_exact': True, 'initialization_exact': True,
                               'all_finite': True, 'nonzero_writer_gradients': True})
    scores = json.loads((OUT / 'evaluation/scores.json').read_text())
    old = json.loads((previous.OUT / 'evaluation/scores.json').read_text())
    assert sha256(previous.OUT / 'evaluation/scores.json') == plan['previous_scores_sha256']
    assert len(scores['rows']) == 66 * 69 and len(scores['full_query']) == 66 * 24
    assert len(scores['updates']) == len(scores['memory_sha256']) == 66 * 9
    for field in ('maps_sha256', 'memory_sha256'):
        for filename, digest in scores[field].items():
            assert sha256(filename) == digest, filename
    replay = []
    for field in ('rows', 'full_query'):
        index = {(r['episode'], r['kind'], r['seed']): r for r in old[field] if r['condition'] == 'normal'}
        for r in scores[field]:
            assert np.isfinite(r['log_mse'])
            if r['kind'] in NEW:
                continue
            before = index[(r['episode'], r['kind'], r['seed'])]
            for metric in ('log_mse', 'hidden_log_mse', 'visible_log_mse', 'rmse_m', 'absrel'):
                if before[metric] is not None:
                    replay.append(abs(r[metric] - before[metric]))
    assert max(replay) < 2e-5
    train_scenes = {m['row']['scene'].split('/')[0] for m in plan['catalog'] if m['id'] in plan['training_ids']}
    primary_scenes = {r['scene'].split('/')[0] for r in scores['rows'] if selected(r, 'primary32')}
    assert len(primary_scenes) == 32 and not train_scenes & primary_scenes
    reads = []; numpy_scores = []; seed = SEEDS[0]
    with torch.inference_mode():
        for rid in ('fit_00', 'confirm_00', 'confirm_15'):
            meta = next(m for m in plan['catalog'] if m['id'] == rid)
            d = previous.load_record(meta); raw, rec = d['raw'], d['rec']
            saved = torch.load(OUT / 'evaluation' / (rid + '_maps.pt'), map_location='cpu', weights_only=False)
            keep = torch.tensor([r == 'evaluation' for r in rec['roles']]); q = rec['query'][keep]
            for kind in NEW:
                # The writer is deliberately not invoked: only the serialized state is read.
                net = previous.model(kind.split('_')[-1], plan).eval()
                net.load_state_dict(torch.load(model_path(kind, seed), map_location='cpu', weights_only=False)['model'])
                memory = torch.load(OUT / 'memories' / f'{rid}_{kind}_{seed}.pt', map_location='cpu', weights_only=False)['memory']
                p = net.read(rec['input_tokens'][keep], memory, raw['c2w'][q], raw['K'][q], raw['diagnostics']['unit_meters'])
                stored = saved['predictions'][f'{kind}_{seed}_normal']; delta = float((p - stored).abs().max())
                assert delta < 2e-5
                reads.append({'episode': rid, 'kind': kind, 'seed': seed, 'max_difference': delta})
                error = stored.numpy().astype('float64') - saved['target'].numpy().astype('float64'); valid = saved['valid'].numpy()
                independent = float(np.mean([np.mean(e[v] ** 2) for e, v in zip(error, valid)]))
                row = next(r for r in scores['rows'] if r['episode'] == rid and r['kind'] == kind and r['seed'] == seed and r['condition'] == 'normal')
                diff = abs(independent - row['log_mse']); assert diff < 2e-6
                numpy_scores.append({'episode': rid, 'kind': kind, 'difference': diff})
    accounting = max(r['accounting_max_diff'] for r in scores['rows'] + scores['full_query'] if r['condition'] == 'normal')
    assert accounting < 1e-10
    result = {'complete': True, 'fit_checks': fit_checks, 'old_models_unchanged': 15,
              'old_metric_replay_max_difference': max(replay), 'old_metric_values_checked': len(replay),
              'saved_memories': len(scores['updates']), 'saved_roundtrip_max_difference': 0.,
              'minimum_covariance_eigenvalue': min(r['minimum_eigenvalue'] for r in scores['updates']),
              'independent_CPU_reads': reads, 'independent_numpy_scores': numpy_scores,
              'decomposition_accounting_max_difference': accounting, 'training_primary_places_disjoint': True,
              'queries': sum(r['queries'] for r in scores['counts']),
              'masked_inputs': sum(r['masked_inputs'] for r in scores['counts']),
              'confirmation_claim': False, 'source_sha256': sha256(__file__)}
    write_json(OUT / 'checks.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('fit_checks', 'independent_CPU_reads', 'independent_numpy_scores')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('analysis', 'checks'))
    globals()[parser.parse_args().stage]()
