"""Development-only paired comparisons and bounded checks for one new control."""
import argparse
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_coverage_report import estimate
from experiments.gaussian_memory_joint_write_analysis import METRICS, matrix
from experiments.gaussian_memory_content_center import (
    OUT, OLD, RECENT, SEEDS, KINDS, NEW, checked, model_path, make_models, load_record, merged_maps)

PAIRS = (('content_center', 'content_gaussian'), ('large_local', 'content_gaussian'),
         ('global', 'content_gaussian'), ('full_gaussian', 'content_gaussian'),
         ('gaussian', 'content_gaussian'), ('center', 'content_center'),
         ('full_center', 'content_center'), ('large_local', 'content_center'),
         ('global', 'content_center'), ('full_center', 'full_gaussian'))


def selected(row, split):
    if split == 'primary51':
        return row['split'] in ('used_clip40', 'used_clip80', 'used_summary20', 'used_writer19')
    if split == 'primary32':
        return row['split'] in ('used_clip40', 'used_clip80', 'used_summary20')
    if split == 'old12':
        return row['split'] in ('used_clip40', 'used_clip80')
    return row['split'] == split


def analysis():
    checked(True); data = json.loads((OUT / 'evaluation/scores.json').read_text()); result = {}
    splits = ['primary51', 'primary32', 'old12'] + sorted({r['split'] for r in data['rows']})
    for split in splits:
        rows = [r for r in data['rows'] if selected(r, split)]
        full = [r for r in data['full_query'] if selected(r, split)]
        scenes = sorted({r['scene'] for r in rows}); summary = {k: {} for k in KINDS}; comparisons = {}
        for metric in METRICS:
            source, key = (full, metric.removeprefix('full_')) if metric.startswith('full_') else (rows, metric)
            values = {kind: matrix(source, kind, key, scenes) for kind in KINDS}
            assert all(np.isfinite(value).all() for value in values.values())
            for kind, value in values.items():
                summary[kind][metric] = float(value.mean())
            comparisons[metric] = {}
            for left, right in PAIRS:
                effect = estimate(values[left] - values[right], scenes)
                effect['relative_reduction_percent'] = 100 * effect['mean_gain'] / values[left].mean()
                comparisons[metric][left + '_minus_' + right] = effect
        interventions = {}
        for kind in ('content_center', 'content_gaussian'):
            normal = matrix(rows, kind, 'log_mse', scenes); interventions[kind] = {}
            for condition in ('removed', 'shuffled', 'wrong_camera', 'initial_geometry', 'initial_memory'):
                effect = estimate(matrix(rows, kind, 'log_mse', scenes, condition) - normal, scenes)
                effect['relative_error_increase_percent'] = 100 * effect['mean_gain'] / normal.mean()
                interventions[kind][condition] = effect
        result[split] = {'summary': summary, 'comparisons': comparisons, 'interventions': interventions,
                         'scenes': scenes, 'episodes': len({r['episode'] for r in rows}),
                         'confirmation_claim': False, 'all_data_previously_used': True}
    assert len(result['primary51']['scenes']) == 51
    write_json(OUT / 'analysis.json', result)
    fields = ('relative_reduction_percent', 'paired_scene_seed_bootstrap_95', 'paired_seed_gain', 'scenes_better', 'scenes_total')
    print(json.dumps({s: {'means': {k: v['log_mse'] for k, v in result[s]['summary'].items()},
        'contrasts': {k: {f: v[f] for f in fields} for k, v in result[s]['comparisons']['log_mse'].items()}}
        for s in ('primary51', 'old12', 'used_summary20', 'used_writer19')}), flush=True)


def checks():
    plan = checked(True); torch.set_num_threads(4); fits = []
    for seed in SEEDS:
        new_dir = model_path('content_center', seed).parent
        old_dir = model_path('content_gaussian', seed).parent
        new_initial = torch.load(new_dir / 'initial.pt', map_location='cpu', weights_only=False)
        old_initial = torch.load(old_dir / 'initial.pt', map_location='cpu', weights_only=False)
        for component in ('model', 'writer'):
            assert all(torch.equal(v, old_initial[component][k]) for k, v in new_initial[component].items())
        rows = [json.loads(s) for s in (new_dir / 'training.jsonl').read_text().splitlines()]
        old_rows = [json.loads(s) for s in (old_dir / 'training.jsonl').read_text().splitlines()]
        assert len(rows) == len(old_rows) == plan['steps'] == 2000
        assert all(all(a[f] == b[f] for f in ('step', 'slot', 'episode', 'sample_ids')) for a, b in zip(rows, old_rows))
        assert all(np.isfinite([r['loss'], r['gradient_norm']]).all() for r in rows)
        status = json.loads((new_dir / 'status.json').read_text()); assert status['complete']
        assert status['writer_gradients'][-1]['content'] > 0
        assert all(g[k] == 0 for g in status['writer_gradients'] for k in ('center', 'shape', 'opacity'))
        fits.append({'seed': seed, 'sampling_exact': True, 'reader_writer_initialization_exact': True,
                     'finite_training': True, 'geometry_gradients_zero': True, 'content_gradient_nonzero': True,
                     'seconds': status['seconds'], 'peak_GiB': status['peak_GiB'],
                     'first100': status['first100'], 'last100': status['last100']})
    data = json.loads((OUT / 'evaluation/scores.json').read_text())
    new = json.loads((OUT / 'evaluation/new_scores.json').read_text())
    assert len(new['rows']) == 85 * 18 and len(new['full_query']) == 85 * 3
    assert len(new['memory_sha256']) == len(new['updates']) == 85 * 3
    assert len(data['rows']) == 85 * 87 and len(data['full_query']) == 85 * 27
    for field in ('maps_sha256', 'memory_sha256'):
        for name, digest in new[field].items():
            assert sha256(name) == digest, name
    for directory in (OLD, RECENT):
        old = json.loads((directory / 'evaluation/scores.json').read_text())
        ids = {r['episode'] for r in old['counts']}
        for field in ('rows', 'full_query', 'updates'):
            expected = [dict(r, split='used_writer19') if directory == RECENT else r for r in old[field]]
            actual = [r for r in data[field] if r['episode'] in ids and r['kind'] not in NEW]
            assert actual == expected
    train_places = {m['row']['scene'].split('/')[0] for m in plan['catalog'] if m['id'] in plan['training_ids']}
    primary_places = {r['scene'].split('/')[0] for r in data['rows'] if selected(r, 'primary51')}
    assert len(primary_places) == 51 and not train_places & primary_places
    geometry_diff = 0.
    for meta in plan['catalog']:
        d = merged_maps(meta['id'])
        for seed in SEEDS:
            delta = float((d['predictions'][f'content_center_{seed}_normal'] -
                           d['predictions'][f'content_center_{seed}_initial_geometry']).abs().max())
            assert delta < 2e-5
            geometry_diff = max(geometry_diff, delta)
    assert max(r['center_displacement'] for r in new['updates']) == 0
    assert max(r['opacity_abs_change'] for r in new['updates']) == 0
    covariance_diff = max(r['covariance_relative_change'] for r in new['updates'])
    assert covariance_diff < 1e-5
    reads = []; scoring = []
    with torch.inference_mode():
        for rid in ('fit_00', 'confirm_00', 'fresh_19'):
            meta = next(m for m in plan['catalog'] if m['id'] == rid)
            raw_data = load_record(meta); raw, rec = raw_data['raw'], raw_data['rec']
            stored = merged_maps(rid); keep = torch.tensor([r == 'evaluation' for r in rec['roles']])
            q = rec['query'][keep]
            for kind, seed in [('content_center', s) for s in SEEDS] + [('content_gaussian', SEEDS[0])]:
                net, _ = make_models(kind, plan); net.eval()
                net.load_state_dict(torch.load(model_path(kind, seed), map_location='cpu', weights_only=False)['model'])
                directory = OUT if kind in NEW else (RECENT if rid.startswith('fresh_') else OLD)
                memory = torch.load(directory / 'memories' / f'{rid}_{kind}_{seed}.pt', map_location='cpu', weights_only=False)['memory']
                pred = net.read(rec['input_tokens'][keep], memory, raw['c2w'][q], raw['K'][q], raw['diagnostics']['unit_meters'])
                saved = stored['predictions'][f'{kind}_{seed}_normal']
                delta = float((pred - saved).abs().max()); assert delta < 2e-5
                reads.append({'episode': rid, 'kind': kind, 'seed': seed, 'max_difference': delta})
                error = saved.numpy().astype('float64') - stored['target'].numpy().astype('float64')
                valid = stored['valid'].numpy()
                independent = float(np.mean([np.mean(e[v] ** 2) for e, v in zip(error, valid)]))
                row = next(r for r in data['rows'] if r['episode'] == rid and r['kind'] == kind and
                           r['seed'] == seed and r['condition'] == 'normal')
                delta = abs(row['log_mse'] - independent); assert delta < 2e-6
                scoring.append(delta)
    accounting = max(r['accounting_max_diff'] for r in data['rows'] + data['full_query'] if r['condition'] == 'normal')
    assert accounting < 1e-10
    result = {'complete': True, 'fits': fits, 'new_fits': 3, 'old_models_unchanged': 24,
              'old_scores_reused_exact': True, 'new_saved_memories': 255, 'all_clips': 85,
              'primary_places': 51, 'training_primary_places_disjoint': True,
              'queries': sum(r['queries'] for r in new['counts']),
              'masked_inputs': sum(r['masked_inputs'] for r in new['counts']),
              'frozen_geometry_read_max_difference': geometry_diff,
              'frozen_covariance_relative_difference_max': covariance_diff,
              'independent_CPU_reads': reads, 'independent_numpy_max_difference': max(scoring),
              'decomposition_accounting_max_difference': accounting, 'confirmation_claim': False,
              'source_sha256': sha256(__file__)}
    write_json(OUT / 'checks.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('independent_CPU_reads', 'fits')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('analysis', 'checks'))
    globals()[p.parse_args().stage]()
