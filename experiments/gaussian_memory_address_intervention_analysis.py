"""Independent exported-score/mixture checks and paired output comparisons."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_address_intervention import (
    ROOT, OUT, ADDRESS, INPUT_CACHE, FAMILIES, POLICIES, CONDITIONS, SEEDS, REGIONS,
    ALPHA, read, load, sha256, write_json)

METRICS = ('hidden_log_mse', 'log_mse', 'visible_log_mse', 'centered_log_mse', 'bias_squared',
           'absrel', 'rmse_m') + tuple(r + '_log_mse' for r in REGIONS)


def independent_scores(prediction, sample, regions):
    depth = prediction.numpy().astype(np.float64); target = sample['target'].numpy().astype(np.float64)
    valid, hidden = sample['valid'].numpy(), sample['hidden'].numpy()
    assert np.isfinite(depth).all() and (depth > 0).all()
    residual = np.log(np.maximum(depth, 1e-6) * sample['metric_scale'] / target)
    rv = residual[valid]; e = depth[valid] * sample['metric_scale'] - target[valid]
    def mse(mask): return float(np.mean(residual[mask] ** 2)) if mask.any() else None
    result = {'log_mse': mse(valid), 'hidden_log_mse': mse(valid & hidden),
              'visible_log_mse': mse(valid & ~hidden), 'centered_log_mse': float(np.var(rv)),
              'bias_squared': float(np.mean(rv) ** 2), 'absrel': float(np.mean(np.abs(e) / target[valid])),
              'rmse_m': float(np.sqrt(np.mean(e ** 2)))}
    for name, region in regions.items():
        mask = region.numpy().astype(bool) & valid
        result[name + '_pixels'] = int(mask.sum()); result[name + '_log_mse'] = mse(mask)
    return result


def check():
    started = time.monotonic(); plan = read(OUT / 'plan.json')
    for path, digest in plan['inputs'].items(): assert sha256(path) == digest, path
    for path, digest in plan['records'].items(): assert sha256(path) == digest, path
    for name, digest in plan['source_sha256'].items(): assert sha256(ROOT / name) == digest == sha256(OUT / 'source_snapshot' / name)
    for model in plan['models']: assert sha256(model['path']) == model['sha256']
    maximum = {m: 0. for m in METRICS}; prediction_count = reference_count = replay_count = empty_count = captures_count = 0
    target_max = mixture_max = replay_max = 0.; records = []
    for meta in plan['catalog']:
        rid = meta['id']; rec = read(OUT / 'evaluation' / f'{rid}.json')
        assert rec['complete'] and rec['plan_sha256'] == sha256(OUT / 'plan.json')
        for key in ('prediction', 'diagnostic'): assert sha256(rec[key + '_path']) == rec[key + '_sha256']
        predictions = load(rec['prediction_path']); diag = load(rec['diagnostic_path'])
        address = load(read(ADDRESS / 'clips' / f'{rid}.json')['artifact_path'])
        perm = np.asarray(plan['permutations'][rid]); assert np.array_equal(np.sort(perm), np.arange(256))
        assert (perm != np.arange(256)).all()
        for state, options in diag['targets'].items():
            binding = address['bindings'][state]
            for label, suffix in [('all', 'all'), ('early_exclusive', 'history')]:
                u = binding['utility'][label].numpy().astype(float); mass = u.sum(-1)
                active = address['roi'].numpy() & (mass > 0)
                t = np.zeros_like(u); t[active] = u[active] / mass[active, None]
                for arm, expected in [('correct_' + suffix, t), ('sham_' + suffix, t[:, perm])]:
                    saved = options[arm]
                    assert np.array_equal(active, saved['active'].numpy())
                    error = np.max(np.abs(saved['target'].numpy() - expected)); target_max = max(target_max, float(error))
                    assert error < 2e-7
                    assert np.allclose(saved['target'].sum(-1).numpy(), active.astype(float), atol=2e-7)
                assert torch.equal(options['correct_' + suffix]['target'].sort(-1).values,
                                   options['sham_' + suffix]['target'].sort(-1).values)
        samples = {ci: load(INPUT_CACHE / rid / f'q11_c{ci}.pt') for ci in CONDITIONS}
        for ci, condition in CONDITIONS.items():
            labs = address['labels'][condition]; binding = address['bindings'][condition + '/history']
            roi = address['roi'].numpy()[labs['query_patch'].numpy()]
            hit = binding['coverage']['retained'].numpy(); hist = labs['exclusive'].numpy()
            expected = {'supported_roi': roi & hit, 'history_only_roi': roi & hit & hist, 'unrepresented_roi': roi & ~hit}
            for name, mask in expected.items(): assert np.array_equal(diag['regions'][condition][name].numpy().ravel(), mask)
        lookup = {tuple(k): i for i, k in enumerate(predictions['keys'])}
        for i, row in enumerate(rec['rows']):
            key = [row['condition'], row['family'], row['seed'], row['policy'], row['arm']]
            assert predictions['keys'][i] == key
            expected = independent_scores(predictions['prediction_native'][i], samples[key[0]], diag['regions'][CONDITIONS[key[0]]])
            for name, value in expected.items():
                if value is None: assert row[name] is None; continue
                difference = abs(row[name] - value)
                if name in maximum: maximum[name] = max(maximum[name], difference)
                assert np.isclose(row[name], value, rtol=2e-5, atol=2e-6), (rid, key, name, difference)
            prediction_count += 1
        for d in diag['diagnostics']:
            assert d['inactive_value_max_abs'] == 0
            if not d['active_patches']:
                key = tuple(d['key']); baseline = (*key[:4], 'original')
                assert torch.equal(predictions['prediction_native'][lookup[key]], predictions['prediction_native'][lookup[baseline]])
                empty_count += 1
        for capture in diag['captures'].values():
            t = capture['target'].numpy().astype(float); v = capture['values'].numpy().astype(float)
            original = capture['original_value'].numpy().astype(float); active = capture['active'].numpy()
            expected = original.copy()
            expected[:, :, active] = ((1 - ALPHA) * original + ALPHA * np.einsum('ps,bhsd->bhpd', t, v))[:, :, active]
            # Saved values have one final BF16 rounding; compare at that precision.
            rounded = torch.from_numpy(expected.astype(np.float32)).to(torch.bfloat16).float().numpy()
            difference = np.abs(rounded - capture['mixed'].numpy()); mixture_max = max(mixture_max, float(difference.max()))
            assert np.sqrt(np.mean(difference ** 2)) < .002, float(difference.max())
            assert np.array_equal(capture['mixed'].numpy()[:, :, ~active], original[:, :, ~active])
            captures_count += 1
        refs = {role: load(spec['predictions']) for role, spec in plan['references'][rid].items()}
        for rep in rec['replays']:
            key = tuple(rep['key']); role = 'old' if key[1] == 'soft_dynamic' else 'new'
            index = refs[role]['keys'].index(list(key[:4])); original = refs[role]['prediction_native'][index]
            difference = float((predictions['prediction_native'][lookup[key]] - original).abs().max())
            replay_max = max(replay_max, difference); assert difference < 1e-4; replay_count += 1
        for row in rec['reference_rows']:
            ci = row['condition']; role = row['reference_role']; key = [ci, row['family'], row['seed'], row['policy']]
            index = refs[role]['keys'].index(key)
            expected = independent_scores(refs[role]['prediction_native'][index], samples[ci], diag['regions'][CONDITIONS[ci]])
            for name, value in expected.items():
                if value is None: assert row[name] is None
                else: assert np.isclose(row[name], value, rtol=2e-5, atol=2e-6), (rid, 'reference', name)
            reference_count += 1
        records.append({'id': rid, 'predictions': len(rec['rows']), 'reference_rows': len(rec['reference_rows'])})
        print(json.dumps({'checked': rid, 'predictions': len(rec['rows'])}), flush=True)
    assert prediction_count == 1152 and replay_count == 288 and reference_count == 672 and captures_count == 12
    result = {'complete': True, 'predictions_scored': prediction_count, 'reused_reference_rows_scored': reference_count,
        'original_replays': replay_count, 'replay_max_abs': replay_max, 'empty_support_exact_replays': empty_count,
        'mixture_capture_cases': captures_count, 'mixture_max_abs_after_bf16_rounding': mixture_max,
        'targets_max_abs': target_max, 'max_score_difference': maximum, 'records': records,
        'parameters_unchanged': True, 'seconds': time.monotonic() - started,
        'checker_sha256': sha256(ROOT / 'experiments/gaussian_memory_address_intervention_analysis.py')}
    write_json(OUT / 'independent_checks.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


def paired(a, b, places):
    a, b = np.asarray(a, float), np.asarray(b, float); assert a.shape == b.shape
    available = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    names = [p for i, p in enumerate(places) if available[i]]; a, b = a[available], b[available]
    if not len(names): return {'contributing_places': 0, 'gain': None}
    rng = np.random.default_rng(20261111); pi = rng.integers(len(names), size=(5000, len(names)))
    si = rng.integers(a.shape[1], size=(5000, a.shape[1]))
    ba = a[pi[:, :, None], si[:, None, :]].mean((1, 2)); bb = b[pi[:, :, None], si[:, None, :]].mean((1, 2))
    difference = a - b
    return {'baseline_mean': float(a.mean()), 'candidate_mean': float(b.mean()), 'gain': float(difference.mean()),
        'relative_gain_percent': float(100 * difference.mean() / a.mean()) if a.mean() else None,
        'paired_95': np.quantile(ba - bb, [.025, .975]).tolist(),
        'relative_gain_percent_95': np.quantile(100 * (ba - bb) / ba, [.025, .975]).tolist() if np.all(ba > 0) else None,
        'seed_gain': difference.mean(0).tolist(), 'seed_relative_gain_percent': (100 * difference.mean(0) / a.mean(0)).tolist(),
        'positive_seeds': int((difference.mean(0) > 0).sum()), 'positive_places': int((difference.mean(1) > 0).sum()),
        'place_gain': dict(zip(names, difference.mean(1).tolist())), 'contributing_places': len(names),
        'definition': 'baseline minus candidate; positive means lower task error'}


def analyze():
    plan = read(OUT / 'plan.json'); catalog = sorted(plan['catalog'], key=lambda m: m['place']); places = [m['place'] for m in catalog]
    recs = {m['id']: read(OUT / 'evaluation' / f'{m["id"]}.json') for m in catalog}
    matrices = {}; results = {}; interactions = {}; diagnostics = []
    for ci, condition in CONDITIONS.items():
        matrices[condition] = {}; results[condition] = {}
        for metric in METRICS:
            methods = sorted({(r['family'], r['policy'], r['arm']) for r in next(iter(recs.values()))['rows']})
            methods += [('content_static', p, 'reference') for p in POLICIES]
            methods += [('center_dynamic', p, 'reference') for p in POLICIES]
            methods += [('new_content', p, 'reference') for p in POLICIES]
            methods += [('new_center', p, 'reference') for p in POLICIES]
            methods += [('native', None, 'reference'), ('local', None, 'reference')]
            data = {}
            for family, policy, arm in methods:
                values = []
                for meta in catalog:
                    source = recs[meta['id']]['reference_rows' if arm == 'reference' else 'rows']
                    values.append([next(r[metric] for r in source if r['condition'] == ci and r['family'] == family
                                  and r['policy'] == policy and r['arm'] == arm and r['seed'] == (None if family == 'native' else s)) for s in SEEDS])
                data['/'.join((family, policy or '-', arm))] = np.asarray(values, float)
            comp = {}
            for fam in FAMILIES:
                for pol in POLICIES:
                    for target in (['all', 'history'] if pol == 'history' else ['all']):
                        prefix = fam + '/' + pol + '/'; correct, sham = 'correct_' + target, 'sham_' + target
                        for baseline, candidate in [('original', correct), (sham, correct), ('original', sham)]:
                            comp[prefix + candidate + '_vs_' + baseline] = paired(data[prefix + baseline], data[prefix + candidate], places)
                    for control in ('content_static', 'center_dynamic', 'new_content', 'new_center'):
                        for arm in ('original', 'correct_all'):
                            comp[f'{fam}/{pol}/{arm}_vs_{control}'] = paired(data[f'{control}/{pol}/reference'], data[f'{fam}/{pol}/{arm}'], places)
                for arm in ('original', 'correct_all', 'sham_all'):
                    comp[f'{fam}/{arm}/history_vs_recent'] = paired(data[f'{fam}/recent/{arm}'], data[f'{fam}/history/{arm}'], places)
            matrices[condition][metric] = {k: [[float(x) if np.isfinite(x) else None for x in row] for row in v]
                                           for k, v in data.items()}
            results[condition][metric] = {'means': {k: float(np.nanmean(v)) if np.isfinite(v).any() else None for k, v in data.items()},
                'contributing_places': {k: int(np.isfinite(v).all(1).sum()) for k, v in data.items()}, 'comparisons': comp}
    for metric in METRICS:
        result = {}
        for fam in FAMILIES:
            def gains(c, arm):
                d = matrices[c][metric]
                return np.asarray(d[fam + '/history/original'], float) - np.asarray(d[fam + '/history/' + arm], float)
            for arm in ('correct_all', 'correct_history'):
                # Positive gain in the comparison means the persistent improvement is greater.
                effect = paired(gains('persistent', arm), gains('full', arm), places)
                # Signed improvements are not losses; percentage ratios of them
                # are unstable and their generic loss labels would be misleading.
                for name in ('relative_gain_percent', 'relative_gain_percent_95', 'seed_relative_gain_percent'):
                    effect.pop(name, None)
                if 'baseline_mean' in effect:
                    effect['persistent_gain_mean'] = effect.pop('baseline_mean')
                    effect['full_gain_mean'] = effect.pop('candidate_mean')
                effect['definition'] = 'persistent (original minus intervention) minus full (original minus intervention), absolute log-MSE; positive means a larger improvement under persistent masking'
                result[fam + '/' + arm + ':persistent_minus_full_gain'] = effect
        interactions[metric] = result
    for meta in catalog:
        record = recs[meta['id']]; saved = load(record['diagnostic_path'])
        parent = load(read(ADDRESS / 'clips' / f'{meta["id"]}.json')['artifact_path'])
        for d in saved['diagnostics']:
            ci, family, seed, policy, arm = d['key']; condition = CONDITIONS[ci]
            counts = parent['labels'][condition]['valid_counts'].numpy() * parent['roi'].numpy()
            active = saved['targets'][condition + '/' + policy][arm]['active'].numpy()
            def mean(x, weights):
                return float((x.numpy() * weights).sum() / weights.sum()) if weights.sum() else None
            diagnostics.append({'id': meta['id'], 'place': meta['place'], 'condition': condition, 'family': family,
                'seed': seed, 'policy': policy, 'arm': arm, 'active_patches': d['active_patches'],
                'active_valid_roi_pixels': int((counts * active).sum()), 'roi_valid_pixels': int(counts.sum()),
                'route_before': mean(d['route_before'], counts), 'route_after': mean(d['route_after'], counts),
                'attention_L1_on_active': mean(d['attention_L1'], counts * active),
                'value_delta_rms_on_active': mean(d['value_delta_rms'], counts * active),
                'original_value_rms_on_active': mean(d['original_value_rms'], counts * active),
                'target_entropy_on_active': mean(d['target_entropy'], counts * active)})
    workers = [read(OUT / f'worker_{i}.json') for i in range(2)]; assert all(w['complete'] for w in workers)
    result = {'complete': True, 'experiment': plan['experiment'], 'places': places, 'conditions': results,
        'matrices': matrices, 'visibility_interactions': interactions, 'routing_diagnostics': diagnostics,
        'cost': {'worker_seconds': [w['seconds'] for w in workers], 'peak_GiB': max(w['peak_GiB'] for w in workers),
                 'exported_outputs': 1152, 'original_replays': 288, 'intervention_outputs': 864,
                 'reused_reference_rows': 672, 'new_training_steps': 0, 'new_download_bytes': 0},
        'scope': 'Query-truth-assisted development intervention; not deployable output gains or fresh confirmation.',
        'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_address_intervention_analysis.py')}
    write_json(OUT / 'analysis.json', result)
    comp = results['persistent']['hidden_log_mse']['comparisons']
    print(json.dumps({key: comp[key] for key in comp if '/history/correct_' in key and key.endswith(('_vs_original', '_vs_sham_all', '_vs_sham_history'))}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('check', 'analyze')); args = p.parse_args()
    torch.set_num_threads(4); (check if args.stage == 'check' else analyze)()
