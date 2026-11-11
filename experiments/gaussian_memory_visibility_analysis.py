"""Place/seed-paired comparisons for the fixed historical visibility conditions."""
import json
import numpy as np

from experiments.gaussian_memory_visibility_data import (
    OUT, CACHE, FAMILIES, SEEDS, CONDITIONS, POLICIES, load_plan, write_json)

METRICS = ('hidden_log_mse', 'log_mse', 'visible_log_mse', 'centered_log_mse',
           'bias_squared', 'absrel', 'rmse_m')


def paired(a, b, places):
    """Positive gain means candidate b has lower error than baseline a."""
    assert a.shape == b.shape == (12, 3)
    rng = np.random.default_rng(20261111)
    pi = rng.integers(12, size=(5000, 12)); si = rng.integers(3, size=(5000, 3))
    ba = a[pi[:, :, None], si[:, None, :]].mean((1, 2))
    bb = b[pi[:, :, None], si[:, None, :]].mean((1, 2))
    d = a - b
    return {'baseline_mean': float(a.mean()), 'candidate_mean': float(b.mean()),
        'gain': float(d.mean()), 'relative_gain_percent': float(100 * d.mean() / a.mean()),
        'paired_place_seed_95': np.quantile(ba - bb, [.025, .975]).tolist(),
        'relative_gain_percent_95': np.quantile(100 * (ba - bb) / ba, [.025, .975]).tolist(),
        'seed_gain': d.mean(0).tolist(), 'seed_relative_gain_percent': (100 * d.mean(0) / a.mean(0)).tolist(),
        'positive_seeds': int((d.mean(0) > 0).sum()), 'positive_places': int((d.mean(1) > 0).sum()),
        'place_gain': dict(zip(places, d.mean(1).tolist())),
        'definition': 'baseline minus candidate; positive is improvement'}


def delta_summary(delta, places):
    rng = np.random.default_rng(20261111)
    pi = rng.integers(len(places), size=(5000, len(places)))
    si = rng.integers(delta.shape[1], size=(5000, delta.shape[1]))
    boot = delta[pi[:, :, None], si[:, None, :]].mean((1, 2))
    return {'mean': float(delta.mean()), 'paired_95': np.quantile(boot, [.025, .975]).tolist(),
            'by_place': dict(zip(places, delta.mean(1).tolist())), 'by_seed': delta.mean(0).tolist(),
            'positive_places': int((delta.mean(1) > 0).sum())}


def main():
    plan = load_plan(); catalog = sorted(plan['catalog'], key=lambda m: m['place'])
    places = [m['place'] for m in catalog]; records = {}; n = 0
    for m in catalog:
        r = json.loads((OUT / 'evaluation' / f'{m["id"]}.json').read_text())
        assert r['complete'] and len(r['rows']) == 102
        records[m['id']] = r['rows']; n += len(r['rows'])
    methods = ['native', 'local'] + [f'{f}/{p}' for f in FAMILIES for p in POLICIES]
    matrices = {}; result = {}; interactions = {}
    for ci, condition in enumerate(CONDITIONS):
        matrices[condition] = {}; result[condition] = {}
        for metric in METRICS:
            data = {}
            for method in methods:
                parts = method.split('/'); fam = parts[0]; policy = parts[1] if len(parts) == 2 else None
                values = []
                for m in catalog:
                    values.append([next(r[metric] for r in records[m['id']] if r['condition'] == ci
                                        and r['family'] == fam and r['policy'] == policy
                                        and r['seed'] == (None if fam == 'native' else seed)) for seed in SEEDS])
                data[method] = np.asarray(values, dtype=float)
                assert np.isfinite(data[method]).all()
            comparisons = {}
            for fam in FAMILIES:
                comparisons[f'{fam}:history_vs_recent'] = paired(data[f'{fam}/recent'], data[f'{fam}/history'], places)
            for policy in POLICIES:
                for control in ('content_static', 'content_dynamic', 'center_dynamic', 'soft_static'):
                    comparisons[f'{policy}:soft_dynamic_vs_{control}'] = paired(data[f'{control}/{policy}'], data[f'soft_dynamic/{policy}'], places)
                for control in ('local', 'native'):
                    for fam in ('soft_dynamic', 'content_static', 'content_dynamic'):
                        comparisons[f'{policy}:{fam}_vs_{control}'] = paired(data[control], data[f'{fam}/{policy}'], places)
            result[condition][metric] = {'means': {k: float(v.mean()) for k, v in data.items()}, 'comparisons': comparisons}
            matrices[condition][metric] = {k: v.tolist() for k, v in data.items()}
    for metric in METRICS:
        interactions[metric] = {}
        def gain(condition, fam):
            d = matrices[condition][metric]
            return np.array(d[f'{fam}/recent']) - np.array(d[f'{fam}/history'])
        for fam in FAMILIES:
            for control in ('full', 'query_only'):
                interactions[metric][f'{fam}:persistent_minus_{control}_history_gain'] = delta_summary(gain('persistent', fam) - gain(control, fam), places)
        for control in ('content_static', 'content_dynamic'):
            interactions[metric][f'persistent:soft_minus_{control}_history_gain'] = delta_summary(gain('persistent', 'soft_dynamic') - gain('persistent', control), places)
    information = {}; inputs = {}; workers = []
    for m in catalog:
        inputs[m['id']] = json.loads((OUT / 'inputs' / f'{m["id"]}.json').read_text())
    keys = [(r['map'], r['region']) for r in next(iter(inputs.values()))['diagnostic']['rows']]
    for map_name, region in keys:
        selected = [next(r for r in inputs[m['id']]['diagnostic']['rows'] if (r['map'], r['region']) == (map_name, region)) for m in catalog]
        info = {}
        for metric in selected[0]:
            if metric in ('map', 'region'): continue
            values = [r[metric] for r in selected]
            available = [i for i, v in enumerate(values) if v is not None]
            if not available:
                info[metric] = {'mean': None, 'contributing_places': 0}; continue
            v = np.array([values[i] for i in available], dtype=float)[:, None]
            info[metric] = dict(delta_summary(v, [places[i] for i in available]), contributing_places=len(available))
        information[f'{map_name}/{region}'] = info
    masks = {m['place']: inputs[m['id']]['diagnostic']['mask_patches_per_frame'] for m in catalog}
    for i in range(2):
        rec = json.loads((OUT / f'worker_{i}_complete.json').read_text()); assert rec['complete']; workers.append(rec)
    output = {'complete': True, 'places': places, 'exported_predictions': n, 'conditions': result,
        'matrices': matrices, 'history_gain_interactions': interactions,
        'information': information, 'mask_patches_by_place': masks,
        'cost': {'worker_seconds': [r['seconds'] for r in workers],
                 'new_backbone_windows': sum(r['new_backbone_inferences'] for r in inputs.values()),
                 'peak_allocated_GiB': max(r['peak_allocated_GiB'] for r in workers),
                 'adapter_predictions': 1188, 'new_native_query_predictions': 12,
                 'reused_native_query_predictions_rescored': 24, 'new_training_steps': 0, 'new_download_bytes': 0},
        'scope': 'All previously used places. The same fixed ROI is scored in all conditions. Native is repeated only to pair model seeds, not treated as three independent fits. Diagnostic intervals, not fresh confirmation.'}
    assert n == 1224 and output['cost']['new_backbone_windows'] == 62
    write_json(OUT / 'analysis.json', output)
    main = result['persistent']['hidden_log_mse']
    print(json.dumps({'analysis_complete': True, 'roi_means': main['means'],
        'soft_history_gain': main['comparisons']['soft_dynamic:history_vs_recent'],
        'soft_vs_strong_content': main['comparisons']['history:soft_dynamic_vs_content_static']}), flush=True)


if __name__ == '__main__': main()
