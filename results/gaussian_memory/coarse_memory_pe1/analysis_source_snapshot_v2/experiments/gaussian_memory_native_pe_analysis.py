"""Prospective paired comparisons and input-only strata for the finite PE experiment."""
import json
import numpy as np

from experiments.gaussian_memory_native_pe_data import OUT, read, write_json, load_plan, sample_key, sha256, ROOT

METRICS = ('hidden_log_mse', 'hidden_centered_log_mse', 'hidden_bias_squared', 'log_mse',
           'centered_log_mse', 'bias_squared', 'absrel', 'hidden_absrel', 'rmse_m')
PAIRS = [('gaussian', 'center'), ('gaussian', 'sphere'), ('sphere', 'center'), ('center', 'native'),
         ('gaussian', 'native'), ('native', 'official'), ('center', 'official'), ('gaussian', 'official')]


def compare(candidate, baseline, places, seeds, draws=5000):
    nseed, nplace = candidate.shape
    rng = np.random.default_rng(20261111)
    si = rng.integers(nseed, size=(draws, nseed)); pi = rng.integers(nplace, size=(draws, nplace))
    a = candidate[si[:, :, None], pi[:, None, :]].mean((1, 2))
    b = baseline[si[:, :, None], pi[:, None, :]].mean((1, 2))
    rel = (b - a) / b * 100
    gain = baseline - candidate; placegain = gain.mean(0); seedgain = gain.mean(1)
    return {'candidate_mean': float(candidate.mean()), 'baseline_mean': float(baseline.mean()),
            'gain': float(gain.mean()), 'relative_gain_percent': float(gain.mean() / baseline.mean() * 100),
            'gain_95': np.percentile(b - a, [2.5, 97.5]).tolist(),
            'relative_gain_percent_95': np.percentile(rel, [2.5, 97.5]).tolist(),
            'positive_places': int((placegain > 0).sum()), 'places': nplace,
            'place_gain': dict(zip(places, map(float, placegain))),
            'seed_gain': dict(zip(map(str, seeds), map(float, seedgain))),
            'positive_seeds': int((seedgain > 0).sum()), 'seeds': nseed,
            'definition': 'baseline minus candidate; positive is lower error; ratio of paired means'}


def analyze():
    plan = load_plan(); check = read(OUT / 'independent_result_checks.json'); assert check['complete']
    rows = read(OUT / 'independent_rows.json'); seeds = plan['seeds']; catalog = plan['catalog']
    ids = [m['id'] for m in catalog]; places = [m['place'] for m in catalog]
    lookup = {(r['id'], r['condition'], r['kind'], r['seed']): r for r in rows}
    def matrix(condition, kind, metric, selected=None):
        subset = selected if selected is not None else ids
        return np.array([[lookup[rid, condition, kind, None if kind == 'official' else seed][metric]
                          for rid in subset] for seed in seeds])
    comparisons = {}; tables = {}
    for condition in plan['conditions']:
        comparisons[condition] = {}; tables[condition] = {}
        for kind in ['official'] + plan['kinds']:
            tables[condition][kind] = {metric: float(matrix(condition, kind, metric).mean()) for metric in METRICS}
        for candidate, baseline in PAIRS:
            comparisons[condition][f'{candidate}_vs_{baseline}'] = {
                metric: compare(matrix(condition, candidate, metric), matrix(condition, baseline, metric), places, seeds)
                for metric in METRICS}
    stats = read(OUT / 'training_statistics.json'); strata = {}; assignments = {}
    for name, threshold in stats['thresholds'].items():
        values = {rid: read(OUT / 'cache' / (sample_key(rid, 11, 'persistent').replace('/', '_') + '.json'))['geometry_statistics'][name] for rid in ids}
        groups = {'low': [rid for rid in ids if values[rid] <= threshold], 'high': [rid for rid in ids if values[rid] > threshold]}
        assignments[name] = {'threshold_from_training': threshold, 'values': values, 'groups': groups}
        strata[name] = {}
        for group, selected in groups.items():
            selected_places = [places[ids.index(rid)] for rid in selected]
            strata[name][group] = {'n': len(selected), 'ids': selected, 'places': selected_places, 'comparisons': {}}
            if not selected: continue
            for candidate, baseline in PAIRS[:3]:
                strata[name][group]['comparisons'][f'{candidate}_vs_{baseline}'] = {
                    metric: compare(matrix('persistent', candidate, metric, selected), matrix('persistent', baseline, metric, selected), selected_places, seeds)
                    for metric in METRICS[:3]}
    delta = matrix('persistent', 'gaussian', 'hidden_log_mse').mean(0) - matrix('persistent', 'center', 'hidden_log_mse').mean(0)
    worst = sorted(zip(ids, delta), key=lambda item: (-item[1], item[0]))[0][0]
    selected = list(dict.fromkeys(plan['fixed_examples'] + [worst]))
    result = {'complete': True, 'experiment': plan['experiment'], 'rows': len(rows), 'tables': tables,
              'comparisons': comparisons, 'strata_assignments': assignments, 'strata': strata,
              'fixed_example_ids': selected, 'worst_id': worst, 'worst_place': places[ids.index(worst)],
              'bootstrap': plan['bootstrap'], 'positive_means_lower_error': True,
              'all_data_previously_used': True, 'fresh_confirmation': False,
              'checkpoints': check['checkpoints'], 'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_native_pe_analysis.py')}
    write_json(OUT / 'analysis.json', result)
    write_json(OUT / 'figure_selection.json', {'fixed': plan['fixed_examples'], 'worst': worst,
               'selection': 'largest persistent ROI Gaussian minus Center error, mean over fixed three seeds',
               'ids': selected, 'visualization_seed': seeds[0]})
    print(json.dumps({condition: {key: value['hidden_log_mse'] for key, value in comparisons[condition].items()}
                      for condition in comparisons}), flush=True)


if __name__ == '__main__': analyze()
