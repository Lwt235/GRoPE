"""Place-equal source support, geometric consistency and state-retention summaries."""
import json
from itertools import product

import numpy as np

from experiments.gaussian_memory_raw_history_data import OUT, VARIANTS, POLICIES, load_plan, write_json

MODES = ('mixed', 'full', 'masked', 'hidden', 'visible')


def select(row, mode):
    if mode == 'mixed': return row['region'] == 'all'
    if mode == 'full': return row['region'] == 'all' and row['condition'] == 0
    if mode == 'masked': return row['region'] == 'all' and row['condition'] > 0
    return row['region'] == mode and row['condition'] > 0


def summarize(values, places):
    valid = np.isfinite(values); a = np.asarray(values)[valid]
    if not len(a): return {'mean': None, 'place_95': None, 'places': 0, 'by_place': dict(zip(places, [None] * len(places)))}
    rng = np.random.default_rng(20261111)
    boot = a[rng.integers(len(a), size=(5000, len(a)))].mean(1)
    return {'mean': float(a.mean()), 'place_95': np.quantile(boot, [.025, .975]).tolist(),
            'places': len(a), 'positive_places': int((a > 0).sum()),
            'by_place': {p: float(v) if np.isfinite(v) else None for p, v in zip(places, values)}}


def aggregate(rows, case_fields, places):
    cases = sorted({tuple(r[k] for k in case_fields) for r in rows}); output = {}; arrays = {}
    excluded = {'id', 'place', 'query', 'condition', 'region', *case_fields}
    metrics = [k for k in rows[0] if k not in excluded]
    for mode in MODES:
        output[mode] = {}; arrays[mode] = {}
        for case in cases:
            key = '/'.join(case); out = {}; arrays[mode][key] = {}
            chosen = [r for r in rows if select(r, mode) and tuple(r[k] for k in case_fields) == case]
            for metric in metrics:
                vals = []
                for place in places:
                    v = [r[metric] for r in chosen if r['place'] == place and r[metric] is not None]
                    vals.append(np.mean(v) if v else np.nan)
                values = np.asarray(vals, dtype='float64'); arrays[mode][key][metric] = values
                out[metric] = summarize(values, places)
            output[mode][key] = out
    return output, arrays


def main():
    plan = load_plan(); rows = {k: [] for k in ('raw_rows', 'history_rows', 'state_rows', 'retention_rows')}
    for rid in plan['late_ids']:
        r = json.loads((OUT / 'evaluation' / f'{rid}.json').read_text()); assert r['complete']
        for k in rows: rows[k] += r[k]
    places = sorted(p['place'] for p in plan['pairs']); arrays = {}; summary = {}
    for key, fields in [('raw_rows', ('variant', 'range')), ('history_rows', ('variant',)),
                        ('state_rows', ('policy',)), ('retention_rows', ('variant', 'policy'))]:
        summary[key], arrays[key] = aggregate(rows[key], fields, places)
    comparisons = {}
    for mode in MODES:
        comparisons[mode] = {}
        for source_range in ('early', 'recent', 'history'):
            for sensor, pred in [('sensor', 'prediction'), ('sensor_matched', 'prediction_matched')]:
                for metric in ('support_fraction', 'consistent_fraction', 'supported_absrel'):
                    a = arrays['raw_rows'][mode][f'{sensor}/{source_range}'][metric]
                    b = arrays['raw_rows'][mode][f'{pred}/{source_range}'][metric]
                    comparisons[mode][f'{sensor}_minus_{pred}/{source_range}/{metric}'] = summarize(a - b, places)
        for policy in ('corrective', 'rebuild'):
            for metric in ('support_fraction', 'consistent_fraction', 'supported_absrel'):
                a = arrays['state_rows'][mode][f'history_{policy}'][metric]
                b = arrays['state_rows'][mode][f'recent_{policy}'][metric]
                comparisons[mode][f'history_minus_recent_{policy}/{metric}'] = summarize(a - b, places)
    result = {'complete': True, 'places': places, 'counts': {k: len(v) for k, v in rows.items()},
        'summary': summary, 'comparisons': comparisons,
        'units': 'Fractions are 0..1; differences in fractions, not percent. place_95 resamples12places, no model seeds.',
        'undefined_conditional_metrics': 'No support yields null conditional error/retention; absolute support/consistent fractions remain0 and all queries retained. Each summary records contributing places.',
        'limits': 'Source sensor diagnostic only; raw points and Gaussian state footprints differ. Support is not Transformer output utility; all places previously used.'}
    assert result['counts'] == {'raw_rows': 4032, 'history_rows': 1344, 'state_rows': 1344, 'retention_rows': 5376}
    write_json(OUT / 'analysis.json', result)
    print(json.dumps({'complete': True, 'history_additions': {v: summary['history_rows']['mixed'][v] for v in ('sensor', 'prediction')}}), flush=True)


if __name__ == '__main__': main()
