"""Summarize the fixed references, including undefined covered-only cases."""
import json
from collections import Counter

import numpy as np

from experiments.gaussian_scene_data import OUT, read
from experiments.gaussian_memory_incremental_data import write_json
from experiments.gaussian_memory_setup import sha256


def place_summary(rows, region, metric):
    places = sorted({row['place'] for row in rows})
    values = {}
    empty = []
    for place in places:
        selected = [row for row in rows if row['place'] == place]
        observed = []
        for row in selected:
            value = row['metrics'][region][metric]
            if value is None:
                empty.append(dict(id=row['id'], query=row['query']))
            else:
                observed.append(value)
        values[place] = float(np.mean(observed)) if observed else None
    estimable = [value for value in values.values() if value is not None]
    return dict(mean=float(np.mean(estimable)) if estimable else None,
                per_place=values, estimable_places=len(estimable), all_places=len(places),
                estimable_views=len(rows)-len(empty), all_views=len(rows), empty_views=empty)


def main():
    # The full references are available before the learned fits finish. Their
    # values are already independently rescored; no new predictions are made.
    score = OUT/'scoring/metrics.json'
    if not score.exists():
        score = OUT/'partial_scoring/metrics.json'
    rows = [row for row in read(score)['rows']
            if row['kind'] in ('native_current', 'official_history', 'render_depth')]
    assert Counter(row['kind'] for row in rows) == dict.fromkeys(
        ('native_current', 'official_history', 'render_depth'), 96)
    plan_hash = sha256(OUT/'plan.json')
    for name in ('native', 'renderer'):
        manifest = read(OUT/'references'/f'{name}.json')
        assert manifest['complete'] and manifest['plan_sha256'] == plan_hash
        assert sha256(manifest['prediction_path']) == manifest['prediction_sha256']
    summary = {}
    for kind in ('native_current', 'official_history', 'render_depth'):
        for condition in ('full', 'missing'):
            selected = [row for row in rows if row['kind'] == kind and row['condition'] == condition]
            assert len(selected) == 48
            regions = ['whole', 'roi'] + (['render_covered'] if kind == 'render_depth' else [])
            for region in regions:
                for metric in ('log_mse', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m'):
                    summary[f'{kind}/{condition}/{region}/{metric}'] = place_summary(selected, region, metric)
    rendered = [row for row in rows if row['kind'] == 'render_depth' and row['condition'] == 'full']
    coverage = {}
    for place in sorted({row['place'] for row in rendered}):
        selected = [row for row in rendered if row['place'] == place]
        coverage[place] = float(np.mean([
            row['metrics']['render_covered']['pixels']/row['metrics']['whole']['pixels'] for row in selected]))
    native = read(OUT/'references/native.json')['costs']
    write_json(OUT/'references/summary.json', dict(
        complete=True, plan_sha256=plan_hash, input_metrics_sha256=sha256(score),
        predictions=288, summaries=summary,
        renderer_valid_pixel_coverage=dict(mean=float(np.mean(list(coverage.values()))), per_place=coverage,
            rule='Within valid sensor target pixels; place-equal and query-equal, alpha>1e-4'),
        official_prefix_tensor_bytes=[dict(id=row['id'], **row['prefix_tensor_bytes']) for row in native],
        scope=['Fixed unadapted backgrounds, no learned-method ranking or fresh confirmation',
               'All-pixel renderer scores retain uncovered source-unit fill and every place',
               'Covered-only scores condition on estimable views; denominators and empty views explicit',
               'Official prefix is neither matched memory budget nor matched adaptation or known-camera input',
               'Reference timings in original manifests are background measurements, not exclusive speed comparisons']))
    print(json.dumps(dict(complete=True, predictions=288,
                         renderer_valid_coverage=float(np.mean(list(coverage.values()))))), flush=True)


if __name__ == '__main__':
    main()
