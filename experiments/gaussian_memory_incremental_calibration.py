"""Describe existing calibration dispersion without changing scales or samples."""
import json

import numpy as np

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import OUT, CACHE, load_plan, write_json


def main():
    groups = {}; identities = {}
    for meta in load_plan()['catalog']:
        rid = meta['id']; path = CACHE / rid / 'identity.json'
        identity = json.loads(path.read_text()); identities[rid] = sha256(path)
        for d in identity['diagnostics']:
            group = d['kind'] + ('_full' if d.get('condition') == 0 else '_masked' if d.get('condition') in (1, 2) else '')
            groups.setdefault(group, []).append(d)
    summary = {}
    for group, rows in groups.items():
        valid = [r for r in rows if r['valid']]
        summary[group] = {'windows': len(rows), 'numerically_available': len(valid),
            'max_min_ratio_quantile_levels': [.5, .9, .95, 1.],
            'max_min_ratio_quantiles': np.quantile([r['ratio_max_over_min'] for r in valid], [.5, .9, .95, 1.]).tolist(),
            'known_median_baseline_quantile_levels': [0., .5, .9, 1.],
            'known_median_baseline_m_quantiles': np.quantile([np.median(r['known_baselines_m']) for r in rows], [0., .5, .9, 1.]).tolist()}
    value = {'complete': True, 'groups': summary, 'cache_identity_sha256': identities,
             'source_sha256': sha256(__file__), 'changes_to_scales_samples_or_models': False,
             'interpretation': 'Numerical availability is not calibration accuracy. Pairwise max/min spread is descriptive and can be sensitive to short baselines. All compared methods use the same frozen per-window scale; no sensor scale alignment, filtering or threshold tuning is performed.'}
    write_json(OUT / 'calibration_dispersion.json', value)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
