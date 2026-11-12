"""Descriptive reanalysis of existing results; no fitting, thresholds, or exclusions."""
from pathlib import Path
import hashlib
import json
import statistics

OUT = Path(__file__).resolve().parent
BASE = OUT.parent

def read(name):
    return json.loads((BASE / name).read_text())

conditions = read('source_conditions.json')
summary = read('scoring/summary.json')
by_split = {}
for split in ('training', 'development'):
    rows = [r for r in conditions['source_conditions'] if r['split'] == split]
    by_split[split] = {}
    for key in ('source_center_rmse_m', 'source_orientation_mean_degrees'):
        values = [r[key] for r in rows]
        by_split[split][key] = dict(n=len(values), mean=statistics.mean(values),
                                   median=statistics.median(values), minimum=min(values), maximum=max(values))
places = []
for r in conditions['source_conditions']:
    if r['split'] != 'development':
        continue
    index = summary['places'].index(r['place'])
    p = summary['summaries']['missing/roi/centered_log_mse/P']['per_place'][index]
    g = summary['summaries']['missing/roi/centered_log_mse/G']['per_place'][index]
    places.append(dict(place=r['place'], id=r['id'],
                       source_orientation_degrees=r['source_orientation_mean_degrees'],
                       source_center_rmse_m=r['source_center_rmse_m'],
                       P=p, G=g, G_minus_P=g-p, G_relative_error_increase_percent=100*(g/p-1)))
result = dict(complete=True, source_residuals=by_split, all_development_places=places,
              source_sha256={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                             for name in ('source_conditions.json','scoring/summary.json')},
              scope='Post-result descriptive analysis of all12 old development places. No new predictions, fitted association, p-value, quality cutoff, exclusion, or fresh confirmation. Metre residuals depend on scene scale; source-camera residuals do not uniquely identify scene geometry error.')
(OUT/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print('Recorded all12 place descriptions and source residual summaries; original results unchanged.')
