"""Extract prespecified comparisons, scene variation and query coverage."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]
result = json.loads((OUT/'result.json').read_text())
assert len(result['records']) == 36
roles = ['relative5', 'distance_mlp', 'geotransformer_rpe', 'fixed_isotropic5']
conditions = ['base', 'fixed_sparse256', 'retokenized256']
seeds = [11,22,33]
lookup = {(r['condition'],r['role'],r['seed']):r['metrics'] for r in result['records']}
summary = {}
for condition in conditions:
    rows = {}
    for role in roles:
        values = [lookup[condition,role,s] for s in seeds]
        rows[role] = {key:None if any(v['macro'][key] is None for v in values)
                           else float(np.mean([v['macro'][key] for v in values]))
                      for key in values[0]['macro']}
    with np.load(result['data_coverage'][condition]['path'],allow_pickle=False) as z:
        overlap = z['overlap']
        valid = np.stack((overlap.max(-1)>=.3,overlap.max(-2)>=.3),1)
        by_scene = {}
        for scene in result['available_scenes']:
            v = valid[z['scenes']==scene]
            by_scene[scene] = dict(pairs=len(v), valid_queries=int(v.sum()), total_queries=v.size,
                                  zero_valid_pairs=int((v.sum((1,2))==0).sum()))
    scene_deltas = {}
    for control in roles[1:]:
        scene_deltas[control] = {}
        for scene in result['planned_scenes']:
            left = [lookup[condition,'relative5',s]['by_scene'][scene] for s in seeds]
            right = [lookup[condition,control,s]['by_scene'][scene] for s in seeds]
            scene_deltas[control][scene] = {
                key:None if any(a[key] is None or b[key] is None for a,b in zip(left,right)) else
                    float(np.mean([a[key]-b[key] for a,b in zip(left,right)]))
                for key in left[0] if key in rows['relative5']}
    summary[condition] = dict(means=rows, observed_scene_coverage=by_scene,
        valid_queries=int(valid.sum()),total_queries=valid.size,
        zero_valid_pairs=int((valid.sum((1,2))==0).sum()),scene_mean_deltas=scene_deltas)

state=json.loads((OUT/'status.json').read_text())
launch=json.loads((OUT/'inference_launch.json').read_text())
summary['execution'] = dict(predictions=len(state['completed']),
    max_peak_allocated_mib=max(r['peak_allocated_bytes'] for r in state['completed'])/2**20,
    inference_wall_seconds=state['observed_unix']-state['started_unix'],
    selected_gpu=launch['selected'],independent_metric_max_error=result['independent_metric_max_error'])
destination=OUT/'actual_review1/analysis.json'
with destination.open('x') as f:
    json.dump(summary,f,indent=2)
    f.write('\n')
print(json.dumps(summary,indent=2))
