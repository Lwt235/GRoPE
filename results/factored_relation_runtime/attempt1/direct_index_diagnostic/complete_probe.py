"""Complete the saved GPU diagnostic after a list-valued-metric serialization fix."""
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.factored_relation_runtime import checked, validation_cache
from experiments import pe_runtime_diagnostic as prior
from experiments.region_confirmation_metric_runtime import BufferedNpz
from experiments.round65_common import read, write, sha

OUT = Path(__file__).resolve().parent
plan = read(OUT / 'completion_plan.json')
for path, digest in plan['identities'].items():
    assert sha(path) == digest, path
prior.native_engine.configure()
lock = checked()
cache = validation_cache('region')
with BufferedNpz(OUT / 'logits.npz') as z:
    assert np.array_equal(z['ids'], cache['ids'])
    outputs = [torch.from_numpy(z[k]) for k in ('original', 'factored')]
with BufferedNpz(lock['checkpoints']['region/direct_mlp']['validation']) as z:
    assert torch.equal(outputs[0], torch.from_numpy(z['logits']))
metrics = [prior.region_engine.metric_torch(v, cache['overlap'], cache['world_centers']) for v in outputs]
world = torch.from_numpy(cache['world_centers']).double()
overlap = torch.from_numpy(cache['overlap']).double()
rows = []
for direction in range(2):
    x, y = [v if direction == 0 else v.transpose(-1, -2) for v in outputs]
    truth = overlap if direction == 0 else overlap.transpose(-1, -2)
    ia, ib = x.argmax(-1), y.argmax(-1)
    for pair, query in (ia != ib).nonzero().tolist():
        ka, kb, view = int(ia[pair, query]), int(ib[pair, query]), 1 - direction
        da = float((world[pair, direction, query] - world[pair, view, ka]).norm())
        db = float((world[pair, direction, query] - world[pair, view, kb]).norm())
        oa, ob = float(truth[pair, query, ka]), float(truth[pair, query, kb])
        rows.append(dict(pair=pair, id=str(cache['ids'][pair]), direction=direction, query=query,
            old_key=ka, new_key=kb, valid_query=bool(truth[pair, query].max() >= .3),
            old_logit_margin=float(x[pair, query, ka] - x[pair, query, kb]),
            new_logit_margin=float(y[pair, query, kb] - y[pair, query, ka]),
            identical_key_inputs=all(np.array_equal(cache[k][pair, view, ka], cache[k][pair, view, kb])
                                     for k in ('features', 'centers', 'covariance')),
            world_center_max_difference=float((world[pair, view, ka] - world[pair, view, kb]).abs().max()),
            whole_label_columns_equal=bool(torch.equal(truth[pair, :, ka], truth[pair, :, kb])),
            overlap_old=oa, overlap_new=ob, center_error_old=da, center_error_new=db,
            region_decision_changed=(oa >= .3) != (ob >= .3),
            center_decision_changed=(da <= .1) != (db <= .1)))
result = dict(completed_unix=time.time(), source_sha256=sha(__file__), completion_plan_sha256=sha(OUT / 'completion_plan.json'),
    logits_sha256=sha(OUT / 'logits.npz'), original_stored_validation_exact=True,
    changed_queries_by_direction=[sum(r['direction'] == d for r in rows) for d in range(2)], changes=rows,
    metrics_original=metrics[0], metrics_factored=metrics[1],
    metric_differences={k: metrics[1][k] - v for k, v in metrics[0].items() if isinstance(v, (int, float))},
    non_scalar_fields_equal={k: metrics[1][k] == v for k, v in metrics[0].items() if not isinstance(v, (int, float))},
    original_mechanical_screen_unchanged=True, model_or_threshold_tuning=False, new_data_access=False,
    gpu_probe_was_not_repeated=True, goal_complete=False)
assert not (OUT / 'result.json').exists()
write(OUT / 'result.json', result)
print('INDEX_DIAGNOSTIC_COMPLETE', result['changed_queries_by_direction'], result['metric_differences'], flush=True)
