"""Post-run confirmation review; run only after all 36 predictions and report."""
import time
from pathlib import Path
import numpy as np

from experiments.region_retokenization_confirmation import OUT, PREP, ROOT, ROLES, SEEDS, CONDITIONS, check, load
from experiments.round65_common import read, write, sha

lock=check()
state=read(OUT/'status.json')
preflight=read(OUT/'preflight.json')
access=read(OUT/'data_access_started.json')
download=read(OUT/'download_complete.json')
data=read(OUT/'data_manifest.json')
result=read(OUT/'result.json')
assert state['stage']=='complete' and state['active'] is None and len(state['completed'])==36
assert preflight['passed'] and len(preflight['records'])==12
assert all(r['maximum_logit_error']==0 and r['prediction_changes']==0 for r in preflight['records'])
assert preflight['conversion']['passed'] and preflight['conversion']['device']=='cuda'
assert data['completed_unix']>download['completed_unix']>access['started_unix']>preflight['completed_unix']>lock['created_unix']
assert access['scenes']==lock['available_scenes']
central=ROOT/'results/research_data_usage/region_retokenization_confirmation_access_20261109.json'
assert read(central)==access
assert sha(OUT/'download_manifest.json')==download['manifest_sha256']
downloads=read(OUT/'download_manifest.json')
assert len(downloads)==download['files'] and sum(x['bytes'] for x in downloads)==download['bytes']
assert len({x['path'] for x in downloads})==len(downloads)
for x in downloads:
    assert Path(x['path']).suffix=='.png' and sha(x['path'])==x['sha256']
assert sha(OUT/'raw_data_lock.json')==data['raw_data_lock_sha256']
assert sha(OUT/'data_manifest.json')==result['data_manifest_sha256']
assert len(lock['planned_scenes'])==10 and len(lock['available_scenes'])==9 and len(lock['unavailable'])==1
assert not result['decision']['complete_planned_cohort_evaluated']
assert result['unavailable']==lock['unavailable'] and result['technical_pair_exclusions']==data['excluded']
caches={c:load(c) for c in CONDITIONS}
for k in ('ids','scenes','scales','overlap','world_centers','permutation'):
    np.testing.assert_array_equal(caches['base'][k],caches['fixed_sparse256'][k])
for k in ('ids','scenes','scales'):
    np.testing.assert_array_equal(caches['base'][k],caches['retokenized256'][k])
pairs=len(caches['base']['ids'])
assert pairs+len(data['excluded'])==lock['scheduled_pairs']==864
expected={(c,r,s) for c in CONDITIONS for r in ROLES for s in SEEDS}
assert {(r['condition'],r['role'],r['seed']) for r in state['completed']}==expected
assert {(r['condition'],r['role'],r['seed']) for r in result['records']}==expected
assert len(result['records'])==36 and len(result['comparisons'])==18
assert [c['bootstrap_seed'] for c in result['comparisons']]==list(range(89001,89019))
lookup={}
for row in result['records']:
    model=next(m for m in lock['models'] if (m['role'],m['seed'])==(row['role'],row['seed']))
    assert row['checkpoint_sha256']==model['checkpoint_sha256']
    assert row['completed_unix']>data['completed_unix']
    assert row['evaluation_lock_sha256']==sha(OUT/'evaluation_lock.json')
    assert row['data_manifest_sha256']==sha(OUT/'data_manifest.json')
    assert row['peak_allocated_bytes']<=3*2**30
    assert sha(row['path'])==row['sha256']
    with np.load(row['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],caches[row['condition']]['ids'])
        assert z['logits'].shape==(pairs,64,64) and np.isfinite(z['logits']).all()
    metrics=row['metrics']
    assert set(metrics['by_scene'])==set(lock['planned_scenes'])
    assert sum(m['total_queries'] for m in metrics['by_scene'].values())==metrics['micro']['total_queries']==128*pairs
    assert sum(m['valid_queries'] for m in metrics['by_scene'].values())==metrics['micro']['valid_queries']
    for key,actual in metrics['macro'].items():
        values=[m[key] for m in metrics['by_scene'].values() if m[key] is not None]
        if values:assert abs(sum(values)/len(values)-actual)<1e-12
        else:assert actual is None
    lookup[row['condition'],row['role'],row['seed']]=metrics
for c in result['comparisons']:
    if c['mean_gain_pp'] is None:continue
    values=[100*(lookup[c['condition'],c['left'],s]['macro'][c['metric']]-
                 lookup[c['condition'],c['right'],s]['macro'][c['metric']]) for s in SEEDS]
    assert np.max(np.abs(np.asarray(values)-c['paired_seed_gains_pp']))<1e-10
    assert abs(sum(values)/3-c['mean_gain_pp'])<1e-10
assert result['new_fits']==0 and result['fresh_cohort_now_used'] and not result['goal_complete']
destination=OUT/'actual_review1/execution_checks.json'
assert not destination.exists()
write(destination,dict(passed=True,completed_unix=time.time(),script_sha256=sha(__file__),
    evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),result_sha256=sha(OUT/'result.json'),
    source_and_provenance_files_checked=len(lock['source_sha256']),download_files_checked=len(downloads),
    available_scenes=9,planned_scenes=10,observed_pairs=pairs,technical_exclusions=len(data['excluded']),
    new_fits=0,prediction_groups=36,comparisons=18,first_access_after_freeze_and_preflight=True,
    all_exports_after_cache_completion=True,all_validation_replays_exact=True,
    independent_metric_max_error=result['independent_metric_max_error'],goal_complete=False))
print('CONFIRMATION_EXECUTION_REVIEW_PASSED',pairs,flush=True)
