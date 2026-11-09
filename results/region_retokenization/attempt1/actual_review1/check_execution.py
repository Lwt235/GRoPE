"""Post-run identity and aggregation review; no fitting or new prediction."""
import time
import numpy as np
from experiments.region_retokenization import OUT, ROLES, SEEDS, CONDITIONS, check
from experiments.round65_common import read, write, sha

lock=check()
state=read(OUT/'status.json')
preflight=read(OUT/'preflight.json')
result=read(OUT/'result.json')
assert state['stage']=='complete' and state['active'] is None
assert len(state['completed'])==12 and preflight['passed']
assert all(r['maximum_logit_error']==0 and r['prediction_changes']==0 for r in preflight['records'])
assert result['lock_sha256']==state['lock_sha256']==preflight['lock_sha256']==sha(OUT/'lock.json')
expected={(c,r,s) for c in CONDITIONS for r in ROLES for s in SEEDS}
assert {(r['condition'],r['role'],r['seed']) for r in result['records']}==expected
assert len(result['records'])==36 and sum(r['reused'] for r in result['records'])==24
assert [c['bootstrap_seed'] for c in result['comparisons']]==list(range(88001,88019))
identities=[]
for entry in state['completed']:
    chosen=next(x for x in lock['models'] if (x['role'],x['seed'])==(entry['role'],entry['seed']))
    assert entry['checkpoint_sha256']==chosen['checkpoint_sha256']
    assert entry['completed_unix']>preflight['completed_unix']>lock['created_unix']
    assert sha(entry['path'])==entry['sha256']
    assert entry['cache_sha256']==lock['provenance_sha256'][str(OUT.parents[1]/'round60_real_scene/attempt1/data/eval_sparse256.npz')]
    assert entry['peak_allocated_bytes']<=3*2**30
    identities.append((entry['role'],entry['seed']))
assert len(set(identities))==12
for r in result['records']:
    assert sha(r['path'])==r['sha256']
    with np.load(r['path'],allow_pickle=False) as z:
        assert len(z['ids'])==len(set(z['ids']))==480
        assert z['logits'].shape==(480,64,64) and np.isfinite(z['logits']).all()
    m=r['metrics']
    assert sum(v['total_queries'] for v in m['by_scene'].values())==m['micro']['total_queries']==61440
    assert sum(v['valid_queries'] for v in m['by_scene'].values())==m['micro']['valid_queries']
    for metric,value in m['macro'].items():
        known=[v[metric] for v in m['by_scene'].values() if v[metric] is not None]
        assert abs(sum(known)/len(known)-value)<1e-12
lookup={(r['condition'],r['role'],r['seed']):r for r in result['records']}
for c in result['comparisons']:
    paired=[]
    for seed in SEEDS:
        a,b=[lookup[c['condition'],r,seed]['metrics']['macro'][c['metric']] for r in (c['left'],c['right'])]
        paired.append(100*(a-b))
    assert np.max(np.abs(np.asarray(paired)-c['paired_seed_gains_pp']))<1e-10
    assert abs(sum(paired)/3-c['mean_gain_pp'])<1e-10
assert result['new_fits']==0 and not result['fresh_confirmation'] and not result['official_test_access']
assert result['candidate']=='relative5' and not result['goal_complete']
record=dict(passed=True,completed_unix=time.time(),script_sha256=sha(__file__),
            lock_sha256=sha(OUT/'lock.json'),result_sha256=sha(OUT/'result.json'),
            source_files_checked=len(lock['source_sha256']),provenance_files_checked=len(lock['provenance_sha256']),
            selected_models=12,prediction_groups=36,comparisons=18,
            exports_after_all_validation_replays=True,unchanged_models_and_sources=True,
            independent_metric_max_error=result['independent_metric_max_error'],
            goal_complete=False)
path=OUT/'actual_review1/execution_checks.json'
assert not path.exists()
write(path,record)
print(record)
