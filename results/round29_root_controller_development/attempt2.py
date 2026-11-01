import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import numpy as np
import sys,time,math,json
from pathlib import Path
root=Path('/home/liwt/GRoPE');sys.path.insert(0,str(root))
from experiments import round29_log_rms_audit as A,round29_log_rms_contract as C,round29_log_rms_fixtures as F
prod=A.DEFAULT_RUN;audit=A.DEFAULT_OUT;path=root/'results/round29_log_rms_root_acceptance_20261031.json'
if path.exists():raise RuntimeError('preserve prior root acceptance')
lock,seal,records,fixtures,banks=A.verify_seal(prod,audit)
pc=A.verify_inventory(prod);ac=A.verify_inventory(audit)
pre=root/'results/round29_log_joint_preflight_20261031.json';pr=C.read(pre)
execution=root/'results/round29_log_rms_execution_20261031.json';exe=C.read(execution)
assert pr['passed'] and pr['metadata_only_freeze_verified'] and pr['source_hashes']==lock['source_hashes'] and pr['runtime']==lock['runtime']
assert pr['finished_unix']<=lock['created_unix'] and pr['tests_log_sha256']==C.sha(pr['tests_log_path'])
assert exe['joint_preflight_sha256']==C.sha(pre) and exe['all_stages_completed']
assert [r['name'] for r in exe['stages']]==['producer_freeze','audit_freeze','producer_run','independent_audit']
assert all(r['exit_code']==0 and r['started_unix']<=r['finished_unix'] for r in exe['stages'])
assert all(a['finished_unix']<=b['started_unix'] for a,b in zip(exe['stages'],exe['stages'][1:]))
assert exe['pipeline_log_sha256']==C.sha(root/'results/round29_log_rms_pipeline_20261031.log')
assert exe['stages'][2]['started_unix']<=seal['started_unix']<=seal['finished_unix']<=exe['stages'][2]['finished_unix']
production=C.read(audit/'production_checks.json');refs=C.read(audit/'reference_records.json');comparisons=C.read(audit/'comparisons.json')
identities=C.read(audit/'identities.json');limit=C.read(audit/'near_constant_limit.json');summary=C.read(audit/'summary.json');verification=C.read(audit/'verification.json')
assert [r['id'] for r in production]==lock['base_ids']+lock['identity_ids']
assert [r['id'] for r in comparisons]==lock['base_ids'] and [r['id'] for r in identities]==lock['identity_ids']
fmap={f['id']:f for f in fixtures};pmap={r['id']:r for r in production}
for row,saved in zip(records,production):
    policy=dict(C.POLICY,**fmap[row['id']]['policy_overrides'])
    try:
        expected=A.production_check(row,banks[row['id']],policy,C.GATES)
        if not row['passed']:expected['partial_delivery']=A.partial_check(row,banks[row['id']],policy)
    except Exception as error:
        expected=dict(passed=False,status='saved_output_failure',failure_reason=type(error).__name__+': '+str(error))
    assert dict(id=row['id'],**expected,**A.SCOPE)==saved
refbanks={};maxima={}
for row in comparisons:
    n=row['id'];p=audit/row['archive_path'];assert C.sha(p)==row['archive_sha256']
    analytic=n in F.ANALYTIC_BASE_IDS
    case_references=[r for r in refs if r['id']==n]
    expected_pass=bool(all(r['passed'] for r in case_references) and len(case_references)==(1 if analytic else 2)
        and (analytic or row.get('reference',{}).get('passed',False)) and row.get('production',{}).get('passed',False))
    assert row['passed']==expected_pass
    with np.load(p,allow_pickle=False) as z:refbanks[n]={k:z[k] for k in z.files}
    assert {k:C.array_sha(v) for k,v in refbanks[n].items()}==row['array_sha256']
    b=banks[n];zero=A.structural_branch(b['input__mu'],b['input__factor'])[1]
    args=(b['input__bounds'],b['input__content_bias'],dict(C.POLICY,**fmap[n]['policy_overrides'])['floor_density'],C.GATES)
    analytic=n in F.ANALYTIC_BASE_IDS;high='analytic' if analytic else '48'
    expected_keys={'log_raw_mass__analytic'} if analytic else {'log_raw_mass__32','log_raw_mass__48'}
    if not expected_keys<=set(refbanks[n]):
        assert row['passed'] is False
        continue
    if not analytic and 'reference' in row:
        expected=A.compare_log_raw(refbanks[n]['log_raw_mass__32'],refbanks[n]['log_raw_mass__48'],*args,reference_comparison=True,structural_zero=zero)
        assert row['reference']==expected
    elif analytic:assert 'reference' not in row
    else:
        assert row['passed'] is False and (row.get('failure_reason') or any(not r['passed'] for r in refs if r['id']==n))
    if 'production' in row:
        assert pmap[n]['passed']
        expected=A.compare_log_raw(b['log_raw_mass'],refbanks[n]['log_raw_mass__'+high],*args,archived_probability=b['probability'],structural_zero=zero)
        assert row['production']==expected
    else:
        assert row['passed'] is False and (not pmap[n]['passed'] or row.get('failure_reason') or any(not r['passed'] for r in refs if r['id']==n))
    for group in ('reference','production'):
        for metric in ('maximum_probability_error','raw_fov_relative','raw_l1_over_reference_fov'):
            value=row.get(group,{}).get(metric)
            if value is not None:
                key=group+'__'+metric
                if key not in maxima or value>maxima[key]['value']:maxima[key]=dict(value=value,id=n)
expected_refs=[(n,o) for n in F.OLD_BASE_IDS for o in C.REFERENCE_POLICY['orders']]+[(n,None) for n in F.ANALYTIC_BASE_IDS]
assert [(r['id'],r['order']) for r in refs]==expected_refs
starts=[json.loads(line) for line in (audit/'reference_starts.jsonl').read_text().splitlines()]
journal=[json.loads(line) for line in (audit/'reference_results.jsonl').read_text().splitlines()]
assert journal==refs and [(r['id'],r['order']) for r in starts]==[(r['id'],r['order']) for r in refs if r['attempted']]
cumulative=cdf=0;failures=[]
for row in refs:
    n=row['id'];b=banks[n];rb=refbanks[n];analytic=n in F.ANALYTIC_BASE_IDS;suffix='__analytic' if analytic else '__'+str(row['order'])
    arrays={k[:-len(suffix)]:v for k,v in rb.items() if k.endswith(suffix)}
    assert {k:C.array_sha(v) for k,v in arrays.items()}==row['available_array_sha256']
    if row['passed']:
        policy=dict(C.POLICY,**fmap[n]['policy_overrides'])
        result=dict(row,**arrays)
        check=A.analytic_check(result,b,policy,C.GATES) if analytic else A.reference_check(result,b['input__mu'],b['input__factor'],b['input__bounds'],b['input__content_bias'],policy,C.REFERENCE_POLICY,C.GATES,row['order'])
        assert all(row[k]==v for k,v in check.items())
        if not analytic:
            zero=A.structural_branch(b['input__mu'],b['input__factor'])[1]
            expected=np.full(arrays['raw_mass'].shape,-np.inf) if zero else np.log(arrays['raw_mass'])
            assert np.array_equal(arrays['log_raw_mass'],expected)
            assert float(arrays['log_raw_fov_mass'][0])==(-np.inf if zero else math.log(row['raw_fov_mass']))
    else:failures.append(dict(id=n,order=row['order'],reason=row.get('failure_reason')))
    cumulative+=row['work']['cell_node_evaluations'];cdf+=row['analytic_cdf_evaluations']
    assert cumulative==row['cumulative_cell_node_evaluations'] and cdf==row['cumulative_analytic_cdf_evaluations']
for row in identities:
    f=fmap[row['id']];parent=f['identity']['parent']
    try:
        if not pmap[row['id']]['passed'] or not pmap[parent]['passed']:raise ValueError('identity/base production failed')
        expected=A.identity_check(f,banks[row['id']],banks[parent],C.GATES)
    except Exception as error:
        expected=dict(id=row['id'],parent=parent,passed=False,failure_reason=type(error).__name__+': '+str(error))
    assert row==dict(expected,**A.SCOPE)
assert limit==dict(A.near_constant_check(banks,pmap,C.GATES),**A.SCOPE)
passed=bool(all(r['passed'] for r in production+refs+comparisons+identities) and limit['passed'] and seal['phase_wall_passed'] and summary['reference_budget_passed'])
assert passed==summary['numerical_preflight_passed']==verification['numerical_preflight_passed']==ac['numerical_preflight_passed']
assert ac['delivery_integrity_passed'] and verification['delivery_integrity_passed']
assert summary['reference_cell_node_evaluations']==cumulative and summary['analytic_cdf_evaluations']==cdf
phase=summary['reference_phase_wall_seconds']
assert math.isfinite(phase) and phase>=0
assert summary['reference_budget_passed']==bool(phase<=C.REFERENCE_POLICY['max_phase_wall_seconds'] and cumulative<=C.REFERENCE_POLICY['max_total_cell_node_evaluations'])
assert summary['reference_call_count']==43 and summary['numerical_reference_call_count']==40 and summary['analytic_reference_call_count']==3
for name,rows in (('production_failure_count',production),('reference_call_failure_count',refs),('base_comparison_failure_count',comparisons),('identity_failure_count',identities)):
    assert summary[name]==sum(not r['passed'] for r in rows)
assert summary['started_unix']>=seal['finished_unix'] and summary['phase_finished_unix']<=summary['finished_unix']<=ac['finished_unix']<=exe['finished_unix']
stage=exe['stages'][3]
assert C.read(audit/'audit_started.json')['started_unix']==summary['started_unix']
assert stage['started_unix']<=summary['started_unix']<=summary['finished_unix']<=verification['finished_unix']<=ac['finished_unix']<=stage['finished_unix']
assert all(summary['started_unix']<=r['started_unix']<=summary['phase_finished_unix'] for r in starts)
assert all(a['started_unix']<=b['started_unix'] for a,b in zip(starts,starts[1:]))
expected_bindings=dict(producer_lock_sha256=C.sha(prod/'lock.json'),audit_lock_sha256=C.sha(audit/'lock.json'),
    producer_completion_sha256=C.sha(prod/'completion.json'),producer_forward_complete_sha256=C.sha(prod/'forward_complete.json'),
    summary_sha256=C.sha(audit/'summary.json'),source_count=len(lock['source_hashes']))
assert all(verification[k]==v for k,v in expected_bindings.items())
assert ac['verification_sha256']==C.sha(audit/'verification.json')
for item in (summary,verification,ac,lock,C.read(audit/'lock.json'),seal,*production,*refs,*comparisons,*identities,limit):
    assert all(item[k] is False for k in A.SCOPE)
oldfixtures=C.read(root/'results/round27_rms_preflight/fixtures.json');oldrecords={r['id']:r for r in C.read(root/'results/round27_rms_preflight/records.json')}
assert [f for f in fixtures if f['id'] not in F.ANALYTIC_BASE_IDS]==oldfixtures
for f in oldfixtures:
    n=f['id'];assert all(C.array_sha(banks[n]['input__'+k])==oldrecords[n]['array_sha256']['input__'+k] for k in ('mu','factor','bounds','content_bias'))
files=[pre,Path(pr['tests_log_path']),execution,root/'results/round29_log_rms_pipeline_20261031.log']
files += [prod/n for n in ('lock.json','lock.sha256','fixtures.json','started.json','records.json','call_starts.jsonl','model_results.jsonl','forward_complete.json','completion.json')]
files += [audit/n for n in ('lock.json','lock.sha256','audit_started.json','production_checks.json','reference_starts.jsonl','reference_results.jsonl','reference_records.json','comparisons.json','identities.json','near_constant_limit.json','summary.json','verification.json','completion.json')]
record=dict(created_unix=time.time(),accepted_for_numerical_interpretation=True,numerical_preflight_passed=passed,**A.SCOPE,goal_remains_active=True,
    source_count=len(lock['source_hashes']),source_hashes=lock['source_hashes'],all_current_and_dual_snapshot_sources_verified=True,
    producer_file_count=len(pc['files'])+1,audit_file_count=len(ac['files'])+1,producer_calls=29,producer_completed=sum(r['passed'] for r in records),
    reference_status_count=43,actual_reference_calls=sum(r['attempted'] for r in refs),reference_failures=len(failures),
    old_numerical_reference_calls=40,new_analytic_reference_calls=3,old26_descriptors_and_input_bytes_identical=True,
    production_phase_wall_seconds=seal['phase_wall_seconds'],reference_phase_wall_seconds=summary['reference_phase_wall_seconds'],
    production_cell_node_evaluations=sum(r['work'].get('cell_node_evaluations',0) for r in records),reference_cell_node_evaluations=cumulative,
    analytic_cdf_evaluations=cdf,failures=failures,maxima=maxima,
    files={str(p.relative_to(root)):C.sha(p) for p in files},
    decision='accept_fixed_log_representation_synthetic_numerics_only' if passed else 'retain_log_representation_numerical_failure_no_promotion',
    notes=['No production or reference integration rerun by root; saved raw/log/P and comparisons rechecked.',
    'Canonical logs preserve positive mass independently of optional ordinary materializations.',
    'R28 remains failed; passing this population is not a real-input or general-domain guarantee.',
    'All prior scientific gates, labels restriction and fresh-data exclusion remain unchanged.'])
C.write(path,record)
print(json.dumps({k:v for k,v in record.items() if k not in ('files','source_hashes')},indent=2))
