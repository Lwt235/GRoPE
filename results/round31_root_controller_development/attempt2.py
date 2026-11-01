import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import sys,time,json,collections,math
from pathlib import Path
root=Path('/home/liwt/GRoPE');sys.path.insert(0,str(root))
from experiments import round31_log_conditional_audit as A,round31_log_conditional_contract as C
run=A.DEFAULT_RUN;out=A.DEFAULT_OUT;destination=root/'results/round31_log_conditional_root_acceptance_20261031.json'
if destination.exists():raise RuntimeError('preserve prior root acceptance')
lock,seal,rows,preprows,specs=A.verify_forward(run,out)
pc=A.complete_inventory(run);ac=A.complete_inventory(out)
prepath=root/'results/round31_log_joint_preflight_20261031.json';pre=C.read(prepath)
exepath=root/'results/round31_log_conditional_execution_20261031.json';exe=C.read(exepath)
assert pre['passed'] and pre['metadata_only_freeze_verified'] and pre['source_hashes']==lock['source_hashes']
assert pre['runtime']==lock['runtime'] and pre['finished_unix']<=lock['created_unix']
assert pre['tests_log_sha256']==C.sha(pre['tests_log_path']) and pre['formal_reference_calls']==0
assert exe['joint_preflight_sha256']==C.sha(prepath) and exe['all_stages_completed']
assert [s['name'] for s in exe['stages']]==['producer_freeze','audit_freeze','prepare_inputs','independent_input_audit','reference_run','independent_output_audit']
assert all(s['exit_code']==0 and s['started_unix']<=s['finished_unix'] for s in exe['stages'])
assert all(a['finished_unix']<=b['started_unix'] for a,b in zip(exe['stages'],exe['stages'][1:]))
assert exe['pipeline_log_sha256']==C.sha(root/'results/round31_log_conditional_pipeline_20261031.log')
assert exe['synthetic_inputs_read'] is True and exe['independent_synthetic_reference']==any(r['attempted'] for r in rows)
assert all(exe[k] is False for k in A.SCOPE)
inputchecks=C.read(out/'input_audit/checks.json');inputreceipt=C.read(out/'input_verification.json')
checks=C.read(out/'checks.json');comparisons=C.read(out/'comparisons.json')
summary=C.read(out/'summary.json');verification=C.read(out/'verification.json')
assert [c['id'] for c in inputchecks]==lock['case_ids']
assert [c['call_id'] for c in checks]==lock['ordered_call_ids']
prepared={};new={};old={};production={};specmap={s['id']:s for s in specs};checked={}
for spec,prep,check in zip(specs,preprows,inputchecks):
    source=A.read_arrays(spec['source_archive_path'],spec['array_sha256'],fields=C.OLD_FIELDS)
    bank=A.read_arrays(run/prep['archive_path'],prep['array_sha256']);prepared[spec['id']]=bank
    expected=A.input_check(source,bank,spec)
    assert expected['passed'] and all(check[k]==v for k,v in expected.items())
    assert check['prepared_archive_sha256']==prep['archive_sha256']
failures=[];maxima={};work=0
for row,check in zip(rows,checks):
    key=(row['id'],row['order']);spec=specmap[row['id']]
    bank=A.read_arrays(run/row['archive_path'],row['array_sha256']);new[key]=bank
    expected=A.validate_call(row,bank,prepared[row['id']],lock['reference_policy'])
    assert all(check[k]==v for k,v in expected.items()) and expected['delivery_integrity_passed']
    policy=dict(lock['policy'],**spec['policy_overrides'])
    arithmetic=A.reference_check(row,bank,policy,lock['reference_policy'],lock['gates'])
    assert check['arithmetic']==arithmetic and check['arithmetic_passed']==arithmetic['passed']
    checked[key]=check;work+=row['charged_cell_node_evaluations']
    if not row['passed']:failures.append(dict(id=row['id'],order=row['order'],attempted=row['attempted'],reason=row['failure_reason'],work=row['work']))
    if arithmetic['status']=='checked':
        for name,value in [('normalization_error',arithmetic['normalization_error']),('scale_restoration_error',arithmetic['scale_restoration']['scale_restoration_error'])]:
            if name not in maxima or value>maxima[name]['value']:maxima[name]=dict(value=value,id=row['id'],order=row['order'])
oracles=C.read(run/'oracle_specs.json')
for spec in oracles:old[(spec['id'],spec['order'])]=A.read_oracle(spec)
for spec in specs:
    wanted=('log_raw_mass','log_raw_fov_mass','probability')
    production[spec['id']]=A.read_arrays(spec['source_archive_path'],{n:spec['production_array_sha256'][n] for n in wanted},wanted)
assert C.read(out/'old_output_delivery_failures.json')==[]
plan=A.comparison_plan(specs,oracles);assert len(plan)==len(comparisons)==155
comparison_maxima={}
for declaration,saved in zip(plan,comparisons):
    description={k:v for k,v in declaration.items() if k not in ('actual_key','reference_key')}
    assert all(saved[k]==v for k,v in description.items())
    akey,rkey=declaration['actual_key'],declaration['reference_key'];kind=declaration['kind']
    try:
        actual=production[akey] if kind=='old_production_vs_new' else new[akey]
        ref=old[rkey] if kind=='new_vs_old_reference' else new[rkey]
        policy=dict(lock['policy'],**specmap[declaration['id']]['policy_overrides'])
        expected=A.compare_saved(actual,ref,prepared[declaration['id']],policy,lock['gates'],kind!='old_production_vs_new')
        keys=[akey] if kind=='new_vs_old_reference' else ([rkey] if kind=='old_production_vs_new' else [akey,rkey])
        expected['all_new_call_checks_passed']=all(checked[k]['arithmetic_passed'] and checked[k]['delivery_integrity_passed'] for k in keys)
        expected['passed']=bool(expected['passed'] and expected['all_new_call_checks_passed'])
    except Exception as error:expected=dict(passed=False,failure_reason=type(error).__name__+': '+str(error))
    assert all(saved[k]==v for k,v in expected.items())
    for metric in ('maximum_probability_error','log_raw_fov_relative','log_raw_l1_over_reference_fov'):
        value=expected.get(metric)
        if value is None:continue
        group=comparison_maxima.setdefault(kind,{})
        if metric not in group or value>group[metric]['value']:group[metric]=dict(value=value,**description)
expected_counts=dict(new_orders=23,new_vs_old_reference=86,old_production_vs_new=46)
assert dict(collections.Counter(c['kind'] for c in comparisons))==expected_counts
inherited=C.read(out/'inherited_checks.json')
assert inherited['identities']==C.read(C.OLD_AUDIT/'identities.json') and inherited['near_constant']==C.read(C.OLD_AUDIT/'near_constant_limit.json')
assert inherited['identities_sha256']==C.sha(C.OLD_AUDIT/'identities.json') and inherited['near_constant_sha256']==C.sha(C.OLD_AUDIT/'near_constant_limit.json')
assert inherited['new_identity_calls']==inherited['new_near_constant_calls']==0 and inherited['inherited_only'] is True
passed=bool(seal['reference_calls_completed'] and all(c['arithmetic_passed'] for c in checks) and all(c['passed'] for c in comparisons))
assert summary['delivery_integrity_passed'] and verification['delivery_integrity_passed'] and ac['delivery_integrity_passed']
assert passed==summary['numerical_preflight_passed']==verification['numerical_preflight_passed']==ac['numerical_preflight_passed']
assert summary['reference_cell_node_evaluations']==seal['total_cell_node_evaluations']==work
assert summary['reference_completed_calls']==sum(r['passed'] for r in rows)
assert summary['actual_reference_calls']==sum(r['attempted'] for r in rows) and summary['calls_not_attempted']==sum(not r['attempted'] for r in rows)
assert summary['comparison_count']==155 and summary['comparison_failures']==sum(not c['passed'] for c in comparisons)
assert summary['reference_check_failures']==sum(not c['arithmetic_passed'] for c in checks)
assert summary['reference_integrations_executed_by_auditor']==verification['reference_integrations_executed_by_auditor']==0
assert all(item[k] is False for item in [summary,verification,ac,inputreceipt,*checks,*inputchecks,*comparisons] for k in A.SCOPE)
assert inputreceipt['all_input_bytes_exact'] and all(c['canonical_input_bytes_exact'] and c['descriptor_bytes_exact'] for c in inputchecks)
assert all(item['synthetic_inputs_read'] is True for item in (summary,verification,ac,inputreceipt))
assert all(item['independent_synthetic_reference']==any(r['attempted'] for r in rows) for item in (summary,verification,ac))
assert summary['reference_phase_wall_seconds']==seal['phase_wall_seconds']
assert summary['phase_wall_passed']==seal['phase_wall_passed'] and summary['phase_node_budget_passed']==seal['phase_node_budget_passed']
assert C.read(out/'audit_started.json')['started_unix']==summary['started_unix']
assert seal['finished_unix']<=summary['started_unix']<=summary['finished_unix']<=verification['finished_unix']<=ac['finished_unix']<=exe['finished_unix']
prepseal=C.read(run/'prepared_complete.json');alock=C.read(out/'lock.json')
for i,start,finish in ((0,lock['created_unix'],lock['created_unix']),(1,alock['created_unix'],alock['created_unix']),(2,prepseal['started_unix'],prepseal['finished_unix']),(3,inputreceipt['started_unix'],C.read(out/'input_audit/completion.json')['finished_unix']),(4,seal['started_unix'],pc['finished_unix']),(5,summary['started_unix'],ac['finished_unix'])):
    assert exe['stages'][i]['started_unix']<=start<=finish<=exe['stages'][i]['finished_unix']
for key,path in dict(producer_lock_sha256=run/'lock.json',audit_lock_sha256=out/'lock.json',producer_completion_sha256=run/'completion.json',producer_forward_complete_sha256=run/'forward_complete.json',input_verification_sha256=out/'input_verification.json',summary_sha256=out/'summary.json',checks_sha256=out/'checks.json',comparisons_sha256=out/'comparisons.json').items():assert verification[key]==C.sha(path)
assert ac['verification_sha256']==C.sha(out/'verification.json')
files=[prepath,Path(pre['tests_log_path']),exepath,root/'results/round31_log_conditional_pipeline_20261031.log']
files += [run/n for n in ('lock.json','lock.sha256','query_specs.json','oracle_specs.json','prepared_complete.json','prepared/started.json','prepared/records.json','prepared/completion.json','forward/started.json','forward/records.json','forward/call_starts.jsonl','forward/reference_results.jsonl','forward/completion.json','forward_complete.json','completion.json')]
files += [out/n for n in ('lock.json','lock.sha256','input_audit/started.json','input_audit/checks.json','input_audit/completion.json','input_verification.json','audit_started.json','checks.json','comparisons.json','old_output_delivery_failures.json','inherited_checks.json','summary.json','verification.json','completion.json')]
record=dict(created_unix=time.time(),accepted_for_numerical_interpretation=True,numerical_preflight_passed=passed,
    **A.SCOPE,goal_remains_active=True,synthetic_inputs_read=True,independent_synthetic_reference=any(r['attempted'] for r in rows),
    fixed_base_count=23,scheduled_reference_calls=46,actual_reference_calls=sum(r['attempted'] for r in rows),
    completed_reference_calls=sum(r['passed'] for r in rows),new_production_calls=0,root_reference_integrations=0,
    source_count=len(lock['source_hashes']),source_hashes=lock['source_hashes'],
    all_current_and_dual_snapshot_sources_verified=True,all_prepared_inputs_bitwise_equal_to_R29=True,
    independent_input_audit_before_all_reference_calls=True,producer_file_count=len(pc['files'])+1,audit_file_count=len(ac['files'])+1,
    reference_phase_wall_seconds=seal['phase_wall_seconds'],reference_cell_node_evaluations=work,
    maximum_process_rss_kib=max(r['peak_rss_kib_process'] for r in rows),comparison_counts=expected_counts,
    comparison_failures=sum(not c['passed'] for c in comparisons),reference_check_failures=sum(not c['arithmetic_passed'] for c in checks),
    failure_reason_counts=dict(collections.Counter(f['reason'] for f in failures)),failures=failures,maxima=maxima,comparison_maxima=comparison_maxima,
    files={str(p.relative_to(root)):C.sha(p) for p in files},
    decision='accept_fixed_synthetic_reference_preflight_only' if passed else 'retain_fixed_synthetic_reference_failure_no_promotion',
    notes=['No model/reference integration replayed by root;23inputs,46saved arithmetic outcomes and155comparison records rechecked.',
    'Both new orders compared against every accepted old reference, plus unchanged R29 production.',
    'Six identities and finite-epsilon evidence are inherited; no new identity calls.',
    'No real-array decode,labels,efficacy or general numerical certificate; all old scientific gates remain mandatory.'])
C.write(destination,record)
print(json.dumps({k:v for k,v in record.items() if k not in ('source_hashes','files','failures')},indent=2))
