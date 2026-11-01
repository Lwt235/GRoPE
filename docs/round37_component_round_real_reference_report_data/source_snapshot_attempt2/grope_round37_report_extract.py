"""Accepted R37 evidence extraction: stdlib JSON and byte hashes only."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math,time
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'docs/round37_component_round_real_reference_report_data'
P=ROOT/'results/round37_component_round_real_reference';A=ROOT/'results/round37_component_round_real_reference_independent_audit'
RECEIPT=ROOT/'results/round37_component_round_real_root_acceptance_20261101.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def verify(files,base=ROOT):
    for n,h in files.items():assert sha(base/n)==h,str(base/n)
def inventory(d):
    files=read(d/'completion.json')['files'];verify(files,d)
    assert set(files)=={str(p.relative_to(d)) for p in d.rglob('*') if p.is_file() and p!=d/'completion.json'}
    return len(files)+1
def csv_write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,separators=(',',':'),allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def maximum(values):return max((v for v in values if type(v) in (int,float)),default=None)
def vstat(rows,key):
    found=[(r[key],r) for r in rows if type(r.get(key)) in (int,float)]
    if not found:return None
    value,r=max(found,key=lambda v:v[0]);return dict(value=value,**{k:r.get(k) for k in ('id','order','new_order','old_order','old_reference_kind')})
def group(c):return 'new_vs_old_'+c['old_reference_kind'] if c['kind']=='new_vs_old_reference' else c['kind']
parser=argparse.ArgumentParser();parser.add_argument('--receipt-sha',required=True);args=parser.parse_args();started=time.time()
assert sha(RECEIPT)==args.receipt_sha;receipt=read(RECEIPT)
assert receipt['accepted_for_numerical_interpretation'] and receipt['delivery_integrity_passed'] and not receipt['labels_read']
verify(receipt['files']);inventories={str(d.relative_to(ROOT)):inventory(d) for d in (P,A,P/'prepared',P/'forward',A/'input_audit')}
for n,h in receipt['source_hashes'].items():
    assert sha(ROOT/n)==h
    for d in (P,A):assert sha(d/'source_snapshot'/n)==h
histories=sorted((ROOT/'results').glob('round37*history*.json'))
for p in histories:verify(read(p)['files'])
capture=read(ROOT/'results/round37_controllers/before_execution.json')
assert capture['all_three_controllers_captured_before_any_execution']
for name,digest in capture['files'].items():assert sha(ROOT/'results/round37_controllers'/Path(name).name)==digest
inventories['results/round37_controllers']=inventory(ROOT/'results/round37_controllers')
rows=read(P/'forward/records.json');checks=read(A/'checks.json');comps=read(A/'comparisons.json')
specs=read(P/'query_specs.json');inputs=read(A/'input_audit/checks.json');oracles=read(P/'oracle_specs.json')
summary=read(A/'summary.json');seal=read(P/'forward_complete.json');inputreceipt=read(A/'input_verification.json')
assert tuple(map(len,(specs,rows,checks,comps,inputs,oracles)))==(72,144,144,296,72,40)
assert len({(o['id'],o['oracle_id']) for o in oracles})==40
assert dict(collections.Counter(o['reference_kind'] for o in oracles))=={'r33_log_conditional':19,'r35_log_real':21}
assert all(o['reference_record']['attempted'] and o['reference_record']['passed'] for o in oracles)
assert all(c['passed'] for c in inputs) and all(c['delivery_integrity_passed'] for c in checks)
assert [r['call_id'] for r in rows]==[c['call_id'] for c in checks]
assert sum(r['charged_cell_node_evaluations'] for r in rows)==receipt['reference_cell_node_evaluations']==seal['total_cell_node_evaluations']
assert summary['numerical_preflight_passed']==receipt['numerical_preflight_passed']
expected={'new_orders':72,'old_production_vs_new':144,'new_vs_old_r33_log_conditional':38,'new_vs_old_r35_log_real':42}
assert dict(collections.Counter(group(c) for c in comps))==expected
assert all(c[k] is False for c in (receipt,summary) for k in ('labels_read','general_numerical_accuracy_established','efficacy_tested','candidate_selected','scientific_gate_passed','goal_complete'))
if receipt['numerical_preflight_passed']:assert all(r['passed'] for r in rows) and all(c['passed'] for c in comps)
callfields=('id','scene','pair','pair_name','query_size','query_index','query_bounds','order','call_id','call_index','status','attempted','passed','failure_stage','failure_reason','not_attempted_reason','real_inputs_read','independent_real_reference','charged_cell_node_evaluations','budget_used_before','budget_used_after','reference_work_consistent','wall_seconds','reference_reported_wall_seconds','peak_rss_kib_process','structural_branch','structural_zero','front_mass','log_front_mass','log_raw_mass_sum','log_total_area_floor','normalization_error','direct_fov_absolute_error_materialized_diagnostic','scaled_outer_error_estimate','outer_convergence_success','normal_logcdf_scalar_evaluations','archive_path','archive_sha256','stop_trigger','depth_edge_coalescing','truncation_bounds','lse_diagnostics','outer_exp_diagnostics','outer_component_roundoff','narrow_normal_quadrature_bound')
calls=[];traces=[]
for r,ch in zip(rows,checks):
    item={k:r.get(k) for k in callfields};item.update(delivery_integrity_passed=ch['delivery_integrity_passed'],arithmetic_passed=ch['arithmetic_passed'],available_arrays=list(r['array_sha256']),independent_component_roundoff_check=ch.get('outer_component_roundoff'))
    item.update({'work_'+k:v for k,v in r['work'].items()});calls.append(item)
    tr=ch.get('outer_component_roundoff');meta=r.get('outer_component_roundoff')
    if tr and tr.get('applicable'):
        target=tr['target'];item=dict(id=r['id'],order=r['order'],reference_passed=r['passed'],trace_complete=meta['complete'],trace_status=meta['status'],pending_interval_bounds=meta['pending_interval_bounds'],**tr)
        for key in ('estimated_error','maximum_historical_roundoff','sum_active_truncation_error'):item[key+'_over_target']=tr[key]/target if target else None
        traces.append(item)
csv_write('reference_calls144.csv',calls);csv_write('outer_trace_summaries.csv',traces)
comparison_rows=[]
for i,c in enumerate(comps):
    item=dict(comparison_index=i,report_group=group(c),**{k:v for k,v in c.items() if not k.startswith('per_cell_')})
    item['maximum_individual_cell_relative_error_diagnostic']=maximum(c.get('per_cell_relative_error',[]))
    item['numeric_result_available']='maximum_probability_error' in c;comparison_rows.append(item)
csv_write('comparisons296.csv',comparison_rows)
csv_write('oracles40.csv',[{k:o[k] for k in ('id','oracle_id','order','scene','pair','pair_name','query_size','query_index','query_bounds','reference_kind','source_archive_path','source_archive_sha256','source_fields','array_sha256')} for o in oracles])
cases=[]
for s,inp in zip(specs,inputs):
    rr=[r for r in rows if r['id']==s['id']];cc=[c for c in comps if c['id']==s['id']]
    assert len(rr)==2 and inp['id']==s['id'] and len(cc)==3+2*sum(o['id']==s['id'] for o in oracles)
    item={k:s[k] for k in ('id','scene','pair','pair_name','query_size','query_index','query_bounds')}
    item.update(input_audit_passed=inp['passed'],actual_reference_calls=sum(r['attempted'] for r in rr),completed_reference_calls=sum(r['passed'] for r in rr),both_orders_complete=all(r['passed'] for r in rr),fail_fast_skipped_calls=sum(r.get('not_attempted_reason')=='fail_fast' for r in rr),comparison_passes=sum(c['passed'] for c in cc),comparison_failures=sum(not c['passed'] for c in cc),charged_nodes=sum(r['charged_cell_node_evaluations'] for r in rr))
    for r in rr:item.update({str(r['order'])+'_'+k:r.get(k) for k in ('status','attempted','passed','failure_reason','failure_stage','not_attempted_reason','wall_seconds','archive_sha256')})
    cases.append(item)
csv_write('cases72.csv',cases)
oldcost=[]
for o in oracles:
    nr=next(r for r in rows if (r['id'],r['order'])==(o['id'],o['order']));old=o['reference_record']
    oldcost.append(dict(id=o['id'],order=o['order'],old_reference_kind=o['reference_kind'],old_oracle_id=o['oracle_id'],new_completed=nr['passed'],old_nodes=old['charged_cell_node_evaluations'],new_nodes=nr['charged_cell_node_evaluations'],node_delta=nr['charged_cell_node_evaluations']-old['charged_cell_node_evaluations'] if nr['passed'] else None,old_reference_reported_seconds=old.get('reference_reported_wall_seconds'),new_reference_reported_seconds=nr.get('reference_reported_wall_seconds')))
csv_write('matched_historical_costs40.csv',oldcost)
attempted=[r for r in rows if r['attempted']];success=[r for r in rows if r['passed']]
valid=[c['arithmetic'] for c in checks if c.get('arithmetic',{}).get('status')=='checked']
workkeys=sorted(set(k for r in rows for k in r['work']));scopes=sorted(set(k for r in attempted for k in r.get('lse_diagnostics',{}).get('by_scope',{})))
stats=dict(receipt_sha256=sha(RECEIPT),receipt_file_bindings=len(receipt['files']),source_count=len(receipt['source_hashes']),inventories=inventories,summary=summary,stop_trigger=seal['stop_trigger'],input_audit_finished_unix=inputreceipt['finished_unix'],forward_started=read(P/'forward/started.json'),
    row_counts=dict(cases72=72,reference_calls144=144,comparisons296=296,oracles40=40,outer_trace_summaries=len(traces)),
    case_outcomes=dict(both_orders_complete=sum(c['both_orders_complete'] for c in cases),one_order_complete=sum(c['completed_reference_calls']==1 for c in cases),neither_order_attempted=sum(c['actual_reference_calls']==0 for c in cases)),
    comparison_failure_reasons=dict(collections.Counter(c.get('failure_reason') for c in comps if not c['passed'])),comparisons={},scene_size=[],
    work_totals={k:sum(r['work'].get(k,0) for r in rows) for k in workkeys},work_maxima={k:maximum(r['work'].get(k) for r in rows) for k in workkeys},
    successful_call_maxima={k:vstat(success,k) for k in ('normalization_error','direct_fov_absolute_error_materialized_diagnostic','scaled_outer_error_estimate')},
    attempted_call_maxima={k:vstat(attempted,k) for k in ('charged_cell_node_evaluations','wall_seconds','peak_rss_kib_process')},
    successful_log_raw_native_range=[min(r['log_raw_mass_sum'] for r in success),max(r['log_raw_mass_sum'] for r in success)] if success else None,
    arithmetic_maxima={k:maximum(c['arithmetic_errors'][k] for c in valid) for k in ('log_floored_mass','log_probability','probability')},
    maximum_scale_restoration=maximum(c['scale_restoration']['scale_restoration_error'] for c in valid),
    successful_tail_maxima={k:maximum(r.get('truncation_bounds',{}).get(k) for r in success) for k in ('raw_l1_over_reference_fov','probability_max_absolute','depth_omitted_mass')},
    coalescing=[dict(id=r['id'],order=r['order'],**r['depth_edge_coalescing']) for r in attempted if r.get('depth_edge_coalescing')],
    outer_underflow_events=sum(r.get('outer_exp_diagnostics',{}).get('total_underflow_terms',0) for r in attempted),
    lse={s:dict(underflow_term_count=sum(r.get('lse_diagnostics',{}).get('by_scope',{}).get(s,{}).get('underflow_term_count',0) for r in attempted),max_log_relative_omitted_bound=maximum(r.get('lse_diagnostics',{}).get('by_scope',{}).get(s,{}).get('maximum_log_relative_omitted_bound') for r in attempted)) for s in scopes},
    outer_trace=dict(call_count=len(traces),complete_count=sum(t['trace_complete'] for t in traces),incomplete_count=sum(not t['trace_complete'] for t in traces),converged_count=sum(t['strict_convergence'] for t in traces),
        totals={k:sum(t[k] for t in traces) for k in ('initial_interval_count','completed_interval_evaluations','active_interval_count','split_count','callback_evaluations')},
        initial_partition_only_completed_calls=sum(t['reference_passed'] and t['split_count']==0 for t in traces),
        maxima={k:vstat(traces,k) for k in ('estimated_error_over_target','maximum_historical_roundoff_over_target','sum_active_truncation_error_over_target','split_count','completed_interval_evaluations')},
        failed_trace=[t for t in traces if not t['reference_passed']]),
    sum_row_wall_seconds=math.fsum(r['wall_seconds'] for r in rows),sum_success_reference_reported_seconds=math.fsum(r.get('reference_reported_wall_seconds',0.) for r in success),
    attempted_reference_reported_seconds=math.fsum(r.get('reference_reported_wall_seconds',0.) for r in attempted),
    reference_reported_seconds_present_attempts=sum('reference_reported_wall_seconds' in r for r in attempted),
    failures=[{k:r.get(k) for k in callfields+('work',)}|{'available_arrays':list(r['array_sha256'])} for r in rows if r['attempted'] and not r['passed']],
    histories={str(p.relative_to(ROOT)):sha(p) for p in histories},
    matched_historical_costs=oldcost)
for name in expected:
    cc=[c for c in comps if group(c)==name];known=[c for c in cc if 'maximum_probability_error' in c]
    stats['comparisons'][name]=dict(scheduled=len(cc),numeric_count=len(known),passes=sum(c['passed'] for c in cc),failures=sum(not c['passed'] for c in cc),argmax_changes=sum(c.get('argmax_changed',False) for c in known),exact_tie_comparisons=sum(max(len(c['actual_exact_maximizers']),len(c['reference_exact_maximizers']))>1 for c in known),near_tie_comparisons=sum(max(len(c['actual_approximate_maximizers']),len(c['reference_approximate_maximizers']))>1 for c in known),maxima={k:vstat(known,k) for k in ('maximum_probability_error','raw_fov_relative','raw_l1_over_reference_fov')})
for scene in dict.fromkeys(s['scene'] for s in specs):
    for size in (8,16,24):
        rr=[r for r in rows if r['scene']==scene and r['query_size']==size]
        stats['scene_size'].append(dict(scene=scene,query_size=size,scheduled=len(rr),attempted=sum(r['attempted'] for r in rr),complete=sum(r['passed'] for r in rr),actual_failure=sum(r['attempted'] and not r['passed'] for r in rr),skipped=sum(not r['attempted'] for r in rr)))
write(OUT/'statistics.json',stats)
names=('cases72.csv','reference_calls144.csv','comparisons296.csv','oracles40.csv','outer_trace_summaries.csv','matched_historical_costs40.csv','statistics.json')
files={n:sha(OUT/n) for n in names}
write(OUT/'completion.json',dict(outputs=files,counts=stats['row_counts'],receipt_sha256=sha(RECEIPT),generator_sha256=sha(__file__),started_unix=started,finished_unix=time.time(),npz_decodes=0,integrator_calls=0,real_inputs_read=False,labels_read=False))
print(json.dumps(dict(outputs=files,inventories=inventories,summary=summary,comparisons=stats['comparisons'],outer_trace=stats['outer_trace'],failures=[{k:r.get(k) for k in ('id','order','failure_reason','charged_cell_node_evaluations','wall_seconds')} for r in stats['failures']]),indent=2))
