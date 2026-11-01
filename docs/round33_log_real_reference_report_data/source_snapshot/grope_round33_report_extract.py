"""Post-acceptance R33 report extraction: stdlib JSON/hash only, no NPZ decode."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import time

ROOT = Path('/home/liwt/GRoPE')
OUT = ROOT/'docs/round33_log_real_reference_report_data'
P = ROOT/'results/round33_log_real_reference'
A = ROOT/'results/round33_log_real_reference_independent_audit'
RECEIPT = ROOT/'results/round33_log_real_root_acceptance_20261101.json'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    with Path(p).open('x') as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
def verify_files(files, base=ROOT):
    for name, digest in files.items():
        assert sha(base/name) == digest, str(base/name)
def inventory(directory):
    c = read(directory/'completion.json')
    verify_files(c['files'], directory)
    assert set(c['files']) == {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p != directory/'completion.json'}
    return len(c['files'])+1
def csv_write(name, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (OUT/name).open('x', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields);w.writeheader()
        for r in records:
            w.writerow({k:json.dumps(v, separators=(',',':'),ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def value_max(records,key):
    values=[(r[key],r) for r in records if isinstance(r.get(key),(int,float)) and not isinstance(r[key],bool)]
    if not values:return None
    value,row=max(values,key=lambda t:t[0])
    return dict(value=value,id=row.get('id'),order=row.get('order'),new_order=row.get('new_order'))

started=time.time()
receipt=read(RECEIPT)
assert receipt['accepted_for_failure_interpretation'] and receipt['delivery_integrity_passed']
assert receipt['numerical_preflight_passed'] is False and receipt['labels_read'] is False
verify_files(receipt['files'])
inventories={str(d.relative_to(ROOT)):inventory(d) for d in (P,A,P/'prepared',P/'forward',A/'input_audit')}
for name,digest in receipt['source_hashes'].items():
    assert sha(ROOT/name)==digest
    for d in (P,A):assert sha(d/'source_snapshot'/name)==digest
histories=['round33_contract_development_history_20261101.json','round33_preflight_development_history_20261101.json',
           'round33_audit_development_history_20261101_v2.json','round33_joint_preflight_development_history_20261101.json']
for name in histories:verify_files(read(ROOT/'results'/name)['files'])
capture=read(ROOT/'results/round33_controllers/capture.json');verify_files(capture['files'])

rows=read(P/'forward/records.json');checks=read(A/'checks.json');comps=read(A/'comparisons.json')
specs=read(P/'query_specs.json');inputs=read(A/'input_audit/checks.json')
assert len(rows)==144 and len(checks)==144 and len(comps)==216 and len(specs)==72 and len(inputs)==72
assert [r['call_id'] for r in rows]==[r['call_id'] for r in checks]
assert sum(r['attempted'] for r in rows)==20 and sum(r['passed'] for r in rows)==19
assert sum(r.get('not_attempted_reason')=='fail_fast' for r in rows)==124
assert sum(not c['passed'] for c in comps)==188
assert all(c['delivery_integrity_passed'] for c in checks) and all(c['passed'] for c in inputs)
assert sum(r['charged_cell_node_evaluations'] for r in rows)==receipt['reference_cell_node_evaluations']

call_csv=[]
for r,ch in zip(rows,checks):
    out={k:r.get(k) for k in ('id','scene','pair','pair_name','query_size','query_index','query_bounds','order','call_id','call_index','status','attempted','passed','failure_stage','not_attempted_reason','failure_reason','real_inputs_read','independent_real_reference','budget_used_before','budget_used_after','charged_cell_node_evaluations','reference_work_consistent','wall_seconds','reference_reported_wall_seconds','peak_rss_kib_process','structural_zero','front_mass','log_front_mass','log_raw_mass_sum','log_total_area_floor','normalization_error','direct_fov_absolute_error_materialized_diagnostic','scaled_outer_error_estimate','outer_convergence_success','normal_logcdf_scalar_evaluations','archive_path','archive_sha256')}
    out.update(delivery_integrity_passed=ch['delivery_integrity_passed'],arithmetic_passed=ch['arithmetic_passed'],
               available_arrays=list(r['array_sha256']),stop_trigger=r['stop_trigger'],
               depth_edge_coalescing=r.get('depth_edge_coalescing'),truncation_bounds=r.get('truncation_bounds'),
               lse_diagnostics=r.get('lse_diagnostics'),outer_exp_diagnostics=r.get('outer_exp_diagnostics'))
    out.update({'work_'+k:v for k,v in r['work'].items()})
    call_csv.append(out)
csv_write('reference_calls144.csv',call_csv)

comp_csv=[]
for i,c in enumerate(comps):
    out=dict(comparison_index=i,**{k:v for k,v in c.items() if not k.startswith('per_cell_')})
    per=c.get('per_cell_relative_error');out['maximum_individual_cell_relative_error_diagnostic']=max(per) if per is not None else None
    out['numeric_result_available']='maximum_probability_error' in c
    comp_csv.append(out)
csv_write('comparisons216.csv',comp_csv)

case_csv=[]
for spec,inp in zip(specs,inputs):
    selected=[r for r in rows if r['id']==spec['id']];cs=[c for c in comps if c['id']==spec['id']]
    assert len(selected)==2 and len(cs)==3 and inp['id']==spec['id']
    out={k:spec[k] for k in ('id','scene','pair','pair_name','query_size','query_index','query_bounds')}
    out.update(input_audit_passed=inp['passed'],actual_reference_calls=sum(r['attempted'] for r in selected),
        completed_reference_calls=sum(r['passed'] for r in selected),fail_fast_skipped_calls=sum(r.get('not_attempted_reason')=='fail_fast' for r in selected),
        both_orders_complete=all(r['passed'] for r in selected),comparison_passes=sum(c['passed'] for c in cs),
        comparison_failures=sum(not c['passed'] for c in cs),all_three_comparisons_passed=all(c['passed'] for c in cs),
        charged_nodes=sum(r['charged_cell_node_evaluations'] for r in selected))
    for r in selected:
        out.update({str(r['order'])+'_'+k:r.get(k) for k in ('status','attempted','passed','failure_reason','failure_stage','not_attempted_reason','wall_seconds','archive_sha256')})
    case_csv.append(out)
csv_write('cases72.csv',case_csv)

success=[r for r in rows if r['passed']];attempted=[r for r in rows if r['attempted']]
stats=dict(receipt_sha256=sha(RECEIPT),receipt_file_bindings=len(receipt['files']),source_count=len(receipt['source_hashes']),
    inventories=inventories,row_counts=dict(cases72=72,reference_calls144=144,comparisons216=216),
    case_outcomes=dict(both_orders_complete=sum(c['both_orders_complete'] for c in case_csv),
        one_order_complete=sum(c['completed_reference_calls']==1 for c in case_csv),neither_order_attempted=sum(c['actual_reference_calls']==0 for c in case_csv)),
    comparison_failure_reasons=dict(collections.Counter(c.get('failure_reason') for c in comps if not c['passed'])),
    comparisons={},scene_size=[],coalescing=[dict(id=r['id'],order=r['order'],**r['depth_edge_coalescing']) for r in attempted],
    successful_call_maxima={k:value_max(success,k) for k in ('normalization_error','direct_fov_absolute_error_materialized_diagnostic','scaled_outer_error_estimate')},
    successful_log_raw_native_range=[min(r['log_raw_mass_sum'] for r in success),max(r['log_raw_mass_sum'] for r in success)],
    sum_all_row_wall_seconds=math.fsum(r['wall_seconds'] for r in rows),
    sum_success_reference_reported_wall_seconds=math.fsum(r['reference_reported_wall_seconds'] for r in success),
    attempted_work_totals={k:sum(r['work'][k] for r in attempted) for k in attempted[0]['work']},
    attempted_work_maxima={k:max(r['work'][k] for r in attempted) for k in attempted[0]['work']},
    successful_tail_maxima={k:max(r['truncation_bounds'][k] for r in success) for k in ('raw_l1_over_reference_fov','probability_max_absolute','depth_omitted_mass')},
    outer_underflow_events_attempted=sum(r['outer_exp_diagnostics']['total_underflow_terms'] for r in attempted),
    outer_underflow_calls_attempted=sum(r['outer_exp_diagnostics']['total_underflow_terms']>0 for r in attempted),
    lse_attempted={scope:dict(underflow_term_count=sum(r['lse_diagnostics']['by_scope'][scope]['underflow_term_count'] for r in attempted),
        max_log_relative_omitted_bound=max([r['lse_diagnostics']['by_scope'][scope]['maximum_log_relative_omitted_bound'] for r in attempted if r['lse_diagnostics']['by_scope'][scope]['maximum_log_relative_omitted_bound'] is not None],default=None)) for scope in attempted[0]['lse_diagnostics']['by_scope']},
    source_jsons={str(p.relative_to(ROOT)):sha(p) for p in [RECEIPT,P/'forward/records.json',A/'checks.json',A/'comparisons.json',A/'input_audit/checks.json',P/'query_specs.json']})
for kind in ('new_orders','old_production_vs_new'):
    selected=[c for c in comps if c['kind']==kind];known=[c for c in selected if 'maximum_probability_error' in c]
    stats['comparisons'][kind]=dict(scheduled=len(selected),numeric_result_count=len(known),passes=sum(c['passed'] for c in selected),
        failures=sum(not c['passed'] for c in selected),argmax_changes=sum(c['argmax_changed'] for c in known),
        exact_tie_comparisons=sum(max(len(c['actual_exact_maximizers']),len(c['reference_exact_maximizers']))>1 for c in known),
        near_tie_comparisons=sum(max(len(c['actual_approximate_maximizers']),len(c['reference_approximate_maximizers']))>1 for c in known),
        maxima={k:value_max(known,k) for k in ('maximum_probability_error','raw_fov_relative','raw_l1_over_reference_fov')})
for scene in dict.fromkeys(s['scene'] for s in specs):
    for size in (8,16,24):
        selected=[r for r in rows if r['scene']==scene and r['query_size']==size]
        stats['scene_size'].append(dict(scene=scene,query_size=size,scheduled=len(selected),attempted=sum(r['attempted'] for r in selected),complete=sum(r['passed'] for r in selected),actual_failure=sum(r['attempted'] and not r['passed'] for r in selected),skipped=sum(not r['attempted'] for r in selected)))
write(OUT/'statistics.json',stats)
outputs={name:sha(OUT/name) for name in ('cases72.csv','reference_calls144.csv','comparisons216.csv','statistics.json')}
write(OUT/'completion.json',dict(outputs=outputs,counts=stats['row_counts'],receipt_sha256=sha(RECEIPT),generator_sha256=sha(__file__),
    started_unix=started,finished_unix=time.time(),npz_decodes=0,integrator_calls=0,labels_read=False))
print(json.dumps(dict(outputs=outputs,receipt_sha256=sha(RECEIPT),inventories=inventories,comparison=stats['comparisons']),indent=2))
