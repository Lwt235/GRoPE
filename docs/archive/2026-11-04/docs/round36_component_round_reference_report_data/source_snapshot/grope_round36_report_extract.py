"""R36 post-acceptance report: stdlib sealed JSON/hash only; never decode NPZ."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math,time
ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'docs/round36_component_round_reference_report_data'
P=ROOT/'results/round36_component_round_preflight'
A=ROOT/'results/round36_component_round_independent_audit'
RECEIPT=ROOT/'results/round36_component_round_root_acceptance_20261101.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def verify_files(files,base=ROOT):
    for n,h in files.items():assert sha(base/n)==h,str(base/n)
def inventory(d):
    files=read(d/'completion.json')['files'];verify_files(files,d)
    assert set(files)=={str(p.relative_to(d)) for p in d.rglob('*') if p.is_file() and p!=d/'completion.json'}
    return len(files)+1
def csv_write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,separators=(',',':'),allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def vmax(rows,key):
    candidates=[(r[key],r) for r in rows if type(r.get(key)) in (int,float)]
    if not candidates:return None
    value,row=max(candidates,key=lambda item:item[0])
    return dict(value=value,**{k:row.get(k) for k in ('id','order','new_order','old_order','old_reference_kind')})
def maximum(values):return max((v for v in values if type(v) in (int,float)),default=None)
def group_name(c):
    if c['kind']!='new_vs_old_reference':return c['kind']
    return {'r32_log_conditional':'new_vs_old_r32','r34_log_narrow':'new_vs_old_r34'}.get(c['old_reference_kind'],'new_vs_old_r29')

args=argparse.ArgumentParser();args.add_argument('--receipt-sha',required=True);args=args.parse_args()
started=time.time();assert sha(RECEIPT)==args.receipt_sha
receipt=read(RECEIPT)
assert receipt['accepted_for_numerical_interpretation'] and receipt['delivery_integrity_passed']
verify_files(receipt['files'])
inventories={str(d.relative_to(ROOT)):inventory(d) for d in (P,A,P/'prepared',P/'forward',A/'input_audit',ROOT/'results/round36_controllers')}
for name,digest in receipt['source_hashes'].items():
    assert sha(ROOT/name)==digest
    for d in (P,A):assert sha(d/'source_snapshot'/name)==digest
histories=sorted((ROOT/'results').glob('round36*history*.json'))
for path in histories:
    h=read(path)
    if 'files' in h:verify_files(h['files'])
rows=read(P/'forward/records.json');checks=read(A/'checks.json');comps=read(A/'comparisons.json')
specs=read(P/'query_specs.json');inputs=read(A/'input_audit/checks.json');oracles=read(P/'oracle_specs.json')
summary=read(A/'summary.json');seal=read(P/'forward_complete.json')
assert tuple(map(len,(specs,rows,checks,comps,inputs,oracles)))==(23,46,46,339,23,135)
assert len({(o['id'],o['oracle_id']) for o in oracles})==135
assert all(c['passed'] and c['canonical_input_bytes_exact'] and c['descriptor_bytes_exact'] for c in inputs)
assert [r['call_id'] for r in rows]==[r['call_id'] for r in checks]
assert sum(r['charged_cell_node_evaluations'] for r in rows)==receipt['reference_cell_node_evaluations']==seal['total_cell_node_evaluations']
assert summary['numerical_preflight_passed']==receipt['numerical_preflight_passed']
assert dict(collections.Counter(group_name(c) for c in comps))==dict(new_orders=23,new_vs_old_r29=86,new_vs_old_r32=92,new_vs_old_r34=92,old_production_vs_new=46)
if receipt['numerical_preflight_passed']:
    assert all(r['attempted'] and r['passed'] for r in rows) and all(c['passed'] for c in comps)
    assert all(c['delivery_integrity_passed'] and c['arithmetic_passed'] for c in checks)
old={(o['id'],o['order']):o['reference_record'] for o in oracles if o['reference_kind']=='r34_log_narrow'}
assert len(old)==46
specmap={s['id']:s for s in specs}
for r in rows:
    if r['attempted']:
        assert all(r['array_sha256'][k]==specmap[r['id']]['array_sha256'][k] for k in ('input__mu','input__factor','input__bounds','input__content_bias'))
    if r['passed']:
        assert r['work']['narrow_y_pdf_nodes']==8*r['lse_diagnostics']['by_scope']['narrow_normal_nodes']['sum_count']

calls=[];traces=[]
fields=('id','order','call_id','status','attempted','passed','failure_reason','structural_branch','structural_zero','front_mass','log_front_mass','log_raw_mass_sum','log_total_area_floor','normalization_error','direct_fov_absolute_error_materialized_diagnostic','scaled_outer_error_estimate','normal_logcdf_scalar_evaluations','wall_seconds','reference_reported_wall_seconds','peak_rss_kib_process','charged_cell_node_evaluations','budget_used_before','budget_used_after','archive_path','archive_sha256','depth_edge_coalescing','truncation_bounds','narrow_normal_quadrature_bound','lse_diagnostics','outer_exp_diagnostics','outer_component_roundoff')
for r,ch in zip(rows,checks):
    o=old[r['id'],r['order']];item={k:r.get(k) for k in fields}
    item.update(delivery_integrity_passed=ch['delivery_integrity_passed'],arithmetic_passed=ch['arithmetic_passed'],r34_charged_nodes=o['charged_cell_node_evaluations'],node_delta_vs_r34=r['charged_cell_node_evaluations']-o['charged_cell_node_evaluations'])
    for k in set(r['work'])|set(o['work']):item['work_'+k]=r['work'].get(k);item['r34_work_'+k]=o['work'].get(k)
    trace=ch.get('outer_component_roundoff')
    item['independent_component_roundoff_check']=trace
    if trace and trace.get('applicable'):
        tr=dict(id=r['id'],order=r['order'],reference_passed=r['passed'],**trace)
        target=trace.get('target')
        tr['estimated_error_over_target']=trace['estimated_error']/target if target else None
        tr['historical_roundoff_over_target']=trace['maximum_historical_roundoff']/target if target else None
        tr['active_truncation_over_target']=trace['sum_active_truncation_error']/target if target else None
        traces.append(tr)
    calls.append(item)
csv_write('reference_calls46.csv',calls)
cc=[]
for i,c in enumerate(comps):
    item=dict(comparison_index=i,report_group=group_name(c),**{k:v for k,v in c.items() if not k.startswith('per_cell_')})
    item['maximum_individual_cell_relative_error_diagnostic']=maximum(c.get('per_cell_relative_error',[]));cc.append(item)
csv_write('comparisons339.csv',cc)
cases=[]
for s,inp in zip(specs,inputs):
    rr=[r for r in rows if r['id']==s['id']];cs=[c for c in comps if c['id']==s['id']]
    item=dict(id=s['id'],cell_count=len(s['fixture']['bounds']),input_passed=inp['passed'],calls=len(rr),attempted=sum(r['attempted'] for r in rr),completed=sum(r['passed'] for r in rr),comparisons=len(cs),comparison_passes=sum(c['passed'] for c in cs),charged_nodes=sum(r['charged_cell_node_evaluations'] for r in rr),r34_charged_nodes=sum(old[r['id'],r['order']]['charged_cell_node_evaluations'] for r in rr),max_P_error=maximum(c.get('maximum_probability_error') for c in cs),argmax_change_records=sum(c.get('argmax_changed',False) for c in cs))
    for r in rr:
        for k in ('passed','failure_reason','structural_zero','log_raw_mass_sum','front_mass','log_front_mass','log_total_area_floor','depth_edge_coalescing','wall_seconds'):item[str(r['order'])+'_'+k]=r.get(k)
    cases.append(item)
csv_write('cases23.csv',cases)
csv_write('oracles135.csv',[{k:o[k] for k in ('id','oracle_id','order','reference_kind','source_archive_path','source_archive_sha256','source_fields','array_sha256')} for o in oracles])
csv_write('outer_trace_summaries.csv',traces)
valid=[c['arithmetic'] for c in checks if c.get('arithmetic',{}).get('status')=='checked']
workkeys=sorted(set(k for r in rows for k in r['work']))
scopes=sorted(set(k for r in rows for k in r.get('lse_diagnostics',{}).get('by_scope',{})))
stats=dict(receipt_sha256=sha(RECEIPT),receipt_file_bindings=len(receipt['files']),source_count=len(receipt['source_hashes']),inventories=inventories,summary=summary,
    row_counts=dict(cases23=23,reference_calls46=46,comparisons339=339,oracles135=135,outer_trace_summaries=len(traces)),comparisons={},
    work_totals={k:sum(r['work'].get(k,0) for r in rows) for k in workkeys},r34_work_totals={k:sum(r['work'].get(k,0) for r in old.values()) for k in workkeys},
    work_maxima={k:maximum(r['work'].get(k) for r in rows) for k in workkeys},
    call_maxima={k:vmax(rows,k) for k in ('charged_cell_node_evaluations','normalization_error','direct_fov_absolute_error_materialized_diagnostic','scaled_outer_error_estimate','wall_seconds')},
    arithmetic_maxima={k:maximum(c['arithmetic_errors'][k] for c in valid) for k in ('log_floored_mass','log_probability','probability')},
    maximum_scale_restore=maximum(c['scale_restoration']['scale_restoration_error'] for c in valid),
    tail_maxima={k:maximum(r.get('truncation_bounds',{}).get(k) for r in rows) for k in ('raw_l1_over_reference_fov','probability_max_absolute','depth_omitted_mass')},
    coalescing=[dict(id=r['id'],order=r['order'],**r['depth_edge_coalescing']) for r in rows if r.get('depth_edge_coalescing')],
    argmax_changes=[{k:v for k,v in c.items() if not k.startswith('per_cell_')} for c in comps if c.get('argmax_changed')],
    outer_underflow_events=sum(r.get('outer_exp_diagnostics',{}).get('total_underflow_terms',0) for r in rows),
    lse={s:dict(underflow_term_count=sum(r.get('lse_diagnostics',{}).get('by_scope',{}).get(s,{}).get('underflow_term_count',0) for r in rows),max_log_relative_omitted_bound=maximum(r.get('lse_diagnostics',{}).get('by_scope',{}).get(s,{}).get('maximum_log_relative_omitted_bound') for r in rows)) for s in scopes},
    outer_trace=dict(call_count=len(traces),totals={k:sum(r[k] for r in traces) for k in ('completed_interval_evaluations','initial_interval_count','active_interval_count','split_count','callback_evaluations')},initial_partition_only_count=sum(r['split_count']==0 for r in traces),
        maxima={k:vmax(traces,k) for k in ('estimated_error','target','maximum_historical_roundoff','sum_active_truncation_error','estimated_error_over_target','historical_roundoff_over_target','active_truncation_over_target','split_count','completed_interval_evaluations')}),
    sum_call_wall_seconds=math.fsum(r['wall_seconds'] for r in rows),sum_reference_reported_seconds=math.fsum(r.get('reference_reported_wall_seconds',0.) for r in rows),
    histories={str(path.relative_to(ROOT)):sha(path) for path in histories})
for group in ('new_orders','new_vs_old_r29','new_vs_old_r32','new_vs_old_r34','old_production_vs_new'):
    selected=[c for c in comps if group_name(c)==group];numeric=[c for c in selected if 'maximum_probability_error' in c]
    stats['comparisons'][group]=dict(count=len(selected),numeric_count=len(numeric),passed=sum(c['passed'] for c in selected),argmax_changes=sum(c.get('argmax_changed',False) for c in selected),exact_tie_records=sum(max(len(c['actual_exact_maximizers']),len(c['reference_exact_maximizers']))>1 for c in numeric),near_tie_records=sum(max(len(c['actual_approximate_maximizers']),len(c['reference_approximate_maximizers']))>1 for c in numeric),maxima={k:vmax(numeric,k) for k in ('maximum_probability_error','raw_fov_relative','raw_l1_over_reference_fov')})
write(OUT/'statistics.json',stats)
names=('cases23.csv','reference_calls46.csv','comparisons339.csv','oracles135.csv','outer_trace_summaries.csv','statistics.json')
files={name:sha(OUT/name) for name in names}
write(OUT/'completion.json',dict(outputs=files,counts=stats['row_counts'],receipt_sha256=sha(RECEIPT),generator_sha256=sha(__file__),started_unix=started,finished_unix=time.time(),npz_decodes=0,integrator_calls=0,real_inputs_read=False,labels_read=False))
print(json.dumps(dict(outputs=files,inventories=inventories,comparisons=stats['comparisons'],outer_trace=stats['outer_trace'],work=stats['work_totals'],old_work=stats['r34_work_totals']),indent=2))
