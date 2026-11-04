"""R34 post-acceptance report: stdlib sealed JSON/hash only; no NPZ decode."""
from pathlib import Path
import collections, csv, hashlib, json, math, time

ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'docs/round34_log_narrow_reference_report_data'
P=ROOT/'results/round34_log_narrow_preflight'
A=ROOT/'results/round34_log_narrow_independent_audit'
RECEIPT=ROOT/'results/round34_log_narrow_root_acceptance_20261101.json'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f: json.dump(x,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
def verify_files(files,base=ROOT):
    for name,digest in files.items(): assert sha(base/name)==digest,str(base/name)
def inventory(d):
    files=read(d/'completion.json')['files'];verify_files(files,d)
    assert set(files)=={str(p.relative_to(d)) for p in d.rglob('*') if p.is_file() and p!=d/'completion.json'}
    return len(files)+1
def csv_write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def vmax(rows,key):
    items=[(r[key],r) for r in rows if type(r.get(key)) in (int,float)]
    if not items:return None
    v,r=max(items,key=lambda t:t[0]);return dict(value=v,**{k:r.get(k) for k in ('id','order','new_order','old_order','old_reference_kind')})

started=time.time();receipt=read(RECEIPT)
assert sha(RECEIPT)=='edeffc6490aecadebba70b4a773ecb2d2fb150911e78ad7fe4c131f8050870b1'
assert receipt['accepted_for_numerical_interpretation'] and receipt['delivery_integrity_passed'] and receipt['numerical_preflight_passed']
verify_files(receipt['files'])
inventories={str(d.relative_to(ROOT)):inventory(d) for d in (P,A,P/'prepared',P/'forward',A/'input_audit',ROOT/'results/round34_controllers')}
for name,digest in receipt['source_hashes'].items():
    assert sha(ROOT/name)==digest
    for d in (P,A):assert sha(d/'source_snapshot'/name)==digest
histories=sorted((ROOT/'results').glob('round34*history*.json'))
for p in histories:verify_files(read(p)['files'])
rows=read(P/'forward/records.json');checks=read(A/'checks.json');comps=read(A/'comparisons.json')
specs=read(P/'query_specs.json');inputs=read(A/'input_audit/checks.json');oracles=read(P/'oracle_specs.json')
assert tuple(map(len,(specs,rows,checks,comps,inputs,oracles)))==(23,46,46,247,23,89)
assert all(r['attempted'] and r['passed'] for r in rows) and all(c['passed'] for c in comps)
assert all(c['delivery_integrity_passed'] and c['arithmetic_passed'] for c in checks)
assert all(c['passed'] and c['canonical_input_bytes_exact'] and c['descriptor_bytes_exact'] for c in inputs)
assert [r['call_id'] for r in rows]==[r['call_id'] for r in checks]
assert sum(r['charged_cell_node_evaluations'] for r in rows)==379773536
assert len({(r['id'],r['oracle_id']) for r in oracles})==89
old={(o['id'],o['order']):o['reference_record'] for o in oracles if o['reference_kind']=='r32_log_conditional'}
assert len(old)==46
for r in rows:
    assert all(r['array_sha256'][k]==next(s for s in specs if s['id']==r['id'])['array_sha256'][k] for k in ('input__mu','input__factor','input__bounds','input__content_bias'))
    assert r['work']['narrow_y_pdf_nodes']==8*r['lse_diagnostics']['by_scope']['narrow_normal_nodes']['sum_count']

calls=[]
for r,ch in zip(rows,checks):
    o=old[r['id'],r['order']]
    out={k:r.get(k) for k in ('id','order','call_id','status','attempted','passed','structural_branch','structural_zero','front_mass','log_front_mass','log_raw_mass_sum','log_total_area_floor','normalization_error','direct_fov_absolute_error_materialized_diagnostic','scaled_outer_error_estimate','normal_logcdf_scalar_evaluations','wall_seconds','reference_reported_wall_seconds','peak_rss_kib_process','charged_cell_node_evaluations','budget_used_before','budget_used_after','archive_path','archive_sha256','depth_edge_coalescing','truncation_bounds','narrow_normal_quadrature_bound','lse_diagnostics','outer_exp_diagnostics')}
    out.update(delivery_integrity_passed=ch['delivery_integrity_passed'],arithmetic_passed=ch['arithmetic_passed'],r32_charged_nodes=o['charged_cell_node_evaluations'],node_delta_vs_r32=r['charged_cell_node_evaluations']-o['charged_cell_node_evaluations'])
    for k,v in r['work'].items():out['work_'+k]=v;out['r32_work_'+k]=o['work'][k]
    calls.append(out)
csv_write('reference_calls46.csv',calls)
cc=[]
for i,c in enumerate(comps):
    out=dict(comparison_index=i,**{k:v for k,v in c.items() if not k.startswith('per_cell_')})
    out['maximum_individual_cell_relative_error_diagnostic']=max((v for v in c.get('per_cell_relative_error',[]) if v is not None),default=None);cc.append(out)
csv_write('comparisons247.csv',cc)
cases=[]
for s,inp in zip(specs,inputs):
    rr=[r for r in rows if r['id']==s['id']];cs=[c for c in comps if c['id']==s['id']]
    out=dict(id=s['id'],cell_count=len(s['fixture']['bounds']),bias_kind=s['fixture'].get('bias_kind'),input_passed=inp['passed'],calls=len(rr),completed=sum(r['passed'] for r in rr),comparisons=len(cs),comparison_passes=sum(c['passed'] for c in cs),charged_nodes=sum(r['charged_cell_node_evaluations'] for r in rr),r32_charged_nodes=sum(old[r['id'],r['order']]['charged_cell_node_evaluations'] for r in rr),max_P_error=max(c['maximum_probability_error'] for c in cs),argmax_change_records=sum(c['argmax_changed'] for c in cs),structural_zero=rr[0]['structural_zero'])
    for r in rr:
        for k in ('log_raw_mass_sum','front_mass','log_front_mass','log_total_area_floor','depth_edge_coalescing','wall_seconds'):out[str(r['order'])+'_'+k]=r.get(k)
    cases.append(out)
csv_write('cases23.csv',cases)
csv_write('oracles89.csv',[{k:o[k] for k in ('id','oracle_id','order','reference_kind','source_archive_path','source_archive_sha256','source_fields','array_sha256')} for o in oracles])

stats=dict(receipt_sha256=sha(RECEIPT),receipt_file_bindings=len(receipt['files']),source_count=len(receipt['source_hashes']),inventories=inventories,
    row_counts=dict(cases23=23,reference_calls46=46,comparisons247=247,oracles89=89),comparisons={},
    work_totals={k:sum(r['work'][k] for r in rows) for k in rows[0]['work']},r32_work_totals={k:sum(r['work'][k] for r in old.values()) for k in rows[0]['work']},
    work_maxima={k:max(r['work'][k] for r in rows) for k in rows[0]['work']},
    calls_changing_nonnarrow_work=[r['call_id'] for r in rows if any(r['work'][k]!=old[r['id'],r['order']]['work'][k] for k in r['work'] if k not in ('narrow_y_pdf_nodes','elementary_inner_nodes','cell_node_evaluations'))],
    calls_not_exactly_halving_narrow_nodes=[r['call_id'] for r in rows if r['work']['narrow_y_pdf_nodes']*2!=old[r['id'],r['order']]['work']['narrow_y_pdf_nodes']],
    call_maxima={k:vmax(rows,k) for k in ('charged_cell_node_evaluations','normalization_error','direct_fov_absolute_error_materialized_diagnostic','scaled_outer_error_estimate','wall_seconds')},
    arithmetic_maxima={k:max(c['arithmetic']['arithmetic_errors'][k] for c in checks) for k in checks[0]['arithmetic']['arithmetic_errors']},
    maximum_scale_restore=max(c['arithmetic']['scale_restoration']['scale_restoration_error'] for c in checks),
    tail_maxima={k:max(r['truncation_bounds'][k] or 0 for r in rows) for k in ('raw_l1_over_reference_fov','probability_max_absolute','depth_omitted_mass')},
    coalescing=[dict(id=r['id'],order=r['order'],**r['depth_edge_coalescing']) for r in rows if r.get('depth_edge_coalescing')],
    argmax_changes=[{k:v for k,v in c.items() if not k.startswith('per_cell_')} for c in comps if c['argmax_changed']],
    outer_underflow_events=sum(r['outer_exp_diagnostics']['total_underflow_terms'] for r in rows),outer_underflow_calls=sum(r['outer_exp_diagnostics']['total_underflow_terms']>0 for r in rows),
    lse={scope:dict(underflow_term_count=sum(r['lse_diagnostics']['by_scope'][scope]['underflow_term_count'] for r in rows),max_log_relative_omitted_bound=max([r['lse_diagnostics']['by_scope'][scope]['maximum_log_relative_omitted_bound'] for r in rows if r['lse_diagnostics']['by_scope'][scope]['maximum_log_relative_omitted_bound'] is not None],default=None)) for scope in rows[0]['lse_diagnostics']['by_scope']},
    sum_call_wall_seconds=math.fsum(r['wall_seconds'] for r in rows),sum_reference_reported_seconds=math.fsum(r['reference_reported_wall_seconds'] for r in rows),
    histories={str(p.relative_to(ROOT)):sha(p) for p in histories})
for group in ('new_orders','new_vs_old_r29','new_vs_old_r32','old_production_vs_new'):
    selected=[c for c in comps if c['kind']==group or (c['kind']=='new_vs_old_reference' and group==('new_vs_old_r32' if c['old_reference_kind']=='r32_log_conditional' else 'new_vs_old_r29'))]
    stats['comparisons'][group]=dict(count=len(selected),passed=sum(c['passed'] for c in selected),argmax_changes=sum(c['argmax_changed'] for c in selected),exact_tie_records=sum(max(len(c['actual_exact_maximizers']),len(c['reference_exact_maximizers']))>1 for c in selected),near_tie_records=sum(max(len(c['actual_approximate_maximizers']),len(c['reference_approximate_maximizers']))>1 for c in selected),maxima={k:vmax(selected,k) for k in ('maximum_probability_error','raw_fov_relative','raw_l1_over_reference_fov')})
write(OUT/'statistics.json',stats)
files={name:sha(OUT/name) for name in ('cases23.csv','reference_calls46.csv','comparisons247.csv','oracles89.csv','statistics.json')}
write(OUT/'completion.json',dict(outputs=files,counts=stats['row_counts'],receipt_sha256=sha(RECEIPT),generator_sha256=sha(__file__),started_unix=started,finished_unix=time.time(),npz_decodes=0,integrator_calls=0,real_inputs_read=False,labels_read=False))
print(json.dumps(dict(outputs=files,inventories=inventories,comparisons=stats['comparisons'],work=stats['work_totals'],old_work=stats['r32_work_totals']),indent=2))
