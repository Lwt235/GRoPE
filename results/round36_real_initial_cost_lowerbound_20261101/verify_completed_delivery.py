"""JSON/integer/byte-hash delivery review only; no NumPy or geometry evaluation."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
REPORT=ROOT/'docs/round36_real_initial_cost_lowerbound_report.md'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=read(OUT/'lock.json');completion=read(OUT/'completion.json');diagnostic=read(OUT/'diagnostic.json')
records=read(OUT/'records.json');progress=[json.loads(line) for line in (OUT/'progress.jsonl').read_text().splitlines()]
assert not (OUT/'execution_receipt.json').exists()
assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
assert completion['passed'] and completion['delivery_integrity_passed']
assert sha(OUT/'diagnostic.json')==completion['diagnostic_sha256']
for name,digest in completion['files'].items():assert sha(OUT/name)==digest,name
for field in ('bindings','local_source_hashes'):
    for name,digest in lock[field].items():assert sha(name)==digest,name
for name,digest in lock['local_source_hashes'].items():
    assert sha(OUT/'source_snapshot'/Path(name).relative_to(ROOT))==digest,name
old=read(ROOT/'results/round35_log_real_reference/lock.json')
assert lock['runtime']==old['runtime']
assert len(records)==len(lock['records'])==72 and records==progress
assert [r['id'] for r in records]==lock['case_ids']
assert len(set(lock['case_ids']))==72
false_fields=('labels_read','independent_real_reference','general_numerical_accuracy_established',
              'efficacy_tested','candidate_selected','scientific_gate_passed','goal_complete')
zero_fields=('reference_integrations','model_integrations','Gaussian_pdf_evaluations',
             'Gaussian_cdf_evaluations','quadrature_weighted_sums')
for item in (lock,diagnostic,completion,*records):
    assert all(item[k] is False for k in false_fields)
    assert all(type(item[k]) is int and item[k]==0 for k in zero_fields)
identity=('id','scene','pair','pair_name','query_size','query_index','query_bounds')
groups={};totals={'32':0,'48':0};maximum={}
for index,(row,spec) in enumerate(zip(records,lock['records'])):
    assert all(row[k]==spec[k] for k in identity) and row['case_index']==index
    assert row['status']=='initial_lower_bound_only' and row['full_initial_geometry_counted']
    assert row['failure_reason'] is None and row['real_inputs_read'] and row['front_probability_not_evaluated']
    assert row['branch']=='stochastic_depth'
    assert row['required_callbacks']==row['completed_required_callbacks']==22*row['initial_interval_count']
    assert 0<=row['zero_inner_callbacks']<=row['completed_required_callbacks']
    assert type(row['minimum_panels_prefix']) is int and row['minimum_panels_prefix']>=0
    assert row['minimum_panels_prefix']>=(row['completed_required_callbacks']-row['zero_inner_callbacks'])*433
    assert row['input_archive_path']==str(ROOT/'results/round35_log_real_reference'/spec['archive_path'])
    assert row['input_archive_sha256']==spec['archive_sha256']==sha(row['input_archive_path'])
    assert sha(OUT/row['geometry_archive_path'])==row['geometry_archive_sha256']
    assert set(row['geometry_array_sha256'])=={'initial_depth_edges','original_depth_edges',
        'dropped_depth_edge_indices','dropped_depth_edge_values','pilot_zeta'}
    for order in ('32','48'):
        value=row['orders'][order]
        assert value['minimum_x_nodes_prefix']==int(order)*row['minimum_panels_prefix']
        assert value['guaranteed_initial_node_budget_failure']==(value['minimum_x_nodes_prefix']>250_000_000)
        assert not value['guaranteed_initial_node_budget_failure'] and value['absence_of_counterexample_is_unknown']
        totals[order]+=value['minimum_x_nodes_prefix']
        if order not in maximum or value['minimum_x_nodes_prefix']>maximum[order]['minimum_x_nodes']:
            maximum[order]=dict(id=row['id'],minimum_x_nodes=value['minimum_x_nodes_prefix'])
    groups.setdefault((row['scene'],row['query_size']),[]).append(row)
assert len(groups)==9 and all(len(x)==8 for x in groups.values())
assert diagnostic['status_counts']=={'initial_lower_bound_only':72}
assert diagnostic['full_geometry_case_count']==diagnostic['case_count']==72
assert diagnostic['scheduled_order_count']==144 and diagnostic['guaranteed_node_counterexamples']==[]
assert diagnostic['conclusion']=='unknown_no_node_counterexample_established'
assert diagnostic['within_diagnostic_wall_budget']==(diagnostic['phase_wall_seconds']<=300.)
freeze=read(OUT/'freeze_invocation.json');freeze_result=read(OUT/'freeze_result.json')
run=read(OUT/'run_invocation.json');started=read(OUT/'started.json')
assert freeze['argv'][-1]=='freeze' and run['command']==lock['command']
assert freeze['script_sha256']==sha(OUT/'diagnose_minimum_cost.py')
assert run['root_reviewed_lock_sha256']==sha(OUT/'lock.json')==freeze_result['lock_sha256']
assert freeze['environment']==run['environment']==lock['runtime']['threads']
assert freeze_result['metadata_freeze_exit_code']==0 and not freeze_result['run_started']
assert not freeze_result['npz_decoded'] and not started['real_inputs_read'] and completion['real_inputs_read']
assert freeze['created_unix']<=lock['created_unix']<=freeze_result['created_unix']<=run['created_unix']<=started['started_unix']<=diagnostic['finished_unix']<=completion['finished_unix']
assert started['lock_sha256']==sha(OUT/'lock.json')
development=read(OUT/'development/attempt1.json')
assert development['exit_code']==0 and development['source_hashes']==development['source_hashes_after']
assert sha(OUT/'development/attempt1.log')==development['log_sha256']
for name,digest in development['source_hashes'].items():
    assert sha(ROOT/development['source_snapshot']/name)==digest
all_files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*'))
           if p.is_file() and p.name!='execution_receipt.json'}
all_files[str(REPORT.relative_to(ROOT))]=sha(REPORT)
assert str((OUT/'execution.log').relative_to(ROOT)) in all_files
assert any('/development/' in name for name in all_files)
receipt=dict(created_unix=time.time(),accepted_for_cost_interpretation=True,delivery_integrity_passed=True,
    case_count=72,order_count=144,all72_initial_geometry_counted=True,
    fixed_population_and_order_verified=True,progress_matches_records=True,
    binding_count=len(lock['bindings']),local_source_count=len(lock['local_source_hashes']),
    source_current_and_snapshot_hashes_verified=True,input_archive_byte_hashes_verified=True,
    geometry_archive_byte_hashes_verified=True,canonical_geometry_arrays_redecoded=False,
    geometry_recomputed=False,new_reference_or_probability_arrays_created=False,
    verifier_read_real_arrays=False,diagnostic_read_real_arrays=True,
    metadata_freeze_precedes_formal_read=True,runtime_equals_R35=True,
    maximum_lower_bounds=maximum,summed_lower_bounds=totals,
    total_geometry_callback_positions=sum(r['completed_required_callbacks'] for r in records),
    phase_wall_seconds=diagnostic['phase_wall_seconds'],guaranteed_initial_node_budget_failures=0,
    budget_feasibility='unknown',numerical_preflight_passed=False,scientific_gate_passed=False,
    labels_read=False,efficacy_tested=False,candidate_selected=False,goal_complete=False,
    source_integrations=0,files=all_files,file_count=len(all_files),
    inventory_includes_execution_log_and_all_development_history=True,
    report_path=str(REPORT.relative_to(ROOT)),report_sha256=sha(REPORT),
    note='Necessary minimum-X admission lower bounds only; no actual total-node upper bound or numerical pass.')
(OUT/'execution_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))
