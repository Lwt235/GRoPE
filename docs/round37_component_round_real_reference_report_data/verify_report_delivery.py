"""Post-extraction delivery verification; JSON, CSV and file-byte hashes only."""
from pathlib import Path
import csv,hashlib,json,time
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'docs/round37_component_round_real_reference_report_data'
REPORT=ROOT/'docs/round37_component_round_real_reference_report.md'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
completion=read(OUT/'completion.json');stats=read(OUT/'statistics.json')
for name,digest in completion['outputs'].items():assert sha(OUT/name)==digest,name
counts={'cases72.csv':72,'reference_calls144.csv':144,'comparisons296.csv':296,'oracles40.csv':40,'outer_trace_summaries.csv':56,'matched_historical_costs40.csv':40}
for name,number in counts.items():
    with (OUT/name).open() as f:assert len(list(csv.DictReader(f)))==number,name
assert stats['summary']['delivery_integrity_passed'] and not stats['summary']['numerical_preflight_passed']
assert stats['summary']['actual_reference_calls']==56 and stats['summary']['reference_completed_calls']==55
assert stats['summary']['fail_fast_skipped_calls']==88 and stats['summary']['comparison_failures']==134
assert sum(v['passes'] for v in stats['comparisons'].values())==162
assert stats['outer_trace']['complete_count']==55 and stats['outer_trace']['incomplete_count']==1
assert completion['npz_decodes']==completion['integrator_calls']==0
attempts=[]
for suffix in ('','_attempt2'):
    invocation=read(OUT/('extraction_invocation'+suffix+'.json'));execution=read(OUT/('extraction_execution'+suffix+'.json'))
    assert invocation['source_and_inputs_captured_before_extraction']
    for name,digest in invocation['source_snapshots'].items():assert sha(ROOT/name)==digest
    for name,digest in invocation['inputs'].items():assert sha(ROOT/name)==digest
    log=OUT/('extraction'+suffix+'.log');assert sha(log)==execution['log_sha256']
    attempts.append(dict(command=execution['command'],exit_code=execution['exit_code'],started_unix=execution['started_unix'],finished_unix=execution['finished_unix'],log_path=str(log.relative_to(ROOT)),log_sha256=sha(log),source_snapshot_hashes=execution['source_snapshots']))
assert [a['exit_code'] for a in attempts]==[1,0]
assert sha(OUT/'source_snapshot_attempt2/grope_round37_report_extract.py')==completion['generator_sha256']
controllers=ROOT/'results/round37_controllers/completion.json'
assert sha(controllers)=='16680dde4419213e9fbdfcf543d52269004c6441c56f3fdd7f86c8b773c55122'
files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='execution_receipt.json'}
files[str(REPORT.relative_to(ROOT))]=sha(REPORT)
record=dict(created_unix=time.time(),delivery_integrity_passed=True,report_complete=True,
    numerical_preflight_passed=False,scientific_gate_passed=False,goal_complete=False,
    root_acceptance_path='results/round37_component_round_real_root_acceptance_20261101.json',root_acceptance_sha256=completion['receipt_sha256'],
    controller_completion_sha256=sha(controllers),report_sha256=sha(REPORT),extractor_sha256=completion['generator_sha256'],
    output_completion_sha256=sha(OUT/'completion.json'),row_counts=counts,files=files,
    attempts=attempts,first_attempt_failure='Old report extractor assumed the R35 controller manifest schema; failed before creating CSV outputs. Only postprocessing manifest fields changed for attempt2.',
    all_extraction_sources_commands_and_input_hashes_captured_before_execution=True,
    npz_decodes=0,integrator_calls=0,new_tests_executed=0,real_inputs_read=False,labels_read=False,
    scope='Report extraction and verification only. Original R37 real-input flags remain unchanged. No postfailure numerical or cost diagnostic included.')
destination=OUT/'execution_receipt.json'
with destination.open('x') as f:json.dump(record,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
print(json.dumps({'receipt_path':str(destination.relative_to(ROOT)),'receipt_sha256':sha(destination),'report_sha256':sha(REPORT),'files':len(files),'completion_sha256':sha(OUT/'completion.json')},indent=2))
