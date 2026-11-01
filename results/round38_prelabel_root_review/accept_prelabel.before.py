"""Root receipt for completed exploratory inputs/production/arithmetic only."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT=Path('/home/liwt/GRoPE')
sys.path.insert(0,str(ROOT))
from experiments import round38_rms_anchor_contract as C

def main():
    C.configure_runtime()
    run,audit=C.DEFAULT_OUT,C.DEFAULT_AUDIT
    destination=ROOT/'results/round38_rms_anchor_prelabel_root_acceptance_20261101.json'
    assert not destination.exists(), 'preserve prior root decision'
    before=C.read(ROOT/'results/round38_prelabel_root_review/before_execution.json')
    assert C.sha(__file__)==before['controller_sha256']
    for name,digest in before['files'].items():assert C.sha(ROOT/name)==digest,name
    lock,alock=C.joint_locks(run,audit)
    checks=C.read(audit/'checks.json');v=C.read(audit/'forward_verification.json')
    input_v=C.read(audit/'input_verification.json')
    pairs=C.read(audit/'input_audit/pair_checks.json')
    inputs=C.read(audit/'input_audit/checks.json')
    rows=C.read(run/'forward/records.json');specs=C.read(run/'query_specs.json')
    seals={}
    for directory in (run/'prepared',run/'forward',audit/'input_audit',audit):
        completion=C.read(directory/'completion.json')
        C.verify_inventory(directory,completion['files'],exclude=('completion.json',))
        seals[str(directory.relative_to(ROOT))]=C.sha(directory/'completion.json')
    assert len(pairs)==48 and all(p['passed'] for p in pairs)
    assert len(inputs)==len(checks)==len(rows)==len(specs)==2304
    ids=[s['id'] for s in specs]
    assert [r['id'] for r in rows]==[r['id'] for r in checks]==[r['id'] for r in inputs]==ids
    assert all(r['passed'] and r['attempted'] and r['status']=='ok' and not r['work_unavailable'] for r in rows)
    assert all(c['passed'] for c in inputs)
    assert all(c['delivery_integrity_passed'] and c['arithmetic_passed'] for c in checks)
    assert input_v['passed'] and input_v['checked_pairs']==48 and input_v['checked_queries']==2304
    assert all(v[k] is True for k in ('all_input_and_output_arithmetic_passed','can_evaluate_train_triage','delivery_integrity_passed','ledger_passed'))
    assert v['checked_pairs']==48 and v['checked_queries']==v['completed_queries']==2304
    assert v['independent_probability_reference'] is False and v['labels_read'] is False
    forward=C.read(run/'forward/completion.json');seal=C.read(run/'forward_complete.json')
    assert forward['all_production_completed'] and seal['all_production_completed']
    assert 0<=forward['wall_seconds']<=C.GATES['production_max_phase_wall_seconds']
    total=sum(r['charged_cell_node_evaluations'] for r in rows)
    assert total==forward['charged_cell_node_evaluations']==seal['charged_cell_node_evaluations']
    assert total<=C.GATES['production_max_phase_cell_node_evaluations']
    assert seal['actual_calls']==seal['completed_calls']==2304 and seal['stop_trigger'] is None
    for key,path in [('producer_lock_sha256',run/'lock.json'),('audit_lock_sha256',audit/'lock.json'),
        ('producer_forward_complete_sha256',run/'forward_complete.json'),('input_verification_sha256',audit/'input_verification.json'),
        ('checks_sha256',audit/'checks.json')]:assert v[key]==C.sha(path),key
    paths=[Path(__file__),ROOT/'results/round38_prelabel_root_review/before_execution.json',
        ROOT/'results/round38_controllers/completion.json',run/'lock.json',audit/'lock.json',run/'query_specs.json',
        run/'prepared_complete.json',run/'prepared/completion.json',run/'forward_complete.json',
        run/'forward/completion.json',run/'forward/records.json',audit/'input_verification.json',
        audit/'input_audit/completion.json',audit/'input_audit/checks.json',audit/'input_audit/pair_checks.json',
        audit/'forward_verification.json',audit/'checks.json',audit/'completion.json']
    files={str(p.relative_to(ROOT)):C.sha(p) for p in paths}
    C.write(destination,dict(created_unix=time.time(),accepted_for_exploratory_label_stage=True,
        all_input_and_output_arithmetic_passed=True,prepared_queries=2304,completed_queries=2304,
        independent_probability_reference=False,numerical_preflight_passed=False,
        labels_read=False,scientific_gate_passed=False,goal_complete=False,
        charged_cell_node_evaluations=total,production_wall_seconds=forward['wall_seconds'],
        maximum_input_normalized_error=max(p['maximum_normalized_error'] for p in pairs),
        scope='Only negative-only old-TRAIN triage authorized by frozen R38 protocol; independent probability reference and every remaining positive gate still mandatory.',
        files=files,verified_stage_completions=seals))
    print('R38_ROOT_PRELABEL_ACCEPTED',C.sha(destination),flush=True)

if __name__=='__main__':main()
