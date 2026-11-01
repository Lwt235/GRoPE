"""Reconcile a controller stdout suffix; never change the sealed experiment."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path('/home/liwt/GRoPE')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments import round38_rms_anchor_contract as C
from experiments import round38_rms_anchor_evaluate as E
import numpy as np


def reject_load(*args, **kwargs):
    raise RuntimeError('Reconciliation must not decode arrays')


def main():
    C.configure_runtime()
    np.load = reject_load
    before = C.read(HERE / 'before_execution.json')
    assert C.sha(__file__) == before['source_sha256']
    for name, digest in before['files'].items():
        assert C.sha(ROOT / name) == digest, name
    assert not (HERE / 'reconciliation.json').exists()
    E.require_prelabel(C.DEFAULT_OUT, C.DEFAULT_AUDIT)
    stages = ROOT / 'results/round38_label_stage_execution'
    completed = C.read(stages / 'completion.json')
    assert completed['error'] is None
    assert completed['state'] == 'stop_this_fixed_implementation_only'
    assert [x['stage'] for x in completed['stages']] == ['evaluate', 'audit']
    assert all(x['exit_code'] == 0 for x in completed['stages'])
    mismatches = [name for name, digest in completed['files'].items()
                  if C.sha(stages / name) != digest]
    assert mismatches == ['execution.log'], mismatches
    log = (stages / 'execution.log').read_bytes()
    suffix = b'R38_EXPLORATORY_DECISION stop_this_fixed_implementation_only\n'
    assert log.endswith(suffix)
    prefix = log[:-len(suffix)]
    assert prefix == b'R38_LABEL_STAGE evaluate 0\nR38_LABEL_STAGE audit 0\n'
    prefix_sha = hashlib.sha256(prefix).hexdigest()
    assert prefix_sha == completed['files']['execution.log']
    for name in ('evaluation', 'metrics_audit'):
        directory = C.DEFAULT_OUT / name
        seal = C.read(directory / 'completion.json')
        assert seal['passed'] is True and seal['goal_complete'] is False
        C.verify_inventory(directory, seal['files'], exclude=('completion.json',))
    verification = C.read(C.DEFAULT_OUT / 'metrics_audit/verification.json')
    screen = C.read(C.DEFAULT_OUT / 'evaluation/screen.json')
    assert verification['passed'] and not verification['exploratory_eight_checks_passed']
    assert len(screen['checks']) == 8 and all(not c['passed'] for c in screen['checks'])
    report = C.read(ROOT / 'docs/round38_rms_anchor_triage_report_data/completion.json')
    for name, digest in report['files'].items():
        assert C.sha(ROOT / name) == digest, name
    tools = ROOT / 'results/round38_report_tools'
    assert not (tools / 'before_execution.json').exists()
    assert C.sha(tools / 'report.py') == C.sha(ROOT / 'docs/round38_rms_anchor_triage_report_data/report_source.py')
    C.write(HERE / 'reconciliation.json', dict(
        created_unix=time.time(), before_execution_sha256=C.sha(HERE / 'before_execution.json'),
        accepted_controller_log_suffix_reconciliation=True,
        original_label_completion_preserved=True,
        label_completion_sha256=C.sha(stages / 'completion.json'),
        only_inventory_mismatch='execution.log',
        original_promised_prefix_sha256=prefix_sha,
        final_log_sha256=C.sha(stages / 'execution.log'),
        appended_text=suffix.decode(),
        cause='Controller inventories its redirected stdout file before printing its final decision line.',
        report_pre_execution_capture_succeeded=False,
        report_generated_despite_failed_capture_check=True,
        report_source_copy_timing='Copied by report.py at invocation; no successful separate pre-execution freeze claimed.',
        report_existing_bindings_verified=len(report['files']),
        new_array_decodes=0, new_model_calls=0, metric_reexecutions=0,
        original_scientific_outputs_modified=False, scientific_gate_passed=False, goal_complete=False))
    print('R38_METADATA_RECONCILED', C.sha(HERE / 'reconciliation.json'), flush=True)


if __name__ == '__main__':
    main()
