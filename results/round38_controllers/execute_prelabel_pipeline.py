"""Run the already selected R38 stages once, retaining every exit and seal.

This controller stops before label access. Its own bytes and the full source
closure must be captured in before_execution.json before it can execute.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/liwt/GRoPE')
HERE = Path(__file__).resolve().parent
PYTHON = '/home/liwt/miniColmap/.venv/bin/python'
RUN = ROOT/'results/round38_rms_anchor_screen'
AUDIT = ROOT/'results/round38_rms_anchor_independent_audit'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')

def main():
    pre = read(HERE/'before_execution.json')
    assert sha(__file__) == pre['controller_sha256']
    assert not (HERE/'started.json').exists(), 'preserve prior controller attempts'
    assert not RUN.exists() and not AUDIT.exists(), 'preserve prior formal run'
    def verify_sources():
        for name, digest in pre['source_hashes'].items():
            assert sha(ROOT/name) == sha(HERE/'source_snapshot'/name) == digest, name
    verify_sources()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    write(HERE/'started.json', dict(started_unix=time.time(), before_execution_sha256=sha(HERE/'before_execution.json'), labels_read=False))
    stages = []
    state = 'incomplete'
    error = None
    def execute(name, module, stage):
        command = [PYTHON, '-m', module, stage]
        row = dict(name=name, command=command, started_unix=time.time())
        with (HERE/(name+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row.update(finished_unix=time.time(), exit_code=result.returncode, log_sha256=sha(HERE/(name+'.log')))
        stages.append(row)
        write(HERE/'stage_records.json', stages)
        verify_sources()
        print('R38_STAGE', name, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError('stage failed: '+name)
    try:
        execute('01_producer_freeze', 'experiments.round38_rms_anchor_screen', 'freeze')
        execute('02_audit_freeze', 'experiments.round38_rms_anchor_audit', 'freeze')
        execute('03_prepare', 'experiments.round38_rms_anchor_screen', 'prepare')
        if not read(RUN/'prepared_complete.json')['all_inputs_prepared']:
            state = 'preparation_failed_labels_blocked'
        else:
            execute('04_input_audit', 'experiments.round38_rms_anchor_audit', 'inputs')
            if not read(AUDIT/'input_verification.json')['passed']:
                state = 'input_audit_failed_labels_blocked'
            else:
                execute('05_forward', 'experiments.round38_rms_anchor_screen', 'forward')
                # Independently inspect saved failures and the entire suffix too.
                execute('06_output_audit', 'experiments.round38_rms_anchor_audit', 'outputs')
                state = ('ready_for_root_prelabel_review' if read(AUDIT/'forward_verification.json')['all_input_and_output_arithmetic_passed']
                         else 'forward_or_arithmetic_failed_labels_blocked')
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        state = 'execution_failed_labels_blocked'
    finally:
        verify_sources()
        files = {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='completion.json'}
        write(HERE/'completion.json', dict(state=state, error=error, stages=stages,
            files=files, finished_unix=time.time(), labels_read=False,
            independent_probability_reference=False, numerical_preflight_passed=False,
            scientific_gate_passed=False, goal_complete=False))
    print('R38_PRELABEL_PIPELINE', state, flush=True)
    return 1 if error else 0

if __name__ == '__main__':
    sys.exit(main())
