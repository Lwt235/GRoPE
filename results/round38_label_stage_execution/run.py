"""Run the two frozen exploratory metric routes once after root acceptance."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path('/home/liwt/GRoPE')
HERE=Path(__file__).resolve().parent
PYTHON='/home/liwt/miniColmap/.venv/bin/python'
sys.path.insert(0,str(ROOT))
from experiments import round38_rms_anchor_contract as C
from experiments import round38_rms_anchor_evaluate as E

def main():
    C.configure_runtime()
    pre=C.read(HERE/'before_execution.json')
    assert C.sha(__file__)==pre['controller_sha256']
    assert not (HERE/'started.json').exists(), 'preserve previous label-stage execution'
    def verify():
        for name,digest in pre['source_hashes'].items():
            assert C.sha(ROOT/name)==C.sha(HERE/'source_snapshot'/name)==digest,name
        for name,digest in pre['files'].items():assert C.sha(ROOT/name)==digest,name
    verify()
    # This check never opens labels; each metric subprocess repeats it itself.
    E.require_prelabel(C.DEFAULT_OUT,C.DEFAULT_AUDIT)
    C.write(HERE/'started.json',dict(started_unix=time.time(),before_execution_sha256=C.sha(HERE/'before_execution.json')))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    stages=[];error=None;state='incomplete'
    try:
        for stage in ('evaluate','audit'):
            command=[PYTHON,'-m','experiments.round38_rms_anchor_evaluate',stage]
            row=dict(stage=stage,command=command,started_unix=time.time())
            with (HERE/(stage+'.log')).open('x') as log:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            row.update(finished_unix=time.time(),exit_code=result.returncode,log_sha256=C.sha(HERE/(stage+'.log')))
            stages.append(row);C.write(HERE/'stage_records.json',stages);verify()
            print('R38_LABEL_STAGE',stage,result.returncode,flush=True)
            if result.returncode:raise RuntimeError('metric stage failed: '+stage)
        v=C.read(C.DEFAULT_OUT/'metrics_audit/verification.json')
        assert v['passed'] and v['eligible_for_promotion'] is False
        assert v['independent_probability_reference'] is False
        state=v['decision']
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        state='metric_execution_failed_no_interpretation'
    finally:
        verify()
        files=C.inventory(HERE)
        C.write(HERE/'completion.json',dict(state=state,error=error,stages=stages,files=files,
            finished_unix=time.time(),independent_probability_reference=False,
            numerical_preflight_passed=False,eligible_for_promotion=False,
            scientific_gate_passed=False,goal_complete=False))
    print('R38_EXPLORATORY_DECISION',state,flush=True)
    return 1 if error else 0

if __name__=='__main__':sys.exit(main())
