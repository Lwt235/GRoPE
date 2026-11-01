"""One saved-output diagnostic; stdout must be redirected outside this folder."""
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
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    pre = read(HERE/'before_execution.json')
    assert sha(__file__) == pre['controller_sha256']
    assert not (HERE/'started.json').exists()
    producer = ROOT/'results/round38_saved_output_diagnostic'
    audit = ROOT/'results/round38_saved_output_diagnostic_audit'
    assert not producer.exists() and not audit.exists()
    def verify():
        for name,digest in pre['source_hashes'].items():
            assert sha(ROOT/name)==sha(HERE/'source_snapshot'/name)==digest,name
        for name,digest in pre['files'].items():
            assert sha(ROOT/name)==digest,name
    verify()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    write(HERE/'started.json',dict(started_unix=time.time(),before_execution_sha256=sha(HERE/'before_execution.json')))
    stages, error, passed = [], None, False
    commands = [
        ('producer_freeze',['-m','experiments.round38_saved_output_diagnostic','freeze']),
        ('audit_freeze',['-m','experiments.round38_saved_output_diagnostic','freeze','--out',str(audit),'--role','audit']),
        ('diagnostic',['-m','experiments.round38_saved_output_diagnostic','run']),
        ('independent_audit',['-m','experiments.round38_saved_output_diagnostic_audit','audit'])]
    try:
        for stage,args in commands:
            command = [PYTHON]+args
            row = dict(stage=stage,command=command,started_unix=time.time())
            with (HERE/(stage+'.log')).open('x') as log:
                process = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            row.update(exit_code=process.returncode,finished_unix=time.time(),log_sha256=sha(HERE/(stage+'.log')))
            stages.append(row)
            write(HERE/'stages.json',stages)
            verify()
            print('SAVED_DIAGNOSTIC_STAGE',stage,process.returncode,flush=True)
            if process.returncode:
                raise RuntimeError('stage failed: '+stage)
            if stage=='diagnostic' and not read(producer/'completion.json')['passed']:
                raise RuntimeError('diagnostic did not pass')
        passed = read(audit/'completion.json')['passed'] is True
        if not passed:
            raise RuntimeError('independent audit did not pass')
    except BaseException as exc:
        error = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        verify()
        files = {str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file()}
        write(HERE/'completion.json',dict(passed=passed,error=error,stages=stages,files=files,
            finished_unix=time.time(),new_model_calls=0,new_reference_calls=0,new_task_metrics_computed=False,
            scientific_gate_passed=False,goal_complete=False))
    print('SAVED_DIAGNOSTIC_COMPLETE',passed,flush=True)
    return 0 if passed and error is None else 1


if __name__=='__main__':
    sys.exit(main())
