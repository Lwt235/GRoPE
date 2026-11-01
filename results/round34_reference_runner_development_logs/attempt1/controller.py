"""Preserve every R34 reference/runner development test attempt before execution."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

ROOT=Path('/home/liwt/GRoPE')
BASE=ROOT/'results/round34_reference_runner_development_logs'
SOURCES=('experiments/round34_log_narrow_reference.py','experiments/round34_log_narrow_preflight.py',
         'tests/test_round34_log_narrow_reference.py','tests/test_round34_log_narrow_preflight.py',
         'experiments/round34_log_narrow_contract.py','ROUND34_LOG_NARROW_REFERENCE_PROTOCOL.md',
         'docs/round34_log_narrow_reference_design.md','experiments/round32_log_conditional_reference.py',
         'experiments/round32_log_conditional_preflight.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')

BASE.mkdir(exist_ok=True)
number=1
while (BASE/('attempt'+str(number))).exists():number+=1
out=BASE/('attempt'+str(number));out.mkdir()
sources={}
for name in SOURCES:
    target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/name,target);sources[name]=sha(target)
    assert sources[name]==sha(ROOT/name)
shutil.copyfile(__file__,out/'controller.py')
cmd=['/home/liwt/miniColmap/.venv/bin/python','-c',
     "import numpy; import pytest; raise SystemExit(pytest.main(['-q','tests/test_round34_log_narrow_reference.py','tests/test_round34_log_narrow_preflight.py']))"]
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(ROOT))
started=time.time()
write(out/'before_execution.json',dict(attempt=number,created_unix=started,source_hashes=sources,
    controller_sha256=sha(out/'controller.py'),command=cmd,cwd=str(ROOT),
    thread_environment={k:env[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
    formal_descriptors_executed=False,real_inputs_read=False,labels_read=False,
    scope='Distinct numerical toys and mock-only runner; no declared formal23 integration'))
with (out/'pytest.log').open('wb') as log:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
log=(out/'pytest.log').read_text()
write(out/'receipt.json',dict(attempt=number,exit_code=result.returncode,started_unix=started,finished_unix=time.time(),
    test_summary=[line for line in log.splitlines() if re.search(r'\d+ (passed|failed|error)',line)],
    pytest_log_sha256=sha(out/'pytest.log'),source_hashes_after={n:sha(ROOT/n) for n in SOURCES},
    sources_unchanged_during_tests=all(sha(ROOT/n)==h for n,h in sources.items()),
    formal_integrations=0,real_inputs_read=False,labels_read=False))
print(log,end='');print('ATTEMPT_DIRECTORY',out);raise SystemExit(result.returncode)
