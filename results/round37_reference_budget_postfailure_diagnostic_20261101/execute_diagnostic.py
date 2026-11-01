"""Execute the frozen geometry diagnostic once, preserving failure output."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'results/round37_reference_budget_postfailure_diagnostic_20261101'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    assert not p.exists()
    p.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
pre=json.loads((OUT/'before_execution.json').read_text())
for name,h in pre['bindings'].items():assert sha(ROOT/name)==h
command=pre['command'];started=time.time()
write(OUT/'invocation.json',dict(started_unix=started,command=command,before_execution_sha256=sha(OUT/'before_execution.json'),controller_sha256=sha(__file__),environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
with (OUT/'diagnostic_stdout.log').open('x') as log:
    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
write(OUT/'execution.json',dict(started_unix=started,finished_unix=time.time(),exit_code=result.returncode,command=command,log_sha256=sha(OUT/'diagnostic_stdout.log'),reference_integrations=0,probability_integrations=0,labels_read=False))
print(json.dumps(dict(exit_code=result.returncode,execution_path=str(OUT/'execution.json'))))
sys.exit(result.returncode)
