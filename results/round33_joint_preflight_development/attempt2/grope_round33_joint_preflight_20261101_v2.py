import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import numpy as np
from pathlib import Path
import sys,json,time,subprocess
root=Path('/home/liwt/GRoPE');sys.path.insert(0,str(root))
from experiments import round33_log_real_preflight as P,round33_log_real_audit as A,round33_log_real_contract as C
from experiments import round32_log_conditional_reference as REF
from models import round29_log_rms_projective_pe as M
out=root/'results/round33_log_real_joint_preflight_20261101_v2.json';log=out.with_suffix('.log')
assert not out.exists() and not log.exists()
source_hashes={n:C.sha(root/n) for n in C.source_names()};started=time.time()
with log.open('w') as stream:
 proc=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_round33_log_real_preflight.py','tests/test_round33_log_real_audit.py'],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
record=dict(started_unix=started,tests_exit_code=proc.returncode,tests_log_sha256=C.sha(log),tests_log_path=str(log),source_hashes=source_hashes,
 real_inputs_read=False,independent_real_reference=False,reference_integrations_executed=0,formal_reference_calls=0,
 scope='Pure saved-array/mock tests; metadata-only joint freeze below prohibits all NPZ decoding and integrator calls.',**C.SCOPE)
try:
 assert proc.returncode==0
 scratch=Path('/tmp/grope_round33_metadata_preflight_20261101_v2');assert not scratch.exists();scratch.mkdir()
 prod=scratch/'producer';audit=scratch/'audit'
 def denied(*a,**k):raise AssertionError('metadata freeze must not decode arrays or integrate')
 old=np.load,REF.integrate,M.integrate;np.load=REF.integrate=M.integrate=denied
 try:P.freeze(prod,audit);A.freeze(prod,audit);P.verify(prod);A.joint_locks(prod,audit)
 finally:np.load,REF.integrate,M.integrate=old
 assert source_hashes=={n:C.sha(root/n) for n in C.source_names()}
 record.update(passed=True,metadata_only_freeze_verified=True,scratch=str(scratch),source_count=len(source_hashes),runtime=C.check_runtime(),producer_lock_sha256=C.sha(prod/'lock.json'),audit_lock_sha256=C.sha(audit/'lock.json'))
except Exception as e:
 record.update(passed=False,error=type(e).__name__+': '+str(e));raise
finally:
 record['finished_unix']=time.time();C.write(out,record)
print(json.dumps({k:v for k,v in record.items() if k not in ('runtime','source_hashes')},indent=2))
