import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import numpy as np
from pathlib import Path
import sys,json,time,subprocess,ast
ROOT=Path('/home/liwt/GRoPE');sys.path.insert(0,str(ROOT))
from experiments import round34_log_narrow_preflight as P,round34_log_narrow_audit as A,round34_log_narrow_contract as C
from models import round29_log_rms_projective_pe as M
from experiments import round34_log_narrow_reference as R
out=ROOT/'results/round34_log_narrow_joint_preflight_20261101.json'
log=ROOT/'results/round34_log_narrow_joint_preflight_20261101.log'
if out.exists() or log.exists():raise RuntimeError('preserve earlier preflight')
names=C.source_names();before={n:C.sha(ROOT/n) for n in names};started=time.time()
oldtree=ast.parse((ROOT/'experiments/round31_log_conditional_reference.py').read_text())
newtree=ast.parse((ROOT/'experiments/round34_log_narrow_reference.py').read_text())
oldnodes={n.name:ast.dump(n,include_attributes=False) for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
newnodes={n.name:ast.dump(n,include_attributes=False) for n in newtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert set(newnodes)==set(oldnodes)|{'coalesce_depth_edges'}
unchanged=sorted(set(oldnodes)-{'_check_policy','integrate'})
assert all(oldnodes[n]==newnodes[n] for n in unchanged)
with log.open('w') as stream:
    proc=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_round34_log_narrow_reference.py','tests/test_round34_log_narrow_preflight.py','tests/test_round34_log_narrow_audit.py'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
record=dict(scope='Formal synthetic archives and reference calls excluded; distinct unit-toy integrations execute only inside the recorded pytest subprocess; following freeze probe is metadata-only.',distinct_unit_toy_reference_integrations_in_subprocess=True,formal_reference_calls=0,started_unix=started,tests_exit_code=proc.returncode,source_hashes=before,tests_log_path=str(log),tests_log_sha256=C.sha(log),real_inputs_read=False,synthetic_inputs_read=False,independent_synthetic_reference=False,labels_read=False,independent_probability_reference=False,general_numerical_accuracy_established=False,efficacy_tested=False,candidate_selected=False,scientific_gate_passed=False,goal_complete=False)
record['unchanged_R31_reference_functions_and_classes']=unchanged
try:
    if proc.returncode:raise RuntimeError('joint toys failed')
    scratch=Path('/tmp/grope_round32_log_metadata_preflight_20261101')
    if scratch.exists():raise RuntimeError('preserve earlier metadata preflight')
    scratch.mkdir();prod=scratch/'producer';audit=scratch/'audit'
    def denied(*args,**kwargs):raise AssertionError('metadata freeze may not load arrays or integrate')
    old=(np.load,M.integrate,R.integrate);np.load=M.integrate=R.integrate=denied
    try:
        P.freeze(prod,audit);A.freeze(prod,audit);P.verify(prod);A.joint_locks(prod,audit)
    finally:np.load,M.integrate,R.integrate=old
    assert before=={n:C.sha(ROOT/n) for n in names}
    record.update(passed=True,metadata_only_freeze_verified=True,scratch=str(scratch),source_count=len(names),runtime=C.check_runtime(),producer_lock_sha256=C.sha(prod/'lock.json'),audit_lock_sha256=C.sha(audit/'lock.json'))
except Exception as e:
    record.update(passed=False,error=type(e).__name__+': '+str(e));raise
finally:
    record['finished_unix']=time.time();C.write(out,record)
print(json.dumps({k:v for k,v in record.items() if k not in ('runtime','source_hashes')},indent=2))
