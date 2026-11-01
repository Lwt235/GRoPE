import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import sys,time,json,subprocess
from pathlib import Path
root=Path('/home/liwt/GRoPE');sys.path.insert(0,str(root))
from experiments import round37_component_round_real_contract as C,round37_component_round_real_preflight as P,round37_component_round_real_audit as A
receipt=root/'results/round37_component_round_real_execution_20261101.json';log=root/'results/round37_component_round_real_pipeline_20261101.log'
assert not receipt.exists() and not log.exists()
prepath=root/'results/round37_component_round_real_joint_preflight_20261101.json';pre=C.read(prepath)
assert pre['passed'] and pre['metadata_only_freeze_verified'] and pre['runtime']==C.check_runtime()
assert pre['source_hashes']=={n:C.sha(root/n) for n in C.source_names()}
assert pre['tests_log_sha256']==C.sha(pre['tests_log_path'])
record=dict(started_unix=time.time(),joint_preflight_sha256=C.sha(prepath),stages=[],real_inputs_read=False,independent_real_reference=False,
 real_prepared_inputs_authorized_after_joint_freeze=True,**C.SCOPE)
steps=[('producer_freeze',['experiments.round37_component_round_real_preflight','freeze']),('audit_freeze',['experiments.round37_component_round_real_audit','freeze']),('prepare_inputs',['experiments.round37_component_round_real_preflight','prepare']),('independent_input_audit',['experiments.round37_component_round_real_audit','input-audit']),('reference_run',['experiments.round37_component_round_real_preflight','run']),('independent_output_audit',['experiments.round37_component_round_real_audit','audit'])]
with log.open('w') as stream:
 for name,args in steps:
  started=time.time();print(name,flush=True);stream.write('\nSTAGE '+name+'\n');stream.flush()
  if name=='prepare_inputs':record['real_inputs_read']=None;C.write(receipt,record)
  if name=='reference_run':record['independent_real_reference']=None;C.write(receipt,record)
  child=subprocess.run([sys.executable,'-m',*args],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
  if name=='prepare_inputs':
   for candidate in ('prepared_complete.json','prepared/failure.json'):
    p=P.DEFAULT_OUT/candidate
    if p.is_file():record['real_inputs_read']=C.read(p)['real_inputs_read'];break
  if name=='reference_run':
   for candidate in ('forward_complete.json','forward/failure.json'):
    p=P.DEFAULT_OUT/candidate
    if p.is_file():record['independent_real_reference']=C.read(p)['independent_real_reference'];break
  record['stages'].append(dict(name=name,started_unix=started,finished_unix=time.time(),exit_code=child.returncode));C.write(receipt,record)
  if child.returncode:raise SystemExit(child.returncode)
record.update(finished_unix=time.time(),all_stages_completed=True,pipeline_log_sha256=C.sha(log));C.write(receipt,record)
print(json.dumps(record,indent=2),flush=True)
