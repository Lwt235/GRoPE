import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import sys,time,json,traceback
import numpy as np
root=Path('/home/liwt/GRoPE');sys.path.insert(0,str(root))
from experiments import round32_log_conditional_audit_repair as RP
from experiments import round32_log_conditional_reference as REF
from models import round29_log_rms_projective_pe as MODEL
C=RP.C
receipt=root/'results/round32_log_conditional_audit_repair_execution_20261031.json'
log=root/'results/round32_log_conditional_audit_repair_pipeline_20261031.log'
assert not receipt.exists() and not log.exists()
history_path=root/'results/round32_audit_repair_development_history_20261031.json'
history=C.read(history_path)
assert history['failed_test_attempts']==0 and history['all_source_snapshots_before_test']
for name,digest in dict(history['files'],**history['source_hashes']).items():assert C.sha(root/name)==digest
record=dict(started_unix=time.time(),stages=[],post_seal_audit_repair=True,original_sixth_stage_failed=True,
 reference_integrations_executed=0,model_integrations_executed=0,synthetic_inputs_read=False,real_inputs_read=False,
 labels_read=False,scientific_gate_passed=False,goal_complete=False,controller_sha256=C.sha(Path(__file__)),
 repair_source_hashes={n:C.sha(root/n) for n in RP.REPAIR_SOURCES})
record['test_history_sha256']=C.sha(history_path)
def denied(*a,**k):raise AssertionError('repair must not execute model or reference integration')
oldref,oldmodel=REF.integrate,MODEL.integrate;REF.integrate=MODEL.integrate=denied
try:
 with log.open('w') as stream:
  for name in ('metadata_only_repair_freeze','saved_output_audit'):
   t=time.time();stage=dict(name=name,started_unix=t);record['stages'].append(stage)
   try:
    if name=='metadata_only_repair_freeze':
     oldload=np.load;np.load=denied
     try:RP.freeze(RP.DEFAULT_RUN,RP.DEFAULT_ORIGINAL,RP.DEFAULT_OUT);RP.verify_repair(RP.DEFAULT_OUT)
     finally:np.load=oldload
     record['metadata_only_freeze_verified']=True
    else:
     record['synthetic_inputs_read']=None;C.write(receipt,record)
     result=RP.run_repair(RP.DEFAULT_OUT)
     record['synthetic_inputs_read']=result['verification']['synthetic_inputs_read']
     record['result']=result
    stage.update(exit_code=0,finished_unix=time.time());stream.write(name+' completed\n');stream.flush()
   except BaseException as error:
    stage.update(exit_code=1,finished_unix=time.time(),error=type(error).__name__+': '+str(error));traceback.print_exc(file=stream);stream.flush();raise
   finally:C.write(receipt,record)
 record.update(all_repair_stages_completed=True,finished_unix=time.time(),pipeline_log_sha256=C.sha(log),repair_lock_sha256=C.sha(RP.DEFAULT_OUT/'repair_lock.json'))
 C.write(receipt,record)
finally:
 REF.integrate,MODEL.integrate=oldref,oldmodel
print(json.dumps(record,indent=2))
