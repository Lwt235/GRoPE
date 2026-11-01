import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='4'
import sys,time,json,subprocess
from pathlib import Path
root=Path('/home/liwt/GRoPE');sys.path.insert(0,str(root))
from experiments import round36_component_round_contract as C
# Load the same numerical dependencies as the metadata freeze before runtime comparison.
from experiments import round36_component_round_preflight as P,round36_component_round_audit as A
receipt=root/'results/round36_component_round_execution_20261101.json';log=root/'results/round36_component_round_pipeline_20261101.log'
if receipt.exists() or log.exists():raise RuntimeError('preserve previous formal execution')
pre=C.read(root/'results/round36_component_round_joint_preflight_20261101.json')
assert pre['passed'] and pre['metadata_only_freeze_verified']
assert pre['source_hashes']=={n:C.sha(root/n) for n in C.source_names()}
assert pre['tests_log_sha256']==C.sha(pre['tests_log_path'])
assert pre['runtime']==C.check_runtime()
record=dict(started_unix=time.time(),joint_preflight_sha256=C.sha(root/'results/round36_component_round_joint_preflight_20261101.json'),stages=[],synthetic_prepared_inputs_authorized_after_joint_freeze=True,synthetic_inputs_read=False,independent_synthetic_reference=False,real_inputs_read=False,labels_read=False,independent_probability_reference=False,general_numerical_accuracy_established=False,efficacy_tested=False,candidate_selected=False,scientific_gate_passed=False,goal_complete=False)
steps=[('producer_freeze',['experiments.round36_component_round_preflight','freeze']),('audit_freeze',['experiments.round36_component_round_audit','freeze']),('prepare_inputs',['experiments.round36_component_round_preflight','prepare']),('independent_input_audit',['experiments.round36_component_round_audit','input-audit']),('reference_run',['experiments.round36_component_round_preflight','run']),('independent_output_audit',['experiments.round36_component_round_audit','audit'])]
with log.open('w') as stream:
    for name,args in steps:
        started=time.time();print(name,flush=True);stream.write('\nSTAGE '+name+'\n');stream.flush()
        if name=='prepare_inputs':
            record['synthetic_inputs_read']=None;C.write(receipt,record)
        if name=='reference_run':
            record['independent_synthetic_reference']=None;C.write(receipt,record)
        child=subprocess.run([sys.executable,'-m',*args],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        if name=='prepare_inputs':
            for candidate in ('prepared_complete.json','prepared/failure.json'):
                source=root/'results/round36_component_round_preflight'/candidate
                if source.is_file():record['synthetic_inputs_read']=C.read(source)['synthetic_inputs_read'];break
        if name=='reference_run':
            for candidate in ('forward_complete.json','forward/failure.json'):
                source=root/'results/round36_component_round_preflight'/candidate
                if source.is_file():record['independent_synthetic_reference']=C.read(source)['independent_synthetic_reference'];break
        record['stages'].append(dict(name=name,started_unix=started,finished_unix=time.time(),exit_code=child.returncode));C.write(receipt,record)
        if child.returncode:raise SystemExit(child.returncode)
record.update(finished_unix=time.time(),all_stages_completed=True,pipeline_log_sha256=C.sha(log));C.write(receipt,record)
print(json.dumps(record,indent=2),flush=True)
