from pathlib import Path
import hashlib,json,shutil,subprocess,time
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'docs/round37_component_round_real_reference_report_data'
assert OUT.is_dir();snapshot=OUT/'source_snapshot_attempt2';snapshot.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=ROOT/'results/round37_component_round_real_root_acceptance_20261101.json';expected='0f7552a673de1422b215972c105eccddeb0fc5119b8716fa1168aded5570caf5'
assert sha(receipt)==expected
record=json.loads(receipt.read_text());assert record['accepted_for_numerical_interpretation'] and record['delivery_integrity_passed']
for n,h in record['files'].items():assert sha(ROOT/n)==h,n
sources={}
for name in ('/tmp/grope_round37_report_extract.py',__file__):
    dest=snapshot/Path(name).name;shutil.copy2(name,dest);sources[str(dest.relative_to(ROOT))]=sha(dest)
inputs={str(receipt.relative_to(ROOT)):expected,**record['files']}
for p in ROOT.glob('results/round37*history*.json'):inputs[str(p.relative_to(ROOT))]=sha(p)
p=ROOT/'results/round37_controllers/before_execution.json';inputs[str(p.relative_to(ROOT))]=sha(p)
command=['/home/liwt/miniColmap/.venv/bin/python','/tmp/grope_round37_report_extract.py','--receipt-sha',expected]
meta=dict(started_unix=time.time(),command=command,source_snapshots=sources,inputs=inputs,npz_decodes=0,integrator_calls=0,real_inputs_read=False,labels_read=False,source_and_inputs_captured_before_extraction=True)
(OUT/'extraction_invocation_attempt2.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
with (OUT/'extraction_attempt2.log').open('x') as log:r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
meta.update(exit_code=r.returncode,finished_unix=time.time(),log_sha256=sha(OUT/'extraction_attempt2.log'))
(OUT/'extraction_execution_attempt2.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
print((OUT/'extraction_attempt2.log').read_text());raise SystemExit(r.returncode)
