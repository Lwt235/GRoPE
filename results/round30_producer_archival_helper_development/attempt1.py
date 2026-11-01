from pathlib import Path
import shutil,hashlib,json,time
root=Path('/home/liwt/GRoPE');source=Path('/tmp/round30_producer_development_logs');target=root/'results/round30_producer_development_logs';history=root/'results/round30_producer_development_history_20261031.json'
assert not target.exists() and not history.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=sorted(p for p in source.rglob('*') if p.is_file());assert len(files)==4
shutil.copytree(source,target)
for p in files:assert sha(p)==sha(target/p.relative_to(source))
receipt=json.loads((target/'attempt1.json').read_text());print('ORIGINAL RECEIPT',json.dumps(receipt,indent=2));print('ORIGINAL LOG',(target/'attempt1.log').read_text())
assert receipt['returncode']==0
assert '20 passed' in (target/'attempt1.log').read_text()
h=dict(created_unix=time.time(),role='producer',scope='Pre-freeze distinct synthetic producer tests only; archive copied after formal execution began',failed_test_attempts=0,formal_fixture_integrations=0,real_inputs_read=False,labels_read=False,files={str(p.relative_to(root)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()},notes=['First producer test attempt: 20 pure toys passed; original log, receipt, producer and test source bytes copied intact from /tmp/round30_producer_development_logs.','This history archives existing development evidence; it does not claim the archival copy was contemporaneous with testing and performs no model or data rerun.'])
history.write_text(json.dumps(h,indent=2,sort_keys=True,allow_nan=False)+'\n')
print('HISTORY',sha(history))
