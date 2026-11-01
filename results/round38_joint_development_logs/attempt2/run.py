from pathlib import Path
import sys,time,json,hashlib,unittest
import numpy as np
import torch
from unittest.mock import patch
r=Path('/home/liwt/GRoPE');sys.path.insert(0,str(r));d=r/'results/round38_joint_development_logs/attempt2'
from experiments import round38_rms_anchor_contract as C
from models import round29_log_rms_projective_pe as MODEL
C.configure_runtime()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pre=json.loads((d/'before_execution.json').read_text());assert all(sha(r/n)==h for n,h in pre['source_hashes'].items())
original=np.load;toy_reads=[]
def guarded(path,*args,**kwargs):
 p=Path(path).resolve()
 if not p.is_relative_to(Path('/tmp')):raise AssertionError('no actual archives in joint development tests: '+str(p))
 toy_reads.append(str(p));return original(path,*args,**kwargs)
started=time.time()
with patch.object(np,'load',side_effect=guarded),patch.object(MODEL,'integrate',side_effect=AssertionError('no real/model integrations in joint mock tests')):
 suite=unittest.defaultTestLoader.discover(str(r/'tests'),pattern='test_round38_rms_anchor_*.py')
 result=unittest.TextTestRunner(verbosity=2).run(suite)
after={n:sha(r/n) for n in pre['source_hashes']}
assert after==pre['source_hashes']
(d/'completion.json').write_text(json.dumps(dict(started_unix=started,finished_unix=time.time(),testsRun=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),source_hashes_after=after,real_array_decodes=0,model_integrations=0,labels_read=False,temporary_toy_archive_reads=toy_reads,runtime=C.check_runtime()),indent=2,sort_keys=True)+'\n')
sys.exit(0 if result.wasSuccessful() else 1)
