import os,sys,json,time,unittest,hashlib
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch
r=Path('/home/liwt/GRoPE');sys.path.insert(0,str(r));d=r/'results/round38_inputs_development_logs/attempt2'
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
pre=json.loads((d/'before_execution.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(r/n)==h for n,h in pre['source_hashes'].items())
started=time.time()
with patch.object(np,'load',side_effect=AssertionError('no archive access')):
 suite=unittest.defaultTestLoader.discover(str(r/'tests'),pattern='test_round38_rms_anchor_inputs.py')
 result=unittest.TextTestRunner(verbosity=2).run(suite)
after={n:sha(r/n) for n in pre['source_hashes']}
assert after==pre['source_hashes']
(d/'completion.json').write_text(json.dumps(dict(started_unix=started,finished_unix=time.time(),testsRun=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),source_hashes_after=after,real_array_decodes=0,model_integrations=0,labels_read=False),indent=2,sort_keys=True)+'\n')
sys.exit(0 if result.wasSuccessful() else 1)
