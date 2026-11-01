from pathlib import Path
import sys,time,json,unittest
import numpy as np
from unittest.mock import patch
r=Path('/home/liwt/GRoPE');sys.path.insert(0,str(r));d=r/'results/round38_review_development_logs/attempt1'
from experiments import round38_rms_anchor_contract as C
from models import round29_log_rms_projective_pe as MODEL
C.configure_runtime()
pre=C.read(d/'before_execution.json');assert all(C.sha(r/n)==h for n,h in pre['source_hashes'].items())
started=time.time()
with patch.object(np,'load',side_effect=AssertionError('no archive decodes in review tests')),patch.object(MODEL,'integrate',side_effect=AssertionError('no integrations in review tests')):
 suite=unittest.defaultTestLoader.discover(str(r/'tests'),pattern='test_round38_rms_anchor_audit.py')
 result=unittest.TextTestRunner(verbosity=2).run(suite)
after={n:C.sha(r/n) for n in pre['source_hashes']};assert after==pre['source_hashes']
C.write(d/'completion.json',dict(started_unix=started,finished_unix=time.time(),testsRun=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),source_hashes_after=after,real_array_decodes=0,model_integrations=0,labels_read=False,runtime=C.check_runtime()))
sys.exit(0 if result.wasSuccessful() else 1)
