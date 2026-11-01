"""Run frozen toys and metadata-only joint freezes with real loads disabled."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

ROOT = Path('/home/liwt/GRoPE')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
pre = json.loads((HERE / 'before_execution.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(__file__) == pre['controller_sha256']
for name, digest in pre['source_hashes'].items():
    assert sha(ROOT / name) == sha(HERE / 'source_snapshot' / name) == digest, name

import numpy as np
from experiments import round38_rms_anchor_contract as C
C.configure_runtime()
def reject_load(*args, **kwargs):
    raise RuntimeError('preflight must not decode real arrays')
np.load = reject_load
try:
    np.load('deliberate_rejection_probe')
except RuntimeError:
    pass
else:
    raise AssertionError('array guard inactive')
from experiments import round38_saved_output_diagnostic as P
guard = P._NoModels()
sys.meta_path.insert(0, guard)
try:
    guard.find_spec('models.deliberate_rejection_probe')
except RuntimeError:
    pass
else:
    raise AssertionError('model guard inactive')
suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_round38_saved_output_diagnostic')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)
producer, audit = HERE / 'metadata_producer', HERE / 'metadata_audit'
P.freeze(producer, 'producer', producer, audit)
P.freeze(audit, 'audit', producer, audit)
P.verify_before(producer)
for name, digest in pre['source_hashes'].items():
    assert sha(ROOT / name) == sha(HERE / 'source_snapshot' / name) == digest, name
print('TOYS_AND_METADATA_FREEZES_PASS', result.testsRun, flush=True)
