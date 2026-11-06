"""Check the changed resource width on the same six previously used fit files."""
import json
from pathlib import Path
import time

import numpy as np

from experiments.native_bounded_prefetch import BoundedPrefetch
from experiments.native_stream_cache import LocalProvider, atomic_json, build_cache
from experiments.native_stream_cache_validation import compare, OUT as PRIOR
from experiments.round65_common import RAW, sha


OUT = Path(__file__).resolve().parent / 'validation'
OUT.mkdir(exist_ok=True)
assert not (OUT / 'local_result.json').exists()
plan = json.loads((PRIOR / 'plan.json').read_text())
rows = plan['local_rows']
started = time.time()
with BoundedPrefetch(rows, LocalProvider(RAW), OUT / 'spool', workers=8,
                     minimum_free_bytes=plan['storage']['validation_minimum_free_bytes']) as provider:
    manifest = build_cache(rows, OUT / 'cache', provider, tokens=plan['cache_tokens'],
                           minimum_free_bytes=plan['storage']['validation_minimum_free_bytes'])
usage = provider.snapshot()
comparison = compare(rows, OUT / 'cache')
reference = json.loads((PRIOR / 'local_cache/manifest.json').read_text())
assert manifest['array_sha256'] == reference['array_sha256']
assert comparison['all_array_values_exact'] and comparison['all_conversion_stats_exact']
assert usage['closed'] and not list((OUT / 'spool').iterdir())
record = dict(workers=8, objects=len(rows), source_bytes=sum(r['size'] for r in rows),
              seconds=time.time()-started, all_array_files_identical_to_prior=True,
              **comparison, usage=usage, closed=usage['closed'],
              manifest_sha256=sha(OUT / 'cache/manifest.json'),
              old_fit_only=True, official_test_inputs_read=False, new_model_predictions=0)
atomic_json(OUT / 'local_result.json', record)
print(json.dumps({key:value for key,value in record.items() if key not in ('fields','conversion_stats')}, indent=2))
