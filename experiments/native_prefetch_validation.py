"""Check bounded prefetch on the same six old-fit objects and two mirror objects."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.native_bounded_prefetch import BoundedPrefetch
from experiments.native_stream_cache import build_cache, LocalProvider, MirrorProvider, atomic_json
from experiments.native_stream_cache_validation import compare, OUT as PRIOR
from experiments.round65_common import RAW, ROOT, sha

OUT = ROOT / 'results/native_confirmation/attempt1/parallel_download1/validation'


def validate(mode):
    plan = json.loads((PRIOR / 'plan.json').read_text())
    rows = plan['local_rows'] if mode == 'local' else plan['remote_rows']
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / (mode + '_result.json')).exists()
    started = time.time()
    provider = LocalProvider(RAW) if mode == 'local' else MirrorProvider(rows, OUT / 'mirror_transport')
    destination = OUT / (mode + '_cache')
    with BoundedPrefetch(rows, provider, OUT / (mode + '_spool'), workers=4,
                         minimum_free_bytes=plan['storage']['validation_minimum_free_bytes']) as prefetched:
        manifest = build_cache(rows, destination, prefetched, tokens=plan['cache_tokens'],
                               minimum_free_bytes=plan['storage']['validation_minimum_free_bytes'])
        usage = prefetched.snapshot()
    comparison = compare(rows, destination)
    reference = json.loads((PRIOR / (mode + '_cache/manifest.json')).read_text())
    assert manifest['array_sha256'] == reference['array_sha256']
    assert comparison['all_array_values_exact'] and comparison['all_conversion_stats_exact']
    assert not list((OUT / (mode + '_spool')).iterdir())
    record = dict(mode=mode, objects=len(rows), source_bytes=sum(r['size'] for r in rows),
        seconds=time.time()-started, all_array_files_identical_to_prior=True,
        **comparison, usage=usage, prior_plan_sha256=sha(PRIOR / 'plan.json'),
        manifest_sha256=sha(destination / 'manifest.json'),
        source_sha256={str(p):sha(p) for p in (Path(__file__), ROOT/'experiments/native_bounded_prefetch.py',
                                               ROOT/'experiments/native_stream_cache.py')},
        old_fit_objects_only=True, official_test_inputs_read=False, model_predictions_generated=False,
        scientific_settings_changed=False)
    atomic_json(OUT / (mode + '_result.json'), record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('fields','conversion_stats','source_sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('local','remote'))
    validate(parser.parse_args().mode)
