"""Resource-only continuation: four raw-byte fetches, original cache and inference."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np

from experiments import native_confirmation_workflow as workflow
from experiments.native_bounded_prefetch import BoundedPrefetch
from experiments.native_stream_cache import atomic_json
from experiments.round65_common import ROOT, sha

REV = workflow.OUT / 'parallel_download1'


def check():
    record = json.loads((REV / 'runtime_record.json').read_text())
    assert record['evaluation_lock_sha256'] == sha(workflow.OUT / 'evaluation_lock.json')
    assert record['workers'] == 4 and record['scientific_settings_changed'] is False
    for path, digest in record['source_sha256'].items():
        assert sha(path) == digest, path
    for path, digest in record['validation_sha256'].items():
        assert sha(path) == digest, path
    assert json.loads((REV / 'units.json').read_text())['returncode'] == 0
    for mode in ('local', 'remote'):
        validation = json.loads((REV / 'validation' / (mode + '_result.json')).read_text())
        assert validation['all_array_values_exact'] and validation['all_conversion_stats_exact']
        assert validation['all_array_files_identical_to_prior']
    workflow.checked_seal()
    return record


def run():
    record = check()
    original_provider = workflow.MirrorProvider
    providers = []

    def factory(rows, transport_directory):
        provider = BoundedPrefetch(rows, original_provider(rows, transport_directory),
            workflow.CACHE.parent / 'prefetch_runtime1', workers=record['workers'],
            minimum_free_bytes=workflow.RESERVE_BYTES)
        providers.append(provider)
        print('BOUNDED_PREFETCH', json.dumps(provider.snapshot()), flush=True)
        return provider

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    workflow.MirrorProvider = factory
    try:
        acquisition = workflow.acquire()
        print('ACQUISITION_COMPLETE', json.dumps(acquisition), flush=True)
    finally:
        for provider in providers:
            provider.close()
        atomic_json(REV / ('prefetch_usage_' + str(time.time_ns()) + '.json'),
                    dict(runtime_record_sha256=sha(REV / 'runtime_record.json'),
                         providers=[p.snapshot() for p in providers]))
        workflow.MirrorProvider = original_provider
    # Unchanged whole-idle GPU selection, inference, model identities and report.
    workflow.queue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'run'))
    args = parser.parse_args()
    if args.action == 'check':
        check()
        print('PREFETCH_RUNTIME_CHECKED')
    else:
        run()
