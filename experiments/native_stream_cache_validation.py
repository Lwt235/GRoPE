"""Exercise bounded native caching against six fixed, already-used fit objects."""
import argparse
import faulthandler
import json
from pathlib import Path
import resource
import time

import numpy as np

from experiments.native_stream_cache import build_cache, LocalProvider, MirrorProvider, atomic_json
from experiments.round65_common import ROOT, RAW, sha

OUT = ROOT / 'results/native_stream_cache_validation/attempt1'
OLD = ROOT / 'results/round65_actual_gs/attempt1'


class PlannedInterruption(RuntimeError):
    pass


def compare(rows, destination):
    manifest = json.loads((OLD / 'data/manifest.json').read_text())
    partition = next(r for r in manifest['partitions'] if r['split'] == 'fit')
    indices = {identity:index for index,identity in enumerate(partition['ids'])}
    reference_stats = {r['path']:r for r in json.loads((OLD / 'data/fit/conversion.json').read_text())}
    progress = json.loads((destination / 'progress.json').read_text())
    fields, all_exact = {}, True
    for name in ('centers','covariance','appearance','indices','labels'):
        original = np.load(OLD / 'data/fit' / (name + '.npy'), mmap_mode='r')
        current = np.load(destination / (name + '.npy'), mmap_mode='r')
        records = []
        for index,row in enumerate(rows):
            expected, actual = original[indices[row['file_path']]], current[index]
            exact = bool(np.array_equal(expected, actual))
            error = float(np.max(np.abs(np.asarray(expected,dtype=np.float64)-np.asarray(actual,dtype=np.float64))))
            records.append(dict(file_path=row['file_path'], exact=exact, maximum_absolute_error=error))
            all_exact &= exact
        fields[name] = records
    stat_records = []
    for record in progress['completed']:
        old = reference_stats[record['file_path']]
        expected = {key:old[key] for key in record['stats']}
        stat_records.append(dict(file_path=record['file_path'], exact=record['stats']==expected))
    return dict(fields=fields, all_array_values_exact=all_exact, conversion_stats=stat_records,
                all_conversion_stats_exact=all(r['exact'] for r in stat_records))


def main(mode):
    plan = json.loads((OUT / 'plan.json').read_text())
    for path,digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    identity = json.loads((OUT / 'implementation.json').read_text())
    for path,digest in identity['source_sha256'].items():
        assert sha(path) == digest, path
    rows = plan['local_rows'] if mode == 'local' else plan['remote_rows']
    if mode == 'remote':
        assert json.loads((OUT / 'local_result.json').read_text())['all_array_values_exact']
    destination = OUT / (mode + '_cache')
    assert not (OUT / (mode + '_result.json')).exists()
    started = time.time()
    atomic_json(OUT / (mode + '_status.json'), dict(state='running', started_unix=started))
    provider = LocalProvider(RAW) if mode == 'local' else MirrorProvider(rows, OUT / 'mirror_transport')
    planned_resume = False
    try:
        if mode == 'local':
            def stop_after_second(index, progress):
                if index == 1:
                    raise PlannedInterruption('Validate resumption after two committed real objects')
            try:
                build_cache(rows,destination,provider,tokens=plan['cache_tokens'],
                            minimum_free_bytes=plan['storage']['validation_minimum_free_bytes'],after_row=stop_after_second)
            except PlannedInterruption:
                planned_resume=True
                atomic_json(OUT / 'planned_resume.json', dict(committed_before_resume=2,
                    progress_sha256=sha(destination / 'progress.json'), budget_or_source_changed=False))
        manifest = build_cache(rows,destination,provider,tokens=plan['cache_tokens'],
                               minimum_free_bytes=plan['storage']['validation_minimum_free_bytes'])
        result = compare(rows,destination)
        result.update(mode=mode, objects=len(rows), source_bytes=sum(r['size'] for r in rows),
            planned_resume_exercised=planned_resume, seconds=time.time()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            manifest_sha256=sha(destination / 'manifest.json'),
            maximum_single_staged_object_bytes=manifest['maximum_raw_staging_bytes'],
            cache_array_bytes=manifest['cache_array_bytes'],
            staged_payloads_remaining=[p.name for p in (destination/'raw_stage').iterdir() if p.name in ('raw.pending','raw.verified.ply')],
            implementation_sha256=sha(OUT / 'implementation.json'),plan_sha256=sha(OUT / 'plan.json'),
            old_cache_manifest_sha256=sha(OLD / 'data/manifest.json'),
            original_raw_files_modified=False,official_test_inputs_read=False,model_predictions_generated=False)
        atomic_json(OUT / (mode + '_result.json'),result)
        assert result['all_array_values_exact'] and result['all_conversion_stats_exact']
        assert not result['staged_payloads_remaining']
        atomic_json(OUT / (mode + '_status.json'),dict(state='complete',result_sha256=sha(OUT/(mode+'_result.json')),seconds=result['seconds']))
        print('VALIDATED',mode,result['objects'],'array_exact',result['all_array_values_exact'],
              'stats_exact',result['all_conversion_stats_exact'],'seconds',result['seconds'],flush=True)
    except BaseException as error:
        atomic_json(OUT / (mode + '_failure.json'),dict(error_type=type(error).__name__,message=str(error),
            source_sha256=identity['source_sha256'],seconds=time.time()-started))
        atomic_json(OUT / (mode + '_status.json'),dict(state='failed',error_type=type(error).__name__))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('local','remote'))
    args=parser.parse_args()
    faulthandler.dump_traceback_later(60,repeat=True)
    main(args.mode)
