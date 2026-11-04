"""Account for a declared metadata failure without replacing a frozen scene."""
from pathlib import Path
import time

from experiments import round6_acquire as acquisition
from experiments.projection_confirmation_metadata import fetch_metadata, listing
from experiments.region_confirmation_common import (
    ROOT, PREP, RAW, read, write, sha, preserved, check_lock)


def finalize():
    check_lock(PREP / 'cohort_lock.json')
    assert not any(RAW.rglob('*.png')) and not any(RAW.rglob('*.jpg'))
    selection = read(PREP / 'scene_selection.json')
    failed = 'home_han/apartment_han_oct_31_2012_scan1_erika'
    failures = sorted(PREP.glob('metadata_failure_*.json'))
    assert failures
    record = read(failures[-1])['failures']
    assert len(record) == 1 and record[0]['scene'] == failed
    assert record[0]['type'] == 'HTTPError' and '404' in record[0]['message']
    assert not (RAW / failed / 'intrinsics.txt').exists()
    # The official listing also has no intrinsics file. Do not invent a camera.
    official_index = PREP / 'home_han_scene_index.html'
    assert 'intrinsics.txt' not in official_index.read_text()
    sources = {str(p): sha(p) for p in [Path(__file__), failures[-1], official_index,
               PREP / 'scene_selection.json']}
    sources.update({str(p): sha(p) for p in (PREP / 'calibration_lookup').glob('*') if p.is_file()})
    preserved(PREP / 'availability_resolution.json', dict(created_unix=time.time(),
        planned_scenes=selection['scenes'], unavailable=record,
        decision='Evaluate the nine available members of the original ten-place cohort; no substitute, guessed calibration, new fit or outcome-dependent selection. Report the missing place and do not call this complete ten-place confirmation.',
        official_reader='https://github.com/jianxiongxiao/SUN3Dtoolbox/blob/master/SUN3Dreader.m',
        official_reader_observation='The official reader directly loads intrinsics.txt; it provides no default camera in this read path.',
        source_sha256=sources, pixels_read=False, original_plan_sha256=sha(ROOT / 'docs/region_confirmation_plan.md')))
    acquisition.fetch, acquisition.listing = fetch_metadata, listing
    rows, provenance = [], []
    for scene in selection['scenes']:
        if scene == failed:
            continue
        row, records = acquisition.one_scene(RAW, 'test', scene)
        rows.append(row)
        provenance.extend(records)
    assert len(rows) == 9 and sum(len(r['pairs']) for r in rows) == 864
    provenance.sort(key=lambda r: r['path'])
    preserved(RAW / 'selection.json', dict(scenes=rows, planned_scenes=selection['scenes'], unavailable=record,
        cohort_lock_sha256=sha(PREP / 'cohort_lock.json'),
        availability_resolution_sha256=sha(PREP / 'availability_resolution.json')))
    preserved(RAW / 'metadata_manifest.json', provenance)
    preserved(PREP / 'metadata_accounted.json', dict(scenes=[r['scene'] for r in rows],
        planned_scenes=selection['scenes'], unavailable=record, pairs=864, planned_pairs=960,
        complete_cohort=False, selection_sha256=sha(RAW / 'selection.json'),
        manifest_sha256=sha(RAW / 'metadata_manifest.json'),
        availability_resolution_sha256=sha(PREP / 'availability_resolution.json'),
        completed_unix=time.time(), pixels_read=False))
    print('REGION_METADATA_ACCOUNTED', '9 available / 10 planned; no pixels, no replacement', flush=True)


if __name__ == '__main__':
    finalize()
