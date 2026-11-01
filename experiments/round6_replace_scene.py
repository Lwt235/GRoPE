"""Preserve v1 and replace possible Brown validation/test overlap before scoring.

Only fresh metadata and selected public RGB/depth files are acquired. No label
exporter, matcher, depth predictor or evaluation routine is imported or run.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.round6_acquire import ROOT, OUT, one_scene, write_preserving
from experiments.round5_acquire import fetch, sha


REJECTED = 'brown_bm_1/brown_bm_1'
REPLACEMENT = 'home_bksh/home_bksh_oct_30_2012_scan2_erika'
V1_SHA256 = 'c7297b2cd040c271c49a90cf85e1d1e7893c4c57b2f77119a16580b72bad9631'


def metadata(out=OUT):
    out = Path(out)
    original = out/'selection.json'
    if sha(original) != V1_SHA256:
        raise ValueError('original v1 selection changed; preserve the agreed source')
    v1 = json.loads(original.read_text())
    row, records = one_scene(out, 'test', REPLACEMENT)
    scenes = [row if scene['scene'] == REJECTED else scene for scene in v1['scenes']]
    if sum(scene['scene'] == REJECTED for scene in v1['scenes']) != 1:
        raise ValueError('expected exactly one rejected v1 scene')
    if any(scene['scene'] == REJECTED for scene in scenes):
        raise ValueError('rejected scene remained in v2')
    places = [scene['scene'].split('/')[0] for scene in scenes]
    if len(places) != len(set(places)) or set(places) & set(v1['previous_place_ids']):
        raise ValueError('v2 fresh place collision')
    v2 = dict(v1, scenes=scenes, scene_split_version=2, source_sha256=sha(__file__),
        upstream_selection_sha256=sha(original),
        helper_source_sha256=sha(ROOT/'experiments/round6_acquire.py'),
        replacement=dict(rejected=REJECTED, replacement=REPLACEMENT,
            reason='possible same-building physical overlap with Brown bm4 validation cannot be excluded from fixed RGB previews; excluded before test labels or model outcomes'))
    write_preserving(out/'selection_v2.json', v2)
    write_preserving(out/'metadata_manifest_v2_addition.json', records)

    # Retain the exact fixed representative identities used for the decision.
    representatives = []
    for scene in v1['scenes']:
        if scene['scene'] not in [REJECTED, 'brown_bm_4/brown_bm_4']:
            continue
        for index in np.linspace(0, len(scene['frames'])-1, 12).astype(int):
            frame = scene['frames'][index]
            path = out/scene['scene']/'image'/frame['rgb']
            representatives.append(dict(scene=scene['scene'], original_index=frame['index'],
                rgb_path=str(path.relative_to(out)), rgb_sha256=sha(path)))
    rejected = dict(rejected_scene=REJECTED, replacement_scene=REPLACEMENT,
        original_selection_sha256=sha(original), replacement_selection_sha256=sha(out/'selection_v2.json'),
        reason=v2['replacement']['reason'],
        observation='both scans include offices and corridors with similar carpet/interior style; no proof of complete physical separation, and no assertion that they are identical rooms',
        decision_inputs='official scene lists and fixed first-to-last RGB representatives only; no GT labels/model scores',
        rejected_data_status='retained as auxiliary development data, excluded from new confirmation test',
        source_sha256=sha(__file__), representatives=representatives)
    write_preserving(out/'rejection_bm1.json', rejected)
    audit = ROOT/'results/round6_split_audit'
    audit.mkdir(parents=True, exist_ok=True)
    source = audit/'source'
    source.mkdir(exist_ok=True)
    for path in [Path(__file__), ROOT/'experiments/round6_acquire.py']:
        target = source/path.name
        if target.exists() and sha(target) != sha(path):
            raise ValueError('split audit source already differs')
        if not target.exists():
            shutil.copy2(path, target)
    preview = Path('/tmp/round6_brown_bm_fixed12.png')
    if preview.exists() and not (audit/preview.name).exists():
        shutil.copy2(preview, audit/preview.name)
    print('V2_SELECTION', sha(out/'selection_v2.json'), 'test_pairs',
          sum(len(scene['pairs']) for scene in scenes if scene['split'] == 'test'), flush=True)
    return v2


def frames(out=OUT, workers=8):
    out = Path(out)
    selection = json.loads((out/'selection_v2.json').read_text())
    if selection['source_sha256'] != sha(__file__) or sha(out/'selection.json') != V1_SHA256:
        raise ValueError('v2 source or original selection changed')
    rows = [scene for scene in selection['scenes'] if scene['scene'] == REPLACEMENT]
    if len(rows) != 1 or any(scene['scene'] == REJECTED for scene in selection['scenes']):
        raise ValueError('invalid v2 replacement')
    scene = rows[0]
    jobs = {}
    for frame in scene['frames']:
        for key, sub in [('rgb', 'image'), ('depth', 'depth')]:
            path = out/scene['scene']/sub/frame[key]
            jobs[str(path)] = (scene['base_url']+sub+'/'+frame[key], path)

    def one(job):
        url, path = job
        fetch(url, path)
        return dict(path=str(path.relative_to(out)), url=url, bytes=path.stat().st_size, sha256=sha(path))

    records = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for record in pool.map(one, jobs.values()):
            records.append(record)
            if len(records) % 50 == 0:
                print('REPLACEMENT_FILES', len(records), '/', len(jobs), flush=True)
    write_preserving(out/'download_manifest_v2_addition.json', records)
    if sha(out/'selection.json') != V1_SHA256:
        raise ValueError('v1 changed during replacement download')
    print('REPLACEMENT_DONE', len(records), 'bytes', sum(row['bytes'] for row in records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', action='store_true')
    parser.add_argument('--frames', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not args.metadata and not args.frames:
        parser.error('choose --metadata and/or --frames')
    if args.workers < 1:
        parser.error('positive --workers required')
    if args.metadata:
        metadata(args.out)
    if args.frames:
        frames(args.out, args.workers)
