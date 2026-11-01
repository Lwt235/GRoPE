"""Acquire five fresh SUN3D scan groups for a sealed downstream V experiment.

Pair selection uses original ordinals/timestamps only. This file imports only
established download/metadata helpers, never label exporters or model code.
Round 5/6 source files and selections are read-only inputs.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path

from experiments.round5_acquire import fetch, sha
from experiments.round6_acquire import one_scene, write_preserving


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/round7_sun3d'
SCENES = [
    'mit_46_6conf/bcs_floor6_conf_1',
    'mit_32_g725/g725_1',
    'mit_w59_conference/conference_w59_nov_6_2012_scan1_erika',
    'harvard_robotics_lab/hv_s1_1',
    'home_md/home_md_scan9_2012_sep_30',
]
PREVIOUS_SELECTIONS = [
    ROOT/'data/round5_sun3d/selection.json',
    ROOT/'data/round6_sun3d/selection.json',
    ROOT/'data/round6_sun3d/selection_v2.json',
]


def metadata(out=OUT, workers=4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    previous = {str(path.relative_to(ROOT)): sha(path) for path in PREVIOUS_SELECTIONS}
    forbidden = {scene['scene'].split('/')[0] for path in PREVIOUS_SELECTIONS
                 for scene in json.loads(path.read_text())['scenes']}
    places = [name.split('/')[0] for name in SCENES]
    if len(places) != 5 or len(set(places)) != 5 or set(places) & forbidden:
        raise ValueError('require five unique fresh top-level place IDs, excluding all previous candidates')
    results, failures = {}, []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one_scene, out, 'test', name): name for name in SCENES}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as error:
                failures.append(dict(scene=name, stage='native metadata integrity', error=str(error)))
    if failures:
        write_preserving(out/'metadata_failures.json', dict(source_sha256=sha(__file__),
            ordered_candidates=SCENES, failures=failures,
            reason='metadata/access integrity only; no labels or model scores read'))
        raise RuntimeError('metadata failure; retain records and choose an explicit new selection version')
    selection = dict(scene_split_version=1, purpose='Round7 fresh held-out downstream depth experiment',
        selection='48 linspace query ordinals30..N-91; offsets15,60; nearest depth timestamp',
        source_sha256=sha(__file__),
        helper_source_sha256={name: sha(ROOT/name) for name in
            ['experiments/round5_acquire.py', 'experiments/round6_acquire.py']},
        previous_selection_sha256=previous, excluded_previous_place_ids=sorted(forbidden),
        no_label_export_or_model_evaluation=True,
        scenes=[results[name][0] for name in SCENES])
    write_preserving(out/'selection.json', selection)
    write_preserving(out/'metadata_manifest.json', [record for name in SCENES for record in results[name][1]])
    print('ROUND7_SELECTION', sha(out/'selection.json'), 'test_pairs',
          sum(len(scene['pairs']) for scene in selection['scenes']), flush=True)
    return selection


def frames(out=OUT, workers=8):
    out = Path(out)
    selection = json.loads((out/'selection.json').read_text())
    if selection['source_sha256'] != sha(__file__):
        raise ValueError('acquisition source changed after metadata selection')
    for path, digest in selection['helper_source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('download helper changed after metadata selection')
    for path, digest in selection['previous_selection_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('previous frozen selection changed')
    jobs = {}
    for scene in selection['scenes']:
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
            if len(records) % 100 == 0:
                print('ROUND7_FILES', len(records), '/', len(jobs), flush=True)
    write_preserving(out/'download_manifest.json', records)
    print('ROUND7_FRAMES_DONE', len(records), 'bytes', sum(record['bytes'] for record in records), flush=True)
    return records


def preview(out=OUT, audit=None):
    """Fixed first/middle/last RGB only; no depth/labels/model-score inspection."""
    from PIL import Image, ImageDraw
    out = Path(out)
    audit = Path(audit or ROOT/'results/round7_split_audit')
    audit.mkdir(parents=True, exist_ok=True)
    selection_path = out/'selection.json'
    selection = json.loads(selection_path.read_text())
    canvas = Image.new('RGB', (960, 264*len(selection['scenes'])), 'white')
    draw, representatives = ImageDraw.Draw(canvas), []
    for row, scene in enumerate(selection['scenes']):
        for column, index in enumerate([0, len(scene['frames'])//2, len(scene['frames'])-1]):
            frame = scene['frames'][index]
            path = out/scene['scene']/'image'/frame['rgb']
            with Image.open(path) as image:
                canvas.paste(image.convert('RGB').resize((320, 240)), (320*column, 264*row+24))
            draw.text((320*column+2, 264*row+2), scene['scene'].split('/')[0]+' idx'+str(frame['index']), fill='black')
            representatives.append(dict(scene=scene['scene'], original_index=frame['index'],
                rgb_path=str(path.relative_to(out)), rgb_sha256=sha(path)))
    destination = audit/'test_rgb_firstmiddlelast.png'
    if destination.exists():
        raise FileExistsError('preserve previous RGB QA preview')
    canvas.save(destination)
    write_preserving(audit/'preview_manifest.json', dict(selection_sha256=sha(selection_path),
        source_sha256=sha(__file__), selection_rule='first/middle/last selected frame by ordinal',
        no_label_or_model_score_access=True, representatives=representatives, preview_sha256=sha(destination)))
    print('ROUND7_RGB_PREVIEW', str(destination), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', action='store_true')
    parser.add_argument('--frames', action='store_true')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not any([args.metadata, args.frames, args.preview]):
        parser.error('choose --metadata, --frames and/or --preview')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.metadata:
        metadata(args.out, args.workers)
    if args.frames:
        frames(args.out, args.workers)
    if args.preview:
        preview(args.out)
