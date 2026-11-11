"""Freeze, acquire and evaluate new-place data with unchanged shared-write models."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import traceback

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_attention_train import write_json
from experiments import gaussian_memory_joint_write as joint
from experiments import gaussian_memory_summary_data as summary

OUT = ROOT / 'results/gaussian_memory/write_confirmation1'
CACHE = DATA / 'write_confirmation_v1'
RAW = CACHE / 'raw'
DOC = ROOT / 'docs/gaussian_memory_write_confirmation_plan.md'
INVENTORY = ROOT / 'results/research_data_usage/sun3d_remaining_after_region_rotary_access_20261110.json'
ACCESS = ROOT / 'results/research_data_usage/gaussian_memory_write_confirmation_access_20261110.json'
SALT = 'GRoPE-shared-writer-confirmation-20261110-v1'
QUOTAS = {'brown': 4, 'harvard': 4, 'hotel': 4, 'mit': 8}
REF = [0, 4, 8, 12]
QUERY = [1, 2, 3, 5, 6, 7, 9, 10, 11]


def freeze():
    old = joint.checked(True); inventory = json.loads(INVENTORY.read_text())
    OUT.mkdir(parents=True, exist_ok=True); assert not (OUT / 'plan.json').exists()
    excluded = set(inventory['excluded_union']); candidates = set(inventory['remaining_directory_candidates'])
    provenance = {str(INVENTORY): sha256(INVENTORY)}
    for name, digest in inventory['source_sha256'].items():
        assert sha256(name) == digest, name
    selections = list(DATA.parent.glob('*/raw/selection.json')) + list((ROOT / 'data').glob('round*_sun3d/selection*.json'))
    for name in selections:
        record = json.loads(name.read_text()); provenance[str(name)] = sha256(name)
        used = {r['scene'].split('/')[0] for r in record.get('scenes', []) if isinstance(r, dict) and 'scene' in r}
        excluded.update(used)
    excluded.update(m['row']['scene'].split('/')[0] for m in old['catalog'])
    candidates -= excluded; chosen = []
    for prefix, count in QUOTAS.items():
        pool = sorted((p for p in candidates if p.startswith(prefix + '_')),
                      key=lambda p: (hashlib.sha256((SALT + ':' + p).encode()).hexdigest(), p))
        assert len(pool) >= count
        chosen.extend(pool[:count])
    assert len(set(chosen)) == 20 and not set(chosen) & excluded
    models = json.loads((joint.OUT / 'evaluation_lock.json').read_text())['models']
    source = dict(old['source_sha256'])
    for name in ('experiments/gaussian_memory_write_confirmation.py', 'experiments/projection_confirmation_metadata.py',
                 'experiments/round6_acquire.py', 'experiments/round5_acquire.py'):
        source[name] = sha256(ROOT / name)
    plan = {'places': chosen, 'excluded_places': sorted(excluded), 'remaining_candidate_count': len(candidates),
            'salt': SALT, 'quotas': QUOTAS, 'center_variance': old['center_variance'],
            'seeds': joint.SEEDS, 'kinds': joint.KINDS, 'models': models, 'source_sha256': source,
            'document_sha256': sha256(DOC), 'selection_sources_sha256': provenance,
            'prior_plan_sha256': sha256(joint.OUT / 'plan.json'), 'prior_report_sha256': sha256(joint.OUT / 'report.md'),
            'PCA_sha256': sha256(DATA / 'cache_v1/pca.pt'),
            'backbone_identity_sha256': sha256(DATA / 'source_identity.json'),
            'new_fits': 0, 'physical_alias_independence_guaranteed': False,
            'cohort_rule': 'first lexical scan, first 13 sorted frames of unchanged metadata rule',
            'fresh_scope': 'unused SUN3D place-directory identities in project usage records'}
    write_json(OUT / 'plan.json', plan)
    write_json(OUT / 'evaluation_lock.json', {'models': models, 'plan_sha256': sha256(OUT / 'plan.json')})
    write_json(OUT / 'status.json', {'complete': False, 'pending': ['metadata', 'download', 'extract', 'evaluate', 'analysis', 'checks', 'review', 'report']})
    print(json.dumps({'frozen_models': len(models), 'new_fits': 0, 'places': chosen, 'excluded': len(excluded)}), flush=True)


def checked():
    plan = json.loads((OUT / 'plan.json').read_text())
    assert sha256(DOC) == plan['document_sha256']
    assert sha256(OUT / 'plan.json') == json.loads((OUT / 'evaluation_lock.json').read_text())['plan_sha256']
    for name, digest in plan['source_sha256'].items():
        assert sha256(ROOT / name) == digest, name
    for name, digest in plan['models'].items():
        assert sha256(name) == digest, name
    assert sha256(DATA / 'cache_v1/pca.pt') == plan['PCA_sha256']
    return plan


def metadata():
    from experiments import round6_acquire as acquisition
    from experiments.projection_confirmation_metadata import fetch_metadata, listing
    from experiments.round5_rgbd_data import read_sun3d_intrinsics, read_sun3d_extrinsics
    plan = checked(); RAW.mkdir(parents=True, exist_ok=True)
    acquisition.fetch, acquisition.listing = fetch_metadata, listing
    results = []; failures = []
    def one(place):
        path = RAW / place / 'scan_index.html'
        fetch_metadata('https://sun3d.cs.princeton.edu/data/' + place + '/', path)
        children = sorted(set(re.findall(r'href="([A-Za-z0-9_-]+)/"', path.read_text())))
        if not children:
            raise ValueError('No child scan')
        row, records = acquisition.one_scene(RAW, 'test', place + '/' + children[0])
        read_sun3d_intrinsics(RAW / row['scene'] / 'intrinsics.txt')
        read_sun3d_extrinsics(RAW / row['pose_file'])
        return row, records
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        tasks = {pool.submit(one, p): p for p in plan['places']}
        for future in concurrent.futures.as_completed(tasks):
            place = tasks[future]
            try:
                row, records = future.result(); results.append({'place': place, 'row': row, 'records': records})
            except Exception as error:
                failures.append({'place': place, 'type': type(error).__name__, 'message': str(error)})
            write_json(OUT / 'metadata_progress.json', {'completed': [r['place'] for r in results], 'failures': failures})
    results.sort(key=lambda r: plan['places'].index(r['place']))
    cohort = []
    for item in results:
        row = item['row']; frames = sorted(row['frames'], key=lambda f: f['index'])[:13]
        assert len(frames) == 13
        record = {'id': f'fresh_{plan["places"].index(item["place"]):02d}', 'scene': row['scene'], 'split': 'confirmation',
                  'root': str(RAW), 'intrinsics': str(RAW / row['scene'] / 'intrinsics.txt'),
                  'poses': str(RAW / row['pose_file']), 'intrinsics_sha256': row['K_sha256'],
                  'poses_sha256': row['pose_sha256'], 'reference': REF, 'training_query': [],
                  'heldout_query': QUERY, 'frames': []}
        for f in frames:
            record['frames'].append({'index': f['index'], 'rgb_path': str(RAW / row['scene'] / 'image' / f['rgb']),
                'rgb_url': row['base_url'] + 'image/' + f['rgb'],
                'depth_path': str(RAW / row['scene'] / 'depth' / f['depth']),
                'depth_url': row['base_url'] + 'depth/' + f['depth'], 'time_gap_us': f['time_gap_us']})
        cohort.append(record)
    write_json(RAW / 'selection.json', {'scenes': [r['row'] for r in results], 'plan_sha256': sha256(OUT / 'plan.json')})
    write_json(OUT / 'cohort.json', {'cohort': cohort, 'metadata_failures': failures,
        'planned_places': plan['places'], 'records': [p for r in results for p in r['records']], 'pixels_read': False})
    print(json.dumps({'metadata_available': len(cohort), 'planned': 20, 'failures': failures}), flush=True)


def download():
    import requests
    plan = checked(); cohort = json.loads((OUT / 'cohort.json').read_text())['cohort']
    write_json(ACCESS, {'experiment': 'write_confirmation1', 'places': plan['places'],
        'started_unix': time.time(), 'plan_sha256': sha256(OUT / 'plan.json'),
        'reserved_places_consumed_regardless_of_download_outcome': True, 'previously_used': False})
    jobs = {}
    for row in cohort:
        for i, frame in enumerate(row['frames']):
            for kind in (('rgb', 'depth') if i in QUERY else ('rgb',)):
                jobs[frame[kind + '_path']] = (frame[kind + '_url'], kind)
    local = threading.local()
    def one(item):
        name, (url, kind) = item; dest = Path(name); dest.parent.mkdir(parents=True, exist_ok=True)
        assert url.startswith('https://sun3d.cs.princeton.edu/data/')
        if not hasattr(local, 'session'):
            local.session = requests.Session(); local.session.trust_env = False
        errors = []
        for attempt in range(5):
            try:
                if not dest.exists():
                    response = local.session.get(url, timeout=(20, 45)); response.raise_for_status()
                    partial = dest.with_suffix(dest.suffix + '.partial'); partial.write_bytes(response.content)
                    with Image.open(partial) as image:
                        image.verify()
                    partial.replace(dest)
                with Image.open(dest) as image:
                    image.verify()
                return {'path': str(dest), 'sha256': sha256(dest), 'bytes': dest.stat().st_size, 'kind': kind, 'recovered_errors': errors}
            except Exception as error:
                errors.append({'attempt': attempt + 1, 'type': type(error).__name__})
                if attempt < 4:
                    time.sleep(min(5 * (attempt + 1), 20))
        return {'path': str(dest), 'failed': True, 'errors': errors, 'kind': kind}
    records = []; failures = []; start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for r in pool.map(one, jobs.items()):
            (failures if r.get('failed') else records).append(r)
            if (len(records) + len(failures)) % 20 == 0:
                write_json(OUT / 'download_progress.json', {'downloaded': len(records), 'total': len(jobs), 'failures': failures, 'proxy_used': False})
                print(json.dumps({'downloaded': len(records), 'total': len(jobs), 'failures': len(failures)}), flush=True)
    digests = {r['path']: r['sha256'] for r in records}; usable = []; unavailable = []
    for row in cohort:
        needed = [f['rgb_path'] for f in row['frames']] + [row['frames'][i]['depth_path'] for i in QUERY]
        missing = [p for p in needed if p not in digests]
        if missing:
            unavailable.append({'id': row['id'], 'scene': row['scene'], 'missing': missing}); continue
        for i, frame in enumerate(row['frames']):
            frame['rgb_sha256'] = digests[frame['rgb_path']]
            if i in QUERY:
                frame['depth_sha256'] = digests[frame['depth_path']]
        usable.append(row)
    write_json(OUT / 'download_complete.json', {'complete': True, 'rows': records, 'failures': failures,
        'unavailable': unavailable, 'usable_cohort': usable, 'bytes': sum(r['bytes'] for r in records),
        'seconds': time.monotonic() - start, 'proxy_used': False, 'planned_places': 20})
    print(json.dumps({'download_accounted': len(records), 'bytes': sum(r['bytes'] for r in records), 'usable_places': len(usable)}), flush=True)


def extract():
    # Reuse the exact, already checked VGGT/PCA/target extraction implementation.
    plan = checked(); downloaded = json.loads((OUT / 'download_complete.json').read_text())
    params = dict(plan, cohort=downloaded['usable_cohort'])
    assert params['cohort'], 'No usable fixed-cohort places'
    summary.OUT, summary.CACHE, summary.SOURCE = OUT, CACHE, RAW
    summary.locked = lambda: params
    summary.extract()


def evaluate():
    plan = checked(); cache = json.loads((OUT / 'cache_identity.json').read_text())
    for name, digest in cache['sha256'].items():
        assert sha256(name) == digest, name
    # Keep the existing reader/writer factory and checkpoint paths. Only catalog and output differ.
    params = dict(plan, catalog=cache['catalog'])
    joint.OUT = OUT
    original_path = joint.model_path
    old_out = ROOT / 'results/gaussian_memory/joint_write1'
    def path(kind, seed, initial=False):
        if kind not in joint.NEW:
            return summary.path(kind, seed, initial)
        return old_out / f'{kind}_{seed}' / ('initial.pt' if initial else 'step_2000.pt')
    joint.model_path = path
    joint.checked = lambda locked=False: params
    joint.evaluate()
    scores_path = OUT / 'evaluation/scores.json'; scores = json.loads(scores_path.read_text())
    for key in ('rows', 'full_query', 'updates', 'counts'):
        for r in scores[key]:
            assert r['split'] == 'used_summary20'
            r['split'] = 'fresh_confirmation'
    for filename in scores['maps_sha256']:
        maps = torch.load(filename, map_location='cpu', weights_only=False)
        maps['split'] = 'fresh_confirmation'
        torch.save(maps, filename)
        scores['maps_sha256'][filename] = sha256(filename)
    scores['confirmation_claim'] = True
    scores['freshness_scope'] = plan['fresh_scope']
    write_json(scores_path, scores)
    joint.model_path = original_path


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('freeze', 'metadata', 'download', 'extract', 'evaluate'))
    args = p.parse_args(); start = time.monotonic()
    try:
        globals()[args.stage]()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / f'failure_{args.stage}_{int(time.time())}.json',
                   {'stage': args.stage, 'traceback': traceback.format_exc(), 'scientific_result': False})
        raise
    write_json(OUT / f'{args.stage}_stage_complete.json', {'complete': True, 'seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
