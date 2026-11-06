"""Frozen-model, full-head transfer to seven prospectively named ModelNet classes.

No fitting, calibration, class masking, permission applications, or proxy use.
Acquisition requires the completed old-data CUDA preflight and the source lock.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import torch

from experiments import native_confirmation_runtime as runtime
from experiments import round65_run as runner
from experiments.native_confirmation_analysis import paired_accuracy_summary
from experiments.native_memmap_runtime import install
from experiments.native_stream_cache import build_cache
from experiments.round65_common import ROOT, DATA_ROOT, CONFIG, VARIANTS, read, write, sha
from models.symmetric_gaussian_relation import SymmetricNativeClassifier

OUT = ROOT / 'results/modelnet_cross_source/attempt1'
DATA = DATA_ROOT / 'GRoPE/modelnet_cross_source/attempt1'
META = ROOT / 'results/research_data_usage/native_candidates_20261106/modelnet620_candidate_metadata.json'
PLAN = ROOT / 'docs/modelnet_cross_source_plan.md'
OLD_LOCK = ROOT / 'results/native_confirmation/attempt1/evaluation_lock.json'
SYMMETRIC = ROOT / 'results/symmetric_relation/attempt1'
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
CLASSES = (1, 4, 11, 12, 13, 15, 20)
CAP_BYTES = 16 * 2**30
BASE_FACTORY = runtime.make_model


def model_for(role):
    return SymmetricNativeClassifier(role) if role in ('symmetric3', 'jeffreys_mlp') else BASE_FACTORY(role)


def source_paths():
    pending = [Path(__file__), ROOT / 'tests/test_modelnet_cross_source.py']
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                     [node.module] + [node.module + '.' + n.name for n in node.names]
                     if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                candidate = ROOT / (name.replace('.', '/') + '.py')
                if name.split('.')[0] in ('models', 'experiments') and candidate.is_file():
                    pending.append(candidate)
    return sorted(found)


def virtual_path(row):
    if not re.fullmatch(r'[a-z_]+_\d+', row['model_id']) or row['taxonomy_id'] not in CLASSES:
        raise ValueError('invalid frozen object')
    return f"{row['taxonomy_id']}/{row['model_id']}.ply"


def choose_member(names, model_id):
    """Exact object identity, then largest numeric iteration; no score-based choice."""
    choices = []
    for name in names:
        path = PurePosixPath(name)
        if path.suffix != '.ply' or (model_id not in path.parts and path.stem != model_id):
            continue
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe archive member')
        # A named object's auxiliary PLY is not its optimized Gaussian model.
        if path.stem != model_id and path.name != 'point_cloud.ply':
            continue
        iterations = [int(m.group(1)) for p in path.parts if (m := re.fullmatch(r'iteration_(\d+)', p))]
        if len(iterations) > 1:
            raise ValueError('ambiguous nested iterations')
        choices.append((iterations[0] if iterations else -1, name))
    if not choices:
        raise ValueError('missing exact source object: ' + model_id)
    maximum = max(x[0] for x in choices)
    top = [name for iteration, name in choices if iteration == maximum]
    if len(top) != 1:
        raise ValueError('duplicated maximum-iteration object: ' + model_id)
    return top[0]


def seal():
    if (OUT / 'lock.json').exists():
        raise ValueError('preserve the existing lock')
    candidate = read(META)
    assert sha(META) == 'd783be4eb40ee2a187f5987de07a230cb986d50c30bc092cdd648098bbaf8125'
    assert len(candidate['rows']) == 620 and len(set(virtual_path(r) for r in candidate['rows'])) == 620
    assert set(r['taxonomy_id'] for r in candidate['rows']) == set(CLASSES)
    review = read(SYMMETRIC / 'review.json')
    assert review['scientific_review_complete'] and not review['symmetric3_promoted']
    assert review['result_sha256'] == sha(SYMMETRIC / 'result.json')
    models = read(OLD_LOCK)['models'].copy()
    for r in read(SYMMETRIC / 'native512/selected.json')['records']:
        path = SYMMETRIC / 'native512/runs' / r['role'] / str(r['seed']) / 'best.pt'
        models.append(dict(role=r['role'], seed=r['seed'], cohort='old3', path=str(path),
            sha256=r['checkpoint_sha256'], selected_epoch=r['best_epoch'],
            selection_path=str(SYMMETRIC / 'native512/selected.json'),
            selection_sha256=sha(SYMMETRIC / 'native512/selected.json')))
    assert len(models) == 37 and len({(m['role'], m['seed']) for m in models}) == 37
    for m in models:
        assert sha(m['path']) == m['sha256']
        assert sha(m['selection_path']) == m['selection_sha256']
        selected = next(r for r in read(m['selection_path'])['records']
                        if (r['role'], r['seed']) == (m['role'], m['seed']))
        assert selected['checkpoint_sha256'] == m['sha256']
        assert selected['best_epoch'] == m['selected_epoch']
    dependencies = source_paths() + [PLAN, META, OLD_LOCK, SYMMETRIC / 'review.json',
        SYMMETRIC / 'result.json', OUT / 'cpu_checks.json',
        ROOT / 'results/round65_actual_gs/preparation/train_catalog.json',
        ROOT / 'results/native_confirmation_metadata/attempt1/test_catalog.json']
    assert read(OUT / 'cpu_checks.json')['passed']
    record = dict(created_unix=time.time(),source_sha256={str(p):sha(p) for p in dependencies},
        models=models, candidate=candidate, conditions=list(VARIANTS), config=CONFIG,
        class_ids=list(CLASSES), primary_seeds=list(SEEDS), training_updates=0,
        new_inputs_accessed=False, proxy_allowed=False, full_sixty_class_output=True,
        bootstrap_replicates=10000, bootstrap_seed=80001, maximum_gpu_jobs=6,
        allocator_cap_bytes=CAP_BYTES, overall_goal_complete=False)
    write(OUT / 'lock.json', record)
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('CROSS_SOURCE_SEALED', len(models), sha(OUT / 'lock.json'), flush=True)


def check():
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    assert lock['config'] == CONFIG and lock['conditions'] == list(VARIANTS)
    for name, digest in lock['source_sha256'].items():
        assert sha(name) == digest, name
    for m in lock['models']:
        assert sha(m['path']) == m['sha256']
        assert sha(m['selection_path']) == m['selection_sha256']
    return lock


def cuda_setup():
    runner.configure()
    install(runner)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(CAP_BYTES / torch.cuda.get_device_properties(0).total_memory)
    runtime.make_model = model_for


@torch.inference_mode()
def preflight(shard):
    lock = check()
    if shard not in (0, 1, 2):
        raise ValueError('three fixed preflight shards')
    target = OUT / 'preflight' / f'{shard}.json'
    if target.exists():
        raise ValueError('preserve prior preflight')
    cuda_setup()
    old = ROOT / 'results/round65_actual_gs/attempt1/data'
    partition = next(r for r in read(old / 'manifest.json')['partitions'] if r['split'] == 'val')
    bundle = {k:np.load(old / 'val' / (k + '.npy'), mmap_mode='r')
              for k in ('centers', 'covariance', 'appearance', 'indices', 'labels')}
    bundle['ids'] = np.asarray(partition['ids'])
    records = []
    for m in lock['models'][shard::3]:
        model = model_for(m['role']).cuda().eval()
        model.load_state_dict(torch.load(m['path'],map_location='cpu',weights_only=True)['state_dict'])
        logits, _ = runner.evaluate(model, bundle, 'base')
        with np.load(Path(m['path']).parent / 'validation.npz', allow_pickle=False) as z:
            assert np.array_equal(z['ids'], bundle['ids'])
            error = float(np.max(np.abs(z['logits'] - logits)))
            changes = int((z['logits'].argmax(1) != logits.argmax(1)).sum())
        assert error < 2e-4 and changes == 0
        # Reuse all six established evaluation conditions on four old objects.
        tiny = {k:v[:4] for k,v in bundle.items()}
        peaks = {}
        for condition in VARIANTS:
            torch.cuda.reset_peak_memory_stats()
            x, _ = runner.evaluate(model, tiny, condition)
            assert x.shape == (4, 60) and np.isfinite(x).all()
            peaks[condition] = torch.cuda.max_memory_allocated()
        records.append(dict(role=m['role'],seed=m['seed'],objects=len(logits),
            gpu_validation_max_error=error,classification_changes=changes,peak_allocated=peaks))
        del model
        torch.cuda.empty_cache()
        print('PREFLIGHT_MODEL',m['role'],m['seed'],error,flush=True)
    write(target,dict(passed=True,records=records,lock_sha256=sha(OUT / 'lock.json'),
        new_inputs_accessed=False,training_updates=0,gpu=torch.cuda.get_device_name()))


def ready():
    lock = check()
    rows = []
    for shard in range(3):
        r = read(OUT / 'preflight' / f'{shard}.json')
        assert r['passed'] and r['lock_sha256'] == sha(OUT / 'lock.json')
        rows.extend(r['records'])
    assert {(r['role'],r['seed']) for r in rows} == {(m['role'],m['seed']) for m in lock['models']}
    return lock


def start_access(kind):
    ready()
    marker = OUT / 'data_access_started.json'
    if not marker.exists():
        write(marker,dict(started_unix=time.time(),kind=kind,lock_sha256=sha(OUT / 'lock.json'),
            all_620_ids_to_be_treated_as_used_after_access=True,archive_contains_other_unscored_objects=True))


def download():
    """Direct mirror with an optional user's existing HF credential; no access request."""
    lock = ready()
    from huggingface_hub import get_token
    token = get_token()
    # Explicitly disable proxy use. Do not print credentials or signed redirects.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    archive_root = DATA / 'archives'
    archive_root.mkdir(parents=True, exist_ok=True)
    for row in lock['candidate']['archive_files']:
        path = archive_root / row['rfilename']
        if path.exists():
            assert path.stat().st_size == row['size'] and sha(path) == row['lfs']['sha256']
            continue
        pending = path.with_suffix('.zip.partial')
        if pending.exists():
            raise ValueError('preserve interrupted download for reviewed recovery: ' + str(pending))
        if shutil.disk_usage(DATA).free < row['size'] + 2 * 2**30:
            raise OSError('insufficient disk reserve')
        url = ('https://hf-mirror.com/datasets/' + lock['candidate']['repository'] + '/resolve/' +
               lock['candidate']['revision'] + '/' + row['rfilename'])
        headers = {'Authorization':'Bearer ' + token} if token else {}
        # A failed HEAD writes no data-access marker and never submits gate forms.
        try:
            with opener.open(urllib.request.Request(url,headers=headers,method='HEAD'),timeout=30):
                pass
        except urllib.error.HTTPError as e:
            write(OUT / 'download_access_failure.json',dict(observed_unix=time.time(),
                status=e.code,error_code=e.headers.get('X-Error-Code'),archive=row['rfilename'],
                local_authentication_present=bool(token),access_request_submitted=False))
            raise RuntimeError('dataset access unavailable; see download_access_failure.json') from None
        start_access('archive_download')
        digest, size = hashlib.sha256(), 0
        with opener.open(urllib.request.Request(url,headers=headers),timeout=60) as source, pending.open('xb') as sink:
            for chunk in iter(lambda:source.read(4 * 2**20),b''):
                size += len(chunk)
                if size > row['size']:
                    raise ValueError('archive larger than declared')
                digest.update(chunk);sink.write(chunk)
        assert size == row['size'] and digest.hexdigest() == row['lfs']['sha256']
        pending.replace(path)
        write(path.with_suffix('.verified.json'),dict(bytes=size,sha256=digest.hexdigest(),
            revision=lock['candidate']['revision'],proxy_used=False))
        print('ARCHIVE_VERIFIED',row['rfilename'],size,flush=True)


def prepare():
    lock = ready()
    archives = {}
    for r in lock['candidate']['archive_files']:
        path = DATA / 'archives' / r['rfilename']
        assert path.stat().st_size == r['size'] and sha(path) == r['lfs']['sha256']
        archives[path.stem] = zipfile.ZipFile(path)
    start_access('archive_objects')
    try:
        rows = []
        prior_files = [ROOT / 'results/round65_actual_gs/preparation/train_catalog.json',
                       ROOT / 'results/native_confirmation_metadata/attempt1/test_catalog.json']
        used_sha = {r['sha256'] for p in prior_files for r in read(p)['rows']}
        for r in lock['candidate']['rows']:
            archive = archives[r['category']]
            member = choose_member(archive.namelist(),r['model_id'])
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for chunk in iter(lambda:stream.read(2**20),b''):
                    digest.update(chunk)
            rows.append(dict(r,file_path=virtual_path(r),source='ModelNet',archive=r['category']+'.zip',
                member=member,size=archive.getinfo(member).file_size,sha256=digest.hexdigest(),
                known_byte_reuse=digest.hexdigest() in used_sha))
        assert len(rows) == 620
        primary = [r for r in rows if not r['known_byte_reuse']]
        assert set(r['taxonomy_id'] for r in primary) == set(CLASSES)
        catalog = dict(rows=rows,repository=lock['candidate']['repository'],revision=lock['candidate']['revision'],
            known_byte_reuse=sum(r['known_byte_reuse'] for r in rows),primary_count=len(primary),
            lock_sha256=sha(OUT/'lock.json'))
        if (OUT/'catalog.json').exists():
            assert read(OUT/'catalog.json') == catalog
        else:
            write(OUT/'catalog.json',catalog)
        provider = lambda row:archives[Path(row['archive']).stem].open(row['member'])
        manifest = build_cache(rows,DATA/'cache',provider,tokens=CONFIG['cache_tokens'],minimum_free_bytes=2*2**30)
        write(OUT/'cache_ready.json',dict(manifest_sha256=sha(DATA/'cache/manifest.json'),
            catalog_sha256=sha(OUT/'catalog.json'),objects=manifest['count'],lock_sha256=sha(OUT/'lock.json')))
    finally:
        for archive in archives.values():
            archive.close()


def export(role, seed):
    lock = ready()
    record = read(OUT/'cache_ready.json')
    assert record['lock_sha256'] == sha(OUT/'lock.json')
    assert record['catalog_sha256'] == sha(OUT/'catalog.json')
    rows = read(OUT/'catalog.json')['rows']
    bundle = runtime.load_cached_population(DATA/'cache',rows,record['manifest_sha256'])
    groups = dict(full_population=np.ones(len(rows),dtype=bool),
        no_known_overlap=np.asarray([not r['known_byte_reuse'] for r in rows]))
    checkpoint = next(m for m in lock['models'] if (m['role'],m['seed']) == (role,seed))
    cuda_setup()
    for condition in VARIANTS:
        runtime.export_one(bundle,checkpoint,condition,groups,OUT/'evaluation'/role/str(seed)/(condition+'.npz'))
        print('EXPORTED',role,seed,condition,flush=True)


def report():
    lock = ready()
    rows = read(OUT/'catalog.json')['rows']
    labels = np.asarray([r['taxonomy_id'] for r in rows])
    ids = np.asarray([r['file_path'] for r in rows])
    primary_mask = np.asarray([not r['known_byte_reuse'] for r in rows])
    records, predictions, ce = [], {}, {}
    for m in lock['models']:
        for condition in VARIANTS:
            path = OUT/'evaluation'/m['role']/str(m['seed'])/(condition+'.npz')
            item = read(path.with_suffix('.json'))
            assert sha(path) == item['prediction_sha256']
            assert item['identity']['checkpoint'] == m and item['identity']['variant'] == condition
            with np.load(path,allow_pickle=False) as z:
                assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],labels)
                x = z['logits'];pred = x.argmax(1)
            key = m['role'],m['seed'],condition
            predictions[key] = pred
            ce[key] = item['groups']['no_known_overlap']['metrics']['ce']
            records.append(dict(role=m['role'],seed=m['seed'],condition=condition,groups=item['groups'],
                outside_seven_fraction=float((~np.isin(pred,CLASSES)).mean()),
                prediction_sha256=item['prediction_sha256']))
    comparisons = [('distance_mlp',SEEDS,'all8'),('distance_mlp',SEEDS[:3],'old3'),
                   ('distance_mlp',SEEDS[3:],'new5')]
    comparisons += [(r,SEEDS[:3],'old3') for r in
        ('fixed_isotropic5','direct_mlp','gaussian_relation','center_rope','nope','symmetric3','jeffreys_mlp')]
    analyses = []
    for right,seeds,cohort in comparisons:
        for condition in VARIANTS:
            a = np.stack([predictions['relative5',s,condition] for s in seeds])[:,primary_mask]
            b = np.stack([predictions[right,s,condition] for s in seeds])[:,primary_mask]
            summary = paired_accuracy_summary(a,b,labels[primary_mask],replicates=10000,seed=80001+len(analyses))
            analyses.append(dict(left='relative5',right=right,seeds=list(seeds),cohort=cohort,
                condition=condition,paired_ce_difference=[ce['relative5',s,condition]-ce[right,s,condition] for s in seeds],**summary))
    p = analyses[0]
    support = p['mean_gain_pp'] >= .5 and p['crossed_seed_object_bootstrap_95_pp'][0] > 0
    permutations = [dict(role=m['role'],seed=m['seed'],classification_changes=int(np.sum(
        predictions[m['role'],m['seed'],'base'] != predictions[m['role'],m['seed'],'permuted'])))
        for m in lock['models']]
    write(OUT/'result.json',dict(records=records,comparisons=analyses,primary=p,
        primary_limited_signal=bool(support),permutation=permutations,objects=len(rows),
        primary_objects=int(primary_mask.sum()),lock_sha256=sha(OUT/'lock.json'),
        scientific_review_pending=True,visual_review_pending=True,goal_complete=False,training_updates=0))
    print('CROSS_SOURCE_REPORT_READY',p['mean_gain_pp'],p['crossed_seed_object_bootstrap_95_pp'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('seal','preflight','download','prepare','export','report'))
    parser.add_argument('--shard',type=int)
    parser.add_argument('--role')
    parser.add_argument('--seed',type=int)
    args = parser.parse_args()
    if args.action == 'preflight':
        preflight(args.shard)
    elif args.action == 'export':
        export(args.role,args.seed)
    else:
        globals()[args.action]()
