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
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import torch

from experiments import native_confirmation_runtime as runtime
from experiments import round65_run as runner
from experiments.native_confirmation_analysis import paired_accuracy_summary
from experiments.native_memmap_runtime import install
from experiments.native_stream_cache import build_cache
from experiments.round65_common import ROOT, DATA_ROOT, CONFIG, read, write, sha

OUT = ROOT / 'results/modelnet_cross_source/attempt1'
DATA = DATA_ROOT / 'GRoPE/modelnet_cross_source/attempt1'
META = ROOT / 'results/research_data_usage/native_candidates_20261106/modelnet620_candidate_metadata.json'
PLAN = ROOT / 'docs/modelnet_cross_source_plan.md'
OLD_LOCK = ROOT / 'results/native_confirmation/attempt1/evaluation_lock.json'
OLD_CATALOG = ROOT / 'results/native_confirmation_metadata/attempt1/test_catalog.json'
OLD_EXPORTS = ROOT / 'results/native_confirmation/attempt1/exports'
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
CLASSES = (1, 4, 11, 12, 13, 15, 20)
ROLES = ('relative5', 'distance_mlp', 'fixed_isotropic5')
CONDITIONS = ('base',)
PREFLIGHT_SHARDS = 2
CAP_BYTES = 16 * 2**30
BASE_FACTORY = runtime.make_model


def model_for(role):
    if role not in ROLES:
        raise ValueError('undeclared fixed model')
    return BASE_FACTORY(role)


def selected_roster(models):
    result = [m for m in models if m['role'] in ROLES]
    expected = {(r, s) for r in ROLES for s in (SEEDS[:3] if r == 'fixed_isotropic5' else SEEDS)}
    if len(result) != 19 or {(m['role'], m['seed']) for m in result} != expected:
        raise ValueError('expected eight F/Distance and three fixed-sphere models')
    return result


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
    models = selected_roster(read(OLD_LOCK)['models'])
    old_exports = []
    for m in models:
        assert sha(m['path']) == m['sha256']
        assert sha(m['selection_path']) == m['selection_sha256']
        selected = next(r for r in read(m['selection_path'])['records']
                        if (r['role'], r['seed']) == (m['role'], m['seed']))
        assert selected['checkpoint_sha256'] == m['sha256']
        assert selected['best_epoch'] == m['selected_epoch']
        prediction = OLD_EXPORTS / m['role'] / str(m['seed']) / 'base.npz'
        sidecar = read(prediction.with_suffix('.json'))
        assert sidecar['identity']['checkpoint'] == m and sidecar['identity']['variant'] == 'base'
        assert sha(prediction) == sidecar['prediction_sha256']
        old_exports.append(dict(role=m['role'], seed=m['seed'], path=str(prediction),
            sha256=sidecar['prediction_sha256'], sidecar_sha256=sha(prediction.with_suffix('.json'))))
    old_rows = [r for r in read(OLD_CATALOG)['rows'] if r['taxonomy_id'] in CLASSES]
    assert len(old_rows) == 422 and {r['source'] for r in old_rows} == {'MACGS'}
    dependencies = source_paths() + [PLAN, META, OLD_LOCK, OLD_CATALOG, OUT / 'cpu_checks.json',
        ROOT / 'results/round65_actual_gs/preparation/train_catalog.json',
        ROOT / 'results/native_region_pe/attempt1/review.json']
    assert read(OUT / 'cpu_checks.json')['passed']
    record = dict(created_unix=time.time(),source_sha256={str(p):sha(p) for p in dependencies},
        models=models, candidate=candidate, conditions=list(CONDITIONS), config=CONFIG,
        old_source_exports=old_exports, old_source_objects=len(old_rows),
        class_ids=list(CLASSES), primary_seeds=list(SEEDS), training_updates=0,
        new_inputs_accessed=False, proxy_allowed=False, full_sixty_class_output=True,
        bootstrap_replicates=10000, bootstrap_seed=80001, maximum_gpu_jobs=2,
        allocator_cap_bytes=CAP_BYTES, overall_goal_complete=False)
    write(OUT / 'lock.json', record)
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('CROSS_SOURCE_SEALED', len(models), sha(OUT / 'lock.json'), flush=True)


def check():
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    assert lock['config'] == CONFIG and lock['conditions'] == list(CONDITIONS)
    selected_roster(lock['models'])
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
    if shard not in range(PREFLIGHT_SHARDS):
        raise ValueError('two fixed preflight shards')
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
    for m in lock['models'][shard::PREFLIGHT_SHARDS]:
        model = model_for(m['role']).cuda().eval()
        model.load_state_dict(torch.load(m['path'],map_location='cpu',weights_only=True)['state_dict'])
        logits, _ = runner.evaluate(model, bundle, 'base')
        with np.load(Path(m['path']).parent / 'validation.npz', allow_pickle=False) as z:
            assert np.array_equal(z['ids'], bundle['ids'])
            error = float(np.max(np.abs(z['logits'] - logits)))
            changes = int((z['logits'].argmax(1) != logits.argmax(1)).sum())
        assert error < 2e-4 and changes == 0
        # Order invariance is an implementation check on old data, not a fresh condition.
        tiny = {k:v[:4] for k,v in bundle.items()}
        peaks = {}
        outputs = {}
        for condition in ('base', 'permuted'):
            torch.cuda.reset_peak_memory_stats()
            x, _ = runner.evaluate(model, tiny, condition)
            assert x.shape == (4, 60) and np.isfinite(x).all()
            peaks[condition] = torch.cuda.max_memory_allocated()
            outputs[condition] = x
        permutation_error = float(np.max(np.abs(outputs['base'] - outputs['permuted'])))
        permutation_changes = int((outputs['base'].argmax(1) != outputs['permuted'].argmax(1)).sum())
        assert permutation_error < 2e-4 and permutation_changes == 0
        records.append(dict(role=m['role'],seed=m['seed'],objects=len(logits),
            gpu_validation_max_error=error,classification_changes=changes,peak_allocated=peaks,
            permutation_max_error=permutation_error,permutation_changes=permutation_changes))
        del model
        torch.cuda.empty_cache()
        print('PREFLIGHT_MODEL',m['role'],m['seed'],error,flush=True)
    write(target,dict(passed=True,records=records,lock_sha256=sha(OUT / 'lock.json'),
        new_inputs_accessed=False,training_updates=0,gpu=torch.cuda.get_device_name()))


def ready():
    lock = check()
    rows = []
    for shard in range(PREFLIGHT_SHARDS):
        r = read(OUT / 'preflight' / f'{shard}.json')
        assert r['passed'] and r['lock_sha256'] == sha(OUT / 'lock.json')
        rows.extend(r['records'])
    assert {(r['role'],r['seed']) for r in rows} == {(m['role'],m['seed']) for m in lock['models']}
    reference = read(OUT / 'old_source_reference.json')
    assert reference['lock_sha256'] == sha(OUT / 'lock.json') and reference['objects'] == 422
    return lock


def start_access(kind):
    ready()
    marker = OUT / 'data_access_started.json'
    if not marker.exists():
        write(marker,dict(started_unix=time.time(),kind=kind,lock_sha256=sha(OUT / 'lock.json'),
            all_620_ids_to_be_treated_as_used_after_access=True,archive_contains_other_unscored_objects=True))
        write(ROOT / 'results/research_data_usage/modelnet_cross_source_access_20261109.json',
            dict(read(marker), candidate_path=str(META), candidate_sha256=sha(META),
                outcome_path=str(OUT), objects=620, no_further_fresh_claim_after_access=True))


class MirrorRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep metadata probes as HEAD and never forward the mirror credential."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != 'https' or not target.hostname or target.username or target.password:
            raise ValueError('dataset redirects must use HTTPS without embedded credentials')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            for name, _ in list(redirected.header_items()):
                if name.lower() in ('authorization', 'proxy-authorization'):
                    redirected.remove_header(name)
            if req.get_method() == 'HEAD':
                redirected.method = 'HEAD'
        return redirected


def mirror_request(url, token=None, method='GET'):
    target = urllib.parse.urlsplit(url)
    if (target.scheme != 'https' or target.hostname != 'hf-mirror.com' or
            target.port not in (None, 443) or target.username or target.password):
        raise ValueError('initial dataset request must use the configured HTTPS mirror')
    if method not in ('GET', 'HEAD'):
        raise ValueError('only dataset GET and HEAD requests are supported')
    request = urllib.request.Request(url, method=method)
    if token:
        request.add_unredirected_header('Authorization', 'Bearer ' + token)
    return request


def mirror_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), MirrorRedirectHandler())


def download():
    """Direct mirror with an optional user's existing HF credential; no access request."""
    lock = ready()
    from huggingface_hub import get_token
    token = get_token()
    # Explicitly disable proxy use. Do not print credentials or signed redirects.
    opener = mirror_opener()
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
        # A failed HEAD writes no data-access marker and never submits gate forms.
        try:
            with opener.open(mirror_request(url, token, 'HEAD'), timeout=30):
                pass
        except urllib.error.HTTPError as e:
            write(OUT / 'download_access_failure.json',dict(observed_unix=time.time(),
                status=e.code,error_code=e.headers.get('X-Error-Code'),archive=row['rfilename'],
                local_authentication_present=bool(token),access_request_submitted=False))
            raise RuntimeError('dataset access unavailable; see download_access_failure.json') from None
        start_access('archive_download')
        digest, size = hashlib.sha256(), 0
        with opener.open(mirror_request(url, token), timeout=60) as source, pending.open('xb') as sink:
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
    for condition in CONDITIONS:
        runtime.export_one(bundle,checkpoint,condition,groups,OUT/'evaluation'/role/str(seed)/(condition+'.npz'))
        print('EXPORTED',role,seed,condition,flush=True)


def collect_population(lock, *, old=False):
    rows = read(OLD_CATALOG if old else OUT / 'catalog.json')['rows']
    labels = np.asarray([r['taxonomy_id'] for r in rows])
    ids = np.asarray([r['file_path'] for r in rows])
    primary_mask = (np.isin(labels, CLASSES) if old else
                    np.asarray([not r['known_byte_reuse'] for r in rows]))
    assert set(labels[primary_mask]) == set(CLASSES)
    records, predictions, ce = [], {}, {}
    for m in lock['models']:
        for condition in CONDITIONS:
            path = ((OLD_EXPORTS if old else OUT / 'evaluation') /
                    m['role'] / str(m['seed']) / (condition + '.npz'))
            item = read(path.with_suffix('.json'))
            assert sha(path) == item['prediction_sha256']
            assert item['identity']['checkpoint'] == m and item['identity']['variant'] == condition
            if old:
                frozen = next(r for r in lock['old_source_exports']
                              if (r['role'], r['seed']) == (m['role'], m['seed']))
                assert frozen['sha256'] == sha(path)
                assert frozen['sidecar_sha256'] == sha(path.with_suffix('.json'))
            with np.load(path,allow_pickle=False) as z:
                assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],labels)
                x = z['logits'];pred = x.argmax(1)
            groups = runtime.summarize_logits(x, labels, dict(selected_population=primary_mask))
            metrics = groups['selected_population']['metrics']
            key = m['role'],m['seed'],condition
            predictions[key] = pred[primary_mask]
            ce[key] = metrics['ce']
            records.append(dict(role=m['role'],seed=m['seed'],condition=condition,metrics=metrics,
                outside_seven_fraction=float((~np.isin(pred[primary_mask],CLASSES)).mean()),
                prediction_sha256=item['prediction_sha256']))
    return records, predictions, ce, labels[primary_mask]


def comparison_specs():
    return [('distance_mlp', SEEDS, 'all8'), ('fixed_isotropic5', SEEDS[:3], 'old3'),
            ('distance_mlp', SEEDS[:3], 'old3'), ('distance_mlp', SEEDS[3:], 'new5')]


def summarize_comparisons(predictions, ce, labels, *, bootstrap_seed):
    analyses = []
    for right,seeds,cohort in comparison_specs():
        for condition in CONDITIONS:
            a = np.stack([predictions['relative5',s,condition] for s in seeds])
            b = np.stack([predictions[right,s,condition] for s in seeds])
            summary = paired_accuracy_summary(a,b,labels,replicates=10000,seed=bootstrap_seed+len(analyses))
            analyses.append(dict(left='relative5',right=right,seeds=list(seeds),cohort=cohort,
                condition=condition,paired_ce_difference=[ce['relative5',s,condition]-ce[right,s,condition] for s in seeds],**summary))
    return analyses


def reference():
    """Read only existing official-test predictions, with seven-class matching."""
    lock = check()
    if (OUT / 'old_source_reference.json').exists():
        raise ValueError('preserve the prospectively fixed old reference')
    records, predictions, ce, labels = collect_population(lock, old=True)
    analyses = summarize_comparisons(predictions, ce, labels, bootstrap_seed=80021)
    write(OUT / 'old_source_reference.json', dict(records=records, comparisons=analyses,
        objects=len(labels), class_ids=list(CLASSES), class_counts=[int((labels == c).sum()) for c in CLASSES],
        lock_sha256=sha(OUT / 'lock.json'), new_inputs_accessed=False, source='MACGS',
        descriptive_used_data=True, historical_confirmation_decision_unchanged=True))
    print('OLD_SOURCE_REFERENCE',len(labels),analyses[0]['mean_gain_pp'],flush=True)


def source_gain_change(new_left, new_right, new_labels, old_left, old_right, old_labels,
                       *, replicates=10000, seed=80101):
    """New-minus-old PE gain; same seeds/classes, independent objects per source.

    This is a descriptive transport contrast, not an isolated reconstruction effect.
    """
    inputs = [(np.asarray(new_left), np.asarray(new_right), np.asarray(new_labels)),
              (np.asarray(old_left), np.asarray(old_right), np.asarray(old_labels))]
    classes = np.unique(inputs[0][2])
    nseeds = inputs[0][0].shape[0]
    deltas, per_class = [], []
    for left, right, labels in inputs:
        if (left.ndim != 2 or left.shape != right.shape or left.shape != (nseeds, len(labels)) or
                not np.array_equal(np.unique(labels), classes) or nseeds < 1):
            raise ValueError('unaligned source comparison')
        delta = 100. * ((left == labels).astype(float) - (right == labels).astype(float))
        indices = [np.flatnonzero(labels == c) for c in classes]
        deltas.append((delta, indices))
        per_class.append(np.stack([delta[:, ix].mean(1) for ix in indices], 1))
    change = per_class[0] - per_class[1]
    rng = np.random.default_rng(seed)
    draws, class_draws = np.empty(replicates), np.empty(replicates)
    for start in range(0, replicates, 128):
        stop = min(start + 128, replicates); count = stop - start
        sw = rng.multinomial(nseeds, np.full(nseeds, 1/nseeds), size=count) / nseeds
        sampled = np.zeros((count, nseeds))
        for sign, (delta, indices) in zip((1, -1), deltas):
            for ix in indices:
                weights = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), size=count)
                sampled += sign * (weights @ delta[:, ix].T) / (len(ix) * len(classes))
        draws[start:stop] = (sw * sampled).sum(1)
        cw = rng.multinomial(len(classes), np.full(len(classes), 1/len(classes)), size=count) / len(classes)
        class_draws[start:stop] = cw @ change.mean(0)
    return dict(mean_gain_change_pp=float(change.mean()), paired_seed_gain_changes_pp=change.mean(1).tolist(),
        class_mean_gain_changes_pp=change.mean(0).tolist(), class_ids=classes.tolist(),
        crossed_seed_object_bootstrap_95_pp=np.quantile(draws,[.025,.975]).tolist(),
        class_sensitivity_bootstrap_95_pp=np.quantile(class_draws,[.025,.975]).tolist(),
        bootstrap_seed=seed,bootstrap_replicates=replicates,
        source_effect_not_causally_isolated=True)


def report():
    lock = ready()
    if (OUT / 'result.json').exists():
        raise ValueError('preserve completed result')
    records, predictions, ce, labels = collect_population(lock)
    analyses = summarize_comparisons(predictions, ce, labels, bootstrap_seed=80001)
    _, old_predictions, _, old_labels = collect_population(lock, old=True)
    changes = []
    for right, seeds, cohort in comparison_specs()[:2]:
        arrays = [np.stack([pop[role,s,'base'] for s in seeds])
                  for pop in (predictions,old_predictions) for role in ('relative5',right)]
        stats = source_gain_change(arrays[0],arrays[1],labels,arrays[2],arrays[3],old_labels,
                                   seed=80101+len(changes))
        changes.append(dict(left='relative5',right=right,seeds=list(seeds),cohort=cohort,**stats))
    p = analyses[0]
    support = p['mean_gain_pp'] >= .5 and p['crossed_seed_object_bootstrap_95_pp'][0] > 0
    write(OUT/'result.json',dict(records=records,comparisons=analyses,primary=p,
        primary_limited_signal=bool(support),source_gain_changes=changes,
        objects=len(read(OUT/'catalog.json')['rows']),primary_objects=len(labels),
        old_reference_sha256=sha(OUT/'old_source_reference.json'),lock_sha256=sha(OUT/'lock.json'),
        scientific_review_pending=True,visual_review_pending=True,goal_complete=False,training_updates=0))
    print('CROSS_SOURCE_REPORT_READY',p['mean_gain_pp'],p['crossed_seed_object_bootstrap_95_pp'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('seal','reference','preflight','download','prepare','export','report'))
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
