"""Sealed exports and reporting for nine fixed source-diversity fits.

Only the original development population and the declared additional TRAIN/dev
objects are scored. The previously used 620-object confirmation is never loaded.
"""
import argparse
from pathlib import Path
import os
import time

import numpy as np

from experiments.round65_common import ROOT, read, write, sha
from experiments.native_source_diversity_data import OUT, DATA, OLD_DATA
from experiments import native_source_diversity_analysis as analysis

EVALUATION_CAP = 4 * 2**30


def source_paths():
    return [ROOT / 'experiments' / ('native_source_diversity_' + name + '.py')
            for name in ('evaluation', 'analysis', 'review', 'postprocess')] + [
                ROOT / 'tests/test_native_source_diversity_evaluation.py']


def population_metadata():
    part = next(p for p in read(OLD_DATA / 'manifest.json')['partitions'] if p['split'] == 'dev_eval')
    extra = [r for r in read(OUT / 'catalog.json')['rows'] if r['split'] == 'dev_eval']
    old_ids = part['ids']
    old_y = np.array([int(s.split('/')[0]) for s in old_ids], dtype=np.int64)
    assert np.array_equal(np.bincount(old_y, minlength=60), part['class_counts'])
    labels = np.concatenate([old_y, np.array([r['taxonomy_id'] for r in extra], dtype=np.int64)])
    ids = np.asarray(old_ids + [r['file_path'] for r in extra])
    sources = np.concatenate([np.where(old_y < 30, 'MACGS', 'ModelNet'), np.full(len(extra), 'ModelNet')])
    old = np.arange(len(ids)) < len(old_ids)
    assert len(ids) == len(set(ids)) == 1027 and old.sum() == 765
    meta = dict(ids=ids, labels=labels, sources=sources, original_native=old)
    masks = analysis.group_masks(labels, sources, old)
    assert {name: int(ix.sum()) for name, ix in masks.items()} == dict(
        primary_seven=326, seven_MACGS=64, seven_ModelNet=262,
        original_60=765, original_other53=701, merged_60=1027)
    return meta


def checked_arrays(path, metadata):
    with np.load(path, allow_pickle=False) as data:
        for field in ('ids', 'labels'):
            if not np.array_equal(data[field], metadata[field]):
                raise ValueError(f'{path}: {field} mismatch.')
        for field in ('sources', 'original_native'):
            if field in data.files and not np.array_equal(data[field], metadata[field]):
                raise ValueError(f'{path}: {field} mismatch.')
        x = np.array(data['logits'])
    if x.shape != (len(metadata['labels']), 60) or not np.isfinite(x).all():
        raise ValueError(f'{path}: invalid full-head logits.')
    return x


def old_export_references(training_lock):
    entries = []
    for model in training_lock['old_models']:
        run = Path(model['path']).parent
        root = run.parents[2]
        path = root / 'evaluation' / model['role'] / str(model['seed']) / 'base.npz'
        record_path = path.parent / 'record.json'
        record = read(record_path)
        selected = read(model['selection_path'])
        chosen = next(r for r in selected['records'] if (r['role'], r['seed']) == (model['role'], model['seed']))
        assert sha(model['selection_path']) == model['selection_sha256'] == record['selection_sha256']
        assert sha(model['path']) == model['sha256'] == record['checkpoint_sha256'] == chosen['checkpoint_sha256']
        assert chosen['best_epoch'] == model['selected_epoch']
        assert record['data_sha256'] == sha(OLD_DATA / 'manifest.json')
        base = next(r for r in record['records'] if r['variant'] == 'base')
        assert sha(path) == base['sha256']
        entries.append(dict(role=model['role'], seed=model['seed'], path=str(path),
            sha256=sha(path), record_path=str(record_path), record_sha256=sha(record_path), checkpoint=model))
    assert {(r['role'], r['seed']) for r in entries} == {(r, s) for r in analysis.ROLES for s in analysis.SEEDS}
    return entries


def old_check():
    from experiments.native_source_diversity import check
    from experiments.native_source_diversity_review import independent_metrics, independent_pair
    lock = check()
    references = old_export_references(lock)
    meta = population_metadata()
    old = meta['original_native']
    old_meta = {k: v[old] for k, v in meta.items()}
    errors, pred = [], {}
    for row in references:
        x = checked_arrays(row['path'], old_meta)
        computed = analysis.metrics(x, old_meta['labels'], old_meta['sources'])
        separate = independent_metrics(x, old_meta['labels'], old_meta['sources'])
        prior = next(r for r in read(row['record_path'])['records'] if r['variant'] == 'base')['metrics']
        errors += [abs(computed[k] - prior[k]) for k in ('mca', 'oa', 'ce')]
        errors += [abs(computed[k] - separate[k]) for k in ('mca', 'oa', 'ce', 'source_equal_accuracy', 'source_equal_ce')]
        pred[row['role'], row['seed']] = x.argmax(1)
    # Verify the bootstrap on already-used objects, without reading any newly
    # scored development predictions. Synthetic tests cover both-source cells.
    for j, role in enumerate(analysis.ROLES[1:]):
        a = np.stack([pred['relative5', s] for s in analysis.SEEDS])
        b = np.stack([pred[role, s] for s in analysis.SEEDS])
        domain = np.full(old.sum(), 'pooled')
        primary = analysis.paired_source_comparison(a, b, old_meta['labels'], domain, seed=92101+j)
        separate = independent_pair(a, b, old_meta['labels'], domain, seed=92101+j)
        errors += [abs(primary['mean_gain_pp'] - separate['mean'])]
        for key in separate['intervals']:
            errors.append(float(np.abs(separate['intervals'][key] - primary['intervals_95_pp'][key]).max()))
    maximum = max(errors)
    assert maximum < 1e-8
    write(OUT / 'evaluation_old_reference_checks.json', dict(passed=True, maximum_error=maximum,
        source_sha256={str(p): sha(p) for p in source_paths()}, old_exports=references,
        original_development_objects=765, new_development_scores_read=False,
        checked_unix=time.time(), goal_complete=False))
    print('OLD_ONLY_EVALUATION_CHECKS', maximum, flush=True)


def freeze():
    from experiments.native_source_diversity import check
    training = check()
    assert not (OUT / 'evaluation_lock.json').exists()
    assert not (OUT / 'evaluation_access_started.json').exists()
    for name in ('evaluation_cpu_checks.json', 'evaluation_old_reference_checks.json'):
        record = read(OUT / name)
        assert record['passed']
        for path, digest in record['source_sha256'].items():
            assert sha(path) == digest, path
    refs = old_export_references(training)
    meta = population_metadata()
    payload = dict(created_unix=time.time(), training_lock_sha256=sha(OUT / 'lock.json'),
        source_sha256={str(p): sha(p) for p in source_paths()},
        checks_sha256={name: sha(OUT / name) for name in ('evaluation_cpu_checks.json', 'evaluation_old_reference_checks.json')},
        old_exports=refs, comparisons=analysis.comparison_specs(), bootstrap_replicates=10000,
        population={k: v.tolist() for k, v in meta.items()}, variant='base', tokens=512,
        new_exports=9, background_added_only_exports=9, full_output_classes=60,
        validation_replay_before_development=True, allocator_cap_bytes=EVALUATION_CAP,
        fresh_confirmation=False, scientific_review_pending=True, goal_complete=False)
    write(OUT / 'evaluation_lock.json', payload)
    (OUT / 'evaluation_lock.sha256').write_text(sha(OUT / 'evaluation_lock.json') + '\n')
    print('SOURCE_DIVERSITY_EVALUATION_SEALED', sha(OUT / 'evaluation_lock.json'), flush=True)


def check_evaluation_lock():
    from experiments.native_source_diversity import check
    check()
    lock = read(OUT / 'evaluation_lock.json')
    assert sha(OUT / 'evaluation_lock.json') == (OUT / 'evaluation_lock.sha256').read_text().strip()
    assert lock['training_lock_sha256'] == sha(OUT / 'lock.json')
    for path, digest in lock['source_sha256'].items():
        assert sha(path) == digest, path
    for name, digest in lock['checks_sha256'].items():
        assert sha(OUT / name) == digest
    for row in lock['old_exports']:
        assert sha(row['path']) == row['sha256'] and sha(row['record_path']) == row['record_sha256']
    assert lock['population'] == {k: v.tolist() for k, v in population_metadata().items()}
    assert lock['comparisons'] == analysis.comparison_specs()
    return lock


def selected_models():
    seal = read(OUT / 'selected.json')
    assert seal['lock_sha256'] == sha(OUT / 'lock.json')
    entries = seal['records']
    assert len(entries) == 9 and {(r['role'], r['seed']) for r in entries} == {
        (r, s) for r in analysis.ROLES for s in analysis.SEEDS}
    for entry in entries:
        run_path = Path(entry['path']).parent / 'run.json'
        run = read(run_path)
        assert sha(entry['path']) == entry['sha256'] and sha(run_path) == entry['run_sha256']
        assert run['lock_sha256'] == seal['lock_sha256'] and run['steps'] == 46800 and run['epochs'] == 120
        assert entry['selected_epoch'] == run['best_epoch']
    return entries


def export_identity(family, model):
    return dict(family=family, checkpoint=model, variant='base',
        evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
        selection_sha256=sha(OUT / 'selected.json'),
        population='merged_1027' if family == 'trained' else 'additional_development_262')


def export_one(model_entry, family, bundle):
    import torch
    from experiments.native_source_diversity import model_for, runner
    destination = OUT / 'evaluation' / family / model_entry['role'] / str(model_entry['seed']) / 'base.npz'
    sidecar = destination.with_suffix('.json')
    identity = export_identity(family, model_entry)
    if sidecar.exists():
        record = read(sidecar)
        assert record['identity'] == identity and sha(destination) == record['prediction_sha256']
        checked_arrays(destination, bundle)
        return record
    temporary = destination.with_suffix('.pending')
    if destination.exists() or temporary.exists():
        raise ValueError('Preserve interrupted export for explicit recovery: ' + str(destination))
    assert sha(model_entry['path']) == model_entry['sha256']
    payload = torch.load(model_entry['path'], map_location='cpu', weights_only=True)
    assert (payload['role'], payload['seed'], payload['epoch']) == (
        model_entry['role'], model_entry['seed'], model_entry['selected_epoch'])
    model = model_for(model_entry['role']).cuda().eval()
    model.load_state_dict(payload['state_dict'])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    logits, _ = runner.evaluate(model, bundle, 'base')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('xb') as stream:
        np.savez(stream, logits=logits, **{k: bundle[k] for k in ('ids', 'labels', 'sources', 'original_native')})
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(destination)
    record = dict(identity=identity, prediction_sha256=sha(destination), optimizer_updates=0,
        seconds=time.monotonic() - started, peak_allocated=torch.cuda.max_memory_allocated(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__,
        numpy_version=np.__version__, allocator_cap_bytes=EVALUATION_CAP)
    write(sidecar, record)
    del model
    torch.cuda.empty_cache()
    print('SOURCE_DIVERSITY_EXPORT', family, model_entry['role'], model_entry['seed'], flush=True)
    return record


def export_all():
    import torch
    from experiments import native_source_diversity as train
    lock = check_evaluation_lock()
    models = selected_models()  # No dev scoring until all nine complete and selected.
    train.runner.configure()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(EVALUATION_CAP / torch.cuda.get_device_properties(0).total_memory)
    status = dict(pid=os.getpid(), state='validation_replay', started_unix=time.time(),
        evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'), completed_exports=[],
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write(OUT / 'evaluation_status.json', status)
    try:
        extra, rows = train.extra_bundle()
        val = train.merged_population('val', extra, rows)
        replay = []
        for entry in models:
            directory = Path(entry['path']).parent
            run = read(directory / 'run.json')
            assert sha(directory / 'validation.npz') == run['hashes']['validation.npz']
            saved = checked_arrays(directory / 'validation.npz', val)
            payload = torch.load(entry['path'], map_location='cpu', weights_only=True)
            assert (payload['role'], payload['seed'], payload['epoch']) == (entry['role'], entry['seed'], entry['selected_epoch'])
            model = train.model_for(entry['role']).cuda().eval()
            model.load_state_dict(payload['state_dict'])
            logits, _ = train.runner.evaluate(model, val, 'base')
            maximum = float(np.abs(logits - saved).max())
            changed = int((logits.argmax(1) != saved.argmax(1)).sum())
            ce = analysis.metrics(logits, val['labels'], val['sources'])['source_equal_ce']
            ce_error = abs(ce - run['best_selection_ce'])
            if maximum > 1e-4 or changed or ce_error > 1e-6:
                raise ValueError(f'Validation replay disagreement: {entry["role"]}/{entry["seed"]}: {maximum}, {changed}, {ce_error}')
            replay.append(dict(role=entry['role'], seed=entry['seed'], maximum_logit_error=maximum,
                prediction_changes=changed, selection_ce_error=ce_error))
            del model
            torch.cuda.empty_cache()
        write(OUT / 'evaluation_validation_replay.json', dict(passed=True, records=replay,
            selection_sha256=sha(OUT / 'selected.json'), evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json')))
        del val
        access = OUT / 'evaluation_access_started.json'
        identity = dict(evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
            selection_sha256=sha(OUT / 'selected.json'), fresh_confirmation=False)
        if access.exists():
            assert read(access)['identity'] == identity
        else:
            write(access, dict(identity=identity, started_unix=time.time(),
                new_models_scored_only_after_all_selected=True, previous_620_objects_used=False))
        bundle = train.merged_population('dev_eval', extra, rows)
        meta = population_metadata()
        for key, values in meta.items():
            assert np.array_equal(bundle[key], values)
        status['state'] = 'exporting'
        for entry in models:
            export_one(entry, 'trained', bundle)
            status['completed_exports'].append(['trained', entry['role'], entry['seed']])
            write(OUT / 'evaluation_status.json', status)
        added = ~bundle['original_native']
        small = {k: v[added] for k, v in bundle.items()}
        for reference in lock['old_exports']:
            entry = reference['checkpoint']
            export_one(entry, 'background', small)
            status['completed_exports'].append(['background', entry['role'], entry['seed']])
            write(OUT / 'evaluation_status.json', status)
        status.update(state='exports_complete', completed_unix=time.time())
        write(OUT / 'evaluation_status.json', status)
    except BaseException as error:
        status.update(state='failed', reason=str(error), error_type=type(error).__name__)
        write(OUT / f'evaluation_failure_{time.time_ns()}.json', status)
        write(OUT / 'evaluation_status.json', status)
        raise


def load_prediction(family, role, seed, lock, meta):
    entry = (next(r for r in selected_models() if (r['role'], r['seed']) == (role, seed)) if family == 'trained'
             else next(r['checkpoint'] for r in lock['old_exports'] if (r['role'], r['seed']) == (role, seed)))
    path = OUT / 'evaluation' / family / role / str(seed) / 'base.npz'
    sidecar = read(path.with_suffix('.json'))
    assert sidecar['identity'] == export_identity(family, entry) and sidecar['optimizer_updates'] == 0
    assert sha(path) == sidecar['prediction_sha256']
    identity = dict(export=str(path), prediction_sha256=sha(path), record_sha256=sha(path.with_suffix('.json')))
    if family == 'trained':
        return checked_arrays(path, meta), identity
    assert family == 'background'
    old = meta['original_native']
    added_x = checked_arrays(path, {k: v[~old] for k, v in meta.items()})
    reference = next(r for r in lock['old_exports'] if (r['role'], r['seed']) == (role, seed))
    assert sha(reference['path']) == reference['sha256']
    original_x = checked_arrays(reference['path'], {k: v[old] for k, v in meta.items()})
    x = np.empty((len(old), 60), dtype=np.result_type(added_x.dtype, original_x.dtype))
    x[old], x[~old] = original_x, added_x
    identity['original_prediction_sha256'] = reference['sha256']
    return x, identity


def report():
    lock = check_evaluation_lock()
    assert not (OUT / 'result.json').exists()
    selected_models()
    meta = population_metadata()
    y, s, old = meta['labels'], meta['sources'], meta['original_native']
    predictions, records = {}, []
    for family in ('trained', 'background'):
        predictions[family] = {}
        for role in analysis.ROLES:
            for seed in analysis.SEEDS:
                x, identity = load_prediction(family, role, seed, lock, meta)
                records.append(dict(family=family, role=role, seed=seed, identity=identity,
                    groups=analysis.summaries(x, y, s, old)))
                predictions[family][role, seed] = x.argmax(1)
    comparisons = [analysis.compare(predictions[spec['family']], y, s, old, spec) for spec in lock['comparisons']]
    changes = []
    for role in analysis.ROLES:
        for group in analysis.GROUPS:
            paired = []
            for seed in analysis.SEEDS:
                a, b = [next(r['groups'][group] for r in records if
                    (r['family'], r['role'], r['seed']) == (family, role, seed)) for family in ('trained', 'background')]
                paired.append(dict(seed=seed, accuracy_change_pp=100 * (a['source_equal_accuracy'] - b['source_equal_accuracy']),
                                   oa_change_pp=100 * (a['oa'] - b['oa']), ce_change=a['ce'] - b['ce']))
            changes.append(dict(role=role, group=group, paired=paired,
                interpretation='Descriptive training-strategy change; instances and validation weighting also changed.'))
    write(OUT / 'result.json', dict(completed_unix=time.time(), evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
        selection_sha256=sha(OUT / 'selected.json'), records=records, comparisons=comparisons,
        strategy_changes=changes, primary_numerical_signal=analysis.primary_signal(comparisons[0]),
        attribution_requires_scientific_review=True, scientific_review_pending=True,
        fresh_confirmation=False, goal_complete=False))
    print('SOURCE_DIVERSITY_RESULT_READY', sha(OUT / 'result.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('old_check', 'freeze', 'export_all', 'report'))
    globals()[parser.parse_args().action]()
