"""Read-only diagnosis of the fixed confirmation's whole-model failure."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import numpy as np

from experiments import modelnet_cross_source as experiment
from experiments.round65_data import native_arrays, read_vertices

ROOT, OUT = experiment.ROOT, experiment.OUT
HERE = Path(__file__).resolve().parent
CLASSES = experiment.CLASSES


def input_summary(directory, catalog, selected_ids):
    directory = Path(directory)
    manifest = experiment.read(directory / 'manifest.json')
    ids = manifest['ids']
    lookup = {r['file_path']: r for r in catalog}
    arrays = {name: np.load(directory / (name + '.npy'), mmap_mode='r', allow_pickle=False)
              for name in ('centers', 'covariance', 'appearance', 'labels')}
    records = []
    for i, identity in enumerate(ids):
        if identity not in selected_ids:
            continue
        row = lookup[identity]
        center, cov, app = [np.asarray(arrays[k][i]) for k in ('centers', 'covariance', 'appearance')]
        assert all(np.isfinite(v).all() for v in (center, cov, app))
        assert int(arrays['labels'][i]) == row['taxonomy_id']
        eig = np.linalg.eigvalsh(cov)
        assert eig.min() > 0
        records.append(dict(file_path=identity, taxonomy_id=row['taxonomy_id'],
            center_rms=float(np.sqrt(np.square(center.astype(np.float64)).sum(-1).mean())),
            median_covariance_trace=float(np.median(eig.sum(-1))),
            median_condition_number=float(np.median(eig[:, -1] / eig[:, 0])),
            median_appearance=np.median(app, axis=0).tolist()))
    assert len(records) == len(selected_ids)
    summary = []
    for label in CLASSES:
        group = [r for r in records if r['taxonomy_id'] == label]
        item = dict(taxonomy_id=label, objects=len(group))
        for key in ('center_rms', 'median_covariance_trace', 'median_condition_number', 'median_appearance'):
            item[key + '_object_quantiles_10_50_90'] = np.quantile([r[key] for r in group], [.1, .5, .9], axis=0).tolist()
        summary.append(item)
    return dict(objects=len(records), tokens_per_object=int(arrays['centers'].shape[1]),
        per_class=summary, records=records, manifest_sha256=experiment.sha(directory / 'manifest.json'))


def main():
    assert not (HERE / 'result.json').exists()
    plan = experiment.read(HERE / 'plan.json')
    assert plan['result_sha256'] == experiment.sha(OUT / 'result.json')
    lock = experiment.check()
    new_rows = experiment.read(OUT / 'catalog.json')['rows']
    old_rows = experiment.read(experiment.OLD_CATALOG)['rows']
    train_rows = experiment.read(ROOT / 'results/round65_actual_gs/preparation/train_catalog.json')['rows']
    category_file = ROOT / 'results/round65_actual_gs/preparation/category.txt'
    names = {int(line.split(',', 1)[0]): line.split(',', 1)[1]
             for line in category_file.read_text().splitlines() if line.strip()}
    source_by_class = {}
    for row in old_rows:
        source_by_class.setdefault(row['taxonomy_id'], set()).add(row['source'])
    assert len(source_by_class) == 60 and all(len(v) == 1 for v in source_by_class.values())
    source_by_class = {k: next(iter(v)) for k, v in source_by_class.items()}
    mapping = []
    for label in CLASSES:
        group = [r for r in new_rows if r['taxonomy_id'] == label]
        train_names = {r['category'] for r in train_rows if r['taxonomy_id'] == label}
        old_names = {r['category'] for r in old_rows if r['taxonomy_id'] == label}
        assert train_names == old_names == {names[label]} == {r['category'] for r in group}
        assert all(r['member'].startswith('test/' + r['model_id'] + '/') for r in group)
        mapping.append(dict(taxonomy_id=label, category=names[label], new_objects=len(group),
                            training_source=source_by_class[label], new_source='ModelNet'))
    prediction_records = []
    for model in lock['models']:
        path = OUT / 'evaluation' / model['role'] / str(model['seed']) / 'base.npz'
        with np.load(path, allow_pickle=False) as z:
            pred = z['logits'].argmax(1)
            assert np.array_equal(z['labels'], [r['taxonomy_id'] for r in new_rows])
        counts = Counter(pred.tolist())
        prediction_records.append(dict(role=model['role'], seed=model['seed'],
            correct_count=int((pred == [r['taxonomy_id'] for r in new_rows]).sum()),
            predicted_source_counts={source: sum(n for label, n in counts.items()
                                                if source_by_class[label] == source)
                                     for source in ('MACGS', 'ModelNet')},
            predicted_class_counts={str(label): count for label, count in sorted(counts.items())}))
    source_summary = []
    for role in experiment.ROLES:
        group = [r for r in prediction_records if r['role'] == role]
        source_summary.append(dict(role=role, seeds=len(group), objects_per_seed=len(new_rows),
            correct_predictions=sum(r['correct_count'] for r in group),
            predicted_ModelNet_fraction=float(np.mean([r['predicted_source_counts']['ModelNet'] / len(new_rows)
                                                       for r in group]))))
    cache_dir = experiment.DATA / 'cache'
    marker = experiment.read(cache_dir / 'identity.json')
    arrays = {name: np.load(cache_dir / (name + '.npy'), mmap_mode='r', allow_pickle=False)
              for name in ('centers', 'covariance', 'appearance', 'indices')}
    conversion_checks = []
    for label in CLASSES:
        index, row = min(((i, r) for i, r in enumerate(new_rows) if r['taxonomy_id'] == label),
                         key=lambda pair: pair[1]['model_id'])
        with tempfile.TemporaryDirectory(prefix='grope_modelnet_check_') as tmp:
            path = Path(tmp) / 'object.ply'
            with zipfile.ZipFile(experiment.DATA / 'archives' / row['archive']) as archive:
                with archive.open(row['member']) as source, path.open('wb') as target:
                    shutil.copyfileobj(source, target)
            assert experiment.sha(path) == row['sha256']
            vertex = read_vertices(path)
            try:
                fields = list(vertex.dtype.names)
                converted, _ = native_arrays(vertex, row['file_path'], marker['tokens'])
            finally:
                vertex._mmap.close()
            errors = {name: float(np.max(np.abs(value.astype(np.float64) - arrays[name][index])))
                      for name, value in converted.items()}
            assert all(v == 0 for v in errors.values())
            conversion_checks.append(dict(file_path=row['file_path'], raw_sha256=row['sha256'],
                fields=fields, maximum_error=errors))
    old_cache = experiment.DATA.parent.parent / 'native_confirmation/attempt1/cache'
    old_ids = {r['file_path'] for r in old_rows if r['taxonomy_id'] in CLASSES}
    summaries = dict(old=input_summary(old_cache, old_rows, old_ids),
                     new=input_summary(cache_dir, new_rows, {r['file_path'] for r in new_rows}))
    result = dict(completed_unix=time.time(), plan_sha256=experiment.sha(HERE / 'plan.json'),
        implementation_sha256=experiment.sha(__file__), original_result_sha256=experiment.sha(OUT / 'result.json'),
        lock_sha256=experiment.sha(OUT / 'lock.json'), category_file_sha256=experiment.sha(category_file),
        mapping_checks=mapping, source_by_output_class=source_by_class, mapping_consistent=True,
        prediction_records=prediction_records, predicted_source_summary=source_summary,
        conversion_checks=conversion_checks, input_summaries=summaries,
        new_inference=False, optimizer_updates=0, scores_or_labels_modified=False,
        causal_source_or_appearance_attribution_established=False)
    experiment.write(HERE / 'result.json', result)
    print('READ_ONLY_DIAGNOSIS_COMPLETE', json.dumps(source_summary), flush=True)


if __name__ == '__main__':
    main()
