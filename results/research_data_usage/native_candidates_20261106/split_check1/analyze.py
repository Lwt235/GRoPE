"""Inspect public object-ID lists and prior usage, without reading new primitives."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/home/liwt/GRoPE')
OUT = Path(__file__).resolve().parent
UPSTREAM = OUT / 'upstream'
EXPECTED = {
    'shapenet_train.txt': 'dea6edcdbf3c94826814ab6b26f0e7a214e6db4b',
    'shapenet_test.txt': 'a9d10ce294b05145f296bf1987b711c48030e011',
    'ShapeNet55GS.yaml': 'ec9a65b21ad24f92b42274aa33f29de983a7d90d',
    'ShapeNet55Gaussian.py.txt': '4cb5f50a3405a5b7150fa9d03f6e4f49f86fa39e',
    'pretrain_enc_full_group_xyz_1k.yaml': 'ebd4a6079a84096fb7696deff732a333de9ba8ad',
    'modelnet40_train.txt': '1b0c58cf52163734345e5411e39baa7d50e33773',
    'modelnet40_test.txt': 'ddf959319649739eeba3b2ebb31dffc98a7e72c6',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def normalized_source(name):
    path = UPSTREAM / name
    data = path.read_bytes()
    added_terminal_newline = False
    if blob(data) != EXPECTED[name]:
        assert data.endswith(b'\n') and blob(data[:-1]) == EXPECTED[name], name
        # apply_patch appends one terminal LF when saving a source without one.
        added_terminal_newline = True
    return dict(path=str(path), local_sha256=sha(path), github_blob_sha=EXPECTED[name],
                local_terminal_newline_added=added_terminal_newline)


def main():
    assert not (OUT / 'result.json').exists()
    sources = {name: normalized_source(name) for name in EXPECTED}
    shape = {split: (UPSTREAM / ('shapenet_' + split + '.txt')).read_text().splitlines()
             for split in ('train', 'test')}
    counts = {split: Counter(line.split('-', 1)[0] for line in lines) for split, lines in shape.items()}
    shape_result = dict(
        counts={split: len(lines) for split, lines in shape.items()},
        unique_paths={split: len(set(lines)) for split, lines in shape.items()},
        classes={split: len(rows) for split, rows in counts.items()},
        class_counts={split: dict(sorted(rows.items())) for split, rows in counts.items()},
        shared_paths=sorted(set(shape['train']) & set(shape['test'])),
        shared_model_ids=sorted({line.split('-', 1)[1][:-4] for line in shape['train']} &
                                {line.split('-', 1)[1][:-4] for line in shape['test']}),
        train_classes_missing_from_test=sorted(set(counts['train']) - set(counts['test'])),
        test_class_size_min=min(counts['test'].values()),
        test_class_size_max=max(counts['test'].values()),
        author_config_purpose='Gaussian_MAE pretraining',
        author_validation_and_test_use_same_ID_file=True,
        ready_as_full_55_class_independent_classification_confirmation=False)

    past = ROOT / 'results/native_confirmation_metadata/attempt1/historical_overlap1'
    old_result, old_plan = read(past / 'result.json'), read(past / 'plan.json')
    official_train = ROOT / 'results/round65_actual_gs/preparation/modelnet40_official_train.txt'
    assert sha(official_train) == old_plan['source_sha256'][str(official_train.relative_to(ROOT))]
    published = set(official_train.read_text().splitlines())
    difference = old_result['original_manifest_difference']
    historical_used = (published - set(difference['shapesplat_only'])) | set(difference['hdf5_only'])
    assert len(historical_used) == old_result['historical_total_used'] == 9840
    native_train_path = ROOT / 'results/round65_actual_gs/preparation/train_catalog.json'
    native_test_path = ROOT / 'results/native_confirmation_metadata/attempt1/test_catalog.json'
    all_native_train = read(native_train_path)['rows']
    head60 = {r['category']: r['taxonomy_id'] for r in all_native_train}
    assert len(head60) == 60 and set(head60.values()) == set(range(60))
    native_train = [r for r in all_native_train if r['taxonomy_id'] >= 30]
    native_test = [r for r in read(native_test_path)['rows'] if r['source'] == 'ModelNet']
    native_used = {r['model_id'] for r in native_train + native_test}
    native_classes = {name.rsplit('_', 1)[0] for name in native_used}
    assert len(native_classes) == 30
    used = historical_used | native_used
    modelnet = {split: set((UPSTREAM / ('modelnet40_' + split + '.txt')).read_text().splitlines())
                for split in ('train', 'test')}
    assert not modelnet['train'] & modelnet['test']
    membership = []
    for split, objects in modelnet.items():
        for object_id in sorted(objects):
            category = object_id.rsplit('_', 1)[0]
            membership.append(dict(model_id=object_id, upstream_split=split, category=category,
                used_R48_R58=object_id in historical_used, used_native=object_id in native_used,
                no_known_used_ID=object_id not in used, category_in_ModelNet_source30=category in native_classes,
                exact_head60_taxonomy_id=head60.get(category)))
    remaining = [r for r in membership if r['no_known_used_ID']]
    matched = [r for r in remaining if r['exact_head60_taxonomy_id'] is not None]
    direct_test = [r for r in matched if r['upstream_split'] == 'test']
    modelnet_result = dict(published_split_counts={k: len(v) for k, v in modelnet.items()},
        historical_point_object_count=len(historical_used), native_used_object_count=len(native_used),
        known_used_in_this_repository=len((modelnet['train'] | modelnet['test']) & used),
        no_known_used_ID_count=len(remaining),
        remaining_by_original_split=dict(Counter(r['upstream_split'] for r in remaining)),
        remaining_by_class=dict(sorted(Counter(r['category'] for r in remaining).items())),
        remaining_with_ModelNet_source30=sum(r['category_in_ModelNet_source30'] for r in remaining),
        ModelNet_source30_class_names=sorted(native_classes),
        remaining_exact_head60_matches=len(matched),
        exact_head60_matches_by_split=dict(Counter(r['upstream_split'] for r in matched)),
        provisional_direct_test_count=len(direct_test),
        provisional_direct_test_mapping=[dict(category=c, taxonomy_id=head60[c], objects=n)
            for c, n in sorted(Counter(r['category'] for r in direct_test).items())],
        remaining_without_exact_head60_match=dict(sorted(Counter(r['category'] for r in remaining
            if r['exact_head60_taxonomy_id'] is None).items())),
        aliases_inferred=False, new_data_or_scores_read=False,
        whole_repository_is_fresh=False,
        limitation='ID-set exclusion only; no proof of absence of near-duplicate source geometry.')
    prior_paths = [past / 'plan.json', past / 'result.json', official_train, native_train_path, native_test_path,
                   ROOT / 'results/native_confirmation/attempt1/review.json',
                   ROOT / 'results/round65_actual_gs/attempt1/data/manifest.json']
    output = dict(observed_unix=time.time(), revision=2,
        correction='The previous explanation confused the ModelNet-source 30 classes with the full 60-class head. Seven remaining classes have exact names in the MACGS-source outputs.',
        previous_result_sha256=sha(OUT / 'result_v1.json'), source_files=sources,
        prior_sha256={str(p): sha(p) for p in prior_paths}, analysis_sha256=sha(Path(__file__)),
        shapenet=shape_result, modelnet=modelnet_result,
        native_primitives_or_new_images_read=False, archive_contents_read=False,
        model_scores_read=False, new_experiment_frozen=False, current_fits_changed=False,
        next_action='After current development results, consider a prospectively fixed cross-source subset using 620 original ModelNet TEST objects in seven exact-name head classes. ShapeNet still needs disjoint splits; no new confirmation cohort is frozen yet.')
    for name, payload in (('result.json', output), ('modelnet_membership.json', dict(rows=membership))):
        with (OUT / name).open('x') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(shapenet_counts=shape_result['counts'], shapenet_classes=shape_result['classes'],
        shapenet_train_test_intersection=len(shape_result['shared_paths']), modelnet=modelnet_result), indent=2))


if __name__ == '__main__':
    main()
