"""Source/split/capacity checks only; never open a test PLY or a prediction."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

ROOT = Path('/home/liwt/GRoPE')
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'results/round65_actual_gs/preparation'
RAW = Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GSModel60')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    assert not (OUT / 'result.json').exists()
    plan, transport = read(OUT / 'plan.json'), read(OUT / 'transport.json')
    assert transport['all_fetched'] and transport['plan_sha256'] == sha(OUT / 'plan.json')
    sources = dict(plan['existing_catalog_sources'])
    for name, digest in sources.items():
        assert sha(Path(name)) == digest
    for record in transport['records']:
        path = OUT / 'metadata_sources' / record['path']
        assert sha(path) == record['sha256']
        sources[str(path)] = sha(path)
    info = read(OLD / 'dataset_info.json')
    assert sha(OUT / 'metadata_sources/test.json') == info['metadata_sha256']['test.json']
    paths = read(OUT / 'metadata_sources/test.json')
    test = [json.loads(line) for line in (OUT / 'metadata_sources/metadata/test.jsonl').read_text().splitlines()]
    train = read(OLD / 'train_catalog.json')['rows']
    assert len(paths) == len(set(paths)) == len(test) == info['num_test'] == 6090
    assert set(paths) == {row['file_path'] for row in test}
    trees = {}
    for category in range(60):
        path = OLD / 'trees' / f'{category}.json'
        sources[str(path)] = sha(path)
        for item in read(path):
            if item['type'] == 'file' and item['path'].endswith('.ply'):
                assert item['path'] not in trees
                assert re.fullmatch('[0-9a-f]{64}', item['lfs']['oid'])
                assert item['size'] == item['lfs']['size']
                trees[item['path']] = item
    assert len(trees) == info['num_samples'] == 13847
    assert sum(item['size'] for item in trees.values()) == info['total_file_bytes']
    for row in train:
        item = trees[row['file_path']]
        assert item['size'] == row['size'] and item['lfs']['oid'] == row['sha256']
    original_train = set((OLD / 'modelnet40_official_train.txt').read_text().splitlines())
    catalog = []
    for row in test:
        path = Path(row['file_path'])
        assert row['split'] == 'test' and len(path.parts) == 2 and '..' not in path.parts
        assert int(path.parts[0]) == row['taxonomy_id'] and path.suffix == '.ply'
        assert path.stem == row['model_id']
        item = trees[row['file_path']]
        catalog.append(dict(row, source='MACGS' if row['taxonomy_id'] < 30 else 'ModelNet',
                            size=item['size'], sha256=item['lfs']['oid'],
                            original_modelnet_train_member=(row['taxonomy_id'] >= 30 and row['model_id'] in original_train)))
    train_hashes, test_hashes = defaultdict(list), defaultdict(list)
    for row in train:
        train_hashes[row['sha256']].append(row['file_path'])
    for row in catalog:
        test_hashes[row['sha256']].append(row['file_path'])
    class_counts = Counter(row['taxonomy_id'] for row in test)
    total = sum(row['size'] for row in catalog)
    free = shutil.disk_usage(RAW).free
    catalog_path = OUT / 'test_catalog.json'
    write_new(catalog_path, dict(repository=plan['repository'], revision=plan['revision'],
                                rows=catalog, total_bytes=total, vertices_decoded=False))
    result = dict(observed_unix=time.time(), repository=plan['repository'], revision=plan['revision'],
        official_test_rows=len(test), official_train_rows=len(train), classes=len(class_counts),
        class_counts={str(i):class_counts[i] for i in range(60)},
        class_count_range=[min(class_counts.values()), max(class_counts.values())],
        groups={group:dict(objects=sum(row['source'] == group for row in catalog),
                           bytes=sum(row['size'] for row in catalog if row['source'] == group))
                for group in ('MACGS', 'ModelNet')},
        train_test_path_overlap=sorted(set(paths) & {row['file_path'] for row in train}),
        train_test_model_id_overlap=sorted({row['model_id'] for row in test} & {row['model_id'] for row in train}),
        train_test_lfs_sha256_overlap=sorted(set(train_hashes) & set(test_hashes)),
        within_test_byte_duplicate_groups={digest:members for digest,members in test_hashes.items() if len(members)>1},
        modelnet_test_members_in_original_modelnet_train=sum(row['original_modelnet_train_member'] for row in catalog),
        remaining_modelnet_membership='Not established as original TEST; only the previously recorded original TRAIN manifest was checked.',
        total_ply_bytes=total, maximum_ply_bytes=max(row['size'] for row in catalog),
        existing_raw_test_paths=[row['file_path'] for row in catalog if (RAW / row['file_path']).exists()],
        observed_free_bytes=free, prior_acquisition_reserve_bytes=50*2**30,
        shortfall_bytes_for_all_raw_plus_prior_reserve=max(0,total+50*2**30-free),
        cached_array_lower_estimate_bytes=6090*2048*15*4,
        cache_estimate_scope='Only float32 centers3/covariance9/appearance3 at 2048 primitives; excludes IDs, labels, metadata, temporary raw objects, safety reserve and implementation differences.',
        test_manifest_sha_verified=True, all_prior_train_tree_identities_verified=True,
        all_tree_population_bytes_match_published=True, raw_vertex_content_read=False,
        test_predictions_generated=False, test_model_selection=False,
        source_sha256=sources, plan_sha256=sha(OUT / 'plan.json'),
        catalog_sha256=sha(catalog_path), analysis_source_sha256=sha(Path(__file__)),
        limitations=['No exact path/ID/file-hash overlap is not a proof of no geometrically related or near-duplicate objects.',
                    'GSModel60 TEST is not identical to the original ModelNet TEST.',
                    'Current native training starts from scratch on GSModel60 TRAIN; original ModelNet TRAIN membership alone does not establish leakage into these models.',
                    'All classes must remain in a prospective full confirmation; no convenient subset was selected.',
                    'Metadata/labels have now been read; test PLY inputs and model predictions remain unread.'],
        decision='Source identities and population are available through the preferred mirror. Full raw retention exceeds current capacity. Before opening test PLY, freeze the final evaluated models/controls and choose a bounded streaming/cache acquisition or free sufficient space; do not replace the full population with a convenient subset.',
        official_confirmation_executed=False, goal_complete=False)
    write_new(OUT / 'result.json', result)
    print(json.dumps({key:result[key] for key in ('official_test_rows','classes','class_count_range',
          'groups','modelnet_test_members_in_original_modelnet_train','total_ply_bytes',
          'maximum_ply_bytes','observed_free_bytes','shortfall_bytes_for_all_raw_plus_prior_reserve',
          'cached_array_lower_estimate_bytes','catalog_sha256')}, indent=2), flush=True)
