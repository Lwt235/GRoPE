"""Read source-ID metadata, never point arrays, native test PLY or scores."""
import collections
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT = Path('/home/liwt/GRoPE')
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    plan = read(OUT / 'plan.json')
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    archive_path = ROOT / plan['archive']
    assert sha(archive_path) == plan['archive_expected_sha256']
    historical = read(ROOT / 'results/round48_modelnet/attempt1/data/manifest.json')
    assert historical['archive_sha256'] == plan['archive_expected_sha256']
    row_to_object, test_ids, members = {}, set(), []
    with zipfile.ZipFile(archive_path) as archive:
        for split, count in (('train', 5), ('test', 2)):
            for index in range(count):
                member = f'modelnet40_ply_hdf5_2048/ply_data_{split}_{index}_id2file.json'
                payload = archive.read(member)
                names = json.loads(payload)
                members.append(dict(member=member, count=len(names),
                                    sha256=hashlib.sha256(payload).hexdigest()))
                for row, name in enumerate(names):
                    identity = Path(name).stem
                    if split == 'train':
                        source_row = f'modelnet40_ply_hdf5_2048/ply_data_train{index}.h5:{row}'
                        assert source_row not in row_to_object
                        row_to_object[source_row] = identity
                    else:
                        assert identity not in test_ids
                        test_ids.add(identity)
    train_ids = set(row_to_object.values())
    published_train = set((ROOT / 'results/round65_actual_gs/preparation/modelnet40_official_train.txt').read_text().splitlines())
    assert len(row_to_object) == len(train_ids) == historical['total'] == 9840
    assert len(test_ids) == 2468 and not train_ids & test_ids
    manifest_difference = dict(hdf5_train_count=len(train_ids),
        shapesplat_manifest_count=len(published_train), intersection=len(train_ids & published_train),
        hdf5_only=sorted(train_ids - published_train), shapesplat_only=sorted(published_train - train_ids))
    used = {}
    for partition in historical['partitions']:
        assert len(partition['ids']) == partition['count']
        for row in partition['ids']:
            identity = row_to_object[row]
            assert identity not in used
            used[identity] = dict(historical_split=partition['split'], source_row=row)
    assert set(used) == train_ids
    catalog = read(OUT.parent / 'test_catalog.json')
    rows = []
    for item in catalog['rows']:
        record = dict(file_path=item['file_path'], model_id=item['model_id'],
                      taxonomy_id=item['taxonomy_id'], source=item['source'])
        if item['source'] == 'ModelNet':
            if item['model_id'] in used:
                record.update(used[item['model_id']], historical_status='used_original_train')
            elif item['model_id'] in test_ids:
                record['historical_status'] = 'original_test_not_used_in_R48_R58'
            else:
                record['historical_status'] = 'unresolved'
        else:
            record['historical_status'] = 'not_ModelNet'
        rows.append(record)
    status_counts = collections.Counter(row['historical_status'] for row in rows)
    split_counts = collections.Counter(row['historical_split'] for row in rows if 'historical_split' in row)
    no_known_overlap = [row for row in rows if row['historical_status'] in
                       ('original_test_not_used_in_R48_R58', 'not_ModelNet')]
    class_counts = collections.Counter(row['taxonomy_id'] for row in no_known_overlap)
    write_new(OUT / 'object_membership.json', dict(rows=rows))
    result = dict(observed_unix=time.time(), plan_sha256=sha(OUT / 'plan.json'),
        analysis_sha256=sha(Path(__file__)), archive_sha256=plan['archive_expected_sha256'],
        metadata_members_read=members, historical_total_used=len(used),
        exact_original_train_ID_set_matches_previous_published_manifest=(train_ids == published_train),
        original_manifest_difference=manifest_difference,
        status_counts=dict(status_counts), reused_by_historical_split=dict(split_counts),
        no_known_historical_source_overlap_count=len(no_known_overlap),
        no_known_overlap_classes=len(class_counts),
        no_known_overlap_class_counts={str(k): v for k, v in sorted(class_counts.items())},
        object_membership_sha256=sha(OUT / 'object_membership.json'),
        test_point_arrays_read=False, native_test_PLY_read=False, model_scores_read=False,
        conclusion='Full GSModel60 official TEST is an unused native-format evaluation, but is not wholly source-object-unseen in this research. Preserve the full benchmark and prospectively separate source reuse.',
        limits=['Historical source reuse affects research-level freshness, not evidence that native scratch-training weights directly consumed these TEST PLYs.',
                'No known source-ID overlap does not rule out related geometry or establish effectiveness.',
                'This metadata classification does not itself authorize claims of fresh confirmation or select models.'],
        new_effectiveness_gate=False, goal_complete=False)
    write_new(OUT / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('status_counts', 'reused_by_historical_split',
          'no_known_historical_source_overlap_count', 'no_known_overlap_classes')}, indent=2))
