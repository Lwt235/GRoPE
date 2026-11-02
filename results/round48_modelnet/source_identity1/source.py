"""Check original mesh IDs across the fixed official-train development split.

Read only the five train ID lists and class-name metadata inside the already
verified archive. No HDF5 member, prediction, or official-test member is opened.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import zipfile

from experiments.round48_acquire import ARCHIVE, EXPECTED_SHA
from experiments.round48_common import ROOT, OUT, checked, read, write, sha


def main():
    checked(OUT)
    if sha(ARCHIVE) != EXPECTED_SHA:
        raise ValueError('archive identity')
    manifest_path = OUT/'data/manifest.json'
    if sha(manifest_path) != manifest_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('partition manifest identity')
    manifest = read(manifest_path)
    dest = ROOT/'results/round48_modelnet/source_identity1'
    if dest.exists():
        raise ValueError('preserve existing metadata audit')
    dest.mkdir(parents=True)
    prefix = 'modelnet40_ply_hdf5_2048/'
    names = [prefix+f'ply_data_train_{i}_id2file.json' for i in range(5)]
    whitelist = names+[prefix+'shape_names.txt']
    source = Path(__file__).read_bytes()
    write(dest/'plan.json', dict(
        frozen_unix=time.time(), source_sha256=hashlib.sha256(source).hexdigest(),
        archive_sha256=EXPECTED_SHA, partition_manifest_sha256=sha(manifest_path),
        metadata_whitelist=whitelist, hdf5_access=False, official_test_access=False,
        purpose='Check original object identity disjointness, without changing the split or fitting a model.',
        canonical_id_rule='Lowercase filename stem; class prefix remains part of identity.',
        limit='Does not detect geometric near-duplicates with different object names.'))
    (dest/'source.py').write_bytes(source)
    payloads, opened = {}, []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in whitelist:
            payloads[name] = archive.read(name)
            opened.append(dict(name=name, sha256=hashlib.sha256(payloads[name]).hexdigest()))
    classes = payloads[prefix+'shape_names.txt'].decode().split()
    if len(classes) != 40 or len(set(classes)) != 40:
        raise ValueError('class-name metadata')
    mapping, source_counts = {}, []
    for i, name in enumerate(names):
        values = json.loads(payloads[name])
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError('train ID-list schema')
        source_counts.append(len(values))
        for row, value in enumerate(values):
            canonical = PurePosixPath(value.replace('\\', '/')).stem.lower()
            matches = [c for c in classes if canonical.startswith(c+'_')]
            if len(matches) != 1:
                raise ValueError('unrecognized object class prefix')
            key = prefix+f'ply_data_train{i}.h5:'+str(row)
            mapping[key] = dict(object_id=canonical, class_name=matches[0])
    if len(mapping) != 9840:
        raise ValueError('official-train metadata population')
    by_split, rows, seen = {}, [], set()
    for record in manifest['partitions']:
        selected = [mapping[identity] for identity in record['ids']]
        if seen.intersection(record['ids']):
            raise ValueError('overlapping declared archive row IDs')
        seen.update(record['ids'])
        object_ids = [item['object_id'] for item in selected]
        counts = Counter(item['class_name'] for item in selected)
        actual = [counts[name] for name in classes]
        by_split[record['split']] = set(object_ids)
        rows.append(dict(split=record['split'], archive_rows=len(object_ids),
                         unique_original_objects=len(set(object_ids)),
                         within_split_duplicate_ids={k:v for k,v in Counter(object_ids).items() if v > 1},
                         metadata_class_counts=actual, manifest_class_counts=record['class_counts'],
                         class_counts_match=actual == record['class_counts']))
    if seen != set(mapping):
        raise ValueError('partition coverage differs from official-train ID lists')
    splits = ('fit', 'val', 'dev_eval')
    intersections = []
    for i, left in enumerate(splits):
        for right in splits[i+1:]:
            shared = sorted(by_split[left] & by_split[right])
            intersections.append(dict(left=left, right=right, shared_count=len(shared), shared_ids=shared))
    passed = all(row['class_counts_match'] for row in rows) and all(r['shared_count'] == 0 for r in intersections)
    result = dict(passed=passed, original_objects=len({r['object_id'] for r in mapping.values()}),
                  archive_rows=len(mapping), metadata_counts=source_counts, partitions=rows,
                  intersections=intersections, metadata_members_opened=opened,
                  hdf5_members_opened=[], official_test_members_opened=[],
                  prediction_arrays_opened=[], scientific_configuration_changed=False,
                  limit='Object-name identity only; not an audit of unnamed geometric near-duplicates or test-set overlap.')
    write(dest/'result.json', result)
    print(json.dumps(dict(passed=passed, original_objects=result['original_objects'],
        partition_objects={r['split']:r['unique_original_objects'] for r in rows},
        cross_partition_shared_counts=[r['shared_count'] for r in intersections],
        class_counts_match=[r['class_counts_match'] for r in rows], array_members_opened=[])))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
