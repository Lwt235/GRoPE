"""Native GS preprocessing; no synthetic patch covariance and no test loading."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.round65_common import (ROOT, OUT, RAW, PREP, CACHE_ROOT, CONFIG, checked,
                                        identity_seed, read, sha, write)


def quaternion_rotation(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if not np.isfinite(quaternion).all() or np.any(norm < 1e-12):
        raise ValueError('invalid native quaternion')
    w, x, y, z = np.moveaxis(quaternion/norm, -1, 0)
    # q=(w,v): R=I+2w[v]_x+2[v]_x^2, independent of q's sign.
    cross = np.zeros((*w.shape, 3, 3), dtype=np.float64)
    cross[..., 0, 1], cross[..., 0, 2] = -z, y
    cross[..., 1, 0], cross[..., 1, 2] = z, -x
    cross[..., 2, 0], cross[..., 2, 1] = -y, x
    return np.eye(3)+2*w[..., None, None]*cross+2*cross@cross


def native_arrays(vertex, identity, count=2048):
    centers = np.column_stack([vertex[k] for k in ('x','y','z')]).astype(np.float64)
    logs = np.column_stack([vertex[f'scale_{i}'] for i in range(3)]).astype(np.float64)
    quaternion = np.column_stack([vertex[f'rot_{i}'] for i in range(4)]).astype(np.float64)
    raw_appearance = np.column_stack([vertex[k] for k in ('opacity','f_dc_0','f_dc_1','f_dc_2')]).astype(np.float64)
    if len(centers) < 1 or any(not np.isfinite(v).all() for v in (centers, logs, quaternion, raw_appearance)):
        raise ValueError('empty/nonfinite native object '+identity)
    origin = centers.mean(0)
    radius = np.sqrt(np.square(centers-origin).sum(-1).mean())
    if not np.isfinite(radius) or radius < 1e-10:
        raise ValueError('invalid native object extent '+identity)
    rng = np.random.default_rng(identity_seed(CONFIG['sample_salt'], identity))
    indices = rng.choice(len(centers), count, replace=len(centers)<count)
    normalized = (centers[indices]-origin)/radius
    scales2 = np.exp(2*(logs[indices]-np.log(radius)))
    rotation = quaternion_rotation(quaternion[indices])
    physical = (rotation*scales2[..., None, :])@rotation.swapaxes(-1,-2)
    covariance = physical+CONFIG['covariance_floor']*np.eye(3)
    appearance = raw_appearance[indices].copy()
    appearance[:, 0] = np.tanh(.5*appearance[:, 0])
    appearance[:, 1:] *= .28209479177387814
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(covariance).all() or eigenvalues.min() <= 0:
        raise ValueError('native covariance conversion invalid '+identity)
    floor_fraction = 3*CONFIG['covariance_floor']/np.trace(covariance, axis1=-2, axis2=-1)
    stats = dict(vertices=len(centers),origin=origin.tolist(),rms_radius=float(radius),
                 minimum_eigenvalue=float(eigenvalues.min()),
                 floor_fraction_quantiles=np.quantile(floor_fraction,[0,.5,.9,1]).tolist(),
                 sampled_with_replacement=len(centers)<count)
    result = dict(centers=normalized.astype(np.float32),covariance=covariance,
                  appearance=appearance.astype(np.float32),indices=indices.astype(np.int32))
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('nonfinite converted model input '+identity)
    return result, stats


def read_vertices(path):
    # Released files have one binary vertex element with scalar float fields.
    # A schema mismatch is exposed instead of silently interpreting other data.
    properties, count, endian, vertex = [], None, None, False
    with Path(path).open('rb') as stream:
        if stream.readline() != b'ply\n':
            raise ValueError('expected PLY signature')
        for _ in range(256):
            line = stream.readline(1024).decode('ascii').strip()
            parts = line.split()
            if parts[:1] == ['format']:
                endian = {'binary_little_endian':'<','binary_big_endian':'>'}.get(parts[1])
            elif parts[:1] == ['element']:
                if parts[1] != 'vertex' or count is not None:
                    raise ValueError('unexpected PLY element')
                count, vertex = int(parts[2]), True
            elif parts[:1] == ['property']:
                if not vertex or len(parts) != 3 or parts[1] not in ('float','float32'):
                    raise ValueError('unexpected PLY property')
                properties.append(parts[2])
            elif line == 'end_header':
                offset = stream.tell()
                break
        else:
            raise ValueError('missing bounded PLY header')
    if not endian or not count or len(set(properties)) != len(properties):
        raise ValueError('unsupported or empty native PLY')
    dtype = np.dtype([(name,endian+'f4') for name in properties])
    if Path(path).stat().st_size != offset+count*dtype.itemsize:
        raise ValueError('native PLY byte length differs from header')
    return np.memmap(path, mode='r', dtype=dtype, offset=offset, shape=(count,))


def split_rows(rows):
    result = {name:[] for name in ('fit','val','dev_eval')}
    for category in range(60):
        group = [r for r in rows if r['taxonomy_id']==category]
        group.sort(key=lambda r:hashlib.sha256((CONFIG['split_salt']+':'+r['file_path']).encode()).hexdigest())
        n = len(group)//10
        if n < 1:
            raise ValueError('class cannot support fixed internal split')
        result['val'].extend(group[:n])
        result['dev_eval'].extend(group[n:2*n])
        result['fit'].extend(group[2*n:])
    seen = {}
    for split, records in result.items():
        for row in records:
            prior = seen.setdefault(row['sha256'], split)
            if prior != split:
                raise ValueError('byte-identical objects cross development partitions')
    return result


def prepare(out=OUT):
    lock = checked(out)
    acquisition = read(PREP/'acquisition_complete.json')
    if acquisition['catalog_sha256'] != lock['catalog_sha256']:
        raise ValueError('acquisition population changed')
    record = Path(acquisition['attempt'])/'completion.json'
    if sha(record) != acquisition['completion_sha256'] or not read(record)['completed']:
        raise ValueError('complete acquisition required')
    partitions = split_rows(read(PREP/'train_catalog.json')['rows'])
    dest = out/'data'
    if dest.exists():
        raise ValueError('preserve existing preprocessing; inspect any failure before retry')
    external=CACHE_ROOT/out.name
    external.mkdir(parents=True,exist_ok=False)
    dest.symlink_to(external,target_is_directory=True)
    records = []
    try:
        for split, rows in partitions.items():
            directory = dest/split
            directory.mkdir()
            count, tokens = len(rows), CONFIG['cache_tokens']
            schema = dict(centers=(np.float32,(count,tokens,3)),
                          covariance=(np.float64,(count,tokens,3,3)),
                          appearance=(np.float32,(count,tokens,4)),
                          indices=(np.int32,(count,tokens)), labels=(np.int64,(count,)))
            arrays = {name:np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
                      for name,(dtype,shape) in schema.items()}
            stats = []
            for i,row in enumerate(rows):
                path = RAW/row['file_path']
                if path.stat().st_size != row['size'] or sha(path) != row['sha256']:
                    raise ValueError('source PLY identity mismatch '+row['file_path'])
                values, item = native_arrays(read_vertices(path),row['file_path'],tokens)
                for name,value in values.items():
                    arrays[name][i] = value
                arrays['labels'][i] = row['taxonomy_id']
                stats.append(dict(path=row['file_path'],source_sha256=row['sha256'],**item))
                if (i+1)%100==0:
                    print('PREPARED',split,i+1,'/',len(rows),flush=True)
            for value in arrays.values():
                value.flush()
            write(directory/'conversion.json',stats)
            records.append(dict(split=split,count=count,ids=[r['file_path'] for r in rows],
                class_counts=np.bincount([r['taxonomy_id'] for r in rows],minlength=60).tolist(),
                hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file()}))
        write(dest/'manifest.json',dict(partitions=records,catalog_sha256=lock['catalog_sha256'],
            official_test_access=False,all_data_declared_development=True,
            higher_SH_used=False,native_primitives_not_patch_aggregates=True))
        (dest/'manifest.sha256').write_text(sha(dest/'manifest.json')+'\n')
    except BaseException as error:
        write(dest/'failure.json',dict(error_type=type(error).__name__,message=str(error)))
        raise


def load(split,out=OUT):
    checked(out)
    if split not in ('fit','val','dev_eval'):
        raise ValueError('only declared development partitions supported')
    if split=='dev_eval' and not (out/'selected.json').exists():
        raise ValueError('seal all selected checkpoints before task evaluation')
    manifest=out/'data/manifest.json'
    if sha(manifest)!=(out/'data/manifest.sha256').read_text().strip():
        raise ValueError('preprocessing manifest identity')
    row=next(r for r in read(manifest)['partitions'] if r['split']==split)
    directory=out/'data'/split
    for name,digest in row['hashes'].items():
        if sha(directory/name)!=digest:
            raise ValueError('cached inputs changed')
    result={name:np.load(directory/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            for name in ('centers','covariance','appearance','indices','labels')}
    result['ids']=np.asarray(row['ids'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    prepare(parser.parse_args().out)
