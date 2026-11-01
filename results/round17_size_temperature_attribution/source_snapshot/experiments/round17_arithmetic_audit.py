"""Frozen extension: same-prepared R17 arithmetic from saved numeric arrays.

No original observation, sensor or target archive is opened. The new audit
uses independent NumPy math N and calls fixed P only for actual tau/bias.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy
import torch

from experiments import round17_trace_numerical_audit as N
from models import round17_trace_pe as P

ROOT=Path(__file__).resolve().parents[1]
SOURCE_NAMES=('docs/round17_arithmetic_addendum.md',
              'experiments/round17_arithmetic_audit.py',
              'tests/test_round17_arithmetic_audit.py')
SCENES=('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
PARTITIONS=('g8','g16','mixed')
CONDITIONS=tuple((q,k) for q in PARTITIONS for k in PARTITIONS)+(
    ('g8','g24'),('g24','g8'),('g16','g24'),('g24','g16'),('g24','g24'))
WIDTHS=(.25,.5,1.,2.)
TEMPERATURES=(0.,5.,10.)
SAME_MASS_TOLERANCE=1e-12
SUPPORT_FIELDS={'indices','actual_mean','actual_cov','actual_content','reference_mean',
                'reference_cov','reference_content','bounds'}
GEOMETRY_FIELDS={'actual_raw','reference64_raw','reference96_raw','hybrid_raw'}|{
    f'{name}_T{t:g}' for name in ('actual','reference64','reference96','hybrid') for t in TEMPERATURES}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def _source_hashes():return {name:sha(ROOT/name) for name in SOURCE_NAMES}


def _base(run):
    """Verify code/partition metadata only; deliberately no data_records call."""
    run=Path(run);lock=read(run/'lock.json');digest=sha(run/'lock.json')
    if (run/'lock.sha256').read_text().strip()!=digest:raise ValueError('base numerical lock changed')
    required={'experiments/round17_trace_numerical_audit.py','models/round17_trace_pe.py'}
    if not required<=set(lock['source_hashes']):raise ValueError('base numerical math source binding missing')
    for name,expected in lock['source_hashes'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('invalid base source name')
        if sha(ROOT/path)!=expected or sha(run/'source_snapshot'/path)!=expected:
            raise ValueError('base numerical source/snapshot changed')
    if sha(run/'partitions.npz')!=lock['partitions_sha256']:raise ValueError('base partitions changed')
    if (lock['conditions']!=[list(x) for x in CONDITIONS]
            or lock['grid']!=[dict(h=h,temperature=t) for h in WIDTHS for t in TEMPERATURES]
            or [e['scene'] for e in lock['cohort']]!=list(SCENES)
            or lock['probability_rows']!=4032 or lock['geometry_rows']!=1344):
        raise ValueError('base numerical cohort/policy differs')
    return lock,digest


def freeze(out,run):
    out,run=Path(out),Path(run).resolve()
    if out.exists():raise FileExistsError('refuse existing arithmetic run')
    if (run/'train').exists():raise ValueError('arithmetic extension must freeze before train starts')
    base,digest=_base(run);sources=_source_hashes()
    out.mkdir(parents=True)
    for name in sources:
        destination=out/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    lock=dict(created_unix=time.time(),run_root=str(run),base_numeric_lock_sha256=digest,
        base_source_hashes=base['source_hashes'],new_source_hashes=sources,train_absent_at_freeze=True,
        support_archives=42,geometry_archives=168,probability_matrices=504,probability_rows=4032,
        same_mass_tolerance=SAME_MASS_TOLERANCE,numpy_version=np.__version__,
        scipy_version=scipy.__version__,torch_version=torch.__version__,
        original_observation_inputs_read=False,labels_read=False)
    # Catch a concurrently started train, rather than issuing a false timing claim.
    if (run/'train').exists():raise ValueError('train started during extension freeze; preserve partial output')
    write(out/'lock.json',lock);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    return lock


def verify_lock(out):
    out=Path(out);lock=read(out/'lock.json')
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('arithmetic lock changed')
    base,digest=_base(lock['run_root'])
    if (digest!=lock['base_numeric_lock_sha256'] or base['source_hashes']!=lock['base_source_hashes']
            or lock['new_source_hashes']!=_source_hashes() or lock['train_absent_at_freeze'] is not True
            or lock['same_mass_tolerance']!=SAME_MASS_TOLERANCE
            or lock['numpy_version']!=np.__version__ or lock['scipy_version']!=scipy.__version__
            or lock['torch_version']!=torch.__version__):
        raise ValueError('arithmetic provenance/runtime changed')
    for name,digest in lock['new_source_hashes'].items():
        if sha(out/'source_snapshot'/name)!=digest:raise ValueError('arithmetic source snapshot changed')
    return lock,base


def _completed(run,stage,numeric_digest):
    directory=run/stage;completion=read(directory/'completion.json')
    if completion['passed'] is not True or completion['lock_sha256']!=numeric_digest:
        raise ValueError('passing original numerical stage required')
    expected={str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p!=directory/'completion.json'}
    if set(completion['files'])!=expected:raise ValueError('original numerical artifact population changed')
    for name,digest in completion['files'].items():
        if sha(directory/name)!=digest:raise ValueError('original numerical artifact hash changed')
    summary=read(directory/'summary.json')
    if summary['passed'] is not True or summary['lock_sha256']!=numeric_digest:
        raise ValueError('original summary is not a numerical pass')
    return completion,summary


@torch.no_grad()
def audit_matrix(raw,content,cov,bounds,saved,h,T):
    """Return one complete arithmetic record, with no I/O or integration."""
    raw=np.asarray(raw);content=np.asarray(content);cov=np.asarray(cov);saved=np.asarray(saved)
    bounds=np.asarray(bounds)
    if (raw.dtype!=np.float64 or raw.ndim!=2 or saved.dtype!=np.float64 or raw.shape!=saved.shape
            or content.dtype!=np.float32 or content.shape!=raw.shape or cov.dtype!=np.float32
            or cov.shape!=(len(raw),2,2) or bounds.shape!=(raw.shape[1],4)
            or not np.isfinite(saved).all() or np.any(saved<0) or np.any(abs(saved.sum(1)-1)>1e-12)):
        raise ValueError('saved same-prepared arithmetic schema differs')
    tc=torch.from_numpy(cov);cc=torch.from_numpy(content)
    actual_tau=P.trace_temperature(tc,h,T).numpy()
    actual_bias=P.content_bias(tc,h,T,cc).numpy()
    reference_tau=N.independent_temperature(cov,h,T).astype(np.float32)
    reference_bias=N.independent_bias(content,cov,h,T).astype(np.float32)
    reference=N.independent_probability(raw,content,cov,h,T,bounds)
    error=np.max(np.abs(saved-reference),axis=1)
    tau_equal=np.array_equal(actual_tau,reference_tau);bias_equal=np.array_equal(actual_bias,reference_bias)
    winner=saved.argmax(1);expected=reference.argmax(1)
    return dict(passed=bool(tau_equal and bias_equal and error.max()<=SAME_MASS_TOLERANCE),
        queries=len(raw),keys=raw.shape[1],tau32_exact=bool(tau_equal),bias32_exact=bool(bias_equal),
        max_same_mass_probability_error=float(error.max()),per_query_probability_error=error.tolist(),
        actual_tau32=actual_tau.tolist(),reference_tau32=reference_tau.tolist(),
        max_tau32_error=float(np.max(abs(actual_tau.astype(float)-reference_tau.astype(float)))),
        max_bias32_error=float(np.max(abs(actual_bias.astype(float)-reference_bias.astype(float)))),
        tau_mismatch_rows=np.flatnonzero(actual_tau!=reference_tau).tolist(),
        bias_mismatch_rows=np.flatnonzero(np.any(actual_bias!=reference_bias,axis=1)).tolist(),
        argmax_mismatch_rows=np.flatnonzero(winner!=expected).tolist(),
        raw_zero_fov_rows=int(np.count_nonzero(raw.sum(1)==0)),
        raw_tiny_fov_rows=int(np.count_nonzero(raw.sum(1)<1e-20)),
        array_hashes={name:array_sha(value) for name,value in (
            ('actual_raw',raw),('actual_cov',cov),('actual_content',content),('bounds',bounds),
            ('saved_probability',saved),('same_mass_reference_probability',reference),
            ('actual_tau32_recomputed',actual_tau),('reference_tau32',reference_tau),
            ('actual_bias32_recomputed',actual_bias),('reference_bias32',reference_bias))})


def audit(out):
    out=Path(out);lock,base=verify_lock(out);run=Path(lock['run_root']);numeric=lock['base_numeric_lock_sha256']
    synthetic,_=_completed(run,'synthetic',numeric)
    train,summary=_completed(run,'train',numeric)
    started=read(run/'train'/'started.json')
    if (started['lock_sha256']!=numeric or train['started_unix']!=started['started_unix']
            or not base['created_unix']<=lock['created_unix']<started['started_unix']
            or synthetic['finished_unix']>started['started_unix']
            or summary['probability_rows']!=4032 or summary['geometry_rows']!=1344 or summary['support_sets']!=42):
        raise ValueError('extension/preflight chronology or complete population differs')
    if any((out/name).exists() for name in ('audit_started.json','verification.json','completion.json')):
        raise FileExistsError('refuse completed or partial arithmetic audit')
    begun=time.time();write(out/'audit_started.json',dict(started_unix=begun,arithmetic_lock_sha256=sha(out/'lock.json')))
    records=[];expected_archives=set();support_count=0;geometry_count=0
    with np.load(run/'partitions.npz',allow_pickle=False) as partitions:
        counts={q:len(partitions[q+'__bounds']) for q in (*PARTITIONS,'g24')}
    for entry in base['cohort']:
        prefix=Path('arrays')/entry['scene'].replace('/','__')
        for q,k in CONDITIONS:
            relative=prefix/f'{q}_{k}_support.npz';expected_archives.add(str(relative))
            support_path=run/'train'/relative;support_count+=1
            with np.load(support_path,allow_pickle=False) as data:
                if set(data.files)!=SUPPORT_FIELDS:raise ValueError('support archive fields differ')
                cov=data['actual_cov'];content=data['actual_content'];bounds=data['bounds'];indices=data['indices']
                if (not np.array_equal(indices,np.linspace(0,counts[q]-1,8,dtype=np.int64))
                        or cov.shape!=(8,2,2) or content.shape!=(8,counts[k])):
                    raise ValueError('saved support query/key population differs')
            for h in WIDTHS:
                relative_geometry=prefix/f'{q}_{k}_h{h:g}.npz';expected_archives.add(str(relative_geometry));geometry_count+=1
                path=run/'train'/relative_geometry
                with np.load(path,allow_pickle=False) as data:
                    if set(data.files)!=GEOMETRY_FIELDS:raise ValueError('geometry archive fields differ')
                    raw=data['actual_raw']
                    for T in TEMPERATURES:
                        record=audit_matrix(raw,content,cov,bounds,data[f'actual_T{T:g}'],h,T)
                        records.append(dict(scene=entry['scene'],pair=entry['pair'],query=q,key=k,h=h,temperature=T,
                            query_indices=indices.tolist(),support_archive=str(relative),
                            support_archive_sha256=train['files'][str(relative)],geometry_archive=str(relative_geometry),
                            geometry_archive_sha256=train['files'][str(relative_geometry)],**record))
    actual_archives={name for name in train['files'] if name.endswith('.npz')}
    if (actual_archives!=expected_archives or support_count!=42 or geometry_count!=168
            or len(records)!=504 or sum(r['queries'] for r in records)!=4032):
        raise ValueError('incomplete same-prepared arithmetic population')
    write(out/'matrix_rows.json',records)
    verification=dict(passed=all(r['passed'] for r in records),
        arithmetic_lock_sha256=sha(out/'lock.json'),base_numeric_lock_sha256=numeric,
        train_summary_sha256=sha(run/'train'/'summary.json'),train_completion_sha256=sha(run/'train'/'completion.json'),
        arithmetic_frozen_unix=lock['created_unix'],train_started_unix=started['started_unix'],
        source_hashes=lock['new_source_hashes'],base_source_hashes=lock['base_source_hashes'],
        support_archives=support_count,geometry_archives=geometry_count,
        probability_matrices=len(records),probability_rows=sum(r['queries'] for r in records),
        tau32_exact=all(r['tau32_exact'] for r in records),bias32_exact=all(r['bias32_exact'] for r in records),
        max_same_mass_probability_error=max(r['max_same_mass_probability_error'] for r in records),
        same_mass_tolerance=SAME_MASS_TOLERANCE,argmax_mismatch_rows=sum(len(r['argmax_mismatch_rows']) for r in records),
        failed_matrix_indices=[i for i,r in enumerate(records) if not r['passed']],
        actual_tau_bias_recomputed_from_locked_function=True,original_tau_bias_arrays_were_not_saved=True,
        original_observation_inputs_read=False,labels_read=False,validation_or_test_read=False,
        new_real_predictions=False,scientific_gate_passed=False,started_unix=begun,finished_unix=time.time())
    verify_lock(out)
    write(out/'verification.json',verification)
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'completion.json',dict(passed=verification['passed'],arithmetic_lock_sha256=sha(out/'lock.json'),
        base_numeric_lock_sha256=numeric,files=files,finished_unix=time.time()))
    return verification


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','audit'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round17_arithmetic_audit')
    parser.add_argument('--run',type=Path,default=ROOT/'results/round17_preflight')
    args=parser.parse_args()
    result=freeze(args.out,args.run) if args.stage=='freeze' else audit(args.out)
    print(json.dumps(result if args.stage=='freeze' else {k:result[k] for k in
        ('passed','probability_matrices','probability_rows','max_same_mass_probability_error')},indent=2))


if __name__=='__main__':main()
