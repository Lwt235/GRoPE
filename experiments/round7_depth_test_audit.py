"""Independent31-archive audit, executable only after root's test-complete notice.

Recompute all metrics from saved predictions, and query labels directly from
native SUN3D PNGs. No test discovery at import, model replay, or fitting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from experiments.round7_depth_audit import (ROOT,SEEDS,ROLES,sha,numpy_rows,numpy_macro,
    compare_tree,boundary_from_labels)


def independent_sensor_centers(encoded):
    """Decode rotated uint16 mm and nearest-sample ALL stride8 query centers."""
    bits=np.asarray(encoded,dtype=np.uint32)
    if bits.ndim!=2 or np.any(bits>65535):raise ValueError('expected native uint16 SUN3D depth')
    millimeters=(bits//8)+(bits%8)*8192
    z=millimeters.astype(np.float32)/np.float32(1000)
    height,width=z.shape
    yy,xx=np.meshgrid(np.arange(18)*8+3.5,np.arange(24)*8+3.5,indexing='ij')
    xn=(xx+.5)*(width/192)-.5;yn=(yy+.5)*(height/144)-.5
    ix=np.floor(xn+.5+1e-9).astype(np.int64);iy=np.floor(yn+.5+1e-9).astype(np.int64)
    if ix.min()<0 or iy.min()<0 or ix.max()>=width or iy.max()>=height:raise ValueError('native center out of bounds')
    target=z[iy,ix].reshape(-1);valid=np.isfinite(target)&(target>0)
    target=np.where(valid,target,0).astype(np.float32)
    return dict(target=target,valid=valid,boundary=boundary_from_labels(target[None],valid[None])[0],
        native_hw=list(z.shape),native_first_xy=[float(xn[0,0]),float(yn[0,0])])


def locked_inputs(confirmation,raw_root):
    lockpath=confirmation/'lock.json';datalockpath=confirmation/'test_data_lock.json'
    completion=json.loads((confirmation/'completion.json').read_text())
    lock=json.loads(lockpath.read_text());datalock=json.loads(datalockpath.read_text())
    if completion['lock_sha256']!=sha(lockpath) or completion['test_data_lock_sha256']!=sha(datalockpath):raise ValueError('completion/lock mismatch')
    if datalock['confirmation_lock_sha256']!=sha(lockpath) or datalock['pair_count']!=480:raise ValueError('test data lock mismatch')
    if lock['seeds']!=list(SEEDS) or set(lock['roles'])!=set(ROLES) or lock.get('test_export_authorized') is not True:raise ValueError('unexpected confirmation protocol')
    if 'experiments/round7_depth_test_audit.py' not in lock['source_hashes']:raise ValueError('audit wrapper was not frozen before test')
    for name,digest in lock['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('locked source differs: '+name)
    if sha(lock['selection_path'])!=lock['selection_sha256']:raise ValueError('changed test selection')
    if len(set(lock['expected_scenes']))!=5 or any(len(pairs)!=96 for pairs in lock['expected_pairs'].values()):raise ValueError('requires five96-pair named groups')
    selection=json.loads(Path(lock['selection_path']).read_text())
    selection_rows={r['scene']:r for r in selection['scenes']}
    if set(selection_rows)!=set(lock['expected_scenes']):raise ValueError('selection named groups differ')
    targets=[];valid=[];boundary=[];ids=[];raw_prediction=[];sensor_report=[]
    for source in datalock['sources']:
        directory=Path(source['path']);m=json.loads((directory/'manifest.json').read_text())
        if m['scene'] not in lock['expected_scenes'] or m['split']!='test':raise ValueError('unexpected label population')
        if m['pairs']!=lock['expected_pairs'][m['scene']]:raise ValueError('changed per-scene pair order')
        if m['selection_sha256']!=lock['selection_sha256']:raise ValueError('export selection mismatch')
        if m['exporter_sha256']!=lock['source_hashes']['experiments/round7_depth_data.py']:raise ValueError('exporter source mismatch')
        for filename,key in [('manifest.json','manifest_sha256'),('inputs.npz','input_sha256'),('labels.npz','label_sha256')]:
            if sha(directory/filename)!=source[key]:raise ValueError('changed test cache/hash')
        with np.load(directory/'labels.npz',allow_pickle=False) as labels:
            targets.append(labels['target'].copy());valid.append(labels['valid'].copy());boundary.append(labels['boundary'].copy())
        with np.load(directory/'inputs.npz',allow_pickle=False) as inputs:raw_prediction.append(inputs['q_depth'].copy())
        frames={int(r['index']):r for r in selection_rows[m['scene']]['frames']}
        query_hashes={int(r['index']):r['sensor_sha256'] for r in m['query_sensor_sources']}
        independent={}
        for qi in sorted({q for q,k in m['pairs']}):
            path=raw_root/m['scene']/'depth'/frames[qi]['depth']
            if sha(path)!=query_hashes[qi]:raise ValueError('native query sensor hash mismatch')
            with Image.open(path) as image:encoded=np.asarray(image).copy()
            independent[qi]=independent_sensor_centers(encoded)
            sensor_report.append(dict(scene=m['scene'],query_index=qi,path=str(path),sha256=sha(path),
                native_hw=independent[qi]['native_hw'],native_first_xy=independent[qi]['native_first_xy']))
        for name,exported in [('target',targets[-1]),('valid',valid[-1]),('boundary',boundary[-1])]:
            original=np.stack([independent[q][name] for q,k in m['pairs']])
            if not np.array_equal(original,exported):raise ValueError('raw sensor '+name+' mismatch')
        ids.extend(dict(scene=m['scene'],q_index=q,k_index=k,pair_id=f'{q}:{k}') for q,k in m['pairs'])
    actual={(r['scene'],r['q_index'],r['k_index']) for r in ids}
    expected={(scene,q,k) for scene,pairs in lock['expected_pairs'].items() for q,k in pairs}
    if len(ids)!=480 or actual!=expected:raise ValueError('incomplete/duplicate480-pair set')
    arrays=dict(target=np.concatenate(targets),valid=np.concatenate(valid),boundary=np.concatenate(boundary),raw=np.concatenate(raw_prediction))
    if int(arrays['valid'].sum())!=datalock['valid_count'] or int(arrays['boundary'].sum())!=datalock['boundary_count']:raise ValueError('locked population count mismatch')
    return lock,datalock,arrays,ids,sensor_report


def audit(confirmation,raw_root,out,authorized=False):
    if not authorized:raise ValueError('execute only after root test-complete notification with explicit flag')
    if not (confirmation/'completion.json').is_file():raise ValueError('confirmation not complete; do not read test labels')
    if out.exists():raise ValueError('refusing overwrite audit output')
    started=time.time();lock,datalock,arrays,ids,sensors=locked_inputs(confirmation,raw_root)
    out.mkdir(parents=True)
    report=dict(scope='independent NumPy recomputation of all30 model+1raw archives and native query PNG center labels; no model replay',
        lock_sha256=sha(confirmation/'lock.json'),test_data_lock_sha256=sha(confirmation/'test_data_lock.json'),
        audit_sha256=sha(__file__),metric_helper_sha256=sha(ROOT/'experiments/round7_depth_audit.py'),
        pair_count=len(ids),valid_count=int(arrays['valid'].sum()),boundary_count=int(arrays['boundary'].sum()),
        raw_sensor_sampling_bitwise_equal=True,sensor_sources=sensors,archives=[])
    expected=[(role,seed,confirmation/role/str(seed)) for role in ROLES for seed in SEEDS]+[('raw',None,confirmation/'baseline_raw')]
    for role,seed,directory in expected:
        stored=json.loads((directory/'test.json').read_text());metadata=stored['metadata']
        if metadata['role']!=role or metadata['lock_sha256']!=report['lock_sha256']:raise ValueError('archive role/lock mismatch')
        if sha(directory/'predictions.npz')!=metadata['prediction_sha256']:raise ValueError('prediction archive hash mismatch')
        if role!='raw':
            entry=lock['roles'][role]['seeds'][str(seed)]
            if metadata['seed']!=seed or metadata['checkpoint_sha256']!=entry['checkpoint_sha256']:raise ValueError('checkpoint identity mismatch')
            if metadata['test_data_lock_sha256']!=report['test_data_lock_sha256']:raise ValueError('different shared test labels')
        with np.load(directory/'predictions.npz',allow_pickle=False) as values:pred=values['depth'].copy()
        if pred.shape!=(480,432):raise ValueError('prediction shape mismatch')
        if role=='raw' and not np.array_equal(pred,arrays['raw']):raise ValueError('raw baseline differs from frozen RGB predictor')
        rows=numpy_rows(pred,arrays['target'],arrays['valid'],arrays['boundary'],ids);summary=numpy_macro(rows)
        comparison=compare_tree(dict(rows=rows,summary=summary),dict(rows=stored['rows'],summary=stored['summary']))
        report['archives'].append(dict(role=role,seed=seed,prediction_sha256=metadata['prediction_sha256'],comparison=comparison,summary=summary))
        (out/f'{role}_{seed if seed is not None else "baseline"}_numpy_rows.json').write_text(json.dumps(dict(rows=rows,summary=summary),indent=2))
    report.update(elapsed_seconds=time.time()-started,archive_count=len(report['archives']),
        pair_metric_rows=31*480,pass_=all(not r['comparison']['mismatches'] for r in report['archives']))
    if report['lock_sha256']!=sha(confirmation/'lock.json') or report['test_data_lock_sha256']!=sha(confirmation/'test_data_lock.json'):raise ValueError('locks changed during audit')
    for name,digest in lock['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source changed during audit')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print('AUDIT31',report['pass_'],'max metric error',max(r['comparison']['max_absolute_error'] for r in report['archives']),flush=True)
    if not report['pass_']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmation',type=Path,default=ROOT/'results/round7_depth_confirmation')
    p.add_argument('--raw-root',type=Path,default=ROOT/'data/round7_sun3d')
    p.add_argument('--out',type=Path,default=ROOT/'results/round7_depth_test_audit')
    p.add_argument('--after-root-test-complete',action='store_true')
    a=p.parse_args();audit(a.confirmation,a.raw_root,a.out,a.after_root_test_complete)
