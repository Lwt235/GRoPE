"""Freeze deterministic calibration, replay it, then evaluate one sealed test set."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.round6_train import load_split,evaluate,macro,GRIDS,sha
from models.gaussian_quadratic_pe import GaussianQuadraticMatcher

ROOT=Path(__file__).resolve().parents[1]
SOURCE_PATHS=(
    'models/gaussian_quadratic_pe.py','models/gaussian_measure_pe.py',
    'models/round6_measure_attention.py','models/round5_cross_attention.py',
    'experiments/round6_confirm.py','experiments/round6_analysis.py',
    'experiments/round6_train.py','experiments/round6_region_data.py',
    'experiments/round6_geometry_dev.py','experiments/round5_rgbd_data.py',
    'experiments/round6_cache.py','experiments/round5_cache.py',
    'experiments/real_frozen_features.py')


def hashes():return {p:sha(ROOT/p) for p in SOURCE_PATHS}


def canonical_sources(sources):
    return [dict(row,path=str(Path(row['path']).resolve())) for row in sources]


def verify_data(groups,ids,selection=None,split=None):
    observed=None
    for grid in GRIDS:
        identities=ids[grid]
        key={(r['scene'],r['q_index'],r['k_index']) for r in identities}
        if len(key)!=len(identities):raise ValueError('duplicate data identity')
        if observed is not None and key!=observed:raise ValueError('unequal grid pair sets')
        observed=key
        if any(v.shape[0]!=len(identities) for v in groups[grid].values()):
            raise ValueError('NPZ rows differ from manifest pairs')
        qs,ks=map(int,grid.replace('q','').replace('k','').split('_'))
        nq,nk=(144//qs)*(192//qs),(144//ks)*(192//ks)
        d=groups[grid]
        if d['soft_labels'].shape[1:]!=(nq,nk) or d['q'].shape[1]!=nq or d['k'].shape[1]!=nk:
            raise ValueError('grid/data shape mismatch')
        for side,size in [('q',qs),('k',ks)]:
            expected=torch.eye(2,device=d[side+'cov'].device)*(size**2/(12*64))
            torch.testing.assert_close(d[side+'cov'],expected.expand_as(d[side+'cov']),atol=0,rtol=0)
    if selection is not None:
        expected={(s['scene'],q,k) for s in selection['scenes'] if s['split']==split for q,k in s['pairs']}
        if observed!=expected:raise ValueError('incomplete or extra selected pair set')
    return observed


def configuration(record):
    return dict(method=record['method'],bandwidth=record['width'],temperature=record['temperature'])


def make_model(config,device):return GaussianQuadraticMatcher(**config).to(device)


def metric_summary(rows):
    return dict(kl=macro(rows),pck8=macro(rows,'pck8_count','n_point',['q8_k8']),
        kl_by_grid={g:macro(rows,grids=[g]) for g in GRIDS})


def freeze(args):
    if args.lock.exists():raise ValueError('refusing to overwrite confirmation lock')
    args.out.mkdir(parents=True,exist_ok=True)
    records_path=ROOT/'results/round6_geometry_two_val/summary.json'
    search=json.loads(records_path.read_text())
    by_method={m:[r for r in search['records'] if r['method']==m] for m in ['point','fixed','isotropic','full']}
    expected_grid={(h,t) for h in [.0625,.125,.25,.5,1.,2.,4.] for t in [0.,2.5,5.,10.,20.]}
    if any(len(v)!=35 or {(r['width'],r['temperature']) for r in v}!=expected_grid for v in by_method.values()):
        raise ValueError('unequal or unexpected calibration budget')
    selected={m:min(rs,key=lambda r:r['kl']) for m,rs in by_method.items()}
    selected['point_pck']=max(by_method['point'],key=lambda r:r['pck8'])
    baseline=min(['point','fixed'],key=lambda m:selected[m]['kl'])
    val,ids,sources=load_split(args.validation,'val',args.device)
    if canonical_sources(sources)!=canonical_sources(search['input_sources']):
        raise ValueError('calibration validation sources changed')
    valid=verify_data(val,ids)
    if len(valid)!=128:raise ValueError('expected two validation scans, 64 pairs each')
    selection=json.loads(args.selection.read_text())
    test_scenes=sorted(s['scene'] for s in selection['scenes'] if s['split']=='test')
    if len(test_scenes)!=5 or sum(len(s['pairs']) for s in selection['scenes'] if s['split']=='test')!=480:
        raise ValueError('unexpected confirmation sample size')
    old=json.loads((ROOT/'data/round5_sun3d/selection.json').read_text())
    forbidden={s['scene'].split('/')[0] for s in old['scenes']}|{r[0].split('/')[0] for r in valid}
    if any(s.split('/')[0] in forbidden or s.startswith('brown_bm_') for s in test_scenes):
        raise ValueError('test/previous/development place collision')
    frozen={p:sha(ROOT/p) for p in ['ROUND6_PROTOCOL.md',
        'data/round6_sun3d/selection_v2.json','data/round6_sun3d/rejection_bm1.json',
        'results/round6_geometry_two_val/summary.json']}
    sourcehash=hashes();roles={};started=time.time()
    for role,record in selected.items():
        config=configuration(record)
        rows=evaluate(make_model(config,args.device),val,ids,args.batch_size)
        metrics=metric_summary(rows)
        if abs(metrics['kl']-record['kl'])>1e-5 or abs(metrics['pck8']-record['pck8'])>1e-7:
            raise ValueError(f'calibration replay mismatch {role}: {metrics} versus {record}')
        for g in GRIDS:
            if abs(metrics['kl_by_grid'][g]-record['kl_by_grid'][g])>1e-5:raise ValueError('grid replay mismatch')
        directory=args.out/role;directory.mkdir(exist_ok=True)
        (directory/'val.json').write_text(json.dumps(dict(rows=rows,config=config,metrics=metrics,
            input_sources=sources,sourcehash=sourcehash),indent=2))
        roles[role]=dict(config=config,validation_metrics=metrics,
            validation_sha256=sha(directory/'val.json'))
        print('VALIDATION_REPLAY',role,metrics,flush=True)
    if hashes()!=sourcehash:raise ValueError('source changed during freeze')
    lock=dict(version=1,created_unix=time.time(),deterministic=True,
        roles=roles,method_configs={role:row['config'] for role,row in roles.items()},
        area_baseline=baseline,point_pck='point_pck',expected_scenes=test_scenes,
        expected_pairs_per_scene=96,expected_grids=list(GRIDS),validation_sources=sources,
        validation_roots=[str(x) for x in args.validation],sourcehash=sourcehash,frozen_files=frozen,
        analysis_source_sha256=sha(ROOT/'experiments/round6_analysis.py'),
        protocol_sha256=sha(ROOT/'ROUND6_PROTOCOL.md'),selection_path=str(args.selection),
        selection_sha256=sha(args.selection),exporter_sha256=sha(ROOT/'experiments/round6_region_data.py'),
        test_export_authorized=True,bootstrap=dict(unit='paired scan/place',replicates=20000,seed=20261030),
        seconds=time.time()-started)
    source_dir=args.out/'source'
    for path in SOURCE_PATHS+('ROUND6_PROTOCOL.md',):
        dest=source_dir/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/path).read_bytes())
    args.lock.write_text(json.dumps(lock,indent=2))
    print('FROZEN',str(args.lock),sha(args.lock),flush=True)


def verify_lock(args):
    lock=json.loads(args.lock.read_text())
    if hashes()!=lock['sourcehash']:raise ValueError('source changed after confirmation freeze')
    for path,digest in lock['frozen_files'].items():
        if sha(ROOT/path)!=digest:raise ValueError('frozen evidence changed: '+path)
    if sha(lock['selection_path'])!=lock['selection_sha256']:raise ValueError('selection changed')
    return lock


def test(args):
    lock=verify_lock(args);started=time.time()
    val,ids,sources=load_split(lock['validation_roots'],'val',args.device)
    verify_data(val,ids)
    if sources!=lock['validation_sources']:raise ValueError('validation sources changed')
    # Replay every selected role before opening the held-out input/label arrays.
    for role,frozen in lock['roles'].items():
        if sha(args.out/role/'val.json')!=frozen['validation_sha256']:raise ValueError('saved validation changed')
        rows=evaluate(make_model(frozen['config'],args.device),val,ids,args.batch_size)
        actual=metric_summary(rows);expected=frozen['validation_metrics']
        if abs(actual['kl']-expected['kl'])>1e-6 or abs(actual['pck8']-expected['pck8'])>1e-7:
            raise ValueError('pre-test validation replay failed')
        print('PRETEST_VALIDATION_OK',role,flush=True)
    del val
    export_path=args.test_data/'frozen_export_protocol.json'
    export=json.loads(export_path.read_text())
    if (export.get('selection_sha256')!=lock['selection_sha256']
            or export.get('source_sha256')!=lock['exporter_sha256']
            or export.get('test_protocol_lock',{}).get('sha256')!=sha(args.lock)):
        raise ValueError('test export was not authorized by this exact frozen protocol')
    data,tids,tsources=load_split(args.test_data,'test',args.device)
    selection=json.loads(Path(lock['selection_path']).read_text())
    identities=verify_data(data,tids,selection,'test')
    observed=sorted({r[0] for r in identities})
    if observed!=lock['expected_scenes']:raise ValueError('test scene set mismatch')
    data_lock=args.out/'test_data_lock.json'
    if data_lock.exists():raise ValueError('test was already opened; refusing silent rerun')
    data_lock.write_text(json.dumps(dict(confirmation_lock_sha256=sha(args.lock),sources=tsources,
        identities=sorted(identities),export_protocol_sha256=sha(args.test_data/'frozen_export_protocol.json'),
        created_unix=time.time()),indent=2))
    for role,frozen in lock['roles'].items():
        output=args.out/role/'test.json'
        if output.exists():raise ValueError('refusing test result overwrite')
        model=make_model(frozen['config'],args.device)
        rows=evaluate(model,data,tids,args.batch_size,args.out/role/'predictions/test')
        for row in rows:row['method']=role
        output.write_text(json.dumps(dict(rows=rows,config=frozen['config'],input_sources=tsources,
            sourcehash=lock['sourcehash'],lock_sha256=sha(args.lock),metrics=metric_summary(rows)),indent=2))
        print('TEST_COMPLETE',role,metric_summary(rows),flush=True)
    if hashes()!=lock['sourcehash']:raise ValueError('source changed during test')
    (args.out/'completion.json').write_text(json.dumps(dict(seconds=time.time()-started,
        lock_sha256=sha(args.lock),test_data_lock_sha256=sha(data_lock),
        roles={role:sha(args.out/role/'test.json') for role in lock['roles']},
        sourcehash=lock['sourcehash'],torch=torch.__version__,numpy=np.__version__,device=args.device,
        max_gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None),indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['freeze','test'])
    p.add_argument('--out',type=Path,default=ROOT/'results/round6_confirmation')
    p.add_argument('--lock',type=Path,default=ROOT/'results/round6_confirmation/lock.json')
    p.add_argument('--selection',type=Path,default=ROOT/'data/round6_sun3d/selection_v2.json')
    p.add_argument('--validation',type=Path,nargs='+',default=[ROOT/'results/round6_region_inputs',ROOT/'results/round6_new_region_inputs'])
    p.add_argument('--test-data',type=Path,default=ROOT/'results/round6_test_region_inputs')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    if args.action=='freeze':freeze(args)
    else:test(args)


if __name__=='__main__':main()
