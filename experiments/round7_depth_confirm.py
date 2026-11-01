"""Freeze development, replay validation, then permit one fresh depth test."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from models.round7_depth_attention import CONFIGS, DepthAttention
from experiments.round7_depth_data import ROOT, sha
from experiments.round7_depth_train import SOURCES, load_data, evaluate, metric_rows, macro

SEEDS=[11,22,33,44,55]
EXTRA_SOURCES=('ROUND7_PROTOCOL.md','experiments/round7_depth_confirm.py',
    'experiments/round7_depth_analysis.py','experiments/round7_depth_audit.py',
    'experiments/round7_depth_test_audit.py',
    'experiments/round7_acquire.py','experiments/round5_cache.py',
    'experiments/round6_cache.py','experiments/real_frozen_features.py')

def verify_sources(mapping):
    for path,digest in mapping.items():
        if sha(ROOT/path)!=digest:raise ValueError('locked source changed '+path)

def canonical_sources(rows):
    return [dict(row,path=str(Path(row['path']).resolve())) for row in rows]

def get_model(path,role,device):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['role']!=role:raise ValueError('wrong checkpoint role')
    model=DepthAttention(role).to(device);model.load_state_dict(state['state_dict'],strict=True)
    return model

def replay(role,entry,data,ids,device):
    if sha(entry['checkpoint_path'])!=entry['checkpoint_sha256']:raise ValueError('checkpoint changed')
    if sha(entry['validation_path'])!=entry['validation_sha256']:raise ValueError('validation changed')
    model=get_model(entry['checkpoint_path'],role,device)
    result,pred=evaluate(model,data,ids)
    previous=json.loads(Path(entry['validation_path']).read_text())
    identity=lambda r:(r['scene'],r['q_index'],r['k_index'],r['pair_id'])
    if list(map(identity,result['rows']))!=list(map(identity,previous['rows'])):
        raise ValueError('validation identities changed')
    for metric,value in result['summary']['metrics'].items():
        if abs(value-previous['summary']['metrics'][metric])>1e-7:
            raise ValueError('validation replay mismatch '+metric)
    run=json.loads(Path(entry['run_path']).read_text())
    if sha(entry['run_path'])!=entry['run_sha256']:raise ValueError('training run metadata changed')
    if abs(result['summary']['metrics']['AbsRel']-run['best_validation_AbsRel'])>1e-7:
        raise ValueError('validation replay does not match selected run score')
    return model

def freeze(a):
    if a.out.exists():raise ValueError('refuse overwrite confirmation directory')
    val,vi,vs=load_data(a.roots,'val',a.device)
    train,ti,ts=load_data(a.roots,'train','cpu')
    if len(ti)!=576 or len(vi)!=128:raise ValueError('wrong development pair counts')
    del train
    selection=json.loads(a.selection.read_text());scenes=selection['scenes']
    if len(scenes)!=5 or any(r['split']!='test' or len(r['pairs'])!=96 for r in scenes):
        raise ValueError('fresh confirmation must contain exactly five 96-pair scans')
    roles={};means={};sequence={};started=time.time()
    for role in CONFIGS:
        entries={};scores=[]
        for seed in SEEDS:
            runpath=a.runs/role/str(seed)/'run.json';run=json.loads(runpath.read_text())
            if run['role']!=role or run['seed']!=seed or run['steps']!=1000:raise ValueError('wrong development budget')
            verify_sources(run['source_hashes'])
            if ({split:canonical_sources(rows) for split,rows in run['data_sources'].items()}
                    !=dict(train=canonical_sources(ts),val=canonical_sources(vs))):
                raise ValueError('development data changed')
            selected=min(run['history'],key=lambda r:r['validation']['metrics']['AbsRel'])
            if selected['step']!=run['best_step'] or selected['validation']['metrics']['AbsRel']!=run['best_validation_AbsRel']:
                raise ValueError('checkpoint selection rule mismatch')
            if seed in sequence and sequence[seed]!=run['batch_sequence_sha256']:raise ValueError('unpaired batches')
            sequence[seed]=run['batch_sequence_sha256']
            entry=dict(checkpoint_path=str((runpath.parent/'best.pt').resolve()),
                checkpoint_sha256=run['checkpoint_sha256'],
                validation_path=str((runpath.parent/'validation.json').resolve()),
                validation_sha256=run['validation_sha256'],run_path=str(runpath.resolve()),
                run_sha256=sha(runpath),best_step=run['best_step'],parameter_count=run['parameter_count'])
            replay(role,entry,val,vi,a.device)
            entries[str(seed)]=entry;scores.append(run['best_validation_AbsRel'])
        roles[role]=dict(seeds=entries);means[role]=float(np.mean(scores))
    baseline=min(('point','point_pck','fixed'),key=lambda r:means[r])
    all_sources={p:sha(ROOT/p) for p in dict.fromkeys(SOURCES+EXTRA_SOURCES)}
    a.out.mkdir(parents=True)
    for path in all_sources:
        dest=a.out/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest)
    rawrows=metric_rows(val['q_depth'],val,vi)
    (a.out/'raw_validation.json').write_text(json.dumps(dict(rows=rawrows,summary=macro(rawrows)),indent=2))
    lock=dict(schema_version=1,seeds=SEEDS,roles=roles,baseline_role=baseline,
        validation_means=means,selection_path=str(a.selection.resolve()),selection_sha256=sha(a.selection),
        expected_scenes=[r['scene'] for r in scenes],
        expected_pairs={r['scene']:r['pairs'] for r in scenes},
        source_hashes=all_sources,development_sources=dict(train=ts,val=vs),
        test_export_authorized=True,test_inputs='frozen RGB predictions/features + shared K/C2W; no sensor inputs',
        bootstrap=dict(draws=20000,seed=20261030,unit='paired crossed seed and named scan/place'),
        protocol_sha256=sha(ROOT/'ROUND7_PROTOCOL.md'),freeze_seconds=time.time()-started)
    (a.out/'lock.json').write_text(json.dumps(lock,indent=2))
    print('LOCKED',sha(a.out/'lock.json'),'baseline',baseline,'validation',means,flush=True)

def test(a):
    lockpath=a.out/'lock.json';lock=json.loads(lockpath.read_text());verify_sources(lock['source_hashes'])
    if (set(lock['roles'])!=set(CONFIGS) or lock['seeds']!=SEEDS
            or any(set(r['seeds'])!={str(s) for s in SEEDS} for r in lock['roles'].values())):
        raise ValueError('locked role/seed matrix is incomplete')
    if sha(lock['selection_path'])!=lock['selection_sha256']:raise ValueError('selection changed')
    if (a.out/'completion.json').exists() or any((a.out/r).exists() for r in CONFIGS):
        raise ValueError('refuse repeat/overwrite fresh test')
    val,vi,vs=load_data(a.roots,'val',a.device)
    if vs!=lock['development_sources']['val']:raise ValueError('validation sources changed')
    # Replaying all locked validation checkpoints precedes any fresh test load.
    for role,record in lock['roles'].items():
        for seed,entry in record['seeds'].items():replay(role,entry,val,vi,a.device)
    export=json.loads((a.test_root/'completion.json').read_text())
    if export['lock_sha256']!=sha(lockpath):raise ValueError('test export not authorized by this lock')
    exported=export['manifests']
    if len(exported)!=5 or {m['scene'] for m in exported}!=set(lock['expected_scenes']):
        raise ValueError('incomplete fresh export manifests')
    for m in exported:
        actual_manifest=a.test_root/'test'/m['scene'].replace('/','__')/'manifest.json'
        if json.loads(actual_manifest.read_text())!=m:raise ValueError('export completion/manifest mismatch')
        if (m['selection_sha256']!=lock['selection_sha256']
                or m['exporter_sha256']!=lock['source_hashes']['experiments/round7_depth_data.py']
                or m['pairs']!=lock['expected_pairs'][m['scene']] or m['split']!='test'):
            raise ValueError('unauthorized fresh export source/selection/population')
    data,ids,sources=load_data([a.test_root],'test',a.device)
    actual={(r['scene'],r['q_index'],r['k_index']) for r in ids}
    expected={(scene,q,k) for scene,pairs in lock['expected_pairs'].items() for q,k in pairs}
    if actual!=expected or len(ids)!=480:raise ValueError('incomplete or altered test identities')
    data_lock=dict(sources=sources,confirmation_lock_sha256=sha(lockpath),pair_count=len(ids),
        total_positions=len(ids)*432,valid_count=int(data['valid'].sum()),
        boundary_count=int(data['boundary'].sum()))
    (a.out/'test_data_lock.json').write_text(json.dumps(data_lock,indent=2));started=time.time()
    for role,record in lock['roles'].items():
        for seed,entry in record['seeds'].items():
            model=get_model(entry['checkpoint_path'],role,a.device);result,pred=evaluate(model,data,ids)
            dest=a.out/role/seed;dest.mkdir(parents=True)
            np.savez_compressed(dest/'predictions.npz',depth=pred.detach().cpu().numpy())
            result['metadata']=dict(role=role,seed=int(seed),checkpoint_sha256=entry['checkpoint_sha256'],
                lock_sha256=sha(lockpath),test_data_lock_sha256=sha(a.out/'test_data_lock.json'),
                prediction_sha256=sha(dest/'predictions.npz'))
            if role=='full':
                result['interventions']={}
                for mode in ('zero_v','reverse_v'):
                    audit,changed=evaluate(model,data,ids,mode)
                    result['interventions'][mode]=dict(summary=audit['summary'],
                        mean_absolute_output_change_m=float((changed-pred).abs().mean()))
            (dest/'test.json').write_text(json.dumps(result,indent=2))
            print('TEST',role,seed,result['summary']['metrics']['AbsRel'],flush=True)
    dest=a.out/'baseline_raw';dest.mkdir()
    rows=metric_rows(data['q_depth'],data,ids)
    np.savez_compressed(dest/'predictions.npz',depth=data['q_depth'].cpu().numpy())
    (dest/'test.json').write_text(json.dumps(dict(rows=rows,summary=macro(rows),metadata=dict(
        role='raw',lock_sha256=sha(lockpath),test_data_lock_sha256=sha(a.out/'test_data_lock.json'),
        prediction_sha256=sha(dest/'predictions.npz'))),indent=2))
    (a.out/'completion.json').write_text(json.dumps(dict(lock_sha256=sha(lockpath),
        test_data_lock_sha256=sha(a.out/'test_data_lock.json'),roles=list(CONFIGS),seeds=SEEDS,
        test_seconds=time.time()-started),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','test'])
    p.add_argument('--runs',type=Path,default=ROOT/'results/round7_depth_development')
    p.add_argument('--roots',nargs='+',type=Path,default=[ROOT/'results/round7_depth_old',ROOT/'results/round7_depth_bm4'])
    p.add_argument('--out',type=Path,default=ROOT/'results/round7_depth_confirmation')
    p.add_argument('--selection',type=Path,default=ROOT/'data/round7_sun3d/selection.json')
    p.add_argument('--test-root',type=Path,default=ROOT/'results/round7_depth_test')
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    (freeze if a.action=='freeze' else test)(a)

if __name__=='__main__':main()
