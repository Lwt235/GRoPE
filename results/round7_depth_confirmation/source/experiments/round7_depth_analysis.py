"""Model-independent locked analysis of the Round7 depth confirmation.

No model/cache imports. Queries are weighted within each scan; five scans and
five training seeds are weighted equally. The strong baseline is locked using
validation, never selected from test scores.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11,22,33,44,55)
ROLES = ('full','fixed','isotropic','point','point_pck','singleview')
BASELINES = ('point','point_pck','fixed')
DRAWS = 20000
RNG_SEED = 20261030
ADDITIVE = ('count','boundary_count','absrel_sum','abs_sum','sq_sum','log_sq_sum',
            'delta1_count','boundary_absrel_sum','boundary_abs_sum')
METRICS = ('AbsRel','MAE','RMSE','log_RMSE','delta1','boundary_AbsRel','boundary_MAE')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file(path):
    path=Path(path);raw=path.read_bytes();obj=json.loads(raw)
    if not isinstance(obj,dict) or not isinstance(obj.get('rows'),list) or not isinstance(obj.get('summary'),dict):
        raise ValueError('expected trainer rows+summary wrapper: '+str(path))
    return obj,dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def identity(row):
    return row['scene'],row['q_index'],row['k_index']


def validate_row_block(rows,expected_scenes,expected_pairs=None,pairs_per_scene=96):
    if not isinstance(rows,list) or not rows:raise ValueError('empty row block')
    normalized=[];seen=set()
    for original in rows:
        if not isinstance(original,dict) or set(('scene','q_index','k_index','pair_id')+ADDITIVE)-set(original):
            raise ValueError('missing identity/additive fields')
        r=dict(original)
        if not isinstance(r['scene'],str) or r['scene'] not in expected_scenes:
            raise ValueError('unexpected scene')
        for f in ('q_index','k_index','count','boundary_count','delta1_count'):
            x=r[f]
            if isinstance(x,bool) or not math.isfinite(float(x)) or int(x)!=x or x<0:
                raise ValueError('nonnegative integer required: '+f)
            r[f]=int(x)
        if r['pair_id']!=f"{r['q_index']}:{r['k_index']}":raise ValueError('pair_id disagrees with indices')
        if not 0<=r['boundary_count']<=r['count']<=432 or r['delta1_count']>r['count']:
            raise ValueError('invalid count or boundary population')
        for f in ADDITIVE:
            if not math.isfinite(float(r[f])) or r[f]<0:raise ValueError('nonfinite/negative additive: '+f)
            if f not in ('count','boundary_count','delta1_count'):r[f]=float(r[f])
        if r['count']==0 and any(r[f]!=0 for f in ADDITIVE):
            raise ValueError('zero-valid pair has nonzero additive sums')
        if r['boundary_count']==0 and any(r[f]!=0 for f in ('boundary_absrel_sum','boundary_abs_sum')):
            raise ValueError('zero-boundary pair has nonzero boundary sums')
        if r['boundary_absrel_sum']>r['absrel_sum']+1e-8 or r['boundary_abs_sum']>r['abs_sum']+1e-8:
            raise ValueError('boundary errors exceed all-valid errors')
        key=identity(r)
        if key in seen:raise ValueError('duplicate pair identity')
        seen.add(key);normalized.append(r)
    if set(r['scene'] for r in normalized)!=set(expected_scenes):raise ValueError('missing scene')
    if expected_pairs is not None and seen!=set(expected_pairs):raise ValueError('pair identity set differs from locked population')
    for scene in expected_scenes:
        if sum(r['scene']==scene for r in normalized)!=pairs_per_scene:
            raise ValueError('incomplete pair count for '+scene)
    return sorted(normalized,key=identity)


def aggregate_block(rows,scenes):
    result={};coverage={}
    for scene in scenes:
        chosen=[r for r in rows if r['scene']==scene]
        sums={f:math.fsum(r[f] for r in chosen) for f in ADDITIVE}
        n,b=sums['count'],sums['boundary_count']
        if n<=0:raise ValueError('scene has no valid query depth')
        result[scene]=dict(AbsRel=sums['absrel_sum']/n,MAE=sums['abs_sum']/n,
            RMSE=math.sqrt(sums['sq_sum']/n),log_RMSE=math.sqrt(sums['log_sq_sum']/n),
            delta1=sums['delta1_count']/n,boundary_AbsRel=sums['boundary_absrel_sum']/b if b else None,
            boundary_MAE=sums['boundary_abs_sum']/b if b else None)
        coverage[scene]=dict(**sums,pairs=len(chosen),zero_valid_pairs=sum(r['count']==0 for r in chosen),
            query_frames=len({r['q_index'] for r in chosen}))
    macro={m:float(np.mean([result[s][m] for s in scenes if result[s][m] is not None]))
           if any(result[s][m] is not None for s in scenes) else None for m in METRICS}
    return dict(metrics=macro,by_scene=result,coverage=coverage,total_count=sum(r['count'] for r in rows),
                total_positions=len(rows)*432)


def check_reported_summary(reported,actual):
    def same(a,b,where):
        if a is None or b is None:
            if a is not None or b is not None:raise ValueError('reported summary mismatch: '+where)
        elif not math.isfinite(float(a)) or not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('reported summary mismatch: '+where)
    for m in METRICS:same(reported['metrics'][m],actual['metrics'][m],m)
    if set(reported['by_scene'])!=set(actual['by_scene']):raise ValueError('reported scene set mismatch')
    for s in actual['by_scene']:
        for m in METRICS:same(reported['by_scene'][s][m],actual['by_scene'][s][m],s+'/'+m)
    for f in ('total_count','total_positions'):same(reported[f],actual[f],f)


def draw_indices(seed_count,scene_count,draws=DRAWS):
    if draws<1:raise ValueError('positive bootstrap draw count required')
    rng=np.random.default_rng(RNG_SEED)
    return rng.integers(seed_count,size=(draws,seed_count)),rng.integers(scene_count,size=(draws,scene_count))


def paired_statistics(candidate,control,seeds,scenes,indices):
    a,b=np.asarray(candidate,float),np.asarray(control,float)
    if a.shape!=(len(seeds),len(scenes)) or b.shape!=a.shape:raise ValueError('seed/scene matrix mismatch')
    available=np.isfinite(a)&np.isfinite(b)
    if not np.array_equal(np.isfinite(a),np.isfinite(b)):raise ValueError('metric availability differs')
    if not available.all():
        return dict(available=False,reason='one or more scans have no boundary population',
            candidate_mean=float(np.nanmean(a)) if available.any() else None,
            control_mean=float(np.nanmean(b)) if available.any() else None,
            paired_finite_cells=int(available.sum()),ci95_difference=None,ci95_relative_difference=None)
    sd,pd=indices
    ab=a[sd[:,:,None],pd[:,None,:]].mean((1,2))
    bb=b[sd[:,:,None],pd[:,None,:]].mean((1,2))
    delta=a-b;observed=float(delta.mean())
    ratios=(ab-bb)/bb if (bb>0).all() else None
    seed_delta=delta.mean(-1);scene_delta=delta.mean(0)
    loo={s:dict(difference=float(np.delete(delta,i,axis=1).mean()),
        relative_difference=float((np.delete(a,i,axis=1).mean()-np.delete(b,i,axis=1).mean())/np.delete(b,i,axis=1).mean())
            if np.delete(b,i,axis=1).mean()>0 else None,remaining_scans=len(scenes)-1,descriptive_only=True)
        for i,s in enumerate(scenes)}
    return dict(available=True,candidate_mean=float(a.mean()),control_mean=float(b.mean()),
        difference=observed,relative_difference=observed/float(b.mean()) if b.mean()>0 else None,
        ci95_difference=np.quantile(ab-bb,[.025,.975]).tolist(),
        ci95_relative_difference=np.quantile(ratios,[.025,.975]).tolist() if ratios is not None else None,
        improved_seed_count=int((seed_delta<0).sum()),
        per_seed_difference={str(s):float(v) for s,v in zip(seeds,seed_delta)},
        per_scene_difference={s:float(v) for s,v in zip(scenes,scene_delta)},leave_one_scan_out=loo)


def analyze_rows(blocks,baseline_role,expected_scenes,*,raw_rows=None,expected_pairs=None,
                 seeds=SEEDS,pairs_per_scene=96,draws=DRAWS):
    seeds=tuple(sorted(seeds));scenes=tuple(sorted(expected_scenes))
    if seeds!=SEEDS or len(scenes)!=5 or len(set(scenes))!=5:raise ValueError('five fixed seeds and five unique scans required')
    if baseline_role not in BASELINES:raise ValueError('invalid locked strong baseline')
    if set(blocks)!=set(ROLES):raise ValueError('incomplete role matrix')
    arrays={r:{m:np.empty((5,5),float) for m in METRICS} for r in ROLES}
    aggregates={};reference_population=None;normalized_blocks={}
    for role in ROLES:
        if set(int(s) for s in blocks[role])!=set(SEEDS) or len(blocks[role])!=5:
            raise ValueError('incomplete seed matrix: '+role)
        aggregates[role]={};normalized_blocks[role]={}
        for si,seed in enumerate(seeds):
            source=blocks[role].get(seed,blocks[role].get(str(seed)))
            rows=validate_row_block(source,scenes,expected_pairs,pairs_per_scene)
            population=[identity(r)+(r['count'],r['boundary_count']) for r in rows]
            if reference_population is None:reference_population=population
            elif population!=reference_population:raise ValueError('pair/valid/boundary populations differ across roles/seeds')
            aggregate=aggregate_block(rows,scenes);aggregates[role][str(seed)]=aggregate;normalized_blocks[role][seed]=rows
            for pi,scene in enumerate(scenes):
                for metric in METRICS:
                    value=aggregate['by_scene'][scene][metric]
                    arrays[role][metric][si,pi]=np.nan if value is None else value
    indices=draw_indices(5,5,draws)
    comparisons={role:{metric:paired_statistics(arrays['full'][metric],arrays[role][metric],seeds,scenes,indices)
        for metric in METRICS} for role in ROLES if role!='full'}
    primary=comparisons[baseline_role]['AbsRel'];single=comparisons['singleview']['AbsRel'];rmse=comparisons[baseline_role]['RMSE']
    criteria=dict(relative_AbsRel_improvement_at_least_2_percent=primary['relative_difference'] is not None and primary['relative_difference']<=-.02,
        AbsRel_difference_CI_upper_below_zero=primary['ci95_difference'][1]<0,
        at_least_4_of_5_seed_macros_improve=primary['improved_seed_count']>=4,
        better_than_singleview_AbsRel_CI_upper_below_zero=single['ci95_difference'][1]<0,
        RMSE_relative_difference_CI_upper_below_2_percent=rmse['ci95_relative_difference'] is not None and rmse['ci95_relative_difference'][1]<.02)
    raw=None
    if raw_rows is not None:
        raw_normalized=validate_row_block(raw_rows,scenes,expected_pairs,pairs_per_scene)
        if [identity(r)+(r['count'],r['boundary_count']) for r in raw_normalized]!=reference_population:
            raise ValueError('raw baseline population differs')
        raw=dict(summary=aggregate_block(raw_normalized,scenes),descriptive_only=True,
            independently_trained_seed_replicates=0,note='one frozen deterministic baseline, not five seed replicates')
    model_summary={r:{m:dict(mean=float(np.nanmean(arrays[r][m])) if np.isfinite(arrays[r][m]).any() else None,
        per_seed={str(s):(float(np.nanmean(arrays[r][m][i]))) if np.isfinite(arrays[r][m][i]).any() else None for i,s in enumerate(seeds)},
        per_scene={s:float(np.nanmean(arrays[r][m][:,i])) if np.isfinite(arrays[r][m][:,i]).any() else None for i,s in enumerate(scenes)})
        for m in METRICS} for r in ROLES}
    fingerprint=hashlib.sha256(json.dumps(reference_population,separators=(',',':')).encode()).hexdigest()
    return dict(baseline_role=baseline_role,baseline_selected_from_test=False,roles=ROLES,seeds=seeds,scenes=scenes,
        models=model_summary,per_role_seed_scan=aggregates,comparisons=comparisons,raw_baseline=raw,
        primary_gate=dict(passed=all(criteria.values()),criteria=criteria,AbsRel=primary,RMSE=rmse,singleview_AbsRel=single),
        bootstrap=dict(draws=draws,seed=RNG_SEED,method='paired crossed resampling of five training seeds and five scans; same indices for every metric/role',
            interval='central 95%',estimand='ratio-of-sums within scan, equal scan and seed weight; RMSE square root within scan before macro'),
        population=dict(pairs_per_role_seed=len(reference_population),pairs_per_scan=pairs_per_scene,
            total_valid_positions=sum(x[-2] for x in reference_population),boundary_positions=sum(x[-1] for x in reference_population),
            pair_count_boundary_fingerprint_sha256=fingerprint,
            exact_pixel_masks_not_inferred_from_counts=True),
        limitations=['five scan/place groups are not five independent institutions',
            'sensor-valid coarse query centers, including cross-view occlusion/out-of-frame cases',
            'supplied RGB-D-reconstructed poses/intrinsics; no GT scale alignment',
            'all negative comparisons retained; isotropic attribution is exploratory',
            'equal population counts do not alone verify pixel-mask identity; shared label archive auditing is separate'])


def resolved(path):
    path=Path(path);return path if path.is_absolute() else ROOT/path


def validate_lock(path):
    raw=Path(path).read_bytes();lock=json.loads(raw)
    if lock.get('test_export_authorized') is not True or tuple(lock.get('seeds',()))!=SEEDS:
        raise ValueError('invalid confirmation authorization/seeds')
    if lock.get('baseline_role') not in BASELINES or set(lock.get('roles',{}))!=set(ROLES):
        raise ValueError('invalid baseline or role lock')
    scenes=lock.get('expected_scenes',[])
    if len(scenes)!=5 or len(set(scenes))!=5 or len({s.split('/')[0] for s in scenes})!=5:
        raise ValueError('five distinct locked scene/place groups required')
    selection_path=resolved(lock['selection_path'])
    if sha(selection_path)!=lock['selection_sha256']:raise ValueError('selection hash changed')
    selection=json.loads(selection_path.read_text())
    picked={s['scene']:s for s in selection['scenes'] if s.get('split')=='test'}
    if set(picked)!=set(scenes):raise ValueError('selection scene set differs from lock')
    pairs={(scene,int(q),int(k)) for scene,item in picked.items() for q,k in item['pairs']}
    if len(pairs)!=480 or any(sum(x[0]==s for x in pairs)!=96 for s in scenes):
        raise ValueError('locked selection must have 96 unique pairs per scan')
    if any(f not in lock['source_hashes'] for f in ('experiments/round7_depth_analysis.py','ROUND7_PROTOCOL.md')):
        raise ValueError('analyzer and protocol must be source-locked before test')
    for source,digest in lock['source_hashes'].items():
        if sha(resolved(source))!=digest:raise ValueError('locked source hash changed: '+source)
    validation_scores={role:[] for role in ROLES};artifacts=[];batches={};capacity=set();validation_population=None
    for role in ROLES:
        entries=lock['roles'][role].get('seeds',{})
        if set(entries)!={str(s) for s in SEEDS}:raise ValueError('incomplete locked checkpoints')
        for seed in SEEDS:
            entry=entries[str(seed)]
            checkpoint=resolved(entry['checkpoint_path']);validation=resolved(entry['validation_path'])
            run_path=checkpoint.parent/'run.json'
            for file,key in ((checkpoint,'checkpoint_sha256'),(validation,'validation_sha256'),(run_path,'run_sha256')):
                if sha(file)!=entry[key]:raise ValueError('locked training artifact changed: '+str(file))
            run=json.loads(run_path.read_text());val,_=load_file(validation)
            if run['role']!=role or run['seed']!=seed or run['steps']!=1000 or run['best_step']!=entry['best_step']:
                raise ValueError('training identity/budget mismatch')
            if any(lock['source_hashes'].get(k)!=v for k,v in run['source_hashes'].items()):
                raise ValueError('training source digest absent/different in lock')
            if run['validation_sha256']!=entry['validation_sha256'] or run['checkpoint_sha256']!=entry['checkpoint_sha256']:
                raise ValueError('run metadata artifact digest mismatch')
            capacity.add(run['parameter_count'])
            if seed in batches and batches[seed]!=run['batch_sequence_sha256']:raise ValueError('unpaired training batches')
            batches[seed]=run['batch_sequence_sha256']
            history=run['history']
            if [h['step'] for h in history]!=list(range(0,1001,100)):raise ValueError('incomplete validation history')
            earliest=min(history,key=lambda h:h['validation']['metrics']['AbsRel'])
            if earliest['step']!=entry['best_step']:raise ValueError('checkpoint is not earliest validation minimum')
            val_scenes=sorted({r['scene'] for r in val['rows']})
            if len(val_scenes)!=2 or {s.split('/')[0] for s in val_scenes}&{s.split('/')[0] for s in scenes}:
                raise ValueError('validation/test places overlap or validation incomplete')
            vrows=validate_row_block(val['rows'],val_scenes,pairs_per_scene=64)
            population=[identity(r)+(r['count'],r['boundary_count']) for r in vrows]
            if validation_population is None:validation_population=population
            elif population!=validation_population:raise ValueError('validation populations differ across roles/seeds')
            computed=aggregate_block(vrows,val_scenes);check_reported_summary(val['summary'],computed)
            score=computed['metrics']['AbsRel']
            if not math.isclose(score,run['best_validation_AbsRel'],rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('validation replay differs from metadata')
            validation_scores[role].append(score)
            artifacts.append(dict(role=role,seed=seed,checkpoint_sha256=entry['checkpoint_sha256'],
                validation_sha256=entry['validation_sha256'],run_sha256=entry['run_sha256'],best_step=entry['best_step']))
    if len(capacity)!=1:raise ValueError('parameter counts differ across controls')
    means={r:float(np.mean(v)) for r,v in validation_scores.items()}
    if means[lock['baseline_role']]>min(means[r] for r in BASELINES)+1e-12:
        raise ValueError('locked baseline is not validation-selected minimum')
    return lock,pairs,dict(lock_path=str(path),lock_sha256=hashlib.sha256(raw).hexdigest(),
        selection_sha256=lock['selection_sha256'],training_artifacts=artifacts,parameter_count=next(iter(capacity)),
        validation_role_mean_AbsRel=means,baseline_role=lock['baseline_role'],test_values_used=False)


def validate_test_data_lock(path,confirmation_sha,expected_pairs,expected_scenes):
    path=Path(path);raw=path.read_bytes();data=json.loads(raw)
    if data.get('confirmation_lock_sha256')!=confirmation_sha:
        raise ValueError('test-data lock belongs to a different confirmation')
    if data.get('pair_count')!=480 or data.get('total_positions')!=480*432:
        raise ValueError('test-data lock population is incomplete')
    for f in ('valid_count','boundary_count'):
        if isinstance(data.get(f),bool) or not isinstance(data.get(f),int) or data[f]<0:
            raise ValueError('invalid test-data lock count')
    if not 0<=data['boundary_count']<=data['valid_count']<=480*432:
        raise ValueError('invalid test-data lock boundary population')
    actual=set();scenes=set();paths=set()
    if not isinstance(data.get('sources'),list) or len(data['sources'])!=5:
        raise ValueError('five test-data sources required')
    for source in data['sources']:
        folder=resolved(source['path']);canonical=str(folder.resolve())
        if canonical in paths:raise ValueError('duplicate test-data source')
        paths.add(canonical)
        for filename,key in (('manifest.json','manifest_sha256'),('inputs.npz','input_sha256'),('labels.npz','label_sha256')):
            if sha(folder/filename)!=source[key]:raise ValueError('test-data source hash changed: '+str(folder/filename))
        manifest=json.loads((folder/'manifest.json').read_text())
        if manifest['split']!='test' or manifest['scene'] not in expected_scenes:
            raise ValueError('unexpected test-data manifest scene/split')
        if any(manifest[k]!=source[k] for k in ('input_sha256','label_sha256')):
            raise ValueError('test-data manifest/source archive digest mismatch')
        if manifest['scene'] in scenes or len(manifest['pairs'])!=96:
            raise ValueError('duplicate/incomplete test-data scene')
        scenes.add(manifest['scene'])
        for q,k in manifest['pairs']:
            key=(manifest['scene'],int(q),int(k))
            if key in actual:raise ValueError('duplicate test-data pair')
            actual.add(key)
    if actual!=set(expected_pairs) or scenes!=set(expected_scenes):
        raise ValueError('test-data identities differ from locked selection')
    return data,hashlib.sha256(raw).hexdigest()


def validate_evaluation_metadata(obj,path,lock_sha,data_sha,role,seed=None,checkpoint_sha=None):
    metadata=obj.get('metadata')
    if not isinstance(metadata,dict) or metadata.get('role')!=role:
        raise ValueError('evaluation role metadata mismatch')
    if metadata.get('lock_sha256')!=lock_sha or metadata.get('test_data_lock_sha256')!=data_sha:
        raise ValueError('evaluation confirmation/test-data lock metadata mismatch')
    if seed is not None:
        if isinstance(metadata.get('seed'),bool) or metadata.get('seed')!=seed:
            raise ValueError('evaluation seed metadata mismatch')
        if metadata.get('checkpoint_sha256')!=checkpoint_sha:
            raise ValueError('evaluation checkpoint metadata mismatch')
    prediction=Path(path).with_name('predictions.npz')
    if not prediction.is_file() or metadata.get('prediction_sha256')!=sha(prediction):
        raise ValueError('prediction archive metadata hash mismatch')
    return dict(verified_metadata=dict(metadata),prediction_path=str(prediction),prediction_sha256=sha(prediction))


def value_interventions(records,scenes):
    """Preserve both frozen interventions; they never select or gate a model."""
    if set(records)!={str(s) for s in SEEDS}:raise ValueError('missing full value interventions')
    summary={}
    for mode in ('zero_v','reverse_v'):
        metrics={m:[] for m in METRICS};changes=[]
        for seed in SEEDS:
            if set(records[str(seed)])!={'zero_v','reverse_v'}:raise ValueError('incomplete value interventions')
            record=records[str(seed)][mode];metric=record['summary']['metrics']
            if set(record['summary']['by_scene'])!=set(scenes):raise ValueError('intervention scene mismatch')
            change=record['mean_absolute_output_change_m']
            if not math.isfinite(change) or change<0:raise ValueError('invalid value-intervention output change')
            changes.append(change)
            for m in METRICS:
                if metric[m] is not None:
                    if not math.isfinite(metric[m]) or metric[m]<0:raise ValueError('nonfinite value-intervention metric')
                    metrics[m].append(metric[m])
        summary[mode]=dict(metrics={m:float(np.mean(v)) if v else None for m,v in metrics.items()},
            mean_absolute_output_change_m=float(np.mean(changes)))
    return dict(by_seed=records,seed_mean=summary,descriptive_only=True,
        note='frozen interventions; improvements after reversed V must be retained, not interpreted as evidence for correspondence dependence')


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    lock,pairs,verification=validate_lock(args.lock)
    data_lock,data_sha=validate_test_data_lock(args.runs/'test_data_lock.json',verification['lock_sha256'],pairs,lock['expected_scenes'])
    completion_path=args.runs/'completion.json';completion=json.loads(completion_path.read_text())
    if (completion.get('lock_sha256')!=verification['lock_sha256'] or completion.get('test_data_lock_sha256')!=data_sha
            or set(completion.get('roles',[]))!=set(ROLES) or completion.get('seeds')!=list(SEEDS)):
        raise ValueError('incomplete or mismatched confirmation completion')
    blocks={};files=[];interventions={}
    for role in ROLES:
        blocks[role]={}
        for seed in SEEDS:
            path=args.runs/role/str(seed)/'test.json';obj,meta=load_file(path)
            rows=validate_row_block(obj['rows'],lock['expected_scenes'],pairs)
            check_reported_summary(obj['summary'],aggregate_block(rows,sorted(lock['expected_scenes'])))
            entry=lock['roles'][role]['seeds'][str(seed)]
            meta.update(role=role,seed=seed,**validate_evaluation_metadata(obj,path,verification['lock_sha256'],data_sha,
                role,seed,entry['checkpoint_sha256']))
            if role=='full':
                if not isinstance(obj.get('interventions'),dict):raise ValueError('full frozen interventions missing')
                for mode,record in obj['interventions'].items():
                    summary=record['summary']
                    if summary['total_count']!=sum(r['count'] for r in rows) or summary['total_positions']!=480*432:
                        raise ValueError('intervention population mismatch')
                    for scene in lock['expected_scenes']:
                        chosen=[r for r in rows if r['scene']==scene]
                        for count in ('count','boundary_count'):
                            if summary['by_scene'][scene][count]!=sum(r[count] for r in chosen):
                                raise ValueError('intervention scene population mismatch')
                interventions[str(seed)]=obj['interventions']
            files.append(meta);blocks[role][seed]=rows
    raw,meta=load_file(args.raw)
    raw_rows=validate_row_block(raw['rows'],lock['expected_scenes'],pairs)
    check_reported_summary(raw['summary'],aggregate_block(raw_rows,sorted(lock['expected_scenes'])))
    meta.update(role='baseline_raw',**validate_evaluation_metadata(raw,args.raw,verification['lock_sha256'],data_sha,'raw'));files.append(meta)
    result=analyze_rows(blocks,lock['baseline_role'],lock['expected_scenes'],raw_rows=raw_rows,expected_pairs=pairs)
    if result['population']['total_valid_positions']!=data_lock['valid_count'] or result['population']['boundary_positions']!=data_lock['boundary_count']:
        raise ValueError('row populations disagree with test-data lock')
    result.update(verification=verification,evaluation_files=files,analyzer_sha256=sha(__file__),
        protocol_sha256=sha(ROOT/'ROUND7_PROTOCOL.md'),test_data_lock_sha256=data_sha,
        completion_sha256=sha(completion_path),frozen_full_value_interventions=value_interventions(interventions,lock['expected_scenes']),
        raw_prediction_metrics_independently_recomputed=False)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    primary=result['primary_gate'];lines=['# Round7 coarse depth confirmation','',
        f"Locked strong baseline: **{lock['baseline_role']}**. Primary research gate: **{'passed' if primary['passed'] else 'failed'}**.",
        '', 'All values use valid-query counts within scan, then equal scan and seed weights. '
        'Intervals use 20,000 paired crossed scan/seed draws; no pixel resampling or deterministic-baseline seed replication.','',
        '| Role | AbsRel | RMSE (m) | MAE (m) | delta1 |','|---|---:|---:|---:|---:|']
    for role in ROLES:
        v=result['models'][role];lines.append(f"| {role} | {v['AbsRel']['mean']:.6f} | {v['RMSE']['mean']:.6f} | {v['MAE']['mean']:.6f} | {v['delta1']['mean']:.6f} |")
    relative=primary['AbsRel']['relative_difference'];relative_text=f'{relative:.3%}' if relative is not None else 'undefined (zero baseline error)'
    lines+=['',f"Full minus strong baseline AbsRel: {primary['AbsRel']['difference']:+.6f}, 95% CI {primary['AbsRel']['ci95_difference']}; relative change {relative_text}.",
        f"Improving seed macros: {primary['AbsRel']['improved_seed_count']}/5. Full minus singleview AbsRel 95% CI: {primary['singleview_AbsRel']['ci95_difference']}.",
        f"RMSE relative-change 95% CI versus baseline: {primary['RMSE']['ci95_relative_difference']}.",'',
        'Every control, per-scan/seed result, leave-one-scan sensitivity, frozen raw baseline and failed criterion is retained in summary.json. '
        'The raw baseline is descriptive. Five place groups do not establish arbitrary institutions or a SOTA native-resolution depth result. '
        'The analyzer verifies additive summaries and locked populations/artifacts; separate prediction auditing must verify arrays against sensor labels.','']
    lines+=['Frozen full-model value interventions (descriptive):']
    for mode,record in result['frozen_full_value_interventions']['seed_mean'].items():
        lines.append(f"- {mode}: AbsRel {record['metrics']['AbsRel']:.6f}, mean absolute output change {record['mean_absolute_output_change_m']:.6f} m.")
    (args.out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(primary,indent=2))


if __name__=='__main__':main()
