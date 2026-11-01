"""Fixed TRAIN-only content/scale contrasts; never a candidate selection grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import round12_stress_probe as S
from experiments import round11_target_conflict as T

A=S.A
M=S.M
ROOT=A.ROOT
SCENES=T.TRAIN_SCENES
QUERIES=('g8','g16','g24')
CONFIGS={
    'full4_t0':dict(method='full4',h=.5,temperature=0.),
    'full4_t5':dict(method='full4',h=.5,temperature=5.),
    'full1_t0':dict(method='full',h=.5,temperature=0.),
    'full1_t5':dict(method='full',h=.5,temperature=5.),
    'point_pck':dict(method='point',h=.5,temperature=5.),
    'quad16':dict(method='quad16',h=.5,temperature=5.),
    'full1_frozen':dict(method='full',h=.25,temperature=0.),
}
CONTRASTS={'full4_content_off':('full4_t0','full4_t5'),
           'full1_content_off':('full1_t0','full1_t5'),
           'full1_old_bandwidth':('full1_frozen','full1_t0')}


def sources():
    result=S.sources()
    for name in ('ROUND14_TEMPERATURE_PROTOCOL.md','experiments/round14_temperature_probe.py',
                 'tests/test_round14_temperature_probe.py','experiments/round11_target_conflict.py'):
        result[name]=A.sha(ROOT/name)
    return result


def target_contract(targets,audit,records):
    """Read source manifests/hash metadata, never target arrays or score summaries."""
    lock=json.loads((targets/'lock.json').read_text())
    completion=json.loads((targets/'completion.json').read_text())
    if A.sha(targets/'lock.json')!=(targets/'lock.sha256').read_text().strip():raise ValueError('changed target lock')
    if completion['lock_sha256']!=A.sha(targets/'lock.json'):raise ValueError('target completion/lock mismatch')
    if lock['training_manifest_records']!=records:raise ValueError('target/cohort mismatch')
    for path,digest in lock['source_hashes'].items():
        if A.sha(ROOT/path)!=digest:raise ValueError('changed target source')
    verification=json.loads(audit.read_text())
    if verification['passed'] is not True:raise ValueError('independent target audit must pass')
    if verification['target_lock_sha256']!=A.sha(targets/'lock.json') or verification['target_completion_sha256']!=A.sha(targets/'completion.json'):
        raise ValueError('independent target audit hash mismatch')
    files={};entries=[]
    for record in records:
        manifest=Path(record['path'])
        slug=record['scene'].replace('/','__')+'__'+manifest.parent.name
        relative='targets/'+slug+'.npz'
        files[str(targets/relative)]=completion['files'][relative]
        entries.append(dict(scene=record['scene'],pair=record['pair'],path=str(manifest.parent),
                            target_path=str(targets/relative)))
    if {p for p in completion['files'] if p.startswith('targets/')}!={str(Path(p).relative_to(targets)) for p in files}:
        raise ValueError('unexpected target archive population')
    return dict(root=str(targets.resolve()),audit_path=str(audit.resolve()),
        lock_sha256=A.sha(targets/'lock.json'),completion_sha256=A.sha(targets/'completion.json'),
        audit_sha256=A.sha(audit),expected_target_hashes=files,entries=entries)


def freeze(out,data,targets,audit):
    if out.exists():raise ValueError('refusing overwrite')
    data,targets,audit=data.resolve(),targets.resolve(),audit.resolve()
    records=T.data_records(data)
    target=target_contract(targets,audit,records)
    inputs={};manifests={}
    for record in records:
        path=Path(record['path'])
        if A.sha(path)!=record['sha256']:raise ValueError('changed train manifest')
        meta=json.loads(path.read_text())
        if meta['split']!='train' or meta['scene']!=record['scene'] or meta['pair']!=record['pair']:
            raise ValueError('training identity mismatch')
        inputs[str(path.parent/'inputs.npz')]=meta['input_sha256'];manifests[str(path)]=record['sha256']
    S.verify_files(inputs)
    stresslock=ROOT/'results/round12_stress/lock.json'
    old=json.loads(stresslock.read_text())
    if A.sha(stresslock)!=(stresslock.parent/'lock.sha256').read_text().strip():raise ValueError('changed frozen operator lock')
    if S.sources()!=old['source_hashes']:raise ValueError('changed inherited frozen operator')
    out.mkdir(parents=True)
    parts=M.partitions()
    np.savez_compressed(out/'partitions.npz',**{name+'__'+k:v for name,p in parts.items() for k,v in p.items()})
    lock=dict(created_unix=time.time(),source_hashes=sources(),configurations=CONFIGS,contrasts=CONTRASTS,
        data_root=str(data.resolve()),data_completion_sha256=A.sha(data/'completion.json'),
        training_manifest_records=records,inputs=inputs,manifests=manifests,target_contract=target,
        partitions_sha256=A.sha(out/'partitions.npz'),operator_parent_lock_sha256=A.sha(stresslock),
        forward_before_all_target_array_reads=True,targets_decoded=False,
        pair_count=48,condition_count=3,role_count=7,probability_dtype='float64',
        no_parameter_selection=True,diagnostic_only=True,fresh_test_read=False,torch=torch.__version__)
    A.write(out/'lock.json',lock);(out/'lock.sha256').write_text(A.sha(out/'lock.json')+'\n')
    print('FROZEN',A.sha(out/'lock.json'),flush=True)


def verify_lock(out):
    lock=json.loads((out/'lock.json').read_text())
    if A.sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('changed R14 lock')
    if sources()!=lock['source_hashes'] or CONFIGS!=lock['configurations']:
        raise ValueError('changed source/configurations')
    if {k:list(v) for k,v in CONTRASTS.items()}!=lock['contrasts']:raise ValueError('changed planned contrasts')
    data=Path(lock['data_root'])
    if A.sha(data/'completion.json')!=lock['data_completion_sha256'] or T.data_records(data)!=lock['training_manifest_records']:
        raise ValueError('changed training cohort')
    S.verify_files(lock['inputs']);S.verify_files(lock['manifests'])
    old=lock['target_contract']
    if target_contract(Path(old['root']),Path(old['audit_path']),lock['training_manifest_records'])!=old:
        raise ValueError('changed independent target contract')
    if A.sha(out/'partitions.npz')!=lock['partitions_sha256']:raise ValueError('changed partitions')
    if A.sha(ROOT/'results/round12_stress/lock.json')!=lock['operator_parent_lock_sha256']:
        raise ValueError('changed parent operator lock')
    return lock


def scene_entries(lock,scene):
    if scene not in SCENES:raise ValueError('only three frozen training scenes allowed')
    entries=[e for e in lock['target_contract']['entries'] if e['scene']==scene]
    if len(entries)!=16:raise ValueError('expected 16 train pairs per scene')
    return sorted(entries,key=lambda e:Path(e['path']).name)


def condition_predictions(queries,key):
    """Compute five geometries once, then seven predeclared probability roles."""
    content=queries['full']['features']@key['features'].T
    geometry={};numerical=[];probabilities={}
    for role,config in CONFIGS.items():
        method,h,temp=config['method'],config['h'],config['temperature'];identity=(method,h)
        prep=method if method in ('full4','quad16') else 'full'
        if identity not in geometry:
            diag=dict(method=method,h=h)
            geometry[identity]=M.geometry_logits(queries[prep],key,method,h,diagnostics=diag)
            numerical.append(diag)
        p=(geometry[identity]+temp*content).softmax(-1)
        if not bool(torch.isfinite(p).all()) or float((p.sum(-1)-1).abs().max())>1.e-12:
            raise ValueError('invalid probability')
        probabilities[role]=p
    return probabilities,numerical


@torch.no_grad()
def forward(out,scene,device):
    lock=verify_lock(out);entries=scene_entries(lock,scene)
    directory=out/'forward'/S.B.scene_slug(scene)
    if directory.exists():raise ValueError('refusing overwrite/partial forward')
    directory.mkdir(parents=True);started=time.time();numerical=[]
    partition=M.partition_tensors(S.read_parts(out),device)
    for index,entry in enumerate(entries):
        path=Path(entry['path']);obs=A.observation_tensors(A.load_npz(path/'inputs.npz'),device)
        key=M.prepare_key(obs,partition['g8']);saved={}
        for qname in QUERIES:
            query={method:M.prepare_query(obs,partition[qname],method) for method in ('full4','full','quad16')}
            probabilities,diagnostics=condition_predictions(query,key)
            saved['predicted_region_mean__'+qname]=(query['full']['mean'].double()*8).cpu().numpy()
            for role,p in probabilities.items():
                name=role+'__'+qname;saved[name]=p.cpu().numpy();saved[name+'__argmax']=p.argmax(-1).cpu().numpy().astype(np.int32)
            numerical.extend(dict(scene=scene,pair=entry['pair'],query=qname,**d) for d in diagnostics)
        np.savez_compressed(directory/(path.name+'.npz'),**saved)
        print('FORWARD',scene,index+1,'/16',flush=True)
    A.write(directory/'numerical.json',numerical)
    S.complete_stage(directory,out,'forward',scene,started,pair_count=16,record_count=16*3*7,
                     targets_decoded=False,device=device)


def all_forward_complete(out):
    result={}
    for scene in SCENES:
        path=out/'forward'/S.B.scene_slug(scene)
        value=S.verify_stage(path,out,'forward',scene)
        result[scene]=dict(completion_sha256=A.sha(path/'completion.json'),finished_unix=value['finished_unix'])
    return result


def measure(p,argmax,mean,key_centers,counts,point,valid):
    if p.dtype!=np.float64 or p.shape!=counts.shape or len(mean)!=len(p):raise ValueError('prediction shape/dtype mismatch')
    if not np.isfinite(p).all() or (p<0).any() or not np.isfinite(mean).all():raise ValueError('invalid prediction')
    np.testing.assert_allclose(p.sum(-1),1.,atol=1.e-12,rtol=0)
    if not np.array_equal(argmax,p.argmax(-1)):raise ValueError('argmax mismatch')
    kl,js,mass=S.kl_js(p,counts)
    winner=key_centers[argmax];expected=p@key_centers
    displacement=np.linalg.norm(winner-mean,axis=-1)
    expected_displacement=np.linalg.norm(expected-mean,axis=-1)
    correct=(np.linalg.norm(winner-point,axis=-1)<=8)&valid
    near=np.linalg.norm(key_centers[None]-mean[:,None],axis=-1)<=8
    near_mass=(p*near).sum(-1)
    mean_error=np.linalg.norm(mean[valid]-point[valid],axis=-1)
    row=dict(kl_sum=kl,js_sum=js,mass=mass,pck_count=int(correct.sum()),n_point=int(valid.sum()),n_query=len(p),
        winner_to_mean_sum_all=float(displacement.sum()),winner_to_mean_sum_center_valid=float(displacement[valid].sum()),
        expected_to_mean_sum_all=float(expected_displacement.sum()),
        expected_to_mean_sum_center_valid=float(expected_displacement[valid].sum()),
        within8_of_mean_probability_sum_all=float(near_mass.sum()),
        within8_of_mean_probability_sum_center_valid=float(near_mass[valid].sum()),
        predicted_mean_to_sensor_center_sum=float(mean_error.sum()))
    return row,correct


def evaluate(out,scene):
    lock=verify_lock(out);entries=scene_entries(lock,scene);chain=all_forward_complete(out)
    directory=out/'evaluation'/S.B.scene_slug(scene)
    if directory.exists():raise ValueError('refusing overwrite/partial evaluation')
    directory.mkdir(parents=True);started=time.time()
    if max(v['finished_unix'] for v in chain.values())>started:raise ValueError('forward chronology')
    key_centers=S.read_parts(out)['g8']['mean']*8;rows=[];paired=[];label_sources={}
    for index,entry in enumerate(entries):
        path=Path(entry['path']);target=Path(entry['target_path'])
        expected_hash=lock['target_contract']['expected_target_hashes'][str(target)]
        if A.sha(target)!=expected_hash:raise ValueError('changed independent training targets')
        labels=A.load_npz(target);label_sources[str(target)]=expected_hash
        prediction=A.load_npz(out/'forward'/S.B.scene_slug(scene)/(path.name+'.npz'))
        expected={role+'__'+q+suffix for role in CONFIGS for q in QUERIES for suffix in ('','__argmax')}
        expected|={'predicted_region_mean__'+q for q in QUERIES}
        if set(prediction)!=expected:raise ValueError('unexpected forward keys')
        for q in QUERIES:
            mean=prediction['predicted_region_mean__'+q];valid=labels[q+'_point_visible'];correct={}
            for role,config in CONFIGS.items():
                name=role+'__'+q
                row,correct[role]=measure(prediction[name],prediction[name+'__argmax'],mean,key_centers,
                    labels[q+'_counts'],labels[q+'_point_xy'],valid)
                row.update(scene=scene,pair=entry['pair'],query=q,role=role,config=config);rows.append(row)
            for name,(left,right) in CONTRASTS.items():
                li=prediction[left+'__'+q+'__argmax'];ri=prediction[right+'__'+q+'__argmax']
                paired.append(dict(scene=scene,pair=entry['pair'],query=q,contrast=name,left=left,right=right,
                    n_query=len(li),n_point=int(valid.sum()),winner_changed_all=int((li!=ri).sum()),
                    winner_changed_center_valid=int(((li!=ri)&valid).sum()),
                    winner_displacement_sum_all=float(np.linalg.norm(key_centers[li]-key_centers[ri],axis=-1).sum()),
                    left_correct_right_wrong=int((correct[left]&~correct[right]&valid).sum()),
                    right_correct_left_wrong=int((correct[right]&~correct[left]&valid).sum())))
        print('EVALUATED',scene,index+1,'/16',flush=True)
    verify_population(rows,entries)
    A.write(directory/'rows.json',rows);A.write(directory/'paired_rows.json',paired)
    S.complete_stage(directory,out,'evaluate',scene,started,forward_chain=chain,label_sources=label_sources,
                     record_count=len(rows),paired_record_count=len(paired))


def verify_population(rows,entries):
    expected={(e['scene'],tuple(e['pair']),q) for e in entries for q in QUERIES};population=None
    for role in CONFIGS:
        selected=[r for r in rows if r['role']==role]
        mapping={(r['scene'],tuple(r['pair']),r['query']):(r['mass'],r['n_point'],r['n_query']) for r in selected}
        if set(mapping)!=expected or len(mapping)!=len(selected):raise ValueError('role population missing/duplicate')
        if population is None:population=mapping
        elif population!=mapping:raise ValueError('role target population changed')
    for (scene,pair,q),values in population.items():
        if values[0]!=population[(scene,pair,'g8')][0]:raise ValueError('scale changed visible pixel mass')


def aggregate(rows):
    def ratio(field,denominator):
        denominator=sum(r[denominator] for r in rows)
        if denominator<=0:raise ValueError('empty metric population')
        return sum(r[field] for r in rows)/denominator
    return dict(direct_kl=ratio('kl_sum','mass'),pck8=ratio('pck_count','n_point'),
        winner_to_predicted_mean_px_all=ratio('winner_to_mean_sum_all','n_query'),
        winner_to_predicted_mean_px_center_valid=ratio('winner_to_mean_sum_center_valid','n_point'),
        expected_to_predicted_mean_px_all=ratio('expected_to_mean_sum_all','n_query'),
        expected_to_predicted_mean_px_center_valid=ratio('expected_to_mean_sum_center_valid','n_point'),
        mass_within8_of_predicted_mean_all=ratio('within8_of_mean_probability_sum_all','n_query'),
        mass_within8_of_predicted_mean_center_valid=ratio('within8_of_mean_probability_sum_center_valid','n_point'),
        predicted_mean_to_sensor_center_px=ratio('predicted_mean_to_sensor_center_sum','n_point'))


def summarize(rows,paired):
    output={}
    for role in CONFIGS:
        scenes={s:{q:aggregate([r for r in rows if r['role']==role and r['scene']==s and r['query']==q])
                   for q in QUERIES} for s in SCENES}
        by_query={q:{metric:float(np.mean([scenes[s][q][metric] for s in SCENES]))
                     for metric in scenes[SCENES[0]][q]} for q in QUERIES}
        output[role]=dict(by_scene_query=scenes,by_query=by_query,
            equal_scale_macro={metric:float(np.mean([by_query[q][metric] for q in QUERIES]))
                               for metric in by_query['g8']})
    contrasts={}
    for name,(left,right) in CONTRASTS.items():
        contrasts[name]={}
        for q in QUERIES:
            l,r=output[left]['by_query'][q],output[right]['by_query'][q]
            scene_values={}
            for scene in SCENES:
                subset=[p for p in paired if p['contrast']==name and p['query']==q and p['scene']==scene]
                if len(subset)!=16:raise ValueError('missing paired contrast population')
                total=sum(p['n_point'] for p in subset);all_count=sum(p['n_query'] for p in subset)
                scene_values[scene]=dict(winner_changed_fraction_all=sum(p['winner_changed_all'] for p in subset)/all_count,
                    winner_changed_fraction_center_valid=sum(p['winner_changed_center_valid'] for p in subset)/total,
                    left_correct_right_wrong_fraction=sum(p['left_correct_right_wrong'] for p in subset)/total,
                    right_correct_left_wrong_fraction=sum(p['right_correct_left_wrong'] for p in subset)/total,
                    winner_displacement_px_all=sum(p['winner_displacement_sum_all'] for p in subset)/all_count)
            contrasts[name][q]=dict(left=left,right=right,direct_kl_relative_change=l['direct_kl']/r['direct_kl']-1,
                pck_difference_pp=100*(l['pck8']-r['pck8']),
                center_valid_winner_offset_difference_px=l['winner_to_predicted_mean_px_center_valid']-r['winner_to_predicted_mean_px_center_valid'],
                by_scene={s:dict(direct_kl_relative_change=output[left]['by_scene_query'][s][q]['direct_kl']/output[right]['by_scene_query'][s][q]['direct_kl']-1,
                    pck_difference_pp=100*(output[left]['by_scene_query'][s][q]['pck8']-output[right]['by_scene_query'][s][q]['pck8']),
                    center_valid_winner_offset_difference_px=output[left]['by_scene_query'][s][q]['winner_to_predicted_mean_px_center_valid']-output[right]['by_scene_query'][s][q]['winner_to_predicted_mean_px_center_valid'],
                    **scene_values[s]) for s in SCENES},
                paired_equal_scene={k:float(np.mean([v[k] for v in scene_values.values()])) for k in scene_values[SCENES[0]]})
    gap={}
    for q in QUERIES:
        before=output['full4_t5']['by_query'][q]['pck8'];after=output['full4_t0']['by_query'][q]['pck8'];point=output['point_pck']['by_query'][q]['pck8']
        gap[q]=dict(t5_minus_point_pp=100*(before-point),t0_minus_point_pp=100*(after-point),
            train_gap_recovered_fraction=(after-before)/(point-before) if point>before else None)
    return dict(roles=output,contrasts=contrasts,full4_train_pck_gap=gap)


def finalize(out):
    lock=verify_lock(out)
    if (out/'completion.json').exists():raise ValueError('refusing overwrite final result')
    chain=all_forward_complete(out);rows=[];paired=[];shards={}
    for scene in SCENES:
        path=out/'evaluation'/S.B.scene_slug(scene);completion=S.verify_stage(path,out,'evaluate',scene)
        if completion['forward_chain']!=chain:raise ValueError('changed forward chain')
        rows.extend(json.loads((path/'rows.json').read_text()));paired.extend(json.loads((path/'paired_rows.json').read_text()))
        shards[scene]=A.sha(path/'completion.json')
    verify_population(rows,lock['target_contract']['entries'])
    result=dict(summary=summarize(rows,paired),configurations=CONFIGS,contrasts=CONTRASTS,
        pair_count=48,probability_records=len(rows),paired_records=len(paired),lock_sha256=A.sha(out/'lock.json'),
        source_hashes=sources(),forward_chain=chain,evaluation_shards=shards,
        diagnostic_only=True,no_parameter_selection=True,fresh_test_read=False,validation_gate_evaluated=False,
        predicted_center_definition='Full-support projected image-area mean used by the point control; not projected sensor center',
        interpretation='Fixed paired TRAIN-only contrasts; no winner/configuration selected, no validation gate or recovery claim for the previous 11.20pp validation loss.')
    A.write(out/'rows.json',rows);A.write(out/'paired_rows.json',paired);A.write(out/'summary.json',result)
    files={str(p.relative_to(out)):A.sha(p) for p in out.rglob('*') if p.is_file()}
    A.write(out/'completion.json',dict(files=files,lock_sha256=A.sha(out/'lock.json'),finished_unix=time.time(),
        probability_records=len(rows),diagnostic_only=True,fresh_test_read=False))
    print(json.dumps(result['summary']['contrasts'],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','forward','evaluate','finalize'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round14_temperature')
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--targets',type=Path,default=ROOT/'results/round13_target_conflict')
    parser.add_argument('--target-audit',type=Path,default=ROOT/'results/round13_target_independent_audit/verification.json')
    parser.add_argument('--scene',choices=SCENES);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.stage=='freeze':freeze(args.out,args.data,args.targets,args.target_audit)
    elif args.stage=='forward':forward(args.out,args.scene,args.device)
    elif args.stage=='evaluate':evaluate(args.out,args.scene)
    else:finalize(args.out)


if __name__=='__main__':main()
