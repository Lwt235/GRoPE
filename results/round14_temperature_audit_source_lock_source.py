"""Independent saved-prediction and NumPy/SciPy audit of fixed TRAIN contrasts.

Never imports the R14 producer or actual model. Existing audited R13 targets
are reused by exact archive hash; this is not another scientific gate.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from experiments import round13_mixture_mechanism_audit as R

ROOT=Path(__file__).resolve().parents[1]
SCENES=R.SCENES
STRIDES=(8,16,24)
CONFIGS={
    'full4_t0':dict(method='full4',h=.5,temperature=0.),
    'full4_t5':dict(method='full4',h=.5,temperature=5.),
    'full1_t0':dict(method='full',h=.5,temperature=0.),
    'full1_t5':dict(method='full',h=.5,temperature=5.),
    'point_pck':dict(method='point',h=.5,temperature=5.),
    'quad16':dict(method='quad16',h=.5,temperature=5.),
    'full1_frozen':dict(method='full',h=.25,temperature=0.),
}
CONTRASTS={'full4_content_off':['full4_t0','full4_t5'],
           'full1_content_off':['full1_t0','full1_t5'],
           'full1_old_bandwidth':['full1_frozen','full1_t0']}
FIELDS=('q_to_k_xy','q_to_k_pixel_cov','q_features','k_features')
OWN_SOURCES=('experiments/round14_temperature_audit.py','tests/test_round14_temperature_audit.py',
    'experiments/round13_mixture_mechanism_audit.py','experiments/round10_cell_result_audit.py',
    'experiments/round10_partition_audit.py')
PRODUCER_SOURCES=('ROUND14_TEMPERATURE_PROTOCOL.md','experiments/round14_temperature_probe.py',
                  'tests/test_round14_temperature_probe.py')
METRICS={
    'direct_kl':('kl_sum','mass'),'pck8':('pck_count','n_point'),
    'winner_to_predicted_mean_px_all':('winner_to_mean_sum_all','n_query'),
    'winner_to_predicted_mean_px_center_valid':('winner_to_mean_sum_center_valid','n_point'),
    'expected_to_predicted_mean_px_all':('expected_to_mean_sum_all','n_query'),
    'expected_to_predicted_mean_px_center_valid':('expected_to_mean_sum_center_valid','n_point'),
    'mass_within8_of_predicted_mean_all':('within8_of_mean_probability_sum_all','n_query'),
    'mass_within8_of_predicted_mean_center_valid':('within8_of_mean_probability_sum_center_valid','n_point'),
    'predicted_mean_to_sensor_center_px':('predicted_mean_to_sensor_center_sum','n_point')}


def archive(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def centers(stride):
    bounds=R.grid(stride)
    return (bounds[:,:2]+bounds[:,2:]-1)/2


def metric_reference(p,index,mean,counts,point,valid,key_centers=None):
    keys=centers(8) if key_centers is None else np.asarray(key_centers)
    if p.dtype!=np.float64 or p.shape!=counts.shape or mean.shape!=(len(p),2):raise ValueError('prediction shape/dtype')
    if valid.dtype!=np.bool_ or valid.shape!=(len(p),) or point.shape!=(len(p),2):raise ValueError('center shape/dtype')
    if np.any(p<0) or not np.isfinite(p).all() or not np.isfinite(mean).all():raise ValueError('invalid probability/mean')
    if not np.isfinite(point[valid]).all() or np.any(counts<0):raise ValueError('invalid target')
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-12,rtol=0)
    if not np.array_equal(p.argmax(-1),index):raise ValueError('native first argmax changed')
    mass=counts.sum(1);truth=counts/np.maximum(mass,1)[:,None]
    # The fixed legacy metric clips log arguments at 1e-15, not the forward mass.
    logg=np.log(np.clip(truth,1e-15,None));logp=np.log(np.clip(p,1e-15,None))
    logm=np.log(np.clip((truth+p)*.5,1e-15,None))
    kl=float((counts*(logg-logp)).sum())
    js=float((mass*.5*(truth*(logg-logm)+p*(logp-logm)).sum(1)).sum())
    winners=keys[index];expected=np.einsum('qk,kd->qd',p,keys)
    offset=np.sqrt(((winners-mean)**2).sum(1));expected_offset=np.sqrt(((expected-mean)**2).sum(1))
    near=((keys[None]-mean[:,None])**2).sum(2)<=64
    probability_near=(p*near).sum(1)
    correct=np.zeros(len(p),bool);correct[valid]=((winners[valid]-point[valid])**2).sum(1)<=64
    result=dict(kl_sum=kl,js_sum=js,mass=int(mass.sum()),pck_count=int(correct.sum()),
        n_point=int(valid.sum()),n_query=len(p),
        winner_to_mean_sum_all=float(offset.sum()),winner_to_mean_sum_center_valid=float(offset[valid].sum()),
        expected_to_mean_sum_all=float(expected_offset.sum()),expected_to_mean_sum_center_valid=float(expected_offset[valid].sum()),
        within8_of_mean_probability_sum_all=float(probability_near.sum()),
        within8_of_mean_probability_sum_center_valid=float(probability_near[valid].sum()),
        predicted_mean_to_sensor_center_sum=float(np.sqrt(((mean[valid]-point[valid])**2).sum(1)).sum()))
    return result,correct


def paired_reference(left,right,left_correct,right_correct,valid,key_centers=None):
    keys=centers(8) if key_centers is None else key_centers
    return dict(n_query=len(left),n_point=int(valid.sum()),winner_changed_all=int((left!=right).sum()),
        winner_changed_center_valid=int(((left!=right)&valid).sum()),
        winner_displacement_sum_all=float(np.sqrt(((keys[left]-keys[right])**2).sum(1)).sum()),
        left_correct_right_wrong=int((left_correct&~right_correct&valid).sum()),
        right_correct_left_wrong=int((right_correct&~left_correct&valid).sum()))


def aggregate_reference(rows):
    result={}
    for metric,(numerator,denominator) in METRICS.items():
        total=sum(r[denominator] for r in rows)
        if total<=0:raise ValueError('empty metric population')
        result[metric]=sum(r[numerator] for r in rows)/total
    return result


def summary_reference(rows,paired):
    queries=tuple('g'+str(s) for s in STRIDES);roles={}
    for role in CONFIGS:
        by_scene={s:{q:aggregate_reference([r for r in rows if (r['role'],r['scene'],r['query'])==(role,s,q)])
                     for q in queries} for s in SCENES}
        by_query={q:{k:sum(by_scene[s][q][k] for s in SCENES)/3 for k in METRICS} for q in queries}
        roles[role]=dict(by_scene_query=by_scene,by_query=by_query,
                        equal_scale_macro={k:sum(by_query[q][k] for q in queries)/3 for k in METRICS})
    contrasts={}
    for name,(left,right) in CONTRASTS.items():
        contrasts[name]={}
        for q in queries:
            by_scene={};paired_scene={}
            for s in SCENES:
                selected=[r for r in paired if (r['contrast'],r['scene'],r['query'])==(name,s,q)]
                if len(selected)!=16:raise ValueError('paired scene population')
                nq=sum(r['n_query'] for r in selected);nv=sum(r['n_point'] for r in selected)
                pair_stats=dict(winner_changed_fraction_all=sum(r['winner_changed_all'] for r in selected)/nq,
                    winner_changed_fraction_center_valid=sum(r['winner_changed_center_valid'] for r in selected)/nv,
                    left_correct_right_wrong_fraction=sum(r['left_correct_right_wrong'] for r in selected)/nv,
                    right_correct_left_wrong_fraction=sum(r['right_correct_left_wrong'] for r in selected)/nv,
                    winner_displacement_px_all=sum(r['winner_displacement_sum_all'] for r in selected)/nq)
                l,r=roles[left]['by_scene_query'][s][q],roles[right]['by_scene_query'][s][q]
                by_scene[s]=dict(direct_kl_relative_change=l['direct_kl']/r['direct_kl']-1,
                    pck_difference_pp=100*(l['pck8']-r['pck8']),
                    center_valid_winner_offset_difference_px=l['winner_to_predicted_mean_px_center_valid']-r['winner_to_predicted_mean_px_center_valid'],
                    **pair_stats)
                paired_scene[s]=pair_stats
            l,r=roles[left]['by_query'][q],roles[right]['by_query'][q]
            contrasts[name][q]=dict(left=left,right=right,direct_kl_relative_change=l['direct_kl']/r['direct_kl']-1,
                pck_difference_pp=100*(l['pck8']-r['pck8']),
                center_valid_winner_offset_difference_px=l['winner_to_predicted_mean_px_center_valid']-r['winner_to_predicted_mean_px_center_valid'],
                by_scene=by_scene,paired_equal_scene={k:sum(paired_scene[s][k] for s in SCENES)/3 for k in paired_scene[SCENES[0]]})
    gaps={}
    for q in queries:
        t0=roles['full4_t0']['by_query'][q]['pck8'];t5=roles['full4_t5']['by_query'][q]['pck8'];point=roles['point_pck']['by_query'][q]['pck8']
        gaps[q]=dict(t5_minus_point_pp=100*(t5-point),t0_minus_point_pp=100*(t0-point),
                    train_gap_recovered_fraction=(t0-t5)/(point-t5) if point>t5 else None)
    return dict(roles=roles,contrasts=contrasts,full4_train_pck_gap=gaps)


def reference_probabilities(inputs,stride,indices):
    bounds=R.grid(stride)[indices];content=R.content_reference(inputs,bounds)
    banks={method:R.component_reference(inputs,bounds,method) for method in ('full4','full1','quad16')}
    geometries={};result={};keys=(R.grid(8)-.5)/8
    for role,cfg in CONFIGS.items():
        method,h=cfg['method'],cfg['h'];identity=(method,h)
        if identity not in geometries:
            bank=banks['full1' if method in ('full','point') else method]
            cov=np.zeros_like(bank['cov']) if method=='point' else bank['cov']
            cells=R.REF.reference_mass(bank['mean'],cov,keys,h,order=64)
            mass=np.array([(cells[bank['owner']==i]*bank['weights'][bank['owner']==i,None]).sum(0) for i in range(len(bounds))])
            geometries[identity]=np.log(mass+1e-30)
        result[role]=softmax(geometries[identity]+cfg['temperature']*content,axis=1)
    return result


def freeze(source_lock,runs):
    if source_lock.exists():raise ValueError('refuse audit lock overwrite')
    source_lock.parent.mkdir(parents=True,exist_ok=True)
    snapshot=source_lock.with_name(source_lock.stem+'_source.py')
    with snapshot.open('xb') as file:file.write(Path(__file__).read_bytes())
    lock=dict(created_unix=time.time(),source_hashes={p:R.sha(ROOT/p) for p in OWN_SOURCES},
        producer_sources={p:R.sha(ROOT/p) for p in PRODUCER_SOURCES},
        source_snapshot=str(snapshot),source_snapshot_sha256=R.sha(snapshot),
        producer_lock_sha256=R.sha(runs/'lock.json') if (runs/'lock.json').exists() else None,
        before_any_r14_arrays_rows_summary=True,probability_tolerance=1e-4,
        matched_temperature_probability_tolerance=1e-5,region_mean_pixel_tolerance=.005,
        row_absolute_tolerance=1e-8,summary_absolute_tolerance=1e-9,
        independent_integration_scope='first metadata pair per train scene, each g8/g16/g24: 8 evenly spaced query indices, all432keys, all7roles / 5uniquegeometries',
        metric_scope='all1008probabilityrecords/all432pairedrecords; all48pairs, exact audited R13 targets',
        diagnosis_only=True,scientific_gate_passed=False,validation_inputs_read=False,fresh_test_read=False)
    R.write(source_lock,lock)
    print('AUDIT_FROZEN',R.sha(source_lock),flush=True)


def stage_contract(directory,runs,stage,scene):
    value=R.read(directory/'completion.json')
    if value['stage']!=stage or value['scene']!=scene or value['lock_sha256']!=R.sha(runs/'lock.json'):
        raise ValueError('stage identity changed')
    paths={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p!=directory/'completion.json'}
    if paths!=set(value['files']):raise ValueError('unbound stage file')
    R.verify_files(value['files'],directory)
    if value['started_unix']>value['finished_unix']:raise ValueError('invalid stage chronology')
    return value


def audit(runs,out,source_lock):
    if out.exists():raise ValueError('refuse audit overwrite')
    started=time.time();own=R.read(source_lock)
    R.verify_files(own['source_hashes'],ROOT);R.verify_files(own['producer_sources'],ROOT)
    if R.sha(own['source_snapshot'])!=own['source_snapshot_sha256'] or R.sha(__file__)!=own['source_snapshot_sha256']:
        raise ValueError('audit snapshot mismatch')
    lock=R.read(runs/'lock.json');complete=R.read(runs/'completion.json');digest=R.sha(runs/'lock.json')
    if digest!=(runs/'lock.sha256').read_text().strip() or complete['lock_sha256']!=digest:
        raise ValueError('run lock mismatch')
    if own['producer_lock_sha256'] is not None and own['producer_lock_sha256']!=digest:raise ValueError('producer lock changed')
    R.verify_files(lock['source_hashes'],ROOT);R.verify_files(complete['files'],runs)
    actual={str(p.relative_to(runs)) for p in runs.rglob('*') if p.is_file() and p!=runs/'completion.json'}
    if actual!=set(complete['files']):raise ValueError('unbound complete artifact')
    if lock['configurations']!=CONFIGS or lock['contrasts']!=CONTRASTS or lock['pair_count']!=48:
        raise ValueError('changed fixed design')
    if not lock['diagnostic_only'] or not lock['no_parameter_selection'] or lock['fresh_test_read']:
        raise ValueError('changed diagnostic scope')
    R.verify_files(lock['inputs']);R.verify_files(lock['manifests'])
    if R.sha(ROOT/'results/round12_stress/lock.json')!=lock['operator_parent_lock_sha256']:
        raise ValueError('parent operator lock changed')
    if R.sha(runs/'partitions.npz')!=lock['partitions_sha256']:raise ValueError('partition hash changed')
    parts=archive(runs/'partitions.npz')
    for stride in STRIDES:
        q='g'+str(stride);bounds=R.grid(stride)
        np.testing.assert_array_equal(parts[q+'__bounds'],bounds)
        np.testing.assert_allclose(parts[q+'__mean'],centers(stride)/8,atol=0,rtol=0)
        np.testing.assert_array_equal(parts[q+'__mass'],np.full(len(bounds),stride**2))
        yy,xx=np.mgrid[:144,:192];owner=(yy//stride*(192//stride)+xx//stride).ravel()
        np.testing.assert_array_equal(parts[q+'__owner'],owner)
    data=Path(lock['data_root']);dc=R.read(data/'completion.json')
    if R.sha(data/'completion.json')!=lock['data_completion_sha256']:raise ValueError('data completion changed')
    records=[r for r in dc['manifests'] if r['split']=='train']
    if records!=lock['training_manifest_records'] or len(records)!=48:raise ValueError('changed old train population')
    target=lock['target_contract'];target_root=Path(target['root']);tc=R.read(target_root/'completion.json')
    ta=R.read(target['audit_path'])
    for path,expected in ((target_root/'lock.json',target['lock_sha256']),
                         (target_root/'completion.json',target['completion_sha256']),
                         (Path(target['audit_path']),target['audit_sha256'])):
        if R.sha(path)!=expected:raise ValueError('target audit contract changed')
    if not ta['passed'] or ta['target_lock_sha256']!=target['lock_sha256'] or ta['target_completion_sha256']!=target['completion_sha256']:
        raise ValueError('independent target audit mismatch')
    if tc['lock_sha256']!=target['lock_sha256']:raise ValueError('target completion lock mismatch')
    entries=target['entries']
    if len(entries)!=48 or {e['scene'] for e in entries}!=set(SCENES):raise ValueError('incorrect scenes')
    for e in entries:
        meta=R.read(Path(e['path'])/'manifest.json')
        if meta['split']!='train' or meta['forward_uses_sensor'] or (meta['scene'],meta['pair'])!=(e['scene'],e['pair']):
            raise ValueError('wrong forward identity')
        if lock['inputs'][str(Path(e['path'])/'inputs.npz')]!=meta['input_sha256']:raise ValueError('input manifest mismatch')
        expected_path=target_root/'targets'/(e['scene'].replace('/','__')+'__'+Path(e['path']).name+'.npz')
        if Path(e['target_path'])!=expected_path:raise ValueError('target/pair path join mismatch')
        relative=str(Path(e['target_path']).relative_to(target_root))
        if tc['files'][relative]!=target['expected_target_hashes'][e['target_path']]:raise ValueError('target file not audit-bound')
    # Verify every forward shard and its complete file identity before reading any target.
    chain={};evaluation={}
    for s in SCENES:
        directory=runs/'forward'/s.replace('/','__');f=stage_contract(directory,runs,'forward',s)
        if f['targets_decoded'] or f['pair_count']!=16 or f['record_count']!=336:raise ValueError('incorrect forward population')
        expected={Path(e['path']).name+'.npz' for e in entries if e['scene']==s}|{'numerical.json'}
        if set(f['files'])!=expected:raise ValueError('wrong prediction archive set')
        chain[s]=dict(completion_sha256=R.sha(directory/'completion.json'),finished_unix=f['finished_unix'])
    for s in SCENES:
        directory=runs/'evaluation'/s.replace('/','__');e=stage_contract(directory,runs,'evaluate',s)
        if e['forward_chain']!=chain or e['started_unix']<max(x['finished_unix'] for x in chain.values()):raise ValueError('labels before all forward completion')
        if e['record_count']!=336 or e['paired_record_count']!=144:raise ValueError('evaluation population changed')
        expected={x['target_path']:target['expected_target_hashes'][x['target_path']] for x in entries if x['scene']==s}
        if e['label_sources']!=expected:raise ValueError('target population differs')
        evaluation[s]=R.sha(directory/'completion.json')
    R.verify_files(target['expected_target_hashes'])
    stored_rows=R.read(runs/'rows.json');stored_pairs=R.read(runs/'paired_rows.json')
    row_map={(r['scene'],tuple(r['pair']),r['query'],r['role']):r for r in stored_rows}
    pair_map={(r['scene'],tuple(r['pair']),r['query'],r['contrast']):r for r in stored_pairs}
    if len(stored_rows)!=1008 or len(row_map)!=1008 or len(stored_pairs)!=432 or len(pair_map)!=432:
        raise ValueError('duplicate/missing metric rows')
    expected_fields={r+'__g'+str(s)+suffix for r in CONFIGS for s in STRIDES for suffix in ('','__argmax')}
    expected_fields|={'predicted_region_mean__g'+str(s) for s in STRIDES}
    rows=[];paired=[];numeric=[];max_mean=max_t=max_row=0.;numerical_count=0
    for scene in SCENES:
        selected=sorted([e for e in entries if e['scene']==scene],key=lambda e:Path(e['path']).name)
        if len(selected)!=16 or selected[0]['pair']!=[30,45]:raise ValueError('fixed first train identity changed')
        diag=R.read(runs/'forward'/scene.replace('/','__')/'numerical.json')
        expected_diags={(tuple(e['pair']),'g'+str(s),c['method'],c['h']) for e in selected for s in STRIDES for c in CONFIGS.values()}
        found={(tuple(d['pair']),d['query'],d['method'],d['h']) for d in diag}
        if len(diag)!=240 or found!=expected_diags:raise ValueError('numerical geometry population changed')
        for d in diag:
            n=144*192//int(d['query'][1:])**2
            if d['scene']!=scene or d['query_rows']!=n or d['key_count']!=432:raise ValueError('wrong geometry dimensions')
            if not 0<=d['raw_zero_fov_rows']<=d['raw_tiny_fov_rows']<=n or not 0<=d['raw_zero_cell_count']<=n*432:
                raise ValueError('invalid numerical counts')
            if not 0<=d['raw_fov_min']<=d['raw_fov_max']<=1+1e-10:raise ValueError('invalid numerical FOV')
        numerical_count+=len(diag)
        for pair_index,e in enumerate(selected):
            with np.load(Path(e['path'])/'inputs.npz',allow_pickle=False) as z:
                if any('gt' in k or 'sensor' in k for k in z.files):raise ValueError('target field in observation archive')
                inputs={k:z[k] for k in FIELDS}
            saved=archive(runs/'forward'/scene.replace('/','__')/(Path(e['path']).name+'.npz'))
            if set(saved)!=expected_fields:raise ValueError('unexpected prediction fields')
            labels=archive(e['target_path']);pair_mass=[]
            for stride in STRIDES:
                q='g'+str(stride);bounds=R.grid(stride);mean=saved['predicted_region_mean__'+q]
                xy=np.asarray(inputs['q_to_k_xy'],float).reshape(144,192,2)
                independent_mean=np.array([xy[y0:y1,x0:x1].mean((0,1)) for x0,y0,x1,y1 in bounds])
                error=float(np.abs(mean-independent_mean).max());max_mean=max(max_mean,error)
                if error>own['region_mean_pixel_tolerance']:raise ValueError('region mean is not projected area mean')
                content=R.content_reference(inputs,bounds)
                for method in ('full4','full1'):
                    expected=softmax(np.log(saved[method+'_t0__'+q])+5*content,axis=1)
                    error=float(np.abs(expected-saved[method+'_t5__'+q]).max());max_t=max(max_t,error)
                    if error>own['matched_temperature_probability_tolerance']:raise ValueError('matched temperature changes geometry')
                counts=labels[q+'_counts'];valid=labels[q+'_point_visible'];point=labels[q+'_point_xy'];correct={}
                if counts.shape!=(len(bounds),432):raise ValueError('native target shape')
                pair_mass.append(int(counts.sum()))
                for role in CONFIGS:
                    name=role+'__'+q
                    metric,correct[role]=metric_reference(saved[name],saved[name+'__argmax'],mean,counts,point,valid)
                    row=dict(**metric,scene=scene,pair=e['pair'],query=q,role=role,config=CONFIGS[role])
                    error=R.compare(row_map.pop((scene,tuple(e['pair']),q,role)),row,own['row_absolute_tolerance'])
                    max_row=max(max_row,error);rows.append(row)
                for name,(left,right) in CONTRASTS.items():
                    metric=paired_reference(saved[left+'__'+q+'__argmax'],saved[right+'__'+q+'__argmax'],correct[left],correct[right],valid)
                    row=dict(**metric,scene=scene,pair=e['pair'],query=q,contrast=name,left=left,right=right)
                    error=R.compare(pair_map.pop((scene,tuple(e['pair']),q,name)),row,own['row_absolute_tolerance'])
                    max_row=max(max_row,error);paired.append(row)
                if pair_index==0:
                    indices=np.linspace(0,len(bounds)-1,8,dtype=int)
                    ref=reference_probabilities(inputs,stride,indices)
                    for role,p in ref.items():
                        error=float(np.abs(saved[role+'__'+q][indices]-p).max())
                        if error>own['probability_tolerance']:raise ValueError('independent SciPy probability mismatch')
                        numeric.append(dict(scene=scene,pair=e['pair'],query=q,role=role,indices=indices.tolist(),probability_error=error))
            if len(set(pair_mass))!=1:raise ValueError('visible mass changes with query partition')
        for name,all_rows in (('rows.json',stored_rows),('paired_rows.json',stored_pairs)):
            shard=R.read(runs/'evaluation'/scene.replace('/','__')/name)
            R.compare(shard,[r for r in all_rows if r['scene']==scene],0)
        print('INDEPENDENT_TEMPERATURE_SCENE',scene,'metric_rows',len(rows),flush=True)
    if row_map or pair_map:raise ValueError('unused metric population')
    summary=R.read(runs/'summary.json');reference=summary_reference(rows,paired)
    summary_error=R.compare(summary['summary'],reference,own['summary_absolute_tolerance'])
    for field,expected in dict(configurations=CONFIGS,contrasts=CONTRASTS,pair_count=48,probability_records=1008,
        paired_records=432,lock_sha256=digest,source_hashes=lock['source_hashes'],forward_chain=chain,
        evaluation_shards=evaluation,diagnostic_only=True,no_parameter_selection=True,fresh_test_read=False,
        validation_gate_evaluated=False).items():R.compare(summary[field],expected)
    if complete['probability_records']!=1008 or not complete['diagnostic_only'] or complete['fresh_test_read']:
        raise ValueError('incorrect final scope')
    out.mkdir(parents=True)
    result=dict(passed=True,diagnosis_only=True,scientific_gate_passed=False,
        temperature_lock_sha256=digest,temperature_summary_sha256=R.sha(runs/'summary.json'),
        temperature_completion_sha256=R.sha(runs/'completion.json'),audit_source_lock_sha256=R.sha(source_lock),
        probability_records=len(rows),paired_records=len(paired),numerical_records=numerical_count,
        independent_integral_probability_matrices=len(numeric),independent_geometry_matrices=45,
        maximum_integral_probability_error=max(x['probability_error'] for x in numeric),
        maximum_matched_temperature_error=max_t,maximum_projected_region_mean_error_px=max_mean,
        maximum_row_metric_error=max_row,maximum_summary_error=summary_error,numerical=numeric,
        reused_target_audit_sha256=target['audit_sha256'],sensor_files_decoded=False,old_train_targets_read=True,
        validation_inputs_read=False,fresh_test_read=False,seconds=time.time()-started,
        interpretation='Independent implementation/artifact audit of fixed TRAIN contrasts only. No candidate selected, no validation gate passed, and no recovery claim for the R12 validation deficit. Region mean is not the projected sensor center; legacy clipped KL is retained unchanged.')
    R.write(out/'verification.json',result)
    print({k:v for k,v in result.items() if k!='numerical'},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','audit'))
    p.add_argument('--runs',type=Path,default=ROOT/'results/round14_temperature')
    p.add_argument('--out',type=Path,default=ROOT/'results/round14_temperature_independent_audit')
    p.add_argument('--source-lock',type=Path,default=ROOT/'results/round14_temperature_audit_source_lock.json')
    args=p.parse_args()
    if args.stage=='freeze':freeze(args.source_lock,args.runs)
    else:audit(args.runs,args.out,args.source_lock)


if __name__=='__main__':main()
