import copy
import itertools
import json

import numpy as np
import pytest

from experiments.round7_depth_analysis import (
    ROLES,SEEDS,ROOT,aggregate_block,analyze_rows,draw_indices,load_file,
    paired_statistics,sha,validate_lock,validate_row_block,
    validate_evaluation_metadata,validate_test_data_lock,value_interventions,
)


SCENES=tuple(f'place{i}/scan' for i in range(5))


def row(scene,index,factor=1.,count=100):
    b=min(5,count)
    return dict(scene=scene,q_index=index*2,k_index=index*2+1,pair_id=f'{index*2}:{index*2+1}',
        count=count,boundary_count=b,absrel_sum=.2*factor*count,abs_sum=factor*count,
        sq_sum=factor**2*count,log_sq_sum=.1*factor**2*count,delta1_count=int(.8*count),
        boundary_absrel_sum=.2*factor*b,boundary_abs_sum=factor*b)


def blocks(full=.96,single=1.1):
    factors=dict(full=full,point=1.,point_pck=1.01,fixed=1.02,isotropic=.9,singleview=single)
    return {role:{seed:[row(scene,i,factors[role]) for scene in SCENES for i in range(2)] for seed in SEEDS} for role in ROLES}


def analyze(data,**kwargs):
    return analyze_rows(data,'point',SCENES,pairs_per_scene=2,draws=2000,**kwargs)


def test_primary_gate_all_controls_and_deterministic_raw_baseline():
    result=analyze(blocks(),raw_rows=[row(s,i,1.2) for s in SCENES for i in range(2)])
    assert result['primary_gate']['passed']
    assert result['primary_gate']['AbsRel']['relative_difference']==pytest.approx(-.04)
    assert result['primary_gate']['AbsRel']['improved_seed_count']==5
    assert set(result['comparisons'])==set(ROLES)-{'full'}
    assert result['comparisons']['isotropic']['AbsRel']['difference']>0
    assert result['raw_baseline']['descriptive_only']
    assert result['raw_baseline']['independently_trained_seed_replicates']==0


def test_practical_improvement_and_singleview_failure_retained():
    result=analyze(blocks(full=.999))
    assert not result['primary_gate']['passed']
    assert not result['primary_gate']['criteria']['relative_AbsRel_improvement_at_least_2_percent']
    result=analyze(blocks(single=.94))
    assert not result['primary_gate']['passed']
    assert not result['primary_gate']['criteria']['better_than_singleview_AbsRel_CI_upper_below_zero']


def test_rmse_guard_uses_relative_bootstrap_upper_bound():
    data=blocks()
    for rows in data['full'].values():
        for r in rows:r['sq_sum']=r['count']*1.08**2
    result=analyze(data)
    assert result['primary_gate']['AbsRel']['relative_difference']<-.02
    assert result['primary_gate']['RMSE']['ci95_relative_difference']==pytest.approx([.08,.08])
    assert not result['primary_gate']['passed']


def test_four_of_five_seed_criterion_is_separate():
    data=blocks()
    for i,seed in enumerate(SEEDS):
        data['full'][seed]=[row(s,j,.8 if i<3 else 1.05) for s in SCENES for j in range(2)]
    result=analyze(data)
    assert result['primary_gate']['AbsRel']['improved_seed_count']==3
    assert not result['primary_gate']['criteria']['at_least_4_of_5_seed_macros_improve']


def test_row_role_seed_order_does_not_change_estimate_or_bootstrap():
    data=blocks();expected=analyze(data)
    shuffled={role:{seed:list(reversed(data[role][seed])) for seed in reversed(SEEDS)} for role in reversed(ROLES)}
    assert expected==analyze(shuffled)


def test_scan_macro_and_rmse_denominators_are_not_micro_or_root_of_macro_mse():
    rows=[]
    for i,s in enumerate(SCENES):
        r=row(s,0,count=100 if i==0 else 1)
        r['sq_sum']=r['count']*(4 if i==0 else 1)
        rows.append(r)
    got=aggregate_block(rows,SCENES)
    assert got['metrics']['RMSE']==pytest.approx(1.2)
    assert got['metrics']['RMSE']!=pytest.approx(np.sqrt(1.6))
    weighted=[row(SCENES[0],0,.5,count=100),row(SCENES[0],1,4.5,count=1)]
    got=aggregate_block(weighted,[SCENES[0]])
    assert got['metrics']['AbsRel']==pytest.approx((10+.9)/101)


def test_paired_crossed_bootstrap_matches_finite_distribution_and_ratio_of_means():
    b=np.ones((2,2));a=b+np.array([[0.,0.],[0.,.04]])
    outcomes=[]
    for sd in itertools.product(range(2),repeat=2):
        for pd in itertools.product(range(2),repeat=2):
            outcomes.append(np.mean([a[s,p]-b[s,p] for s in sd for p in pd]))
    got=paired_statistics(a,b,[11,22],['a','b'],draw_indices(2,2,20000))
    np.testing.assert_allclose(got['ci95_difference'],np.quantile(outcomes,[.025,.975],method='inverted_cdf'))
    a=np.array([[2.,2.],[1.,1.]]);b=np.array([[1.,1.],[100.,100.]])
    got=paired_statistics(a,b,[11,22],['a','b'],draw_indices(2,2,1000))
    assert got['relative_difference']==pytest.approx((1.5-50.5)/50.5)
    assert got['relative_difference']!=pytest.approx(np.mean((a-b)/b))


def test_missing_roles_seeds_duplicate_identity_and_population_changes_rejected():
    data=blocks();del data['fixed'][55]
    with pytest.raises(ValueError,match='seed matrix'):analyze(data)
    data=blocks();del data['singleview']
    with pytest.raises(ValueError,match='role matrix'):analyze(data)
    data=blocks();data['fixed'][11][0]=copy.deepcopy(data['fixed'][11][1])
    with pytest.raises(ValueError,match='duplicate'):analyze(data)
    data=blocks();data['fixed'][11][0]['count']+=1
    with pytest.raises(ValueError,match='populations differ'):analyze(data)
    data=blocks();data['fixed'][11][0]['boundary_count']-=1
    with pytest.raises(ValueError,match='populations differ'):analyze(data)


def test_zero_populations_and_raw_population_are_checked():
    r=row(SCENES[0],0,count=0);r['sq_sum']=1
    with pytest.raises(ValueError,match='zero-valid'):validate_row_block([r],[SCENES[0]],pairs_per_scene=1)
    raw=[row(s,i) for s in SCENES for i in range(2)];raw[0]['count']+=1
    with pytest.raises(ValueError,match='raw baseline population'):analyze(blocks(),raw_rows=raw)
    data=blocks()
    for role in ROLES:
        for seed in SEEDS:
            for r in data[role][seed]:
                if r['scene']==SCENES[0]:
                    r.update(boundary_count=0,boundary_absrel_sum=0,boundary_abs_sum=0)
    result=analyze(data)
    assert result['primary_gate']['passed']
    assert not result['comparisons']['point']['boundary_AbsRel']['available']
    assert result['comparisons']['point']['boundary_AbsRel']['ci95_difference'] is None


def locked_fixture(tmp_path):
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(scenes=[dict(scene=s,split='test',pairs=[[i*2,i*2+1] for i in range(96)]) for s in SCENES])))
    source=tmp_path/'sealed.py';source.write_text('sealed=true\n')
    sources={str(source):sha(source),'experiments/round7_depth_analysis.py':sha(ROOT/'experiments/round7_depth_analysis.py'),
             'ROUND7_PROTOCOL.md':sha(ROOT/'ROUND7_PROTOCOL.md')}
    lock=dict(test_export_authorized=True,seeds=list(SEEDS),expected_scenes=list(SCENES),baseline_role='point',
        selection_path=str(selection),selection_sha256=sha(selection),source_hashes=sources,roles={})
    for role in ROLES:
        entries={}
        factor=dict(point=1.,point_pck=1.1,fixed=1.2,full=.98,isotropic=1.05,singleview=1.3)[role]
        for seed in SEEDS:
            folder=tmp_path/role/str(seed);folder.mkdir(parents=True)
            checkpoint=folder/'best.pt';checkpoint.write_bytes(b'checkpoint')
            rows=[row(s,i,factor) for s in ('val0/scan','val1/scan') for i in range(64)]
            summary=aggregate_block(rows,['val0/scan','val1/scan'])
            validation=folder/'validation.json';validation.write_text(json.dumps(dict(rows=rows,summary=summary)))
            score=summary['metrics']['AbsRel']
            run=dict(role=role,seed=seed,steps=1000,best_step=1000,parameter_count=500,
                source_hashes={str(source):sha(source)},batch_sequence_sha256=str(seed),
                validation_sha256=sha(validation),checkpoint_sha256=sha(checkpoint),best_validation_AbsRel=score,
                history=[dict(step=step,validation=dict(metrics=dict(AbsRel=score+(1000-step)*1e-5))) for step in range(0,1001,100)])
            runfile=folder/'run.json';runfile.write_text(json.dumps(run))
            entries[str(seed)]=dict(checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                validation_path=str(validation),validation_sha256=sha(validation),run_sha256=sha(runfile),best_step=1000)
        lock['roles'][role]=dict(seeds=entries)
    path=tmp_path/'lock.json';path.write_text(json.dumps(lock));return path,lock


def test_lock_checks_artifacts_validation_selection_and_earliest_checkpoint(tmp_path):
    path,lock=locked_fixture(tmp_path)
    _,pairs,verification=validate_lock(path)
    assert len(pairs)==480 and verification['baseline_role']=='point'
    lock['baseline_role']='fixed';path.write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='validation-selected'):validate_lock(path)
    lock['baseline_role']='point'
    entry=lock['roles']['full']['seeds']['11'];run_path=tmp_path/'full/11/run.json'
    run=json.loads(run_path.read_text());run['history'][0]['validation']['metrics']['AbsRel']=0.
    run_path.write_text(json.dumps(run));entry['run_sha256']=sha(run_path);path.write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='earliest'):validate_lock(path)


def test_lock_rejects_changed_checkpoint_and_wrapper_schema(tmp_path):
    path,lock=locked_fixture(tmp_path)
    checkpoint=tmp_path/'full/11/best.pt';checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):validate_lock(path)
    malformed=tmp_path/'bad.json';malformed.write_text(json.dumps([]))
    with pytest.raises(ValueError,match=r'rows\+summary'):load_file(malformed)


def test_prediction_metadata_binds_role_seed_checkpoint_both_locks_and_actual_array(tmp_path):
    path=tmp_path/'test.json';prediction=tmp_path/'predictions.npz'
    np.savez_compressed(prediction,depth=np.ones((2,432),np.float32))
    metadata=dict(role='full',seed=11,checkpoint_sha256='checkpoint',lock_sha256='lock',
        test_data_lock_sha256='data',prediction_sha256=sha(prediction))
    obj=dict(metadata=metadata)
    checked=validate_evaluation_metadata(obj,path,'lock','data','full',11,'checkpoint')
    assert checked['prediction_sha256']==sha(prediction)
    for field in metadata:
        changed=dict(obj,metadata=dict(metadata,**{field:'wrong'}))
        with pytest.raises(ValueError):validate_evaluation_metadata(changed,path,'lock','data','full',11,'checkpoint')
    raw=dict(metadata=dict(metadata,role='raw'))
    validate_evaluation_metadata(raw,path,'lock','data','raw')
    prediction.write_bytes(b'changed')
    with pytest.raises(ValueError,match='prediction archive'):validate_evaluation_metadata(raw,path,'lock','data','raw')


def test_test_data_lock_verifies_actual_archives_and_full_scene_pair_population(tmp_path):
    sources=[];expected=set()
    for scene in SCENES:
        folder=tmp_path/scene.replace('/','_');folder.mkdir()
        (folder/'inputs.npz').write_bytes(b'inputs');(folder/'labels.npz').write_bytes(b'labels')
        pairs=[[i*2,i*2+1] for i in range(96)]
        expected.update((scene,q,k) for q,k in pairs)
        manifest=dict(scene=scene,split='test',pairs=pairs,input_sha256=sha(folder/'inputs.npz'),label_sha256=sha(folder/'labels.npz'))
        (folder/'manifest.json').write_text(json.dumps(manifest))
        sources.append(dict(path=str(folder),manifest_sha256=sha(folder/'manifest.json'),
            input_sha256=manifest['input_sha256'],label_sha256=manifest['label_sha256']))
    path=tmp_path/'test_data_lock.json'
    value=dict(confirmation_lock_sha256='lock',pair_count=480,total_positions=480*432,
        valid_count=48000,boundary_count=2400,sources=sources)
    path.write_text(json.dumps(value))
    _,digest=validate_test_data_lock(path,'lock',expected,SCENES)
    assert digest==sha(path)
    with pytest.raises(ValueError,match='different confirmation'):validate_test_data_lock(path,'other',expected,SCENES)
    with pytest.raises(ValueError,match='identities differ'):validate_test_data_lock(path,'lock',set(list(expected)[1:]),SCENES)
    (tmp_path/SCENES[0].replace('/','_')/'labels.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source hash changed'):validate_test_data_lock(path,'lock',expected,SCENES)


def test_value_interventions_preserve_reverse_improvement_without_new_selection():
    records={}
    for seed in SEEDS:
        records[str(seed)]={}
        for mode,factor in [('zero_v',1.1),('reverse_v',.9)]:
            rows=[row(s,i,factor) for s in SCENES for i in range(2)]
            records[str(seed)][mode]=dict(summary=aggregate_block(rows,SCENES),mean_absolute_output_change_m=.1)
    result=value_interventions(records,SCENES)
    assert result['descriptive_only']
    assert result['seed_mean']['reverse_v']['metrics']['AbsRel']==pytest.approx(.18)
    assert result['seed_mean']['zero_v']['metrics']['AbsRel']==pytest.approx(.22)
