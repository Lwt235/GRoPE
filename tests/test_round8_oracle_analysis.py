import copy
import json

import numpy as np
import pytest

from experiments.round8_oracle_analysis import (
    ROLES, SCENES, SEEDS, analyze_directory, analyze_predictions, compare_tree,
    inspect_training_metadata, load_validation_sources, numpy_macro, numpy_rows,
    paired_description, population_masks, sha,
)


def fixture_arrays(factors=None):
    factors = factors or dict(pred_pred=1.10, gt_route=1.08, gt_value=1.06,
                              gt_both=1.02, singleview=1.10)
    ids = [dict(scene=scene,q_index=i*2,k_index=i*2+1,pair_id=f'{i*2}:{i*2+1}')
           for scene in SCENES for i in range(2)]
    labels = dict(target=np.ones((4,4)), valid=np.ones((4,4),bool),
        boundary=np.tile([True,False,False,False],(4,1)),
        oracle_covisible=np.tile([True,True,False,False],(4,1)),
        gt_key_coverage=np.tile([.25,1.,0.,0.],(4,1)))
    predictions = {role:{seed:np.full((4,4),factors[role]) for seed in SEEDS} for role in ROLES}
    return predictions,labels,ids


def test_numpy_metrics_select_before_arithmetic_and_strict_delta1():
    ids=[dict(scene=SCENES[0],q_index=0,k_index=1,pair_id='0:1')]
    rows=numpy_rows([[1.25,.8,1.]],[[1.,1.,np.nan]],[[True,True,False]],[[True,False,True]],ids)
    assert rows[0]['count']==2 and rows[0]['boundary_count']==1
    assert rows[0]['delta1_count']==0
    assert rows[0]['absrel_sum']==pytest.approx(.45)
    assert rows[0]['sq_sum']==pytest.approx(.1025)
    assert rows[0]['boundary_abs_sum']==pytest.approx(.25)
    with pytest.raises(ValueError,match='finite and positive'):
        numpy_rows([[1.,1.,np.nan]],[[1.,1.,np.nan]],[[True,True,False]],[[False]*3],ids)


def test_macro_is_scan_equal_within_scan_count_weighted_and_root_before_macro():
    _,labels,ids=fixture_arrays()
    labels['valid'][1,1:]=False
    pred=np.array([[2.]*4,[5.]*4,[3.]*4,[3.]*4])
    rows=numpy_rows(pred,labels['target'],labels['valid'],labels['boundary'],ids)
    result=numpy_macro(rows,4)
    assert result['by_scene'][SCENES[0]]['MAE']==pytest.approx(8/5)
    assert result['metrics']['MAE']==pytest.approx(1.8)
    assert result['metrics']['RMSE']==pytest.approx((np.sqrt(4)+2)/2)
    assert result['total_positions']==16


def test_empty_subset_is_missing_and_not_artificially_perfect():
    pred,labels,ids=fixture_arrays()
    labels['oracle_covisible'][:2]=False
    result,_=analyze_predictions(pred,labels,ids)
    subset=result['per_role_stratum_seed']['gt_route']['oracle_covisible']['11']
    assert subset['by_scene'][SCENES[0]]['AbsRel'] is None
    assert subset['metrics']['AbsRel']==pytest.approx(.08)
    assert not result['comparisons']['oracle_covisible']['gt_route_minus_singleview']['AbsRel']['complete_seed_scene_matrix']


def test_population_partition_and_key_coverage_never_include_fallback():
    _,labels,_=fixture_arrays()
    masks=population_masks(labels)
    np.testing.assert_array_equal(masks['oracle_covisible']|masks['oracle_fallback'],masks['all_query'])
    assert not np.any(masks['oracle_covisible']&masks['oracle_fallback'])
    strata=[mask for name,mask in masks.items() if name.startswith('key_coverage_')]
    np.testing.assert_array_equal(np.sum(strata,axis=0),masks['oracle_covisible'].astype(int))
    assert masks['key_coverage_0.25_0.5'].sum()==4
    assert masks['key_coverage_0.75_1'].sum()==4
    assert masks['key_coverage_0_0.25'].sum()==0
    labels['valid'][0,0]=False
    with pytest.raises(ValueError,match='invalid query'):
        population_masks(labels)


def test_primary_and_diagnostic_gates_and_factorial_are_paired():
    predictions,labels,ids=fixture_arrays()
    result,_=analyze_predictions(predictions,labels,ids)
    assert result['primary_gate']['passed']
    assert result['diagnostic_gates']['gt_value']['passed']
    assert result['diagnostic_gates']['gt_both']['passed']
    assert result['primary_gate']['comparison']['relative_difference']==pytest.approx(-.2)
    assert result['primary_gate']['comparison']['ci95'] is None
    effects=result['factorial']['all_query']['AbsRel']
    assert effects['route_at_predicted_value']['mean']==pytest.approx(-.02)
    assert effects['route_at_sensor_value']['mean']==pytest.approx(-.04)
    assert effects['value_at_predicted_route']['mean']==pytest.approx(-.04)
    assert effects['value_at_sensor_route']['mean']==pytest.approx(-.06)
    assert effects['interaction']['mean']==pytest.approx(-.02)
    assert effects['mean_route_effect']['mean']==pytest.approx(-.03)


def test_primary_gate_requires_both_scenes_and_four_seeds_separately():
    predictions,labels,ids=fixture_arrays()
    for seed in SEEDS:predictions['gt_route'][seed][2:]=1.11
    result,_=analyze_predictions(predictions,labels,ids)
    assert not result['primary_gate']['passed']
    assert not result['primary_gate']['criteria']['both_validation_scene_seed_means_improve']
    predictions,labels,ids=fixture_arrays()
    for seed in SEEDS[3:]:predictions['gt_route'][seed][:]=1.11
    result,_=analyze_predictions(predictions,labels,ids)
    assert not result['primary_gate']['passed']
    assert result['primary_gate']['comparison']['improved_seed_count']==3


def test_factorial_retains_harm_and_gate_does_not_select_best_oracle_role():
    predictions,labels,ids=fixture_arrays(dict(pred_pred=1.1,gt_route=1.101,
        gt_value=1.01,gt_both=1.005,singleview=1.1))
    result,_=analyze_predictions(predictions,labels,ids)
    assert not result['primary_gate']['passed']
    assert result['diagnostic_gates']['gt_value']['passed']
    assert result['diagnostic_gates']['gt_both']['passed']
    assert result['factorial']['all_query']['AbsRel']['route_at_predicted_value']['mean']>0


def test_missing_roles_seeds_wrong_scene_duplicate_ids_rejected():
    predictions,labels,ids=fixture_arrays()
    del predictions['gt_value'][55]
    with pytest.raises(ValueError,match='seed matrix'):analyze_predictions(predictions,labels,ids)
    predictions,labels,ids=fixture_arrays();del predictions['gt_value']
    with pytest.raises(ValueError,match='role matrix'):analyze_predictions(predictions,labels,ids)
    predictions,labels,ids=fixture_arrays();ids[0]=ids[1]
    with pytest.raises(ValueError,match='duplicate'):analyze_predictions(predictions,labels,ids)
    predictions,labels,ids=fixture_arrays();ids[0]=dict(ids[0],scene='fresh/scan')
    with pytest.raises(ValueError,match='scene population'):analyze_predictions(predictions,labels,ids)


def test_ratio_is_difference_of_means_over_control_mean_not_mean_ratios():
    a=np.tile([2.,1.],(5,1));b=np.tile([1.,100.],(5,1))
    result=paired_description(a,b)
    assert result['relative_difference']==pytest.approx((1.5-50.5)/50.5)
    assert result['relative_difference']!=pytest.approx(((2-1)/1+(1-100)/100)/2)


def test_delta1_improvement_direction_is_positive():
    a=np.full((5,2),.9);b=np.full((5,2),.8)
    result=paired_description(a,b,higher_is_better=True)
    assert result['improved_seed_count']==5 and result['improved_scene_count']==2
    assert result['higher_is_better']


def test_metric_comparison_rejects_schema_and_nonfinite():
    assert compare_tree(dict(a=[1.,None]),dict(a=[1.+1e-10,None]))['passed']
    assert not compare_tree(dict(a=1.),dict(a=float('nan')))['passed']
    assert not compare_tree(dict(a=1.),dict(b=1.))['passed']
    assert not compare_tree(dict(a=1.),dict(a=1.001))['passed']


def disk_sources(tmp_path):
    base_sources=[];oracle_sources=[]
    for scene in SCENES:
        slug=scene.replace('/','__');base=tmp_path/'base'/'val'/slug;oracle=tmp_path/'oracle'/'val'/slug
        base.mkdir(parents=True);oracle.mkdir(parents=True)
        pairs=[[2*i,2*i+1] for i in range(64)]
        np.savez_compressed(base/'inputs.npz',q=np.zeros(1))
        np.savez_compressed(base/'labels.npz',target=np.ones((64,432)),valid=np.ones((64,432),bool),boundary=np.zeros((64,432),bool))
        manifest=dict(scene=scene,split='val',pairs=pairs,input_sha256=sha(base/'inputs.npz'),label_sha256=sha(base/'labels.npz'))
        (base/'manifest.json').write_text(json.dumps(manifest))
        base_sources.append(dict(path=str(base),manifest_sha256=sha(base/'manifest.json'),input_sha256=manifest['input_sha256'],label_sha256=manifest['label_sha256']))
        weights=np.zeros((64,432,4));weights[...,0]=1
        np.savez_compressed(oracle/'oracle.npz',oracle_valid=np.ones((64,432),bool),oracle_indices=np.zeros((64,432,4),np.int64),
            oracle_weights=weights,k_sensor_fraction=np.full((64,432),.5))
        om=dict(scene=scene,split='val',pairs=pairs,oracle_sha256=sha(oracle/'oracle.npz'))
        (oracle/'manifest.json').write_text(json.dumps(om))
        oracle_sources.append(dict(path=str(oracle),manifest_sha256=sha(oracle/'manifest.json'),oracle_sha256=om['oracle_sha256']))
    return base_sources,oracle_sources


def test_loader_binds_original_labels_and_separate_oracle_arrays(tmp_path):
    base,oracle=disk_sources(tmp_path)
    arrays,ids,records=load_validation_sources(base,oracle)
    assert arrays['target'].shape==(128,432) and len(ids)==128
    np.testing.assert_allclose(arrays['gt_key_coverage'],.5)
    assert arrays['oracle_covisible'].all()
    assert len(records['base'])==len(records['oracle'])==2
    reordered,reordered_ids,_=load_validation_sources(base,list(reversed(oracle)))
    assert reordered_ids==ids
    np.testing.assert_array_equal(reordered['oracle_covisible'],arrays['oracle_covisible'])
    np.testing.assert_array_equal(reordered['gt_key_coverage'],arrays['gt_key_coverage'])


def test_loader_rejects_test_before_reading_arrays_and_hash_changes(tmp_path):
    with pytest.raises(ValueError,match='old validation'):
        load_validation_sources([dict(path=str(tmp_path/'test'/'scene'))],[])
    base,oracle=disk_sources(tmp_path)
    with open(base[0]['path']+'/labels.npz','ab') as file:file.write(b'changed')
    with pytest.raises(ValueError,match='array hash'):
        load_validation_sources(base,oracle)


def metadata_fixture(tmp_path):
    for filename in ('best.pt','validation.json','val_predictions.npz'):
        (tmp_path/filename).write_text(filename)
    source=tmp_path/'source.py';source.write_text('source\n')
    sources=dict(train=[],val=[],oracle_train=[],oracle_val=[])
    m=dict(role='pred_pred',seed=11,steps=1000,best_step=200,best_validation_AbsRel=.2,
        history=[dict(step=step,validation=dict(metrics=dict(AbsRel=.2 if step>=200 else .3))) for step in range(0,1001,100)],
        data_sources=sources,source_hashes={str(source):sha(source)},batch_sequence_sha256='batch',parameter_count=100,
        checkpoint_sha256=sha(tmp_path/'best.pt'),validation_sha256=sha(tmp_path/'validation.json'),prediction_sha256=sha(tmp_path/'val_predictions.npz'))
    return m,sources


def test_training_audit_checks_paired_batches_and_earliest_minimum(tmp_path):
    m,sources=metadata_fixture(tmp_path)
    batches={};counts=set()
    inspect_training_metadata(m,'pred_pred',11,tmp_path,sources,batches,counts)
    assert counts=={100} and batches=={11:'batch'}
    changed=copy.deepcopy(m);changed['best_step']=300
    with pytest.raises(ValueError,match='earliest minimum'):
        inspect_training_metadata(changed,'pred_pred',11,tmp_path,sources,batches,counts)
    changed=copy.deepcopy(m);changed['batch_sequence_sha256']='other'
    with pytest.raises(ValueError,match='batch sequences'):
        inspect_training_metadata(changed,'pred_pred',11,tmp_path,sources,batches,counts)
    changed=copy.deepcopy(m);changed['steps']=500
    with pytest.raises(ValueError,match='budget'):
        inspect_training_metadata(changed,'pred_pred',11,tmp_path,sources,batches,counts)


def test_complete_artifact_analysis_independently_recomputes_all_predictions(tmp_path):
    base,oracle=disk_sources(tmp_path/'data')
    labels,ids,_=load_validation_sources(base,oracle)
    sources=dict(train=[],val=base,oracle_train=[],oracle_val=oracle)
    source=tmp_path/'model_source.py';source.write_text('frozen\n')
    development=tmp_path/'development'
    for role in ROLES:
        for seed in SEEDS:
            run=development/role/str(seed);run.mkdir(parents=True)
            factor=dict(pred_pred=1.1,gt_route=1.08,gt_value=1.06,gt_both=1.02,singleview=1.1)[role]
            prediction=np.full((128,432),factor)
            rows=numpy_rows(prediction,labels['target'],labels['valid'],labels['boundary'],ids)
            summary=numpy_macro(rows)
            score=summary['metrics']['AbsRel']
            (run/'best.pt').write_bytes(b'checkpoint')
            (run/'validation.json').write_text(json.dumps(dict(rows=rows,summary=summary)))
            np.savez_compressed(run/'val_predictions.npz',depth=prediction)
            metadata=dict(role=role,seed=seed,steps=1000,best_step=0,best_validation_AbsRel=score,
                history=[dict(step=step,validation=dict(metrics=dict(AbsRel=score))) for step in range(0,1001,100)],
                data_sources=sources,source_hashes={str(source):sha(source)},batch_sequence_sha256=str(seed),parameter_count=100,
                checkpoint_sha256=sha(run/'best.pt'),validation_sha256=sha(run/'validation.json'),prediction_sha256=sha(run/'val_predictions.npz'))
            (run/'run.json').write_text(json.dumps(metadata))
    result=analyze_directory(development,tmp_path/'analysis')
    assert result['primary_gate']['passed']
    assert result['verification']['independently_recomputed_prediction_archives']==25
    assert result['verification']['max_metric_absolute_error']==0
    assert (tmp_path/'analysis'/'README.md').exists()
