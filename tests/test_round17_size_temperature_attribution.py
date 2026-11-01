"""Synthetic-only tests for the unrun native-size attribution draft."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
import torch
from models import round17_size_temperature_pe as M
from experiments import round17_size_temperature_attribution as A


def test_native_rectangle_trace_and_exact_float32_analytic_temperatures():
    bounds=torch.tensor([[0,0,8,8],[0,0,16,16],[0,0,24,24],[0,0,16,8],[0,0,8,16]],dtype=torch.int64)
    np.testing.assert_allclose(M.native_trace(bounds).numpy(),[1/6,2/3,1.5,5/12,5/12],rtol=0,atol=1e-16)
    expected=np.array([30/11,6/7,.4,30/23,30/23],np.float32)
    assert M.size_temperature(bounds).numpy().tobytes()==expected.tobytes()
    # Same area is not the covariance trace of an anisotropic rectangle.
    area_tau=np.float32(30/19)
    assert M.size_temperature(bounds)[3].item()!=area_tau


def test_bounds_contract_rejects_floats_degenerate_and_nonfinite_parameters():
    good=torch.tensor([[0,0,8,8]],dtype=torch.int64)
    for bad in (good.float(),good.int(),torch.tensor([[0,0,0,8]]),torch.ones((2,3),dtype=torch.int64)):
        with pytest.raises(ValueError):M.native_trace(bad)
    for h,t in ((0.,10.),(.25,-1.),(float('inf'),10.),(.25,float('nan'))):
        with pytest.raises(ValueError):M.size_temperature(good,h,t)


def test_only_bias_changes_and_t0_exact_with_float32_bias_order():
    bounds=torch.tensor([[0,0,8,8],[0,0,16,8]],dtype=torch.int64)
    geometry=torch.tensor([[-1.,-2.,-3.],[-3.,-.1,-.7]],dtype=torch.float64)
    content=torch.tensor([[.1,.8,-.3],[-.2,.1234567,.9876543]],dtype=torch.float32)
    before=geometry.clone(); tau=M.size_temperature(bounds)
    expected=(geometry+(tau[:,None]*content).double()).softmax(-1)
    assert torch.equal(M.probabilities(geometry,content,bounds),expected)
    assert torch.equal(geometry,before)
    assert torch.equal(M.probabilities(geometry,content,bounds,.25,0.),geometry.softmax(-1))
    # Native footprint remains nonzero even if geometry hypothetically is a point.
    assert bool((tau<10.).all())
    with pytest.raises(ValueError):M.probabilities(geometry,content.double(),bounds)


def test_original_squared_pck_boundary_and_invalid_harm_benefit_conservation():
    key_bounds=np.array([[0,0,1,1],[16,0,17,1],[32,0,33,1]])
    primary=np.eye(3)[[0,0,2,0]].astype(np.float64)
    size=np.eye(3)[[1,2,0,2]].astype(np.float64)
    target=np.array([[8.,0.],[32.,0.],[32.,0.],[np.nan,np.nan]])
    valid=np.array([True,True,True,False])
    ac,bc,result=A.center_pair_metrics(primary,size,key_bounds,target,valid)
    assert ac.tolist()==[True,False,True,False] and bc.tolist()==[True,True,False,False]
    assert result['n_point']==3 and result['n_query']==4
    assert result['primary_correct']==result['size_correct']==2
    assert result['size_harm_count']==result['size_benefit_count']==1
    assert result['flip_all_count']==4 and result['flip_valid_count']==3


def test_all_fixed_role_configuration_and_condition_population_required():
    counts=dict(g8=432,g16=108,mixed=243,g24=48)
    entries=[dict(scene=scene,pair=[i,i+1],input_sha256='a'*64) for scene in A.SCENES for i in range(16)]
    rows=[dict(scene=e['scene'],pair=e['pair'],role=role,config=A.CONFIG,**c,
        shape=[counts[c['query']],counts[c['key']]],dtype='float64',input_sha256=e['input_sha256'])
        for e in entries for c in A.CONDITIONS for role in A.ROLES]
    A.population(rows,entries)
    for changed in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError,match='population'):A.population(changed,entries)
    changed=deepcopy(rows);changed[0]['config']=dict(h=.5,temperature=10.)
    with pytest.raises(ValueError,match='configuration'):A.population(changed,entries)


def test_labels_cannot_open_before_global_forward_seal(monkeypatch,tmp_path):
    monkeypatch.setattr(A,'no_active_validation_benchmark',lambda:None)
    monkeypatch.setattr(A,'verify_lock',lambda out:{})
    def unsealed(*args):raise ValueError('no global seal')
    monkeypatch.setattr(A,'require_seal',unsealed)
    monkeypatch.setattr(A,'load_labels',lambda *args:pytest.fail('premature label read'))
    with pytest.raises(ValueError,match='global seal'):A.evaluate(tmp_path,A.SCENES[0])
    assert not (tmp_path/'evaluation').exists()


def test_paired_macro_is_equal_scene_and_exact_not_population_pooled():
    rows=[]
    for scene,n,p,s in zip(A.SCENES,(1,2,4),(1,0,0),(0,1,1)):
        for i in range(16):
            rows.append(dict(scene=scene,pair=[i,i+1],condition='g8_g8',n_query=4,n_point=n,
                primary_correct=p,size_correct=s,size_benefit_count=max(s-p,0),size_harm_count=max(p-s,0),
                flip_all_count=1,flip_valid_count=1))
    result=A.aggregate_pairs(rows)['g8_g8']
    exact=result['macro']['pck_difference_exact']
    assert Fraction(exact['numerator'],exact['denominator'])==Fraction(-1,12)
    assert result['macro']['every_scene_has_center_population'] is True
    # Zero-population stratum cannot masquerade as zero accuracy.
    for row in rows:
        row.update(n_point=0,primary_correct=0,size_correct=0,size_benefit_count=0,size_harm_count=0,flip_valid_count=0)
    empty=A.aggregate_pairs(rows)['g8_g8']
    assert empty['macro']['pck_difference_exact'] is None
    assert all(v['primary_pck'] is None for v in empty['by_scene'].values())


def test_active_benchmark_prevents_work(monkeypatch,tmp_path):
    monkeypatch.setattr(A,'ROOT',tmp_path)
    timing=tmp_path/'results/round17_validation/timing';timing.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='benchmark'):A.no_active_validation_benchmark()
    (timing/'completion.json').write_text('{}')
    A.no_active_validation_benchmark()


def test_input_statistics_unique_query_identity_and_fixed_native_shape_bins():
    rows=[dict(scene='synthetic',pair=[0,1],query='mixed',query_index=i,width=w,height=h,
        tau_obs32=t,tau_size32=2.,observed_trace=.2,native_trace=.25)
        for i,(w,h,t) in enumerate(((8,16,1.),(8,16,3.),(16,8,4.)))]
    summary=A.summarize_supports(rows,expected_queries=3)
    assert summary['unique_queries']==3 and len(summary['groups'])==2
    first=summary['groups'][0]
    assert first['n']==2 and first['tau_obs32']['median']==2.
    assert first['tau_obs_over_tau_size']['median']==1.
    assert (first['width'],first['height'])==(8,16)
    with pytest.raises(ValueError,match='duplicate support query'):
        A.summarize_supports(rows+[rows[0]],expected_queries=4)
    with pytest.raises(ValueError,match='population incomplete'):
        A.summarize_supports(rows,expected_queries=4)


def test_center_sources_inherit_exact_stage_a_targets_and_only_mixed_from_r10(tmp_path):
    old=tmp_path/'old';old.mkdir()
    np.savez(old/'labels.npz',gt_xy=np.zeros((2,2)),gt_visible=np.ones(2,bool),
        mixed_point_xy=np.array([[1.,2.]]),mixed_point_visible=np.ones(1,bool))
    target=tmp_path/'r13.npz'
    np.savez(target,**{q+'_point_'+f:np.array([[10.+i,20.]]) if f=='xy' else np.ones(1,bool)
        for i,q in enumerate(('g8','g16','g24')) for f in ('xy','visible')})
    result=A.load_labels(dict(path=str(old),label_sha256=A.sha(old/'labels.npz'),
        target_path=str(target),target_sha256=A.sha(target)))
    assert result['mixed_point_xy'].tolist()==[[1.,2.]]
    assert [result[q+'_point_xy'][0,0] for q in ('g8','g16','g24')]==[10.,11.,12.]


def test_full_native_end_to_end_numeric_interface(tmp_path):
    """One fixed full-size affine fixture, never selected by numerical outcome."""
    import json
    import os
    import time
    from experiments import round17_size_temperature_audit as independent
    artifact=Path(os.environ.get('ROUND17_SIZE_INTERFACE_OUT',str(tmp_path)))
    artifact.mkdir(parents=True,exist_ok=True)
    started=time.time()
    seed=171709
    transform=np.array([[1.0125,.015],[-.01,.9875]],np.float64)
    translation=np.array([.2,-.15],np.float64)
    fixture=dict(synthetic_only=True,real_inputs_or_labels_read=False,seed=seed,
        image_height=144,image_width=192,feature_shape=[432,128],affine=transform.tolist(),
        translation=translation.tolist(),pixel_covariance='A @ A.T / 12',
        conditions=14,roles=2,query_rows_per_matrix=8,key_rows='all native',
        fixture_adjusted_after_error=False)
    (artifact/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    yy,xx=np.mgrid[:144,:192]
    xy=np.stack((xx,yy),axis=-1).reshape(-1,2)
    rng=np.random.default_rng(seed)
    qfeatures=rng.standard_normal((432,128)).astype(np.float32)
    qfeatures/=np.sqrt(np.square(qfeatures).sum(-1,keepdims=True))
    kfeatures=qfeatures+.02*rng.standard_normal((432,128)).astype(np.float32)
    kfeatures/=np.sqrt(np.square(kfeatures).sum(-1,keepdims=True))
    inputs=dict(q_to_k_xy=(xy@transform.T+translation).astype(np.float32),
        q_to_k_pixel_cov=np.broadcast_to((transform@transform.T/12).astype(np.float32),(144*192,2,2)).copy(),
        q_features=qfeatures,k_features=kfeatures)
    parts=A.R.S.M.partitions();partition=A.R.S.M.partition_tensors(parts,'cpu')
    supports={};samples={}
    with torch.no_grad():
        queries,keys=A.R.prepare_pair(inputs,partition)
        for q,query in queries.items():
            supports[q+'__mean']=query['mean'].numpy();supports[q+'__cov']=query['cov'].numpy()
        for c in A.CONDITIONS:
            q,k=c['query'],c['key'];query,key=queries[q],keys[k]
            geometry,content,raw,_=A.geometry_content(query,key)
            indices=np.linspace(0,len(query['mean'])-1,8,dtype=np.int64)
            observed_tau=A.R.P.trace_temperature(query['cov'],.25,10.)
            native_tau=M.size_temperature(partition[q]['bounds'],.25,10.)
            observed_bias=A.R.P.content_bias(query['cov'],.25,10.,content)
            native_bias=native_tau[:,None]*content
            samples[c['condition']]=dict(
                trace_full=A.R.P.probabilities(geometry,content,query['cov'],.25,10.).numpy(),
                size_temperature=M.probabilities(geometry,content,partition[q]['bounds'],.25,10.).numpy(),
                indices8=indices,raw8=raw.numpy()[indices],content8=content.numpy()[indices],
                tau_obs32_8=observed_tau.numpy()[indices],tau_size32_8=native_tau.numpy()[indices],
                bias_obs32_8=observed_bias.numpy()[indices],bias_size32_8=native_bias.numpy()[indices])
    fixture['input_array_sha256']={n:A.array_sha(v) for n,v in inputs.items()}
    fixture['complete_probability_sha256']={c:{role:A.array_sha(s[role]) for role in A.ROLES} for c,s in samples.items()}
    (artifact/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    try:
        checks=independent.numeric_pair(inputs,parts,supports,samples)
    except Exception as error:
        (artifact/'interface_exception.json').write_text(json.dumps(dict(
            type=type(error).__name__,message=str(error),seconds=time.time()-started),indent=2)+'\n')
        raise
    summary=dict(synthetic_only=True,real_inputs_or_labels_read=False,matrices=len(checks),
        all_passed=all(c['passed'] for c in checks),
        maximum_same_mass_error=max(c['same_mass_probability_error'] for c in checks),
        maximum_reference_error=max(c['max_probability_error'] for c in checks),
        maximum_reference_convergence=max(c['reference_convergence_error'] for c in checks),
        maximum_hybrid_error=max(c['hybrid_probability_error'] for c in checks),
        tau32_exact=all(c['same_prepared_tau32_exact'] for c in checks),
        bias32_exact=all(c['same_prepared_bias32_exact'] for c in checks),seconds=time.time()-started)
    (artifact/'numerical_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (artifact/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert len(checks)==28 and summary['all_passed']
    assert summary['tau32_exact'] and summary['bias32_exact']
    assert summary['maximum_same_mass_error']<=1e-12
    assert summary['maximum_reference_error']<=1e-4 and summary['maximum_hybrid_error']<=1e-4
    assert summary['maximum_reference_convergence']<=1e-8
