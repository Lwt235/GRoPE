"""R21 synthetic-only operator, population, paired-metric and chronology tests."""
from fractions import Fraction
import copy
import json
from math import sqrt
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from models import round21_matched_decomposition as M
from models import round20_anchor_pe as A
from models import round15_validation_pe as OLD
from experiments import round21_matched_decomposition_metrics as MET
from tests.test_round20_anchor_pe import synthetic as old_synthetic


@pytest.fixture(scope='module',autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def synthetic():
    observations,parts=old_synthetic.__wrapped__()
    # Avoid an exactly binary affine fixture whose reductions legitimately
    # have zero residual; exercise the declared finite-precision limitation.
    perturb=torch.arange(observations['q_to_k_xy'].numel(),dtype=torch.float32).reshape_as(observations['q_to_k_xy'])
    observations['q_to_k_xy']=observations['q_to_k_xy']+.003*torch.sin(perturb)
    banks={}
    for q in ('g8','g16','g24'):
        mean=OLD.prepare_query(observations,parts[q],'gaussian_full1')['mean']
        anchor=mean.double()+mean.new_tensor([.125,-.0625]).double()+1.e-10
        banks[q]=M.prepare(observations,parts[q],parts['g8'],anchor)
    return observations,parts,banks


def test_fixed_eight_roles_ten_contrasts_no_search():
    assert len(M.FAMILIES)==8 and len(MET.CONTRASTS)==10
    assert len({c['name'] for c in MET.CONTRASTS})==10
    assert all(v['h']==.25 and v['temperature']==10 for v in M.ROLE_CONFIGS.values())
    assert json.loads(json.dumps(M.ROLE_CONFIGS))==M.ROLE_CONFIGS
    assert not hasattr(MET,'screen')


@pytest.mark.parametrize('q',['g8','g16','g24'])
def test_original_T0_and_R20_content_exact(synthetic,q):
    obs,parts,banks=synthetic;p=banks[q]
    for geometry,oldfamily in [('mu_c','gaussian_full1'),('full4','gaussian_full4')]:
        actual,_=M.probabilities(p,geometry)
        query=OLD.prepare_query(obs,parts[q],oldfamily);key=OLD.prepare_key(obs,parts['g8'],oldfamily)
        expected,_=OLD.predict(query,key,oldfamily,.25,0.)
        assert torch.equal(actual['zero'],expected)
    for geometry,oldfamily in [('a_c','mean_only_matched_tau'),('a_ca','anchor_trace')]:
        actual,_=M.probabilities(p,geometry)
        query=A.prepare_query(obs,parts[q],oldfamily,p['a64']);key=A.prepare_key(obs,parts['g8'],oldfamily)
        expected,_=A.predict(query,key,oldfamily)
        assert torch.equal(actual['content'],expected)


@pytest.mark.parametrize('geometry',M.GEOMETRIES)
def test_common_bias_and_zero_same_raw_no_rows_removed(synthetic,geometry):
    p=synthetic[2]['g24'];raw=M.raw_mass(p,geometry);actual,_=M.probabilities(p,geometry,raw)
    floor=1.e-30*p['key']['mass'].double()/64
    logits=(raw+floor[None]).log()
    assert torch.equal(actual['zero'],(logits+torch.zeros_like(p['bias32']).double()).softmax(-1))
    assert torch.equal(actual['content'],(logits+p['bias32'].double()).softmax(-1))
    assert actual['content'].dtype==torch.float64 and actual['content'].shape==p['content32'].shape
    assert torch.equal(p['bias32'],p['tau32'][:,None]*p['content32'])


def test_mixture_moments_keep_components_and_report_float_residual(synthetic):
    p=synthetic[2]['g16'];n=p['full4_nodes32'].numpy().astype(np.float64)
    cov=p['full4_node_cov32'].numpy().astype(np.float64);w=p['full4_weights32'].numpy().astype(np.float64)
    mean=(w[...,None]*n).sum(1);d=n-mean[:,None]
    total=(w[...,None,None]*(cov+d[..., :,None]*d[...,None,:])).sum(1)
    assert np.array_equal(p['mixture_mean64'].numpy(),mean)
    assert np.array_equal(p['mixture_cov64'].numpy(),total)
    assert np.array_equal(p['mean_residual64'].numpy(),mean-p['mu32'].numpy().astype(np.float64))
    assert np.array_equal(p['cov_residual64'].numpy(),total-p['cov32'].numpy().astype(np.float64))
    report=M.moment_diagnostics(p);assert report['components_repaired'] is False
    assert report['cov_max_absolute_residual']>0  # Do not silently force moments to agree.
    shift=p['full4_nodes32'].new_tensor([2.,-1.])
    moved,mcov=M.mixture_moments(p['full4_nodes32']+shift,p['full4_node_cov32'],p['full4_weights32'])
    assert torch.allclose(moved,p['mixture_mean64']+shift.double(),atol=1e-6,rtol=0)
    assert torch.allclose(mcov,p['mixture_cov64'],atol=1e-6,rtol=0)


def test_full4_collapse_and_within_component_covariance(synthetic):
    p=dict(synthetic[2]['g24']);p['full4_nodes32']=p['mu32'][:,None,:].expand(-1,4,-1).clone()
    p['full4_node_cov32']=p['cov32'][:,None,:,:].expand(-1,4,-1,-1).clone()
    p['full4_weights32']=torch.full_like(p['full4_weights32'],.25)
    mean,cov=M.mixture_moments(p['full4_nodes32'],p['full4_node_cov32'],p['full4_weights32'])
    assert torch.equal(mean,p['mu32'].double());assert torch.equal(cov,p['cov32'].double())
    assert torch.equal(M.raw_mass(p,'full4'),M.raw_mass(p,'mu_c'))


def test_floor_once_after_mixture_sum_and_zero_raw_rows():
    p=dict(a64=torch.zeros((2,2),dtype=torch.float64),content32=torch.zeros((2,2)),
           bias32=torch.tensor([[0.,1.],[0.,0.]]),key={'mass':torch.tensor([64.,256.])})
    raw=torch.zeros((2,2),dtype=torch.float64)
    actual,diag=M.probabilities(p,'full4',raw)
    assert torch.allclose(actual['zero'],torch.tensor([[.2,.8],[.2,.8]],dtype=torch.float64),atol=1e-15,rtol=0)
    assert diag['raw_zero_fov_rows']==2 and diag['raw_tiny_fov_rows']==2
    expected=(torch.tensor([[1.e-30,4.e-30]],dtype=torch.float64).log()+p['bias32'].double()).softmax(-1)
    assert torch.equal(actual['content'],expected)


def test_labels_extra_fields_nonfinite_and_wrong_anchor_rejected(synthetic):
    obs,parts,banks=synthetic
    poison=dict(obs,gt_xy=torch.zeros((1,2)))
    with pytest.raises(ValueError):M.prepare(poison,parts['g8'],parts['g8'],banks['g8']['a64'])
    with pytest.raises(ValueError):M.prepare(obs,parts['g8'],parts['g8'],banks['g8']['a64'].float())
    poison=dict(obs);poison['q_to_k_xy']=obs['q_to_k_xy'].clone();poison['q_to_k_xy'][0,0]=float('nan')
    with pytest.raises(ValueError):M.prepare(poison,parts['g8'],parts['g8'],banks['g8']['a64'])


def test_paired_integer_boundary_and_continuous_signed_differences():
    bounds=np.array([[0,0,1,1],[16,0,17,1]],np.int64)
    before=np.array([0,0,1,0],np.int32);after=np.array([1,1,0,1],np.int32)
    target=np.array([[16.,0.],[8.,0.],[16.,0.],[0.,0.]])
    valid=np.array([True,True,True,False])
    r=MET.paired_center(before,after,bounds,target,valid)
    assert (r['benefit'],r['harm'],r['both_correct'],r['both_wrong'])==(1,1,1,0)
    assert r['n_point']==3 and r['n_query']==4 and r['flip_valid']==3 and r['flip_all']==4
    assert r['distance_delta_sum']==0 and r['squared_distance_delta_sum']==0 and r['rmse_delta']==0
    assert (r['distance_benefit'],r['distance_harm'],r['distance_equal'])==(1,1,1)
    empty=MET.paired_center(before,after,bounds,target,np.zeros(4,dtype=bool))
    assert empty['n_point']==0 and empty['rmse_delta'] is None and empty['flip_all']==4


def test_paired_RMSE_is_difference_after_aggregate_not_mean_pair_RMSE():
    bounds=np.array([[0,0,1,1],[16,0,17,1]],np.int64);valid=np.ones(1,dtype=bool)
    a=MET.paired_center(np.array([0],np.int32),np.array([1],np.int32),bounds,np.array([[0.,0.]]),valid)
    b=MET.paired_center(np.array([0],np.int32),np.array([0],np.int32),bounds,np.array([[10.,0.]]),valid)
    total=MET._paired_aggregate([a,b])
    assert total['rmse_delta']==sqrt((256+100)/2)-sqrt(100/2)
    assert total['rmse_delta']!=(a['rmse_delta']+b['rmse_delta'])/2
    assert MET.fraction(total['pck_delta'])==Fraction(-1,2)


def test_complete_scene_scale_summary_fraction_and_interaction():
    rows=[];pair=[]
    cm=dict(n_query=1,n_point=1,pck_count=1,distance_sum=2.,distance_squared_sum=4.,mean=2.,rmse=2.)
    bounds=np.array([[0,0,1,1],[16,0,17,1]],np.int64)
    pc=MET.paired_center(np.array([0],np.int32),np.array([1],np.int32),bounds,np.array([[16.,0.]]),np.array([True]))
    for scene in MET.SCENES:
        for c in MET.CONDITIONS:
            for i in range(16):
                base=dict(scene=scene,pair=[i,i+1],condition=c['condition'],mass=2)
                rows.append(dict(base,**{k+'_sum':2. for k in MET.METRICS},center=cm))
                pair.append(dict(base,**{k+'_delta_sum':-1. for k in MET.METRICS},center=pc))
    r=MET.summarize(rows);assert r['macro']=={k:1. for k in MET.METRICS}
    assert MET.fraction(r['center']['macro']['pck'])==Fraction(1)
    ps=MET.summarize(pair,paired=True);assert MET.fraction(ps['center']['macro']['pck_delta'])==Fraction(1)
    contrasts={c['name']:dict(from_role=c['from_role'],to_role=c['to_role'],**copy.deepcopy(ps)) for c in MET.CONTRASTS}
    effects=MET.interactions(contrasts)
    assert set(effects)=={'anchor','widening','shape'}
    assert all(v['macro']=={k+'_delta':0. for k in MET.METRICS} for v in effects.values())
    assert all(MET.fraction(v['center']['macro']['pck_delta'])==0 for v in effects.values())
    with pytest.raises(ValueError):MET.summarize(rows[:-1])


def test_failed_numeric_audit_blocks_every_array_and_preserves_failure():
    from experiments import round21_matched_decomposition as P
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)
        with patch.object(P,'verify_lock',return_value={}),patch.object(P,'require_forward_audit',side_effect=ValueError('numeric failure')), \
             patch.object(P.np,'load',side_effect=AssertionError('array accessed')) as load:
            with pytest.raises(ValueError,match='numeric failure'):P.evaluate(out)
            load.assert_not_called()
        assert json.loads((out/'evaluation/completion.json').read_text())['passed'] is False
        original=(out/'evaluation/failure.json').read_bytes()
        with pytest.raises(ValueError,match='preserve existing'):P.evaluate(out)
        assert (out/'evaluation/failure.json').read_bytes()==original


def test_legacy_canonical_hash_is_distinct_and_sample_claim_is_explicit():
    from experiments import round21_matched_decomposition as P
    a=np.array([[.25,.75]],np.float64)
    assert P.array_sha(a)!=P.legacy_array_sha(a)
    source=dict(replay_kind='r15_commitment',array_sha256=P.legacy_array_sha(a),source_archive_path=None)
    r=P.check_inherited_probability(a,source)
    assert r['commitment_exact'] is True and r['saved_array_checked'] is False
    with pytest.raises(ValueError):P.check_inherited_probability(a,dict(source,array_sha256=P.array_sha(a)))
