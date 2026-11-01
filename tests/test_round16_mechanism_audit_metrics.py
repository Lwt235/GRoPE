"""Synthetic counterexamples for the independent Round16 diagnostic math."""
from copy import deepcopy
import math

import numpy as np
import pytest

from experiments import round16_mechanism_audit_metrics as A


def tilted(p, content, temperature):
    logits=np.log(p)+(np.float32(temperature)*content).astype(np.float64)
    weights=np.exp(logits-logits.max(-1,keepdims=True))
    return weights/weights.sum(-1,keepdims=True)


def toy():
    p0=np.array([[.6,.3,.1],[.6,.3,.1],[.4,.4,.2],[.6,.3,.1]],np.float64)
    content=np.array([[0,.3,0],[0,.3,0],[0,0,0],[0,0,0]],np.float32)
    mu=np.array([[0.,0.],[8.,0.],[24.,0.],[-4.,0.]])
    covariance=np.broadcast_to(np.eye(2),(4,2,2)).copy()
    keys=np.array([[0.,0.],[8.,0.],[16.,0.]])
    target=np.array([[-4.,0.],[12.,0.],[np.nan,np.nan],[8.,0.]])
    valid=np.array([True,True,False,True])
    return p0,tilted(p0,content,5),content,mu,covariance,keys,target,valid


def analyze_toy():return A.analyze(*toy(),h=.5,temperature=5.)


def test_first_argmax_second_winner_ties_and_original_valid_population():
    value=analyze_toy()
    np.testing.assert_array_equal(value['winner'],[1,1,0,0])
    np.testing.assert_array_equal(value['zero_winner'],[0,0,0,0])
    np.testing.assert_array_equal(value['j2'],[1,1,1,1])
    np.testing.assert_array_equal(value['tie_count'],[1,1,2,1])
    np.testing.assert_array_equal(value['nearest_mu'],[0,1,2,0])
    np.testing.assert_array_equal(value['correct'],[0,1,-1,1])
    np.testing.assert_array_equal(value['zero_correct'],[1,0,-1,1])
    assert np.isnan(value['mu_target_dist_px'][2])
    assert value['correct'].dtype==np.int8
    assert A.validate_arrays(value)==4


def test_geom_content_identity_pixel_units_and_fov_half_open_boundary():
    value=analyze_toy()
    assert value['top_geometry_gap'][0]==pytest.approx(math.log(2))
    assert value['geom_penalty'][0]==pytest.approx(math.log(2))
    assert value['content_gain'][0]==1.5
    assert value['advantage'][0]==pytest.approx(1.5-math.log(2))
    assert value['identity_error'].max()<1e-15
    np.testing.assert_array_equal(value['winner_to_zero_dx_px'],[8,8,0,0])
    np.testing.assert_array_equal(value['winner_to_zero_dist_px'],[8,8,0,0])
    np.testing.assert_array_equal(value['mu_in_fov'],[True,True,False,True])
    assert value['fov_distance_px'][2]==4.
    assert value['mu_phase_x'][0]==.5 and value['mu_phase_x'][3]==0.
    inputs=list(toy());inputs[3]=inputs[3].copy();inputs[3][0]=[20.,0.]
    boundary=A.analyze(*inputs,h=.5,temperature=5.)
    assert not boundary['mu_in_fov'][0] and boundary['fov_distance_px'][0]==0


def test_float32_products_are_rounded_before_the_content_difference():
    # A sizeable common offset makes the distributive-but-different fp32
    # expression detectably wrong, independently of the geometry identity.
    p=np.array([[.50000001,.49999999]],np.float64)
    content=np.array([[.1,.10000003]],np.float32)
    x=np.zeros((1,2));cov=np.zeros((1,2,2));keys=np.array([[0.,0.],[8.,0.]])
    expected=float(np.float32(10)*content[0,1])-float(np.float32(10)*content[0,0])
    wrong=float(np.float32(10)*(content[0,1]-content[0,0]))
    assert expected!=wrong
    result=A.analyze(p,tilted(p,content,10),content,x,cov,keys,x,np.ones(1,bool),h=.25,temperature=10.)
    assert result['winner'][0]==1
    assert result['content_gain'][0]==expected
    assert result['identity_error'][0]<1e-15


def test_covariance_context_is_reference8_with_h_added_exactly_once():
    inputs=list(toy());inputs[4]=np.broadcast_to([[1.,1.],[1.,1.]],(4,2,2)).copy()
    result=A.analyze(*inputs,h=.5,temperature=5.)
    np.testing.assert_allclose(result['S_condition'],9.)
    np.testing.assert_allclose(result['S_major_angle_rad'],math.pi/4)
    assert not result['isotropic'].any()
    isotropic=analyze_toy()
    assert isotropic['isotropic'].all()
    np.testing.assert_array_equal(isotropic['S_major_angle_rad'],0.)


def test_covariance_symmetry_domain_and_native_grid_are_checked():
    values=list(toy())
    values[4]=np.broadcast_to([[1e6,1.],[1.+1e-7,1e6]],(4,2,2)).copy()
    assert A.analyze(*values,h=.5,temperature=5.)['S_condition'].min()>1
    values[4]=np.broadcast_to([[1.,0.],[1e-10,1.]],(4,2,2)).copy()
    with pytest.raises(ValueError,match='symmetric'):
        A.analyze(*values,h=.5,temperature=5.)
    for invalid_keys in (np.array([[0.,0.],[16.,0.],[32.,0.]]),
                         np.array([[0.,0.],[8.,0.],[0.,8.]]),
                         np.array([[0.,0.],[8.,0.],[8.,0.]])):
        values=list(toy());values[5]=invalid_keys
        with pytest.raises(ValueError,match='8-pixel'):
            A.analyze(*values,h=.5,temperature=5.)


def test_stats_harm_benefit_closure_and_all_vs_valid_denominators():
    summary=A.stats(analyze_toy());allq=summary['all'];valid=summary['valid']
    assert allq['n']==4 and valid['n']==allq['n_valid']==valid['n_valid']==3
    assert allq['n_flip']==valid['n_flip']==2
    assert allq['n_geometry_tie']==1 and valid['n_geometry_tie']==0
    assert allq['n_harm']==allq['n_benefit']==allq['n_both_correct']==1
    assert allq['n_both_wrong']==0
    assert allq['n_correct']-allq['n_zero_correct']==allq['n_benefit']-allq['n_harm']==0
    assert allq['mu_target_dist_px_count']==valid['mu_target_dist_px_count']==3
    assert allq['dist_mu_px_count']==4 and valid['dist_mu_px_count']==3
    assert A.validate_stats(summary)


def test_paired_statistics_match_queries_and_preserve_target_sentinels():
    args=list(toy());args[0]=np.tile([.95,.04,.01],(4,1)).astype(np.float64)
    args[1]=tilted(args[0],args[2],5)
    point=A.analyze(*args,h=.5,temperature=5.)
    full=analyze_toy();summary=A.paired_stats(full,point)
    s=summary['all']
    assert s['n']==4 and s['n_valid']==3
    assert s['n_same_winner']==2 and s['n_same_zero_winner']==4
    assert s['n_full_flip']==2 and s['n_point_flip']==0 and s['n_full_flip_point_not']==2
    assert s['n_full_correct_point_wrong']==s['n_full_wrong_point_correct']==s['n_both_correct']==1
    assert not A.validate_stats(summary)
    point['S_condition']=point['S_condition'].copy();point['S_condition'][0]=2.
    with pytest.raises(ValueError,match='shared input'):
        A.paired_stats(full,point)


def test_no_valid_centers_keep_all_geometry_and_null_correctness_means():
    args=list(toy());args[-1]=np.zeros(4,bool);args[-2]=np.full((4,2),np.nan)
    arrays=A.analyze(*args,h=.5,temperature=5.)
    summary=A.stats(arrays)
    assert summary['all']['n']==4 and summary['valid']['n']==0
    for population in summary.values():
        assert population['n_correct']==population['n_harm']==population['n_benefit']==0
        assert population['mu_target_dist_px_count']==population['mu_target_dist_px_sum']==0
    rows=[dict(scene=A.SCENES[0],pair=[1,2],family='quartic_full',grid_index=4,query='g8',stats=summary)]
    result=A.summarize(rows)['quartic_full']['4']['g8']
    assert result['macro']['all']['pck8'] is None
    assert result['macro']['valid']['dist_mu_px_mean'] is None
    assert result['macro']['all']['flip_rate']==.5


@pytest.mark.parametrize('change', ['zero_probability','wrong_dtype','negative_probability','content_dtype','visible_target_nan'])
def test_analysis_rejects_unidentifiable_logs_or_changed_input_contract(change):
    values=list(toy())
    if change=='zero_probability':values[0][0]=[1.,0.,0.]
    elif change=='wrong_dtype':values[0]=values[0].astype(np.float32)
    elif change=='negative_probability':values[1][0]=[1.1,-.1,0.]
    elif change=='content_dtype':values[2]=values[2].astype(np.float64)
    else:values[-2][0]=np.nan
    with pytest.raises(ValueError):A.analyze(*values,h=.5,temperature=5.)


def test_verification_is_exact_for_discrete_fields_and_keeps_invalid_nan():
    value=analyze_toy();same=deepcopy(value)
    assert A.compare_arrays(value,same)==0.
    same['mu_target_dist_px'][2]=0.
    with pytest.raises(ValueError,match='target distance validity'):A.compare_arrays(value,same)
    same=deepcopy(value);same['tie_count'][0]=2
    with pytest.raises(ValueError,match='discrete array'):A.compare_arrays(value,same)
    additive=A.stats(value);additive['all']['n_harm']=0
    with pytest.raises(ValueError,match='closure'):A.validate_stats(additive)


def synthetic_stats(n,valid,correct,offset):
    # Construct additive totals from a literal hand-specified query population.
    p=np.tile([.6,.3,.1],(n,1)).astype(np.float64)
    c=np.zeros_like(p,dtype=np.float32);mu=np.tile([offset,0.],(n,1))
    covariance=np.tile(np.eye(2),(n,1,1));keys=np.array([[0.,0.],[8.,0.],[16.,0.]])
    target=np.full((n,2),np.nan);target[:valid]=[16.,0.];target[:correct]=[0.,0.]
    mask=np.arange(n)<valid
    return A.stats(A.analyze(p,p,c,mu,covariance,keys,target,mask,h=.5,temperature=0.))


def test_scene_pooling_precedes_macro_and_keeps_scale_separate():
    rows=[]
    specifications=[[(1,1,0.),(9,0,10.)],[(10,10,20.),(10,10,20.)],[(5,0,30.),(5,0,30.)]]
    for scene,specs in zip(A.SCENES,specifications):
        for pair,(valid,correct,offset) in enumerate(specs):
            rows.append(dict(scene=scene,pair=[pair,pair+10],family='quartic_full',grid_index=0,query='g8',
                stats=synthetic_stats(10,valid,correct,offset)))
    report=A.summarize(rows)['quartic_full']['0']['g8']
    assert report['by_scene'][A.SCENES[0]]['valid']['derived']['pck8']==.1
    assert report['macro']['valid']['pck8']==pytest.approx((.1+1+0)/3)
    assert report['macro']['all']['dist_mu_px_mean']==pytest.approx((5+20+30)/3)
    assert report['macro']['valid']['dist_mu_px_mean']==pytest.approx((9+20+30)/3)
    assert report['macro']['valid']['pck8']!=pytest.approx(21/40)
    shuffled=list(reversed(rows))
    assert A.summarize(shuffled)==A.summarize(rows)
    empty=deepcopy(rows);empty[4]['stats']=synthetic_stats(10,0,0,30.);empty[5]['stats']=synthetic_stats(10,0,0,30.)
    assert A.summarize(empty)['quartic_full']['0']['g8']['macro']['valid']['pck8'] is None
    with pytest.raises(ValueError,match='duplicate'):A.summarize(rows+[rows[0]])


@pytest.fixture(scope='module')
def complete_rows():
    template={q:synthetic_stats(n,n//2,n//4,0.) for q,n in A.QUERY_COUNTS.items()}
    return [dict(scene=s,pair=[i,i+100],family=f,grid_index=g,query=q,stats=template[q])
            for s in A.SCENES for i in range(16) for f in A.FAMILIES for g in range(12) for q in A.QUERIES]


def test_full_5184_population_counts_every_configuration_and_query(complete_rows):
    result=A.verify_population(complete_rows)
    assert result['rows']==5184 and result['query_states']==1016064 and result['unique_queries']==28224
    assert len(A.summarize(complete_rows))==3
    with pytest.raises(ValueError,match='incomplete'):A.verify_population(complete_rows[:-1])
    with pytest.raises(ValueError,match='duplicate'):A.verify_population(complete_rows+[complete_rows[0]])
    changed=deepcopy(complete_rows);changed[0]['stats']=synthetic_stats(432,1,0,0.)
    with pytest.raises(ValueError,match='valid-center population'):A.verify_population(changed)


def test_independent_and_producer_synthetic_schema_agree():
    # Cross-implementation imports belong only to tests, never either module.
    from experiments import round16_mechanism_metrics as P
    inputs=toy();independent=A.analyze(*inputs,h=.5,temperature=5.)
    produced=P.analyze(*inputs,h=.5,temperature=5.)
    assert A.compare_arrays(produced,independent,atol=1e-12)<1e-12
    assert A.stats(independent)==P.stats(produced)
    assert A.paired_stats(independent,independent)==P.paired_stats(produced,produced)
    rows=[dict(scene=s,pair=[1,2],family='quartic_full',grid_index=4,query='g8',stats=A.stats(independent)) for s in A.SCENES]
    assert A.summarize(rows)==P.summarize(rows)
