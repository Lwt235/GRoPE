import numpy as np
import pytest
from scipy.optimize import minimize

from experiments import round13_target_conflict as P


@pytest.mark.parametrize('stride',[8,16,24])
def test_identity_raster_all_scales_and_exact_center_geometry(stride):
    yy,xx=np.mgrid[:144,:192];xy=np.stack([xx,yy],-1).reshape(-1,2).astype(float)
    counts=P.reconstruct_counts(xy,np.ones(len(xy),bool),stride)
    assert counts.sum()==144*192
    np.testing.assert_array_equal(counts.sum(-1),stride*stride)
    np.testing.assert_array_equal((counts>0).sum(-1),(stride//8)**2)
    np.testing.assert_array_equal(counts[counts>0],64)
    rows=P.diagnose_counts(counts,P.native_centers(stride),np.ones(len(counts),bool),'identity',[1,2],stride)
    assert all(r['optimistic_top_tie_correct'] and r['min_kl']==0 for r in rows)
    assert all(r['native_first_argmax_correct']==(stride!=24) for r in rows)
    if stride==24:
        assert rows[0]['allowed_key_count']==5
        assert rows[0]['global_max_bin_mass']==pytest.approx(1/9)
        assert rows[0]['best_allowed_bin_mass']==pytest.approx(1/9)
        assert rows[0]['first_argmax_distance']==pytest.approx(np.sqrt(128))


def test_summary_unconstrained_rows_keep_all_query_and_area_denominators():
    count=np.zeros((48,432),np.int64);count[0,0]=80;count[0,3]=20;count[1,0]=50
    centers=np.zeros((48,2));centers[0]=[27.5,3.5]
    valid=np.zeros(48,bool);valid[0]=True
    rows=P.diagnose_counts(count,centers,valid,'synthetic',[1,2],24)
    s=P.summarize(rows);cost=rows[0]['min_kl']
    assert s['joint_valid_queries']==1 and s['area_valid_center_invalid']==1
    assert s['neither_valid']==46
    assert s['min_kl_mean_joint']==cost
    assert s['min_kl_mean_all_query_unconstrained_zero']==pytest.approx(cost/48)
    assert s['min_kl_mass_weighted_all_area_unconstrained_zero']==pytest.approx(cost*100/150)
    assert s['strict_conflict_fraction_all_query']==pytest.approx(1/48)


def test_projection_independent_scipy_with_fixed_allowed_pooling():
    g=np.array([.52,.23,.13,.08,.04]);allowed=np.array([False,False,False,True,True])
    actual=P.T.minimum_allowed_argmax_kl(g,allowed)
    reference=[]
    for j in np.flatnonzero(allowed):
        others=np.array([i for i in range(len(g)) if i!=j])
        objective=lambda p:float(np.sum(g*(np.log(g)-np.log(p))))
        result=minimize(objective,np.ones(5)/5,method='SLSQP',bounds=[(1.e-12,1)]*5,
            constraints=[{'type':'eq','fun':lambda p:p.sum()-1},
                         {'type':'ineq','fun':lambda p:p[j]-p[others]}],
            options={'ftol':1.e-13,'maxiter':500})
        assert result.success
        reference.append(result.fun)
    assert actual['value']==pytest.approx(min(reference),abs=1.e-10)


def test_scale8_matches_frozen_oracle_reconstruction():
    rng=np.random.default_rng(130024)
    xy=rng.uniform([-.5,-.5],[191.5,143.5],(144*192,2));visible=rng.random(len(xy))>.3
    np.testing.assert_array_equal(P.reconstruct_counts(xy,visible,8),P.T.reconstruct_counts(xy,visible))


def test_invalid_visible_projection_rejected():
    xy=np.zeros((144*192,2));visible=np.zeros(len(xy),bool);visible[0]=True;xy[0]=[192.,0]
    with pytest.raises(ValueError):P.reconstruct_counts(xy,visible,24)
