import numpy as np
import pytest
from scipy.optimize import minimize
from experiments import round13_target_audit as A


@pytest.mark.parametrize('g,j',[(np.array([.52,.23,.13,.08,.04]),4),
    (np.array([.7,.3,0,0]),2),(np.array([.2,.2,.2,.2,.2]),3),
    (np.array([.49,.25,.25,.01]),3)])
def test_independent_cap_root_matches_general_constrained_optimizer(g,j):
    p,value,active=A.cap_projection(g,j)
    positive=g>0;others=np.delete(np.arange(len(g)),j)
    objective=lambda v:np.sum(g[positive]*np.log(g[positive]/v[positive]))
    res=minimize(objective,np.full(len(g),1/len(g)),method='SLSQP',bounds=[(1e-12,1)]*len(g),
        constraints=[{'type':'eq','fun':lambda v:v.sum()-1},
            {'type':'ineq','fun':lambda v:v[j]-v[others]}],options={'ftol':1e-13,'maxiter':500})
    assert res.success
    assert value==pytest.approx(res.fun,abs=2e-10)
    assert p[j]>=p.max() and p.sum()==pytest.approx(1,abs=2e-13)


def test_first_index_tie_can_fail_with_zero_relaxed_kl():
    count=np.zeros(432,int);count[[0,1,2,24,25,26,48,49,50]]=64
    row,n=A.row_reference(count,np.array([11.5,11.5]),True,'synthetic',[1,2],24,0)
    assert row['native_first_argmax_correct'] is False
    assert row['optimistic_top_tie_correct'] is True
    assert row['min_kl']==0 and row['strict_conflict'] is False
    assert row['allowed_key_count']==5


def test_exact_cap_tie_has_unique_projection_but_nonunique_active_count():
    count=np.array([15,20,11,3,8,7,14,11]);g=count/count.sum()
    p,_,_=A.cap_projection(g,5)
    assert p[5]==pytest.approx(14/count.sum(),abs=2e-15)
    assert A.active_count_interval(g,5)==(3,4)
    # Including the pre-existing 14/count.sum() coordinate changes no probability.
    assert p[6]==pytest.approx(g[6],abs=2e-15)


def test_unconstrained_denominators_do_not_claim_localization():
    counts=np.zeros((3,432),int);counts[0,[0,3]]=[80,20];counts[1,0]=50
    rows=[A.row_reference(counts[i],np.array([27.5,3.5]),i==0,'s',[1,2],24,i)[0] for i in range(3)]
    result=A.summary_reference(rows);cost=rows[0]['min_kl']
    assert result['min_kl_mean_joint']==cost
    assert result['min_kl_mean_all_query_unconstrained_zero']==pytest.approx(cost/3)
    assert result['min_kl_mean_area_valid_unconstrained_zero']==pytest.approx(cost/2)
    assert result['min_kl_mass_weighted_all_area_unconstrained_zero']==pytest.approx(cost*100/150)
    assert result['area_valid_center_invalid']==1 and result['neither_valid']==1


def test_sensor_identity_histograms_at_all_scales():
    y,x=np.mgrid[:144,:192];pixels=np.c_[x.ravel(),y.ravel()];good=np.ones(len(pixels),bool)
    for stride in (8,16,24):
        count=A.histogram(pixels,good,stride)
        np.testing.assert_array_equal(count.sum(-1),stride*stride)
        np.testing.assert_array_equal(count[count>0],64)
        assert count.sum()==27648
