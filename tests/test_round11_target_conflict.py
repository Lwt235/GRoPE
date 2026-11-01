import numpy as np
import pytest
from scipy.optimize import minimize

from experiments.round11_target_conflict import project_to_argmax,minimum_allowed_argmax_kl,reconstruct_counts,diagnose_pair


def constrained_reference(g,j):
    n=len(g);others=np.array([i for i in range(n) if i!=j])
    positive=g>0
    def objective(p):return float(np.sum(g[positive]*(np.log(g[positive])-np.log(p[positive]))))
    result=minimize(objective,np.full(n,1/n),method='SLSQP',bounds=[(1e-12,1)]*n,
        constraints=[{'type':'eq','fun':lambda p:p.sum()-1},
                     {'type':'ineq','fun':lambda p:p[j]-p[others]}],
        options={'ftol':1e-12,'maxiter':500})
    assert result.success,result.message
    return result.fun


def test_independent_scipy_convex_projection():
    rng=np.random.default_rng(110031)
    for n in (3,5,8):
        for repeat in range(3):
            g=rng.dirichlet(np.ones(n)*2)
            for j in (0,n-1):
                p,value,_=project_to_argmax(g,j)
                assert p.sum()==pytest.approx(1.,abs=2e-15)
                assert p[j]>=p.max()-1e-15
                assert value==pytest.approx(constrained_reference(g,j),abs=2e-9)


def test_best_allowed_candidate_matches_brute_force():
    rng=np.random.default_rng(110032)
    for _ in range(30):
        g=rng.dirichlet(np.ones(9));allowed=rng.random(9)>.5
        if not allowed.any():allowed[0]=True
        result=minimum_allowed_argmax_kl(g,allowed)
        expected=min(project_to_argmax(g,j)[1] for j in np.flatnonzero(allowed))
        assert result['value']==pytest.approx(expected,abs=2e-15)


def test_analytic_two_cell_conflict_zeros_and_ties():
    p,value,_=project_to_argmax(np.array([.8,.2]),1)
    np.testing.assert_array_equal(p,[.5,.5])
    assert value==pytest.approx(.8*np.log(1.6)+.2*np.log(.4),abs=2e-15)
    p,value,_=project_to_argmax(np.array([1.,0.,0.]),1)
    np.testing.assert_array_equal(p,[.5,.5,0.]);assert value==pytest.approx(np.log(2))
    result=minimum_allowed_argmax_kl(np.array([.5,.5,0.]),np.array([False,True,False]))
    assert result['value']==0 and result['index']==1


def test_full_raster_histogram_mass_and_native_half_pixel_edges():
    y,x=np.mgrid[:144,:192];xy=np.stack([x,y],-1).reshape(-1,2).astype(float)
    visible=np.ones(len(xy),bool)
    counts=reconstruct_counts(xy,visible)
    np.testing.assert_array_equal(counts,np.eye(432,dtype=np.int64)*64)
    xy[0]=[-.5,-.5];xy[1]=[7.5,0]
    counts=reconstruct_counts(xy,visible)
    assert counts[0,0]==63 and counts[0,1]==1 and counts.sum()==144*192


def test_deterministic_first_tie_differs_from_optimistic_tie():
    xy=np.zeros((144*192,2),float);visible=np.zeros(144*192,bool)
    visible[0:2]=True;xy[0]=[3.5,3.5];xy[1]=[27.5,3.5]
    center=np.zeros((432,2));center[0]=[27.5,3.5]
    valid=np.zeros(432,bool);valid[0]=True
    rows=diagnose_pair(dict(gt_xy=xy,gt_visible=visible,g8_point_xy=center,g8_point_visible=valid),'synthetic',[1,2])
    assert len(rows)==432
    assert rows[0]['joint_valid']
    assert not rows[0]['native_first_argmax_correct']
    assert rows[0]['optimistic_top_tie_correct']
    assert rows[0]['min_kl']==0
    assert not any(r['joint_valid'] for r in rows[1:])


@pytest.mark.parametrize('g',[[-1.,2.],[.2,.2],[np.nan,0.],[np.inf,0.]])
def test_invalid_distributions_rejected(g):
    with pytest.raises(ValueError):project_to_argmax(g,0)
