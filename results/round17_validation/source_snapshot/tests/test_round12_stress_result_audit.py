import numpy as np
import pytest
from experiments import round12_stress_result_audit as A


def test_independent_native_center_identity_hole_and_depth_consistency():
    centers=np.array([[11.5,11.5],[35.5,11.5],[59.5,11.5]])
    q=np.ones((288,384),np.float32)*2;k=q.copy()
    # Exact native addresses under half-pixel scaling are (24,24),(72,24),(120,24).
    q[24,72]=0;k[24,120]=1
    K=np.array([[100.,0,191.5],[0,100.,143.5],[0,0,1.]])
    xy,visible,status=A.center_reference(centers,q,k,K,K,np.eye(4),np.eye(4))
    np.testing.assert_allclose(xy[[0,2]],centers[[0,2]],atol=1e-12,rtol=0)
    assert np.isnan(xy[1]).all()
    np.testing.assert_array_equal(visible,[True,False,False])
    np.testing.assert_array_equal(status,[0,1,5])


def test_nearest_sample_halfopen_boundary_and_guard():
    d=np.arange(1,10,dtype=np.float32).reshape(3,3)
    value,known=A.native_sample(d,np.array([[-.5,0],[2.5,0],[.5-1e-12,.5]]))
    np.testing.assert_array_equal(known,[True,False,True])
    assert value[0]==1 and value[2]==5 and np.isnan(value[1])


def test_independent_projected_noise_matches_frozen_policy():
    from tests.test_round12_stress_pe import synthetic_inputs
    from models.round12_stress_pe import perturbed_observations
    inputs=synthetic_inputs()
    for state in A.STATES:
        ref,_=A.perturb_reference(inputs,state);got,_=perturbed_observations(inputs,state)
        np.testing.assert_array_equal(ref['q_to_k_xy'],got['q_to_k_xy'])
        np.testing.assert_array_equal(ref['q_to_k_pixel_cov'],got['q_to_k_pixel_cov'])


def test_common_reconstruction_and_direct_counts_on_nonaligned_scale():
    from models.round12_stress_pe import partitions
    parts=partitions();lookups={n:A.part_lookup(p) for n,p in parts.items()}
    y,x=np.mgrid[:144,:192];xy=np.stack([x,y],-1).reshape(-1,2)
    labels=dict(gt_xy=xy,gt_visible=np.ones(len(xy),bool))
    counts,common=A.counts_reference(labels,parts,lookups,'g24','g16')
    assert counts.sum()==common.sum()==27648
    np.testing.assert_array_equal(counts.sum(-1),np.full(48,576))
    with pytest.raises(ValueError):A.common_reference(counts/576,parts['g24'],parts['g16'])


def test_audit_equality_rejects_decision_or_population_changes():
    assert A.compare({'passed':True,'mass':5},{'passed':True,'mass':5})==0
    with pytest.raises(ValueError):A.compare({'passed':False},{'passed':True})
    with pytest.raises(ValueError):A.compare({'mass':4},{'mass':5})
