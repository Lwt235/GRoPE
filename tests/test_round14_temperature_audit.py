import numpy as np
import pytest
from scipy.special import ndtr
from experiments import round14_temperature_audit as A


def test_metric_populations_include_zero_area_center_and_invalid_center_query():
    probability=np.array([[.25,.75],[.9,.1]],float)
    keys=np.array([[3.5,3.5],[11.5,3.5]])
    counts=np.array([[1,3],[0,0]],int)
    mean=np.array([[7.5,3.5],[3.5,3.5]])
    # First center invalid does not remove its regional metric or all-query offset.
    points=np.array([[np.nan,np.nan],[3.5,3.5]])
    row,correct=A.metric_reference(probability,np.array([1,0]),mean,counts,points,np.array([False,True]),keys)
    assert row['kl_sum']==0 and row['mass']==4
    assert row['pck_count']==1 and row['n_point']==1 and row['n_query']==2
    assert correct.tolist()==[False,True]
    assert row['winner_to_mean_sum_all']==4 and row['winner_to_mean_sum_center_valid']==0
    assert row['expected_to_mean_sum_all']==pytest.approx(2.8)
    assert row['expected_to_mean_sum_center_valid']==pytest.approx(.8)
    with pytest.raises(ValueError,match='argmax'):
        A.metric_reference(probability,np.array([0,0]),mean,counts,points,np.array([False,True]),keys)


def test_paired_population_does_not_drop_invalid_queries_or_change_center_mask():
    keys=np.array([[0.,0.],[8.,0.],[16.,0.]])
    r=A.paired_reference(np.array([0,2,1]),np.array([1,0,1]),
        np.array([True,False,False]),np.array([False,False,True]),np.array([True,False,True]),keys)
    assert r==dict(n_query=3,n_point=2,winner_changed_all=2,winner_changed_center_valid=1,
                  winner_displacement_sum_all=24.,left_correct_right_wrong=1,right_correct_left_wrong=1)


def make_rows():
    rows=[];paired=[]
    for si,scene in enumerate(A.SCENES):
        for i in range(16):
            for stride in A.STRIDES:
                q='g'+str(stride);mass=(i+1)*(si+1)*10;nv=(si+1)*10;nq=144*192//stride**2
                for role in A.CONFIGS:
                    # Unequal scene populations must not change equal-scene PCK.
                    pck=(.8-.1*si) if role=='full4_t0' else .9 if role=='point_pck' else .6
                    kl=.11 if role=='full4_t0' else .1
                    rows.append(dict(scene=scene,pair=[i,i+15],query=q,role=role,
                        kl_sum=kl*mass,js_sum=0.,mass=mass,pck_count=round(pck*nv),n_point=nv,n_query=nq,
                        winner_to_mean_sum_all=2*nq,winner_to_mean_sum_center_valid=2*nv,
                        expected_to_mean_sum_all=nq,expected_to_mean_sum_center_valid=nv,
                        within8_of_mean_probability_sum_all=.5*nq,within8_of_mean_probability_sum_center_valid=.5*nv,
                        predicted_mean_to_sensor_center_sum=3*nv))
                for name,(left,right) in A.CONTRASTS.items():
                    paired.append(dict(scene=scene,pair=[i,i+15],query=q,contrast=name,left=left,right=right,
                        n_query=nq,n_point=nv,winner_changed_all=nq//2,winner_changed_center_valid=nv//2,
                        winner_displacement_sum_all=8*nq,left_correct_right_wrong=nv//10,right_correct_left_wrong=0))
    return rows,paired


def test_summary_merges_mass_then_averages_scenes_and_keeps_scale_contrasts():
    rows,paired=make_rows();summary=A.summary_reference(rows,paired)
    for stride in A.STRIDES:
        q='g'+str(stride);contrast=summary['contrasts']['full4_content_off'][q]
        assert summary['roles']['full4_t0']['by_query'][q]['pck8']==pytest.approx(.7)
        assert contrast['pck_difference_pp']==pytest.approx(10.)
        assert contrast['direct_kl_relative_change']==pytest.approx(.1)
        assert contrast['paired_equal_scene']['left_correct_right_wrong_fraction']==pytest.approx(.1)
        assert summary['full4_train_pck_gap'][q]['train_gap_recovered_fraction']==pytest.approx(1/3)
    # One scene combines pair numerators/denominators, not mean of pair ratios.
    assert A.aggregate_reference([dict(rows[0],kl_sum=1.,mass=1),dict(rows[0],kl_sum=9.,mass=99)])['direct_kl']==pytest.approx(.1)


def test_scipy_bank_reference_reduces_to_exact_separable_gaussian():
    mean=np.array([.5,.8]);cov=np.diag([.3,.2])
    inputs=dict(q_to_k_xy=np.broadcast_to(mean*8,(144*192,2)),
        q_to_k_pixel_cov=np.broadcast_to(cov*64,(144*192,2,2)),
        q_features=np.zeros((432,2)),k_features=np.zeros((432,2)))
    p=A.reference_probabilities(inputs,24,np.array([0]))
    bounds=(A.R.grid(8)-.5)/8;sd=np.sqrt(np.diag(cov)+.25)
    mass=(ndtr((bounds[:,2:]-mean)/sd)-ndtr((bounds[:,:2]-mean)/sd)).prod(-1)+1e-30
    expected=mass/mass.sum()
    for role in ('full4_t0','full4_t5','full1_t0','full1_t5','quad16'):
        np.testing.assert_allclose(p[role][0],expected,atol=2e-14,rtol=0)
    assert np.max(abs(p['full1_frozen'][0]-expected))>.001
