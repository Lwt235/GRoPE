import numpy as np
import pytest
from experiments import round15_train_numerical as P
from experiments import round15_quartic_numerical_audit as R


def observations():
    y,x=np.mgrid[:144,:192];xy=np.stack([x,y],-1).reshape(-1,2).astype(np.float64)
    return dict(q_to_k_xy=xy,q_to_k_pixel_cov=np.broadcast_to(np.eye(2)/12,(len(xy),2,2)).copy(),
                q_features=np.broadcast_to([3.,4.],(432,2)).copy(),k_features=np.broadcast_to([3.,4.],(432,2)).copy())


@pytest.mark.parametrize('stride',[8,16,24])
def test_real_production_and_independent_moments_use_identical_complete_support(stride):
    inputs=observations();indices=P.selected_indices(stride)
    reference=P.independent_support(inputs,stride,indices);actual=P.production_support(inputs,stride,indices)
    expected_center=(P.native_bounds(stride)[indices,:2]+(stride-1)/2)/8
    np.testing.assert_allclose(reference['mean'],expected_center,atol=1e-13,rtol=0)
    np.testing.assert_allclose(reference['cov'],np.broadcast_to(np.eye(2)*stride**2/(12*64),(8,2,2)),atol=2e-13,rtol=0)
    for name in ('mean','cov','content'):
        np.testing.assert_allclose(actual[name].numpy(),reference[name],atol=2e-6,rtol=0)
    assert reference['content'].shape==(8,432)
    np.testing.assert_allclose(reference['content'],1.,atol=1e-14,rtol=0)


def test_high_order_independent_cartesian_reference_matches_adaptive_tail_with_content():
    bounds=np.array([[-.0625,-.0625,.9375,.9375],[-.0625,.9375,.9375,1.9375],[.9375,-.0625,1.9375,.9375]])
    mean=np.array([[-1.25,.4375]]);cov=np.zeros((1,2,2));content=np.array([[-.7,.8,.2]])
    adaptive=np.array([[R.adaptive_rectangle(mean[0],cov[0],b,.25)['mass'] for b in bounds]])
    r64=P.independent_mass(mean,cov,bounds,.25,64);r96=P.independent_mass(mean,cov,bounds,.25,96)
    for temperature in P.TEMPERATURES:
        p=P.probability(adaptive,content,temperature,bounds)
        np.testing.assert_allclose(P.probability(r64,content,temperature,bounds),p,atol=1e-11,rtol=0)
        np.testing.assert_allclose(P.probability(r96,content,temperature,bounds),p,atol=1e-11,rtol=0)


def test_original_key_floor_area_added_once_before_content():
    bounds=np.array([[0.,0.,1.,1.],[1.,0.,3.,1.]])
    p=P.probability(np.zeros((1,2)),np.zeros((1,2)),10.,bounds)
    np.testing.assert_allclose(p,[[1/3,2/3]],atol=2e-15,rtol=0)


def test_near_tie_argmax_changes_are_retained_and_described_not_labelled_correct():
    reference=np.array([.500001,.499999]);actual=reference[::-1].copy()
    row=P.row_comparison(actual,reference,reference.copy(),reference.copy())
    assert row['argmax_changed'] and row['actual_argmax']==1 and row['reference_argmax']==0
    assert row['reference_top_gap']==pytest.approx(2e-6)
    assert row['actual_winner_margin_against_reference']==pytest.approx(2e-6)
    assert row['reference_winner_margin_against_actual']==pytest.approx(2e-6)
    assert row['max_probability_error']==pytest.approx(2e-6)
    assert all('correct' not in key for key in row)


def test_fixed_population_counts_and_no_label_fields():
    assert P.FIELDS==('q_to_k_xy','q_to_k_pixel_cov','q_features','k_features')
    supports=len(P.SCENES)*sum(len(P.selected_indices(s)) for s in P.STRIDES)
    assert supports==72 and supports*len(P.WIDTHS)*len(P.METHODS)==576
    assert supports*len(P.WIDTHS)*len(P.METHODS)*len(P.TEMPERATURES)==1728
