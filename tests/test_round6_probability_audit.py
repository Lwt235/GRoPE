import math
import numpy as np
import pytest
from experiments.round6_probability_audit import (
    gaussian_energy,stable_log_probability,probability_metrics,pck_bounds,
)


def inputs():
    return {'q_to_k':np.array([[0.,0.]]),'q_to_kcov':np.array([np.diag([1.,3.])]),
        'qcov':np.array([np.eye(2)*5]),'kcov':np.array([np.eye(2)*2]*2),
        'kmean':np.array([[0.,0.],[2.,0.]]),'q':np.array([[1.,0.]]),
        'k':np.array([[1.,0.],[0.,1.]])}


@pytest.mark.parametrize('method,last',[('full',-.5),('fixed',-.25),('isotropic',-.4),('point',-2.)])
def test_independent_gaussian_energy_controls(method,last):
    score=gaussian_energy(inputs(),{'method':method,'bandwidth':1.,'temperature':3.})
    np.testing.assert_allclose(score,[[3.,last]],atol=1e-14)


def labels():
    return {'soft_labels':np.array([[.5,.5]]),'gt_valid':np.array([True]),
        'visible_fraction':np.array([1.]),'sample_count':np.array([64]),
        'point_gt_valid':np.array([True]),'point_pixcontinuous':np.array([[8.,0.]]),
        'point_key_index':np.array([1]),'key_grid':np.array([[0.,0.],[8.,0.]]),'k_stride':8}


def test_zero_probability_does_not_fake_restore_ce_with_log_clamp():
    lp=np.array([[0.,-1000.]])
    p=np.exp(lp)
    unresolved=probability_metrics(p,labels())
    assert unresolved['ce_sum'] is None and unresolved['kl_sum'] is None
    assert unresolved['unresolved_area_ce_rows']==1 and unresolved['unresolved_point_ce_rows']==1
    restored=probability_metrics(p,labels(),lp)
    assert restored['ce_sum']==32000
    assert restored['kl_sum']==pytest.approx((500-math.log(2))*64)
    assert restored['point_ce_sum']==1000
    assert math.isfinite(restored['js_sum'])


def test_stable_logsumexp_is_shift_invariant_without_underflow_loss():
    score=np.array([[-10000.,-10001.]])
    reference=np.log(np.array([[1.,math.exp(-1)]]));reference-=math.log(1+math.exp(-1))
    np.testing.assert_allclose(stable_log_probability(score),reference,atol=1e-13)
    np.testing.assert_allclose(stable_log_probability(score),stable_log_probability(score+1e6),atol=1e-13)


def test_pck_tie_bounds_report_ambiguity_instead_of_changing_counts_silently():
    d=labels();d['key_grid']=np.array([[0.,0.],[16.,0.]])
    d['point_pixcontinuous']=np.array([[0.,0.]])
    bounds,ties=pck_bounds(np.array([[.5,.5]]),d)
    assert ties==1 and bounds['pck8_count']==[0,1]
    assert bounds['point_epe_sum']==[0.,16.]
