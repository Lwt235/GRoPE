"""Independent analytic probability, area-weighting, mask and macro checks."""
import math

import numpy as np
import pytest
import torch

from experiments.round6_train import per_query, measure_rows, macro, evaluate
from models.round6_measure_attention import FIELDS


def numpy_metrics(p, y):
    p,y=np.asarray(p,float),np.asarray(y,float)
    midpoint=(p+y)/2
    entropy=-np.sum(y[y>0]*np.log(y[y>0]))
    ce=-np.sum(y[y>0]*np.log(p[y>0]))
    js=.5*np.sum(y[y>0]*np.log(y[y>0]/midpoint[y>0]))
    js+=.5*np.sum(p[p>0]*np.log(p[p>0]/midpoint[p>0]))
    return ce,entropy,js


@pytest.mark.parametrize('p,y',[
    ([.5,.5],[.25,.75]),([.2,.3,.5],[1.,0.,0.]),([.15,.25,.6],[.15,.25,.6]),
])
def test_kl_js_match_independent_probability_formula(p,y):
    logits=torch.tensor(np.log(p)[None,None,:],dtype=torch.float64)
    labels={'soft_labels':torch.tensor([[y]],dtype=torch.float64)}
    ce,entropy,js=per_query(logits,labels)
    expected=numpy_metrics(p,y)
    np.testing.assert_allclose([ce.item(),entropy.item(),js.item()],expected,atol=1e-12,rtol=0)
    assert abs((ce-entropy).item()-(expected[0]-expected[1]))<1e-12
    assert -.000000000001<=js.item()<=math.log(2)


def metric_example():
    probability=np.array([[[.8,.2],[.25,.75],[.9,.1]]])
    logits=torch.tensor(np.log(probability),dtype=torch.float64)
    d={'soft_labels':torch.tensor([[[1.,0.],[0.,1.],[0.,0.]]],dtype=torch.float64),
       'gt_valid':torch.tensor([[True,True,False]]),
       'visible_fraction':torch.tensor([[1/64,3/64,0]],dtype=torch.float64),
       'sample_count':torch.tensor([[64.,64.,64.]],dtype=torch.float64),
       'kmean':torch.tensor([[[3.5,3.5],[11.5,3.5]]],dtype=torch.float64)/8,
       'point_gt_valid':torch.tensor([[False,True,True]]),
       'point_pixcontinuous':torch.tensor([[[float('nan'),float('nan')],[11.5,3.5],[11.5,3.5]]],dtype=torch.float64),
       'point_key_index':torch.tensor([[-1,1,1]])}
    ids=[{'scene':'room','grid':'q8_k8','q_stride':8,'k_stride':8}]
    return logits,d,ids


def test_visible_source_pixel_mass_and_independent_point_population():
    logits,d,ids=metric_example()
    row=measure_rows(logits,d,ids)[0]
    ce0,_,js0=numpy_metrics([.8,.2],[1.,0.])
    ce1,_,js1=numpy_metrics([.25,.75],[0.,1.])
    assert row['n_valid']==2 and row['visible_mass']==4
    assert row['ce_sum']==pytest.approx(ce0+3*ce1,abs=1e-12)
    assert row['kl_sum']==pytest.approx(ce0+3*ce1,abs=1e-12)
    assert row['js_sum']==pytest.approx(js0+3*js1,abs=1e-12)
    assert row['row_kl_sum']==pytest.approx(ce0+ce1,abs=1e-12)
    assert row['entropy_sum']==0
    # Invalid point coords/sentinel do not pollute metrics. Conversely, an
    # area-invalid but point-valid row still contributes to secondary PCK/CE.
    assert row['n_point']==2
    assert row['pck8_count']==2 and row['pck4_count']==1 and row['pck_stride_count']==2
    assert row['point_ce_sum']==pytest.approx(-math.log(.75)-math.log(.1),abs=1e-12)
    assert row['point_epe_sum']==pytest.approx(8.)
    assert all(math.isfinite(value) for key,value in row.items() if isinstance(value,(float,int)))


def test_point_pck_threshold_tracks_target_stride_and_only_point_mask():
    logits,d,ids=metric_example()
    d['point_pixcontinuous'][0,2]=torch.tensor([19.5,3.5])
    ids[0]['k_stride']=16
    row=measure_rows(logits,d,ids)[0]
    assert row['pck8_count']==1 and row['pck_stride_count']==2
    d['point_gt_valid'].zero_()
    row=measure_rows(logits,d,ids)[0]
    assert row['n_point']==0 and row['point_ce_sum']==0 and row['point_epe_sum']==0
    assert row['visible_mass']==4 and row['n_valid']==2


def test_scene_condition_macro_uses_pixel_mass_within_then_equal_groups():
    rows=[{'scene':'a','grid':'q8_k8','kl_sum':1.,'visible_mass':1.},
          {'scene':'a','grid':'q8_k8','kl_sum':3.,'visible_mass':1.},
          {'scene':'a','grid':'q16_k8','kl_sum':100.,'visible_mass':10.},
          {'scene':'b','grid':'q8_k8','kl_sum':300.,'visible_mass':100.},
          {'scene':'b','grid':'q16_k8','kl_sum':20.,'visible_mass':1.}]
    assert macro(rows)==pytest.approx((2+10+3+20)/4)
    assert macro(rows,grids=['q8_k8'])==pytest.approx((2+3)/2)
    assert macro(rows)!=pytest.approx(sum(r['kl_sum'] for r in rows)/sum(r['visible_mass'] for r in rows))
    # Splitting an observation with the same numerator/denominator preserves it.
    split=rows[:-1]+[dict(rows[-1],kl_sum=10.,visible_mass=.5)]*2
    assert macro(split)==macro(rows)
    with pytest.raises(ValueError,match='empty'):
        macro([{'scene':'a','grid':'q8_k8','kl_sum':0.,'visible_mass':0.}])


def test_extreme_finite_scores_and_zero_target_entries_are_numerically_finite():
    logits=torch.tensor([[[10000.,-10000.],[-10000.,10000.]]])
    labels={'soft_labels':torch.tensor([[[0.,1.],[0.,1.]]])}
    ce,entropy,js=per_query(logits,labels)
    assert torch.isfinite(ce).all() and torch.isfinite(entropy).all() and torch.isfinite(js).all()
    torch.testing.assert_close(ce,torch.tensor([[20000.,0.]]))
    torch.testing.assert_close(js,torch.tensor([[math.log(2),0.]]))


def test_evaluator_exposes_only_forward_fields_to_model():
    logits,d,ids=metric_example()
    for name in FIELDS:
        if name=='kmean':continue
        n=2 if name in ['k','kcov','k_to_q','k_to_qcov'] else 3
        tail=(128,) if name in ['q','k'] else (2,2) if name.endswith('cov') else (2,)
        d[name]=torch.zeros((1,n,*tail),dtype=torch.float64)
    class Capture(torch.nn.Module):
        def forward(self,inputs):
            assert set(inputs)==set(FIELDS)
            assert not any('valid' in name or 'label' in name or 'visible' in name for name in inputs)
            return logits
    rows=evaluate(Capture(),{'q8_k8':d},{'q8_k8':ids})
    assert len(rows)==1 and rows[0]['visible_mass']==4
