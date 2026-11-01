import numpy as np
import pytest
import torch
from experiments import round7_scale_probe as p
from experiments import round7_scale_data as d


def row(scene,grid,kl,mass):
    r=dict(scene=scene,grid=grid)
    for name,(metric,denom) in p.METRICS.items():r[metric]=kl*mass;r[denom]=mass
    return r


def test_macro_uses_area_within_place_then_equal_places():
    rows=[row('a','g',1.,1.),row('a','g',3.,3.),row('b','g',10.,100.)]
    g=p.group_metrics(rows)
    assert g['a','g']['kl']==2.5
    assert g['b','g']['kl']==10.
    result=p.paired_description(g,g,['a','b'],['g'],True)
    assert result['full']==6.25
    assert result['relative']==0
    assert result['descriptive_relative_ci95']==[0.,0.]


def test_paired_ci_keeps_place_group_and_all_conditions_together():
    a={(s,g):{'kl':float(v)} for s,v in [('a',2),('b',4),('c',8)] for g in ('g1','g2')}
    b={key:{'kl':v['kl']*2} for key,v in a.items()}
    out=p.paired_description(a,b,['a','b','c'],['g1','g2'],True)
    assert out['improving_places']==3
    np.testing.assert_allclose(out['descriptive_relative_ci95'],[-.5,-.5])


def test_24_fraction_keeps_exact_integer_mass_and_no_forward_gt():
    depth=np.full((48,48),2.);depth[0,0]=0
    K=np.array([[100.,0.,23.5],[0.,100.,23.5],[0.,0.,1.]])
    labels=d.region_labels(depth,depth,K,K,np.eye(4),np.eye(4),24,24,(48,48))
    batch=p.tensor_labels(labels,{'kmean':torch.zeros(1,4,2)})
    assert batch['visible_fraction'].dtype==torch.float64
    assert batch['point_key_index'].dtype==torch.int64
    np.testing.assert_allclose((batch['visible_fraction']*batch['sample_count']).numpy()[0],d.exact_visible_mass(labels),atol=1e-12)
    _,_,measure=p.frozen_helpers()
    rows=measure(torch.zeros(1,4,4),batch,[dict(k_stride=24)])
    assert rows[0]['visible_mass']==2303


def test_array_hash_is_order_invariant_but_shape_sensitive():
    a=np.arange(4,dtype=np.float32)
    assert p.array_sha({'a':a,'b':a+1})==p.array_sha({'b':a+1,'a':a})
    assert p.array_sha({'a':a})!=p.array_sha({'a':a.reshape(2,2)})


def test_frozen_helpers_execute_sealed_archive_and_role_configuration():
    lock,model,measure=p.frozen_helpers()
    assert lock['roles']['full']['config']=={'method':'full','bandwidth':.5,'temperature':5.}
    assert len(list(model(**lock['roles']['full']['config']).parameters()))==0
    assert measure.__code__.co_filename.endswith('source/experiments/round6_train.py')
    assert p.GRIDS==('q8_k24','q24_k8','q16_k24','q24_k16','q24_k24')
