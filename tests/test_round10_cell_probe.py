"""B dispatch, numerical floor and unchanged protocol helper contracts."""
import pytest
import torch

from experiments import round10_cell_probe as B
from experiments import round10_partition_probe as A
from models.round10_cell_integral_fast import gaussian_cell_mass_fast


def tensor(value):return torch.tensor(value,dtype=torch.float64)


def example():
    mean=tensor([[.2,.3],[1.4,.1]])
    cov=tensor([[[.5,.2],[.2,.3]],[[.3,-.1],[-.1,.2]]])
    q=dict(mean=mean,cov=cov,mass=tensor([64.,64.]))
    k=dict(mean=tensor([[.5,.5],[1.5,.5]]),cov=torch.eye(2,dtype=torch.float64).repeat(2,1,1)/12,
        mass=tensor([64.,64.]),cell_bounds=tensor([[0.,0.,1.,1.],[1.,0.,2.,1.]]))
    return q,k


def test_full_is_native_rectangle_mass_without_double_area():
    q,k=example();diagnostics={}
    logits=B.geometry_logits(q,k,'full',.25,diagnostics=diagnostics)
    expected=gaussian_cell_mass_fast(q['mean'],q['cov'],k['cell_bounds'])+1e-30*(k['mass']/64)[None]
    torch.testing.assert_close(logits.exp(),expected,rtol=2e-15,atol=2e-15)
    assert diagnostics['query_rows']==2
    assert diagnostics['raw_zero_fov_rows']==0
    assert diagnostics['floor_fov_mass']==2e-30


def test_point_center_positive_control_and_diagonal_controls():
    q,k=example()
    point=B.geometry_logits(q,k,'point',.25)
    center=B.geometry_logits(q,k,'center',.25)
    torch.testing.assert_close(point,center,rtol=0,atol=0)
    for method in ('fixed','isotropic','rotated','full'):
        value=B.geometry_logits(q,k,method,.25)
        assert torch.isfinite(value).all()
    with pytest.raises(ValueError):B.geometry_logits(q,k,'unknown',.25)


def test_quadrature_mass_mixture_preserves_node_weights():
    q,k=example()
    q['nodes']=q['mean'][:,None].expand(-1,4,-1).clone()
    q['nodes'][:,0,0]-=.1
    q['node_cov']=q['cov'][:,None].expand(-1,4,-1,-1)
    q['node_weights']=tensor([[.1,.2,.3,.4],[.4,.3,.2,.1]])
    result=B.geometry_logits(q,k,'quad4',.5).exp()
    expected=sum(q['node_weights'][:,i,None]*gaussian_cell_mass_fast(q['nodes'][:,i],q['node_cov'][:,i],
        k['cell_bounds'],bandwidth=.5) for i in range(4))+1e-30*(k['mass']/64)[None]
    torch.testing.assert_close(result,expected,rtol=2e-15,atol=2e-15)


def test_uniform_density_fallback_preserves_area_and_records_zeros():
    q,k=example();q['mean']=tensor([[1000.,1000.],[1000.,1000.]])
    k['cell_bounds']=tensor([[0.,0.,1.,1.],[1.,0.,3.,1.]])
    k['mass']=tensor([64.,128.])
    diagnostics={}
    probability=B.geometry_logits(q,k,'point',.25,diagnostics=diagnostics).softmax(-1)
    torch.testing.assert_close(probability,tensor([[1/3,2/3],[1/3,2/3]]),rtol=2e-14,atol=2e-14)
    assert diagnostics['raw_zero_fov_rows']==diagnostics['raw_tiny_fov_rows']==2
    assert diagnostics['raw_zero_cell_count']==4


def test_benchmark_patch_changes_only_runtime_bindings_and_restores():
    before_key,before_geometry=A.prepare_key,A.geometry_logits
    with B.benchmark_dispatch():
        assert A.geometry_logits is B.geometry_logits
        assert A.prepare_key is not before_key
    assert A.prepare_key is before_key
    assert A.geometry_logits is before_geometry


def test_calibration_budget_and_tie_order_unchanged():
    configs=B.configurations()
    assert len(configs)==7*4*3
    assert tuple(B.METHODS)==tuple(A.METHODS)
    assert list(configs)[0]=='point_h0.25_t0.0'
    assert B.WIDTHS==A.WIDTHS and B.TEMPERATURES==A.TEMPERATURES
