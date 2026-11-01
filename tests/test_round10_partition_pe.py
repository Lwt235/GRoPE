"""Independent density, quadrature, partition-measure and gradient checks."""
import inspect

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import torch

from models.round10_partition_pe import (
    common_prediction, geometry_logits, group_mean, pool_content, prepare_support,
)


DTYPE = torch.float64


def tensor(value):
    return torch.as_tensor(value, dtype=DTYPE)


def banks():
    query = dict(mean=tensor([[.2,-.1],[1.3,.9]]),
                 cov=tensor([[[.5,.15],[.15,.2]],[[.1,-.03],[-.03,.8]]]),
                 mass=tensor([64,256]), parent=torch.tensor([0,1]))
    key = dict(mean=tensor([[-.4,.3],[.7,-.2],[1.2,1.8]]),
               cov=tensor([[[.2,0],[0,.08]],[[.3,.04],[.04,.15]],[[.07,0],[0,.5]]]),
               mass=tensor([64,128,256]))
    return query, key


def scipy_overlap(query, key, method, bandwidth):
    qm, km = query['mean'].numpy(), key['mean'].numpy()
    qc, kc = query['cov'].numpy().copy(), key['cov'].numpy().copy()
    if method == 'point':
        qc[:] = 0; kc[:] = 0
    elif method == 'center':
        qc[:] = 0
    elif method == 'isotropic':
        qc = np.trace(qc, axis1=1, axis2=2)[:,None,None]*np.eye(2)/2
        kc = np.trace(kc, axis1=1, axis2=2)[:,None,None]*np.eye(2)/2
    elif method == 'fixed':
        qc = query['mass'].numpy()[:,None,None]*np.eye(2)/(64*12)
        kc = key['mass'].numpy()[:,None,None]*np.eye(2)/(64*12)
    elif method == 'rotated':
        rotation = np.array([[0.,-1.],[1.,0.]])
        qc = rotation@qc@rotation.T
    output = np.empty((len(qm),len(km)))
    for i in range(len(qm)):
        for j in range(len(km)):
            output[i,j] = (multivariate_normal.logpdf(qm[i],mean=km[j],
                cov=qc[i]+kc[j]+bandwidth**2*np.eye(2)) + np.log(key['mass'][j]))
    return output


@pytest.mark.parametrize('method',['point','center','fixed','isotropic','full','rotated'])
@pytest.mark.parametrize('bandwidth',[.25,2.])
def test_each_control_matches_scipy_density_with_variable_key_covariance(method,bandwidth):
    query,key = banks()
    actual = geometry_logits(query,key,method,bandwidth).numpy()
    np.testing.assert_allclose(actual,scipy_overlap(query,key,method,bandwidth),
                               atol=3e-13,rtol=2e-14)


def support_input():
    yy,xx = np.mgrid[:16,:32]
    # Nonlinear spatial field makes omitted within-stratum scatter measurable.
    xy = np.stack([xx/8+.03*(yy/8)**2, yy/8+.02*np.sin(xx)],-1).reshape(-1,2)
    cov = np.zeros((512,2,2))
    cov[:,0,0] = .001+xx.ravel()/100000
    cov[:,1,1] = .002+yy.ravel()/100000
    owner = torch.tensor((xx//16).ravel())
    bounds = torch.tensor([[0,0,16,16],[16,0,32,16]])
    return tensor(xy),tensor(cov),owner,bounds,torch.tensor([0,1])


@pytest.mark.parametrize('samples',[1,4,16])
def test_support_moments_and_strata_match_direct_pixel_slices(samples):
    xy,cov,owner,bounds,parent = support_input()
    result = prepare_support(xy,cov,owner,bounds,parent,samples=samples)
    dense_xy = xy.numpy().reshape(16,32,2)
    dense_cov = cov.numpy().reshape(16,32,2,2)
    for region,x0 in enumerate([0,16]):
        points = dense_xy[:,x0:x0+16].reshape(-1,2)
        mean = points.mean(0)
        reference = np.cov(points,rowvar=False,bias=True)+dense_cov[:,x0:x0+16].mean((0,1))
        np.testing.assert_allclose(result['mean'][region],mean,atol=2e-15)
        np.testing.assert_allclose(result['cov'][region],reference,atol=2e-15)
        assert result['mass'][region] == 256
        if samples == 1:
            assert 'nodes' not in result
            continue
        side = int(np.sqrt(samples)); size = 16//side
        expected_nodes, expected_cov = [],[]
        for sy in range(side):
            for sx in range(side):
                ys,xs = slice(sy*size,(sy+1)*size),slice(x0+sx*size,x0+(sx+1)*size)
                expected_nodes.append(dense_xy[ys,xs].mean((0,1)))
                expected_cov.append(dense_cov[ys,xs].mean((0,1)))
        np.testing.assert_allclose(result['nodes'][region],expected_nodes,atol=2e-15)
        np.testing.assert_allclose(result['node_cov'][region],expected_cov,atol=2e-15)
        np.testing.assert_allclose(result['node_weights'][region],1/samples,atol=0)
        nodes = result['nodes'][region].numpy()
        np.testing.assert_allclose(nodes.mean(0),mean,atol=2e-15)
        mixture_cov = np.cov(nodes,rowvar=False,bias=True)+np.mean(expected_cov,axis=0)
        # Quadrature is not silently a within-stratum Gaussian mixture: that
        # different baseline would preserve the full covariance exactly.
        omitted = reference-mixture_cov
        assert np.linalg.eigvalsh(omitted).min() > .001


@pytest.mark.parametrize('samples',[4,16])
def test_quadrature_is_weighted_density_mixture_not_average_log_density(samples):
    query = prepare_support(*support_input(),samples=samples)
    _,key = banks(); width=.5
    actual = geometry_logits(query,key,'quad'+str(samples),width).numpy()
    reference = np.zeros_like(actual)
    for i in range(len(query['mean'])):
        for j in range(len(key['mean'])):
            components=[]
            for s in range(samples):
                components.append(multivariate_normal.logpdf(query['nodes'][i,s].numpy(),
                    mean=key['mean'][j].numpy(),
                    cov=query['node_cov'][i,s].numpy()+key['cov'][j].numpy()+width**2*np.eye(2))
                    +np.log(query['node_weights'][i,s].item()))
            reference[i,j]=logsumexp(components)+np.log(key['mass'][j].item())
    np.testing.assert_allclose(actual,reference,atol=2e-13,rtol=2e-14)


@pytest.mark.parametrize('samples',[4,16])
def test_minimal_quadrature_preparation_omits_unused_scatter_without_changing_output(samples):
    complete=prepare_support(*support_input(),samples=samples)
    minimal=prepare_support(*support_input(),samples=samples,include_full_moments=False)
    assert 'cov' not in minimal
    for field in minimal:
        torch.testing.assert_close(minimal[field],complete[field])
    _,key=banks()
    torch.testing.assert_close(geometry_logits(minimal,key,'quad'+str(samples),.5),
                               geometry_logits(complete,key,'quad'+str(samples),.5))


@pytest.mark.parametrize('method',['point','center','fixed','isotropic','full','rotated','quad4','quad16'])
def test_key_mass_global_scaling_leaves_attention_unchanged(method):
    query,key=banks()
    if method.startswith('quad'):
        query=prepare_support(*support_input(),samples=int(method[4:]))
    changed={**key,'mass':key['mass']*11}
    # fixed derives covariance from native mass, so changing mass alone also
    # changes declared support geometry; its mass-only identity is inapplicable.
    if method=='fixed':
        assert not torch.allclose(geometry_logits(query,key,method,.5).softmax(-1),
                                  geometry_logits(query,changed,method,.5).softmax(-1))
    else:
        original=geometry_logits(query,key,method,.5)
        scaled=geometry_logits(query,changed,method,.5)
        torch.testing.assert_close(scaled-original,torch.full_like(original,np.log(11.)),atol=1e-14,rtol=1e-14)
        torch.testing.assert_close(original.softmax(-1),scaled.softmax(-1),atol=1e-14,rtol=1e-14)


@pytest.mark.parametrize('method',['point','center','isotropic','full','rotated','quad4','quad16'])
def test_duplicate_key_mass_splitting_preserves_value_transport(method):
    query,key=banks()
    if method.startswith('quad'):
        query=prepare_support(*support_input(),samples=int(method[4:]))
    index=torch.tensor([0,1,1,2])
    split={field:value[index].clone() for field,value in key.items()}
    split['mass'][1:3] *= tensor([.25,.75])
    values=tensor([[.4,2.],[-1.,.3],[.5,.9]])
    original=geometry_logits(query,key,method,.5).softmax(-1)@values
    duplicated=geometry_logits(query,split,method,.5).softmax(-1)@values[index]
    torch.testing.assert_close(original,duplicated,atol=3e-14,rtol=3e-14)


def test_common_grid_aggregation_preserves_mass_with_actual_mixed_cell_sizes():
    # Two 16px query parents: four 8px children, and two horizontal half-cells.
    query=dict(mass=tensor([64,64,64,64,128,128]),parent=torch.tensor([0,0,0,0,1,1]))
    # Target: one 16px cell, two 8x16 half-cells, four 8px cells.
    key=dict(mass=tensor([256,128,128,64,64,64,64]))
    atom_owner=torch.tensor([0,0,0,0,1,1,2,2,3,4,5,6])
    rng=np.random.default_rng(93)
    p=rng.uniform(.1,1,(6,7));p/=p.sum(-1,keepdims=True)
    expected=np.zeros((2,12))
    for q in range(6):
        for atom,k in enumerate(atom_owner.tolist()):
            expected[query['parent'][q],atom]+=p[q,k]*64/key['mass'][k].item()*query['mass'][q].item()/256
    actual=common_prediction(tensor(p),query,key,atom_owner,parents=2)
    np.testing.assert_allclose(actual,expected,atol=1e-16)
    np.testing.assert_allclose(actual.sum(-1),1,atol=2e-16)
    poisoned_query={**query,'gt_visible':torch.zeros(6),'sensor_counts':tensor([0,0,0,0,1,99])}
    poisoned_key={**key,'gt_labels':torch.full((6,7),float('nan'))}
    torch.testing.assert_close(actual,common_prediction(tensor(p),poisoned_query,poisoned_key,atom_owner,parents=2))
    assert not any('gt' in name or 'visible' in name for name in inspect.signature(common_prediction).parameters)


def test_same_content_duplicate_query_subdivision_preserves_common_prediction():
    key=dict(mass=tensor([64,64,64,64]));atoms=torch.arange(4)
    p=tensor([[.1,.2,.3,.4]])
    parent=dict(mass=tensor([256]),parent=torch.tensor([0]))
    children=dict(mass=tensor([64,64,128]),parent=torch.tensor([0,0,0]))
    torch.testing.assert_close(common_prediction(p,parent,key,atoms,parents=1),
        common_prediction(p.expand(3,-1),children,key,atoms,parents=1))
    # This is duplicate conditional-row splitting, not an assertion that
    # different actual child supports have the same probability distribution.


def test_full_density_transport_has_correct_geometry_and_bandwidth_gradients():
    query,key=banks()
    qa=torch.linalg.cholesky(query['cov']).requires_grad_()
    ka=torch.linalg.cholesky(key['cov']).requires_grad_()
    qm=query['mean'].clone().requires_grad_();km=key['mean'].clone().requires_grad_()
    bandwidth=tensor(.6).requires_grad_()
    values=tensor([[.2,-1.],[2.,.3],[-.5,1.4]])
    def operation(qm,qa,km,ka,bw):
        q={**query,'mean':qm,'cov':qa@qa.transpose(-1,-2)}
        k={**key,'mean':km,'cov':ka@ka.transpose(-1,-2)}
        return geometry_logits(q,k,'full',bw).softmax(-1)@values
    assert torch.autograd.gradcheck(operation,(qm,qa,km,ka,bandwidth),eps=1e-6,atol=2e-6,rtol=2e-5)
    loss=operation(qm,qa,km,ka,bandwidth).square().sum()
    grads=torch.autograd.grad(loss,(qm,qa,km,ka,bandwidth))
    assert all(torch.isfinite(g).all() and g.abs().max()>1e-7 for g in grads)


def test_quadrature_and_support_preparation_retain_pixel_gradients():
    xy,cov,owner,bounds,parent=support_input()
    xy=xy.clone().requires_grad_();cov=cov.clone().requires_grad_()
    query=prepare_support(xy,cov,owner,bounds,parent,samples=4)
    _,key=banks()
    value=geometry_logits(query,key,'quad4',.5).softmax(-1)@tensor([[1.],[-2.],[.3]])
    grad_xy,grad_cov=torch.autograd.grad(value.square().sum(),(xy,cov))
    assert torch.isfinite(grad_xy).all() and torch.isfinite(grad_cov).all()
    assert (grad_xy.abs().sum(-1)>0).all()
    assert (grad_cov.abs().sum((-1,-2))>0).all()


def test_normalized_content_pool_matches_direct_mean_and_handles_zero_content():
    features=tensor([[1.,2.],[3.,4.],[0.,0.]])
    owner=torch.tensor([0,0,1])
    expected=tensor([[2.,3.],[0.,0.]])
    expected[0]/=expected[0].norm()
    torch.testing.assert_close(pool_content(features,owner,2),expected)


def test_empty_support_groups_and_invalid_quadrature_budgets_fail_closed():
    with pytest.raises(ValueError,match='empty support group'):
        group_mean(torch.empty((0,2),dtype=DTYPE),torch.empty(0,dtype=torch.long),1)
    with pytest.raises(ValueError,match='empty support group'):
        group_mean(tensor([[1.,2.]]),torch.tensor([0]),2)
    with pytest.raises(ValueError,match='square quadrature budget'):
        prepare_support(*support_input(),samples=3)
    # More strata than pixels must not generate NaN mixture nodes.
    with pytest.raises(ValueError,match='empty support group'):
        prepare_support(tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]),
            torch.eye(2,dtype=DTYPE).expand(4,-1,-1),torch.zeros(4,dtype=torch.long),
            torch.tensor([[0,0,2,2]]),torch.tensor([0]),samples=16)


@pytest.mark.parametrize('bad_cov',[np.zeros((2,2)),np.full((2,2),np.nan),np.diag([-1.,1.])])
def test_nonpositive_or_nonfinite_pair_covariance_is_rejected(bad_cov):
    query,key=banks()
    q={**query,'cov':tensor(bad_cov).expand(2,-1,-1)}
    k={**key,'cov':torch.zeros_like(key['cov'])}
    with pytest.raises(ValueError,match='nonpositive pair covariance'):
        geometry_logits(q,k,'full',0.)


def test_unknown_control_fails_closed():
    query,key=banks()
    with pytest.raises(ValueError):geometry_logits(query,key,'candidate_chosen_after_test',.5)
