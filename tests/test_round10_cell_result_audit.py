"""Independent replay reference: actual cells, tails, mixture and area floor."""
import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import _mvn

from experiments.round10_cell_result_audit import (
    FLOOR, normal_interval, reference_logits, reference_mass, scalar_reference,
    transformed_covariance,
)


@pytest.mark.parametrize('rho', [-.99,0.,.83])
def test_positive_reference_matches_independent_adaptive_and_scipy_bivariate(rho):
    means=np.array([[.2,-.3],[2.,1.5]])
    cov=np.array([[[.5,rho*.5],[rho*.5,.5]],[[2.,rho],[rho,.5]]])
    bounds=np.array([[-.5,-.5,.5,.5],[1.,1.,2.,3.],[-3.,-2.,-1.,0.]])
    mass=reference_mass(means,cov,bounds,.25)
    for i in range(2):
        for j in range(3):
            scalar,error=scalar_reference(means[i],cov[i],bounds[j],.25)
            scipy,status=_mvn.mvnun(bounds[j,:2],bounds[j,2:],means[i],cov[i]+.25**2*np.eye(2),100000,1e-13)
            assert status==0 and error<1e-10
            assert mass[i,j]==pytest.approx(scalar,abs=1e-13,rel=1e-11)
            assert mass[i,j]==pytest.approx(scipy,abs=1e-13,rel=1e-10)


def test_extreme_tail_keeps_positive_mass_below_common_absolute_tolerances():
    mean=np.array([[0.,0.]])
    cov=np.array([[[1.,0.],[0.,1.]]])
    bounds=np.array([[8.,-.5,8.2,.5],[10.,-.5,10.1,.5]])
    sigma=np.sqrt(1+.25**2)
    exact=(ndtr(-bounds[:,0]/sigma)-ndtr(-bounds[:,2]/sigma))*(ndtr(bounds[:,3]/sigma)-ndtr(bounds[:,1]/sigma))
    actual=reference_mass(mean,cov,bounds,.25)[0]
    assert np.all(actual>0)
    np.testing.assert_allclose(actual,exact,rtol=2e-13,atol=0.)
    for j in range(2):
        scalar,error=scalar_reference(mean[0],cov[0],bounds[j],.25)
        assert scalar==pytest.approx(exact[j],rel=2e-13,abs=0.)


def test_rectangle_key_subdivision_and_density_floor_are_additive():
    query=dict(mean=np.array([[.3,-.2],[1.2,.5]]),
        cov=np.array([[[.7,.2],[.2,.3]],[[.3,-.2],[-.2,.7]]]),mass=np.array([256.,256.]))
    parent=np.array([[0.,0.,2.,2.]])
    child=np.array([[0.,0.,1.,1.],[1.,0.,2.,1.],[0.,1.,1.,2.],[1.,1.,2.,2.]])
    a=np.exp(reference_logits(query,dict(mass=np.array([256.])),parent,'full',.5))
    b=np.exp(reference_logits(query,dict(mass=np.full(4,64.)),child,'full',.5))
    np.testing.assert_allclose(a[:,0],b.sum(-1),atol=1e-15,rtol=1e-13)
    remote={**query,'mean':query['mean']+1e4}
    af=np.exp(reference_logits(remote,dict(mass=np.array([256.])),parent,'full',.5))
    bf=np.exp(reference_logits(remote,dict(mass=np.full(4,64.)),child,'full',.5))
    np.testing.assert_allclose(af[:,0],FLOOR*4,rtol=1e-14,atol=0)
    np.testing.assert_allclose(af[:,0],bf.sum(-1),rtol=1e-14,atol=0)


def test_quadrature_weights_average_mass_not_logarithm():
    query=dict(nodes=np.array([[[0.,0.],[1.,1.]]]),node_cov=np.zeros((1,2,2,2)),
        node_weights=np.array([[.2,.8]]))
    boxes=np.array([[-.5,-.5,.5,.5],[.5,.5,1.5,1.5]])
    key=dict(mass=np.array([64.,64.]))
    expected=(reference_mass(query['nodes'],query['node_cov'],boxes,.5)*query['node_weights'][:,:,None]).sum(1)+FLOOR
    np.testing.assert_allclose(np.exp(reference_logits(query,key,boxes,'quad4',.5)),expected,rtol=1e-14)


def test_query_only_controls_preserve_mean_and_native_mass_semantics():
    query=dict(cov=np.array([[[2.,.2],[.2,.5]]]),mass=np.array([128.]))
    np.testing.assert_array_equal(transformed_covariance(query,'point'),np.zeros((1,2,2)))
    np.testing.assert_array_equal(transformed_covariance(query,'center'),transformed_covariance(query,'point'))
    np.testing.assert_allclose(transformed_covariance(query,'fixed'),np.eye(2)[None]/6)
    np.testing.assert_allclose(transformed_covariance(query,'isotropic'),np.eye(2)[None]*1.25)
    np.testing.assert_allclose(np.linalg.eigvalsh(transformed_covariance(query,'rotated')),np.linalg.eigvalsh(query['cov']))
