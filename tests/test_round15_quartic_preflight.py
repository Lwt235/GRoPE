import numpy as np
import pytest
from experiments.round15_quartic_preflight import synthetic_geometry,ROLES


@pytest.mark.parametrize('stride',[8,16,24])
def test_synthetic_identity_same_parent_and_moment_conservation(stride):
    g=synthetic_geometry(stride);roles=g['roles']
    assert tuple(roles)==ROLES and len(g['key_bounds'])==432
    parent=roles['full1'];full=roles['full4'];quad=roles['quad16']
    assert len(parent['means'])==144*192//stride**2
    np.testing.assert_array_equal(parent['means'],roles['quartic']['means'])
    np.testing.assert_array_equal(parent['cov'],roles['quartic']['cov'])
    mean=(full['means']*full['weights'][...,None]).sum(1)
    delta=full['means']-mean[:,None]
    cov=((full['cov']+delta[...,None]*delta[...,None,:])*full['weights'][...,None,None]).sum(1)
    np.testing.assert_allclose(mean,parent['means'][:,0],atol=1e-14,rtol=0)
    np.testing.assert_allclose(cov,parent['cov'][:,0],atol=1e-14,rtol=0)
    assert full['means'].shape[1]==4 and quad['means'].shape[1]==16
    assert np.all(quad['cov'][:,0,0,0]<parent['cov'][:,0,0,0])
    assert np.all(g['key_bounds'][:,2:]-g['key_bounds'][:,:2]==1)
