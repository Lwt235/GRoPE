"""Reference-level checks for actual local covariance and moment conservation."""
import numpy as np

from experiments.round10_partition_data import partition_bounds,partition_index,native_support,pixel_centers
from experiments.round11_result_audit import reference_four_support,mixture_moments,reference_four_logits,R,BREF


def fixture():
    xy=pixel_centers();rng=np.random.default_rng(81)
    # Curved projected field includes between-stratum and within-stratum scatter.
    projected=np.stack([xy[:,0]+.002*xy[:,1]**2,xy[:,1]+.3*np.sin(xy[:,0]/6)],axis=-1)
    a=rng.normal(size=(len(xy),2,2));cov=a@a.transpose(0,2,1)/20
    inputs=dict(q_to_k_xy=projected,q_to_k_pixel_cov=cov,
                q_features=rng.normal(size=(432,3)),k_features=rng.normal(size=(432,3)))
    parts={}
    for name in ('g8','g16','mixed'):
        bounds=partition_bounds(name);mean,cov,area=native_support(bounds)
        parts.update({name+'_bounds':bounds,name+'_centers':mean,name+'_covariance':cov,name+'_area':area,
            name+'_pixel_index':partition_index(bounds),
            name+'_parent_g16':(mean[:,1].astype(int)//16)*12+mean[:,0].astype(int)//16})
    return inputs,parts


def test_four_components_keep_quad4_means_weights_and_add_only_local_scatter():
    inputs,parts=fixture()
    for name in ('g8','g16','mixed'):
        four=reference_four_support(inputs,parts,name)
        previous=R.reference_support(inputs,parts,name,4)
        np.testing.assert_array_equal(four['nodes'],previous['nodes'])
        np.testing.assert_array_equal(four['node_weights'],previous['node_weights'])
        np.testing.assert_array_equal(four['pixel_only_node_cov'],previous['node_cov'])
        extra=four['node_cov']-four['pixel_only_node_cov']
        assert np.linalg.eigvalsh(extra).min()>0
        assert np.max(np.abs(extra))>.01


def test_global_first_two_moments_are_preserved_under_true_four_way_split():
    inputs,parts=fixture()
    for name in ('g8','g16','mixed'):
        four=reference_four_support(inputs,parts,name)
        mean,cov=mixture_moments(four)
        np.testing.assert_allclose(mean,four['mean'],atol=3e-14,rtol=2e-15)
        np.testing.assert_allclose(cov,four['cov'],atol=3e-14,rtol=2e-14)
        np.testing.assert_array_equal(four['node_weights'].sum(-1),np.ones(len(mean)))


def test_covariance_identity_constant_plane_and_no_sensor_field_dependence():
    inputs,parts=fixture();inputs['q_to_k_xy']=pixel_centers()
    inputs['q_to_k_pixel_cov']=np.broadcast_to(np.eye(2)/12,(27648,2,2)).copy()
    original=reference_four_support(inputs,parts,'g16')
    expected=np.broadcast_to(np.eye(2)/12,original['node_cov'].shape)
    np.testing.assert_allclose(original['node_cov'],expected,atol=1e-15,rtol=0)
    inputs.update(gt_visible=np.zeros(27648,bool),sensor_depth=np.full((144,192),np.nan),gt_cov=np.full((27648,2,2),np.nan))
    poisoned=reference_four_support(inputs,parts,'g16')
    for field in original:np.testing.assert_array_equal(original[field],poisoned[field])


def test_four_gaussian_bias_is_probability_mixture_with_single_area_floor():
    inputs,parts=fixture();allq=reference_four_support(inputs,parts,'g16')
    query={k:v[:2] for k,v in allq.items()};key=R.reference_key(inputs,parts,'g16')
    bounds=(parts['g16_bounds']-.5)/8
    for method in ('full4','iso4'):
        cov=query['node_cov'].copy()
        if method=='iso4':cov=np.trace(cov,axis1=-2,axis2=-1)[:,:,None,None]*np.eye(2)/2
        mass=BREF.reference_mass(query['nodes'],cov,bounds,.5)
        expected=np.sum(mass*query['node_weights'][:,:,None],axis=1)+BREF.FLOOR*key['mass'][None]/64
        np.testing.assert_allclose(np.exp(reference_four_logits(query,key,bounds,method,.5)),expected,atol=1e-17,rtol=1e-14)
