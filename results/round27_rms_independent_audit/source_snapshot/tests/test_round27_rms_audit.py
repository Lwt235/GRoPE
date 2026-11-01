"""R27 independent audit toys; no formal fixtures or empirical archives."""
from copy import deepcopy
import math
import numpy as np
import pytest
from experiments import round27_rms_audit as A
from experiments import round27_rms_contract as C
from experiments import round27_rms_reference as R
from models import round26_rms_projective_pe as M

MU=np.array([.13,-.19,1.3])
F=np.array([[.23,0.,0.],[.09,.31,0.],[.03,-.07,.22]])
B=np.array([[-.37,-.51,.08,.43],[.08,-.51,.61,.43]])
BIAS=np.array([.2,-.35])

def model_case(mu=MU,factor=F):
    result=M.integrate(mu,factor,B,BIAS,policy=C.POLICY)
    assert result['passed']
    bank={'input__mu':mu,'input__factor':factor,'input__bounds':B,'input__content_bias':BIAS}
    bank.update({k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    return result,bank

def test_production_metadata_and_floor_arithmetic():
    row,bank=model_case()
    assert A.production_check(row,bank,C.POLICY,C.GATES)['passed']
    row['front_mass']*=.5
    assert not A.production_check(row,bank,C.POLICY,C.GATES)['passed']

@pytest.mark.parametrize('corruption',['nodes','bounds','floor','normalizer','argmax'])
def test_production_metadata_tampering_rejected(corruption):
    row,bank=model_case()
    if corruption=='nodes':row['density_diagnostics']['positive_branch_nodes']+=1
    elif corruption=='bounds':row['analytic_truncation_bound']['raw_native_l1_absolute']*=2
    elif corruption=='floor':bank['floored_mass']=bank['floored_mass']*2
    elif corruption=='normalizer':row['normalizer']*=2
    else:row['argmax_index']=1-row['argmax_index']
    assert not A.production_check(row,bank,C.POLICY,C.GATES)['passed']

def test_structural_zero_must_be_explicit_and_all_raw_zero():
    raw=np.zeros(2)
    with pytest.raises(ValueError):A.compare_raw(raw,raw,B,BIAS,1e-30,C.GATES)
    good=A.compare_raw(raw,raw,B,BIAS,1e-30,C.GATES,structural_zero=True)
    assert good['passed'] and good['raw_fov_relative'] is None and good['raw_relative_applicable'] is False
    with pytest.raises(ValueError):A.compare_raw(raw+1e-35,raw,B,BIAS,1e-30,C.GATES,structural_zero=True)
    row,bank=model_case(np.array([.2,.1,-.7]),np.zeros((3,3)))
    assert A.production_check(row,bank,C.POLICY,C.GATES)['passed']

def test_subfloor_raw_error_cannot_hide_behind_probability():
    raw=np.array([2e-37,3e-37])
    check=A.compare_raw(raw*2,raw,B,BIAS,1e-30,C.GATES)
    assert check['maximum_probability_error']<1e-4 and not check['passed']
    assert check['raw_fov_relative']==1. and check['raw_l1_over_reference_fov']==1.

def test_floor_once_then_native_bias_and_unequal_areas():
    bounds=np.array([[0.,0.,1.,1.],[1.,0.,3.,1.]])
    p,floor,floored,_=A.probabilities(np.zeros(2),bounds,np.array([0.,1.]),1e-30)
    expected=np.array([1.,2.*math.e]);expected/=expected.sum()
    np.testing.assert_allclose(p,expected,rtol=0,atol=3e-15)
    np.testing.assert_array_equal(floor,floored)

def test_reference_independent_tail_metadata_and_call_work():
    result=R.integrate(MU,F,B,BIAS,policy=C.POLICY,reference_policy=C.REFERENCE_POLICY,order=32)
    def check(value):return A.reference_check(value,MU,F,B,BIAS,C.POLICY,C.REFERENCE_POLICY,C.GATES,32)
    assert check(result)['passed']
    bad=deepcopy(result);bad['truncation_bounds']['native_raw_l1_absolute']=0.;bad['truncation_bounds']['raw_l1_over_reference_fov']=0.
    assert not check(bad)['passed']
    bad=deepcopy(result);bad['order']=48
    assert not check(bad)['passed']
    bad=deepcopy(result);bad['work']['depth_evaluations']=C.REFERENCE_POLICY['maximum_depth_evaluations']+1
    assert not check(bad)['passed']

def test_merge_identity_audits_floored_mass_as_well_as_probability():
    parent={'raw_mass':np.array([.1,.2,.3]),'floored_mass':np.array([.2,.3,.5]),'probability':np.array([.1,.3,.6])}
    actual={k:np.array([v[:2].sum(),v[2]]) for k,v in parent.items()}
    fixture=dict(id='toy_merge',identity=dict(parent='toy_base',relation='merge',groups=[[0,1],[2]]))
    assert A.identity_check(fixture,actual,parent,C.GATES)['passed']
    actual['floored_mass'][0]+=.01
    assert not A.identity_check(fixture,actual,parent,C.GATES)['passed']

def test_near_constant_limit_requires_nonzero_depth_branch():
    near='11_near_constant';constant='03_constant_correlated'
    bank=dict(raw_mass=np.array([.2,.3]),probability=np.array([.4,.6]),input__factor=np.diag([.1,.2,1e-9]))
    banks={near:deepcopy(bank),constant:deepcopy(bank)}
    production={near:dict(passed=True,structural_branch='variable_depth'),constant:dict(passed=True)}
    assert A.near_constant_check(banks,production,C.GATES)['passed']
    production[near]['structural_branch']='positive_constant_depth_gaussian'
    assert not A.near_constant_check(banks,production,C.GATES)['passed']
