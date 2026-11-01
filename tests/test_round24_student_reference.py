"""Independent t4 reference/audit toys, never formal fixtures or real data."""
from copy import deepcopy
import math
import time
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from experiments import round24_student_contract as C
from experiments import round24_student_reference as R
from experiments import round24_student_audit as A


BOUNDS = np.array([[-.6, -.4, -.1, .7], [-.1, -.4, .4, .7]], np.float64)
MEAN = np.array([-.1, .2]); COV = np.array([[.25, .075], [.075, .4]])


def reference(order=64, mean=MEAN, cov=COV, bounds=BOUNDS, bias=None, **kwargs):
    return R.integrate(mean, cov, bounds, bias, policy=dict(C.POLICY, h=.3), reference_policy=C.REFERENCE_POLICY, order=order, **kwargs)


@pytest.mark.parametrize('lower,upper', [(-.2,.4),(.3,.300001),(-9.,-8.),(8.,9.),(1e4,1e4+1.)])
def test_stable_t5_intervals_against_positive_density_integral(lower, upper):
    nodes, weights = leggauss(96); middle, half = (lower+upper)/2, (upper-lower)/2
    x = middle + half*nodes
    expected = half*np.sum(weights*8/(3*np.pi*np.sqrt(5))*(1+x*x/5)**-3)
    actual = float(R.t5_interval(lower, upper))
    assert actual > 0
    assert abs(actual-expected)/expected < 2e-9
    assert actual == float(R.t5_interval(-upper,-lower))


def test_t5_center_infinite_endpoints_and_invalid_interval():
    assert R.t5_interval(-np.inf,np.inf) == 1.
    assert R.t5_interval(0.,np.inf) == .5
    assert R.t5_interval(np.inf,np.inf) == 0.
    tiny = float(R.t5_interval(-1e-12,1e-12))
    assert abs(tiny/(16e-12/(3*np.pi*np.sqrt(5)))-1) < 1e-14
    with pytest.raises(ValueError): R.t5_interval(2.,1.)


def joint_rectangle(mean,cov,bounds,h):
    total=cov+h*h*np.eye(2); precision=np.linalg.inv(total); nodes,weights=leggauss(128)
    result=[]
    for box in bounds:
        half=(box[2:]-box[:2])/2; middle=(box[2:]+box[:2])/2
        xx,yy=np.meshgrid(middle[0]+half[0]*nodes,middle[1]+half[1]*nodes,indexing='ij')
        d=np.stack((xx-mean[0],yy-mean[1]),-1)
        q=np.einsum('...i,ij,...j->...',d,precision,d)
        density=(1+q/2)**-3/(np.pi*np.sqrt(np.linalg.det(total)))
        result.append(float(np.sum(density*np.outer(weights,weights))*half.prod()))
    return np.asarray(result)


def test_conditional_t5_reference_matches_separate_direct_toy_integral():
    low,high=reference(64),reference(96)
    expected=joint_rectangle(MEAN,COV,BOUNDS,.3)
    np.testing.assert_allclose(high['raw_mass'],expected,atol=3e-15,rtol=0)
    np.testing.assert_allclose(low['raw_mass'],high['raw_mass'],atol=3e-15,rtol=0)
    assert high['work']['cell_node_evaluations']==A.reference_work(BOUNDS,.3,96,C.REFERENCE_POLICY)
    assert high['tail_truncation_used'] is False and high['quadrature_error_bound_available'] is False


def test_fixed_bias_does_not_change_raw_geometry_and_tail_is_not_discarded():
    plain=reference();biased=reference(bias=np.array([-.7,1.2]))
    np.testing.assert_array_equal(plain['raw_mass'],biased['raw_mass'])
    far=np.array([2e6,-.3]);result=reference(mean=far)
    assert np.all(result['raw_mass']>0) and result['raw_mass'].sum()<1e-30
    expected=joint_rectangle(far,COV,BOUNDS,.3)
    assert np.max(abs(result['raw_mass']-expected))/expected.sum()<1e-8


@pytest.mark.parametrize('kind',['asymmetric','indefinite','nonfinite_S','wrongdtype','overlap','order'])
def test_invalid_input_or_policy_fails_without_repair(kind):
    cov=COV.copy();bounds=BOUNDS.copy();order=64;policy=dict(C.POLICY,h=.3)
    if kind=='asymmetric':cov[0,1]+=.001
    elif kind=='indefinite':cov[0,0]=-1.
    elif kind=='nonfinite_S':cov=np.eye(2)*1.5e308;policy['h']=1e154
    elif kind=='wrongdtype':cov=cov.astype(np.float32)
    elif kind=='overlap':bounds[1,0]-=.1
    else:order=32
    with np.errstate(over='ignore'):
        with pytest.raises((ValueError,np.linalg.LinAlgError)):
            R.integrate(MEAN,cov,bounds,policy=policy,reference_policy=C.REFERENCE_POLICY,order=order)


def test_small_negative_admitted_C_is_not_modified_and_S_stays_strict():
    cov=np.diag([-1e-13,.2]);before=cov.copy()
    result=reference(cov=cov)
    assert result['admitted_covariance_min_eigenvalue']==-1e-13
    np.testing.assert_array_equal(cov,before)


def test_deadline_and_failed_work_are_retained(monkeypatch):
    with pytest.raises(R.ReferenceFailure) as error:reference(deadline=time.perf_counter()-1)
    assert error.value.reference_work['cell_node_evaluations']==0
    original=R._rectangle;calls=[]
    def fail_second(*args):
        calls.append(1)
        if len(calls)==2:raise ValueError('invented CDF failure')
        return original(*args)
    monkeypatch.setattr(R,'_rectangle',fail_second)
    with pytest.raises(R.ReferenceFailure) as error:reference()
    assert error.value.reference_work['cell_node_evaluations']==128
    assert error.value.partial_raw_mass.shape==(1,)


def test_one_native_floor_precedes_content_and_preserves_area():
    bounds=np.array([[0.,0.,1.,1.],[1.,0.,3.,1.]])
    raw=np.zeros(2);bias=np.array([0.,1.])
    actual,floor,floored,_=A.probabilities(raw,bounds,bias,1e-30)
    expected=np.array([1.,2*np.e]);expected/=expected.sum()
    np.testing.assert_allclose(actual,expected,atol=3e-15,rtol=0)
    np.testing.assert_array_equal(floor,floored)


def test_raw_subfloor_relative_gate_and_zero_reference_are_not_hidden():
    ref=np.array([1e-35,1e-35]);actual=ref*2;bias=np.zeros(2)
    result=A.compare_raw(actual,ref,BOUNDS,bias,1e-30,C.GATES)
    assert result['maximum_probability_error']==0. and not result['passed']
    assert result['raw_fov_relative']==result['raw_l1_over_reference_fov']==1.
    with pytest.raises(ValueError,match='unresolved'):
        A.compare_raw(actual,np.zeros(2),BOUNDS,bias,1e-30,C.GATES)


def test_P_gate_uses_archived_values_and_records_ties():
    raw=np.array([.50009999999975,.49990000000025]);ref=np.array([.5,.5]);bias=np.zeros(2)
    gates=dict(C.GATES,raw_fov_relative=1.,raw_l1_over_reference_fov=1.)
    assert A.compare_raw(raw,ref,BOUNDS,bias,1e-30,gates)['passed']
    archived=np.array([.50010000000025,.49989999999975])
    result=A.compare_raw(raw,ref,BOUNDS,bias,1e-30,gates,archived_probability=archived)
    assert not result['passed'] and result['reference_exact_maximizers']==[0,1]


def test_identity_native_merge_and_permutation_are_array_comparisons():
    parent=dict(raw_mass=np.array([.1,.2,.3,.4]),probability=np.array([.1,.2,.3,.4]))
    fixture=dict(id='toy',identity=dict(parent='parent',relation='permutation',permutation=[3,2,1,0]))
    assert A.identity_check(fixture,{n:v[::-1] for n,v in parent.items()},parent,C.GATES)['passed']
    fixture['identity']=dict(parent='parent',relation='merge',groups=[[0],[1,2,3]])
    merged={n:np.array([v[0],math.fsum(v[1:])]) for n,v in parent.items()}
    assert A.identity_check(fixture,merged,parent,C.GATES)['passed']


def test_freeze_guard_precedes_arrays_and_preserves_failed_audit(tmp_path,monkeypatch):
    run=tmp_path/'producer';run.mkdir();(run/'started.json').write_text('{}')
    monkeypatch.setattr(A.np,'load',lambda *args,**kwargs:pytest.fail('array opened'))
    with pytest.raises(ValueError,match='precede every formal'):A.freeze(run,tmp_path/'audit')
    out=tmp_path/'audit';out.mkdir()
    def reject(*args):raise KeyboardInterrupt('invented stop')
    monkeypatch.setattr(A,'verify_seal',reject)
    with pytest.raises(KeyboardInterrupt):A.audit(run,out)
    assert A.verify_inventory(out)['delivery_integrity_passed'] is False
    original=(out/'failure.json').read_bytes()
    with pytest.raises(ValueError,match='preserve every prior'):A.audit(run,out)
    assert (out/'failure.json').read_bytes()==original


def test_runtime_source_closure_and_work_contract_match_metadata_only():
    from experiments import round24_student_preflight as P
    assert A.source_names()==P.source_names()
    assert A.check_runtime()==P.check_runtime()
    fixtures=C.fixtures()
    assert sum(r['kind']=='base' for r in fixtures)==36 and sum(r['kind']=='identity' for r in fixtures)==6
    assert A.array_sha(np.array([1.,2.]))==P.array_sha(np.array([1.,2.]))
