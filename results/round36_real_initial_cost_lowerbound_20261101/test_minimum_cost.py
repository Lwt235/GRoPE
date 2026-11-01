"""Distinct in-memory geometry toys; no saved model inputs are decoded."""
import ast
import importlib.util
import math
from pathlib import Path
import time
import numpy as np
import pytest

SOURCE=Path(__file__).with_name('diagnose_minimum_cost.py')
spec=importlib.util.spec_from_file_location('minimum_cost_toy_module',SOURCE)
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


def boxes():
    return np.asarray([[x,y,x+1.,y+1.] for y in range(-9,9) for x in range(-12,12)],np.float64)


def policies():
    return dict(h=.25),dict(standard_x_panel_width=2.,depth_cutoff=18.,maximum_depth_intervals=512,
        outer_geometry_breaks=[-8.,-1.,0.,1.,8.])


def bank(mu=(.13,-.27,2.3),factor=None):
    return dict(input__mu=np.asarray(mu,np.float64),input__factor=np.diag([.31,.47,.62]) if factor is None else factor,
        input__bounds=boxes(),input__content_bias=np.zeros(432))


def test_geometry_matches_direct_diagonal_covariance_without_cdf(monkeypatch):
    monkeypatch.setattr(D.scipy.special,'log_ndtr',lambda *a:pytest.fail('no CDF permitted'))
    mu=np.asarray([.13,-.27,2.3]);factor=np.diag([.31,.47,.62]);g=D.geometry(mu,factor,.25)
    rms=math.hypot(2.3,.62)
    assert g['branch']=='stochastic_depth'
    assert abs(g['normalized_depth_mean']-2.3/rms)<2e-16
    assert abs(g['sx']-math.hypot(.31/rms,.25))<2e-16
    assert abs(g['marginal_sy']-math.hypot(.47/rms,.25))<2e-16
    np.testing.assert_array_equal(g['beta'],[0.,0.])


@pytest.mark.parametrize('z',(0.,-2.3))
def test_structural_branches_do_not_normalize_irrelevant_xy(z):
    g=D.geometry(np.asarray([1e308,-1e308,z]),np.zeros((3,3)),.25)
    assert g['structural_zero']


def test_nonzero_depth_factor_underflow_is_not_constant_replacement():
    factor=np.zeros((3,3));factor[2,2]=1e-300
    with pytest.raises(FloatingPointError,match='stochastic depth lost'):
        D.geometry(np.asarray([0.,0.,1e308]),factor,.25)


def test_unrepresentable_normalized_xy_is_explicit_failure():
    with pytest.raises(FloatingPointError):
        D.geometry(np.asarray([1e308,0.,1e-308]),np.zeros((3,3)),.25)


@pytest.mark.parametrize('a,b,expected',((0.,4.,2),(-2.,2.,2),(-1.,1.,2),(0.,2.,1),(-2.,0.,1)))
def test_python_integer_minimum_x_admission(a,b,expected):
    value=D.minimum_panels(np.zeros(2),1.,np.asarray([[a,0.,b,1.]]),1.,2.)
    assert type(value) is int and value==expected


def test_no_grid_or_inner_node_arrays_are_constructed(monkeypatch):
    monkeypatch.setattr(np,'repeat',lambda *a,**k:pytest.fail('grid allocation forbidden'))
    monkeypatch.setattr(np.polynomial.legendre,'leggauss',lambda *a:pytest.fail('inner GL nodes forbidden'))
    value=D.minimum_panels(np.zeros(2),1.,np.asarray([[-1000001.,0.,1000001.,1.]]),1.,2.)
    assert value==1000002


def test_collapsed_float_interval_keeps_original_positive_halfwidth():
    assert D.minimum_panels(np.asarray([-1e20,0.]),1.,np.asarray([[0.,0.,1.,1.]]),1.,2.)==1
    with pytest.raises(FloatingPointError,match='rounded interval'):
        D.minimum_panels(np.asarray([-1e20,0.]),1.,np.asarray([[0.,0.,3.,1.]]),1.,2.)


@pytest.mark.parametrize('z',(0.,-1.,math.inf))
def test_nonpositive_or_nonfinite_depth_is_not_counted(z):
    with pytest.raises(FloatingPointError):
        D.minimum_panels(np.zeros(2),z,np.asarray([[0.,0.,1.,1.]]),1.,2.)


def test_constant_depth_is_one_rectangle_call_and_structural_zero_none():
    p,r=policies();functions=D.pure_geometry_functions()
    report,arrays=D.count_case(bank(factor=np.zeros((3,3))),p,r,time.perf_counter()+10,functions)
    assert report['status']=='initial_lower_bound_only' and report['required_callbacks']==1
    assert report['completed_required_callbacks']==1 and report['minimum_panels_prefix']>0 and not arrays
    report,_=D.count_case(bank(mu=(0.,0.,-2.),factor=np.zeros((3,3))),p,r,time.perf_counter()+10,functions)
    assert report['status']=='structural_zero' and report['minimum_panels_prefix']==0


def test_all_pilots_and_initial_gk_callbacks_are_mandatory():
    p,r=policies();_,coalesce,nodes=D.pure_geometry_functions()
    report,arrays=D.count_case(bank(),p,r,time.perf_counter()+10,(lambda *a:[],coalesce,nodes))
    assert report['initial_interval_count']==1 and report['full_initial_geometry_counted']
    assert report['required_callbacks']==report['completed_required_callbacks']==22
    assert len(arrays['pilot_zeta'])==1 and report['minimum_panels_prefix']>=22*433


def test_rounded_endpoint_zero_skips_inner_nodes_without_clamping(monkeypatch):
    p,r=policies();_,coalesce,nodes=D.pure_geometry_functions()
    one=np.nextafter(-1.,np.inf);two=np.nextafter(one,np.inf)
    original=D.geometry
    def geometry(*args):
        value=original(*args);value.update(normalized_depth_mean=1.,normalized_depth_std=1.)
        return value
    monkeypatch.setattr(D,'geometry',geometry)
    report,_=D.count_case(bank(),p,r,time.perf_counter()+10,(lambda *a:[two],coalesce,nodes))
    assert report['full_initial_geometry_counted'] and report['zero_inner_callbacks']>0
    assert report['completed_required_callbacks']==44


def test_positive_front_beyond_cutoff_is_not_low_cost_success():
    p,r=policies()
    report,_=D.count_case(bank(mu=(.1,.2,-9.),factor=np.diag([.2,.3,.1])),p,r,
        time.perf_counter()+10,D.pure_geometry_functions())
    assert report['status']=='geometry_failure' and 'outside' in report['failure_reason']
    assert not report['full_initial_geometry_counted']


def test_timeout_keeps_completed_integer_prefix(monkeypatch):
    p,r=policies();_,coalesce,nodes=D.pure_geometry_functions();calls=[]
    def panel(*args):
        calls.append(True)
        if len(calls)==3:raise TimeoutError('toy deadline')
        return 3_000_001
    monkeypatch.setattr(D,'minimum_panels',panel)
    report,_=D.count_case(bank(),p,r,time.perf_counter()+10,(lambda *a:[],coalesce,nodes))
    assert report['status']=='timed_out' and report['completed_required_callbacks']==2
    assert report['minimum_panels_prefix']==6_000_002
    assert 32*report['minimum_panels_prefix']<250_000_000<48*report['minimum_panels_prefix']
    assert not report['full_initial_geometry_counted']


def test_deadline_before_geometry_does_not_manufacture_completed_work():
    p,r=policies()
    report,_=D.count_case(bank(),p,r,time.perf_counter()-1,D.pure_geometry_functions())
    assert report['status']=='timed_out' and report['completed_required_callbacks']==0
    assert report['minimum_panels_prefix']==0


def test_source_does_not_import_integrator_or_reference_and_freeze_cannot_decode_npz():
    tree=ast.parse(SOURCE.read_text())
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        if isinstance(node,ast.Import):imports.extend(n.name for n in node.names)
    assert not any(x.startswith(('experiments','models','scipy.integrate')) for x in imports)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name!='run':
            assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)
                           and isinstance(x.func.value,ast.Name) and x.func.value.id=='np' and x.func.attr=='load'
                           for x in ast.walk(node))


def test_timeout_preserves_all72_statuses_and_never_decodes_remaining_inputs(tmp_path,monkeypatch):
    import json
    data=bank();loads=[]
    records=[]
    for index in range(72):
        records.append(dict(id=f'toy_{index}',scene='toy',pair=[1,2],pair_name='toy_pair',
            query_size=8,query_index=index,query_bounds=[0,0,8,8],archive_path=f'input_{index}.npz',
            archive_sha256='toy_digest',array_sha256={k:D.array_sha(v) for k,v in data.items()}))
    monkeypatch.setattr(D,'OUT',tmp_path);monkeypatch.setattr(D,'RUN',tmp_path/'inputs')
    (tmp_path/'lock.json').write_text('{}')
    policy,ref=policies();lock=dict(records=records,policy=policy,reference_policy=ref)
    monkeypatch.setattr(D,'verify',lambda:lock)
    original_sha=D.sha
    monkeypatch.setattr(D,'sha',lambda path:'toy_digest' if Path(path).suffix=='.npz' else original_sha(path))
    class Archive:
        files=list(D.FIELDS)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,key):return data[key]
    def load(path,**kwargs):loads.append(path);return Archive()
    monkeypatch.setattr(D.np,'load',load)
    def count(*args):
        return dict(status='timed_out',failure_reason='toy deadline',minimum_panels_prefix=7,
            completed_required_callbacks=1,required_callbacks=22,full_initial_geometry_counted=False,
            front_probability_not_evaluated=True),{}
    monkeypatch.setattr(D,'count_case',count)
    D.run()
    rows=json.loads((tmp_path/'records.json').read_text())
    assert len(rows)==72 and len(loads)==1 and rows[0]['real_inputs_read']
    assert rows[0]['orders']['48']['minimum_x_nodes_prefix']==336
    assert all(r['status']=='not_processed' and not r['real_inputs_read'] for r in rows[1:])
    assert all(r['orders']['32']['minimum_x_nodes_prefix']==0 for r in rows[1:])
    assert len((tmp_path/'progress.jsonl').read_text().splitlines())==72
    assert all(not r['labels_read'] and r['reference_integrations']==0 for r in rows)
