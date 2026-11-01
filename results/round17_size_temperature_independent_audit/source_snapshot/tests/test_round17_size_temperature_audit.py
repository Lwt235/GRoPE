"""Synthetic tests only; no observations, labels or current validation results."""
from copy import deepcopy
from fractions import Fraction
import numpy as np
import pytest
from scipy.special import ndtr,softmax

from experiments import round17_size_temperature_audit as A


def test_native_bounds_trace_keeps_rectangle_aspect_ratio_and_units():
    bounds=np.array([[0,0,8,8],[0,0,16,16],[0,0,24,24],[0,0,16,8],[0,0,8,16]],np.int64)
    np.testing.assert_allclose(A.native_trace(bounds),[1/6,2/3,1.5,5/12,5/12],atol=1e-16,rtol=0)
    assert A.size_temperature(bounds).tobytes()==np.array([30/11,6/7,.4,30/23,30/23],np.float32).tobytes()
    assert A.size_temperature(bounds)[3]!=np.float32(30/19)
    assert np.array_equal(A.native_trace(bounds+np.array([20,30,20,30])),A.native_trace(bounds))
    # Historical partition archives may store int32; actual prepared Torch
    # bounds and saved support records are int64. Both encode the same cells.
    assert np.array_equal(A.native_trace(bounds.astype(np.int32)),A.native_trace(bounds))
    assert np.array_equal(A.size_temperature(bounds.astype(np.int32)),A.size_temperature(bounds))


@pytest.mark.parametrize('bounds',[
    np.array([[0,0,8,8]],np.float64),np.array([[0,0,8,8]],np.int16),
    np.array([[0,0,0,8]],np.int64),np.empty((0,4),np.int64),np.ones((2,3),np.int64)])
def test_bounds_schema_rejects_changed_population_dtype_and_degenerate_cells(bounds):
    with pytest.raises(ValueError):A.native_trace(bounds)


def test_float32_bias_order_and_area_floor_once():
    raw=np.array([[0.,2e-31,1e-30],[.1,.5,.3]],np.float64)
    content=np.array([[.123456789,.987654321,-.27182818],[2.,-.2,.8]],np.float32)
    qbounds=np.array([[0,0,16,8],[0,0,24,24]],np.int64)
    kbounds=np.array([[0,0,1,1],[1,0,3,1],[3,0,6,1]],np.float64)
    tau=A.size_temperature(qbounds)
    actual=A.independent_probability(raw,content,tau,kbounds)
    expected=softmax(np.log(raw+1e-30*np.array([1.,2.,3.]))+(tau[:,None]*content).astype(np.float64),axis=1)
    assert np.array_equal(actual,expected)
    wrong=softmax(np.log(raw+1e-30*np.array([1.,2.,3.]))+tau.astype(np.float64)[:,None]*content,axis=1)
    assert np.max(abs(actual-wrong))>1e-10
    assert np.allclose(A.independent_probability(raw,content,np.zeros(2,np.float32),kbounds),
        softmax(np.log(raw+1e-30*np.array([1.,2.,3.])),axis=1),atol=0,rtol=0)
    with pytest.raises(ValueError):A.independent_probability(raw,content.astype(np.float64),tau,kbounds)


def test_gaussian_whole_cell_reference_and_size_bias_against_analytic_diagonal():
    mean=np.array([[.2,.3],[1.2,.8]],np.float32)
    cov=np.array([np.diag([.2,.4]),np.diag([.5,.3])],np.float32)
    bounds=np.array([[0.,0.,1.,1.],[1.,0.,2.,1.],[0.,1.,2.,2.]])
    s=np.sqrt(cov.astype(np.float64).diagonal(axis1=1,axis2=2)+.25**2)
    exact=np.prod(ndtr((bounds[None,:,2:]-mean[:,None])/s[:,None])-
                  ndtr((bounds[None,:,:2]-mean[:,None])/s[:,None]),axis=2)
    for order in (64,96):
        raw=A.TRACE.independent_mass(mean,cov,bounds,.25,order)
        np.testing.assert_allclose(raw,exact,atol=1e-15,rtol=1e-13)
        content=np.array([[.1,.9,-.3],[.4,.1,.3]],np.float32)
        tau=A.size_temperature(np.array([[0,0,8,16],[0,0,16,8]],np.int64))
        p=A.independent_probability(raw,content,tau,bounds)
        reference=A.independent_probability(exact,content,tau,bounds)
        assert np.max(abs(p-reference))<1e-14


def test_squared_pck_boundary_ties_invalid_and_paired_conservation():
    bounds=np.array([[0,0,1,1],[16,0,17,1],[32,0,33,1]],np.int64)
    primary=np.array([[.5,.5,0],[1,0,0],[0,0,1],[1,0,0]],np.float64)
    size=np.eye(3)[[1,2,0,2]]
    target=np.array([[8.,0.],[32.,0.],[32.,0.],[np.nan,np.nan]])
    visible=np.array([True,True,True,False])
    ac,bc,counts=A.center_pair(primary,size,bounds,target,visible)
    assert ac.tolist()==[True,False,True,False] and bc.tolist()==[True,True,False,False]
    assert counts==dict(n_query=4,n_point=3,primary_correct=2,size_correct=2,flip_all_count=4,
        flip_valid_count=3,size_benefit_count=1,size_harm_count=1)


def test_center_sources_preserve_r13_coordinates_and_visibility_for_all_squares(tmp_path):
    old = tmp_path / 'old'
    old.mkdir()
    dense = dict(gt_xy=np.array([[3., 4.]]), gt_visible=np.ones(1, bool),
                 mixed_point_xy=np.array([[5., 6.]]), mixed_point_visible=np.ones(1, bool))
    # Close coordinates can differ at the inclusive radius boundary. These old
    # R10 square fields are deliberately present but must never be selected.
    decoys = {q + '_point_' + f:
              np.array([[8. + 1e-7, 0.]]) if f == 'xy' else np.zeros(1, bool)
              for q in ('g8', 'g16') for f in ('xy', 'visible')}
    np.savez(old / 'labels.npz', **dense, **decoys)
    target = tmp_path / 'r13.npz'
    centers = {q + '_point_' + f:
               np.array([[8., 0.]]) if f == 'xy' else np.ones(1, bool)
               for q in ('g8', 'g16', 'g24') for f in ('xy', 'visible')}
    np.savez(target, **centers)
    entry = dict(path=str(old), label_sha256=A.sha(old / 'labels.npz'),
                 target_path=str(target), target_sha256=A.sha(target))
    labels = A.load_labels(entry)
    assert set(labels) == set(dense) | set(centers)
    for field, expected in dict(dense, **centers).items():
        assert labels[field].dtype == expected.dtype
        assert labels[field].tobytes() == expected.tobytes()
    p = np.ones((1, 1), np.float64)
    bounds = np.array([[0, 0, 1, 1]], np.int64)
    for q in ('g8', 'g16', 'g24'):
        _, _, counts = A.center_pair(p, p, bounds, labels[q + '_point_xy'], labels[q + '_point_visible'])
        assert counts['n_point'] == counts['primary_correct'] == 1
    with pytest.raises(ValueError, match='label bytes'):
        A.load_labels(dict(entry, target_sha256='0' * 64))


def synthetic_rows():
    entries=[dict(scene=s,pair=[i,i+1],input_sha256='a'*64) for s in A.SCENES for i in range(16)]
    rows=[]
    for e in entries:
        scene_index=A.SCENES.index(e['scene'])
        for c in A.CONDITIONS:
            for role in A.ROLES:
                n=scene_index+1;good=1 if role=='trace_full' else 0
                rows.append(dict(scene=e['scene'],pair=e['pair'],role=role,config=A.CONFIG,**c,
                    shape=[A.C.NATIVE_SIZES[c['query']],A.C.NATIVE_SIZES[c['key']]],dtype='float64',
                    input_sha256=e['input_sha256'],mass=100+scene_index,n_point=n,pck_count=good,
                    n_query=A.C.NATIVE_SIZES[c['query']],n_key=A.C.NATIVE_SIZES[c['key']],
                    kl_sum=2*(100+scene_index),js_sum=.5*(100+scene_index),
                    direct_kl_sum=3*(100+scene_index),direct_js_sum=.7*(100+scene_index)))
    return entries,rows


def test_complete_population_exact_pck_scene_macro_not_pooled():
    entries,rows=synthetic_rows();A.validate_population(rows,entries)
    summary=A.aggregate(rows,entries)['trace_full']
    exact=summary['by_condition']['g8_g8']['pck_exact']
    assert Fraction(exact['numerator'],exact['denominator'])==Fraction(11,18)
    assert summary['common9_kl']==2. and summary['direct5_kl']==2.
    assert summary['J_descriptive']==2.
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError,match='rows'):A.validate_population(bad,entries)
    bad=deepcopy(rows);bad[0]['n_point']+=1
    with pytest.raises(ValueError,match='visibility'):A.validate_population(bad,entries)
    bad=deepcopy(rows);bad[0]['mass']+=1
    with pytest.raises(ValueError,match='dense population'):A.validate_population(bad,entries)


def paired_rows():
    rows=[]
    for scene,n,p,s in zip(A.SCENES,(1,2,4),(1,0,0),(0,1,1)):
        for i in range(16):
            rows.append(dict(scene=scene,pair=[i,i+1],condition='mixed_g8',n_query=4,n_point=n,
                primary_correct=p,size_correct=s,size_benefit_count=max(s-p,0),size_harm_count=max(p-s,0),
                flip_all_count=1,flip_valid_count=1))
    return rows


def test_paired_fraction_empty_scene_and_duplicates():
    rows=paired_rows();macro=A.aggregate_pairs(rows)['mixed_g8']['macro']
    assert macro['pck_difference_exact']==dict(numerator=-1,denominator=12)
    assert macro['size_minus_primary_pck_pp']==float(Fraction(-100,12))
    with pytest.raises(ValueError,match='duplicate'):A.aggregate_pairs(rows+[rows[0]])
    for row in rows:
        if row['scene']==A.SCENES[0]:
            row.update(n_point=0,primary_correct=0,size_correct=0,size_benefit_count=0,size_harm_count=0,flip_valid_count=0)
    macro=A.aggregate_pairs(rows)['mixed_g8']['macro']
    assert macro['pck_difference_exact'] is None and macro['every_scene_has_center_population'] is False


def test_native_size_strata_have_only_direct_regional_metrics():
    rows=paired_rows()
    for row in rows:
        row.update(width=16,height=8,native={r:dict(mass=10,direct_kl_sum=20.,direct_js_sum=3.) for r in A.ROLES})
    result=A.aggregate_pairs(rows,stratified=True)['mixed_g8__16x8']
    assert result['macro']['native']['trace_full']==dict(direct_kl=2.,direct_js=.3)
    assert 'common_kl' not in result['macro']['native']['trace_full']


def synthetic_supports():
    parts=A.C.literal_parts();collection=[]
    for scene in A.SCENES:
        for i in range(16):
            support={}
            for q,part in parts.items():
                bounds=part['bounds'];cov=part['cov'].astype(np.float32)
                support.update({q+'__'+n:v for n,v in dict(bounds=bounds,cov=cov,
                    tau_obs32=A.TRACE.independent_temperature(cov,.25,10.).astype(np.float32),
                    tau_size32=A.size_temperature(bounds)).items()})
            collection.append((dict(scene=scene,pair=[i,i+1]),support))
    return collection


def test_all_unique_supports_fixed_mixed_orientations_no_key_repetition():
    collection=synthetic_supports();result=A.support_statistics(collection)
    assert result['unique_queries']==39888 and len(result['groups'])==21
    mixed=[g for g in result['groups'] if g['query']=='mixed' and g['scene']==A.SCENES[0]]
    assert {(g['width'],g['height']) for g in mixed}=={(8,8),(16,8),(8,16),(16,16)}
    assert sum(g['n'] for g in mixed)==16*243
    assert result['key_condition_repetition'] is False and result['significance_test'] is False
    with pytest.raises(ValueError,match='more than once'):A.support_statistics(collection+[collection[0]])


def test_inventory_detects_missing_added_changed_and_unsafe_files(tmp_path):
    p=tmp_path/'a.txt';p.write_text('a');files={'a.txt':A.sha(p)}
    A.inventory(tmp_path,files)
    (tmp_path/'extra').write_text('x')
    with pytest.raises(ValueError,match='population'):A.inventory(tmp_path,files)
    (tmp_path/'extra').unlink();p.write_text('changed')
    with pytest.raises(ValueError,match='changed'):A.inventory(tmp_path,files)
    with pytest.raises(ValueError,match='unsafe'):A.inventory(tmp_path,{'../outside':'x'})


def test_pre_forward_design_binds_exact_producer_and_source_snapshots(tmp_path,monkeypatch):
    run=tmp_path/'run';out=tmp_path/'audit';run.mkdir();out.mkdir()
    A.write(run/'lock.json',dict(created_unix=1.))
    monkeypatch.setattr(A,'SOURCE_NAMES',('source.py',))
    snapshot=out/'source_snapshot';snapshot.mkdir();(snapshot/'source.py').write_text('x')
    lock=dict(created_unix=1.,source_hashes={'source.py':A.sha(snapshot/'source.py')})
    design=dict(created_unix=2.,run=str(run),producer_lock_sha256=A.sha(run/'lock.json'),
        source_hashes=lock['source_hashes'],**A.AUDIT_POLICY)
    def write_design():
        (out/'lock.json').unlink(missing_ok=True)  # Synthetic tampering fixture only.
        A.write(out/'lock.json',design);(out/'lock.sha256').write_text(A.sha(out/'lock.json'))
    write_design();A.verify_audit_lock(run,out,lock)
    design['producer_lock_sha256']='0'*64;write_design()
    with pytest.raises(ValueError,match='chronology'):A.verify_audit_lock(run,out,lock)
    design['producer_lock_sha256']=A.sha(run/'lock.json');design['created_unix']=0.;write_design()
    with pytest.raises(ValueError,match='chronology'):A.verify_audit_lock(run,out,lock)
    design['created_unix']=2.;design['probability_tolerance']=1e-3;write_design()
    with pytest.raises(ValueError,match='chronology'):A.verify_audit_lock(run,out,lock)
    design['probability_tolerance']=1e-4;design['independent_new_model_import']=0;write_design()
    with pytest.raises(ValueError,match='chronology'):A.verify_audit_lock(run,out,lock)


def test_joint_sources_include_producer_review_inherited_chain_and_import_closure(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir()
    monkeypatch.setattr(A,'SOURCE_NAMES',('audit.py',))
    monkeypatch.setattr(A,'PRODUCER_SOURCES',('producer.py','review.md'))
    original=dict(source_hashes={'audit.py':'a','producer.py':'b','review.md':'c','inherited.py':'d'},
                  prior=dict(source_hashes={'inherited.py':'d'}))
    def make_lock(value):
        (run/'lock.json').unlink(missing_ok=True)
        A.write(run/'lock.json',value)
        (run/'lock.sha256').write_text(A.sha(run/'lock.json'))
    for missing in ('audit.py','producer.py','review.md','inherited.py'):
        changed=deepcopy(original);del changed['source_hashes'][missing];make_lock(changed)
        with pytest.raises(ValueError,match='source population'):A.checked_lock(run)
    changed=deepcopy(original);changed['source_hashes']['unbound.py']='e';make_lock(changed)
    with pytest.raises(ValueError,match='source population'):A.checked_lock(run)


def test_comparison_never_turns_integer_or_fraction_difference_into_tolerance():
    A.compare(dict(value=1.,count=2),dict(value=1.+1e-11,count=2))
    with pytest.raises(ValueError):A.compare(dict(value=1.,count=2),dict(value=1.,count=3))
    with pytest.raises(ValueError):A.compare(dict(numerator=1,denominator=3),dict(numerator=2,denominator=6))


def test_current_benchmark_guard_reads_only_existence(monkeypatch,tmp_path):
    monkeypatch.setattr(A,'ROOT',tmp_path)
    directory=tmp_path/'results/round17_validation/timing';directory.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='benchmark'):A.no_active_benchmark()
    (directory/'completion.json').write_text('not decoded')
    A.no_active_benchmark()


def test_complete_synthetic_summary_schema_matches_independent_calculation():
    # Cross-check is intentionally confined to synthetic tests. The actual
    # independent auditor never imports the new producer's metrics or model.
    from experiments import round17_size_temperature_attribution as P
    entries,rows=synthetic_rows();paired=paired_rows();strata=deepcopy(paired)
    for r in strata:
        r.update(width=16,height=8,native={role:dict(mass=10,direct_kl_sum=20.,direct_js_sum=3.) for role in A.ROLES})
    collection=synthetic_supports()
    def descriptors():
        for entry,support in collection:
            for q in A.PARTITIONS:
                bounds=support[q+'__bounds'];sizes=bounds[:,2:]-bounds[:,:2]
                trace=np.trace(support[q+'__cov'].astype(np.float64),axis1=-2,axis2=-1)
                native=A.native_trace(bounds)
                for i,(width,height) in enumerate(sizes):
                    yield dict(scene=entry['scene'],pair=entry['pair'],query=q,query_index=i,width=int(width),height=int(height),
                        observed_trace=trace[i],native_trace=native[i],tau_obs32=support[q+'__tau_obs32'][i],
                        tau_size32=support[q+'__tau_size32'][i])
    actual=dict(metrics=P.aggregate(rows),paired_summary=P.aggregate_pairs(paired),
        size_strata_summary=P.aggregate_pairs(strata,stratified=True),support_statistics=P.summarize_supports(descriptors()))
    independent=dict(metrics=A.aggregate(rows,entries),paired_summary=A.aggregate_pairs(paired),
        size_strata_summary=A.aggregate_pairs(strata,stratified=True),support_statistics=A.support_statistics(collection))
    assert A.compare(actual,independent)==0.


def test_all_fourteen_numerical_sample_interfaces_and_actual_bias_detection():
    parts=A.C.literal_parts();schema=A.REF.reference_partition_schema(parts)
    yy,xx=np.mgrid[:144,:192]
    features=np.column_stack((np.ones(432),np.sin(np.arange(432)*.17))).astype(np.float32)
    inputs=dict(q_to_k_xy=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32),
        q_to_k_pixel_cov=np.broadcast_to(np.eye(2)/12,(27648,2,2)).copy().astype(np.float32),
        q_features=features,k_features=features[::-1].copy())
    supports={};queries={};keys={}
    for q in A.PARTITIONS:
        queries[q]=A.REF.AREF.reference_support(inputs,schema,q)
        keys[q]=A.REF.AREF.reference_key(inputs,schema,q)
        supports[q+'__mean']=queries[q]['mean'].astype(np.float32)
        supports[q+'__cov']=queries[q]['cov'].astype(np.float32)
    samples={}
    for c in A.CONDITIONS:
        q,k=c['query'],c['key'];idx=A.REF.selected_indices(len(parts[q]['bounds']))
        bounds=(parts[k]['bounds'].astype(np.float64)-.5)/8
        mean,cov=supports[q+'__mean'][idx],supports[q+'__cov'][idx]
        raw=A.TRACE.independent_mass(mean,cov,bounds,.25,96)
        content=(queries[q]['features'].astype(np.float32)@keys[k]['features'].astype(np.float32).T)[idx]
        sample=dict(indices8=idx,raw8=raw,content8=content)
        for role,suffix in (('trace_full','obs'),('size_temperature','size')):
            tau=(A.TRACE.independent_temperature(cov,.25,10.).astype(np.float32) if suffix=='obs'
                 else A.size_temperature(parts[q]['bounds'][idx]))
            p=np.full((len(parts[q]['bounds']),len(bounds)),1/len(bounds),np.float64)
            p[idx]=A.independent_probability(raw,content,tau,bounds)
            sample.update({role:p,'tau_'+suffix+'32_8':tau,'bias_'+suffix+'32_8':tau[:,None]*content})
        samples[c['condition']]=sample
    checks=A.numeric_pair(inputs,parts,supports,samples)
    assert len(checks)==28 and all(r['passed'] for r in checks)
    assert sum(len(r['indices']) for r in checks)==224
    assert {r['role'] for r in checks}==set(A.ROLES)
    bad=deepcopy(samples);bad['g8_g8']['bias_size32_8'][0,0]+=np.float32(.01)
    with pytest.raises(ValueError,match='tau32/bias32'):A.numeric_pair(inputs,parts,supports,bad)
