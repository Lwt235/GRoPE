"""Synthetic artifact and independent geometry tests; never read real results."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from experiments import round17_validation_audit as A


def selected_fixture():
    def spec(family, index=4):
        return dict(family=family, grid_index=index, config=A.GRID[index], J=.1+index/10)
    controls = {family:spec(family) for family in A.FAMILIES[1:]}
    return dict(selection=dict(primary=spec('trace_full', 6), controls=controls,
        strong_point=controls['quartic_point'], strong_quadrature=controls['gaussian_quad4'],
        eligible_indices=[6, 9]),
        point_localization_selections={family:dict(grid_index=1, config=A.GRID[1])
                                      for family in ('gaussian_point', 'quartic_point')},
        legacy_configurations={role:spec(family, A.GRID.index(dict(h=h, temperature=t)))
                               for role, (family, h, t) in A.LEGACY.items()})


def test_independent_roles_preserve_aliases_exact_family_and_matched_configuration():
    selected = selected_fixture(); roles = A.assemble_roles(selected)
    assert len(roles['aliases']) == 19 and 9 <= len(roles['operators']) <= 19
    assert roles['aliases']['regional_gaussian_full4'] == roles['aliases']['legacy_full4']
    assert roles['aliases']['primary'] != roles['aliases']['matched_isotropic']
    constant = roles['operators'][roles['aliases']['matched_constant_temperature']]
    assert constant == dict(family='gaussian_full1', h=1., temperature=0.)
    assert roles['aliases']['primary'] != roles['aliases']['matched_constant_temperature']
    iso = roles['operators'][roles['aliases']['matched_isotropic']]
    assert iso == dict(family='trace_isotropic_matched', h=1., temperature=0.)
    assert roles['strong_point'] == 'regional_quartic_point'
    bad = deepcopy(selected); bad['selection']['primary']['grid_index'] = 4
    bad['selection']['primary']['config'] = A.GRID[4]
    with pytest.raises(ValueError, match='feasible'):
        A.assemble_roles(bad)
    bad = deepcopy(selected); bad['legacy_configurations']['point_pck']['config'] = A.GRID[7]
    bad['legacy_configurations']['point_pck']['grid_index'] = 7
    with pytest.raises(ValueError, match='legacy'):
        A.assemble_roles(bad)
    bad = deepcopy(selected); bad['selection']['strong_point'] = dict(bad['selection']['strong_point'], J=100.)
    with pytest.raises(ValueError, match='accepted regional control'):
        A.assemble_roles(bad)


def synthetic_forward_rows():
    parts = A.REGION.literal_parts(); roles = A.assemble_roles(selected_fixture())
    entry = dict(scene='synthetic/scene', pair=[1, 2], input_sha256='a'*64,
                 target_sha256='b'*64, path='/metadata/pair1')
    rows = []
    for op, cfg in roles['operators'].items():
        for condition in A.CONDITIONS:
            q, k = condition['query'], condition['key']
            rows.append(dict(scene=entry['scene'], pair=entry['pair'], operator=op, config=cfg,
                **condition, shape=[len(parts[q]['mass']), len(parts[k]['mass'])], dtype='float64',
                input_sha256=entry['input_sha256'], array_sha256='c'*64, argmax_sha256='d'*64,
                numerical=dict(all_finite=True, max_normalization_error=0., raw_zero_cells=0,
                    raw_zero_fov_rows=0,raw_tiny_fov_rows=0,raw_fov_min=.1,raw_fov_max=.9,
                    raw_floor_fov_mass=1e-30*432)))
    return rows, entry, roles, parts


def test_all_condition_population_and_committed_schema_are_required():
    rows, entry, roles, parts = synthetic_forward_rows()
    mapping = A.validate_forward_rows(rows, [entry], roles, parts)
    assert len(mapping) == 68*len(roles['operators'])
    for changed in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError, match='forward row'):
            A.validate_forward_rows(changed, [entry], roles, parts)
    for key, value in (('input_sha256', '0'*64), ('state', 'scale'), ('shape', [108,432])):
        bad = deepcopy(rows); bad[0][key] = value
        with pytest.raises(ValueError, match='configuration'):
            A.validate_forward_rows(bad, [entry], roles, parts)
    bad = deepcopy(rows); bad[0]['numerical']['max_normalization_error'] = 2.e-12
    with pytest.raises(ValueError, match='numerics'):
        A.validate_forward_rows(bad, [entry], roles, parts)


def test_replayed_metric_record_cannot_change_source_population_or_probability():
    rows, entry, roles, _ = synthetic_forward_rows(); forward = rows[0]
    direct = np.ones(forward['shape'], dtype=np.int64)
    labels = {forward['query']+'_point_visible':np.ones(forward['shape'][0], bool)}
    primary = roles['aliases']['primary']
    measured = dict(**forward, replayed_array_sha256=forward['array_sha256'], target_sha256=entry['target_sha256'],
        n_query=forward['shape'][0], n_key=forward['shape'][1], mass=int(direct.sum()),
        n_point=forward['shape'][0], probability_source='retained_primary')
    A.check_measured_row(forward, measured, entry, primary, direct, labels)
    for key, value in (('array_sha256','x'*64), ('replayed_array_sha256','x'*64),
                       ('target_sha256','x'*64), ('mass',int(direct.sum())-1), ('n_point',1)):
        with pytest.raises(ValueError, match='changed|mismatch'):
            A.check_measured_row(forward, dict(measured, **{key:value}), entry, primary, direct, labels)


def test_native_argmax_floor_geometry_and_independent_pck_metrics():
    parts = A.REGION.literal_parts(); yy, xx = np.mgrid[:144,:192]
    labels = dict(gt_xy=np.stack((xx,yy),-1).reshape(-1,2), gt_visible=np.ones(27648,bool),
        g24_point_xy=parts['g24']['mean']*8, g24_point_visible=np.ones(48,bool))
    direct, common = A.REGION.independent_counts(labels, parts, 'g24', 'g8')
    probability = direct/direct.sum(-1,keepdims=True)
    record = dict(query='g24',key='g8',scenario='scale',shape=[48,432],
        array_sha256=A.array_sha(probability), argmax_sha256=A.array_sha(probability.argmax(-1).astype(np.int32)))
    metric = A.check_probability(probability, record, labels, parts, direct, common)
    assert metric['kl_sum'] == pytest.approx(0.) and metric['mass'] == 27648
    # Native first-mode tie is a corner of the 3x3 footprint, >8px from center.
    assert metric['pck_count'] == 0 and metric['n_point'] == 48
    changed = probability.copy(); changed[0,0] = np.nextafter(changed[0,0],1.)
    with pytest.raises(ValueError, match='hash'):
        A.check_probability(changed, record, labels, parts, direct, common)
    with pytest.raises(ValueError, match='argmax'):
        A.check_probability(probability, dict(record,argmax_sha256='a'*64),labels,parts,direct,common)


def test_sensor_reference_replays_all_four_native_center_sets(tmp_path, monkeypatch):
    parts = A.REGION.literal_parts()
    qdepth = np.full((144,192), 2000, np.uint16); qdepth[4,4] = 0
    kdepth = np.full((144,192), 2000, np.uint16)
    paths = []
    for name, depth in (('q',qdepth),('k',kdepth)):
        path = tmp_path/(name+'.png')
        encoded = (((depth.astype(np.uint32)<<3)|(depth.astype(np.uint32)>>13))&65535).astype(np.uint16)
        Image.fromarray(encoded).save(path); paths.append(path)
    selection = tmp_path/'selection.json'; A.write(selection, dict(scenes=[dict(scene='synthetic')]))
    camera = tmp_path/'camera.txt'; camera.write_text('synthetic identity camera')
    index = tmp_path/'index.html'; index.write_text('synthetic timestamp index')
    frame = lambda p:SimpleNamespace(depth_path=p, K_native=np.eye(3), c2w=np.eye(4))
    from experiments import round6_region_data
    monkeypatch.setattr(round6_region_data, 'read_scene_selection', lambda *_:[frame(paths[0]),frame(paths[1])])
    meta = dict(scene='synthetic',pair=[0,1],source_selection=str(selection),source_selection_sha256=A.sha(selection),
        camera_files={str(camera):A.sha(camera)},source_index_files={str(index):A.sha(index)},
        sensor_sources={str(p):A.sha(p) for p in paths})
    provenance = dict(sensor_sha256=meta['sensor_sources'],query_rgb_index=0,key_rgb_index=1,
                      query_sensor=str(paths[0]),key_sensor=str(paths[1]))
    labels = {}
    for name, part in parts.items():
        centers = part['mean']*8; address = np.floor(centers+.5).astype(int)
        known = qdepth[address[:,1],address[:,0]] > 0
        labels.update({name+'_point_xy':np.where(known[:,None],centers,np.nan),
            name+'_point_visible':known, name+'_point_status':np.where(known,0,1).astype(np.uint8)})
    counts = A.check_sensor_labels(meta, labels, provenance, parts)
    assert counts == dict(g8=432,g16=108,mixed=243,g24=48) and sum(counts.values()) == 831
    bad = deepcopy(labels); bad['g16_point_xy'][0,0] += .1
    with pytest.raises(AssertionError):
        A.check_sensor_labels(meta,bad,provenance,parts)


def test_projection_hashes_are_exact_and_extra_float32_checks_are_separate():
    reference = dict(state='zero',transform=np.eye(4).tolist(),projected_xy_sha256='a'*64,
                     zero_float32_xy_equal=True,zero_float32_cov_equal=False)
    saved = dict(scene='s',pair=[0,1],state='zero',transform=np.eye(4).tolist(),projected_xy_sha256='a'*64)
    A.check_projection(saved,reference,'s',[0,1])
    with pytest.raises(ValueError, match='identity'):
        A.check_projection(dict(saved,projected_xy_sha256='b'*64),reference,'s',[0,1])


def test_raw_fov_diagnostics_keep_zero_tiny_rows_and_native_area_floor():
    value = dict(raw_zero_cells=4,raw_zero_fov_rows=1,raw_tiny_fov_rows=2,
        raw_fov_min=0.,raw_fov_max=1e-25,raw_floor_fov_mass=1e-30*5,
        all_finite=True,max_normalization_error=0.)
    # One zero row and one tiny nonzero row remain in the two-query population.
    A.check_numerical_diagnostics(value,(2,3),np.array([64.,64.,192.]))
    for field,new in (('raw_floor_fov_mass',3e-30),('raw_zero_cells',2),('raw_tiny_fov_rows',0),
                      ('raw_fov_min',float('nan')),('raw_zero_fov_rows',True)):
        with pytest.raises(ValueError,match='FOV'):
            A.check_numerical_diagnostics(dict(value,**{field:new}),(2,3),np.array([64.,64.,192.]))


def test_chronology_requires_design_before_forward_both_scenes_before_labels_and_timing():
    producer, design, seal = dict(created_unix=1.), dict(created_unix=2.), dict(sealed_unix=5.)
    forwards = {s:dict(started_unix=3.,finished_unix=4.) for s in A.SCENES}
    evaluations = {s:dict(started_unix=6.,finished_unix=7.) for s in A.SCENES}
    timing = dict(started_unix=8.,finished_unix=9.)
    A.verify_chronology(producer,design,seal,forwards,evaluations,timing)
    with pytest.raises(ValueError, match='chronology'):
        A.verify_chronology(producer,dict(created_unix=3.1),seal,forwards,evaluations,timing)
    bad = deepcopy(evaluations); bad[A.SCENES[0]]['started_unix'] = 4.9
    with pytest.raises(ValueError, match='chronology'):
        A.verify_chronology(producer,design,seal,forwards,bad,timing)
    with pytest.raises(ValueError, match='timing'):
        A.verify_chronology(producer,design,seal,forwards,evaluations,dict(started_unix=6.9,finished_unix=9.))


def test_freeze_is_metadata_only_refuses_late_or_overwritten_design(tmp_path, monkeypatch):
    run = tmp_path/'producer'; run.mkdir()
    source = tmp_path/'audit.py'; source.write_text('synthetic frozen source\n')
    A.write(run/'lock.json', dict(created_unix=1.,source_hashes={'audit.py':A.sha(source)}))
    (run/'lock.sha256').write_text(A.sha(run/'lock.json')+'\n')
    (run/'source_snapshot').mkdir(); (run/'source_snapshot/audit.py').write_bytes(source.read_bytes())
    monkeypatch.setattr(A,'ROOT',tmp_path)
    monkeypatch.setattr(A,'verify_files',lambda files,base=None:A.H.verify_files(files,tmp_path if base is None else base))
    monkeypatch.setattr(A,'source_files',lambda:{'audit.py':A.sha(source)})
    monkeypatch.setattr(A,'verify_regional',lambda _:(None,A.assemble_roles(selected_fixture())))
    monkeypatch.setattr(A,'verify_target_metadata',lambda _:[])  # Metadata contract separately tested upstream.
    monkeypatch.setattr(np,'load',lambda *_args,**_kw:pytest.fail('freeze decoded an array'))
    output = tmp_path/'audit'; A.freeze(output,run)
    saved = A.read(output/'lock.json')
    assert saved['created_unix'] > 1. and saved['producer_lock_sha256'] == A.sha(run/'lock.json')
    assert (output/'source_snapshot/audit.py').read_bytes() == source.read_bytes()
    assert saved['validation_labels_decoded'] is False and saved['validation_effects_read'] is False
    assert saved['expected_center_labels'] == 26592
    assert saved['expected_replayed_arrays'] == 2176*len(A.assemble_roles(selected_fixture())['operators'])
    assert saved['expected_attribution_supports'] == 704 and saved['expected_attribution_queries'] == 176928
    with pytest.raises(ValueError,match='overwrite'):
        A.freeze(output,run)
    (run/'forward').mkdir()
    with pytest.raises(ValueError,match='before validation forward'):
        A.freeze(tmp_path/'late',run)
    assert not (tmp_path/'late').exists()


def test_timing_rotates_all_operators_and_binds_inputs_and_configs():
    roles = A.assemble_roles(selected_fixture()); entries = [dict(scene='synthetic',pair=[0,1],input_sha256='a'*64)]
    order = A.expected_execution_order(entries,roles); opids = list(roles['operators'])
    assert len(order) == 9*18*len(opids)
    assert [r['operator'] for r in order[:len(opids)]] == opids
    assert [r['operator'] for r in order[len(opids):2*len(opids)]] == opids[1:]+opids[:1]
    rows = [dict(**entries[0],operator=op,config=cfg) for op,cfg in roles['operators'].items()]
    A.check_timing_sources(rows,entries,roles)
    changed = deepcopy(rows); changed[0]['input_sha256'] = 'b'*64
    with pytest.raises(ValueError,match='timed operator/input'):
        A.check_timing_sources(changed,entries,roles)
    changed = deepcopy(rows); changed[0]['config']['temperature'] = 10.
    with pytest.raises(ValueError,match='timed operator/input'):
        A.check_timing_sources(changed,entries,roles)


def test_numeric_failure_cannot_be_reported_as_scientific_success():
    assert A.decision_fields(False,dict(passed=True)) == dict(
        passed=False,metric_gates_passed=True,scientific_gate_passed=False)
    assert A.decision_fields(True,dict(passed=False)) == dict(
        passed=True,metric_gates_passed=False,scientific_gate_passed=False)
    assert A.decision_fields(True,dict(passed=True)) == dict(
        passed=True,metric_gates_passed=True,scientific_gate_passed=True)
    with pytest.raises(ValueError,match='explicit booleans'):
        A.decision_fields(True,dict(passed=1))


def test_numerical_policy_binds_gaussian32_and_quartic_control24_to_regional():
    producer = dict(numerical_policy=dict(A.NUMERICAL_POLICY))
    regional = dict(numerical_policy={k:v for k,v in A.NUMERICAL_POLICY.items() if k != 'quartic_control_quadrature_order'})
    A.check_numerical_policy(producer,regional)
    for name,value in (('gaussian_quadrature_order',24),('quartic_control_quadrature_order',16),
                       ('floor',1e-20),('content_bias_dtype','float64')):
        altered = dict(numerical_policy=dict(A.NUMERICAL_POLICY,**{name:value}))
        with pytest.raises(ValueError,match='policy changed'):
            A.check_numerical_policy(altered,regional)
        with pytest.raises(ValueError,match='policy changed'):
            A.check_numerical_policy(altered,dict(numerical_policy={k:v for k,v in altered['numerical_policy'].items()
                if k != 'quartic_control_quadrature_order'}))


def test_native_archive_paths_bind_scene_pair_state_and_condition():
    entry = dict(scene='a/b',path='/old/000_1_2')
    condition = dict(scenario='scale',condition='g24_g8')
    assert A.retained_path(entry,condition) == Path('forward/a__b/arrays/000_1_2/scale__g24_g8.npz')


def test_auditor_shares_only_explicit_preparation_forward_not_production_metrics():
    tree = ast.parse(Path(A.__file__).read_text())
    forbidden = {'round17_validation_metrics'}
    producer_imports = []
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert not any(name.name in forbidden for name in node.names)
            assert not (node.module and node.module.rsplit('.',1)[-1] in forbidden)
            producer_imports.extend(name for name in node.names if name.name == 'round17_validation')
    assert len(producer_imports) == 1 and producer_imports[0].asname == 'SHARED'
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
             and n.func.value.id == 'SHARED'}
    assert calls == {'runtime','prepare_state','predict_prepared'}


def test_every_shared_replay_requires_exact_probability_argmax_and_raw_diagnostics():
    p = np.array([[.4,.6],[.5,.5]], np.float64)
    diagnostics = dict(raw_zero_fov_rows=0, all_finite=True)
    record = dict(shape=list(p.shape), array_sha256=A.array_sha(p),
        argmax_sha256=A.array_sha(p.argmax(-1).astype(np.int32)), numerical=diagnostics)
    assert A.check_replayed_probability(p, diagnostics, record) is p
    altered = p.copy(); altered[0,0] = np.nextafter(altered[0,0], 1.)
    with pytest.raises(ValueError,match='replay/hash'):
        A.check_replayed_probability(altered,diagnostics,record)
    with pytest.raises(ValueError,match='replay/hash'):
        A.check_replayed_probability(p,dict(diagnostics,raw_zero_fov_rows=1),record)
    with pytest.raises(ValueError,match='replay/hash'):
        A.check_replayed_probability(p,diagnostics,dict(record,argmax_sha256='f'*64))


def attribution_fixture():
    from models import round17_trace_pe as TRACE
    roles = A.assemble_roles(selected_fixture())
    cfg=roles['operators'][roles['aliases']['primary']]
    cfg['temperature']=10.
    roles['operators'][roles['aliases']['matched_isotropic']]['temperature']=10.
    banks={}
    for name,n in (('g8',12),('g16',6),('g24',3)):
        covariance=torch.tensor([[.1,.03],[.03,.4]],dtype=torch.float32).expand(n,-1,-1).clone()
        base=dict(mean=torch.arange(n*2,dtype=torch.float32).reshape(n,2),
                  features=torch.ones((n,4),dtype=torch.float32))
        full=dict(base,cov=covariance,tau_cov=covariance)
        iso_cov=covariance.double().diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*torch.eye(2,dtype=torch.float64)
        iso=dict(base,cov=iso_cov,tau_cov=covariance.clone())
        banks[name]=dict(trace_full=full,trace_isotropic_matched=iso)
    return banks,roles,TRACE.trace_temperature


def test_attribution_checks_all_query_sets_original_cross_terms_and_exact_tau():
    banks,roles,temperature=attribution_fixture()
    rows=A.trace_attribution(banks,roles,temperature)
    assert len(rows)==3 and sum(r['query_count'] for r in rows)==21
    assert all(r['tau32_exact'] and r['maximum_tau_difference']==0 for r in rows)
    assert all(r['full_covariance_sha256']==r['isotropic_original_covariance_sha256'] for r in rows)
    assert all(r['shared_production_replay'] and not r['independent_support_preparation'] for r in rows)
    changed=deepcopy(banks);changed['g16']['trace_isotropic_matched']['tau_cov'][:,0,1]=0.
    with pytest.raises(ValueError,match='original full covariance'):
        A.trace_attribution(changed,roles,temperature)
    changed=deepcopy(banks);changed['g24']['trace_isotropic_matched']['cov'][0,0,1]=1.e-20
    with pytest.raises(ValueError,match='original full covariance'):
        A.trace_attribution(changed,roles,temperature)


def test_attribution_rejects_different_tau_arithmetic_even_if_covariances_match():
    banks,roles,temperature=attribution_fixture();calls=[0]
    def changed(cov,h,t):
        calls[0]+=1
        value=temperature(cov,h,t)
        return torch.nextafter(value,torch.full_like(value,float('inf'))) if calls[0]%2==0 else value
    with pytest.raises(ValueError,match='effective tau'):
        A.trace_attribution(banks,roles,changed)


def test_prepared_numeric_loader_replays_full_content_before_fixed_eight_rows():
    from models import round17_validation_pe as MODEL
    previous=torch.get_num_threads();torch.set_num_threads(2)
    try:
        n=12
        cov=torch.tensor([[.12,.03],[.03,.24]],dtype=torch.float32).expand(n,-1,-1).clone()
        mean=torch.stack((torch.linspace(.1,2.8,n),torch.linspace(.2,1.8,n)),-1)
        features=torch.nn.functional.normalize(torch.arange(n*4,dtype=torch.float32).reshape(n,4)+1,dim=-1)
        family='trace_isotropic_matched'
        query=dict(mean=mean,cov=cov.double().diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*torch.eye(2,dtype=torch.float64),
                   tau_cov=cov,features=features,family=family)
        bounds=torch.tensor([[0,0,8,8],[8,0,16,8],[0,8,8,16],[8,8,16,16]],dtype=torch.int64)
        key=MODEL.OLD._key_cells(bounds,'gaussian_full1');key['features']=torch.eye(4)
        cfg=dict(family=family,h=.25,temperature=10.)
        def predict(qs,ks,config):
            p,d=MODEL.predict(qs[family],ks[family],family,config['h'],config['temperature'])
            return p.detach().numpy(),d
        backend=SimpleNamespace(P=MODEL,predict_prepared=predict)
        probability,diagnostics=predict({family:query},{family:key},cfg)
        record=dict(shape=list(probability.shape),array_sha256=A.array_sha(probability),
            argmax_sha256=A.array_sha(probability.argmax(-1).astype(np.int32)),numerical=diagnostics)
        prepared=A.prepared_numeric_reference(backend,{family:query},{family:key},cfg,record)
        indices=np.linspace(0,n-1,8,dtype=np.int64)
        np.testing.assert_array_equal(prepared['indices'],indices)
        np.testing.assert_array_equal(prepared['content'],(features@key['features'].T).numpy()[indices])
        np.testing.assert_array_equal(prepared['probability'],probability[indices])
        assert prepared['mean'].dtype==prepared['tau_cov'].dtype==prepared['content'].dtype==np.float32
        assert prepared['cov'].dtype==prepared['raw'].dtype==prepared['probability'].dtype==np.float64
        assert prepared['tau32'].shape==(8,) and prepared['bias32'].shape==(8,4)
        assert prepared['full_probability_sha256']==prepared['committed_probability_sha256']==record['array_sha256']
        assert prepared['recomputed_not_original_intermediates'] is True
        with pytest.raises(ValueError,match='replay/hash'):
            A.prepared_numeric_reference(backend,{family:query},{family:key},cfg,dict(record,array_sha256='f'*64))
    finally:
        torch.set_num_threads(previous)


def repair_fixture(tmp_path,monkeypatch):
    directory=tmp_path/'regional';directory.mkdir()
    regional_lock=dict(created_unix=1.,source_hashes={'old.py':'a'*64})
    A.write(directory/'lock.json',regional_lock)
    root=tmp_path/'repair';nested=root/'audit';nested.mkdir(parents=True)
    repaired=dict(created_unix=3.,source_hashes={'repair.py':'b'*64},scope='only historical snapshot check',
        audit_output=str(nested.resolve()),original_contract=dict(run=str(directory.resolve()),
          joint_lock_sha256=A.sha(directory/'lock.json'),original_source_hashes=regional_lock['source_hashes'],
          original_design_created_unix=1.,bound_files={'historical_failure.log':'c'*64}))
    A.write(root/'lock.json',repaired);digest=A.sha(root/'lock.json')
    receipt=dict(repair_lock_sha256=digest,repair_frozen_unix=3.,repair_source_hashes=repaired['source_hashes'],
        original_joint_lock_sha256=repaired['original_contract']['joint_lock_sha256'],
        path_repair_scope=repaired['scope'],path_repair_frozen_before_regional_forward=False,
        original_failure_bound_files=repaired['original_contract']['bound_files'])
    audit_lock=dict(path_repair=receipt)
    audit=dict(path_repair=receipt,started_unix=4.,finished_unix=5.)
    A.write(nested/'lock.json',audit_lock);A.write(nested/'verification.json',audit)
    A.write(nested/'completion.json',dict(passed=True))
    A.write(root/'completion.json',dict(passed=True,scientific_gate_passed=False,repair_lock_sha256=digest,
        verification_path=str(nested/'verification.json'),verification_sha256=A.sha(nested/'verification.json'),
        finished_unix=6.,files={str(p.relative_to(root)):A.sha(p) for p in root.rglob('*') if p.is_file()}))
    prior=dict(audit_repair=dict(root=str(root.resolve()),lock_sha256=digest,
        completion_sha256=A.sha(root/'completion.json'),audit_completion_sha256=A.sha(nested/'completion.json'),
        source_hashes=repaired['source_hashes'],scope=repaired['scope'],frozen_before_regional_forward=False))
    monkeypatch.setattr(A.REPAIR,'verify_repair',lambda _:repaired)
    return prior,directory,regional_lock,nested/'verification.json',audit_lock,audit,dict(created_unix=7.)


def test_regional_repair_is_bound_as_honest_later_code_not_fabricated_preforward_snapshot(tmp_path,monkeypatch):
    args=repair_fixture(tmp_path,monkeypatch)
    result=A.verify_regional_repair(*args)
    assert result['frozen_before_regional_forward'] is False
    assert not (args[3].parent/'source_snapshot').exists()
    bad=list(deepcopy(args));bad[0]['audit_repair']['completion_sha256']='d'*64
    with pytest.raises(ValueError,match='engineering repair'):
        A.verify_regional_repair(*bad)
    bad=list(deepcopy(args));bad[5]['path_repair']['path_repair_frozen_before_regional_forward']=True
    with pytest.raises(ValueError,match='chronology'):
        A.verify_regional_repair(*bad)
    bad=list(deepcopy(args));bad[6]['created_unix']=5.9
    with pytest.raises(ValueError,match='chronology'):
        A.verify_regional_repair(*bad)


def test_regional_repair_rejects_extra_unbound_artifact(tmp_path,monkeypatch):
    args=repair_fixture(tmp_path,monkeypatch)
    (args[3].parent.parent/'unbound.txt').write_text('unbound')
    with pytest.raises(ValueError,match='file population'):
        A.verify_regional_repair(*args)
