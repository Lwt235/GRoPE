from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import round17_validation as P


def test_ten_field_loader_does_not_decode_unlisted_object_array(tmp_path):
    path=tmp_path/'inputs.npz'
    np.savez(path,**{name:np.zeros(1) for name in P.FIELDS},sensor_secret=np.array([{'never':'decode'}],dtype=object))
    values=P.load_inputs(path)
    assert len(values)==10 and set(values)==set(P.FIELDS)


def test_freeze_requires_accepted_calibration_before_any_new_output(tmp_path,monkeypatch):
    def rejected(_):raise ValueError('independently accepted Stage B selection required')
    monkeypatch.setattr(P,'selected_contract',rejected)
    def forbidden():raise AssertionError('must not access old targets before accepted calibration')
    monkeypatch.setattr(P,'old_target_contract',forbidden)
    output=tmp_path/'new_run'
    with pytest.raises(ValueError,match='accepted Stage B'):
        P.freeze(output,tmp_path/'regional')
    assert not output.exists()


def test_evaluation_requires_both_scene_forward_seal_before_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'verify_lock',lambda _: {})
    def unsealed(_):raise ValueError('global forward missing')
    monkeypatch.setattr(P,'verify_forward_complete',unsealed)
    monkeypatch.setattr(P,'load_labels',lambda _:pytest.fail('label access before full forward'))
    with pytest.raises(ValueError,match='global forward'):
        P.evaluate(tmp_path,P.SCENES[0])
    assert not (tmp_path/'evaluation').exists()


def test_storage_counts_cover_primary_and_two_control_sample_pairs():
    parts=P.S.M.partitions()
    n=sum(len(parts[c['query']]['mass'])*len(parts[c['key']]['mass']) for c in P.CONDITIONS)
    assert len(P.CONDITIONS)==68 and n==4345767
    for count in (9,19):
        roles={'operators':{str(i):{} for i in range(count)}}
        assert P.sample_bytes(roles)==n*8*(32+(count-1)*2)
    with pytest.raises(ValueError):P.sample_bytes({'operators':{}})


def test_native_argmax_and_distinct_common_vs_scale_measurement():
    parts=P.S.M.partitions()
    yy,xx=np.mgrid[:144,:192]
    labels=dict(gt_xy=np.stack([xx,yy],-1).reshape(-1,2),gt_visible=np.ones(144*192,bool))
    for q in parts:
        labels[q+'_point_xy']=parts[q]['mean']*8
        labels[q+'_point_visible']=np.ones(len(parts[q]['mass']),bool)
    direct,common=P.S.target_counts(labels,parts,'g24','g8')
    probability=direct/direct.sum(-1,keepdims=True)
    condition=dict(query='g24',key='g8',scenario='scale')
    value=P.measure(probability,labels,parts,condition,direct,common)
    assert value['kl_sum']==pytest.approx(0.) and value['direct_kl_sum']==pytest.approx(0.)
    # The first mode is a corner of the 3x3 target support, farther than 8px.
    assert value['pck_count']==0 and value['n_point']==48
    direct,common=P.S.target_counts(labels,parts,'g8','g8')
    probability=direct/direct.sum(-1,keepdims=True)
    metric=P.measure(probability,labels,parts,dict(query='g8',key='g8',scenario='ry_plus1'),direct,common)
    assert metric['pck_count']==432 and metric['kl_sum']==pytest.approx(0.)


def test_forward_population_rejects_missing_duplicate_and_changed_config():
    entries=[dict(scene='synthetic',pair=[0,1])]
    cfg=dict(family='trace_full',h=1.,temperature=0.)
    roles={'operators':{'o00':cfg}}
    parts=P.S.M.partitions()
    rows=[dict(scene='synthetic',pair=[0,1],operator='o00',config=cfg,**c,dtype='float64',
               shape=[len(parts[c['query']]['mass']),len(parts[c['key']]['mass'])]) for c in P.CONDITIONS]
    P.forward_population(rows,entries,roles)
    with pytest.raises(ValueError):P.forward_population(rows[:-1],entries,roles)
    with pytest.raises(ValueError):P.forward_population(rows+[rows[0]],entries,roles)
    changed=[dict(r) for r in rows];changed[0]['config']=dict(cfg,h=2.)
    with pytest.raises(ValueError):P.forward_population(changed,entries,roles)


def test_zero_path_check_requires_bitwise_float32_agreement(tmp_path,monkeypatch):
    entries=[]
    for scene in P.R.SCENES:
        directory=tmp_path/scene.replace('/','__');directory.mkdir()
        path=directory/'inputs.npz';path.write_bytes(b'fixture')
        entries.append(dict(scene=scene,pair=[0,1],path=str(directory),input_sha256=P.sha(path)))
    inputs={name:np.zeros((1,2),np.float64) for name in P.A.FIELDS}
    monkeypatch.setattr(P,'load_inputs',lambda _:inputs)
    def tensors(raw,_):return {name:torch.tensor(raw[name],dtype=torch.float32) for name in P.A.FIELDS}
    monkeypatch.setattr(P.S.A,'observation_tensors',tensors)
    diag=dict(zero_xy_max_error=0.,zero_cov_max_error=0.)
    monkeypatch.setattr(P.S.M,'perturbed_observations',lambda *_:(inputs,diag))
    assert P.zero_path_check(dict(entries=entries))['passed']
    def changed(*_):
        raw={k:v.copy() for k,v in inputs.items()};raw['q_to_k_xy'][0,0]=1.e-7
        return raw,diag
    monkeypatch.setattr(P.S.M,'perturbed_observations',changed)
    assert P.zero_path_check(dict(entries=entries))['passed'] is False


def test_retained_path_contains_state_and_condition_but_no_label_path(tmp_path):
    entry=dict(scene='a/b',path='/metadata/000_1_2')
    condition=dict(scenario='scale',condition='g24_g8')
    assert P.native_path(tmp_path,entry,condition)==tmp_path/'forward/a__b/arrays/000_1_2/scale__g24_g8.npz'


def test_production_prepares_only_each_familys_key_geometry(monkeypatch):
    operators={str(i):dict(family=f,h=.5,temperature=5.) for i,f in enumerate(
        ('trace_full','gaussian_quad16','trace_isotropic_matched'))}
    partition={name:name for name in P.S.M.PARTITIONS}
    monkeypatch.setattr(P.S.M,'perturbed_observations',lambda *_:({},{}))
    monkeypatch.setattr(P.S.A,'observation_tensors',lambda raw,_:raw)
    monkeypatch.setattr(P.P,'prepare_query',lambda obs,part,family:{'query_family':family})
    calls=[]
    def key(obs,part,family):
        calls.append((part,family));return {'key_family':family}
    monkeypatch.setattr(P.P,'prepare_key',key)
    queries,keys,_=P.prepare_state({},'zero',partition,operators)
    assert len(calls)==12 and len(queries)==len(keys)==4
    for part in partition:
        for cfg in operators.values():
            assert keys[part][cfg['family']]=={'key_family':cfg['family']}
            assert (part,cfg['family']) in calls


def acceptance_fixture(tmp_path,monkeypatch,numerical_passed,metric_passed):
    run=tmp_path/'run';run.mkdir();audit=tmp_path/'audit';audit.mkdir()
    P.write(run/'lock.json',dict(synthetic=True))
    P.write(run/'forward_complete.json',dict(synthetic=True))
    summary=dict(roles={'synthetic':'roles'},gates=dict(passed=metric_passed))
    P.write(run/'summary.json',summary)
    P.write(run/'completion.json',dict(files={p.name:P.sha(p) for p in run.iterdir()}))
    design=dict(producer_lock_sha256=P.sha(run/'lock.json'),source_hashes={})
    P.write(audit/'lock.json',design)
    (audit/'lock.sha256').write_text(P.sha(audit/'lock.json')+'\n')
    verification=dict(passed=numerical_passed,scientific_gate_passed=numerical_passed and metric_passed,
        metric_gates_passed=metric_passed,roles=summary['roles'],gates=summary['gates'],
        independent_audit_lock_sha256=P.sha(audit/'lock.json'),
        **{f'validation_{name}_sha256':P.sha(run/(name+'.json')) for name in ('lock','forward_complete','summary','completion')})
    P.write(audit/'verification.json',verification)
    P.write(audit/'completion.json',dict(passed=numerical_passed,scientific_gate_passed=numerical_passed and metric_passed,
                                        files={p.name:P.sha(p) for p in audit.iterdir()}))
    monkeypatch.setattr(P,'verify_lock',lambda _: {})
    return run,audit/'verification.json'


def test_failed_numerical_audit_cannot_promote_even_positive_metric_gates(tmp_path,monkeypatch):
    run,audit=acceptance_fixture(tmp_path,monkeypatch,False,True)
    with pytest.raises(ValueError,match='audit did not pass'):P.accept_audit(run,audit)
    assert not (run/'decision.json').exists()


@pytest.mark.parametrize('metric_passed',[False,True])
def test_audit_acceptance_preserves_scientific_failure_or_remaining_tasks(tmp_path,monkeypatch,metric_passed):
    run,audit=acceptance_fixture(tmp_path,monkeypatch,True,metric_passed)
    result=P.accept_audit(run,audit)
    assert result['scientific_gate_passed'] is metric_passed
    assert result['next_action']==('review_remaining_actual_value_and_gs_designs' if metric_passed else 'stop_unchanged_candidate_preserve_failure')
    assert result['general_research_goal_achieved'] is False and result['fresh_confirmation_authorized'] is False
    with pytest.raises(ValueError,match='overwrite'):P.accept_audit(run,audit)


def regional_fixture(tmp_path, monkeypatch, screen_passed=True):
    directory=tmp_path/'regional'; directory.mkdir()
    audit_dir=tmp_path/'audit'; audit_dir.mkdir()
    config=dict(h=.5,temperature=10.)
    primary=dict(family='trace_full',grid_index=5,config=config,J=.3)
    controls={family:dict(family=family,grid_index=4,config=dict(h=.5,temperature=5.),J=.5)
              for family in P.V.REGIONAL_FAMILIES}
    selection=dict(primary=primary,controls=controls,strong_point=controls['gaussian_point'],
                   strong_quadrature=controls['gaussian_quad16'])
    screen=dict(passed=screen_passed,next_action='prepare_old_validation_protocol')
    localizers={f:dict(grid_index=1,config=dict(h=.25,temperature=5.))
                for f in ('gaussian_point','quartic_point')}
    legacy={name:dict(family=f,config=dict(h=h,temperature=t)) for name,(f,h,t) in P.V.LEGACY.items()}
    lock=dict(stage_a=dict(point_localization_selections=localizers),legacy=dict(configurations=legacy))
    P.write(directory/'lock.json',lock); P.write(directory/'forward_complete.json',{})
    P.write(directory/'summary.json',dict(selection=selection,train_screen=screen))
    P.write(directory/'completion.json',dict(files={}))
    hashes={f'regional_{name}_sha256':P.sha(directory/(name+'.json'))
            for name in ('lock','forward_complete','summary','completion')}
    audit=dict(passed=True,scientific_gate_passed=False,selection=selection,train_screen=screen,**hashes)
    P.write(audit_dir/'verification.json',audit)
    selected=dict(selection=selection,train_screen=screen,**hashes,
        audit_path=str(audit_dir/'verification.json'),audit_sha256=P.sha(audit_dir/'verification.json'),
        next_action='prepare_old_validation_protocol',validation_run=False,scientific_gate_passed=False,
        point_localization_selections=localizers,legacy_configurations=legacy)
    P.write(directory/'selection.json',selected)
    (directory/'selection.sha256').write_text(P.sha(directory/'selection.json')+'\n')
    monkeypatch.setattr(P.R,'verify_lock',lambda _:lock)
    monkeypatch.setattr(P,'regional_audit_repair',lambda *_:dict(synthetic=True))
    return directory


def test_accepted_audit_with_failed_train_screen_cannot_start_validation(tmp_path,monkeypatch):
    directory=regional_fixture(tmp_path,monkeypatch,False)
    with pytest.raises(ValueError,match='accepted Stage B'):
        P.selected_contract(directory)


def test_fixed_primary_and_both_matched_controls_inherit_same_configuration(tmp_path,monkeypatch):
    directory=regional_fixture(tmp_path,monkeypatch)
    result=P.selected_contract(directory); roles=result['roles']
    assert len(roles['aliases'])==19 and result['train_screen']['passed']
    for alias,family in [('primary','trace_full'),('matched_constant_temperature','gaussian_full1'),
                         ('matched_isotropic','trace_isotropic_matched')]:
        assert roles['operators'][roles['aliases'][alias]]==dict(family=family,h=.5,temperature=10.)


