"""Literal R17 validation invariants on complete synthetic populations only."""
from copy import deepcopy
from fractions import Fraction
import ast
from pathlib import Path

import pytest
from experiments import round17_validation_audit_metrics as A


def role_manifest():
    families={'primary':'trace_full','matched_isotropic':'trace_isotropic_matched',
        'matched_constant_temperature':'gaussian_full1',
        **{'regional_'+f:f for f in A.CONTROL_FAMILIES},
        **{'localizer_'+f:f for f in ('gaussian_point','quartic_point')},
        **{'legacy_'+n:v[0] for n,v in A.LEGACY.items()}}
    specs={a:dict(family=f,h=.5,temperature=5.) for a,f in families.items()}
    for a in ('primary','matched_isotropic','matched_constant_temperature'):
        specs[a].update(h=1.,temperature=0.)
    specs['localizer_gaussian_point']['h']=1.
    for name,(f,h,t) in A.LEGACY.items():specs['legacy_'+name]=dict(family=f,h=h,temperature=t)
    aliases={};operators={};seen={}
    for alias in A.ALIASES:
        cfg=specs[alias];identity=(cfg['family'],cfg['h'],cfg['temperature'])
        op=seen.setdefault(identity,f'o{len(seen):02d}')
        aliases[alias]=op;operators[op]=cfg
    return dict(aliases=aliases,operators=operators,strong_point='regional_gaussian_point',
                strong_quad='regional_gaussian_quad4',primary='primary')


def cohort():
    return [dict(scene=s,pair=[2*i,2*i+1]) for s in A.SCENES for i in range(16)]


def rows_for(roles,entries):
    return [dict(operator=o,**e,scenario=z,condition=q+'_'+k,query=q,key=k,config=dict(cfg),
        kl_sum=800. if o==roles['aliases']['primary'] else 1000.,js_sum=100.,
        direct_kl_sum=800. if o==roles['aliases']['primary'] else 1000.,direct_js_sum=100.,
        mass=1000,pck_count=20,n_point=25,n_query=A.COUNTS[q],n_key=A.COUNTS[k],
        numerical=dict(all_finite=True,max_normalization_error=0.))
        for o,cfg in roles['operators'].items() for e in entries
        for z,conditions in A.CONDITIONS.items() for q,k in conditions]


def timing_for(roles):
    entries=[dict(scene=s,pair=[0,1]) for s in (*A.TRAIN,*A.SCENES)]
    rows=[dict(**e,operator=o,condition=q+'_'+k,query=q,key=k,
        samples_seconds=[.8 if o==roles['aliases']['primary'] else 1.]*15,warmup_seconds=[1.]*3)
        for o in roles['operators'] for e in entries for q,k in A.ORIGINAL]
    return entries,rows


@pytest.fixture(scope='module')
def case():
    roles=role_manifest();entries=cohort();rows=rows_for(roles,entries);te,tr=timing_for(roles)
    return roles,entries,rows,A.summarize(rows,entries,roles),te,tr,A.aggregate_timing(tr,te,roles)


def test_complete_population_and_attribution_roles(case):
    roles,_,rows,metrics,_,_,timing=case
    assert len(A.ALIASES)==19 and len(rows)==2176*len(roles['operators'])
    assert sum(map(len,A.CONDITIONS.values()))==68
    assert roles['aliases']['matched_constant_temperature']!=roles['aliases']['primary']
    result=A.gates(metrics,roles,timing)
    assert result['passed'] and result['isotropic_is_attribution_only']
    assert result['constant_temperature_is_attribution_only']
    assert not result['general_research_goal_achieved'] and not result['fresh_confirmation_authorized']


@pytest.mark.parametrize('alias',['matched_constant_temperature','matched_isotropic'])
def test_matched_controls_cannot_be_retuned(alias):
    roles=role_manifest();roles['operators'][roles['aliases'][alias]]['temperature']=10.
    with pytest.raises(ValueError):A.validate_roles(roles)


def test_legacy_and_canonical_role_constraints():
    roles=role_manifest();roles['operators'][roles['aliases']['legacy_full4']]['h']=2.
    with pytest.raises(ValueError):A.validate_roles(roles)
    roles=role_manifest();roles['aliases']['primary']='unknown'
    with pytest.raises((KeyError,ValueError)):A.validate_roles(roles)


@pytest.mark.parametrize('kind',['missing','duplicate','mass','point','shape','config','nan','normalization'])
def test_bad_population_is_not_silently_reduced(case,kind):
    roles,entries,rows,*_=case;changed=list(rows)
    if kind=='missing':changed=changed[:-1]
    elif kind=='duplicate':changed[-1]=changed[0]
    else:
        changed[0]=deepcopy(changed[0]);r=changed[0]
        if kind=='mass':r['mass']=999
        elif kind=='point':r['n_point']=24
        elif kind=='shape':r['n_query']=48
        elif kind=='config':r['config']['h']=2.
        elif kind=='nan':r['kl_sum']=float('nan')
        else:r['numerical']['max_normalization_error']=1.1e-12
    with pytest.raises(ValueError):A.summarize(changed,entries,roles)


def test_scene_macros_and_fraction_counts_keep_different_denominators(case):
    roles,entries,rows,*_=case;changed=[]
    for r in rows:
        i=A.SCENES.index(r['scene']);ci=A.CONDITIONS[r['scenario']].index((r['query'],r['key']))
        mass=(100,1000)[i];v=1+2*i+ci/10
        changed.append(dict(r,mass=mass,kl_sum=mass*v,direct_kl_sum=mass*(v if r['scenario']=='scale' else v+5),
                            n_point=(20,40)[i],pck_count=(10,36)[i]))
    m=A.summarize(changed,entries,roles)
    assert m['zero']['primary']['kl']==pytest.approx(2.4)
    assert m['zero']['primary']['direct_kl']==pytest.approx(7.4)
    assert A.pck(m['zero']['primary'],'g8_g8')==Fraction(7,10)


@pytest.mark.parametrize('scenario,condition,section',[
    ('zero','g8_g8','original'),('zero','g16_g8','new_q16_protection'),
    ('scale','g24_g8','scale'),('depth09','g8_g8','stress')])
def test_exact_half_pp_boundary_and_next_count_fail(case,scenario,condition,section):
    roles,entries,rows,_,_,_,timing=case;changed=deepcopy(rows);selected=[]
    for r in changed:
        if r['operator']==roles['aliases']['primary'] and r['scenario']==scenario and r['condition']==condition and r['pair']==[0,1]:
            r['pck_count']=18;selected.append(r)
    assert len(selected)==2
    assert A.gates(A.summarize(changed,entries,roles),roles,timing)['passed']
    selected[0]['pck_count']=17
    result=A.gates(A.summarize(changed,entries,roles),roles,timing)
    assert not result['passed']
    part=result[section][scenario] if section=='stress' else result[section]
    assert not part['passed']


def test_each_legacy_and_new_reference_remains_binding(case):
    roles,entries,rows,_,_,_,timing=case
    for alias in ('legacy_point','legacy_quad16','regional_gaussian_point','regional_gaussian_quad4'):
        changed=[dict(r,kl_sum=.7*r['mass'],direct_kl_sum=.7*r['mass']) if r['operator']==roles['aliases'][alias] else r for r in rows]
        assert not A.gates(A.summarize(changed,entries,roles),roles,timing)['passed']
    for alias in A.LOCALIZERS:
        changed=[dict(r,pck_count=21) if r['operator']==roles['aliases'][alias] else r for r in rows]
        assert not A.gates(A.summarize(changed,entries,roles),roles,timing)['passed']


def test_each_six_stress_guards_and_each_scale_condition(case):
    roles,entries,rows,_,_,_,timing=case;primary=roles['aliases']['primary']
    for state in A.STATES[1:]:
        changed=[dict(r,kl_sum=1030.) if r['operator']==primary and r['scenario']==state else r for r in rows]
        assert not A.gates(A.summarize(changed,entries,roles),roles,timing)['stress'][state]['passed']
    for q,k in A.SCALE:
        changed=[dict(r,kl_sum=1030.,direct_kl_sum=1030.) if r['operator']==primary and r['scenario']=='scale' and r['condition']==q+'_'+k else r for r in rows]
        assert not A.gates(A.summarize(changed,entries,roles),roles,timing)['scale']['partition']['passed']


def test_attribution_is_reported_without_primary_reselection(case):
    roles,entries,rows,_,_,_,timing=case
    attribution={roles['aliases'][a] for a in ('matched_isotropic','matched_constant_temperature')}
    changed=[dict(r,kl_sum=1.,direct_kl_sum=1.) if r['operator'] in attribution else r for r in rows]
    result=A.gates(A.summarize(changed,entries,roles),roles,timing)
    assert result['passed'] and roles['primary']=='primary'


def test_cost_uses_per_cell_median_and_requires_whole_population(case):
    roles,_,_,_,entries,rows,_=case;changed=deepcopy(rows);op=roles['aliases']['primary']
    for i,r in enumerate(r for r in changed if r['operator']==op):r['samples_seconds']=[.01]*7+[float(i+1)]*8
    assert A.aggregate_timing(changed,entries,roles)[op]['mean_seconds']==23.
    for invalid in (rows[:-1],[dict(rows[0],samples_seconds=[1.]*14),*rows[1:]],
                    [dict(rows[0],warmup_seconds=[1.,0.,1.]),*rows[1:]]):
        with pytest.raises(ValueError):A.aggregate_timing(invalid,entries,roles)


def test_no_producer_or_model_import():
    tree=ast.parse(Path(A.__file__).read_text())
    assert all(not isinstance(n,ast.ImportFrom) or not (n.module or '').startswith(('experiments','models')) for n in ast.walk(tree))
