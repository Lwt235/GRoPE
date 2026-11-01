"""R17 independent literal validation reductions, with inherited R15 gates.

Only the role manifest changes: trace_full primary plus two matched attribution
controls. The arithmetic/thresholds retain the frozen independent R15 formulas;
no new producer, model, data loader, or selector is imported.
"""
from __future__ import annotations

from fractions import Fraction
from collections import defaultdict
import math
from numbers import Integral, Real

SCENES=('harvard_c8/hv_c8_3','brown_bm_4/brown_bm_4')
TRAIN=('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
STATES=('zero','depth09','depth11','ry_plus1','ry_minus1','tx_plus01','tx_minus01')
ORIGINAL=tuple((q,k) for q in ('g8','g16','mixed') for k in ('g8','g16','mixed'))
SCALE=(('g8','g24'),('g24','g8'),('g16','g24'),('g24','g16'),('g24','g24'))
CONDITIONS={**{s:ORIGINAL for s in STATES},'scale':SCALE}
COUNTS=dict(g8=432,g16=108,mixed=243,g24=48)
CONTROL_FAMILIES=('quartic_point','gaussian_point','gaussian_fixed','gaussian_full1','gaussian_full4','gaussian_quad4','gaussian_quad16')
LEGACY={'point':('gaussian_point',1.,5.),'point_pck':('gaussian_point',.5,5.),
        'fixed':('gaussian_fixed',.5,5.),'full1':('gaussian_full1',.25,0.),
        'full4':('gaussian_full4',.5,5.),'quad4':('gaussian_quad4',.5,5.),'quad16':('gaussian_quad16',.5,5.)}
ALIASES=('primary',*('regional_'+f for f in CONTROL_FAMILIES),'localizer_gaussian_point','localizer_quartic_point',
         *('legacy_'+n for n in LEGACY),'matched_constant_temperature','matched_isotropic')
LOCALIZERS=('localizer_gaussian_point','localizer_quartic_point','legacy_point_pck')
FIELDS=('kl_sum','js_sum','direct_kl_sum','direct_js_sum')


def finite(value):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):
        raise ValueError('nonfinite or nonnumeric audit value')
    return float(value)


def integer(value,minimum=0,maximum=None):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<minimum or (maximum is not None and value>maximum):
        raise ValueError('invalid integer audit population')
    return int(value)


def mean(values):
    values=list(values)
    if not values:raise ValueError('empty audit macro population')
    return math.fsum(values)/len(values)


def rational(value):
    return dict(numerator=value.numerator,denominator=value.denominator)


def validate_roles(roles):
    if set(roles)!={'aliases','operators','strong_point','strong_quad','primary'} or roles['primary']!='primary':
        raise ValueError('invalid role manifest')
    if tuple(roles['aliases'])!=ALIASES or not 9<=len(roles['operators'])<=19:
        raise ValueError('wrong alias/operator population')
    expected_families={'primary':'trace_full','matched_isotropic':'trace_isotropic_matched',
        'matched_constant_temperature':'gaussian_full1',
        **{'regional_'+f:f for f in CONTROL_FAMILIES},
        **{'localizer_'+f:f for f in ('gaussian_point','quartic_point')},
        **{'legacy_'+n:v[0] for n,v in LEGACY.items()}}
    encountered={};configs={}
    for alias in ALIASES:
        op=roles['aliases'][alias];cfg=roles['operators'][op]
        if set(cfg)!={'family','h','temperature'} or cfg['family']!=expected_families[alias]:
            raise ValueError('wrong alias family')
        h,t=finite(cfg['h']),finite(cfg['temperature'])
        if h not in (.25,.5,1.,2.) or t not in (0.,5.,10.):raise ValueError('configuration outside grid')
        identity=(cfg['family'],h,t)
        if alias.startswith('legacy_') and identity!=LEGACY[alias.removeprefix('legacy_')]:
            raise ValueError('legacy configuration changed')
        expected=encountered.setdefault(identity,f'o{len(encountered):02d}')
        if op!=expected:raise ValueError('noncanonical alias deduplication')
        configs[op]=cfg
    if configs!=roles['operators']:raise ValueError('extra/missing operator')
    primary=configs[roles['aliases']['primary']];iso=configs[roles['aliases']['matched_isotropic']]
    if any(primary[k]!=iso[k] for k in ('h','temperature')):raise ValueError('isotropic retuned')
    constant=configs[roles['aliases']['matched_constant_temperature']]
    if any(primary[k]!=constant[k] for k in ('h','temperature')):raise ValueError('constant-temperature attribution retuned')
    if roles['strong_point'] not in ('regional_gaussian_point','regional_quartic_point'):
        raise ValueError('wrong strong point reference')
    if roles['strong_quad'] not in ('regional_gaussian_quad4','regional_gaussian_quad16'):
        raise ValueError('wrong strong quad reference')
    return roles


def cohort(entries,timing=False):
    scenes=(*TRAIN,*SCENES) if timing else SCENES
    count=1 if timing else 16
    ids=[]
    for e in entries:
        if not isinstance(e['pair'],(tuple,list)) or len(e['pair'])!=2:raise ValueError('wrong pair shape')
        ids.append((e['scene'],tuple(integer(v) for v in e['pair'])))
    if len(ids)!=len(scenes)*count or len(set(ids))!=len(ids) or any(sum(s==v for s,_ in ids)!=count for v in scenes):
        raise ValueError('wrong complete audit cohort')
    return ids


def summarize(rows,entries,roles):
    validate_roles(roles);pairs=cohort(entries)
    expected={(s,p,o,z,q+'_'+k) for s,p in pairs for o in roles['operators'] for z,conditions in CONDITIONS.items() for q,k in conditions}
    indexed={};mass_by_pair={};point_by_pair={}
    grouped=defaultdict(list)
    for r in rows:
        if not isinstance(r['pair'],(tuple,list)) or len(r['pair'])!=2:raise ValueError('wrong metric pair')
        pair_indices=tuple(integer(v) for v in r['pair'])
        key=(r['scene'],pair_indices,r['operator'],r['scenario'],r['condition'])
        if key in indexed or key not in expected:raise ValueError('duplicate or unauthorized metric population')
        indexed[key]=r
        q,k=r['query'],r['key']
        if (q,k) not in CONDITIONS[r['scenario']] or r['condition']!=q+'_'+k or r['config']!=roles['operators'][r['operator']]:
            raise ValueError('metric configuration/condition mismatch')
        finite(r['config']['h']);finite(r['config']['temperature'])
        if integer(r['n_query'])!=COUNTS[q] or integer(r['n_key'])!=COUNTS[k]:raise ValueError('wrong native shape population')
        mass=integer(r['mass'],maximum=27648);npoint=integer(r['n_point'],maximum=COUNTS[q])
        integer(r['pck_count'],maximum=npoint)
        for f in FIELDS:
            value=finite(r[f])
            if mass==0 and value!=0:raise ValueError('nonzero metric in empty population')
        if r['scenario']=='scale' and (r['kl_sum']!=r['direct_kl_sum'] or r['js_sum']!=r['direct_js_sum']):
            raise ValueError('scale did not use native metrics')
        pair=r['scene'],pair_indices
        if mass_by_pair.setdefault(pair,mass)!=mass or point_by_pair.setdefault((*pair,q),npoint)!=npoint:
            raise ValueError('changed label population across operator/state/partition')
        error=finite(r['numerical']['max_normalization_error'])
        if r['numerical'].get('all_finite') is not True or not 0<=error<=1e-12:
            raise ValueError('invalid prediction numerical evidence')
        grouped[(r['operator'],r['scenario'],r['scene'],r['condition'])].append(r)
    if set(indexed)!=expected:raise ValueError('incomplete metric population')
    output={}
    for scenario,conditions in CONDITIONS.items():
        operators={}
        for operator in roles['operators']:
            detail={s:{} for s in SCENES};pcks={};all_numeric=[]
            for q,k in conditions:
                condition=q+'_'+k;counts={};fractions={}
                for scene in SCENES:
                    items=grouped[(operator,scenario,scene,condition)]
                    denominator=sum(r['mass'] for r in items)
                    if denominator<=0:raise ValueError('empty regional denominator')
                    detail[scene][condition]={f[:-4]:math.fsum(r[f] for r in items)/denominator for f in FIELDS}
                    detail[scene][condition]['mass']=denominator
                    n=sum(r['n_point'] for r in items);correct=sum(r['pck_count'] for r in items)
                    if n<=0:raise ValueError('empty center denominator')
                    counts[scene]=dict(pck_count=correct,n_point=n);fractions[scene]=Fraction(correct,n)
                    all_numeric.extend(r['numerical'] for r in items)
                exact=(fractions[SCENES[0]]+fractions[SCENES[1]])/2
                pcks[condition]=dict(macro=float(exact),by_scene={s:float(fractions[s]) for s in SCENES},
                                    counts_by_scene=counts,exact=rational(exact))
            by_sc={s:{c:v['kl'] for c,v in detail[s].items()} for s in SCENES}
            operators[operator]=dict(**{f:mean(mean(v[f] for v in detail[s].values()) for s in SCENES)
                                       for f in ('kl','js','direct_kl','direct_js')},
                by_scene={s:mean(by_sc[s].values()) for s in SCENES},
                by_condition={q+'_'+k:mean(by_sc[s][q+'_'+k] for s in SCENES) for q,k in conditions},
                by_scene_condition=by_sc,detail=detail,pck_by_condition=pcks,
                probability_checks=dict(all_finite=all(v['all_finite'] for v in all_numeric),matrices=len(all_numeric),
                                        max_normalization_error=max(v['max_normalization_error'] for v in all_numeric)))
        output[scenario]={alias:operators[roles['aliases'][alias]] for alias in ALIASES}
    return output


def aggregate_timing(rows,entries,roles):
    validate_roles(roles);pairs=cohort(entries,True)
    expected={(s,p,o,q+'_'+k) for s,p in pairs for o in roles['operators'] for q,k in ORIGINAL}
    seen=set();grouped={o:[] for o in roles['operators']}
    for r in rows:
        if not isinstance(r['pair'],(tuple,list)) or len(r['pair'])!=2:raise ValueError('wrong timing pair')
        identity=r['scene'],tuple(integer(v) for v in r['pair']),r['operator'],r['condition']
        if identity in seen or identity not in expected:raise ValueError('wrong paired timing population')
        seen.add(identity)
        if (r['query'],r['key']) not in ORIGINAL or r['condition']!=r['query']+'_'+r['key']:
            raise ValueError('wrong timing condition')
        for name,count in (('samples_seconds',15),('warmup_seconds',3)):
            if len(r[name])!=count or any(finite(v)<=0 for v in r[name]):raise ValueError('wrong positive timing samples')
        median_seconds=float(sorted(r['samples_seconds'])[7])
        grouped[r['operator']].append(dict(r,median_seconds=median_seconds))
    if seen!=expected:raise ValueError('incomplete paired timing')
    return {o:dict(mean_seconds=mean(c['median_seconds'] for c in cells),cells=cells) for o,cells in grouped.items()}


def gain(a,b):
    a,b=finite(a),finite(b)
    if b<=0:raise ValueError('nonpositive relative KL denominator')
    return 1-a/b


def pck(metric,condition):
    value=metric['pck_by_condition'][condition];counts=value['counts_by_scene']
    if set(counts)!=set(SCENES):raise ValueError('PCK scene set changed')
    fractions={s:Fraction(integer(counts[s]['pck_count'],maximum=integer(counts[s]['n_point'],minimum=1)),counts[s]['n_point']) for s in SCENES}
    result=sum(fractions.values(),Fraction())/2
    if value['exact']!=rational(result) or value['macro']!=float(result) or value['by_scene']!={s:float(v) for s,v in fractions.items()}:
        raise ValueError('PCK not derived from exact counts')
    return result


def pck_checks(metrics,condition):
    comparisons={}
    for name in LOCALIZERS:
        delta=pck(metrics['primary'],condition)-pck(metrics[name],condition)
        comparisons[name]=dict(passed=delta>=Fraction(-1,200),difference_pp=float(delta*100),exact_difference=rational(delta))
    return dict(passed=all(v['passed'] for v in comparisons.values()),condition=condition,comparisons=comparisons)


def point_checks(metrics,references):
    primary=metrics['primary'];out={}
    for name in references:
        control=metrics[name]
        checks=dict(relative_gain_at_least_2pct=primary['kl']<=control['kl']*.98,
                    both_scenes_improve=all(primary['by_scene'][s]<control['by_scene'][s] for s in SCENES))
        out[name]=dict(passed=all(checks.values()),checks=checks,relative_gain=gain(primary['kl'],control['kl']))
    return dict(passed=all(v['passed'] for v in out.values()),comparisons=out)


def partition_checks(metrics,references):
    out={}
    for name in references:
        values={c:dict(passed=v<=metrics[name]['by_condition'][c]*1.02,
                       relative_gain=gain(v,metrics[name]['by_condition'][c])) for c,v in metrics['primary']['by_condition'].items()}
        out[name]=dict(passed=all(v['passed'] for v in values.values()),conditions=values)
    return dict(passed=all(v['passed'] for v in out.values()),comparisons=out)


def gates(metrics,roles,timing):
    validate_roles(roles)
    if set(metrics)!=set(CONDITIONS) or set(timing)!=set(roles['operators']):raise ValueError('wrong gate population')
    numerical={}
    for state,conditions in CONDITIONS.items():
        if set(metrics[state])!=set(ALIASES):raise ValueError('wrong gate alias population')
        names={q+'_'+k for q,k in conditions}
        for alias,value in metrics[state].items():
            if set(value['by_scene'])!=set(SCENES) or set(value['by_condition'])!=names or set(value['pck_by_condition'])!=names:
                raise ValueError('incomplete gate scenario conditions')
            check=value['probability_checks'];error=finite(check['max_normalization_error'])
            if error<0 or check['matrices']!=len(conditions)*32:raise ValueError('wrong numerical coverage')
            numerical[state+'__'+alias]=check['all_finite'] is True and error<=1e-12
    for value in timing.values():
        if len(value['cells'])!=45 or finite(value['mean_seconds'])<=0 or value['mean_seconds']!=mean(finite(c['median_seconds']) for c in value['cells']):
            raise ValueError('inconsistent cost summary')
    points=(roles['strong_point'],'legacy_point');quads=(roles['strong_quad'],'legacy_quad16')
    zero=metrics['zero'];p=zero['primary']['kl'];seconds=timing[roles['aliases']['primary']]['mean_seconds'];cost={}
    for name in quads:
        control=zero[name]['kl'];t=timing[roles['aliases'][name]]['mean_seconds']
        accurate=p<=.98*control and seconds<=1.25*t
        fast=p<=1.01*control and seconds<=.5*t
        cost[name]=dict(passed=accurate or fast,accuracy_route=accurate,efficiency_route=fast,
                        relative_gain=gain(p,control),time_ratio=seconds/t)
    original=dict(point=point_checks(zero,points),
                  accuracy_cost=dict(passed=all(v['passed'] for v in cost.values()),comparisons=cost),
                  partition=partition_checks(zero,quads),pck_g8=pck_checks(zero,'g8_g8'))
    original['passed']=all(v['passed'] for v in original.values())
    q16=dict(pck_checks(zero,'g16_g8'),new_predeclared_guard=True,historical_R12_gate=False)
    scale=dict(point=point_checks(metrics['scale'],points),partition=partition_checks(metrics['scale'],quads),
               pck_g24=pck_checks(metrics['scale'],'g24_g8'))
    scale['passed']=all(v['passed'] for v in scale.values())
    stress={}
    for state in STATES:
        full=metrics[state]['primary']['kl']
        comparisons={name:dict(passed=full<=1.02*metrics[state][name]['kl'],relative_gain=gain(full,metrics[state][name]['kl'])) for name in quads}
        center=pck_checks(metrics[state],'g8_g8')
        stress[state]=dict(passed=center['passed'] and all(v['passed'] for v in comparisons.values()),quad_comparisons=comparisons,pck_g8=center)
    predictions=dict(passed=all(numerical.values()),checks=numerical)
    passed=all([original['passed'],q16['passed'],scale['passed'],predictions['passed'],*(v['passed'] for v in stress.values())])
    return dict(passed=passed,original=original,new_q16_protection=q16,scale=scale,stress=stress,
                predictions=predictions,general_research_goal_achieved=False,fresh_confirmation_authorized=False,isotropic_is_attribution_only=True,constant_temperature_is_attribution_only=True)


SOURCE_NAMES = ('experiments/round17_validation_audit_metrics.py',
                'tests/test_round17_validation_audit_metrics.py')
