"""R20 producer metrics; no file access, model calls or parameter selection."""
from __future__ import annotations

from fractions import Fraction
from math import fsum, sqrt
import numpy as np

SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
COUNTS = dict(g8=432, g16=108, mixed=243, g24=48)
CONDITIONS = tuple(dict(scenario=s, query=q, key=k, condition=q+'_'+k)
    for s, pairs in (('partition', tuple((q,k) for q in ('g8','g16','mixed') for k in ('g8','g16','mixed'))),
                     ('scale', (('g8','g24'),('g24','g8'),('g16','g24'),('g24','g16'),('g24','g24'))))
    for q,k in pairs)
METRICS = ('kl', 'js', 'direct_kl', 'direct_js')


def fraction_dict(value):
    return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))


def fraction(value):
    return Fraction(value['numerator'], value['denominator'])


def mean(values):
    values = list(values)
    if not values: raise ValueError('empty averaging population')
    return fsum(values)/len(values)


def probability(value, shape):
    if (not isinstance(value,np.ndarray) or value.dtype!=np.float64 or value.shape!=tuple(shape)
            or not np.isfinite(value).all() or np.any(value<0)
            or np.max(np.abs(value.sum(-1)-1))>1.e-12):
        raise ValueError('invalid native float64 probability')


def target_counts(labels, parts, query, key):
    xy, visible = np.asarray(labels['gt_xy'],np.float64), labels['gt_visible']
    if xy.shape!=(144*192,2) or visible.shape!=(144*192,) or visible.dtype!=np.bool_:
        raise ValueError('original dense target population required')
    good=xy[visible]
    if (not np.isfinite(good).all() or np.any(good<-.5) or np.any(good>=[191.5,143.5])):
        raise ValueError('visible target outside original image')
    pixel=np.clip(np.floor(good+.5+1.e-9),[0,0],[191,143]).astype(np.int64)
    flat=pixel[:,1]*192+pixel[:,0]
    qi=parts[query]['owner'][visible]; ki=parts[key]['owner'][flat]
    direct=np.bincount(qi*COUNTS[key]+ki,minlength=COUNTS[query]*COUNTS[key]).reshape(COUNTS[query],COUNTS[key])
    common=np.bincount(parts['g16']['owner'][visible]*432+parts['g8']['owner'][flat],
                       minlength=108*432).reshape(108,432)
    if direct.sum()!=visible.sum() or common.sum()!=visible.sum():
        raise ValueError('target population lost')
    return direct,common


def divergence(p, counts):
    mass=counts.sum(-1); truth=counts/np.maximum(mass,1)[:,None]
    logtruth=np.log(np.maximum(truth,1.e-15)); logp=np.log(np.maximum(p,1.e-15))
    midpoint=(truth+p)*.5; logmid=np.log(np.maximum(midpoint,1.e-15))
    kl=float(np.sum(counts*(logtruth-logp)))
    js=.5*(truth*(logtruth-logmid)+p*(logp-logmid)).sum(-1)
    return kl,float(np.sum(js*mass)),int(mass.sum())


def region_metrics(p, parts, condition, direct, common):
    q,k=condition['query'],condition['key']
    probability(p,(COUNTS[q],COUNTS[k]))
    dk,dj,mass=divergence(p,direct)
    if condition['scenario']=='partition':
        qp,kp=parts[q],parts[k]
        if np.any(qp['parent']<0): raise ValueError('invalid common parent')
        lifted=p[:,kp['atoms']]*(64/kp['mass'][kp['atoms']])[None]
        mapped=np.zeros((108,432),np.float64)
        np.add.at(mapped,qp['parent'],lifted*qp['mass'][:,None]/256)
        probability(mapped,(108,432))
        kl,js,cmass=divergence(mapped,common)
        if cmass!=mass:raise ValueError('native/common target mismatch')
    elif condition['scenario']=='scale': kl,js=dk,dj
    else:raise ValueError('unknown condition')
    return dict(kl_sum=kl,js_sum=js,direct_kl_sum=dk,direct_js_sum=dj,
                mass=mass,n_query=len(p),n_key=p.shape[1])


def center_metrics(argmax, bounds, point, valid):
    if (point.shape!=(len(argmax),2) or valid.shape!=(len(argmax),) or valid.dtype!=np.bool_
            or not np.isfinite(point[valid]).all() or argmax.dtype!=np.int32
            or np.any(argmax<0) or np.any(argmax>=len(bounds))):
        raise ValueError('invalid original center population or winner')
    centers=(bounds[:,:2]+bounds[:,2:]-1)/2
    distance2=np.square(centers[argmax][valid]-point[valid]).sum(-1)
    distance=np.sqrt(distance2); count=int(valid.sum())
    total=float(distance.sum()); squared=float(distance2.sum())
    return dict(pck_count=int((distance2<=64.).sum()),n_point=count,n_query=len(argmax),
        distance_sum=total,distance_squared_sum=squared,
        mean=total/count if count else None,rmse=sqrt(squared/count) if count else None)


def _center_aggregate(rows):
    n=sum(r['n_point'] for r in rows)
    if n<=0:raise ValueError('empty scene center denominator')
    out={k:sum(r[k] for r in rows) for k in ('pck_count','n_point','n_query')}
    out['pck']=fraction_dict(Fraction(out['pck_count'],n))
    has=['distance_sum' in r for r in rows]
    if any(has) and not all(has):raise ValueError('mixed continuous provenance')
    if all(has):
        total=fsum(r['distance_sum'] for r in rows); squared=fsum(r['distance_squared_sum'] for r in rows)
        out.update(distance_sum=total,distance_squared_sum=squared,mean=total/n,rmse=sqrt(squared/n))
    return out


def summarize_role(regional_rows, center_rows):
    """Already identity-checked rows; preserve scene weighting and separate masks."""
    by_scene_condition={};by_scene={};by_condition={};macro={}
    if regional_rows:
        for scene in SCENES:
            by_scene_condition[scene]={}
            for c in CONDITIONS:
                selected=[r for r in regional_rows if r['scene']==scene and r['condition']==c['condition']]
                if len(selected)!=16:raise ValueError('incomplete scene/condition regional population')
                mass=sum(r['mass'] for r in selected)
                if mass<=0:raise ValueError('empty regional population')
                by_scene_condition[scene][c['condition']]=dict(
                    **{m:fsum(r[m+'_sum'] for r in selected)/mass for m in METRICS},mass=mass)
            v={name:mean(by_scene_condition[scene][c['condition']]['kl'] for c in CONDITIONS if c['scenario']==scenario)
                for name,scenario in (('common9_kl','partition'),('direct5_kl','scale'))}
            by_scene[scene]=dict(v,J=.5*(v['common9_kl']+v['direct5_kl']))
        by_condition={c['condition']:{m:mean(by_scene_condition[s][c['condition']][m] for s in SCENES)
                                      for m in METRICS} for c in CONDITIONS}
        macro={m:mean(by_scene[s][m] for s in SCENES) for m in ('common9_kl','direct5_kl')}
        macro['J']=.5*(macro['common9_kl']+macro['direct5_kl'])
    center_scene={};center_condition={}
    if center_rows:
        conditions=[c['condition'] for c in CONDITIONS if any(r['condition']==c['condition'] for r in center_rows)]
        for scene in SCENES:
            center_scene[scene]={}
            for c in conditions:
                selected=[r for r in center_rows if r['scene']==scene and r['condition']==c]
                if len(selected)!=16:raise ValueError('incomplete scene/condition center population')
                center_scene[scene][c]=_center_aggregate(selected)
        for c in conditions:
            v=dict(pck=fraction_dict(sum((fraction(center_scene[s][c]['pck']) for s in SCENES),Fraction())/3))
            if all('mean' in center_scene[s][c] for s in SCENES):
                v.update(mean=mean(center_scene[s][c]['mean'] for s in SCENES),
                         rmse=mean(center_scene[s][c]['rmse'] for s in SCENES))
            center_condition[c]=v
    return dict(by_scene_condition=by_scene_condition,by_condition=by_condition,by_scene=by_scene,macro=macro,
                center=dict(by_scene_condition=center_scene,by_condition=center_condition))


def screen(by_role,old_by_role):
    primary=by_role['anchor_trace']; checks=[]
    def regional(category,name,reference,metric,ref,multiplier,condition=None):
        a=primary['by_condition'][condition]['kl'] if condition else primary['macro'][metric]
        b=ref['by_condition'][condition]['kl'] if condition else ref['macro'][metric]
        bound=b*multiplier
        checks.append(dict(category=category,name=name,metric=metric,reference=reference,
                           candidate_value=float(a),reference_value=float(b),bound=float(bound),passed=bool(b>0 and a<=bound)))
    for role in ('strong_point','legacy_point','strong_quad','legacy_quad16'):
        for metric in ('common9_kl','direct5_kl'):
            multiple=.98 if 'point' in role else 1.01 if metric=='common9_kl' else 1.02
            regional('old_macro',metric+'_vs_'+role,role,metric,old_by_role[role],multiple)
    for c in CONDITIONS:
        for role in ('strong_quad','legacy_quad16'):
            regional('old_condition',c['condition']+'_vs_'+role,role,'kl',old_by_role[role],1.02,c['condition'])
    for role in ('anchor_point_regional','anchor_quartic_regional','anchor_fixed_regional'):
        for metric in ('common9_kl','direct5_kl'):
            regional('same_a_macro',metric+'_vs_'+role,role,metric,by_role[role],.98)
    for category,refs,source in (
        ('old_center',('point_localizer','quartic_localizer','legacy_point_localizer'),old_by_role),
        ('same_a_center',('anchor_point_localizer','anchor_quartic_localizer','anchor_point_legacy'),by_role)):
        for q in ('g8','g16','g24'):
            c=q+'_g8';a=fraction(primary['center']['by_condition'][c]['pck'])
            for role in refs:
                b=fraction(source[role]['center']['by_condition'][c]['pck']);bound=b-Fraction(1,200)
                checks.append(dict(category=category,name=c+'_vs_'+role,metric='pck',reference=role,
                    candidate_value=fraction_dict(a),reference_value=fraction_dict(b),bound=fraction_dict(bound),
                    passed=a>=bound))
    if len(checks)!=60:raise ValueError('all sixty fixed comparisons required')
    passed=all(c['passed'] for c in checks)
    return dict(passed=passed,checks=checks,primary='anchor_trace',scientific_gate_passed=False,
                next_action='consider_separately_frozen_same_information_calibration' if passed else 'stop_fixed_anchor_candidate')
