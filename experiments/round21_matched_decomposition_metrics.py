"""Pure R21 diagnostic metrics; no file access, effect gate or model selection."""
from fractions import Fraction
from math import fsum, sqrt
import numpy as np

from experiments import round20_anchor_fixed_metrics as OLD

SCENES=OLD.SCENES
COUNTS=dict(g8=432,g16=108,g24=48)
CONDITIONS=tuple(dict(scenario='partition' if q!='g24' else 'scale',query=q,key='g8',condition=q+'_g8')
                 for q in COUNTS)
CONTRASTS=tuple([dict(name='content_'+g,from_role=g+'_zero',to_role=g+'_content')
                 for g in ('mu_c','a_c','a_ca','full4')]+[
    dict(name=name+'_'+mode,from_role=before+'_'+mode,to_role=after+'_'+mode)
    for name,before,after in (('anchor','mu_c','a_c'),('widening','a_c','a_ca'),('shape','mu_c','full4'))
    for mode in ('zero','content')])
METRICS=OLD.METRICS
probability,target_counts,region_metrics,center_metrics=OLD.probability,OLD.target_counts,OLD.region_metrics,OLD.center_metrics
fraction_dict,fraction,mean=OLD.fraction_dict,OLD.fraction,OLD.mean


def paired_center(before,after,bounds,point,valid):
    """Unfiltered label-valid pairs; improvements are to-correct/from-incorrect."""
    a=center_metrics(before,bounds,point,valid);b=center_metrics(after,bounds,point,valid)
    centers=(bounds[:,:2]+bounds[:,2:]-1)/2
    da2=np.square(centers[before][valid]-point[valid]).sum(-1)
    db2=np.square(centers[after][valid]-point[valid]).sum(-1)
    da,db=np.sqrt(da2),np.sqrt(db2)
    ca,cb=da2<=64,db2<=64
    return dict(n_query=len(before),n_point=int(valid.sum()),from_correct=a['pck_count'],to_correct=b['pck_count'],
        both_correct=int((ca&cb).sum()),both_wrong=int((~ca&~cb).sum()),benefit=int((~ca&cb).sum()),
        harm=int((ca&~cb).sum()),flip_valid=int((before[valid]!=after[valid]).sum()),
        flip_all=int((before!=after).sum()),distance_delta_sum=float((db-da).sum()),
        squared_distance_delta_sum=float((db2-da2).sum()),from_distance_squared_sum=float(da2.sum()),
        to_distance_squared_sum=float(db2.sum()),rmse_delta=b['rmse']-a['rmse'] if a['n_point'] else None,
        distance_benefit=int((db2<da2).sum()),
        distance_harm=int((db2>da2).sum()),distance_equal=int((db2==da2).sum()))


def _average_center(values,paired=False):
    pkey='pck_delta' if paired else 'pck'
    fkeys=('mean_distance_delta','mean_squared_distance_delta','rmse_delta') if paired else ('mean','rmse')
    return dict({pkey:fraction_dict(sum((fraction(v[pkey]) for v in values),Fraction())/len(values))},
                **{k:mean(v[k] for v in values) for k in fkeys})


def _paired_aggregate(rows):
    integers=('n_query','n_point','from_correct','to_correct','both_correct','both_wrong','benefit','harm',
              'flip_valid','flip_all','distance_benefit','distance_harm','distance_equal')
    out={k:sum(r[k] for r in rows) for k in integers}
    n=out['n_point']
    if (n<=0 or out['both_correct']+out['both_wrong']+out['benefit']+out['harm']!=n
            or out['to_correct']-out['from_correct']!=out['benefit']-out['harm']
            or out['distance_benefit']+out['distance_harm']+out['distance_equal']!=n):
        raise ValueError('paired center population or transitions changed')
    for k in ('distance_delta_sum','squared_distance_delta_sum','from_distance_squared_sum','to_distance_squared_sum'):
        out[k]=fsum(r[k] for r in rows)
    out.update(pck_delta=fraction_dict(Fraction(out['benefit']-out['harm'],n)),
               mean_distance_delta=out['distance_delta_sum']/n,
               mean_squared_distance_delta=out['squared_distance_delta_sum']/n,
               rmse_delta=sqrt(out['to_distance_squared_sum']/n)-sqrt(out['from_distance_squared_sum']/n))
    return out


def summarize(rows,paired=False):
    """Exactly 16 pairs per scene/condition, equal-condition/equal-scene macros.

    Overall PCK is a mean of exact scene/scale Fractions, never a pooled count.
    RMSE macros average scene RMSEs; no pooled-RMSE interpretation is implied.
    """
    names=tuple(k+('_delta' if paired else '') for k in METRICS)
    region={};centers={}
    for scene in SCENES:
        region[scene]={};centers[scene]={}
        for spec in CONDITIONS:
            c=spec['condition'];selected=[r for r in rows if r['scene']==scene and r['condition']==c]
            if len(selected)!=16 or len({tuple(r['pair']) for r in selected})!=16:
                raise ValueError('complete fixed sixteen-pair scene/condition required')
            mass=sum(r['mass'] for r in selected)
            if mass<=0:raise ValueError('empty regional population')
            region[scene][c]=dict({k:fsum(r[k+'_sum'] for r in selected)/mass for k in names},mass=mass)
            centers[scene][c]=(_paired_aggregate if paired else OLD._center_aggregate)([r['center'] for r in selected])
    by_condition={c['condition']:{k:mean(region[s][c['condition']][k] for s in SCENES) for k in names} for c in CONDITIONS}
    by_scene={s:{k:mean(region[s][c['condition']][k] for c in CONDITIONS) for k in names} for s in SCENES}
    macro={k:mean(by_scene[s][k] for s in SCENES) for k in names}
    cc={c['condition']:_average_center([centers[s][c['condition']] for s in SCENES],paired) for c in CONDITIONS}
    cs={s:_average_center([centers[s][c['condition']] for c in CONDITIONS],paired) for s in SCENES}
    return dict(by_scene_condition=region,by_condition=by_condition,by_scene=by_scene,macro=macro,
        center=dict(by_scene_condition=centers,by_condition=cc,by_scene=cs,
                    macro=_average_center([cs[s] for s in SCENES],paired)))


def paired_row(before,after,winner_before,winner_after,bounds,point,valid,contrast):
    for key in ('scene','pair','condition','query','key','mass','n_query','n_key','label_sha256','target_sha256'):
        if before[key]!=after[key]:raise ValueError('paired rows must share complete population and label identity')
    if before['family']!=contrast['from_role'] or after['family']!=contrast['to_role']:
        raise ValueError('contrast role mismatch')
    return dict({k:before[k] for k in ('scene','pair','condition','query','key','mass','n_query','n_key',
                                      'label_sha256','target_sha256')},contrast=contrast['name'],
        from_role=contrast['from_role'],to_role=contrast['to_role'],
        **{m+'_delta_sum':after[m+'_sum']-before[m+'_sum'] for m in METRICS},
        center=paired_center(winner_before,winner_after,bounds,point,valid))


def interactions(contrasts):
    """Content-minus-zero differences of three fixed geometric contrasts."""
    result={}
    def regional(a,b):
        out={k+'_delta':b[k+'_delta']-a[k+'_delta'] for k in METRICS}
        if 'mass' in a:
            if a['mass']!=b['mass']:raise ValueError('interaction regional population changed')
            out['mass']=a['mass']
        return out
    def centers(a,b):
        out=dict(pck_delta=fraction_dict(fraction(b['pck_delta'])-fraction(a['pck_delta'])),
                 mean_distance_delta=b['mean_distance_delta']-a['mean_distance_delta'],
                 mean_squared_distance_delta=b['mean_squared_distance_delta']-a['mean_squared_distance_delta'],
                 rmse_delta=b['rmse_delta']-a['rmse_delta'])
        if 'n_point' in a:
            if a['n_point']!=b['n_point']:raise ValueError('interaction center population changed')
            out.update(n_point=a['n_point'],pck_delta_count=(b['to_correct']-b['from_correct'])-(a['to_correct']-a['from_correct']),
                       distance_delta_sum=b['distance_delta_sum']-a['distance_delta_sum'],
                       squared_distance_delta_sum=b['squared_distance_delta_sum']-a['squared_distance_delta_sum'])
        return out
    def levels(a,b,fn):
        return dict(by_scene_condition={s:{c['condition']:fn(a['by_scene_condition'][s][c['condition']],
                            b['by_scene_condition'][s][c['condition']]) for c in CONDITIONS} for s in SCENES},
            by_condition={c['condition']:fn(a['by_condition'][c['condition']],b['by_condition'][c['condition']]) for c in CONDITIONS},
            by_scene={s:fn(a['by_scene'][s],b['by_scene'][s]) for s in SCENES},macro=fn(a['macro'],b['macro']))
    for name in ('anchor','widening','shape'):
        a,b=contrasts[name+'_zero'],contrasts[name+'_content']
        result[name]=dict(zero_contrast=name+'_zero',content_contrast=name+'_content',
            **levels(a,b,regional),center=levels(a['center'],b['center'],centers))
    return result
