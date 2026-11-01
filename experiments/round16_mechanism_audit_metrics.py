"""Independent NumPy mathematics for the Round16 saved-probability diagnostic.

No new producer/analytics, forward operator, file, source loader or selector is
imported. The source-owning auditor supplies the already-frozen shared support
and content preparation and verifies every original Stage A probability hash.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math
from numbers import Integral, Real

import numpy as np


SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
FAMILIES = ('quartic_full', 'quartic_point', 'gaussian_point')
QUERIES = ('g8', 'g16', 'g24')
QUERY_COUNTS = dict(g8=432, g16=108, g24=48)
GRID = tuple((h,t) for h in (.25,.5,1.,2.) for t in (0.,5.,10.))
INT_FIELDS = ('winner','zero_winner','j2','nearest_mu','tie_count','final_tie_count')
BOOL_FIELDS = ('flip','valid','isotropic','mu_in_fov')
CORRECT_FIELDS = ('correct','zero_correct','nearest_correct')
NUMERIC_FIELDS = ('dist_mu_px','zero_dist_mu_px','nearest_dist_mu_px','mu_target_dist_px',
    'winner_to_zero_dx_px','winner_to_zero_dy_px','winner_to_zero_dist_px',
    'top_geometry_gap','geom_penalty','content_gain','advantage','identity_residual',
    'identity_error','fov_distance_px','S_condition')
EXTRA_FLOAT_FIELDS = ('mu_phase_x','mu_phase_y','S_major_angle_rad')
ARRAY_FIELDS = INT_FIELDS+BOOL_FIELDS+CORRECT_FIELDS+NUMERIC_FIELDS+EXTRA_FLOAT_FIELDS
PAIRED_NUMERIC_FIELDS = ('dist_mu_px_difference','zero_dist_mu_px_difference','top_geometry_gap_difference')


def _integer(value, name, minimum=0):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<minimum:
        raise ValueError('invalid integer '+name)
    return int(value)


def _finite(value, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not math.isfinite(value):
        raise ValueError('invalid finite number '+name)
    return float(value)


def _probability(value, shape=None):
    x=np.asarray(value)
    if (x.dtype!=np.float64 or x.ndim!=2 or min(x.shape)<1 or x.shape[1]<2
            or (shape is not None and x.shape!=shape) or not np.isfinite(x).all()
            or np.any(x<=0) or np.any(np.abs(x.sum(-1)-1)>1e-12)):
        raise ValueError('expected positive normalized float64 probability')
    return x


def analyze(p0,pT,content32,mu_px,cov_reference8,key_px,target_px,valid,*,h,temperature):
    """Per-query diagnostics; all distances are pixels, covariance reference8.

    p0/pT are retained final distributions. Their log odds recover the floored
    positional mass difference without recomputing an integral. Temperature
    multiplication is float32 BEFORE double conversion, matching the frozen
    old content path; no probability is clamped to invent a finite log margin.
    """
    p0=_probability(p0);pT=_probability(pT,p0.shape);n,k=p0.shape
    content=np.asarray(content32)
    mu=np.asarray(mu_px,np.float64);cov=np.asarray(cov_reference8,np.float64)
    keys=np.asarray(key_px,np.float64);target=np.asarray(target_px,np.float64)
    visible=np.asarray(valid)
    if content.dtype!=np.float32 or content.shape!=p0.shape or not np.isfinite(content).all():
        raise ValueError('content must be the shared float32 matrix')
    if (mu.shape!=(n,2) or cov.shape!=(n,2,2) or keys.shape!=(k,2) or target.shape!=(n,2)
            or visible.shape!=(n,) or visible.dtype!=np.bool_
            or any(not np.isfinite(v).all() for v in (mu,cov,keys))
            or not np.isfinite(target[visible]).all()):
        raise ValueError('wrong shared geometry/center population')
    width=_finite(h,'h');temp=_finite(temperature,'temperature')
    if width not in (.25,.5,1.,2.) or temp not in (0.,5.,10.):
        raise ValueError('configuration outside frozen grid')
    symmetry_tolerance=1e-12*np.maximum(1.,np.max(np.abs(cov),axis=(1,2)))
    if np.any(np.abs(cov-cov.transpose(0,2,1))>symmetry_tolerance[:,None,None]):
        raise ValueError('query covariance must be symmetric')
    axes=[np.unique(keys[:,dimension]) for dimension in range(2)]
    if (len(axes[0])*len(axes[1])!=k or len(np.unique(keys,axis=0))!=k
            or any(np.any(np.diff(axis)!=8.) for axis in axes)):
        raise ValueError('key centers must form a complete native 8-pixel Cartesian grid')
    S=cov+width**2*np.eye(2)
    eigenvectors=[];eigenvalues=[]
    for matrix in S:
        values,vectors=np.linalg.eigh(matrix)
        if values[0]<=0:raise ValueError('effective covariance must be positive definite')
        eigenvalues.append(values);eigenvectors.append(vectors[:,1])
    eig=np.asarray(eigenvalues);direction=np.asarray(eigenvectors)
    isotropic=eig[:,0]==eig[:,1]
    angle=np.mod(np.arctan2(direction[:,1],direction[:,0]),np.pi)
    angle=np.where(isotropic | (angle>=np.pi),0.,angle)
    j0=np.argmax(p0,axis=1);winner=np.argmax(pT,axis=1);row=np.arange(n)
    competitors=p0.copy();competitors[row,j0]=-np.inf
    j2=np.argmax(competitors,axis=1)
    squared=np.sum((mu[:,None,:]-keys[None,:,:])**2,axis=2)
    nearest=np.argmin(squared,axis=1)
    log0=np.log(p0);logT=np.log(pT)
    # Each float32 product is rounded before the two selected values differ.
    bias=(np.float32(temp)*content).astype(np.float64)
    penalty=log0[row,j0]-log0[row,winner]
    gain=bias[row,winner]-bias[row,j0]
    advantage=logT[row,winner]-logT[row,j0]
    residual=advantage-(gain-penalty)
    lower=keys.min(axis=0)-4.;upper=keys.max(axis=0)+4.
    displacement=keys[winner]-keys[j0]
    target_distance=np.full(n,np.nan,np.float64)
    target_distance[visible]=np.sqrt(np.sum((mu[visible]-target[visible])**2,axis=1))
    correct={}
    for name,indices in (('correct',winner),('zero_correct',j0),('nearest_correct',nearest)):
        value=np.full(n,-1,np.int8)
        value[visible]=(np.sum((keys[indices[visible]]-target[visible])**2,axis=1)<=64).astype(np.int8)
        correct[name]=value
    result=dict(winner=winner.astype(np.int64),zero_winner=j0.astype(np.int64),j2=j2.astype(np.int64),
        nearest_mu=nearest.astype(np.int64),
        tie_count=(p0==p0[row,j0,None]).sum(-1).astype(np.int64),
        final_tie_count=(pT==pT[row,winner,None]).sum(-1).astype(np.int64),
        flip=winner!=j0,valid=visible.copy(),isotropic=isotropic,mu_in_fov=((mu>=lower)&(mu<upper)).all(-1),
        **correct,mu_target_dist_px=target_distance,
        dist_mu_px=np.sqrt(squared[row,winner]),zero_dist_mu_px=np.sqrt(squared[row,j0]),
        nearest_dist_mu_px=np.sqrt(squared[row,nearest]),
        winner_to_zero_dx_px=displacement[:,0],winner_to_zero_dy_px=displacement[:,1],
        winner_to_zero_dist_px=np.sqrt(np.sum(displacement**2,axis=1)),
        top_geometry_gap=log0[row,j0]-log0[row,j2],geom_penalty=penalty,content_gain=gain,
        advantage=advantage,identity_residual=residual,identity_error=np.abs(residual),
        mu_phase_x=np.mod(mu[:,0]-lower[0],8.)/8.,mu_phase_y=np.mod(mu[:,1]-lower[1],8.)/8.,
        fov_distance_px=np.sqrt(np.sum(np.maximum(np.maximum(lower-mu,mu-upper),0.)**2,axis=1)),
        S_condition=eig[:,1]/eig[:,0],S_major_angle_rad=angle)
    validate_arrays(result)
    return result


def validate_arrays(arrays):
    """Check complete per-query schema without requiring probability access."""
    if not isinstance(arrays,dict) or set(arrays)!=set(ARRAY_FIELDS):
        raise ValueError('wrong complete diagnostic array fields')
    values={name:np.asarray(value) for name,value in arrays.items()}
    n=len(values['valid'])
    if n<1 or any(value.shape!=(n,) for value in values.values()):
        raise ValueError('wrong one-dimensional query population')
    if any(values[f].dtype!=np.int64 for f in INT_FIELDS):raise ValueError('integer diagnostics require int64')
    if any(values[f].dtype!=np.bool_ for f in BOOL_FIELDS):raise ValueError('boolean diagnostics require bool')
    if any(values[f].dtype!=np.int8 for f in CORRECT_FIELDS):raise ValueError('correctness diagnostics require int8')
    if any(values[f].dtype!=np.float64 for f in (*NUMERIC_FIELDS,*EXTRA_FLOAT_FIELDS)):
        raise ValueError('real diagnostics require float64')
    if any(np.any(values[f]<0) for f in INT_FIELDS[:4]) or any(np.any(values[f]<1) for f in INT_FIELDS[4:]):
        raise ValueError('invalid native index/tie count')
    valid=values['valid']
    for field in CORRECT_FIELDS:
        if not np.all(np.isin(values[field][valid],[0,1])) or np.any(values[field][~valid]!=-1):
            raise ValueError('correctness must preserve invalid-center sentinel')
    for field in (*NUMERIC_FIELDS,*EXTRA_FLOAT_FIELDS):
        x=values[field]
        if field=='mu_target_dist_px':
            if not np.isfinite(x[valid]).all() or not np.isnan(x[~valid]).all():
                raise ValueError('target distance validity changed')
        elif not np.isfinite(x).all():raise ValueError('nonfinite query diagnostic')
    if (np.any(values['flip']!=(values['winner']!=values['zero_winner']))
            or not np.array_equal(values['identity_error'],np.abs(values['identity_residual']))
            or np.any(values['S_condition']<1)
            or any(np.any((values[f]<0)|(values[f]>=1)) for f in ('mu_phase_x','mu_phase_y'))
            or np.any((values['S_major_angle_rad']<0)|(values['S_major_angle_rad']>=np.pi))
            or np.any(values['S_major_angle_rad'][values['isotropic']]!=0)):
        raise ValueError('inconsistent query diagnostic invariant')
    return n


def _moments(values, mask):
    selected=np.asarray(values,np.float64)[mask]
    return len(selected),math.fsum(map(float,selected)),math.fsum(float(x)*float(x) for x in selected)


def stats(arrays):
    """Additive full-query and original-valid-center populations, separately."""
    n=validate_arrays(arrays);output={}
    visible=arrays['valid'];correct=arrays['correct']==1;zero=arrays['zero_correct']==1
    nearest=arrays['nearest_correct']==1
    for name,mask in (('all',np.ones(n,dtype=bool)),('valid',visible)):
        truth=mask & visible
        conditions=dict(n_valid=truth,n_flip=mask & arrays['flip'],
            n_geometry_tie=mask & (arrays['tie_count']>1),n_final_tie=mask & (arrays['final_tie_count']>1),
            n_isotropic=mask & arrays['isotropic'],n_mu_in_fov=mask & arrays['mu_in_fov'],
            n_correct=truth & correct,n_zero_correct=truth & zero,n_nearest_correct=truth & nearest,
            n_both_correct=truth & correct & zero,n_both_wrong=truth & ~correct & ~zero,
            n_harm=truth & zero & ~correct,n_benefit=truth & ~zero & correct,
            n_nearest_harm=truth & nearest & ~correct,n_nearest_benefit=truth & ~nearest & correct)
        value={'n':int(mask.sum()),**{field:int(condition.sum()) for field,condition in conditions.items()}}
        for field in NUMERIC_FIELDS:
            population=truth if field=='mu_target_dist_px' else mask
            count,total,squared=_moments(arrays[field],population)
            value.update({field+'_count':count,field+'_sum':total,field+'_sum_sq':squared})
        output[name]=value
    return output


def paired_stats(full,point):
    n=validate_arrays(full)
    if validate_arrays(point)!=n:raise ValueError('paired query populations differ')
    for field in ('valid','nearest_mu','nearest_correct','nearest_dist_mu_px','mu_target_dist_px',
                  'mu_in_fov','mu_phase_x','mu_phase_y','fov_distance_px','isotropic','S_condition','S_major_angle_rad'):
        if not np.array_equal(full[field],point[field],equal_nan=True):
            raise ValueError('paired shared input changed: '+field)
    output={};visible=full['valid'];fc=full['correct']==1;pc=point['correct']==1
    for name,mask in (('all',np.ones(n,dtype=bool)),('valid',visible)):
        truth=mask & visible;ff=full['flip'];pf=point['flip']
        conditions=dict(n_valid=truth,n_same_winner=mask & (full['winner']==point['winner']),
            n_same_zero_winner=mask & (full['zero_winner']==point['zero_winner']),
            n_full_flip=mask & ff,n_point_flip=mask & pf,
            n_full_flip_point_not=mask & ff & ~pf,n_point_flip_full_not=mask & pf & ~ff,
            n_full_correct=truth & fc,n_point_correct=truth & pc,n_both_correct=truth & fc & pc,
            n_both_wrong=truth & ~fc & ~pc,n_full_correct_point_wrong=truth & fc & ~pc,
            n_full_wrong_point_correct=truth & ~fc & pc)
        value={'n':int(mask.sum()),**{field:int(condition.sum()) for field,condition in conditions.items()}}
        for name2 in PAIRED_NUMERIC_FIELDS:
            field=name2.removesuffix('_difference')
            count,total,squared=_moments(full[field]-point[field],mask)
            value.update({name2+'_count':count,name2+'_sum':total,name2+'_sum_sq':squared})
        output[name]=value
    return output


ORDINARY_RATES = {
    'n_flip':'flip_rate','n_geometry_tie':'geometry_tie_rate','n_final_tie':'final_tie_rate',
    'n_isotropic':'isotropic_rate','n_mu_in_fov':'mu_in_fov_rate','n_same_winner':'same_winner_rate',
    'n_same_zero_winner':'same_zero_winner_rate','n_full_flip':'full_flip_rate','n_point_flip':'point_flip_rate',
    'n_full_flip_point_not':'full_only_flip_rate','n_point_flip_full_not':'point_only_flip_rate'}
LABELED_RATES = {
    'n_correct':'pck8','n_zero_correct':'zero_pck8','n_nearest_correct':'nearest_pck8',
    'n_both_correct':'both_correct_rate','n_both_wrong':'both_wrong_rate','n_harm':'harm_rate',
    'n_benefit':'benefit_rate','n_nearest_harm':'nearest_harm_rate','n_nearest_benefit':'nearest_benefit_rate',
    'n_full_correct':'full_pck8','n_point_correct':'point_pck8',
    'n_full_correct_point_wrong':'full_benefit_rate','n_full_wrong_point_correct':'full_harm_rate'}
ORDINARY_COUNTS = ('n','n_valid','n_flip','n_geometry_tie','n_final_tie','n_isotropic','n_mu_in_fov',
    'n_correct','n_zero_correct','n_nearest_correct','n_both_correct','n_both_wrong','n_harm','n_benefit',
    'n_nearest_harm','n_nearest_benefit')
PAIRED_COUNTS = ('n','n_valid','n_same_winner','n_same_zero_winner','n_full_flip','n_point_flip',
    'n_full_flip_point_not','n_point_flip_full_not','n_full_correct','n_point_correct','n_both_correct',
    'n_both_wrong','n_full_correct_point_wrong','n_full_wrong_point_correct')


def validate_stats(value):
    """Validate additive denominators and exact correctness transition identities."""
    if not isinstance(value,dict) or set(value)!={'all','valid'}:
        raise ValueError('both all and valid reporting populations required')
    ordinary='n_correct' in value['all']
    count_fields=ORDINARY_COUNTS if ordinary else PAIRED_COUNTS
    numeric=NUMERIC_FIELDS if ordinary else PAIRED_NUMERIC_FIELDS
    expected=set(count_fields)|{f+suffix for f in numeric for suffix in ('_count','_sum','_sum_sq')}
    for population,a in value.items():
        if set(a)!=expected:raise ValueError('incomplete or unexpected additive schema')
        for field in count_fields:
            count=_integer(a[field],field)
            denominator=a['n_valid'] if field in LABELED_RATES else a['n']
            if field not in ('n','n_valid') and count>denominator:
                raise ValueError('count exceeds reporting denominator')
        if a['n_valid']>a['n'] or (population=='valid' and a['n_valid']!=a['n']):
            raise ValueError('invalid valid-center denominator')
        if ordinary:
            if (a['n_both_correct']+a['n_both_wrong']+a['n_harm']+a['n_benefit']!=a['n_valid']
                    or a['n_correct']!=a['n_both_correct']+a['n_benefit']
                    or a['n_zero_correct']!=a['n_both_correct']+a['n_harm']
                    or a['n_nearest_correct']-a['n_correct']!=a['n_nearest_harm']-a['n_nearest_benefit']):
                raise ValueError('incorrect harm/benefit count closure')
        elif (a['n_both_correct']+a['n_both_wrong']+a['n_full_correct_point_wrong']+a['n_full_wrong_point_correct']!=a['n_valid']
                or a['n_full_correct']!=a['n_both_correct']+a['n_full_correct_point_wrong']
                or a['n_point_correct']!=a['n_both_correct']+a['n_full_wrong_point_correct']
                or a['n_full_flip']-a['n_point_flip']!=a['n_full_flip_point_not']-a['n_point_flip_full_not']):
            raise ValueError('incorrect paired count closure')
        for field in numeric:
            count=_integer(a[field+'_count'],field+'_count')
            expected_count=a['n_valid'] if field=='mu_target_dist_px' else a['n']
            total=_finite(a[field+'_sum'],field+'_sum');square=_finite(a[field+'_sum_sq'],field+'_sum_sq')
            if count!=expected_count or square<0 or (count==0 and (total!=0 or square!=0)):
                raise ValueError('numeric population or empty moments changed')
    for field in ('n_valid',*(f for f in LABELED_RATES if f in value['all'])):
        if value['all'][field]!=value['valid'][field]:
            raise ValueError('all/valid correctness populations disagree')
    return ordinary


def _derive(a):
    result={}
    for source,destination in (*ORDINARY_RATES.items(),*LABELED_RATES.items()):
        if source in a:
            denominator=a['n_valid'] if source in LABELED_RATES else a['n']
            result[destination]=float(Fraction(a[source],denominator)) if denominator else None
    if 'n_correct' in a:
        result['net_pck_change']=(float(Fraction(a['n_benefit']-a['n_harm'],a['n_valid']))
                                  if a['n_valid'] else None)
    else:
        result['full_minus_point_pck']=(float(Fraction(a['n_full_correct_point_wrong']-a['n_full_wrong_point_correct'],a['n_valid']))
                                        if a['n_valid'] else None)
    for name in sorted(k[:-6] for k in a if k.endswith('_count')):
        count=a[name+'_count']
        average=a[name+'_sum']/count if count else None
        variance=max(0.,a[name+'_sum_sq']/count-average*average) if count else None
        result[name+'_mean']=average
        result[name+'_std']=math.sqrt(variance) if variance is not None else None
    return result


def _pair_key(value):
    if isinstance(value,str) and value:return value
    if isinstance(value,(list,tuple)) and len(value)==2:
        return tuple(_integer(v,'frame index') for v in value)
    raise ValueError('pair must be a nonempty identifier or two frame indices')


def _identity(row):
    grid=_integer(row['grid_index'],'grid index')
    if grid>=12 or row['query'] not in QUERIES or row['scene'] not in SCENES:
        raise ValueError('row outside declared training grid/scenes/scales')
    if not isinstance(row['family'],str) or not row['family']:
        raise ValueError('invalid family or paired comparison identity')
    return row['scene'],_pair_key(row['pair']),row['family'],grid,row['query']


def verify_population(rows,expected_families=FAMILIES):
    """Explicit complete 48-pair population check for the source-owning auditor.

    The generic reducer also supports paired comparison labels. Call this
    separately with their fixed label tuple; a missing scene is never silently
    accepted as a complete diagnostic experiment.
    """
    families=tuple(expected_families)
    if not families or len(set(families))!=len(families):raise ValueError('invalid expected families')
    ids=[];pairs=defaultdict(set);populations={}
    for row in rows:
        identity=_identity(row);scene,pair,family,_,query=identity
        if family not in families:raise ValueError('undeclared diagnostic family')
        validate_stats(row['stats']);a=row['stats']['all']
        if a['n']!=QUERY_COUNTS[query]:raise ValueError('native query count changed')
        key=scene,pair,query
        if populations.setdefault(key,a['n_valid'])!=a['n_valid']:
            raise ValueError('configuration changed original valid-center population')
        ids.append(identity);pairs[scene].add(pair)
    if (len(ids)!=48*len(families)*12*3 or len(set(ids))!=len(ids)
            or set(pairs)!=set(SCENES) or any(len(pairs[s])!=16 for s in SCENES)):
        raise ValueError('incomplete or duplicate full diagnostic population')
    expected={(s,p,f,i,q) for s in SCENES for p in pairs[s] for f in families for i in range(12) for q in QUERIES}
    if set(ids)!=expected:raise ValueError('configuration/scale pair cohorts differ')
    return dict(pairs=48,rows=len(ids),query_states=sum(row['stats']['all']['n'] for row in rows),
                unique_queries=48*sum(QUERY_COUNTS.values()),families=list(families))


def summarize(rows):
    """Sum pairs within each scene then macro-average scene statistics.

    Source/cohort completeness is checked by ``verify_population`` or the
    source-owning auditor. No scale/configuration averaging or selection is
    provided. Empty denominators yield None, never silent scene exclusion.
    """
    groups=defaultdict(list);seen=set()
    for row in rows:
        identity=_identity(row)
        if identity in seen:raise ValueError('duplicate diagnostic row')
        seen.add(identity);validate_stats(row['stats'])
        scene,_,family,index,query=identity
        groups[family,index,query,scene].append(row['stats'])
    if not groups:raise ValueError('empty diagnostic summary population')
    result={}
    for family,index,query in sorted({key[:3] for key in groups}):
        scenes=sorted(s for f,i,q,s in groups if (f,i,q)==(family,index,query))
        detail={}
        for scene in scenes:
            items=groups[family,index,query,scene];detail[scene]={}
            for pop in ('all','valid'):
                fields=set(items[0][pop])
                if any(set(v[pop])!=fields for v in items):raise ValueError('mixed additive schemas')
                merged={}
                for field in sorted(fields):
                    values=[v[pop][field] for v in items]
                    merged[field]=sum(values) if field.startswith('n') or field.endswith('_count') else math.fsum(values)
                detail[scene][pop]=dict(additive=merged,derived=_derive(merged))
        macros={}
        for pop in ('all','valid'):
            names=set(detail[scenes[0]][pop]['derived'])
            if any(set(detail[s][pop]['derived'])!=names for s in scenes):raise ValueError('mixed derived schemas')
            macros[pop]={}
            for name in sorted(names):
                values=[detail[s][pop]['derived'][name] for s in scenes]
                macros[pop][name]=None if any(v is None for v in values) else math.fsum(values)/len(values)
        result.setdefault(family,{}).setdefault(str(index),{})[query]=dict(by_scene=detail,macro=macros)
    return result


def compare_arrays(actual,expected,*,atol=1e-9,rtol=1e-12):
    """Schema plus exact discrete fields; finite floats use declared tolerance."""
    if validate_arrays(actual)!=validate_arrays(expected):raise ValueError('array population changed')
    maximum=0.
    for field in ARRAY_FIELDS:
        a,b=actual[field],expected[field]
        if field in (*INT_FIELDS,*BOOL_FIELDS,*CORRECT_FIELDS):
            if not np.array_equal(a,b):raise ValueError('discrete array differs: '+field)
        else:
            if not np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True):
                raise ValueError('real array differs: '+field)
            finite=np.isfinite(a)&np.isfinite(b)
            maximum=max(maximum,float(np.max(np.abs(a[finite]-b[finite]),initial=0.)))
    return maximum
