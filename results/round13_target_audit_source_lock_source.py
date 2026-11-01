"""Independent target-compatibility audit: scalar capped-mass convex projection.

Does not import the Round13 producer or its sorted-pooling KL helper. A sensor
reference is reused from the already frozen independent Round12 audit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import brentq
from experiments import round12_stress_result_audit as S

ROOT=Path(__file__).resolve().parents[1]
SCENES=('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
STRIDES=(8,16,24)
SOURCE_NAMES=('experiments/round13_target_audit.py','tests/test_round13_target_audit.py',
    'experiments/round12_stress_result_audit.py','experiments/round6_region_data.py',
    'experiments/round5_rgbd_data.py')
EQUAL_FIELDS=('native_first_argmax_pck_joint','optimistic_top_tie_pck_joint','strict_conflict_fraction_joint',
    'min_kl_mean_joint','min_kl_mass_weighted_joint','min_kl_mean_all_query_unconstrained_zero',
    'min_kl_mass_weighted_all_area_unconstrained_zero')


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def centers(stride):
    x=np.arange(stride/2-.5,192,stride);y=np.arange(stride/2-.5,144,stride)
    xx,yy=np.meshgrid(x,y)
    return np.column_stack([xx.ravel(),yy.ravel()])


def cap_projection(g,index):
    """Independent scalar root from normalization under p_i=min(g_i,t).

    KKT yields a common top cap t and multiplier one. A selected candidate has
    p_j=t; every other coordinate below t stays at g_i. Normalization gives
    t + sum_{i!=j} min(g_i,t) = 1, solved without sorting or active-pool scans.
    """
    g=np.asarray(g,np.float64)
    if g.ndim!=1 or not len(g) or np.any(g<0) or not np.isfinite(g).all() or abs(g.sum()-1)>1e-12:
        raise ValueError('expected normalized finite histogram')
    if not 0<=index<len(g):raise ValueError('candidate index')
    if g[index]==g.max():return g.copy(),0.,1
    others=np.delete(g,index)
    level=brentq(lambda t:t+np.minimum(others,t).sum()-1.,0.,1.,xtol=5e-16,rtol=1e-14)
    p=np.minimum(g,level);p[index]=level
    if abs(p.sum()-1)>2e-13 or p[index]<p.max():raise ValueError('infeasible cap projection')
    positive=g>0
    value=max(0.,float(np.sum(g[positive]*np.log(g[positive]/p[positive]))))
    # Count only strict active inequalities, excluding rational exact ties.
    active=1+int((others>level+2e-14).sum())
    return p,value,active


def min_allowed(g,indices):
    results=[cap_projection(g,int(j)) for j in indices]
    # Native-index tie convention: largest original allowed g, first index.
    chosen=int(np.argmax(np.asarray(g)[indices]));value=results[chosen][1]
    if abs(value-min(r[1] for r in results))>1e-11:raise ValueError('largest-allowed candidate is not optimal')
    return value,int(indices[chosen]),results[chosen][2],len(results)


def active_count_interval(g,index):
    """A coordinate exactly at the cap can be counted in either active set.

    This changes no probability or KL. The original floating pooling can include
    such a tied coordinate after a one-ulp mean roundoff, so audit the admissible
    interval instead of falsely requiring one unique active-set cardinality.
    """
    p,_,_=cap_projection(g,index);others=np.delete(g,index);level=p[index]
    return 1+int((others>level+2e-14).sum()),1+int((others>=level-2e-14).sum())


def histogram(xy,visible,stride):
    xy=np.asarray(xy);visible=np.asarray(visible)
    projected=xy[visible]
    if not np.isfinite(projected).all() or np.any(projected<[-.5,-.5]) or np.any(projected>=[191.5,143.5]):
        raise ValueError('invalid visible sensor projection')
    key=np.floor((projected+.5+1e-9)/8).astype(int);key=np.clip(key,[0,0],[23,17]);key=key[:,1]*24+key[:,0]
    source=np.flatnonzero(visible);q=(source//192//stride)*(192//stride)+(source%192//stride)
    count=np.bincount(q*432+key,minlength=(144//stride)*(192//stride)*432).reshape(-1,432)
    if int(count.sum())!=int(visible.sum()):raise ValueError('source mass changed')
    return count


def row_reference(count,center,visible,scene,pair,stride,qi):
    mass=int(count.sum());joint=bool(mass>0 and visible)
    row=dict(scene=scene,pair=pair,stride=stride,query=qi,visible_source_pixels=mass,
        area_valid=bool(mass>0),center_valid=bool(visible),joint_valid=joint,
        native_first_argmax_correct=None,optimistic_top_tie_correct=None,strict_conflict=None,
        min_kl=None,allowed_key_indices=None,allowed_key_count=None,
        global_max_bin_mass=None,best_allowed_bin_mass=None,allowed_set_mass=None,
        max_mass_gap=None,allowed_to_global_max_ratio=None,first_argmax_key=None,
        first_argmax_distance=None,top_tie_count=None,best_allowed_key=None,pooled_active_count=None)
    if not joint:return row,0
    if not np.isfinite(center).all():raise ValueError('invalid visible center')
    distance=np.sqrt(np.sum((centers(8)-center)**2,axis=1));allowed=np.flatnonzero(distance<=8)
    if not len(allowed):raise ValueError('valid center has no allowed key')
    g=count/mass;cost,best,active,projections=min_allowed(g,allowed)
    first=int(np.argmax(count));maximum=int(count.max());best_count=int(count[allowed].max())
    row.update(native_first_argmax_correct=bool(first in allowed),optimistic_top_tie_correct=bool(best_count==maximum),
        strict_conflict=bool(best_count<maximum),min_kl=cost,allowed_key_indices=allowed.tolist(),allowed_key_count=len(allowed),
        global_max_bin_mass=maximum/mass,best_allowed_bin_mass=best_count/mass,allowed_set_mass=float(g[allowed].sum()),
        max_mass_gap=(maximum-best_count)/mass,allowed_to_global_max_ratio=best_count/maximum,
        first_argmax_key=first,first_argmax_distance=float(distance[first]),top_tie_count=int(np.count_nonzero(count==maximum)),
        best_allowed_key=best,pooled_active_count=active)
    return row,projections


def compare(actual,expected,tolerance=1e-10):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('result fields differ')
        return max([compare(actual[k],v,tolerance) for k,v in expected.items()]+[0.])
    if isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError('list population differs')
        return max([compare(a,b,tolerance) for a,b in zip(actual,expected)]+[0.])
    if expected is None or isinstance(expected,(bool,str)):
        if actual!=expected:raise ValueError('categorical result differs')
        return 0.
    difference=float(abs(actual-expected))
    if not np.isfinite(difference) or difference>tolerance:raise ValueError('numeric result differs: '+str(difference))
    return difference


def quantiles(values):
    if not len(values):return None
    levels=(0.,.25,.5,.75,.9,.95,.99,1.)
    return dict(zip(map(str,levels),map(float,np.quantile(values,levels))))


def summary_reference(rows):
    n=len(rows);area=np.array([r['area_valid'] for r in rows],bool);center=np.array([r['center_valid'] for r in rows],bool)
    joint=area&center;mass=np.array([r['visible_source_pixels'] for r in rows],np.int64)
    values=np.array([r['min_kl'] if r['joint_valid'] else 0. for r in rows],float)
    selected=[r for r in rows if r['joint_valid']]
    if not len(selected):raise ValueError('empty joint population')
    strict=np.array([bool(r['strict_conflict']) for r in rows]);first=sum(r['native_first_argmax_correct'] for r in selected)
    optimistic=sum(r['optimistic_top_tie_correct'] for r in selected)
    output=dict(total_queries=n,area_valid_queries=int(area.sum()),center_valid_queries=int(center.sum()),
        joint_valid_queries=int(joint.sum()),area_valid_center_invalid=int((area&~center).sum()),
        center_valid_area_invalid=int((center&~area).sum()),neither_valid=int((~center&~area).sum()),
        all_visible_source_pixels=int(mass.sum()),joint_visible_source_pixels=int(mass[joint].sum()),
        native_first_argmax_pck_joint=first/len(selected),optimistic_top_tie_pck_joint=optimistic/len(selected),
        native_first_conflict_fraction_joint=1-first/len(selected),strict_conflict_fraction_joint=float(strict.sum()/joint.sum()),
        strict_conflict_fraction_all_query=float(strict.sum()/n),top_tie_only_conflicts=optimistic-first,
        min_kl_mean_joint=float(values[joint].mean()),min_kl_mass_weighted_joint=float(np.average(values[joint],weights=mass[joint])),
        min_kl_mean_all_query_unconstrained_zero=float(values.mean()),
        min_kl_mean_area_valid_unconstrained_zero=float(values[area].mean()),
        min_kl_mass_weighted_all_area_unconstrained_zero=float(np.average(values,weights=mass)),
        min_kl_quantiles_joint=quantiles(values[joint]),min_kl_quantiles_strict=quantiles(values[strict]),
        min_kl_quantiles_all_query_unconstrained_zero=quantiles(values))
    for key in ('global_max_bin_mass','best_allowed_bin_mass','allowed_set_mass','allowed_to_global_max_ratio','max_mass_gap'):
        output[key+'_quantiles']=quantiles([r[key] for r in selected])
    sizes=np.array([r['allowed_key_count'] for r in selected]);counts=np.bincount(sizes)
    output['allowed_count_histogram']={str(i):int(counts[i]) for i in np.flatnonzero(counts)}
    return output


def sensor_replay(meta,saved,original):
    from experiments.round6_region_data import read_scene_selection
    selection=Path(meta['source_selection']);row=next(r for r in S.read(selection)['scenes'] if r['scene']==meta['scene'])
    frames=read_scene_selection(selection.parent,row);q,k=[frames[i] for i in meta['pair']]
    if set(meta['sensor_sources'])!={str(q.depth_path),str(k.depth_path)}:raise ValueError('wrong sensor timestamp association')
    S.verify_files(meta['sensor_sources']);qd,kd=S.decode_sensor(q.depth_path),S.decode_sensor(k.depth_path)
    yy,xx=np.mgrid[:144,:192];pixels=np.column_stack([xx.ravel(),yy.ravel()])
    xy,visible,status=S.center_reference(pixels,qd,kd,q.K_native,k.K_native,q.c2w,k.c2w)
    np.testing.assert_allclose(xy,original['gt_xy'],atol=1e-10,rtol=0,equal_nan=True)
    np.testing.assert_array_equal(visible,original['gt_visible'])
    if 'gt_status' in original:np.testing.assert_array_equal(status,original['gt_status'])
    count=0
    for stride in STRIDES:
        c,v,s=S.center_reference(centers(stride),qd,kd,q.K_native,k.K_native,q.c2w,k.c2w)
        np.testing.assert_allclose(c,saved[f'g{stride}_point_xy'],atol=1e-10,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(v,saved[f'g{stride}_point_visible']);np.testing.assert_array_equal(s,saved[f'g{stride}_point_status'])
        np.testing.assert_array_equal(histogram(xy,visible,stride),saved[f'g{stride}_counts']);count+=len(c)
    return dict(scene=meta['scene'],pair=meta['pair'],dense_sensor_points=len(pixels),center_sensor_points=count,
        sensor_files=meta['sensor_sources'])


def freeze(source_lock,runs):
    if source_lock.exists():raise ValueError('refuse overwrite source lock')
    source_lock.parent.mkdir(parents=True,exist_ok=True)
    snapshot=source_lock.with_name(source_lock.stem+'_source.py')
    if snapshot.exists():raise ValueError('refuse overwrite source snapshot')
    snapshot.write_bytes(Path(__file__).read_bytes())
    write(source_lock,dict(created_unix=time.time(),source_hashes={n:S.sha(ROOT/n) for n in SOURCE_NAMES},
        source_snapshot=str(snapshot),source_snapshot_sha256=S.sha(snapshot),target_lock_sha256=S.sha(runs/'lock.json'),
        stage='after producer completion, before auditor opens rows/targets/summary',
        prior_information='Theory message disclosed approximate q24 conflict 29.72% and all-mass cost 2.2194e-4; audit algorithm and scope already sent to root before that message.',
        scope='all28224 rows and all allowed scalar-root projections; all48 histograms from frozen dense GT; first metadata pair per3 train scenes raw dense and8/16/24 center replay',
        probability_input='sensor histogram only',cost_tolerance=1e-10,summary_tolerance=1e-10,
        active_set_ties='Only auxiliary pooled_active_count may include coordinates exactly at cap (2e-14); probabilities/KL and all decisions retain original tolerances.',
        no_method_claim=True,no_gate_exemption=True))


def audit(runs,out,source_lock):
    if out.exists():raise ValueError('refuse overwrite audit')
    own=S.read(source_lock);S.verify_files(own['source_hashes'],ROOT)
    if S.sha(own['source_snapshot'])!=own['source_snapshot_sha256'] or own['source_snapshot_sha256']!=S.sha(__file__):raise ValueError('audit snapshot changed')
    lock=S.read(runs/'lock.json');complete=S.read(runs/'completion.json')
    if S.sha(runs/'lock.json')!=own['target_lock_sha256'] or S.sha(runs/'lock.json')!=(runs/'lock.sha256').read_text().strip():raise ValueError('changed target lock')
    S.verify_files(lock['source_hashes'],ROOT);S.verify_files(lock['source_metadata']);S.verify_files(lock['expected_sensor_hashes'])
    S.verify_files(complete['files'],runs)
    actual={str(p.relative_to(runs)) for p in runs.rglob('*') if p.is_file() and p!=runs/'completion.json'}
    if actual!=set(complete['files']) or complete['lock_sha256']!=S.sha(runs/'lock.json'):raise ValueError('completion file set/lock')
    previous=ROOT/'results/round11_target_conflict/lock.json'
    if S.sha(previous)!=lock['previous_oracle_lock_sha256'] or S.sha(previous)!=(previous.with_suffix('.sha256')).read_text().strip():raise ValueError('previous oracle lock changed')
    old=S.read(previous);S.verify_files(old['source_hashes'],ROOT)
    data=Path(lock['data_root']);data_completion=S.read(data/'completion.json')
    if S.sha(data/'completion.json')!=lock['data_completion_sha256']:raise ValueError('cohort completion changed')
    cohort=[r for r in data_completion['manifests'] if r['split']=='train']
    if cohort!=lock['training_manifest_records'] or len(cohort)!=48 or {r['scene'] for r in cohort}!=set(SCENES):raise ValueError('wrong train-only cohort')
    if any(sum(r['scene']==s for r in cohort)!=16 for s in SCENES):raise ValueError('wrong per-scene cohort')
    summary=S.read(runs/'summary.json');stored=S.read(runs/'rows.json')
    mapping={(r['scene'],tuple(r['pair']),r['stride'],r['query']):r for r in stored}
    if len(mapping)!=len(stored) or len(stored)!=28224:raise ValueError('query population duplicate/incomplete')
    rows=[];maxrow=0.;maxcost=0.;projections=0;replays=[];started=time.time();first=set();active_ties=0
    provenance={(r['scene'],tuple(r['pair'])):r for r in summary['label_sources']}
    for i,record in enumerate(cohort):
        path=Path(record['path']);meta=S.read(path)
        if S.sha(path)!=record['sha256'] or meta['split']!='train' or meta['pair']!=record['pair'] or meta['scene']!=record['scene']:
            raise ValueError('source metadata changed')
        labels_path=path.parent/'labels.npz'
        if S.sha(labels_path)!=meta['label_sha256']:raise ValueError('old GT changed')
        prov=provenance.pop((meta['scene'],tuple(meta['pair'])))
        if prov!=dict(scene=meta['scene'],pair=meta['pair'],manifest_sha256=record['sha256'],labels_path=str(labels_path),labels_sha256=meta['label_sha256'],sensor_sources=meta['sensor_sources']):
            raise ValueError('label provenance mismatch')
        original=S.archive(labels_path);slug=meta['scene'].replace('/','__')+'__'+path.parent.name
        target=S.archive(runs/'targets'/(slug+'.npz'))
        expected={f'g{s}_{n}' for s in STRIDES for n in ('counts','point_xy','point_visible','point_status')}
        if set(target)!=expected:raise ValueError('target fields changed')
        for stride in STRIDES:
            count=histogram(original['gt_xy'],original['gt_visible'],stride)
            if target[f'g{stride}_counts'].dtype!=np.int16:raise ValueError('saved count dtype')
            np.testing.assert_array_equal(count,target[f'g{stride}_counts'])
            if stride in (8,16):
                for name in ('point_xy','point_visible','point_status'):
                    np.testing.assert_array_equal(target[f'g{stride}_{name}'],original[f'g{stride}_{name}'])
            for qi in range(len(count)):
                row,npools=row_reference(count[qi],target[f'g{stride}_point_xy'][qi],target[f'g{stride}_point_visible'][qi],meta['scene'],meta['pair'],stride,qi)
                oldrow=mapping.pop((meta['scene'],tuple(meta['pair']),stride,qi))
                if row['joint_valid']:
                    low,high=active_count_interval(count[qi]/count[qi].sum(),row['best_allowed_key'])
                    if not low<=oldrow['pooled_active_count']<=high:raise ValueError('infeasible active-pool cardinality')
                    active_ties+=int(oldrow['pooled_active_count']!=low)
                    row['pooled_active_count']=oldrow['pooled_active_count']
                error=compare(oldrow,row)
                maxrow=max(maxrow,error)
                if row['joint_valid']:maxcost=max(maxcost,abs(oldrow['min_kl']-row['min_kl']))
                projections+=npools;rows.append(row)
        if meta['scene'] not in first:
            # First manifest record is also the original metadata-first [30,45] pair.
            if meta['pair']!=[30,45]:raise ValueError('predeclared sensor replay pair changed')
            replays.append(sensor_replay(meta,target,original));first.add(meta['scene'])
        if (i+1)%8==0:print('AUDIT',i+1,'/48',len(rows),'queries',flush=True)
    if mapping or provenance:raise ValueError('extra row or provenance identity')
    by_stride={}
    for stride in STRIDES:
        subset=[r for r in rows if r['stride']==stride]
        scenes={s:summary_reference([r for r in subset if r['scene']==s]) for s in SCENES}
        equal={k:float(np.mean([v[k] for v in scenes.values()])) for k in EQUAL_FIELDS}
        by_stride[str(stride)]=dict(pooled=summary_reference(subset),by_scene=scenes,scene_equal_means=equal)
    summary_error=compare(summary['by_stride'],by_stride)
    ref=lock['reference'];reference=Path(ref['root'])
    if S.sha(reference/'summary.json')!=ref['summary_sha256'] or S.sha(reference/'completion.json')!=ref['completion_sha256']:
        raise ValueError('descriptive magnitude reference changed')
    completed=S.read(reference/'completion.json')
    if completed['files']['summary.json']!=ref['summary_sha256']:raise ValueError('descriptive reference completion mismatch')
    quad=S.read(reference/'summary.json')['summary']['scale']['quad16']
    expected=dict(quad16_scale_macro_direct_kl=quad['kl'],quad16_q24_to_g8_direct_kl=quad['by_condition']['g24_g8'])
    compare(ref['values'],expected);compare(ref['two_percent'],{k:.02*v for k,v in expected.items()})
    compare(summary['reference'],ref)
    ratios={metric:{name:by_stride['24']['scene_equal_means'][metric]/value for name,value in ref['two_percent'].items()}
        for metric in ('min_kl_mean_joint','min_kl_mass_weighted_joint','min_kl_mean_all_query_unconstrained_zero','min_kl_mass_weighted_all_area_unconstrained_zero')}
    ratio_error=compare(summary['q24_ratio_to_reference_two_percent'],ratios)
    for field in ('candidate_forward_read','new_validation_or_test_labels_read','configuration_selection','gate_exemption'):
        if summary[field] is not False:raise ValueError('diagnostic interpretation/scope changed')
    if summary['lock_sha256']!=S.sha(runs/'lock.json') or summary['record_count']!=28224 or summary['pair_count']!=48 or complete['record_count']!=28224:
        raise ValueError('summary population/lock')
    out.mkdir(parents=True)
    write(out/'verification.json',dict(passed=True,target_lock_sha256=S.sha(runs/'lock.json'),
        target_summary_sha256=S.sha(runs/'summary.json'),target_completion_sha256=S.sha(runs/'completion.json'),
        audit_source_lock_sha256=S.sha(source_lock),query_rows=len(rows),independent_scalar_projections=projections,
        joint_queries=sum(r['joint_valid'] for r in rows),maximum_row_error=maxrow,maximum_min_kl_error=maxcost,
        degenerate_auxiliary_active_set_rows=active_ties,
        maximum_summary_error=summary_error,maximum_cross_cohort_ratio_error=ratio_error,
        raw_sensor_replays=replays,dense_sensor_points=sum(r['dense_sensor_points'] for r in replays),
        center_sensor_points=sum(r['center_sensor_points'] for r in replays),
        by_stride=by_stride,seconds=time.time()-started,new_validation_labels_read=False,fresh_test_read=False,
        candidate_ability_validated=False,gate_exemption=False,
        interpretation='Relaxed sensor-target compatibility only; ties allowed. Small KL cost does not show an RGB-only candidate can attain it, or native first-index argmax succeeds. Invalid-joint rows have zero relaxed constraint cost, not correct localization. Cross-cohort magnitude ratios cannot alter R12 or any later gate.'))
    print(json.dumps({k:v for k,v in S.read(out/'verification.json').items() if k not in ('by_stride','raw_sensor_replays')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','audit'))
    p.add_argument('--runs',type=Path,default=ROOT/'results/round13_target_conflict')
    p.add_argument('--out',type=Path,default=ROOT/'results/round13_target_independent_audit')
    p.add_argument('--source-lock',type=Path,default=ROOT/'results/round13_target_audit_source_lock.json')
    a=p.parse_args()
    if a.stage=='freeze':freeze(a.source_lock,a.runs)
    else:audit(a.runs,a.out,a.source_lock)

if __name__=='__main__':main()
