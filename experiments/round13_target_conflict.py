"""Old-training target compatibility across scale; no PE/model forward access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments import round11_target_conflict as T
from experiments.round6_region_data import read_scene_selection, _project_gt_pixels
from experiments.round5_rgbd_data import load_sensor_depth

ROOT=T.ROOT
STRIDES=(8,16,24)
SOURCE_FILES=('ROUND13_TARGET_PROTOCOL.md','experiments/round13_target_conflict.py',
    'tests/test_round13_target_conflict.py','experiments/round11_target_conflict.py',
    'docs/round11_target_conflict_protocol.md','experiments/round6_region_data.py',
    'experiments/round5_rgbd_data.py')


def source_hashes():return {p:T.sha(ROOT/p) for p in SOURCE_FILES}


def verify_files(files):
    for path,digest in files.items():
        if T.sha(path)!=digest:raise ValueError('changed source file: '+path)


def native_centers(stride):
    y,x=np.mgrid[:144//stride,:192//stride]
    return np.stack([x*stride+(stride-1)/2,y*stride+(stride-1)/2],-1).reshape(-1,2)


def reconstruct_counts(xy,visible,stride):
    if stride not in STRIDES:raise ValueError('unsupported source scale')
    xy=np.asarray(xy,np.float64);visible=np.asarray(visible)
    if xy.shape!=(144*192,2) or visible.shape!=(144*192,) or visible.dtype!=np.bool_:
        raise ValueError('dense sensor label contract')
    inside=np.isfinite(xy).all(-1)&(xy[:,0]>=-.5)&(xy[:,0]<191.5)&(xy[:,1]>=-.5)&(xy[:,1]<143.5)
    if np.any(visible&~inside):raise ValueError('visible projected pixel outside key image')
    y,x=np.mgrid[:144,:192]
    query=(y//stride*(192//stride)+x//stride).ravel()
    key_xy=np.floor((xy[visible]+.5+1.e-9)/8).astype(np.int64)
    key_xy=np.clip(key_xy,[0,0],[23,17]);key=key_xy[:,1]*24+key_xy[:,0]
    count=np.zeros(((144//stride)*(192//stride),432),np.int64)
    np.add.at(count,(query[visible],key),1)
    if count.sum()!=visible.sum():raise ValueError('lost source mass')
    return count


def diagnose_counts(counts,center,valid,scene,pair,stride):
    counts=np.asarray(counts);center=np.asarray(center,np.float64);valid=np.asarray(valid)
    n=(144//stride)*(192//stride)
    if counts.shape!=(n,432) or center.shape!=(n,2) or valid.shape!=(n,) or valid.dtype!=np.bool_:
        raise ValueError('target shape or validity contract')
    if np.any(valid&~np.isfinite(center).all(-1)):raise ValueError('nonfinite valid center')
    keys=native_centers(8);rows=[]
    for qi,count in enumerate(counts):
        mass=int(count.sum());joint=mass>0 and bool(valid[qi])
        row=dict(scene=scene,pair=pair,stride=stride,query=qi,visible_source_pixels=mass,
            area_valid=mass>0,center_valid=bool(valid[qi]),joint_valid=joint,
            native_first_argmax_correct=None,optimistic_top_tie_correct=None,strict_conflict=None,
            min_kl=None,allowed_key_indices=None,allowed_key_count=None,
            global_max_bin_mass=None,best_allowed_bin_mass=None,allowed_set_mass=None,
            max_mass_gap=None,allowed_to_global_max_ratio=None,first_argmax_key=None,
            first_argmax_distance=None,top_tie_count=None,best_allowed_key=None,pooled_active_count=None)
        if joint:
            g=count/mass;distance=np.linalg.norm(keys-center[qi],axis=-1)
            allowed=distance<=8.;indices=np.flatnonzero(allowed)
            if not len(indices):raise ValueError('visible center has empty allowed key set')
            first=int(count.argmax());max_count=int(count.max());best_count=int(count[allowed].max())
            minimum=T.minimum_allowed_argmax_kl(g,allowed)
            brute=min(T.project_to_argmax(g,int(j))[1] for j in indices)
            if abs(brute-minimum['value'])>1.e-12:raise ValueError('allowed-candidate optimum mismatch')
            optimistic=bool(best_count==max_count)
            if optimistic and minimum['value']>1.e-12:raise ValueError('tied feasible target has positive cost')
            row.update(native_first_argmax_correct=bool(allowed[first]),optimistic_top_tie_correct=optimistic,
                strict_conflict=best_count<max_count,min_kl=minimum['value'],allowed_key_indices=indices.tolist(),
                allowed_key_count=len(indices),global_max_bin_mass=max_count/mass,
                best_allowed_bin_mass=best_count/mass,allowed_set_mass=float(g[allowed].sum()),
                max_mass_gap=(max_count-best_count)/mass,allowed_to_global_max_ratio=best_count/max_count,
                first_argmax_key=first,first_argmax_distance=float(distance[first]),
                top_tie_count=int((count==max_count).sum()),best_allowed_key=minimum['index'],
                pooled_active_count=minimum['active_count'])
        rows.append(row)
    return rows


def quantiles(values):
    a=np.asarray(values,np.float64)
    if not len(a):return None
    return {str(q):float(np.quantile(a,q)) for q in (0.,.25,.5,.75,.9,.95,.99,1.)}


def summarize(rows):
    joint=[r for r in rows if r['joint_valid']]
    if not joint:raise ValueError('no joint-valid queries')
    values=np.array([r['min_kl'] for r in joint]);weight=np.array([r['visible_source_pixels'] for r in joint])
    total_mass=sum(r['visible_source_pixels'] for r in rows)
    area_count=sum(r['area_valid'] for r in rows);strict=[r for r in joint if r['strict_conflict']]
    first=sum(r['native_first_argmax_correct'] for r in joint);optimistic=sum(r['optimistic_top_tie_correct'] for r in joint)
    return dict(total_queries=len(rows),area_valid_queries=area_count,
        center_valid_queries=sum(r['center_valid'] for r in rows),joint_valid_queries=len(joint),
        area_valid_center_invalid=sum(r['area_valid'] and not r['center_valid'] for r in rows),
        center_valid_area_invalid=sum(r['center_valid'] and not r['area_valid'] for r in rows),
        neither_valid=sum(not r['area_valid'] and not r['center_valid'] for r in rows),
        all_visible_source_pixels=total_mass,joint_visible_source_pixels=int(weight.sum()),
        native_first_argmax_pck_joint=first/len(joint),optimistic_top_tie_pck_joint=optimistic/len(joint),
        native_first_conflict_fraction_joint=1-first/len(joint),strict_conflict_fraction_joint=len(strict)/len(joint),
        strict_conflict_fraction_all_query=len(strict)/len(rows),top_tie_only_conflicts=optimistic-first,
        min_kl_mean_joint=float(values.mean()),min_kl_mass_weighted_joint=float(values@weight/weight.sum()),
        min_kl_mean_all_query_unconstrained_zero=float(values.sum()/len(rows)),
        min_kl_mean_area_valid_unconstrained_zero=float(values.sum()/area_count),
        min_kl_mass_weighted_all_area_unconstrained_zero=float(values@weight/total_mass),
        min_kl_quantiles_joint=quantiles(values),min_kl_quantiles_strict=quantiles([r['min_kl'] for r in strict]),
        min_kl_quantiles_all_query_unconstrained_zero=quantiles([r['min_kl'] if r['joint_valid'] else 0. for r in rows]),
        global_max_bin_mass_quantiles=quantiles([r['global_max_bin_mass'] for r in joint]),
        best_allowed_bin_mass_quantiles=quantiles([r['best_allowed_bin_mass'] for r in joint]),
        allowed_set_mass_quantiles=quantiles([r['allowed_set_mass'] for r in joint]),
        allowed_to_global_max_ratio_quantiles=quantiles([r['allowed_to_global_max_ratio'] for r in joint]),
        max_mass_gap_quantiles=quantiles([r['max_mass_gap'] for r in joint]),
        allowed_count_histogram={str(n):sum(r['allowed_key_count']==n for r in joint)
                                  for n in sorted({r['allowed_key_count'] for r in joint})})


def reference_contract(reference):
    completion=json.loads((reference/'completion.json').read_text())
    path=reference/'summary.json'
    if T.sha(path)!=completion['files']['summary.json']:raise ValueError('changed R12 scale reference')
    result=json.loads(path.read_text());quad=result['summary']['scale']['quad16']
    values=dict(quad16_scale_macro_direct_kl=quad['kl'],quad16_q24_to_g8_direct_kl=quad['by_condition']['g24_g8'])
    return dict(root=str(reference.resolve()),summary_sha256=T.sha(path),completion_sha256=T.sha(reference/'completion.json'),
        values=values,two_percent={k:.02*v for k,v in values.items()},
        interpretation='Descriptive KL magnitude only: 32 old validation pairs versus this diagnostic 48 train pairs; neither training supervision nor gate exemption. No method/configuration selected here.')


def freeze(out,data,reference):
    if out.exists():raise ValueError('refusing overwrite')
    records=T.data_records(data);metadata={};sensor={}
    for record in records:
        path=Path(record['path'])
        if T.sha(path)!=record['sha256']:raise ValueError('changed training manifest')
        meta=json.loads(path.read_text())
        if meta['split']!='train' or meta['scene']!=record['scene'] or meta['pair']!=record['pair']:
            raise ValueError('training identity mismatch')
        metadata[str(path)]=record['sha256'];metadata[meta['source_selection']]=meta['source_selection_sha256']
        metadata.update(meta['camera_files']);metadata.update(meta['source_index_files'])
        sensor.update(meta['sensor_sources'])  # Expected hashes only; sensor bytes are not opened yet.
    verify_files(metadata)
    previous=ROOT/'results/round11_target_conflict/lock.json'
    old=json.loads(previous.read_text())
    if T.sha(previous)!=(previous.with_suffix('.sha256')).read_text().strip():raise ValueError('changed R11 oracle lock')
    verify_files({str(ROOT/p):h for p,h in old['source_hashes'].items()})
    lock=dict(source_hashes=source_hashes(),created_unix=time.time(),data_root=str(data.resolve()),
        data_completion_sha256=T.sha(data/'completion.json'),training_manifest_records=records,
        source_metadata=metadata,expected_sensor_hashes=sensor,previous_oracle_lock_sha256=T.sha(previous),
        reference=reference_contract(reference),strides=list(STRIDES),key_stride=8,
        stage='before this diagnostic reads old training labels or decodes sensor images',
        candidate_forward_read=False,new_validation_or_test_labels_read=False,configurations_selected=False)
    out.mkdir(parents=True);T.write(out/'lock.json',lock);(out/'lock.sha256').write_text(T.sha(out/'lock.json')+'\n')
    print('FROZEN',T.sha(out/'lock.json'),flush=True)


def verify(out,data):
    lock=json.loads((out/'lock.json').read_text())
    if T.sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('changed diagnostic lock')
    if source_hashes()!=lock['source_hashes']:raise ValueError('changed source/protocol')
    if str(data.resolve())!=lock['data_root'] or T.sha(data/'completion.json')!=lock['data_completion_sha256']:
        raise ValueError('changed cohort root/completion')
    if T.data_records(data)!=lock['training_manifest_records']:raise ValueError('changed training cohort')
    verify_files(lock['source_metadata'])
    if reference_contract(Path(lock['reference']['root']))!=lock['reference']:raise ValueError('changed magnitude reference')
    return lock


def sensor_centers(meta):
    selection_path=Path(meta['source_selection']);selection=json.loads(selection_path.read_text())
    row=next(r for r in selection['scenes'] if r['scene']==meta['scene'])
    frames=read_scene_selection(selection_path.parent,row);q,k=(frames[i] for i in meta['pair'])
    if {str(q.depth_path),str(k.depth_path)}!=set(meta['sensor_sources']):raise ValueError('timestamp mapping mismatch')
    verify_files(meta['sensor_sources'])
    qdepth,kdepth=load_sensor_depth(q.depth_path),load_sensor_depth(k.depth_path)
    result={}
    for stride in STRIDES:
        xy,valid,status=_project_gt_pixels(native_centers(stride),qdepth,kdepth,q.K_native,k.K_native,q.c2w,k.c2w,(144,192),.05,.05)
        result[stride]=(xy,valid,status)
    return result


def run(out,data):
    lock=verify(out,data)
    if (out/'rows.json').exists() or (out/'targets').exists():raise ValueError('refusing overwrite/partial run')
    (out/'targets').mkdir();started=time.time();rows=[];sources=[]
    for index,record in enumerate(lock['training_manifest_records']):
        path=Path(record['path']);meta=json.loads(path.read_text());labelpath=path.parent/'labels.npz'
        if T.sha(labelpath)!=meta['label_sha256']:raise ValueError('changed dense labels')
        with np.load(labelpath,allow_pickle=False) as archive:
            labels={k:archive[k] for k in ('gt_xy','gt_visible','g8_point_xy','g8_point_visible','g8_point_status',
                                          'g16_point_xy','g16_point_visible','g16_point_status')}
        centers=sensor_centers(meta);saved={}
        for stride in STRIDES:
            xy,valid,status=centers[stride]
            if stride in (8,16):
                np.testing.assert_allclose(xy,labels[f'g{stride}_point_xy'],atol=1.e-10,rtol=1.e-12,equal_nan=True)
                np.testing.assert_array_equal(valid,labels[f'g{stride}_point_visible'])
                np.testing.assert_array_equal(status,labels[f'g{stride}_point_status'])
            counts=reconstruct_counts(labels['gt_xy'],labels['gt_visible'],stride)
            rows.extend(diagnose_counts(counts,xy,valid,meta['scene'],meta['pair'],stride))
            saved.update({f'g{stride}_counts':counts.astype(np.int16),f'g{stride}_point_xy':xy,
                          f'g{stride}_point_visible':valid,f'g{stride}_point_status':status})
        slug=meta['scene'].replace('/','__')+'__'+path.parent.name
        np.savez_compressed(out/'targets'/(slug+'.npz'),**saved)
        sources.append(dict(scene=meta['scene'],pair=meta['pair'],manifest_sha256=record['sha256'],
            labels_path=str(labelpath),labels_sha256=meta['label_sha256'],sensor_sources=meta['sensor_sources']))
        if (index+1)%8==0:print('PAIRS',index+1,'/48',flush=True)
    if len(rows)!=48*sum((144//s)*(192//s) for s in STRIDES):raise ValueError('lost query population')
    by_stride={}
    for stride in STRIDES:
        selected=[r for r in rows if r['stride']==stride]
        scenes={scene:summarize([r for r in selected if r['scene']==scene]) for scene in T.TRAIN_SCENES}
        fields=('native_first_argmax_pck_joint','optimistic_top_tie_pck_joint','strict_conflict_fraction_joint',
                'min_kl_mean_joint','min_kl_mass_weighted_joint','min_kl_mean_all_query_unconstrained_zero',
                'min_kl_mass_weighted_all_area_unconstrained_zero')
        equal={k:float(np.mean([v[k] for v in scenes.values()])) for k in fields}
        by_stride[str(stride)]=dict(pooled=summarize(selected),by_scene=scenes,scene_equal_means=equal)
    compare={metric:{ref:by_stride['24']['scene_equal_means'][metric]/value
                     for ref,value in lock['reference']['two_percent'].items()}
             for metric in ('min_kl_mean_joint','min_kl_mass_weighted_joint',
                            'min_kl_mean_all_query_unconstrained_zero','min_kl_mass_weighted_all_area_unconstrained_zero')}
    result=dict(by_stride=by_stride,reference=lock['reference'],q24_ratio_to_reference_two_percent=compare,
        record_count=len(rows),pair_count=48,label_sources=sources,lock_sha256=T.sha(out/'lock.json'),
        source_hashes=source_hashes(),candidate_forward_read=False,new_validation_or_test_labels_read=False,
        sensor_images_read=True,configuration_selection=False,gate_exemption=False,
        interpretation='Target compatibility on reused TRAIN data only. KL projection allows a top tie and is not a guarantee for first-index argmax, a deployable oracle, or a bound on available-input PE accuracy. All-query/all-area means assign zero cost where no regional-center joint constraint is defined. Cross-cohort R12 KL reference is descriptive only.')
    verify(out,data);T.write(out/'rows.json',rows);T.write(out/'summary.json',result)
    files={str(p.relative_to(out)):T.sha(p) for p in out.rglob('*') if p.is_file()}
    T.write(out/'completion.json',dict(files=files,lock_sha256=T.sha(out/'lock.json'),
        seconds=time.time()-started,record_count=len(rows),fresh_test_read=False))
    print(json.dumps({s:v['pooled'] for s,v in by_stride.items()},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round13_target_conflict')
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--reference',type=Path,default=ROOT/'results/round12_stress')
    args=parser.parse_args()
    if args.stage=='freeze':freeze(args.out,args.data,args.reference)
    else:run(args.out,args.data)


if __name__=='__main__':main()
