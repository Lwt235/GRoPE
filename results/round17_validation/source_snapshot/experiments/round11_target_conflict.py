"""Independent old-training-only regional/center-target compatibility audit.

No candidate implementation, forward geometry or experiment result is imported.
This consumes only already frozen sensor-projection labels after its own source
and protocol are locked. Its KL is for direct g8 regional histograms, not the
Round11 aggregated common-grid score, and is not a PE performance bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
TRAIN_SCENES=('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
FIELDS=('gt_xy','gt_visible','g8_point_xy','g8_point_visible')
SOURCE_FILES=('experiments/round11_target_conflict.py','docs/round11_target_conflict_protocol.md')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def project_to_argmax(probability,index):
    """KL(g||p) minimizer subject to p[index]>=p[i] for all i, allowing ties."""
    g=np.asarray(probability,np.float64)
    if g.ndim!=1 or not len(g) or not np.isfinite(g).all() or (g<0).any() or abs(g.sum()-1.)>1.e-12:
        raise ValueError('expected a finite normalized nonnegative vector')
    if not 0<=index<len(g):raise ValueError('invalid candidate argmax')
    active=[int(index)];total=float(g[index]);level=total
    for other in np.argsort(-g,kind='stable'):
        if other==index:continue
        if g[other]<=level:break
        active.append(int(other));total+=float(g[other]);level=total/len(active)
    p=g.copy();p[active]=level
    positive=g>0
    value=float(np.sum(g[positive]*(np.log(g[positive])-np.log(p[positive]))))
    return p,max(value,0.),len(active)


def minimum_allowed_argmax_kl(probability,allowed):
    """Choose the allowed cell with largest g, then perform the convex pooling.

    A swap argument proves this choice is optimal: if g[a]>=g[b] and p[b]
    is maximal, swapping p[a],p[b] cannot worsen cross entropy and makes a
    maximal. Ties between eligible candidates use their first native index.
    """
    g=np.asarray(probability,np.float64);allowed=np.asarray(allowed)
    if allowed.shape!=g.shape or allowed.dtype!=np.bool_ or not allowed.any():
        raise ValueError('nonempty boolean allowed set required')
    candidates=np.flatnonzero(allowed)
    index=int(candidates[np.argmax(g[candidates])])
    p,value,count=project_to_argmax(g,index)
    return dict(value=value,index=index,active_count=count,probability=p)


def reconstruct_counts(xy,visible):
    """Literal 192x144 raster -> native g8 hard-cell count, independently."""
    xy=np.asarray(xy,np.float64);visible=np.asarray(visible)
    if xy.shape!=(144*192,2) or visible.shape!=(144*192,) or visible.dtype!=np.bool_:
        raise ValueError('unexpected dense sensor-label shape/dtype')
    finite=np.isfinite(xy).all(-1)
    inside=finite&(xy[:,0]>=-.5)&(xy[:,0]<191.5)&(xy[:,1]>=-.5)&(xy[:,1]<143.5)
    if np.any(visible&~inside):raise ValueError('visible projection outside fixed image')
    y,x=np.mgrid[:144,:192]
    query=(y//8*24+x//8).reshape(-1)
    target=np.floor((xy[visible]+.5+1.e-9)/8).astype(np.int64)
    target=np.minimum(np.maximum(target,[0,0]),[23,17])
    key=target[:,1]*24+target[:,0]
    counts=np.zeros((432,432),np.int64)
    np.add.at(counts,(query[visible],key),1)
    if int(counts.sum())!=int(visible.sum()):raise ValueError('lost sensor mass')
    return counts


def diagnose_pair(labels,scene,pair):
    counts=reconstruct_counts(labels['gt_xy'],labels['gt_visible'])
    center=np.asarray(labels['g8_point_xy'],np.float64)
    center_valid=np.asarray(labels['g8_point_visible'])
    if center.shape!=(432,2) or center_valid.shape!=(432,) or center_valid.dtype!=np.bool_:
        raise ValueError('unexpected independent center-label shape/dtype')
    if np.any(center_valid&~np.isfinite(center).all(-1)):
        raise ValueError('valid center has nonfinite projection')
    yy,xx=np.mgrid[:18,:24]
    key_centers=np.stack([xx*8+3.5,yy*8+3.5],-1).reshape(-1,2)
    rows=[]
    for qi,count in enumerate(counts):
        mass=int(count.sum());valid_center=bool(center_valid[qi])
        row=dict(scene=scene,pair=pair,query=qi,visible_source_pixels=mass,
            area_valid=mass>0,center_valid=valid_center,joint_valid=mass>0 and valid_center,
            native_first_argmax_correct=None,optimistic_top_tie_correct=None,min_kl=None,
            allowed_key_count=None,best_allowed_key=None,pooled_active_count=None)
        if row['joint_valid']:
            g=count/mass
            allowed=np.square(key_centers-center[qi]).sum(-1)<=64.
            if not allowed.any():raise ValueError('valid center has no native key within 8 pixels')
            first=int(np.argmax(count))
            optimistic=bool(np.any(allowed&(count==count.max())))
            minimum=minimum_allowed_argmax_kl(g,allowed)
            row.update(native_first_argmax_correct=bool(allowed[first]),optimistic_top_tie_correct=optimistic,
                min_kl=minimum['value'],allowed_key_count=int(allowed.sum()),best_allowed_key=minimum['index'],
                pooled_active_count=minimum['active_count'])
            if optimistic and minimum['value']>1.e-12:raise ValueError('zero-conflict tie has positive KL bound')
        rows.append(row)
    return rows


def summarize(rows):
    joint=[r for r in rows if r['joint_valid']]
    if not joint:raise ValueError('empty joint-valid diagnostic population')
    values=np.array([r['min_kl'] for r in joint]);weights=np.array([r['visible_source_pixels'] for r in joint])
    first=sum(r['native_first_argmax_correct'] for r in joint)
    optimistic=sum(r['optimistic_top_tie_correct'] for r in joint)
    return dict(total_queries=len(rows),area_valid_queries=sum(r['area_valid'] for r in rows),
        center_valid_queries=sum(r['center_valid'] for r in rows),joint_valid_queries=len(joint),
        area_valid_center_invalid=sum(r['area_valid'] and not r['center_valid'] for r in rows),
        center_valid_area_invalid=sum(r['center_valid'] and not r['area_valid'] for r in rows),
        neither_valid=sum(not r['area_valid'] and not r['center_valid'] for r in rows),
        all_visible_source_pixels=sum(r['visible_source_pixels'] for r in rows),
        joint_visible_source_pixels=int(weights.sum()),native_first_argmax_pck=first/len(joint),
        optimistic_top_tie_pck=optimistic/len(joint),native_first_conflict_fraction=1-first/len(joint),
        optimistic_conflict_fraction=1-optimistic/len(joint),
        min_kl_mean_per_joint_query=float(values.mean()),min_kl_visible_mass_weighted=float(np.dot(values,weights)/weights.sum()),
        min_kl_quantiles={str(q):float(np.quantile(values,q)) for q in (0.,.5,.9,.95,.99,1.)},
        positive_lower_bound_queries=int((values>1.e-12).sum()),
        top_tie_only_conflicts=int(optimistic-first))


def data_records(data):
    completion=json.loads((data/'completion.json').read_text())
    if completion['fresh_test_read'] or completion['pair_count']!=80:
        raise ValueError('unexpected old-development data contract')
    if sha(data/'contract.json')!=completion['contract_sha256']:
        raise ValueError('changed source data contract')
    records=[r for r in completion['manifests'] if r['split']=='train']
    if len(records)!=48 or set(r['scene'] for r in records)!=set(TRAIN_SCENES):
        raise ValueError('only the exact 48 old-training pairs are allowed')
    for scene in TRAIN_SCENES:
        if sum(r['scene']==scene for r in records)!=16:raise ValueError('changed per-scene population')
    return records


def freeze(out,data):
    if out.exists():raise ValueError('refusing overwrite diagnostic directory')
    records=data_records(data)
    lock=dict(source_hashes={name:sha(ROOT/name) for name in SOURCE_FILES},
        data_root=str(data.resolve()),data_completion_sha256=sha(data/'completion.json'),
        contract_sha256=sha(data/'contract.json'),training_manifest_records=records,
        created_unix=time.time(),stage='before any training label archive is read by this diagnostic',
        candidate_results_read=False,validation_labels_read=False)
    out.mkdir(parents=True);write(out/'lock.json',lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN',sha(out/'lock.json'),flush=True)


def verify(out,data):
    lock=json.loads((out/'lock.json').read_text())
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('changed diagnostic lock')
    if lock['source_hashes']!={name:sha(ROOT/name) for name in SOURCE_FILES}:raise ValueError('changed diagnostic source/protocol')
    if str(data.resolve())!=lock['data_root'] or sha(data/'completion.json')!=lock['data_completion_sha256']:
        raise ValueError('changed data root/completion')
    if data_records(data)!=lock['training_manifest_records']:raise ValueError('changed training identities')
    return lock


def run(out,data):
    lock=verify(out,data)
    if (out/'rows.json').exists() or (out/'summary.json').exists():raise ValueError('refusing overwrite diagnostic result')
    rows=[];label_sources=[]
    for record in lock['training_manifest_records']:
        path=Path(record['path'])
        if sha(path)!=record['sha256']:raise ValueError('changed training pair manifest')
        meta=json.loads(path.read_text())
        if meta['split']!='train' or meta['scene'] not in TRAIN_SCENES or meta['scene']!=record['scene'] or meta['pair']!=record['pair']:
            raise ValueError('training manifest identity mismatch')
        label_path=path.parent/'labels.npz'
        if sha(label_path)!=meta['label_sha256']:raise ValueError('changed training labels')
        with np.load(label_path,allow_pickle=False) as archive:
            labels={name:archive[name].copy() for name in FIELDS}
        rows.extend(diagnose_pair(labels,meta['scene'],meta['pair']))
        label_sources.append(dict(path=str(label_path.resolve()),sha256=meta['label_sha256'],
            manifest_path=str(path.resolve()),manifest_sha256=record['sha256']))
    if len(rows)!=48*432:raise ValueError('lost query rows')
    by_scene={scene:summarize([r for r in rows if r['scene']==scene]) for scene in TRAIN_SCENES}
    pooled=summarize(rows)
    metrics=('native_first_argmax_pck','optimistic_top_tie_pck','native_first_conflict_fraction',
             'optimistic_conflict_fraction','min_kl_mean_per_joint_query','min_kl_visible_mass_weighted')
    result=dict(pooled=pooled,by_scene=by_scene,
        scene_equal_means={name:float(np.mean([by_scene[scene][name] for scene in TRAIN_SCENES])) for name in metrics},
        label_sources=label_sources,fields_read=list(FIELDS),lock_sha256=sha(out/'lock.json'),
        source_hashes=lock['source_hashes'],candidate_results_read=False,validation_labels_read=False,
        sensor_images_read=False,predicted_geometry_read=False,
        interpretation='Direct g8 sensor-region targets versus independent center targets, joint-valid rows only for PCK/KL. Ties relax argmax; this is not a method result, deployable oracle, common-grid benchmark lower bound or gate exemption.')
    verify(out,data);write(out/'rows.json',rows);write(out/'summary.json',result)
    files={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}
    write(out/'completion.json',dict(files=files,seconds=time.time()-lock['created_unix'],fresh_test_read=False))
    print(json.dumps(dict(pooled=pooled,scene_equal_means=result['scene_equal_means'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    args=parser.parse_args()
    if args.stage=='freeze':freeze(args.out,args.data)
    else:run(args.out,args.data)


if __name__=='__main__':main()
