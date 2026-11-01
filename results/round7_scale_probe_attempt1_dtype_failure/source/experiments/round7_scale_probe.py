"""Fixed Round6 configurations on five NEW 24px grids; old-data diagnosis only.

Freeze protocol first, then stream RGB-only inputs and sensor-only labels.
No optimization, calibration, condition selection, or future Round7 input path.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from experiments import round7_scale_data as data

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'results/round6_confirmation/lock.json'
LOCK_SHA='878ce25b0273333e9e58438261d1a58c7ba1fd28005aade37d9321986815cd92'
ARCHIVE=ROOT/'results/round6_confirmation/source'
ROLES=('point','fixed','isotropic','full','point_pck')
GRIDS=tuple(f'q{q}_k{k}' for q,k in data.CONDITIONS)
VAL_SCENES=('harvard_c8/hv_c8_3','brown_bm_4/brown_bm_4')
TEST_SCENES=('brown_cogsci_1/brown_cogsci_1','brown_cs_3/brown_cs3',
    'home_at/home_at_scan1_2013_jan_1','home_bksh/home_bksh_oct_30_2012_scan2_erika','mit_32_d507/d507_2')
NEW_SOURCES=('experiments/round7_scale_data.py','experiments/round7_scale_probe.py',
    'tests/test_round7_scale_data.py','tests/test_round7_scale_probe.py')
METRICS={'kl':('kl_sum','visible_mass'),'ce':('ce_sum','visible_mass'),
    'js':('js_sum','visible_mass'),'row_kl':('row_kl_sum','n_valid'),
    'pck8':('pck8_count','n_point'),'pck4':('pck4_count','n_point'),
    'pck_stride':('pck_stride_count','n_point'),'epe':('point_epe_sum','n_point'),
    'soft_epe':('point_soft_epe_sum','n_point'),'point_ce':('point_ce_sum','n_point')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(values):
    digest=hashlib.sha256()
    for key in sorted(values):
        value=np.asarray(values[key]);digest.update(key.encode())
        digest.update(str(value.shape).encode());digest.update(value.dtype.str.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def frozen_helpers():
    lock=json.loads(LOCK.read_text())
    if sha(LOCK)!=LOCK_SHA:raise ValueError('Round6 lock changed')
    if tuple(lock['expected_scenes'])!=TEST_SCENES:raise ValueError('old-test scene allowlist differs')
    for name in ('experiments/round6_region_data.py','experiments/round5_rgbd_data.py'):
        if sha(ROOT/name)!=lock['sourcehash'][name]:raise ValueError('sealed data helper changed: '+name)
    for name in ('models/gaussian_quadratic_pe.py','experiments/round6_train.py'):
        if sha(ARCHIVE/name)!=lock['sourcehash'][name]:raise ValueError('archive mismatch: '+name)
    spec=importlib.util.spec_from_file_location('round7_archived_quadratic',ARCHIVE/'models/gaussian_quadratic_pe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Execute only the two frozen metric functions, without importing live models.
    tree=ast.parse((ARCHIVE/'experiments/round6_train.py').read_text())
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('per_query','measure_rows')]
    if len(selected)!=2:raise ValueError('missing archived metric functions')
    namespace={'torch':torch,'F':F}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(ARCHIVE/'experiments/round6_train.py'),'exec'),namespace)
    return lock,module.GaussianQuadraticMatcher,namespace['measure_rows']


def scene_jobs(lock):
    jobs=[]
    for raw,cache,selection,allowed in [
        ('data/round5_sun3d','results/round5_inputs/cache','data/round5_sun3d/selection.json',VAL_SCENES[:1]),
        ('data/round6_sun3d','results/round6_inputs/cache','data/round6_sun3d/selection_v2.json',VAL_SCENES[1:]+TEST_SCENES)]:
        path=ROOT/selection
        if 'round6' in selection and sha(path)!=lock['selection_sha256']:raise ValueError('Round6 selected pairs changed')
        rows={r['scene']:r for r in json.loads(path.read_text())['scenes'] if r['scene'] in allowed}
        if set(rows)!=set(allowed):raise ValueError('missing old scene')
        for scene in allowed:
            row=rows[scene];split='test' if scene in TEST_SCENES else 'val';limit=96 if split=='test' else 64
            if row['split']!=split or len(row['pairs'])<limit:raise ValueError('old fixed split/pair count changed')
            # Every selected first pair must match the exact already-opened Round6 export.
            old=ROOT/('results/round6_test_region_inputs' if split=='test' else
                'results/round6_region_inputs' if scene==VAL_SCENES[0] else 'results/round6_new_region_inputs')
            manifest=old/split/scene.replace('/','__')/'q8_k8/manifest.json'
            if json.loads(manifest.read_text())['pairs'][:limit]!=row['pairs'][:limit]:raise ValueError('pair identity mismatch')
            jobs.append(dict(raw_root=str(ROOT/raw),cache_root=str(ROOT/cache),selection_path=str(path),
                selection_sha256=sha(path),scene=scene,split=split,pairs=row['pairs'][:limit],row=row,
                prior_pair_manifest=str(manifest),prior_pair_manifest_sha256=sha(manifest)))
    return jobs


def protocol():
    lock,_,_=frozen_helpers();jobs=scene_jobs(lock)
    return dict(scope='diagnostic extension of already used two validation and five Round6 test scan/place groups; NOT a new independent confirmation',
        round6_lock_path=str(LOCK),round6_lock_sha256=LOCK_SHA,roles={r:lock['roles'][r]['config'] for r in ROLES},
        conditions=[list(c) for c in data.CONDITIONS],grids=list(GRIDS),reference_pixels=8,
        token24_count=48,feature_pool='3x3 spatial mean of base8 frozen128D visual features then L2; no new learned parameters',
        support='same RGB-predicted 3D scatter plus within-pixel continuous covariance; native s^2/(12*64) I; no optimized GS R/S',
        label='same sensor-nearest all192x144 source-pixel hard target-cell area measure; visible>0 only; no GT mask forward',
        exact_mass='sum integer target_counts, encode ratio/sample_count as float64 for archived metric helper; avoids float32 576-denominator rounding',
        point='same query-center sensor label; report target-stride PCK and fixed8px/4px PCK, EPE, and nearest-grid oracle ceilings; no new guard or selection',
        aggregation='sum visible-pixel mass within scene/grid then equal scan and equal five NEW grids; val and old test separate',
        uncertainty='descriptive paired scan/place percentile bootstrap20000 RNG20261030 only on five old test groups; no seed replicates or independence claim',
        no_tuning=True,no_condition_selection=True,no_future_round7_test=True,device='cpu',threads=1,
        jobs=jobs,sourcehash={name:sha(ROOT/name) for name in NEW_SOURCES},
        sealed_dependencies={name:lock['sourcehash'][name] for name in ('models/gaussian_quadratic_pe.py',
            'experiments/round6_train.py','experiments/round6_region_data.py','experiments/round5_rgbd_data.py')})


def freeze(out):
    if out.exists() and any(out.iterdir()):raise ValueError('refusing to replace existing Round7 output')
    p=protocol();out.mkdir(parents=True,exist_ok=True)
    for name in NEW_SOURCES:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    (out/'frozen_protocol.json').write_text(json.dumps(p,indent=2))
    print('frozen',sha(out/'frozen_protocol.json'),flush=True)


def group_metrics(rows):
    grouped={}
    for row in rows:
        key=(row['scene'],row['grid']);acc=grouped.setdefault(key,defaultdict(float))
        for metric,denominator in METRICS.values():
            acc[metric]+=row[metric]
        # Add shared denominators only once, unlike metrics with shared denominator.
        for denominator in {d for _,d in METRICS.values()}:acc[denominator]+=row[denominator]
    result={}
    for (scene,grid),acc in grouped.items():
        if any(acc[d]<=0 for _,d in METRICS.values()):raise ValueError('empty scene/grid supervision')
        result[scene,grid]={name:acc[m]/acc[d] for name,(m,d) in METRICS.items()}
    return result


def paired_description(a,b,scenes,grids,bootstrap=False):
    # One scene retains ALL its selected pairs and requested grids in each draw.
    av=np.array([[a[s,g]['kl'] for g in grids] for s in scenes]).mean(-1)
    bv=np.array([[b[s,g]['kl'] for g in grids] for s in scenes]).mean(-1)
    record=dict(full=float(av.mean()),fixed=float(bv.mean()),absolute=float((av-bv).mean()),
        relative=float(av.mean()/bv.mean()-1),improving_places=int((av<bv).sum()),n_places=len(scenes),
        by_scene={s:dict(full=float(x),fixed=float(y),relative=float(x/y-1)) for s,x,y in zip(scenes,av,bv)})
    if bootstrap:
        draws=np.random.default_rng(20261030).integers(0,len(scenes),(20000,len(scenes)))
        am,bm=av[draws].mean(-1),bv[draws].mean(-1)
        record['descriptive_relative_ci95']=np.quantile(am/bm-1,[.025,.975]).tolist()
        record['descriptive_absolute_ci95']=np.quantile(am-bm,[.025,.975]).tolist()
    return record


def summarize(rows,labels):
    summary={}
    for split,scenes in [('val',VAL_SCENES),('test',TEST_SCENES)]:
        grouped={role:group_metrics([r for r in rows[role] if r['split']==split]) for role in ROLES}
        expected={(s,g) for s in scenes for g in GRIDS}
        if any(set(value)!=expected for value in grouped.values()):raise ValueError('incomplete reporting grid')
        role_summary={role:dict(macro={m:float(np.mean([v[m] for v in table.values()])) for m in METRICS},
            by_scene={s:{m:float(np.mean([table[s,g][m] for g in GRIDS])) for m in METRICS} for s in scenes},
            by_grid={g:{m:float(np.mean([table[s,g][m] for s in scenes])) for m in METRICS} for g in GRIDS},
            scene_grid=[dict(scene=s,grid=g,**v) for (s,g),v in table.items()]) for role,table in grouped.items()}
        comparisons={g:paired_description(grouped['full'],grouped['fixed'],scenes,[g],split=='test') for g in GRIDS}
        comparisons['all_new_grids']=paired_description(grouped['full'],grouped['fixed'],scenes,GRIDS,split=='test')
        oracle=defaultdict(lambda:defaultdict(float))
        for r in labels:
            if r['split']!=split:continue
            for name in ('n_point','pck8_count','pck_stride_count'):oracle[r['scene'],r['grid']][name]+=r['oracle'][name]
        summary[split]=dict(roles=role_summary,full_minus_fixed_kl=comparisons,
            oracle={g:{'pck8':float(np.mean([oracle[s,g]['pck8_count']/oracle[s,g]['n_point'] for s in scenes])),
                       'pck_stride':float(np.mean([oracle[s,g]['pck_stride_count']/oracle[s,g]['n_point'] for s in scenes]))} for g in GRIDS},
            all_grid_kl_better=all(v['relative']<0 for v in comparisons.values()),
            claim='old-data extension diagnostic; no candidate or h/T selection from these results')
    return summary


def tensor_labels(label,inputs):
    mass=data.exact_visible_mass(label);counts=np.asarray(label.sample_count,dtype=np.float64)
    d={name:torch.as_tensor(np.asarray(getattr(label,name))).unsqueeze(0) for name in
        ('soft_labels','gt_valid','point_gt_valid','point_pixcontinuous','point_key_index')}
    d['visible_fraction']=torch.as_tensor(mass/counts).unsqueeze(0)
    d['sample_count']=torch.as_tensor(counts).unsqueeze(0)
    d['kmean']=inputs['kmean']
    return d


def run(out):
    started=time.time();p=json.loads((out/'frozen_protocol.json').read_text())
    if (out/'completion.json').exists():raise ValueError('already completed; refusing overwrite')
    if p!=protocol():raise ValueError('frozen protocol/source/selection changed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    _,Matcher,measure_rows=frozen_helpers()
    models={role:Matcher(**p['roles'][role]).eval() for role in ROLES}
    rows={role:[] for role in ROLES};label_rows=[];provenance=[]
    for job in p['jobs']:
        frames=data.frozen_data.read_scene_selection(Path(job['raw_root']),job['row'])
        selected=sorted({index for pair in job['pairs'] for index in pair})
        prepared={};sourcehash={};cache_provenance={}
        def record(path):
            sourcehash[str(path)]=sha(path)
        directory=Path(job['raw_root'])/job['scene']
        for path in (directory/'intrinsics.txt',Path(job['raw_root'])/job['row']['pose_file'],
                     directory/'image_index.html',directory/'depth_index.html'):record(path)
        for index in selected:
            path=Path(job['cache_root'])/job['scene']/f'{index:07d}.npz'
            cache,sidecar=data.frozen_data._load_cache(path,frames[index],job['row'])
            cache_provenance[str(path)]=sidecar
            for file in (path,path.with_suffix('.json'),frames[index].rgb_path,frames[index].depth_path):record(file)
            for stride in data.STRIDES:prepared[index,stride]=data.prepare_frame(cache,stride)
        @lru_cache(maxsize=8)
        def depth(index):return data.frozen_data.load_sensor_depth(frames[index].depth_path)
        for pi,(qi,ki) in enumerate(job['pairs']):
            for qs,ks in data.CONDITIONS:
                arrays,diag=data.make_inputs(prepared[qi,qs],prepared[ki,ks],qs,ks)
                inputs={name:torch.from_numpy(value).unsqueeze(0).float() for name,value in arrays.items()}
                label=data.region_labels(depth(qi),depth(ki),frames[qi].K_native,frames[ki].K_native,
                    frames[qi].c2w,frames[ki].c2w,qs,ks)
                identity=dict(scene=job['scene'],split=job['split'],grid=f'q{qs}_k{ks}',
                    q_index=qi,k_index=ki,pair_id=f'{qi}:{ki}',q_stride=qs,k_stride=ks)
                d=tensor_labels(label,inputs)
                label_rows.append(dict(identity,input_sha256=array_sha(arrays),
                    label_sha256=array_sha({name:getattr(label,name) for name in
                        ('soft_labels','target_counts','gt_valid','counts_by_reason','point_gt_valid','point_pixcontinuous','point_key_index')}),
                    visible_mass=int(data.exact_visible_mass(label).sum()),n_valid=int(label.gt_valid.sum()),
                    sample_count=int(label.sample_count.sum()),counts_by_reason=label.counts_by_reason.sum(0).tolist(),
                    oracle=data.oracle_pck(label,ks),projection_diagnostics=diag))
                with torch.no_grad():
                    for role,model in models.items():
                        logits=model(inputs)
                        if not torch.isfinite(logits).all():raise ValueError('nonfinite logits')
                        measured=measure_rows(logits,d,[dict(identity,method=role)])[0]
                        if abs(measured['visible_mass']-label_rows[-1]['visible_mass'])>1e-8:raise ValueError('mass rounding error')
                        rows[role].append(measured)
            if (pi+1)%16==0:print(job['scene'],pi+1,'/',len(job['pairs']),'pairs',round(time.time()-started,2),'s',flush=True)
        depth.cache_clear()
        # Inputs and source files stay identical across all roles and throughout this scene.
        if any(sha(path)!=value for path,value in sourcehash.items()):raise ValueError('scene source changed during execution')
        provenance.append(dict(scene=job['scene'],split=job['split'],sourcehash=sourcehash,rgb_cache_provenance=cache_provenance))
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
        for role in ROLES:(out/f'{role}_rows.json').write_text(json.dumps(dict(rows=rows[role],config=p['roles'][role]),indent=2))
        print('completed scene',job['scene'],flush=True)
    if p!=protocol():raise ValueError('sealed source/protocol changed during run')
    expected=(2*64+5*96)*len(GRIDS)
    if len(label_rows)!=expected or any(len(value)!=expected for value in rows.values()):raise ValueError('incomplete row coverage')
    (out/'label_diagnostics.json').write_text(json.dumps(label_rows,indent=2))
    summary=summarize(rows,label_rows);(out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Round7 24px extension diagnostic','',p['scope'],
        '', 'The five Round6 role configurations were frozen unchanged. All first64 validation/96 old-test pairs and all five new grids are included. No future Round7 test was opened.',
        '', 'Source/RGB/cache/depth hashes, input/label digests, all per-pair role rows, supervision reasons, and quantization ceilings are saved. Features24 use3x3 pooled frozen features; no fitting was performed.',
        '', '|split|grid|full KL|fixed KL|relative full−fixed|improving places|', '|---|---|---:|---:|---:|---:|']
    for split,value in summary.items():
        for grid,r in value['full_minus_fixed_kl'].items():lines.append(f"|{split}|{grid}|{r['full']:.6f}|{r['fixed']:.6f}|{100*r['relative']:+.3f}%|{r['improving_places']}/{r['n_places']}|")
    lines+=['', 'PCK at8/4px changes its quantization ceiling for24px targets; target-stride PCK and oracle ceilings are reported alongside it. All point metrics use the separate sampled query-center label.',
        '', 'Confidence intervals are descriptive paired resampling of five already-used named scan/place groups; they are not a fresh confirmation, independent rooms guarantee, or five seed repetitions. Val n=2 is descriptive. Changes in token pooling and quantization are coupled to scale. This does not establish optimized3DGS covariance value.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'completion.json').write_text(json.dumps(dict(elapsed_seconds=time.time()-started,
        role_pair_grid_rows=sum(len(x) for x in rows.values()),label_pair_grid_conditions=len(label_rows),
        frozen_protocol_sha256=sha(out/'frozen_protocol.json'),sourcehash=p['sourcehash'],
        summary_sha256=sha(out/'summary.json')),indent=2))
    print('completed',time.time()-started,'s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/round7_scale_probe')
    parser.add_argument('mode',choices=('freeze','run'))
    args=parser.parse_args()
    (freeze if args.mode=='freeze' else run)(args.out)
