"""Attribute actual 48D normalized-PE checkpoints to internal GS covariance.

Frozen old-camera input interventions only. Full, key-center-only,
query-center-only and both-center-only keep identical projected means, visual
features, masks, labels, resolution and learned parameters. No fitting or test
selection occurs. Actual archived normalized_shape code is executed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.isometric_confirmation_analysis import (
    SEEDS,archived_matcher,verify_source_snapshot,model_probability,
    real_probability_metrics,paired_camera_seed_statistics,real_row_identity,
    compare_summary,sha256)
from experiments.isometric_cov_counterfactual import state_digest
from experiments.matching_probe import digest
from experiments.query_support import use_fitted_query
from experiments.real_support_data import project_gaussian
from experiments.region_pe_probe import real_dataset


SCENES=['teatime','waldo_kitchen','figurines','temple']
CONDITIONS=['full','key_centeronly','query_centeronly','both_centeronly']
METRICS=['cross_entropy','js','top1_target_mass','dominant_region_accuracy']
METHOD='normalized_shape'


def protocol():
    return dict(purpose='Descriptive frozen 48D normalized_shape primitive-covariance attribution, not retrained ablation or prospective independent confirmation',
        checkpoints='results/round3_normalized_confirmation/training_raw/checkpoints/{seed}-normalized_shape.pt',
        seeds=SEEDS,method=METHOD,scenes=SCENES,partitions=['primary','alt'],conditions=CONDITIONS,
        main_comparison='full minus both_centeronly; key/query-only additionally detect asymmetric filter mismatch',
        covariance='Fixed law-of-total-covariance decomposition: remove weighted individual primitive covariance while retaining covariance of primitive centers; same native numerical eigenfloor',
        intervention='Project centeronly 3D covariance at ORIGINAL serialized world mean using original query camera J_KR; divide by24^2, add loader1e-6I once; model adds original .25^2I once',
        unchanged=['all projected means/phases','RGB features','3D means','visibility','candidate masks','>=95% retained-target-mass rows','targets','encoder/checkpoint','temperature'],
        query='fitted query geometry exactly matching these actual checkpoints; not the old native-rectangle/isometric experiment',
        data='Existing decomposition camera pairs: teatime/waldo old test; figurines/temple all original train/validation/test blocks. These are not the prospective 203 fresh pairs.',
        inference='paired five-seed/query-camera bootstrap stratified by fixed scene,4000 draws; no scene-population claim',
        primary_metric='cross_entropy',difference='full minus intervention; negative CE favors retained internal GS covariance',
        tuning='none; no temperature recalibration, checkpoint selection or retraining',
        limitations=['Covariance intervention changes the input distribution of full-trained encoders.',
                    'GS means are trained geometry even in centeronly; this is not raw SfM versus GS.',
                    'Opaque center visibility is held fixed; no re-rasterization or alpha compositing.',
                    'Sparse observed-track labels are conditional transfer proxies, not dense visibility truth.'])


def load_completed(root):
    manifest=json.loads((root/'manifest.json').read_text())
    config=manifest['config'];seeds=[int(s) for s in config['seeds'].split(',')]
    expected={(s,m) for s in seeds for m in config['methods'].split(',')}
    actual=[(r['seed'],r['method']) for r in manifest['records']]
    if ('elapsed_seconds' not in manifest or seeds!=SEEDS or len(actual)!=len(set(actual)) or set(actual)!=expected
            or len(actual)!=50 or config['query_geometry']!='fitted' or config['resolution']!=.25):
        raise ValueError('actual normalized training is incomplete or differs from requested frozen matrix')
    checks=verify_source_snapshot(root,manifest,['models/geometry.py','experiments/query_support.py',
        ('experiments/region_pe_probe.py',['real_dataset','evaluate'])])
    cls,implementation=archived_matcher(root,manifest)
    return manifest,cls,dict(source=checks,executed_archived_model=implementation,manifest_sha256=sha256(root/'manifest.json'))


def load_original_and_center(root,decomposition,scene,alternate,hashes):
    splits=['test'] if scene in ['teatime','waldo_kitchen'] else ['train','validation','test']
    blocks=[];pairs=[];metadatas=[]
    folder=root/scene;hashes[str(folder/'manifest.json')]=sha256(folder/'manifest.json')
    for split in splits:
        data,meta=real_dataset(root,scene,split,alternate)
        data,meta=use_fitted_query(data,meta,folder,alternate)
        blocks.append(data);pairs.extend(meta['pairs']);metadatas.append(meta)
    full={key:torch.cat([block[key] for block in blocks]) for key in blocks[0]}
    qcenter=full['projected_qcov'].clone();kcenter=full['projected_kcov'].clone()
    suffix='_alt' if alternate else '';diagnostics=[]
    for index,pair in enumerate(pairs):
        source=folder/pair['file']
        component=decomposition/'confirmation'/'pairs'/scene/pair['file']
        hashes[str(source)]=sha256(source);hashes[str(component)]=sha256(component)
        if hashes[str(source)]!=pair['sha256']:raise ValueError('original pair source changed')
        with np.load(source,allow_pickle=False) as data:
            K,R,t=data['K_query'],data['R_query'],data['t_query']
            values={f'{side}_{field}':data[f'{side}_{field}{suffix}'].copy()
                    for side in ['q','k'] for field in ['mean','cov','valid','projected_cov']}
        with np.load(component,allow_pickle=False) as data:
            components={side:data[f'{side}_centeronly{suffix}'].copy() for side in ['q','k']}
            full_precision={side:data[f'{side}_full{suffix}'].copy() for side in ['q','k']}
            for side in ['q','k']:
                np.testing.assert_array_equal(data[f'{side}_valid{suffix}'],values[f'{side}_valid'])
                np.testing.assert_allclose(data[f'{side}_mean{suffix}'],values[f'{side}_mean'],rtol=1e-6,atol=1e-7)
        for side in ['q','k']:
            valid=values[f'{side}_valid'].astype(bool)
            original_mean=values[f'{side}_mean'].astype(np.float64)
            # The decomposition stores the original fitter's float64 full C.
            # Serialized float32 world C can lose projected variance through
            # cancellation of a large radial component; keep that quantization
            # audit separate rather than loosening the projection tolerance.
            _,reprojected,_=project_gaussian(original_mean,full_precision[side],K,R,t)
            _,serialized_projection,_=project_gaussian(original_mean,values[f'{side}_cov'].astype(np.float64),K,R,t)
            _,projected,_=project_gaussian(original_mean,components[side],K,R,t)
            original_C2=values[f'{side}_projected_cov'].astype(np.float64)
            relative=np.linalg.norm(reprojected[valid]-original_C2[valid],axis=(-2,-1))/np.maximum(np.linalg.norm(original_C2[valid],axis=(-2,-1)),1e-20)
            serialized_error=np.linalg.norm(serialized_projection[valid]-original_C2[valid],axis=(-2,-1))/np.maximum(np.linalg.norm(original_C2[valid],axis=(-2,-1)),1e-20)
            if len(relative) and relative.max()>1e-4:raise ValueError('covariance projection did not reproduce original exporter')
            target=qcenter if side=='q' else kcenter
            target[index,valid]=torch.as_tensor(projected[valid]/24**2+np.eye(2)*1e-6,dtype=target.dtype)
            if torch.linalg.eigvalsh(target[index]).min()<=0:raise ValueError('centeronly projection not SPD')
            ratio=1-np.trace(projected[valid],axis1=-2,axis2=-1)/np.maximum(np.trace(original_C2[valid],axis1=-2,axis2=-1),1e-20)
            diagnostics.append(dict(scene=scene,alternate=alternate,file=pair['file'],side=side,
                valid_regions=int(valid.sum()),projection_max_relative_error=float(relative.max()) if len(relative) else None,
                serialized_world_cov_projection_max_relative_error=float(serialized_error.max()) if len(serialized_error) else None,
                removed_projected_trace_fraction_mean=float(ratio.mean()) if len(ratio) else None))
    datasets={'full':full}
    for condition in CONDITIONS[1:]:
        changed=dict(full)
        if condition in ['query_centeronly','both_centeronly']:changed['projected_qcov']=qcenter
        if condition in ['key_centeronly','both_centeronly']:changed['projected_kcov']=kcenter
        for key in full:
            if key not in ['projected_qcov','projected_kcov']:assert torch.equal(full[key],changed[key]),key
        datasets[condition]=changed
    identities=real_row_identity(pairs,full['label_valid'].numpy(),scene)
    return datasets,identities,dict(scene=scene,alternate=alternate,pairs=pairs,metadata=metadatas,
        dataset_sha256=digest(full),valid_rows=len(identities),query_cameras=len({r['q_id'] for r in identities})),diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,default=Path('results/round3_normalized_confirmation/training_raw'))
    p.add_argument('--development-data',type=Path,default=Path('results/round3_real_support_data_adjacent'))
    p.add_argument('--external-data',type=Path,default=Path('results/round3_external_scenes'))
    p.add_argument('--decomposition',type=Path,default=Path('results/round3_cov_decomposition'))
    p.add_argument('--out',type=Path,default=Path('results/round4_primitive_cov_audit'))
    args=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'summary.json').exists():raise ValueError('refusing to overwrite completed primitive attribution')
    manifest,matcher,implementation=load_completed(args.training)
    code_hash=sha256(__file__);protocol_path=args.out/'frozen_protocol.json'
    protocol_path.write_text(json.dumps(dict(protocol(),code_sha256=code_hash,training_manifest_sha256=implementation['manifest_sha256']),indent=2))
    started=time.time();hashes={};metadata={};diagnostics=[];cache={};identities={}
    # Validate frozen selected checkpoints against their actual old validation
    # inputs/metrics; historical probabilities for these old test pairs were
    # not archived by this development-phase training run.
    val,vm=real_dataset(Path(manifest['config']['real_root']),'teatime','validation')
    val,vm=use_fitted_query(val,vm,Path(manifest['config']['real_root'])/'teatime')
    for seed in SEEDS:
        if digest(val)!=manifest['datasets'][str(seed)]['validation']:raise ValueError('validation digest mismatch')
    for scene in SCENES:
        root=args.development_data if scene in ['teatime','waldo_kitchen'] else args.external_data
        for alternate in [False,True]:
            data,ids,meta,diag=load_original_and_center(root,args.decomposition,scene,alternate,hashes)
            cache[scene,alternate]=data;identities[scene,alternate]=ids
            metadata[f'{scene}/{int(alternate)}']=meta;diagnostics.extend(diag)
    rows={};records=[];validation_replay=[];probability_hashes={};array_archive={}
    archive=args.out/'predictions';archive.mkdir(exist_ok=True)
    for seed in SEEDS:
        path=args.training/'checkpoints'/f'{seed}-{METHOD}.pt';hashes[str(path)]=sha256(path)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['method']!=METHOD or payload['resolution']!=.25:raise ValueError('wrong checkpoint metadata')
        model=matcher(METHOD,payload['features'],payload['resolution']).eval()
        model.load_state_dict(payload['state_dict'],strict=True);state_before=state_digest(model)
        record=next(r for r in manifest['records'] if r['seed']==seed and r['method']==METHOD)
        val_metrics,_=real_probability_metrics(model_probability(model,val),val)
        errors=compare_summary(val_metrics,record['best_validation'])
        validation_replay.append(dict(seed=seed,errors=errors))
        for (scene,alternate),datasets in cache.items():
            saved={};valid=datasets['full']['label_valid'].numpy()
            for condition,data in datasets.items():
                probability=model_probability(model,data)
                summary,metrics=real_probability_metrics(probability,data)
                key=(scene,alternate,condition)
                rows.setdefault(key,{m:[] for m in METRICS})
                for metric in METRICS:
                    values=metrics[metric][valid]
                    rows[key][metric].append(values)
                    array_archive[f'{seed}/{scene}/{int(alternate)}/{condition}/{metric}']=values
                records.append(dict(seed=seed,scene=scene,alternate=alternate,condition=condition,metrics=summary))
                saved[condition]=probability
            if not np.array_equal(saved['full'],model_probability(model,datasets['full'])):raise ValueError('full prediction mutated')
            filename=archive/f'{seed}-{scene}-{int(alternate)}.npz'
            np.savez_compressed(filename,**saved,valid=valid,targets=datasets['full']['soft_labels'].numpy())
            probability_hashes[str(filename)]=sha256(filename)
        if state_digest(model)!=state_before:raise ValueError('model weights/state mutated')
        print(f'audited normalized_shape seed {seed}',flush=True)
    rows={k:{m:np.stack(v) for m,v in metrics.items()} for k,metrics in rows.items()}
    groups={s:[s] for s in SCENES};groups.update(external=['figurines','temple'],all=SCENES)
    means,differences={},{}
    for alternate in [False,True]:
        for group,scenes in groups.items():
            ids=[r for scene in scenes for r in identities[scene,alternate]]
            for condition in CONDITIONS:
                means[f'{int(alternate)}/{group}/{condition}']={m:float(np.concatenate([rows[scene,alternate,condition][m] for scene in scenes],axis=1).mean()) for m in METRICS}
                if condition=='full':continue
                for metric in METRICS:
                    a=np.concatenate([rows[scene,alternate,'full'][metric] for scene in scenes],axis=1)
                    b=np.concatenate([rows[scene,alternate,condition][metric] for scene in scenes],axis=1)
                    differences[f'{int(alternate)}/{group}/full_minus_{condition}/{metric}']=paired_camera_seed_statistics(a,b,ids)
    for path,h in hashes.items():
        if sha256(path)!=h:raise ValueError('source/checkpoint changed during audit')
    if sha256(args.training/'manifest.json')!=implementation['manifest_sha256'] or sha256(__file__)!=code_hash:raise ValueError('training manifest/audit code changed')
    output=dict(protocol_sha256=sha256(protocol_path),implementation=implementation,means=means,differences=differences,
        records=records,validation_replay=validation_replay,metadata=metadata,projection_audit=diagnostics,
        source_checkpoint_sha256=hashes,probability_sha256=probability_hashes,elapsed_seconds=time.time()-started)
    (args.out/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    np.savez_compressed(args.out/'row_metrics.npz',**array_archive)
    with (args.out/'per_row.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','scene','alternate','condition','q_id','k_id','query_region',*METRICS]);writer.writeheader()
        for (scene,alternate,condition),metrics in rows.items():
            for i,seed in enumerate(SEEDS):
                for j,identity in enumerate(identities[scene,alternate]):
                    writer.writerow(dict(seed=seed,scene=scene,alternate=alternate,condition=condition,
                        **{k:identity[k] for k in ['q_id','k_id','query_region']},**{m:float(metrics[m][i,j]) for m in METRICS}))
    shutil.copy2(__file__,args.out/'source_snapshot.py')
    lines=['# Actual 48D normalized PE: primitive covariance attribution','',
        'Five actual raw-RGB normalized_shape checkpoints execute their archived model implementation. Fitted query support is preserved, matching training. No fitting or temperature calibration occurs. Full and center-only inputs retain identical means/phases, features, labels, masks, resolution and checkpoint parameters.','',
        'This descriptive repeat uses the OLD decomposition camera pairs, not the 203 prospective fresh pairs. Both-side removal is the primary attribution; one-sided changes can misalign the still-fitted opposite spectral filter.','',
        '| parity | scene | intervention | full CE | changed CE | full−changed CE | seed/camera 95% CI |','|---|---|---|---:|---:|---:|---|']
    for alternate in [False,True]:
        for group in groups:
            for condition in CONDITIONS[1:]:
                d=differences[f'{int(alternate)}/{group}/full_minus_{condition}/cross_entropy']
                lines.append(f"| {int(alternate)} | {group} | {condition} | {means[f'{int(alternate)}/{group}/full']['cross_entropy']:.6f} | {means[f'{int(alternate)}/{group}/{condition}']['cross_entropy']:.6f} | {d['mean']:+.6f} | [{d['ci95_seed_camera'][0]:+.6f},{d['ci95_seed_camera'][1]:+.6f}] |")
    lines += ['', 'Negative full-minus-center CE favors retaining weighted internal primitive covariance; positive favors removing it. These frozen interventions are not retrained structural comparisons. Their scene-stratified camera intervals concern these reconstructed scenes, not a population of scenes. A positive old isometric/world-kernel result cannot substitute for this actual normalized-PE attribution, and this audit does not prove 3DGS-unique representational value over all point-cloud estimators.','',
        'Selected validation digests and metrics are replayed. The old camera pairs had no historical test probability archive for these development-trained checkpoints, so full-test verification is independent repeated-forward equality plus archived implementation/checkpoint hashes. Probabilities, row metrics, every decomposition/original NPZ hash and projection replay errors are retained.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in differences.items() if '/full_minus_both_centeronly/cross_entropy' in k},indent=2),flush=True)


if __name__=='__main__':main()
