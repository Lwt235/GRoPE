"""Exploratory real-GS scope audit; no fresh/legacy test split is opened.

Unlabeled individual-primitive operator diagnostics are deliberately separate
from regional matching scored against independently observed SfM tracks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.real_support_data import DEFAULT_SOURCES, load_scene, project_gaussian
from models.gaussian_measure_pe import gaussian_measure_log_bias
from models.gaussian_quadratic_pe import quadratic_features


ROOT = Path(__file__).resolve().parents[1]
SCENES = ('teatime', 'waldo_kitchen')
UNIT = 24.
METHODS = ('center_rbf', 'center_dispersion', 'trace_full', 'full')
SOURCES = ('experiments/round7_primitive_audit.py', 'experiments/real_support_data.py',
           'models/gaussian_measure_pe.py', 'models/gaussian_quadratic_pe.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_hash(hashes, path):
    hashes[str(path)] = sha(path)
    return hashes[str(path)]


def symmetric(c):
    return (c + c.swapaxes(-1, -2))*.5


def full_score(qm, qc, km, kc, width):
    args = [torch.as_tensor(x, dtype=torch.float64) for x in (qm, qc, km, kc)]
    return gaussian_measure_log_bias(*args, key_mass=1., bandwidth=width).numpy()


def independent_score(qm, qc, km, kc, width):
    s = symmetric(qc[:, None] + kc[None]) + width**2*np.eye(2)
    delta = qm[:, None]-km[None]
    logdet = np.linalg.slogdet(s)
    if (logdet.sign <= 0).any(): raise ValueError('non-SPD effective support')
    mahal = (delta*np.linalg.solve(s, delta[..., None])[..., 0]).sum(-1)
    return -.5*(mahal+logdet.logabsdet)-np.log(2*np.pi)


def logprob(score):
    offset = score.max(-1, keepdims=True)
    return score-offset-np.log(np.exp(score-offset).sum(-1, keepdims=True))


def lift_score(qm, qc, km, common_kc, width):
    q,k = quadratic_features(*(torch.as_tensor(x,dtype=torch.float64)
        for x in (qm,qc,km,common_kc)), width)
    return (q@k.T).numpy()


def covariance_statistics(c):
    e = np.linalg.eigvalsh(symmetric(c))
    traces = np.trace(c,axis1=-2,axis2=-1)
    return dict(tokens=len(c), trace_quantiles=np.quantile(traces,[0,.25,.5,.75,1]).tolist(),
        eigenvalue_min=float(e.min()),
        axis_ratio_quantiles=np.quantile(np.sqrt(np.maximum(e[:,-1],0)/np.maximum(e[:,0],1e-30)),[.25,.5,.75,1]).tolist(),
        relative_deviation_from_common_mean=float(np.linalg.norm(c-c.mean(0))/np.maximum(np.linalg.norm(c),1e-30)))


def operator_diagnostic(qm,qc,km,kc,width,ids):
    exact = full_score(qm,qc,km,kc,width)
    independent = independent_score(qm,qc,km,kc,width)
    exact_lp = logprob(exact)
    common = symmetric(kc.mean(0))
    approximate = lift_score(qm,qc,km,common,width)
    approximate_lp = logprob(approximate)
    common_exact = full_score(qm,qc,km,np.broadcast_to(common,kc.shape).copy(),width)
    common_error = float(np.abs(np.exp(logprob(common_exact))-np.exp(approximate_lp)).max())
    p, a = np.exp(exact_lp),np.exp(approximate_lp)
    centered = exact-exact.mean(-1,keepdims=True)
    singular = np.linalg.svd(centered,compute_uv=False)
    residual = float(np.linalg.norm(singular[5:])/np.maximum(np.linalg.norm(singular),1e-30))
    # Fixed bounded value probe, NOT a matching label or task prediction target.
    values = np.sin(np.asarray(ids)[:,None]*np.array([[.01,.023,.047,.071]]))
    return dict(bandwidth=width, full_pair_reference_max_probability_error=float(
        np.abs(np.exp(exact_lp)-np.exp(logprob(independent))).max()),
        common_covariance_counterfactual_max_probability_error=common_error,
        mean_key_covariance_fiveD_approximation=dict(
            maximum_probability_difference=float(np.abs(p-a).max()),
            mean_probability_total_variation=float(np.abs(p-a).sum(-1).mean()/2),
            mean_KL_exact_to_approx=float((p*(exact_lp-approximate_lp)).sum(-1).mean()),
            argmax_disagreements=int((p.argmax(-1)!=a.argmax(-1)).sum()),
            value_probe_max_difference=float(np.abs(p@values-a@values).max())),
        row_centered_logkernel=dict(singular_values=singular.tolist(),
            rank_relative_1e_8=int((singular>max(singular[0]*1e-8,1e-10)).sum()),
            rank_relative_1e_6=int((singular>max(singular[0]*1e-6,1e-10)).sum()),
            best_rank5_relative_frobenius_residual=residual))


def visible_ids(geometry,frame):
    means = geometry['means']; R,t = np.asarray(frame['R']),np.asarray(frame['t'])
    camera = means@R.T+t
    z = camera[:,2]
    pixels = (camera@frame['K'].T)[:,:2]/np.maximum(z[:,None],1e-8)
    valid = np.isfinite(camera).all(-1)&(z>1e-6)&(geometry['opacities'].reshape(-1)>=.3)
    valid &= (pixels[:,0]>=-.5)&(pixels[:,0]<191.5)&(pixels[:,1]>=-.5)&(pixels[:,1]<143.5)
    ids = np.flatnonzero(valid)
    xy = np.floor(pixels[ids]+.5).astype(np.int64)
    linear = xy[:,1]*192+xy[:,0]
    buffer = np.full(192*144,np.inf)
    np.minimum.at(buffer,linear,z[ids])
    return ids[z[ids]<=buffer[linear]*1.015]


def subsample(ids,count=64):
    return ids[np.linspace(0,len(ids)-1,min(count,len(ids)),dtype=np.int64)]


def covariance_from_rs(scales,quaternions):
    q = quaternions/np.linalg.norm(quaternions,axis=-1,keepdims=True)
    w,x,y,z = q.T
    rotation = np.stack((
        np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),-1),
        np.stack((2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),-1),
        np.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1)),-2)
    return (rotation*scales[:,None,:]**2)@rotation.swapaxes(-1,-2)


def primitive_diagnostic(scene, manifest, out, hashes):
    # This camera pair is fixed by manifest order before reading geometry.
    pair = next(p for p in manifest['pairs'] if p['split']=='validation')
    source = Path(DEFAULT_SOURCES[scene])
    model,frames,_,geometry,dataset,normalization = load_scene(source)
    del model
    for path in (source/'gaussians.npz',source/'source_model.json',source/'run_status.json',
                 source/'training_scene.json',dataset): record_hash(hashes,path)
    qframe,kframe = frames[pair['q_id']],frames[pair['k_id']]
    qids = visible_ids(geometry,qframe); qids=qids[qids%2==0]
    kids = visible_ids(geometry,kframe); kids=kids[kids%2==1]
    # Same query image frame; only geometric projection validity limits the bank.
    kp,_,front = project_gaussian(geometry['means'][kids],geometry['covariances'][kids],
        qframe['K'],np.asarray(qframe['R']),np.asarray(qframe['t']))
    inside = front&(kp[:,0]>=-.5)&(kp[:,0]<191.5)&(kp[:,1]>=-.5)&(kp[:,1]<143.5)
    kids=kids[inside]
    qids,kids=subsample(qids),subsample(kids)
    if min(len(qids),len(kids))<8 or np.intersect1d(qids,kids).size:
        raise ValueError('insufficient deterministic disjoint primitive sample')
    projected=[]
    for ids in (qids,kids):
        mean,cov,front=project_gaussian(geometry['means'][ids],geometry['covariances'][ids],
            qframe['K'],np.asarray(qframe['R']),np.asarray(qframe['t']))
        if not front.all():raise ValueError('sample projection invalid')
        projected.extend([mean/UNIT,symmetric(cov)/UNIT**2])
    allids=np.concatenate([qids,kids])
    with np.load(source/'gaussians.npz') as z:
        scales=z['scales'][allids].astype(np.float64)
        quaternions=z['quaternions'][allids].astype(np.float64)
    reconstructed=covariance_from_rs(scales,quaternions)
    exported=geometry['covariances'][allids].astype(np.float64)
    relative=np.linalg.norm(reconstructed-exported,axis=(-2,-1))/np.maximum(np.linalg.norm(exported,axis=(-2,-1)),1e-30)
    if relative.max()>1e-4:raise ValueError('exported covariance does not match R/S convention')
    dest=out/'primitives'/f'{scene}.npz';dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,query_ids=qids,key_ids=kids,qmean=projected[0],qcov=projected[1],
        kmean=projected[2],kcov=projected[3],scales=scales,quaternions_wxyz=quaternions,
        exported_covariance=exported,reconstructed_covariance=reconstructed)
    result=dict(scene=scene,q_id=pair['q_id'],k_id=pair['k_id'],query_name=qframe['name'],key_name=kframe['name'],
        primitive_count=len(geometry['means']),selected_query_ids=qids.tolist(),selected_key_ids=kids.tolist(),
        selection='first validation pair; opacity>=.3; native opaque-center zbuffer1.5%; disjoint parity; uniformly spaced sorted visible IDs; no covariance/label selection',
        no_primitive_correspondence_labels=True,normalization=normalization,
        rs_covariance_max_relative_error=float(relative.max()),query_covariance=covariance_statistics(projected[1]),
        key_covariance=covariance_statistics(projected[3]),sample_npz_sha256=sha(dest),
        diagnostics=[operator_diagnostic(*projected,width,kids) for width in (.5,1/UNIT)])
    del geometry
    return result


def row_metrics(logp,y):
    ce=-(y*logp).sum(-1)
    entropy=-(y*np.log(np.maximum(y,1e-300))).sum(-1)
    p=logp.argmax(-1)
    mass=y[np.arange(len(y)),p]
    return dict(ce=ce,kl=ce-entropy,top1_target_mass=mass,
        dominant_region_accuracy=(mass>=y.max(-1)-1e-8).astype(float))


def region_audit(scene,manifest,out,hashes,width):
    rows=[];diagnostics=[]
    for pair in manifest['pairs']:
        if pair['split']!='validation':continue
        original=ROOT/'results/round3_real_support_data_adjacent'/scene/pair['file']
        component=ROOT/'results/round3_cov_decomposition/development/pairs'/scene/pair['file']
        if record_hash(hashes,original)!=pair['npz_sha256']:raise ValueError('original NPZ hash mismatch')
        record_hash(hashes,component)
        with np.load(original) as z:
            # RGB is intentionally not scored in this geometry-only mechanism test.
            fields=['soft_labels','label_valid','K_query','R_query','t_query']
            fields += [s+'_'+f+suffix for s in ('q','k') for f in ('mean','valid','projected_cov') for suffix in ('','_alt')]
            source={f:z[f] for f in fields}
        with np.load(component) as z:
            part={f:z[f] for f in z.files}
        if not np.array_equal(part['soft_labels'],source['soft_labels']) or not np.array_equal(part['label_valid'],source['label_valid']):
            raise ValueError('component labels changed')
        for alternate in (False,True):
            suffix='_alt' if alternate else ''
            projections={};projection_error=[]
            for side in ('q','k'):
                if not np.array_equal(part[side+'_valid'+suffix],source[side+'_valid'+suffix]):raise ValueError('validity mismatch')
                mean=source[side+'_mean'+suffix].astype(np.float64)
                np.testing.assert_allclose(mean,part[side+'_mean'+suffix],rtol=1e-6,atol=1e-7)
                raw=part[side+'_raw_center'+suffix]+part[side+'_raw_internal'+suffix]
                np.testing.assert_array_equal(raw,part[side+'_raw_total'+suffix])
                for name in ('full','centeronly'):
                    xy,cov,_=project_gaussian(mean,part[side+'_'+name+suffix],source['K_query'],source['R_query'],source['t_query'])
                    projections[side+'mean']=xy/UNIT
                    projections[side+'_'+name]=symmetric(cov)/UNIT**2
                valid=source[side+'_valid'+suffix].astype(bool)
                ref=source[side+'_projected_cov'+suffix][valid]/UNIT**2
                error=np.linalg.norm(projections[side+'_full'][valid]-ref,axis=(-2,-1))/np.maximum(np.linalg.norm(ref,axis=(-2,-1)),1e-20)
                projection_error.append(float(error.max()) if len(error) else 0.)
            kv=source['k_valid'+suffix].astype(bool)
            target=source['soft_labels'].astype(np.float64)
            retained=target[:,kv].sum(-1)
            qv=source['q_valid'+suffix].astype(bool)&source['label_valid'].astype(bool)&(retained>0)
            qi,ki=np.flatnonzero(qv),np.flatnonzero(kv)
            if not len(qi) or not len(ki):continue
            y=target[np.ix_(qi,ki)]/retained[qi,None]
            qm,km=projections['qmean'][qi],projections['kmean'][ki]
            qc,kc=projections['q_full'][qi],projections['k_full'][ki]
            probs=[]
            for method in METHODS:
                if method=='center_rbf':cq,ck=qc*0,kc*0
                elif method=='center_dispersion':cq,ck=projections['q_centeronly'][qi],projections['k_centeronly'][ki]
                elif method=='trace_full':
                    cq=np.trace(qc,axis1=-2,axis2=-1)[:,None,None]/2*np.eye(2)
                    ck=np.trace(kc,axis1=-2,axis2=-1)[:,None,None]/2*np.eye(2)
                else:cq,ck=qc,kc
                lp=logprob(full_score(qm,cq,km,ck,width));probs.append(np.exp(lp))
                metrics=row_metrics(lp,y)
                for i,region in enumerate(qi):
                    for mask in ('conditional','strict95'):
                        if mask=='strict95' and retained[region]<.95:continue
                        rows.append(dict(scene=scene,q_id=pair['q_id'],k_id=pair['k_id'],file=pair['file'],
                            alternate=alternate,query_region=int(region),method=method,mask=mask,
                            retained_target_mass=float(retained[region]),**{k:float(v[i]) for k,v in metrics.items()}))
            folder=out/'regional_probabilities'/scene;folder.mkdir(parents=True,exist_ok=True)
            dest=folder/(Path(pair['file']).stem+f'_alt{int(alternate)}.npz')
            np.savez_compressed(dest,methods=np.array(METHODS),probability=np.stack(probs),targets=y,
                query_indices=qi,key_indices=ki,retained=retained[qi])
            diagnostics.append(dict(scene=scene,file=pair['file'],alternate=alternate,q_id=pair['q_id'],
                k_id=pair['k_id'],valid_rows=len(qi),valid_keys=len(ki),projection_relative_error_max=max(projection_error),
                key_covariance=covariance_statistics(kc),output_sha256=sha(dest),
                operator=operator_diagnostic(qm,qc,km,kc,width,ki)))
    return rows,diagnostics


def summaries(rows):
    result={}
    for alternate in (False,True):
        for mask in ('conditional','strict95'):
            for scene in ('all',)+SCENES:
                group=[r for r in rows if r['alternate']==alternate and r['mask']==mask and (scene=='all' or r['scene']==scene)]
                if not group:continue
                key=f'alt{int(alternate)}/{mask}/{scene}'
                result[key]={m:dict(rows=sum(r['method']==m for r in group),
                    **{metric:float(np.mean([r[metric] for r in group if r['method']==m])) for metric in ('ce','kl','top1_target_mass','dominant_region_accuracy')}) for m in METHODS}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/round7_primitive_audit')
    args=p.parse_args();torch.set_num_threads(1);torch.set_grad_enabled(False)
    args.out.mkdir(parents=True,exist_ok=False)
    hashes={};began=time.time()
    selection=ROOT/'results/round3_real_geometry_dev/summary.json';record_hash(hashes,selection)
    old=json.loads(selection.read_text())['selected']['projected_center_rbf/normalise_kernel']
    width=float(old['bandwidth'])
    if width!=.5 or old['temperature']!=1.:raise ValueError('archived old development operating point changed')
    sourcehash={f:sha(ROOT/f) for f in SOURCES}
    for file in SOURCES:
        dest=args.out/'source'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,dest)
    protocol=dict(scope='old Tea/Waldo validation only; exploratory mechanism audit, not independent confirmation',
        regional_source='results/round3_real_support_data_adjacent',methods=METHODS,
        bandwidth=width,coordinate_unit_pixels=UNIT,bandwidth_origin=str(selection),old_selected=old,
        shared='same fitted means, original masks, independent SfM targets, unit key mass, bandwidth and no visual content term',
        covariance='center_dispersion retains regional GS-center scatter; full adds weighted optimized primitive internal R/S; trace_full preserves total trace but removes anisotropy',
        labels='observed common SfM track region counts; >=4 tracks; conditional and >=95% retained target mass; never Gaussian-generated',
        primary_mask='strict95',parities=['even-query/odd-key','odd-query/even-key'],
        primitive_label_scope='none: individual primitive sample tests operator identities only',
        primitive_widths=[.5,1/UNIT],primitive_widths_reason='archived region blur and a fixed one-pixel mechanism blur, not metric selected',
        source_sha256=sourcehash,fresh_round7_test_read=False,old_test_read=False,parameters_fitted=0)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    primitives=[];rows=[];diagnostics=[]
    for scene in SCENES:
        path=ROOT/'results/round3_real_support_data_adjacent'/scene/'manifest.json'
        record_hash(hashes,path);manifest=json.loads(path.read_text())
        primitives.append(primitive_diagnostic(scene,manifest,args.out,hashes))
        print('PRIMITIVES',scene,flush=True)
        r,d=region_audit(scene,manifest,args.out,hashes,width);rows.extend(r);diagnostics.extend(d)
        print('REGIONS',scene,len(r),'rows',flush=True)
    if sourcehash!={f:sha(ROOT/f) for f in SOURCES}:raise ValueError('source changed during audit')
    output=dict(protocol=protocol,primitives=primitives,regional_summary=summaries(rows),
        regional_loading_audit=diagnostics,input_sha256=hashes,source_sha256=sourcehash,seconds=time.time()-began,
        interpretation='optimized-primitive identity diagnostics and region-level SfM matching are distinct; no primitive accuracy or new-scene claim')
    (args.out/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    (args.out/'rows.json').write_text(json.dumps(rows,separators=(',',':'),allow_nan=False))
    print('DONE',round(time.time()-began,2),'seconds',flush=True)


if __name__=='__main__':main()
