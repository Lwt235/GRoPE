"""R63: fixed-weight fit/val interventions, parameter traces and independent sums."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import gc
import time
import unittest
from pathlib import Path
import numpy as np
import torch
from experiments.round62_common import (ROOT, OUT as BASE, SOURCES as PARENT_SOURCES,
    SEEDS, METRICS, checked as parent_checked, load as parent_load, read, write, sha,
    run_path, pixels)
from experiments.round55_train import setup, tensor_hash, subset, metric_rows, macro, query_ce
from experiments.round62_audit import independent_metrics
from experiments.round5_rgbd_data import load_sensor_depth, resize_pixel_coordinates, sample_nearest_depth
from models.round62_rgb_relations import FLOOR, POINT_VARIANCE
from models.round63_interventions import InterventionMatcher, INTERVENTIONS

OUT = ROOT/'results/round63_rgb_diagnostic/attempt1'
PREP = ROOT/'results/round63_rgb_diagnostic/preparation/availability.json'
PLAN = ROOT/'docs/round63_rgb_diagnostic_plan.md'
ROLES = ('predicted_full', 'predicted_point', 'observed_full')
CONFIG = dict(batch=8, fit_pairs=48, val_pairs=64, groups=54, training_updates=0,
    coverage_samples=16, saturation=.95, cholesky_jitter=1e-12, coordinate_tolerance=1e-8,
    old_eval_access=False, confirmation_access=False)
SOURCES = tuple(sorted(set(PARENT_SOURCES)|{
    'ROUND63_RGB_DIAGNOSTIC_PROTOCOL.md','docs/round63_rgb_diagnostic_plan.md',
    'models/round63_interventions.py','experiments/round63_diagnostic.py',
    'experiments/round63_queue.py','tests/test_round63_diagnostic.py'}))


def cases(role):
    return INTERVENTIONS if role == 'predicted_full' else ('baseline',)


def checked():
    parent_checked()
    lock = read(OUT/'lock.json')
    if sha(OUT/'lock.json') != (OUT/'lock.sha256').read_text().strip() or lock['config'] != CONFIG:
        raise ValueError('R63 scope/lock changed')
    for name, digest in lock['sources'].items():
        if sha(ROOT/name) != digest:raise ValueError('changed source '+name)
    for name, digest in lock['input_records'].items():
        if sha(name) != digest:raise ValueError('changed input identity '+name)
    return lock


def freeze():
    parent_checked()
    if (OUT/'lock.json').exists():raise ValueError('already frozen')
    prep = read(PREP)
    if prep['plan_sha256'] != sha(PLAN) or prep['parent_completion_sha256'] != sha(BASE/'completion.json'):
        raise ValueError('prepared scope changed')
    if not read(BASE/'completion.json')['completed']:raise ValueError('finish R62')
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_round63_diagnostic.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise ValueError('intervention tests failed')
    records = {str(p):sha(p) for p in (PREP, PLAN, BASE/'completion.json', BASE/'lock.json',
        BASE/'selected.json', BASE/'data/manifest.json')}
    for split in ('fit','val'):
        for ext in ('npz','json'):
            p = BASE/'data'/f'observed_{split}.{ext}';records[str(p)] = sha(p)
    for rows in prep['samples'].values():
        for r in rows:
            if sha(r['rgb_cache']) != r['rgb_sha256']:raise ValueError('RGB cache identity')
            records[r['rgb_cache']] = r['rgb_sha256']
    for r in prep['models']:
        path = Path(r['path']);run = read(path.with_name('run.json'))
        if sha(path) != r['sha256'] or sha(path.with_name('run.json')) != r['run_sha256']:
            raise ValueError('selected model identity')
        if sha(path.with_name('validation.npz')) != run['validation_sha256']:
            raise ValueError('R62 validation identity')
        records[str(path)] = r['sha256']
        records[str(path.with_name('run.json'))] = r['run_sha256']
        records[str(path.with_name('validation.npz'))] = run['validation_sha256']
    for r in prep['frames'].values():
        if sha(r['path']) != r['sha256']:raise ValueError('native depth identity')
        records[r['path']] = r['sha256']
    for name in SOURCES:
        p = OUT/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/name).read_bytes())
    write(OUT/'units.json',dict(passed=True,count=result.testsRun,unix=time.time()))
    write(OUT/'lock.json',dict(config=CONFIG,sources={n:sha(ROOT/n) for n in SOURCES},
        input_records=records,samples=prep['samples'],frames=prep['frames'],models=prep['models'],
        frozen_unix=time.time(),software=dict(torch=torch.__version__,numpy=np.__version__),
        fresh_confirmation=False,goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('FROZEN R63',sha(OUT/'lock.json'),len(SOURCES),'sources',flush=True)


def native_coverage(depth):
    centers = resize_pixel_coordinates(pixels(),(144,192),depth.shape)
    _, known = sample_nearest_depth(depth,centers)
    y,x = np.mgrid[:144,:192]
    samples = np.stack((x,y),-1).reshape(18,8,24,8,2).transpose(0,2,1,3,4).reshape(-1,2)
    _, patch = sample_nearest_depth(depth,resize_pixel_coordinates(samples,(144,192),depth.shape))
    return known,patch.reshape(432,64).sum(-1).astype(np.int16)


def prepare():
    lock = checked();setup();coverage = {}
    for key,r in lock['frames'].items():
        coverage[key] = native_coverage(load_sensor_depth(r['path']))
    for split in ('fit','val'):
        destination = OUT/'data'/f'{split}.npz'
        if destination.exists():raise ValueError('preserve prepared data')
        rows = lock['samples'][split];data,identities = parent_load(split,'cpu',observed=True)
        ix = np.array([r['global_index'] for r in rows])
        ids = [identities[i] for i in ix]
        if [(r['scene'],r['pair_id']) for r in ids] != [(r['scene'],r['pair_id']) for r in rows]:
            raise ValueError('declared sample identity')
        arrays = {k:v.numpy()[ix] for k,v in data.items()}
        for prefix,key in (('q','q_index'),('k','k_index')):
            c = [coverage[f"{r['scene']}:{r[key]}"] for r in rows]
            arrays[prefix+'native_valid'] = np.stack([x[0] for x in c])
            arrays[prefix+'patch_count'] = np.stack([x[1] for x in c])
        destination.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination,**arrays)
        write(destination.with_suffix('.json'),dict(ids=ids,split=split,count=len(ids),
            sha256=sha(destination),lock_sha256=sha(OUT/'lock.json'),analysis_masks_not_model_inputs=True))
        print('PREPARED',split,len(ids),flush=True)
        del data,arrays
    write(OUT/'data/prepared.json',dict(unique_frames=len(coverage),splits=dict(fit=48,val=64),
        old_eval_access=False,confirmation_access=False))


def load_data(split,device='cpu'):
    if split not in ('fit','val'):raise ValueError('diagnostic data scope')
    p = OUT/'data'/f'{split}.npz';meta = read(p.with_suffix('.json'))
    if meta['lock_sha256'] != sha(OUT/'lock.json') or sha(p) != meta['sha256']:
        raise ValueError('prepared input changed')
    with np.load(p,allow_pickle=False) as z:
        data = {k:torch.from_numpy(z[k]).to(device) for k in z.files}
    return data,meta['ids']


def probe():
    checked();device = setup(0)
    actual = float(torch.arange(128,device=device,dtype=torch.float32).sum())
    if actual != 8128:raise ValueError('CUDA probe')
    write(OUT/'cuda_probe.json',dict(device=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda=torch.version.cuda,sum=actual,available=torch.cuda.is_available(),unix=time.time()))
    print('CUDA READY',torch.cuda.get_device_name(),flush=True)


def prediction_path(role,seed,split,case):
    return OUT/'predictions'/role/f'{seed}_{split}_{case}.npz'


@torch.no_grad()
def run(seed):
    lock = checked();device = setup(0)
    if seed not in SEEDS:raise ValueError(seed)
    datasets = {s:load_data(s,device) for s in ('fit','val')}
    grid = torch.from_numpy(pixels()).to(device)
    for role in ROLES:
        record = next(r for r in lock['models'] if (r['role'],r['seed']) == (role,seed))
        state = torch.load(record['path'],map_location=device,weights_only=True)
        original_hash = tensor_hash(state)
        with np.load(Path(record['path']).with_name('validation.npz'),allow_pickle=False) as z:
            reference_logits = z['logits']
        reference_ce = read(Path(record['path']).with_name('run.json'))['val_ce']
        for split,(data,ids) in datasets.items():
            for case in cases(role):
                dest = prediction_path(role,seed,split,case)
                if dest.exists():raise ValueError('preserve diagnostic predictions')
                capture = case == 'baseline' and role != 'observed_full'
                model = InterventionMatcher(role,case,capture).to(device).eval().requires_grad_(False)
                model.load_state_dict(state,strict=True)
                if tensor_hash(model.state_dict()) != original_hash:raise ValueError('model loading changed weights')
                result = {k:[] for k in ('argmax','soft_xy','target_logp')};traces = {}
                max_difference = 0.;ce_sum = 0.;n_valid = 0
                torch.cuda.reset_peak_memory_stats();start_time = time.monotonic()
                for start in range(0,len(ids),CONFIG['batch']):
                    d = subset(data,slice(start,start+CONFIG['batch']))
                    logits = model(d)
                    if not torch.isfinite(logits).all():raise ValueError('nonfinite diagnostic logits')
                    result['argmax'].append(logits.argmax(-1).cpu().numpy().astype(np.int16))
                    result['soft_xy'].append((logits.softmax(-1)@grid).cpu().numpy())
                    result['target_logp'].append(logits.log_softmax(-1).gather(-1,d['key_indices'].clamp_min(0)).cpu().numpy())
                    if split == 'val' and case == 'baseline':
                        max_difference = max(max_difference,float(np.abs(logits.cpu().numpy()-reference_logits[start:start+len(logits)]).max()))
                        ce_sum += float(query_ce(logits,d)[d['gt_valid']].sum());n_valid += int(d['gt_valid'].sum())
                    if capture:
                        if len(model.captured) != 4:raise ValueError('two-layer geometry capture')
                        for t in model.captured:
                            for key in ('raw','mean','cov'):
                                traces.setdefault(f"layer{t['layer']}_{t['prefix']}{key}",[]).append(t[key])
                if tensor_hash(model.state_dict()) != original_hash:raise ValueError('diagnostic updated weights')
                validation = None
                if split == 'val' and case == 'baseline':
                    validation = dict(max_logit_difference=max_difference,ce=ce_sum/n_valid,
                        ce_error=abs(ce_sum/n_valid-reference_ce))
                    if max_difference > 1e-3 or validation['ce_error'] > 1e-5:
                        raise ValueError('original branch differs from R62')
                result = {k:np.concatenate(v) for k,v in result.items()}
                dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**result)
                rows,tails = metric_rows(result,data,ids)
                trace_path = None
                if capture:
                    trace_path = OUT/'geometry'/role/f'{seed}_{split}.npz'
                    trace_path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(trace_path,**{k:np.concatenate(v) for k,v in traces.items()})
                write(dest.with_suffix('.json'),dict(role=role,seed=seed,split=split,case=case,
                    rows=rows,metrics=macro(rows),scene_tails=tails,validation=validation,
                    checkpoint_sha256=record['sha256'],state_sha256=original_hash,weights_unchanged=True,
                    prediction_sha256=sha(dest),data_sha256=sha(OUT/'data'/f'{split}.npz'),
                    lock_sha256=sha(OUT/'lock.json'),geometry_path=None if trace_path is None else str(trace_path),
                    geometry_sha256=None if trace_path is None else sha(trace_path),seconds=time.monotonic()-start_time,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),confirmation_access=False))
                print('FORWARD',role,seed,split,case,flush=True)
                del model;gc.collect();torch.cuda.empty_cache()
        del state,reference_logits


def describe(values):
    x = np.asarray(values,dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all():raise ValueError('nonfinite statistic')
    if not len(x):return dict(n=0,mean=None,median=None,p10=None,p90=None,min=None,max=None)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),
        p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)),min=float(x.min()),max=float(x.max()))


def covariance_distance(predicted,observed):
    values,vectors = np.linalg.eigh(observed)
    if (values <= 0).any():raise ValueError('observed covariance not SPD')
    whitening = (vectors*(values**-.5)[...,None,:])@vectors.swapaxes(-1,-2)
    matrix = whitening@predicted@whitening
    values = np.linalg.eigvalsh((matrix+matrix.swapaxes(-1,-2))*.5)
    if (values <= 0).any():raise ValueError('relative covariance not SPD')
    return np.linalg.norm(np.log(values),axis=-1)


def covariance_range(observed):
    observed = np.asarray(observed,dtype=np.float64)
    moment = observed-FLOOR**2*np.eye(3);moment = (moment+moment.swapaxes(-1,-2))*.5
    smallest = np.linalg.eigvalsh(moment)[...,0]
    jitter = np.maximum(1e-12,1e-12-smallest)
    linear = np.linalg.cholesky(moment+jitter[...,None,None]*np.eye(3))
    base = np.array([.025,.030,.010]);low,high = base*np.exp(-2),base*np.exp(2)
    diagonal = np.diagonal(linear,axis1=-2,axis2=-1)
    ij = np.tril_indices(3,-1);lower = linear[...,ij[0],ij[1]]
    clipped = linear.copy();axis = np.arange(3)
    clipped[...,axis,axis] = np.clip(diagonal,low,high)
    clipped[...,ij[0],ij[1]] = np.clip(lower,-.03,.03)
    reconstructed = clipped@clipped.swapaxes(-1,-2)+FLOOR**2*np.eye(3)
    return dict(nonpositive_moment=(smallest<=0),jitter=jitter,
        diagonal_below=(diagonal < low-1e-8),diagonal_above=(diagonal > high+1e-8),
        lower_outside=(np.abs(lower) > .03+1e-8),
        any_coordinate_outside=((diagonal < low-1e-8)|(diagonal > high+1e-8)|(np.abs(lower) > .03+1e-8)).any(-1),
        relative_frobenius=np.linalg.norm(reconstructed-observed,axis=(-2,-1))/np.linalg.norm(observed,axis=(-2,-1)),
        log_spectrum_change=covariance_distance(reconstructed,observed))


def strata(data):
    native = np.concatenate([data[p+'native_valid'].numpy() for p in ('q','k')])
    count = np.concatenate([data[p+'patch_count'].numpy() for p in ('q','k')])
    return dict(all=np.ones_like(native,dtype=bool),native_center=native,
                patch16=count>=16,empty_patch=count==0)


def geometry_statistics(data,ids,trace):
    masks = strata(data);observed_mean = np.concatenate([data[p+'observed_mean'].numpy() for p in ('q','k')]).astype(float)
    observed_cov = np.concatenate([data[p+'observed_cov'].numpy() for p in ('q','k')]).astype(float)
    scene_ids = np.array([r['scene'] for r in ids]*2);rows = []
    for layer in range(2):
        mean,cov,raw = [np.concatenate([trace[f'layer{layer}_{p}{key}'] for p in ('q','k')]).astype(float)
                       for key in ('mean','cov','raw')]
        values = dict(center_distance_m=3*np.linalg.norm(mean-observed_mean,axis=-1),
            depth_abs_error_m=3*np.abs(mean[...,2]-observed_mean[...,2]),
            depth_signed_error_m=3*(mean[...,2]-observed_mean[...,2]),
            predicted_depth_m=3*mean[...,2],observed_depth_m=3*observed_mean[...,2],
            covariance_log_error=covariance_distance(cov,observed_cov))
        def summarize(selection):
            output = {}
            for name,mask in masks.items():
                take = mask & selection[:,None]
                item = {k:describe(v[take]) for k,v in values.items()}
                item['raw_coordinates'] = [describe(raw[...,j][take]) for j in range(raw.shape[-1])]
                item['diagonal_saturation_fraction'] = ([float((np.abs(np.tanh(raw[...,j][take]/2))>.95).mean())
                    if take.any() else None for j in range(1,4)] if raw.shape[-1]==7 else [])
                item['lower_saturation_fraction'] = ([float((np.abs(np.tanh(raw[...,j][take]))>.95).mean())
                    if take.any() else None for j in range(4,7)] if raw.shape[-1]==7 else [])
                output[name] = item
            return output
        rows.append(dict(layer=layer,strata=summarize(np.ones(len(scene_ids),dtype=bool)),
            by_scene={s:summarize(scene_ids==s) for s in sorted(set(scene_ids))}))
    return rows


def audit():
    checked();records=[];geometry=[];ranges={};maximum=0.;replay=[]
    datasets = {s:load_data(s) for s in ('fit','val')}
    for split,(data,ids) in datasets.items():
        observed = np.concatenate([data[p+'observed_cov'].numpy() for p in ('q','k')]).astype(float)
        arrays = covariance_range(observed);masks = strata(data)
        ranges[split] = {name:{k:([describe(v[...,i][mask]) for i in range(v.shape[-1])]
            if v.ndim==3 else describe(v[mask])) for k,v in arrays.items()} for name,mask in masks.items()}
        for role in ROLES:
            for seed in SEEDS:
                for case in cases(role):
                    p = prediction_path(role,seed,split,case);meta = read(p.with_suffix('.json'))
                    if sha(p)!=meta['prediction_sha256'] or not meta['weights_unchanged']:
                        raise ValueError('prediction/weight identity')
                    if meta['lock_sha256']!=sha(OUT/'lock.json') or meta['data_sha256']!=sha(OUT/'data'/f'{split}.npz'):
                        raise ValueError('source/data identity')
                    with np.load(p,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
                    rows,scenes,values = independent_metrics(pred,data,ids)
                    if [(r['scene'],r['pair_id'],r['n_valid']) for r in rows] != [(r['scene'],r['pair_id'],r['n_valid']) for r in meta['rows']]:
                        raise ValueError('query population')
                    error = max(abs(values[k]-meta['metrics'][k]) for k in METRICS)
                    maximum = max(maximum,error)
                    if error>1e-5:raise ValueError('metric discrepancy')
                    records.append(dict(role=role,seed=seed,split=split,case=case,metrics=values,by_scene=scenes,rows=rows))
                    if meta['validation'] is not None:replay.append(dict(role=role,seed=seed,**meta['validation']))
                    if meta['geometry_path'] is not None:
                        if sha(meta['geometry_path'])!=meta['geometry_sha256']:raise ValueError('trace identity')
                        with np.load(meta['geometry_path'],allow_pickle=False) as z:trace={k:z[k] for k in z.files}
                        geometry.append(dict(role=role,seed=seed,split=split,layers=geometry_statistics(data,ids,trace)))
                    print('AUDIT',role,seed,split,case,error,flush=True)
    if len(records)!=54 or len(replay)!=9 or len(geometry)!=12:raise ValueError('diagnostic scope')
    write(OUT/'analysis/summary.json',dict(records=records,geometry=geometry,observed_range=ranges,
        baseline_val_replay=replay,max_metric_error=maximum,groups=54,training_updates=0,
        distribution_shift_diagnostic=True,fresh_confirmation=False,goal_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','prepare','probe','run','audit'))
    parser.add_argument('--seed',type=int,choices=SEEDS);args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=='freeze':freeze()
    elif args.stage=='prepare':prepare()
    elif args.stage=='probe':probe()
    elif args.stage=='run':run(args.seed)
    else:audit()
