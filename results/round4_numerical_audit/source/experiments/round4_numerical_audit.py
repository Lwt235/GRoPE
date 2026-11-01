"""CPU-only accuracy, gradient and speed audit of an opt-in exact PE fast path.

This does not train or modify a model. Only development cameras are profiled;
no task labels, held-out cameras or task scores are accessed. Thresholds select
the same real-valued normalized Gaussian map's numerical evaluation path.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.gaussian_pe import GaussianPEMatcher,normalized_rotary
from models.gaussian_pe_fast import (CONTENT_SCALE_MAX,CONTENT_SCALE_MIN,
    MIN_FILTERED_NORM_RATIO,PHASE_ABSOLUTE_SUM_LIMIT,VARIANCE_ABSOLUTE_SUM_LIMIT,
    _ordinary_float32,eligibility,normalized_rotary_fast)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(value):
    value=np.asarray(value,np.float64)
    return dict(zip(('min','p50','p90','p95','p99','max'),np.quantile(value,[0,.5,.9,.95,.99,1]).tolist()))


def protocol():
    return dict(stage='numerical implementation audit, not a new kernel or efficacy experiment',
        reference='unchanged models.gaussian_pe.normalized_rotary float64 intermediates',
        candidate='explicit opt-in models.gaussian_pe_fast.normalized_rotary_fast',
        map='norm(c) * ExpectedRoPE(c,mu,C) / norm(ExpectedRoPE(c,mu,C)) in real arithmetic',
        no_attenuation_floor=True,no_kernel_changed=True,current_model_not_patched=True,
        eligibility=dict(dtype='all inputs float32',dimension=2,matching_leading_shapes=True,
            finite_inputs=True,nonnegative_evaluated_variances=True,
            max_sum_absolute_quadratic_terms=VARIANCE_ABSOLUTE_SUM_LIMIT,
            max_sum_absolute_phase_terms=PHASE_ABSOLUTE_SUM_LIMIT,
            content_max_abs_min=CONTENT_SCALE_MIN,content_max_abs_max=CONTENT_SCALE_MAX,
            min_filtered_norm_ratio=MIN_FILTERED_NORM_RATIO),
        fallback='existing float64 helper on every ineligible/unsupported token, gathered separately from fast autograd graph',
        datasets=dict(scenes=['teatime','waldo_kitchen'],splits=['train','validation'],partitions=['primary','alt'],
            geometry='original real-support projected key covariance /24^2 +1e-6 I and .25^2 resolution; query fixed rectangle covariance +.25^2 I',
            content='actual RGB through a fixed untrained seed11 encoder, unit-normalized; no task training or labels'),
        tolerances=dict(value_max_abs=5e-6,gradient_atol=5e-5,gradient_rtol=5e-4),
        limits=['Float32 numerically equivalent within stated tolerances, not bitwise float64 equivalence.',
                'Eligibility constants are numerical dispatch bounds, not fitted geometry/kernel hyperparameters.',
                'CPU timing includes eligibility/dispatch; CUDA mask synchronization may erase any speed gain.',
                'The original extreme zero-channel derivative may be mathematically unrepresentable; fallback cannot fix that mathematical range limit.'])


def real_development_profile(root):
    torch.manual_seed(11)
    encoder=GaussianPEMatcher('normalized_shape',count=16,resolution=.25).encoder.eval()
    frequency=GaussianPEMatcher('normalized_shape',count=16,resolution=.25).freq2
    parts=[];hashes={};identities=[]
    with torch.no_grad():
        for scene in ['teatime','waldo_kitchen']:
            folder=root/scene;manifest_path=folder/'manifest.json'
            manifest=json.loads(manifest_path.read_text());hashes[str(manifest_path)]=sha256(manifest_path)
            for pair in manifest['pairs']:
                if pair['split'] not in ['train','validation']:continue
                path=folder/pair['file'];hashes[str(path)]=sha256(path)
                if hashes[str(path)]!=pair['npz_sha256']:raise ValueError('source pair changed')
                with np.load(path,allow_pickle=False) as data:
                    # Labels/counts and test files are not read.
                    for suffix in ['', '_alt']:
                        for side in ['q','k']:
                            valid=data[side+'_valid'+suffix].astype(bool)
                            rgb=torch.as_tensor(data[side+'_features'][valid],dtype=torch.float32)
                            content=F.normalize(encoder(rgb),dim=-1)
                            if side=='q':
                                mean=data['q_pixel_mean'][valid]/24
                                covariance=data['q_pixel_cov'][valid]/24**2+np.eye(2)*.25**2
                            else:
                                mean=data['k_projected_mean'+suffix][valid]/24
                                covariance=data['k_projected_cov'+suffix][valid]/24**2+np.eye(2)*(.25**2+1e-6)
                            parts.append((content,torch.as_tensor(mean,dtype=torch.float32),torch.as_tensor(covariance,dtype=torch.float32)))
                            identities.extend([dict(scene=scene,side=side,split=pair['split'],alternate=bool(suffix)) for _ in range(int(valid.sum()))])
    tensors=[torch.cat([p[i] for p in parts]) for i in range(3)]
    h=hashlib.sha256()
    for name,value in encoder.state_dict().items():h.update(name.encode());h.update(value.numpy().tobytes())
    return (*tensors,frequency),identities,hashes,h.hexdigest()


def accuracy(inputs):
    reference=normalized_rotary(*inputs)
    candidate,info=normalized_rotary_fast(*inputs,return_diagnostics=True)
    error=(reference-candidate).abs()
    value=dict(max_absolute_error=float(error.max()),rms_error=float(error.square().mean().sqrt()),
               norm_max_absolute_error=float((reference.norm(dim=-1)-candidate.norm(dim=-1)).abs().max()),dispatch=info)
    if value['max_absolute_error']>protocol()['tolerances']['value_max_abs']:raise ValueError('fast-path forward error exceeds declared tolerance')
    # A fixed content/geometry sample audits gradients without task labels.
    index=torch.linspace(0,len(inputs[0])-1,min(2048,len(inputs[0]))).long()
    subset=[x[index].clone() for x in inputs[:3]]+[inputs[3].clone()]
    generator=torch.Generator().manual_seed(38399)
    weight=torch.randn(subset[0].shape,generator=generator)
    derivatives=[]
    for function in [normalized_rotary,normalized_rotary_fast]:
        args=[x.clone().requires_grad_() for x in subset]
        result=function(*args)
        derivatives.append(torch.autograd.grad((result*weight).sum(),args))
    gradient={}
    for name,a,b in zip(['content','mean','covariance','frequency'],*derivatives):
        error=(a-b).abs()
        gradient[name]=dict(max_absolute_error=float(error.max()),rms_error=float(error.square().mean().sqrt()),
            reference_max_absolute_value=float(a.abs().max()),finite=bool(torch.isfinite(b).all()))
        torch.testing.assert_close(b,a,atol=protocol()['tolerances']['gradient_atol'],rtol=protocol()['tolerances']['gradient_rtol'])
    return value,gradient


def time_function(function,inputs,backward=False,repeats=5,iterations=25):
    weight=torch.linspace(-1,1,inputs[0].numel()).reshape_as(inputs[0])
    def operation():
        if backward:
            args=[x.detach().requires_grad_() for x in inputs[:3]]+[inputs[3]]
            out=function(*args)
            torch.autograd.grad((out*weight).sum(),args[:3])
        else:
            with torch.no_grad():function(*inputs)
    for _ in range(8):operation()
    values=[]
    for _ in range(repeats):
        started=time.perf_counter()
        for __ in range(iterations):operation()
        values.append((time.perf_counter()-started)/iterations*1000)
    return dict(median_ms=statistics.median(values),minimum_ms=min(values),maximum_ms=max(values),repetitions_ms=values)


def timing(real,repeats,iterations):
    f,m,c,w=real
    g=torch.Generator().manual_seed(133)
    samples={}
    index=torch.arange(16*48)%len(f)
    samples['real_mixed_16x48']=(f[index].reshape(16,48,-1),m[index].reshape(16,48,2),c[index].reshape(16,48,2,2),w)
    qcov=torch.eye(2).expand(16,48,2,2)*(1/12+.25**2)
    samples['fixed_query_16x48']=(torch.randn(16,48,48,generator=g),torch.rand(16,48,2,generator=g)*5,qcov,w)
    index=torch.linspace(0,len(f)-1,16*48).long()
    samples['real_distributed_16x48']=(f[index].reshape(16,48,-1),m[index].reshape(16,48,2),c[index].reshape(16,48,2,2),w)
    samples['extreme_fallback_16x48']=(torch.randn(16,48,48,generator=g),torch.rand(16,48,2,generator=g),torch.eye(2).expand(16,48,2,2)*1000,w)
    output={}
    for name,inputs in samples.items():
        mask=eligibility(*inputs)
        item=dict(tokens=inputs[0].numel()//48,eligible_fraction=float(mask.float().mean()),timings={})
        functions=[('reference',normalized_rotary),('dispatched_fast',normalized_rotary_fast)]
        if bool(mask.all()):functions.append(('ordinary_kernel_without_dispatch',_ordinary_float32))
        for backward in [False,True]:
            mode='forward_backward' if backward else 'forward'
            item['timings'][mode]={label:time_function(function,inputs,backward,repeats,iterations) for label,function in functions}
            t=item['timings'][mode]
            t['dispatch_speedup']=t['reference']['median_ms']/t['dispatched_fast']['median_ms']
        output[name]=item
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('results/round3_real_support_data_adjacent'))
    p.add_argument('--out',type=Path,default=Path('results/round4_numerical_audit'))
    p.add_argument('--timing-repeats',type=int,default=5)
    p.add_argument('--timing-iterations',type=int,default=25)
    args=p.parse_args();torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'summary.json').exists():raise ValueError('refusing to overwrite completed numerical audit')
    source_paths=[Path('experiments/round4_numerical_audit.py'),Path('models/gaussian_pe.py'),Path('models/gaussian_pe_fast.py'),Path('models/geometry.py'),Path('tests/test_round4_numerics.py')]
    hashes={str(x):sha256(x) for x in source_paths}
    hashes['normalized_rotary_function_source']=hashlib.sha256(inspect.getsource(normalized_rotary).encode()).hexdigest()
    (args.out/'frozen_protocol.json').write_text(json.dumps(protocol(),indent=2))
    (args.out/'execution_source_sha256.json').write_text(json.dumps(hashes,indent=2))
    inputs,identities,data_hashes,encoder_hash=real_development_profile(args.data)
    f,m,c,w=inputs
    with torch.no_grad():
        variance=torch.einsum('fd,nde,fe->nf',w,c,w)
        spread=(variance.amax(-1)-variance.amin(-1))/2
        absolute=torch.einsum('fd,nde,fe->nf',w.abs(),c.abs(),w.abs()).amax(-1)
        phase=(m.abs()@w.abs().T).amax(-1)
        mask=eligibility(*inputs)
    profiles={}
    for scene in ['all','teatime','waldo_kitchen']:
        for side in ['all','q','k']:
            take=np.array([(scene=='all' or r['scene']==scene) and (side=='all' or r['side']==side) for r in identities])
            profiles[f'{scene}/{side}']=dict(tokens=int(take.sum()),eligible_fraction=float(mask[take].float().mean()),
                log_amplitude_spread=quantiles(spread[take]),absolute_variance_sum=quantiles(absolute[take]),absolute_phase_sum=quantiles(phase[take]))
    value,gradient=accuracy(inputs)
    print(json.dumps(dict(tokens=len(f),value=value,gradient=gradient),indent=2),flush=True)
    speed=timing(inputs,args.timing_repeats,args.timing_iterations)
    for path in source_paths:
        if sha256(path)!=hashes[str(path)]:raise ValueError(f'numerical audit source changed during execution: {path}')
    output=dict(protocol=protocol(),source_sha256=hashes,data_sha256=data_hashes,encoder_parameter_sha256=encoder_hash,
        profiles=profiles,value_accuracy=value,gradient_accuracy=gradient,speed=speed,
        environment=dict(pytorch=torch.__version__,python=platform.python_version(),platform=platform.platform(),
                         cpu_threads=torch.get_num_threads(),load_average=os.getloadavg(),cuda_benchmarked=False))
    (args.out/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    snapshot=args.out/'source'
    for path in source_paths:
        target=snapshot/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    lines=['# Normalized Gaussian PE numerical audit','',
        'The frozen float64 helper was not modified. A separate, opt-in float32 path computes the same real-valued normalized expected-RoPE map and falls back per token when an explicit numerical predicate fails. This is an implementation audit, not efficacy evidence.','',
        f"Real development tokens: {len(f):,}; eligible fraction: {float(mask.float().mean()):.4%}. No task labels or held-out pair NPZs were read. Fixed untrained seed11 RGB features profile content. More details and every source/data hash are in summary.json.",'',
        '| Workload | eligible | forward speedup | forward+backward speedup |','|---|---:|---:|---:|']
    for name,item in speed.items():lines.append(f"| {name} | {item['eligible_fraction']:.2%} | {item['timings']['forward']['dispatch_speedup']:.3f}x | {item['timings']['forward_backward']['dispatch_speedup']:.3f}x |")
    lines += ['',f"Maximum observed forward absolute error: {value['max_absolute_error']:.3g}; RMS: {value['rms_error']:.3g}. Content, mean, covariance and frequency gradients pass the declared float32 tolerances; exact values are in JSON.",'',
        'Float32 outputs are not bitwise equal to float64. No covariance, relative Gaussian amplitude, or content is floored to make the fast path pass. Eligibility controls only numerical dispatch. It bounds sum-absolute quadratic terms by 32, sum-absolute phase terms by 64, content scale to [2^-32,2^32], and the observed normalized-filter conditioning by 16. Every other token is evaluated by the unchanged reference. Ordinary zero-block content gradients are retained.', '',
        'CPU timings include runtime eligibility and mixed-token dispatch. The ordinary kernel timing without dispatch is listed separately when all tokens qualify. CPU load can influence wall-clock measurements; GPU timings are unmeasured, and CUDA synchronization in the dispatcher may remove the gain. Do not replace a frozen experiment implementation without rerunning its source audit and confirming numerical tolerance requirements.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:{mode:value['dispatch_speedup'] for mode,value in row['timings'].items()} for k,row in speed.items()},indent=2),flush=True)


if __name__=='__main__':main()
