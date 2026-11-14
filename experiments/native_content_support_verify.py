"""Independent CPU-only G/P/S reductions and paired-place intervals from saved energies."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import collections, json, math, pathlib, resource, time
import numpy as np
import torch

start=time.monotonic();torch.set_num_threads(1)
out=pathlib.Path('/home/liwt/GRoPE/results/gaussian_memory/native_content_support1')
rows=json.loads((out/'token_measurements.json').read_text());summary=json.loads((out/'summary.json').read_text())
byimage=collections.defaultdict(list)
for r in rows:byimage[r['key']].append(r)
metrics=('CG_CP_pp','CG_CS_pp','CS_CP_pp');imageresults={};scalar_difference=0.
for key,rr in byimage.items():
    e=np.load(out/'images'/key/'energy.npy',mmap_mode='r')
    layout=torch.load(out/'images'/key/'layout.pt',map_location='cpu',weights_only=True,mmap=True)
    values={m:[] for m in metrics}
    for r in rr:
        if not r['read_defined'] or r['total_derivative_zero']:continue
        # fsum over all four probes directly: no reuse of authoritative C or averaged energy.
        ti=r['token_index'];den=math.fsum(map(float,e[ti].ravel()));cs={}
        for arm in ('G','P','S'):
            ids=[int(p) for p in torch.nonzero(layout['regions'][arm][ti]).flatten() if int(p)!=r['token']]
            cs[arm]=math.fsum(float(e[ti,v,p]) for v in range(4) for p in ids)/den
            scalar_difference=max(scalar_difference,abs(cs[arm]-r['C'][arm]))
        for m,v in zip(metrics,(100*(cs['G']-cs['P']),100*(cs['G']-cs['S']),100*(cs['S']-cs['P']))):values[m].append(v)
    imageresults[key]={m:math.fsum(v)/len(v) if v else None for m,v in values.items()}
placevalues=collections.defaultdict(list)
for cohort in ('query','source'):
    for rid in sorted({r['id'] for r in rows}):
        keys=sorted(k for k,rr in byimage.items() if rr[0]['id']==rid and rr[0]['cohort']==cohort)
        for m in metrics:
            vals=[imageresults[k][m] for k in keys if imageresults[k][m] is not None]
            placevalues[cohort,m].append(math.fsum(vals)/len(vals))
verified={};endpoint_difference=0.
for (cohort,m),vv in placevalues.items():
    a=np.array(vv);draws=np.random.default_rng(20261114).integers(0,len(a),(5000,len(a)))
    result=dict(mean=math.fsum(vv)/len(vv),interval95=np.percentile(a[draws].mean(1),[2.5,97.5]).tolist())
    ref=summary['cohorts'][cohort]['metrics'][m]
    endpoint_difference=max(endpoint_difference,abs(result['mean']-ref['mean']),max(abs(x-y) for x,y in zip(result['interval95'],ref['interval95'])))
    verified[cohort+'_'+m]=result
assert scalar_difference<1e-12 and endpoint_difference<1e-10
result=dict(complete=True,saved_energy_only=True,arms=['G','P','S'],independent_reduction='Python math.fsum of all saved four-probe entries; same fixed layout masks, hierarchical equal-image/place means and independent5000paired-place draws.',
    maximum_C_absolute_difference=scalar_difference,maximum_paired_endpoint_pp_difference=endpoint_difference,verified=verified,
    model_gradient_render_calls=0,old_scores_recomputed=False,CUDA_initialized=torch.cuda.is_initialized(),body_wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(out/'independent_metric_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
