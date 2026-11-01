"""Saved R35 arrays only: no model, reference import or integration.

The failed48 restored logs are diagnostic approximations, not accepted
reference probabilities. No new density, Gaussian CDF, or quadrature call.
"""
from pathlib import Path
import json,hashlib,math,time
import numpy as np
from scipy.special import logsumexp
ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/round35_roundoff_postfailure_diagnostic_20261101'
RUN=ROOT/'results/round35_log_real_reference'
CASE='harvard_c3__hv_c3_1__g16__q0030'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):
 a=np.ascontiguousarray(a);h=json.dumps(dict(dtype=str(a.dtype),shape=list(a.shape)),sort_keys=True,separators=(',',':')).encode()
 return hashlib.sha256(h+b'\n'+a.tobytes()).hexdigest()
def logabs(a,b):
 a,b=np.broadcast_arrays(np.asarray(a),np.asarray(b));high=np.maximum(a,b);v=np.full(a.shape,-np.inf);mask=a!=b
 v[mask]=high[mask]+np.log(-np.expm1(-abs(a[mask]-b[mask])))
 return v
pre=read(OUT/'before_execution.json');assert pre['script_sha256']==sha(__file__)
rows=read(RUN/'forward/records.json');case=[r for r in rows if r['id']==CASE];assert [r['order'] for r in case]==[32,48]
arrays={}
for row in case:
 p=RUN/row['archive_path'];assert sha(p)==row['archive_sha256']
 names=['input__bounds','input__content_bias','outer_log_scale','outer_scaled_integral']
 names+=['log_raw_mass','log_raw_fov_mass','probability'] if row['order']==32 else ['partial_log_raw_mass','partial_log_raw_fov_mass','partial_raw_computed']
 with np.load(p,allow_pickle=False) as archive:a={n:archive[n] for n in names}
 assert all(array_sha(v)==row['array_sha256'][n] for n,v in a.items());arrays[row['order']]=a
b32,b48=arrays[32],arrays[48];m32=b32['log_raw_mass'];m48=b48['partial_log_raw_mass']
f32=float(logsumexp(m32));f48=float(logsumexp(m48));ld=logabs(m48,m32)
# Reconstructed normalization solely describes the failed approximate raw vector.
bounds=b48['input__bounds'];bias=b48['input__content_bias'];floor=np.prod(bounds[:,2:]-bounds[:,:2],axis=1)*1e-30
logit=np.logaddexp(m48,np.log(floor))+bias;derived=np.exp(logit-logsumexp(logit))
outer={}
for row in case:
 order=row['order'];a=arrays[order];norm=float(np.max(abs(a['outer_scaled_integral'])))
 epsrel=1e-11 if order==32 else 1e-12;tol=epsrel*norm;error=row['scaled_outer_error_estimate']
 outer[order]=dict(scaled_integral_norm_max=norm,epsrel=epsrel,tol=tol,internal_global_error_target=tol/8,
  returned_total_error_estimate=error,returned_error_over_tol=error/tol,
  saved_scale_min=float(a['outer_log_scale'].min()),saved_scale_max=float(a['outer_log_scale'].max()),
  scaled_integral_all_positive_finite=bool(np.isfinite(a['outer_scaled_integral']).all() and np.all(a['outer_scaled_integral']>0)))
report=dict(post_failure_saved_arithmetic_only=True,failed_result_remains_failed=True,
 root_acceptance_sha256=sha(ROOT/'results/round35_log_real_root_acceptance_20261101.json'),
 records_sha256=sha(RUN/'forward/records.json'),script_sha256=sha(__file__),
 source_sha256=sha(ROOT/'experiments/round34_log_narrow_reference.py'),
 archive_hashes={str(r['order']):r['archive_sha256'] for r in case},
 actual_selected_array_hashes={str(o):{n:array_sha(v) for n,v in a.items()} for o,a in arrays.items()},
 reference_integrations=0,model_integrations=0,Gaussian_pdf_calls=0,Gaussian_cdf_calls=0,
 raw_partial_computed_count=int(b48['partial_raw_computed'].sum()),raw_partial_component_count=len(b48['partial_raw_computed']),
 partial_raw_all_finite=bool(np.isfinite(m48).all() and np.isfinite(b48['partial_log_raw_fov_mass']).all()),
 partial_vs32=dict(log_raw_FOV32=f32,log_raw_FOV48=f48,
  raw_FOV_relative=float(np.exp(logabs(f48,f32)-f32)),raw_L1_over_FOV32=float(np.exp(logsumexp(ld)-f32)),
  maximum_log_raw_absolute_difference=float(np.max(abs(m48-m32))),
  reconstructed_failed_probability_vs32_max_absolute=float(np.max(abs(derived-b32['probability']))),
  reconstructed_failed_argmax=int(derived.argmax()),accepted32_argmax=int(b32['probability'].argmax())),
 outer=outer,scope='No recomputed integral. Failed48 normalization is a post-failure diagnostic, not a saved or accepted P.',finished_unix=time.time())
assert not (OUT/'diagnostic.json').exists();(OUT/'diagnostic.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ('raw_partial_computed_count','raw_partial_component_count','partial_vs32','outer')},indent=2))
