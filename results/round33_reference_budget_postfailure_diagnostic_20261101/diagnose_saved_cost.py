"""Post-failure saved-array geometry count only, not a probability experiment.

No producer/reference module import, no integrate, no Gaussian PDF/CDF,
no log integrand, no quadrature weighted sums. SVD is used only to rebuild
fixed rectangle panel geometry. GK21 abscissae are read as source literals.
"""
from pathlib import Path
import ast,hashlib,json,math,time
from fractions import Fraction
from decimal import Decimal,localcontext
import numpy as np
ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'results/round33_reference_budget_postfailure_diagnostic_20261101'
RUN=ROOT/'results/round33_log_real_reference'
AUDIT=ROOT/'results/round33_log_real_reference_independent_audit'
CASE='harvard_c3__hv_c3_1__g16__q0015'
SCIPY_SOURCE=Path('/home/liwt/miniColmap/.venv/lib/python3.11/site-packages/scipy/integrate/_quad_vec.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def array_sha(value):
 value=np.ascontiguousarray(value)
 header=json.dumps(dict(dtype=str(value.dtype),shape=list(value.shape)),sort_keys=True,separators=(',',':')).encode()
 return hashlib.sha256(header+b'\n'+value.tobytes()).hexdigest()

def rebuild_geometry(mu,factor,h):
 ds=max(abs(float(mu[2])),float(np.max(abs(factor[2]))))
 rr=math.hypot(*map(float,np.r_[mu[2]/ds,factor[2]/ds]))
 nm=(mu/ds)/rr;nf=(factor/ds)/rr
 sigma=math.hypot(*map(float,nf[2]));direction=nf[2]/sigma
 _,_,vh=np.linalg.svd(nf[2:3],full_matrices=True)
 base=nf[:2]@vh[1:].T;beta=nf[:2]@direction
 cf=np.column_stack((base,h*np.eye(2)))
 sx=math.hypot(*map(float,cf[0]));dx=cf[0]/sx
 ky=float(np.dot(cf[1],dx));sy=math.hypot(*map(float,cf[1]-ky*dx))
 return dict(m=float(nm[2]),sigma=sigma,xy=nm[:2],beta=beta,sx=sx,ky=ky,sy=sy)

def count_geometry(zeta,order,boxes,g,policy,nodes):
 """Exact fixed partition/node-admission arithmetic only; no density values."""
 z=g['m']+g['sigma']*zeta;location=g['xy']+g['beta']*zeta
 xmid=(z*((boxes[:,0]+boxes[:,2])/2)-location[0])/g['sx']
 xhalf=z*((boxes[:,2]-boxes[:,0])/2)/g['sx']
 low,high=xmid-xhalf,xmid+xhalf;width=policy['standard_x_panel_width']
 regular=high>low
 first=[];counts=[]
 for a,b,good in zip(low,high,regular):
  start=math.floor(float(a)/width)+1 if good else 0
  stop=math.ceil(float(b)/width)-1 if good else -1
  first.append(start);counts.append(max(0,stop-start+1))
 minimum=(sum(counts)+len(boxes))*order
 ca=np.asarray(counts,np.int64);go=np.repeat(np.arange(len(boxes)),ca)
 offsets=np.arange(len(go))-np.repeat(np.cumsum(ca)-ca,ca)
 grid=np.asarray([float(first[int(owner)]+int(offset))*width for owner,offset in zip(go,offsets)],np.float64)
 owner_parts=[np.flatnonzero(regular),np.flatnonzero(regular),go]
 point_parts=[low[regular],high[regular],grid]
 if g['ky']!=0:
  for axis in (1,3):
   for c in policy['inner_geometry_breaks']:
    roots=(z*boxes[:,axis]-location[1]-c*g['sy'])/g['ky']
    active=regular&np.isfinite(roots)&(low<roots)&(roots<high)
    owner_parts.append(np.flatnonzero(active));point_parts.append(roots[active])
 owners=np.concatenate(owner_parts);points=np.concatenate(point_parts)
 perm=np.lexsort((points,owners));owners,points=owners[perm],points[perm]
 valid=(owners[1:]==owners[:-1])&(points[1:]>points[:-1])
 po=owners[:-1][valid];a,b=points[:-1][valid],points[1:][valid]
 mid,half=(a+b)/2,(b-a)/2
 nper=np.bincount(po,minlength=len(boxes));single=nper[po]==1
 mid[single],half[single]=xmid[po[single]],xhalf[po[single]]
 collapsed=~regular
 if np.any(collapsed):
  ix=np.flatnonzero(collapsed);po=np.r_[po,ix];mid=np.r_[mid,xmid[ix]];half=np.r_[half,xhalf[ix]]
 assert np.all(half>0) and np.all(2*half<=width)
 u=mid[:,None]+half[:,None]*nodes[None]
 ymid=(z*((boxes[po,1]+boxes[po,3])/2)[:,None]-location[1]-g['ky']*u)/g['sy']
 yhalf=(z*((boxes[po,3]-boxes[po,1])/2)/g['sy'])[:,None]
 narrow=int(np.count_nonzero(yhalf*(np.abs(ymid)+yhalf)<=policy['narrow_cdf_exponent_span']))
 return dict(inner_x_nodes=int(len(po)*order),narrow_intervals=narrow,narrow_y_nodes_GL16=narrow*16,
  narrow_y_nodes_GL8=narrow*8,panels=int(len(po)),minimum_x_admission=minimum,
  total_GL16=int(len(po)*order+narrow*16),total_GL8=int(len(po)*order+narrow*8))

def sum_counts(rows):
 return {k:sum(r[k] for r in rows) for k in rows[0]} if rows else {}

started=time.time()
pre=read(OUT/'before_execution.json')
assert pre['script_sha256']==sha(__file__)
accept=read(ROOT/'results/round33_log_real_root_acceptance_20261101.json')
for name,h in accept['files'].items():assert sha(ROOT/name)==h
for name,h in accept['source_hashes'].items():assert sha(ROOT/name)==h
lock=read(RUN/'lock.json');rows=read(RUN/'forward/records.json');case=[r for r in rows if r['id']==CASE]
assert [r['order'] for r in case]==[32,48] and case[0]['passed'] and not case[1]['passed']
assert sha(SCIPY_SOURCE)==pre['scipy_quad_source_sha256']
arrays={};bindings={}
for row in case:
 path=RUN/row['archive_path'];assert sha(path)==row['archive_sha256'];bindings[str(path)]=sha(path)
 wanted=['input__mu','input__factor','input__bounds','input__content_bias','initial_depth_edges','original_depth_edges','pilot_zeta']
 if row['order']==48:wanted+=['pilot_completed','partial_raw_computed','partial_log_raw_mass','partial_log_raw_fov_mass']
 with np.load(path,allow_pickle=False) as archive:bank={n:archive[n] for n in wanted}
 for n,a in bank.items():assert array_sha(a)==row['array_sha256'][n]
 arrays[row['order']]=bank
b=arrays[48]
assert all(array_sha(arrays[32][n])==array_sha(b[n]) for n in ('input__mu','input__factor','input__bounds','initial_depth_edges','pilot_zeta'))
bounds=b['input__bounds'];fov=np.array([bounds[:,0].min(),bounds[:,1].min(),bounds[:,2].max(),bounds[:,3].max()])
boxes=np.vstack((bounds,fov));g=rebuild_geometry(b['input__mu'],b['input__factor'],lock['policy']['h'])
# Only read the 21 abscissa tuple from installed SciPy source; do not call quad.
tree=ast.parse(SCIPY_SOURCE.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_quadrature_gk21')
assignment=next(n for n in function.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='x')
gk=np.asarray(ast.literal_eval(assignment.value),np.float64);assert len(gk)==21
edges=b['initial_depth_edges'];pilot=b['pilot_zeta'];intervals=len(edges)-1
quad=np.concatenate([.5*(a+c)+.5*(c-a)*gk for a,c in zip(edges[:-1],edges[1:])])
projection={}
for order in (32,48):
 nodes=np.polynomial.legendre.leggauss(order)[0]
 pilot_counts=[count_geometry(float(z),order,boxes,g,lock['reference_policy'],nodes) for z in pilot]
 quad_counts=[count_geometry(float(z),order,boxes,g,lock['reference_policy'],nodes) for z in quad]
 initial=sum_counts(pilot_counts+quad_counts)
 item=dict(pilot=sum_counts(pilot_counts),initial_GK21=sum_counts(quad_counts),initial_total=initial,
  pilot_depth_count=len(pilot),initial_GK21_depth_count=len(quad))
 if order==48:
  evaluated=case[1]['work']['quadrature_depth_evaluations'];done=evaluated-1
  prefix=sum_counts(pilot_counts+quad_counts[:done]);next_count=quad_counts[done]
  assert prefix['total_GL16']==case[1]['charged_cell_node_evaluations']
  assert prefix['inner_x_nodes']==case[1]['work']['inner_x_nodes']
  assert prefix['narrow_y_nodes_GL16']==case[1]['work']['narrow_y_pdf_nodes']
  remaining=lock['reference_policy']['max_call_cell_node_evaluations']-prefix['total_GL16']
  assert next_count['minimum_x_admission']>remaining
  item.update(exact_completed_prefix=prefix,next_failed_callback_zeta=float(quad[done]),
   next_failed_callback_index_1based=evaluated,next_failed_initial_interval_zero_based=done//21,
   next_failed_GK21_node_zero_based=done%21,next_failed_interval_edges=edges[done//21:done//21+2].tolist(),
   requested_next_callback=next_count,remaining_call_budget=remaining,
   remaining_initial_GK21_callbacks=len(quad)-done,
   saved_failure_at_initialization_before_adaptive_refinement=True)
 projection[order]=item

def exact_gl8_relative_bound(span):
 n=8;coeff=Fraction(2**17*math.factorial(8)**4,17*math.factorial(16)**3)
 summation=sum((span**(16-k)/Fraction(2**k*math.factorial(k)*math.factorial(16-2*k)) for k in range(9)),Fraction())
 constant=coeff*math.factorial(16)*summation/Fraction(2)
 with localcontext() as context:
  context.prec=70;s=Decimal(span.numerator)/Decimal(span.denominator)
  return str(Decimal(constant.numerator)/Decimal(constant.denominator)*(2*s).exp())
checks=read(AUDIT/'checks.json');comparisons=read(AUDIT/'comparisons.json')
report=dict(post_failure_diagnostic=True,not_selected_reference_change=True,
 saved_probability_results_only=True,new_probability_results_created=False,reference_integrations=0,model_integrations=0,
 Gaussian_pdf_evaluations=0,Gaussian_cdf_evaluations=0,quadrature_weighted_sums=0,
 observed_saved_array_reads=list(bindings),input_bindings=bindings,root_acceptance_sha256=sha(ROOT/'results/round33_log_real_root_acceptance_20261101.json'),
 frozen_source_hashes=accept['source_hashes'],scipy_quad_source_sha256=sha(SCIPY_SOURCE),script_sha256=sha(__file__),
 saved_case_records=case,selected_array_sha256={str(o):{n:array_sha(a) for n,a in bank.items()} for o,bank in arrays.items()},
 original_edge_count=len(b['original_depth_edges']),retained_edge_count=len(edges),
 pilot_completed_count=int(b['pilot_completed'].sum()),partial_raw_computed_count=int(b['partial_raw_computed'].sum()),
 partial_native_all_negative_infinity=bool(np.isneginf(b['partial_log_raw_mass']).all()),
 partial_fov_all_negative_infinity=bool(np.isneginf(b['partial_log_raw_fov_mass']).all()),
 cost_projection=projection,GL8_relative_exact_arithmetic_bound_at_span_one_half=exact_gl8_relative_bound(Fraction(1,2)),
 actual_frozen_narrow_span=lock['reference_policy']['narrow_cdf_exponent_span'],
 failed_case_same_order32_audit=next(r for r in checks if r['id']==CASE and r['order']==32),
 comparison_counts={kind:dict(total=sum(r['kind']==kind for r in comparisons),passed=sum(r['kind']==kind and r['passed'] for r in comparisons)) for kind in ('new_orders','old_production_vs_new')},
 cost_geometry=dict(m=g['m'],sigma=g['sigma'],sx=g['sx'],ky=g['ky'],sy=g['sy'],xy=g['xy'].tolist(),beta=g['beta'].tolist()),
 finished_unix=time.time(),wall_seconds=time.time()-started,
 caveat='Geometry counts at fixed saved pilots and initial SciPy GK21 abscissae only; GL8 probability/adaptivity/accuracy not executed or certified.')
assert not (OUT/'diagnostic.json').exists()
(OUT/'diagnostic.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps(dict(projection=projection,bound=report['GL8_relative_exact_arithmetic_bound_at_span_one_half'],wall_seconds=report['wall_seconds']),indent=2))
