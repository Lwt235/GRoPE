"""R37 saved geometry only; no Gaussian values or probability integration.
Reconstruct complete initial X partitions and narrow Y branch membership.
Uniform lower-order projections and fixed span bins are cost diagnostics only.
"""
from pathlib import Path
import ast,hashlib,json,math,time,sys
import numpy as np
ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'results/round37_reference_budget_postfailure_diagnostic_20261101'
RUN=ROOT/'results/round37_component_round_real_reference'
CASE='harvard_c5__hv_c5_1__g8__q0184'
HELPER=ROOT/'experiments/round36_component_round_quadrature.py'
SPAN_EXPONENTS=(-80,-64,-48,-40,-32,-24,-18,-16,-12,-11,-10,-8,-7,-6,-4,-3,-2)
ADAPTIVE_THRESHOLDS=((1,-40),(2,-18),(3,-11),(4,-7),(6,-3))
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
 span=yhalf*(np.abs(ymid)+yhalf)
 narrow_mask=span<=policy['narrow_cdf_exponent_span'];narrow=int(np.count_nonzero(narrow_mask))
 bins={f'span_le_2pow_{exponent}':int(np.count_nonzero(narrow_mask&(span<=2.**exponent))) for exponent in SPAN_EXPONENTS}
 with np.errstate(over='ignore',under='ignore',invalid='ignore'):
  span_upper=np.nextafter(yhalf*np.nextafter(np.abs(ymid)+yhalf,np.inf),np.inf)
 selected=np.full(span.shape,8,np.int64)
 for n,exponent in reversed(ADAPTIVE_THRESHOLDS):selected[span_upper<=2.**exponent]=n
 adaptive_counts={f'adaptive_GL{n}_intervals':int(np.count_nonzero(narrow_mask&(selected==n))) for n in (1,2,3,4,6,8)}
 adaptive_y_nodes=sum(n*adaptive_counts[f'adaptive_GL{n}_intervals'] for n in (1,2,3,4,6,8))
 assert sum(adaptive_counts.values())==narrow
 return dict(inner_x_nodes=int(len(po)*order),narrow_intervals=narrow,narrow_y_nodes_GL16=narrow*16,
  narrow_y_nodes_GL8=narrow*8,panels=int(len(po)),minimum_x_admission=minimum,
  total_GL16=int(len(po)*order+narrow*16),total_GL8=int(len(po)*order+narrow*8),
  **{f'total_uniform_GL{n}':int(len(po)*order+narrow*n) for n in (1,2,3,4,6,8)},adaptive_y_nodes=adaptive_y_nodes,total_adaptive_GL=int(len(po)*order+adaptive_y_nodes),**adaptive_counts,**bins)

def sum_counts(rows):
 return {k:sum(r[k] for r in rows) for k in rows[0]} if rows else {}

def main():
 started=time.time();deadline=time.perf_counter()+300
 pre=read(OUT/'before_execution.json')
 for name,h in pre['bindings'].items():assert sha(ROOT/name)==h
 assert pre['script_sha256']==sha(__file__)
 accept=read(ROOT/'results/round37_component_round_real_root_acceptance_20261101.json')
 assert accept['delivery_integrity_passed'] and accept['accepted_for_failure_interpretation'] and not accept['numerical_preflight_passed']
 lock=read(RUN/'lock.json');rows=read(RUN/'forward/records.json');case=[r for r in rows if r['id']==CASE]
 assert [r['order'] for r in case]==[32,48] and case[0]['passed'] and not case[1]['passed']
 arrays={};bindings={}
 fields=('input__mu','input__factor','input__bounds','input__content_bias','initial_depth_edges','original_depth_edges','pilot_zeta','outer_interval_bounds','outer_active_interval_indices')
 for row in case:
  path=RUN/row['archive_path'];assert sha(path)==row['archive_sha256'];bindings[str(path.relative_to(ROOT))]=sha(path)
  wanted=fields+(('pilot_completed',) if row['order']==48 else ())
  with np.load(path,allow_pickle=False) as archive:bank={n:archive[n] for n in wanted}
  for n,a in bank.items():assert array_sha(a)==row['array_sha256'][n]
  arrays[row['order']]=bank
 b=arrays[48]
 for n in fields[:7]:assert array_sha(arrays[32][n])==array_sha(b[n])
 assert b['pilot_completed'].all() and len(b['outer_interval_bounds'])==56
 assert len(arrays[32]['outer_interval_bounds'])==len(arrays[32]['initial_depth_edges'])-1
 bounds=b['input__bounds'];fov=np.array([bounds[:,0].min(),bounds[:,1].min(),bounds[:,2].max(),bounds[:,3].max()])
 boxes=np.vstack((bounds,fov));g=rebuild_geometry(b['input__mu'],b['input__factor'],lock['policy']['h'])
 tree=ast.parse(HELPER.read_text());assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='X')
 gk=np.asarray(ast.literal_eval(assignment.value),np.float64);assert len(gk)==21
 edges=b['initial_depth_edges'];pilot=b['pilot_zeta'];quad=np.concatenate([.5*(a+c)+.5*(c-a)*gk for a,c in zip(edges[:-1],edges[1:])])
 assert len(pilot)==133 and len(quad)==2793
 projection={}
 def count(z,order,nodes):
  if time.perf_counter()>deadline:raise TimeoutError('300s geometry diagnostic deadline')
  assert g['m']+g['sigma']*float(z)>0
  return count_geometry(float(z),order,boxes,g,lock['reference_policy'],nodes)
 for order in (32,48):
  nodes=np.polynomial.legendre.leggauss(order)[0]
  pc=[count(z,order,nodes) for z in pilot];qc=[count(z,order,nodes) for z in quad]
  initial=sum_counts(pc+qc)
  item=dict(pilot=sum_counts(pc),initial_GK21=sum_counts(qc),initial_total=initial,pilot_depth_count=len(pilot),initial_GK21_depth_count=len(quad))
  if order==32:
   assert initial['total_GL8']==case[0]['charged_cell_node_evaluations']
   assert initial['inner_x_nodes']==case[0]['work']['inner_x_nodes']
   assert initial['narrow_y_nodes_GL8']==case[0]['work']['narrow_y_pdf_nodes']
   item['completed_order32_all_initial_nodes_exact']=True
  else:
   evaluated=case[1]['work']['quadrature_depth_evaluations'];done=evaluated-1
   prefix=sum_counts(pc+qc[:done]);next_count=qc[done]
   assert prefix['total_GL8']==case[1]['charged_cell_node_evaluations']
   assert prefix['inner_x_nodes']==case[1]['work']['inner_x_nodes']
   assert prefix['narrow_y_nodes_GL8']==case[1]['work']['narrow_y_pdf_nodes']
   remaining=lock['reference_policy']['max_call_cell_node_evaluations']-prefix['total_GL8']
   assert next_count['minimum_x_admission']>remaining
   item.update(exact_completed_prefix=prefix,next_failed_callback_zeta=float(quad[done]),next_failed_callback_index_1based=evaluated,next_failed_initial_interval_zero_based=done//21,next_failed_GK21_node_zero_based=done%21,next_failed_interval_edges=edges[done//21:done//21+2].tolist(),requested_next_callback=next_count,remaining_call_budget=remaining,remaining_initial_GK21_callbacks=len(quad)-done,saved_failure_before_adaptive_refinement=True)
  projection[order]=item
  (OUT/f'order{order}_cost.json').write_text(json.dumps(item,indent=2,sort_keys=True,allow_nan=False)+'\n')
 report=dict(post_failure_diagnostic=True,not_selected_reference_change=True,new_probability_results_created=False,reference_integrations=0,model_integrations=0,Gaussian_pdf_evaluations=0,Gaussian_cdf_evaluations=0,quadrature_weighted_sums=0,real_inputs_read=True,labels_read=False,observed_saved_array_reads=list(bindings),input_bindings=bindings,root_acceptance_sha256=sha(ROOT/'results/round37_component_round_real_root_acceptance_20261101.json'),script_sha256=sha(__file__),helper_sha256=sha(HELPER),before_execution_sha256=sha(OUT/'before_execution.json'),saved_case_records=case,selected_array_sha256={str(o):{n:array_sha(a) for n,a in bank.items()} for o,bank in arrays.items()},cost_projection=projection,span_cumulative_bin_exponents=list(SPAN_EXPONENTS),adaptive_thresholds=[list(t) for t in ADAPTIVE_THRESHOLDS],adaptive_span_upper_method='nextafter(half*nextafter(abs(mid)+half,+inf),+inf)',geometry={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in g.items()},finished_unix=time.time(),wall_seconds=time.time()-started,caveat='Fixed saved initial geometry only. Lower-order uniform costs do not establish accuracy, local bounds or convergence. No alternative integrator executed, no method selected. Adaptive refinements and all remaining real inputs unknown.')
 assert not (OUT/'diagnostic.json').exists()
 (OUT/'diagnostic.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps(dict(projection=projection,wall_seconds=report['wall_seconds']),indent=2))

if __name__=='__main__':main()
