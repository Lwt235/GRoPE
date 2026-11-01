"""R25 post-numerical labels, independent metrics and sealed eight-gate screen."""
from __future__ import annotations
import argparse
from fractions import Fraction
import math
from pathlib import Path
import time
import traceback
import numpy as np
from experiments import round25_student_contract as C
from experiments import round20_anchor_fixed_metrics as M
from experiments import round25_student_metrics as A

CONDITION=dict(condition='g24_g8',scenario='scale',query='g24',key='g8')

def frac(x):return dict(numerator=x.numerator,denominator=x.denominator,value=float(x))
def primary_summary(rows,entries):
 expected=[(e['scene'],tuple(e['pair'])) for e in entries]
 actual=[(r['scene'],tuple(r['pair'])) for r in rows]
 if actual!=expected or len(set(actual))!=48:raise ValueError('ordered complete old population required')
 rs={};cs={}
 for scene,valid in zip(C.SCENES,(382,402,312)):
  chosen=[r for r in rows if r['scene']==scene]
  if len(chosen)!=16:raise ValueError('scene count changed')
  reg=[r['region'] for r in chosen];cen=[r['center'] for r in chosen]
  if any(r['n_query']!=48 or r['n_key']!=432 for r in reg):raise ValueError('native matrix changed')
  mass=sum(r['mass'] for r in reg)
  sums={m+'_sum':math.fsum(r[m+'_sum'] for r in reg) for m in M.METRICS}
  rs[scene]=dict(mass=mass,**sums,**{m:sums[m+'_sum']/mass for m in M.METRICS})
  count={k:sum(r[k] for r in cen) for k in ('pck_count','n_point','n_query')}
  if count['n_point']!=valid or count['n_query']!=768:raise ValueError('center mask population changed')
  total=math.fsum(r['distance_sum'] for r in cen);square=math.fsum(r['distance_squared_sum'] for r in cen)
  cs[scene]=dict(**count,pck=frac(Fraction(count['pck_count'],valid)),distance_sum=total,distance_squared_sum=square,mean=total/valid,rmse=math.sqrt(square/valid))
 pck=sum((Fraction(cs[s]['pck_count'],cs[s]['n_point']) for s in C.SCENES),Fraction())/3
 return dict(condition='g24_g8',population=dict(pairs=48,queries=2304,valid_centers=1096),region=dict(by_scene=rs,macro={m:math.fsum(rs[s][m] for s in C.SCENES)/3 for m in M.METRICS}),center=dict(by_scene=cs,macro=dict(pck=frac(pck),mean=math.fsum(cs[s]['mean'] for s in C.SCENES)/3,rmse=math.fsum(cs[s]['rmse'] for s in C.SCENES)/3)))

def primary_checks(summary,baseline):
 kl=summary['region']['macro']['kl'];p=summary['center']['macro']['pck'];p=Fraction(p['numerator'],p['denominator'])
 checks=[]
 for role in ('strong_quad','legacy_quad16'):
  r=baseline['old_by_role'][role]['by_condition']['g24_g8']['kl'];b=1.02*r
  checks.append(dict(reference=role,metric='kl',candidate_value=kl,reference_value=r,bound=b,passed=bool(r>0 and kl<=b)))
 for source,roles in (('old_by_role',('point_localizer','quartic_localizer','legacy_point_localizer')),('by_role',('anchor_point_localizer','anchor_quartic_localizer','anchor_point_legacy'))):
  for role in roles:
   v=baseline[source][role]['center']['by_condition']['g24_g8']['pck'];r=Fraction(v['numerator'],v['denominator']);b=r-Fraction(1,200)
   checks.append(dict(reference=role,metric='pck',candidate_value=frac(p),reference_value=frac(r),bound=frac(b),passed=p>=b))
 return dict(passed=all(r['passed'] for r in checks),checks=checks)

def require_numerical(out,audit):
 lock=C.verify(out);alock=C.verify(audit)
 if lock['source_hashes']!=alock['source_hashes'] or lock['runtime']!=alock['runtime']:raise ValueError('joint freeze differs')
 completion=C.read(audit/'completion.json');C.verify_inventory(audit,completion['files'],exclude=('completion.json',))
 v=C.read(audit/'forward_verification.json')
 if (not completion['passed'] or not v['passed'] or not v['numerical_gate_passed'] or v['checked_pairs']!=48 or v['checked_queries']!=2304 or v['reference_call_count']!=4608 or not v['all_query_reference_population_checked'] or v['labels_read']):raise ValueError('complete accepted prelabel numerical verification required')
 for key,path in [('producer_lock_sha256',out/'lock.json'),('audit_lock_sha256',audit/'lock.json'),('producer_forward_complete_sha256',out/'forward_complete.json'),('producer_records_sha256',out/'forward/records.json')]:
  if v[key]!=C.sha(path):raise ValueError('audit producer binding changed '+key)
 if completion['forward_verification_sha256']!=C.sha(audit/'forward_verification.json'):raise ValueError('audit verification changed')
 seal=C.read(out/'forward_complete.json')
 if seal['forward_completion_sha256']!=C.sha(out/'forward/completion.json'):raise ValueError('forward inventory seal changed')
 forward=C.read(out/'forward/completion.json')
 C.verify_inventory(out/'forward',forward['files'],exclude=('completion.json',))
 return lock

def load_parts(lock):
 if C.sha(lock['partitions_path'])!=lock['partitions_sha256']:raise ValueError('inherited native partition changed')
 with np.load(lock['partitions_path'],allow_pickle=False) as z:
  return {q:{name.split('__')[1]:z[name] for name in z.files if name.startswith(q+'__')} for q in ('g8','g16','g24')}

def load_probability(out,records,entry):
 selected=[r for r in records if r['scene']==entry['scene'] and r['pair']==entry['pair']]
 if len(selected)!=48 or [r['query_index'] for r in selected]!=list(range(48)):raise ValueError('complete ordered 48 query records required')
 probs=[];winners=[]
 for r in selected:
  if r['status']!='ok':raise ValueError('failed numerical row cannot enter efficacy')
  path=out/r['archive_path']
  if C.sha(path)!=r['archive_sha256']:raise ValueError('prediction archive changed')
  with np.load(path,allow_pickle=False) as z:
   p=z['probability'];win=z['argmax']
  for name,val in [('probability',p),('argmax',win)]:
   if C.array_sha(val)!=r['array_sha256'][name]:raise ValueError('sealed prediction array changed')
  if p.shape!=(432,) or p.dtype!=np.float64 or win.dtype!=np.int32 or win.size!=1 or int(win.reshape(-1)[0])!=int(p.argmax()):raise ValueError('invalid native P/first winner')
  probs.append(p);winners.append(int(win.reshape(-1)[0]))
 return np.stack(probs),np.asarray(winners,np.int32)

def generate_rows(out,lock,independent=False):
 parts=load_parts(lock);records=C.read(out/'forward/records.json');rows=[];sources={};module=A if independent else M
 for entry in lock['entries']:
  # These are the first actual label-byte accesses of the producer stage.
  for field in ('label','target'):
   path=entry[field+'_path'];digest=entry[field+'_sha256']
   if C.sha(path)!=digest:raise ValueError('inherited label bytes changed')
   sources[path]=digest
  with np.load(entry['label_path'],allow_pickle=False) as z:labels={k:z[k] for k in ('gt_xy','gt_visible')}
  with np.load(entry['target_path'],allow_pickle=False) as z:point=z['g24_point_xy'];valid=z['g24_point_visible']
  p,winner=load_probability(out,records,entry)
  direct,common=module.target_counts(labels,parts,'g24','g8')
  r=module.region_metrics(p,parts,CONDITION,direct,common)
  center=module.center_metrics(winner,parts['g8']['bounds'],point,valid)
  rows.append(dict(scene=entry['scene'],pair=entry['pair'],**CONDITION,region=r,center=center))
  print('R25_METRICS', 'independent' if independent else 'producer',entry['scene'],entry['pair_name'],flush=True)
 return rows,sources

def discrepancy(a,b):
 if isinstance(a,dict):
  if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('metric dictionary schema differs')
  return max([discrepancy(a[k],b[k]) for k in a] or [0.])
 if isinstance(a,list):
  if not isinstance(b,list) or len(a)!=len(b):raise ValueError('metric row population differs')
  return max([discrepancy(x,y) for x,y in zip(a,b)] or [0.])
 if isinstance(a,(str,bool)) or a is None or isinstance(a,int):
  if a!=b:raise ValueError('exact identity/count/gate mismatch')
  return 0.
 if not math.isfinite(a) or not math.isfinite(b):raise ValueError('nonfinite metric')
 delta=abs(a-b)
 if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9):raise ValueError('metric disagreement beyond frozen replay tolerance')
 return delta

def run(out,audit,independent=False):
 directory=out/('metrics_audit' if independent else 'evaluation')
 if directory.exists():raise ValueError('retain previous label-stage attempt')
 lock=require_numerical(out,audit)
 if independent:
  comp=C.read(out/'evaluation/completion.json');C.verify_inventory(out/'evaluation',comp['files'],exclude=('completion.json',))
  if not comp['passed']:raise ValueError('completed producer metric execution required')
 directory.mkdir();started=time.time();C.write(directory/'started.json',dict(created_unix=started,prelabel_audit_sha256=C.sha(audit/'forward_verification.json')))
 try:
  rows,sources=generate_rows(out,lock,independent)
  old=C.read(C.ROOT/'results/round20_anchor_fixed/summary.json')
  summary=A.summarize(rows,lock['entries']) if independent else primary_summary(rows,lock['entries'])
  checks=A.screen(summary,old) if independent else primary_checks(summary,old)
  C.write(directory/'rows.json',rows);C.write(directory/'summary.json',summary);C.write(directory/'screen.json',checks)
  if independent:
   primary=C.read(out/'evaluation/screen.json')
   simplified=[{k:c[k] for k in ('reference','metric','candidate_value','reference_value','bound','passed')} for c in checks['checks']]
   difference=dict(rows=discrepancy(C.read(out/'evaluation/rows.json'),rows),summary=discrepancy(C.read(out/'evaluation/summary.json'),summary),checks=discrepancy(primary,dict(passed=checks['passed'],checks=simplified)))
   C.write(directory/'verification.json',dict(passed=True,maximum_discrepancy=difference,necessary_screen_passed=checks['passed'],scientific_gate_passed=False,goal_complete=False))
  matched=old['by_role']['anchor_trace']
  C.write(directory/'matched_gaussian_diagnostic.json',dict(role='anchor_trace',region=matched['by_condition']['g24_g8'],center=matched['center']['by_condition']['g24_g8'],is_gate=False))
  C.write(directory/'completion.json',dict(passed=True,files=C.inventory(directory),label_sources=sources,started_unix=started,finished_unix=time.time(),numerical_audit_sha256=C.sha(audit/'forward_verification.json'),necessary_screen_passed=checks['passed'],scientific_gate_passed=False,goal_complete=False))
 except BaseException as e:
  C.write(directory/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),finished_unix=time.time(),scientific_gate_passed=False,goal_complete=False));raise

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('evaluate','audit'));p.add_argument('--out',type=Path,default=C.DEFAULT_OUT);p.add_argument('--audit-root',type=Path,default=C.DEFAULT_AUDIT);a=p.parse_args();C.configure_runtime();run(a.out,a.audit_root,a.stage=='audit')
if __name__=='__main__':main()
