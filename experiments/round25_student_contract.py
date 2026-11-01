"""Prospective R25 old-TRAIN necessary screen; metadata-only freeze helpers."""
from __future__ import annotations
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import scipy
import scipy.special
import torch
from threadpoolctl import threadpool_info
from experiments.round24_student_contract import POLICY as OLD_POLICY, GATES as OLD_GATES

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'results/round25_student_screen'
DEFAULT_AUDIT=ROOT/'results/round25_student_independent_audit'
SCENES=('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
FIELDS=('g24__a','g24__anchor_cov','g24__tau32','g24__q_features','g8__k_features','g8__bounds_px')
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')
POLICY=dict(OLD_POLICY)
REFERENCE_POLICY=dict(orders=[64,96],maximum_panel_side_in_h=4.,max_total_cell_node_evaluations=250000000,max_phase_wall_seconds=1800.)
GATES=dict(OLD_GATES,production_max_phase_wall_seconds=600.)
SOURCE_FILES=('ROUND25_STUDENT_SCREEN_PROTOCOL.md','experiments/round25_student_contract.py',
 'experiments/round25_student_screen.py','experiments/round25_student_audit.py',
 'experiments/round25_student_metrics.py','experiments/round25_student_evaluate.py',
 'tests/test_round25_student_screen.py','tests/test_round25_student_audit.py','tests/test_round25_student_metrics.py',
 'models/round24_student_t_pe.py','experiments/round24_student_reference.py')
BINDINGS=('results/round24_student_root_acceptance_20261031.json',
 'results/round24_student_post_result_decision_20261031.json',
 'results/round20_anchor_root_acceptance_20261031.json',
 'results/round20_anchor_fixed/lock.json','results/round20_anchor_fixed/forward/support_records.json',
 'results/round20_anchor_fixed/forward_complete.json','results/round20_anchor_fixed/summary.json')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def array_sha(value):
 value=np.ascontiguousarray(value)
 header=json.dumps(dict(dtype=str(value.dtype),shape=list(value.shape)),sort_keys=True,separators=(',',':')).encode()
 return hashlib.sha256(header+b'\n'+value.tobytes()).hexdigest()
def inventory(directory):
 directory=Path(directory)
 return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
def verify_inventory(directory,files,exclude=()):
 actual=inventory(Path(directory))
 if set(actual)!=set(files)|set(exclude):raise ValueError('inventory membership changed')
 if any(actual[n]!=h for n,h in files.items()):raise ValueError('inventory digest changed')
def source_names():
 found=set();pending=list(SOURCE_FILES)
 while pending:
  name=pending.pop()
  if name in found:continue
  path=ROOT/name
  if not path.is_file():raise ValueError('missing joint source '+name)
  found.add(name)
  if path.suffix!='.py':continue
  for node in ast.walk(ast.parse(path.read_text())):
   modules=[]
   if isinstance(node,ast.Import):modules=[n.name for n in node.names]
   elif isinstance(node,ast.ImportFrom):
    if node.level:raise ValueError('relative import needs explicit review')
    if node.module:modules=[node.module]+[node.module+'.'+n.name for n in node.names]
   for module in modules:
    parts=module.split('.')
    candidates=[Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py']+[Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
    pending.extend(str(p) for p in candidates if (ROOT/p).is_file() and str(p) not in found)
 return sorted(found)
def configure_runtime():
 torch.set_num_threads(4)
 if torch.get_num_interop_threads()!=4:torch.set_num_interop_threads(4)
 torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 return check_runtime()
def check_runtime():
 threads={n:os.environ.get(n) for n in THREAD_ENV}
 pools=threadpool_info()
 if any(v!='4' for v in threads.values()) or any(p['num_threads']!=4 for p in pools):raise ValueError('frozen numerical pools must all use four threads')
 if torch.get_num_threads()!=4 or torch.get_num_interop_threads()!=4 or not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:raise ValueError('frozen Torch runtime changed')
 builds={}
 for n,f in [('numpy',np.show_config),('scipy',scipy.show_config)]:
  s=io.StringIO()
  with redirect_stdout(s):f()
  builds[n+'_build_sha256']=hashlib.sha256(s.getvalue().encode()).hexdigest()
 return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,threads=threads,threadpools=sorted(pools,key=lambda p:p['filepath']),intraop=4,interop=4,deterministic=True,tf32=False,**builds)
def metadata():
 bindings={str(ROOT/n):sha(ROOT/n) for n in BINDINGS}
 r24=read(ROOT/BINDINGS[0]);r20=read(ROOT/BINDINGS[2]);decision=read(ROOT/BINDINGS[1])
 if not r24['accepted_for_numerical_interpretation'] or not r24['numerical_preflight_passed'] or not r20['accepted_for_development_interpretation']:raise ValueError('accepted inherited evidence required')
 if decision['root_acceptance_sha256']!=sha(ROOT/BINDINGS[0]) or decision['next_selected_work']['round']!='R25':raise ValueError('prospective R25 selection mismatch')
 for receipt in (r24,r20):
  for name,digest in receipt['files'].items():
   if Path(name).suffix=='.npz':raise ValueError('no actual arrays allowed during metadata freeze')
   path=ROOT/name
   if sha(path)!=digest:raise ValueError('accepted evidence changed '+name)
   bindings[str(path)]=digest
 for old_root in ('results/round24_student_preflight','results/round20_anchor_fixed'):
  inherited=read(ROOT/old_root/'lock.json')
  for name,digest in inherited['source_hashes'].items():
   if sha(ROOT/name)!=digest or sha(ROOT/old_root/'source_snapshot'/name)!=digest:raise ValueError('inherited frozen implementation changed '+name)
   bindings[str(ROOT/name)]=digest
 old=read(ROOT/'results/round20_anchor_fixed/lock.json')
 oldforward=read(ROOT/'results/round20_anchor_fixed/forward_complete.json')
 records=read(ROOT/'results/round20_anchor_fixed/forward/support_records.json')
 # The record index is authenticated by the old forward seal; cache bytes are deferred.
 if oldforward['support_records_sha256']!=sha(ROOT/'results/round20_anchor_fixed/forward/support_records.json'):raise ValueError('old support index seal mismatch')
 lookup={(e['scene'],tuple(e['pair'])):e for e in records}
 entries=[]
 for e in old['entries']:
  r=lookup[(e['scene'],tuple(e['pair']))]
  if r['pair_name']!=e['pair_name'] or r['input_sha256']!=e['input_sha256']:raise ValueError('old cache identity mismatch')
  entries.append(dict(e,cache_path=str(ROOT/'results/round20_anchor_fixed'/r['archive_path']),cache_sha256=r['archive_sha256'],cache_array_sha256={f:r['array_sha256'][f] for f in FIELDS}))
 if len(entries)!=48 or len(lookup)!=48 or len({(e['scene'],tuple(e['pair'])) for e in entries})!=48 or any(sum(e['scene']==s for e in entries)!=16 for s in SCENES):raise ValueError('old cohort changed')
 return dict(entries=entries,bindings=bindings,partitions_path=old['partitions_path'],partitions_sha256=old['partitions_sha256'])
def freeze(out,kind,producer_root=None,audit_root=None):
 out=Path(out)
 if out.exists():raise ValueError('refusing existing frozen directory')
 if kind not in ('producer','audit'):raise ValueError('unknown freeze role')
 data=metadata();runtime=check_runtime();sources={n:sha(ROOT/n) for n in source_names()}
 if kind=='audit':
  p=verify(Path(producer_root))
  if p['kind']!='producer' or p['source_hashes']!=sources or p['runtime']!=runtime or p['entries']!=data['entries']:raise ValueError('joint freezes differ')
 out.mkdir(parents=True)
 for n,h in sources.items():
  dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
  if sha(dest)!=h:raise ValueError('source changed during snapshot')
 lock=dict(version=1,kind=kind,created_unix=time.time(),runtime=runtime,source_hashes=sources,policy=POLICY,reference_policy=REFERENCE_POLICY,gates=GATES,fields=list(FIELDS),producer_root=str(Path(producer_root or out).resolve()),audit_root=str(Path(audit_root or out).resolve()),labels_read=False,real_arrays_read=False,scientific_gate_passed=False,goal_complete=False,**data)
 write(out/'lock.json',lock);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n');verify(out);return lock

def verify(out):
 out=Path(out);lock=read(out/'lock.json')
 if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('lock bytes changed')
 expected=dict(runtime=check_runtime(),policy=POLICY,reference_policy=REFERENCE_POLICY,gates=GATES,fields=list(FIELDS),labels_read=False,real_arrays_read=False,scientific_gate_passed=False,goal_complete=False,**metadata())
 if any(lock.get(k)!=v for k,v in expected.items()):raise ValueError('frozen metadata/runtime changed')
 if set(lock['source_hashes'])!=set(source_names()):raise ValueError('source closure changed')
 for n,h in lock['source_hashes'].items():
  if sha(ROOT/n)!=h or sha(out/'source_snapshot'/n)!=h:raise ValueError('source changed '+n)
 return lock
