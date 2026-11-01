"""R38 source/runtime/provenance contract; never decode archives or run a model."""
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

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'results/round38_rms_anchor_screen'
DEFAULT_AUDIT=ROOT/'results/round38_rms_anchor_independent_audit'
SCENES=('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
PAIR_COUNT,QUERY_COUNT,CASE_COUNT,KEY_COUNT=48,48,2304,432
IMAGE_HW=(144,192)
OBSERVATION_FIELDS=('q_points3','q_pixel_cov3','q_K','k_K','q_to_k_transform','q_predicted_depth')
CACHE_FIELDS=('g24__a','g24__anchor_cov','g24__tau32','g24__q_features','g8__k_features','g8__bounds_px')
FIELDS=CACHE_FIELDS
OLD_FIELDS=('input__mu','input__factor','input__bounds','input__content_bias')
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')
FAIL_FAST='first_failed_scheduled_outcome'
SCOPE=dict(labels_read=False,efficacy_tested=False,candidate_selected=False,
    general_numerical_accuracy_established=False,independent_probability_reference=False,
    scientific_gate_passed=False,goal_complete=False,selected_for_exploratory_triage=True)
POLICY=dict(h=.25,floor_density=1e-30,quadrature_order=24,maximum_tile_side_in_h=2.,
    max_cell_node_evaluations=4000000,negative_tail_threshold=10.,negative_tail_series_terms=20,
    max_negative_tail_relative_remainder=1e-10,projection_regularization='rms_depth_xy_gaussian')
# Preparation/input-audit budgets are frozen with the root protocol before any data.
GATES=dict(input_normalized_absolute=1e-10,normalization_absolute=2e-12,
    direct_fov_additivity_absolute=1e-8,production_max_phase_wall_seconds=7200.,
    production_max_phase_cell_node_evaluations=CASE_COUNT*POLICY['max_cell_node_evaluations'],
    preparation_max_phase_wall_seconds=1200.,input_audit_max_phase_wall_seconds=1200.,
    arithmetic_audit_max_phase_wall_seconds=1800.,metrics_max_phase_wall_seconds=1200.)
SOURCE_FILES=('ROUND38_RMS_ANCHOR_TRIAGE_PROTOCOL.md','docs/round38_rms_anchor_triage_design.md',
    'experiments/round38_rms_anchor_contract.py','experiments/round38_rms_anchor_inputs.py',
    'experiments/round38_rms_anchor_screen.py','experiments/round38_rms_anchor_audit.py',
    'experiments/round38_rms_anchor_evaluate.py',
    'tests/test_round38_rms_anchor_contract.py','tests/test_round38_rms_anchor_inputs.py',
    'tests/test_round38_rms_anchor_screen.py','tests/test_round38_rms_anchor_audit.py',
    'tests/test_round38_rms_anchor_evaluate.py','models/round29_log_rms_projective_pe.py',
    'docs/round29_log_rms_representation.md','docs/round26_rms_quadrature_derivation.md')
DESIGN='docs/round38_rms_anchor_triage_design.md'
DECISION='results/round37_component_round_real_post_result_decision_20261101.json'
R20='results/round20_anchor_root_acceptance_20261031.json'
R29='results/round29_log_rms_root_acceptance_20261031.json'
R30='results/round30_log_rms_root_acceptance_20261031.json'
R37='results/round37_component_round_real_root_acceptance_20261101.json'

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
def pair_id(entry):
 return entry['scene'].replace('/','__')+'__'+entry['pair_name']

def query_specs(entries):
 specs=[]
 for entry in entries:
  for index in range(QUERY_COUNT):
   y,x=divmod(index,8);x*=24;y*=24
   specs.append(dict(id=pair_id(entry)+'__g24__q%04d'%index,pair_id=pair_id(entry),
     **{k:entry[k] for k in ('scene','pair','pair_name')},query_size=24,query_index=index,
     query_bounds=[x,y,x+24,y+24]))
 if len(specs)!=CASE_COUNT or len({s['id'] for s in specs})!=CASE_COUNT:raise ValueError('fixed unique2304 population required')
 return specs

def metadata():
 """Authenticate old accepted metadata and allowlisted archive bytes, no np.load."""
 bindings={}
 def bind(path,expected=None):
  path=Path(path).resolve();digest=sha(path)
  if expected is not None and digest!=expected:raise ValueError('accepted bytes changed: '+str(path))
  if str(path) in bindings and bindings[str(path)]!=digest:raise ValueError('conflicting byte binding')
  bindings[str(path)]=digest;return digest
 receipts={}
 for name in (R20,R29,R30,R37):
  bind(ROOT/name);receipt=read(ROOT/name);receipts[name]=receipt
  for rel,digest in receipt['files'].items():bind(ROOT/rel,digest)
 if (receipts[R20].get('accepted_for_development_interpretation') is not True
     or receipts[R20].get('full_forward_and_final_independent_audits_passed') is not True
     or receipts[R29].get('numerical_preflight_passed') is not True
     or receipts[R29].get('accepted_for_numerical_interpretation') is not True
     or receipts[R30].get('accepted_for_applicability_interpretation') is not True
     or receipts[R30].get('applicability_calls_completed') is not True
     or receipts[R37].get('delivery_integrity_passed') is not True
     or receipts[R37].get('accepted_for_failure_interpretation') is not True
     or receipts[R37].get('numerical_preflight_passed') is not False):
  raise ValueError('accepted R20 inputs/R29 model and retained R37 failure required')
 bind(ROOT/DECISION);bind(ROOT/DESIGN);decision=read(ROOT/DECISION)
 selected=decision['next_selected_work']
 if (decision['root_acceptance_sha256']!=sha(ROOT/R37) or selected['round']!='R38'
     or selected['design_path']!=DESIGN or selected['design_sha256']!=sha(ROOT/DESIGN)):
  raise ValueError('prospective R38 design selection differs')
 for name,digest in decision['files'].items():bind(ROOT/name,digest)
 if selected['protocol_sha256']!=sha(ROOT/selected['protocol_path']):raise ValueError('prospective protocol changed')
 oldroot=ROOT/'results/round20_anchor_fixed';old=read(oldroot/'lock.json')
 for receipt,producer,auditor,sources in (
     (receipts[R20],oldroot,ROOT/'results/round20_anchor_independent_audit',old['source_hashes']),
     (receipts[R29],ROOT/'results/round29_log_rms_preflight',ROOT/'results/round29_log_rms_independent_audit',receipts[R29]['source_hashes']),
     (receipts[R30],ROOT/'results/round30_log_rms_applicability',ROOT/'results/round30_log_rms_applicability_independent_audit',receipts[R30]['source_hashes']),
     (receipts[R37],ROOT/'results/round37_component_round_real_reference',ROOT/'results/round37_component_round_real_reference_independent_audit',receipts[R37]['source_hashes'])):
  for name,digest in sources.items():
   bind(ROOT/name,digest)
   for base in (producer,auditor):bind(base/'source_snapshot'/name,digest)
 # The old audited implementations define the legal construction being copied,
 # but are not imported by the new independent implementation.
 for name in ('experiments/round23_ray_applicability.py','experiments/round13_mixture_mechanism.py',
              'experiments/round25_student_screen.py','experiments/round28_rms_applicability.py'):
  bind(ROOT/name)
 forward=read(oldroot/'forward_complete.json')
 bind(oldroot/'lock.json',forward['producer_lock_sha256'])
 bind(oldroot/'forward/support_records.json',forward['support_records_sha256'])
 records=read(oldroot/'forward/support_records.json')
 lookup={(r['scene'],tuple(r['pair'])):r for r in records}
 if len(records)!=PAIR_COUNT or len(lookup)!=PAIR_COUNT:raise ValueError('R20 support population differs')
 entries=[]
 fields=('scene','pair','pair_name','input_path','input_sha256','manifest_path','manifest_sha256',
         'target_path','target_sha256','label_path','label_sha256')
 for original in old['entries']:
  entry={k:original[k] for k in fields};r=lookup[(entry['scene'],tuple(entry['pair']))]
  if r['pair_name']!=entry['pair_name'] or r['input_sha256']!=entry['input_sha256']:raise ValueError('R20 support identity differs')
  entry.update(pair_id=pair_id(entry),cache_path=str(oldroot/r['archive_path']),cache_sha256=r['archive_sha256'],
    cache_array_sha256={k:r['array_sha256'][k] for k in CACHE_FIELDS})
  bind(entry['input_path'],entry['input_sha256']);bind(entry['manifest_path'],entry['manifest_sha256'])
  bind(entry['cache_path'],entry['cache_sha256'])
  manifest=read(entry['manifest_path'])
  if (manifest['scene']!=entry['scene'] or manifest['pair']!=entry['pair'] or manifest['split']!='train'
      or manifest['forward_uses_sensor'] is not False or manifest['input_sha256']!=entry['input_sha256']):
   raise ValueError('legal old TRAIN observation provenance differs')
  # Label paths/hashes above are authenticated JSON declarations only. Do not
  # hash, open or decode any label/target file before the root's new label gate.
  entries.append(entry)
 if (len(entries)!=PAIR_COUNT or len({pair_id(e) for e in entries})!=PAIR_COUNT
     or any(sum(e['scene']==s for e in entries)!=16 for s in SCENES)):
  raise ValueError('fixed48 pair population differs')
 bind(old['partitions_path'],old['partitions_sha256'])
 return dict(entries=entries,bindings=bindings,partitions_path=old['partitions_path'],partitions_sha256=old['partitions_sha256'])

def inherited_evidence():
 data=metadata();return data['bindings'],data['entries'],query_specs(data['entries'])

def freeze(out,kind,producer_root=None,audit_root=None):
 out=Path(out)
 if out.exists():raise ValueError('preserve every attempted R38 freeze')
 if kind not in ('producer','audit'):raise ValueError('unknown freeze role')
 data=metadata();runtime=check_runtime();sources={n:sha(ROOT/n) for n in source_names()}
 if kind=='audit':
  previous=verify(producer_root)
  if (previous['kind']!='producer' or previous['source_hashes']!=sources
      or previous['runtime']!=runtime or any(previous[k]!=v for k,v in data.items())):
   raise ValueError('joint freezes differ')
 out.mkdir(parents=True)
 for n,h in sources.items():
  destination=out/'source_snapshot'/n;destination.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(ROOT/n,destination)
  if sha(destination)!=h:raise ValueError('source changed during snapshot')
 specs=query_specs(data['entries']);write(out/'query_specs.json',specs)
 lock=dict(version=1,kind=kind,created_unix=time.time(),runtime=runtime,source_hashes=sources,
  policy=POLICY,gates=GATES,fail_fast=FAIL_FAST,case_count=CASE_COUNT,pair_count=PAIR_COUNT,
  key_count=KEY_COUNT,query_specs_sha256=sha(out/'query_specs.json'),
  ordered_query_ids=[s['id'] for s in specs],
  producer_root=str(Path(producer_root or out).resolve()),audit_root=str(Path(audit_root or out).resolve()),
  real_inputs_read=False,**SCOPE,**data)
 write(out/'lock.json',lock);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
 verify(out);return lock

def verify(out):
 out=Path(out);lock=read(out/'lock.json')
 if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('lock bytes changed')
 data=metadata();specs=query_specs(data['entries'])
 expected=dict(version=1,runtime=check_runtime(),policy=POLICY,gates=GATES,fail_fast=FAIL_FAST,
  case_count=CASE_COUNT,pair_count=PAIR_COUNT,key_count=KEY_COUNT,
  query_specs_sha256=sha(out/'query_specs.json'),ordered_query_ids=[s['id'] for s in specs],
  real_inputs_read=False,**SCOPE,**data)
 if any(lock.get(k)!=v for k,v in expected.items()) or read(out/'query_specs.json')!=specs:
  raise ValueError('frozen metadata/runtime/population differs')
 if set(lock['source_hashes'])!=set(source_names()):raise ValueError('source closure differs')
 for name,h in lock['source_hashes'].items():
  if sha(ROOT/name)!=h or sha(out/'source_snapshot'/name)!=h:raise ValueError('source changed: '+name)
 return lock

def joint_locks(run,audit):
 p,a=verify(run),verify(audit)
 if (p['kind']!='producer' or a['kind']!='audit'
     or a['producer_root']!=str(Path(run).resolve()) or p['audit_root']!=str(Path(audit).resolve())
     or a['audit_root']!=p['audit_root'] or a['created_unix']<p['created_unix']
     or any(p[k]!=a[k] for k in ('source_hashes','runtime','entries','bindings','ordered_query_ids'))):
  raise ValueError('joint producer/auditor freeze differs')
 return p,a
