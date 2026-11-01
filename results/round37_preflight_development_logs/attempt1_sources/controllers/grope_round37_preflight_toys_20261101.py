import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import ast,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/liwt/GRoPE');sys.path.insert(0,str(ROOT))
from experiments import round37_component_round_real_contract as C
OUT=ROOT/'results/round37_preflight_development_logs';OUT.mkdir(exist_ok=True)
number=max([int(p.stem.removeprefix('attempt')) for p in OUT.glob('attempt*.json')],default=0)+1;stem=f'attempt{number}'
snapshot=OUT/(stem+'_sources');snapshot.mkdir()
found=set();missing=set();pending=list(C.SOURCE_FILES)
while pending:
 name=pending.pop()
 if name in found or name in missing:continue
 path=ROOT/name
 if not path.is_file():missing.add(name);continue
 found.add(name)
 if path.suffix!='.py':continue
 for node in ast.walk(ast.parse(path.read_text())):
  names=[]
  if isinstance(node,ast.Import):names=[n.name for n in node.names]
  elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]+[node.module+'.'+n.name for n in node.names]
  for module in names:
   parts=module.split('.')
   paths=[Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py']+[Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
   pending.extend(str(p) for p in paths if (ROOT/p).is_file())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes={}
for name in sorted(found):
 dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest);hashes[name]=sha(dest)
controller_hashes={}
for name in ('/tmp/grope_round37_preflight_toys_20261101.py','/tmp/grope_round37_preflight_toy_child_20261101.py'):
 dest=snapshot/'controllers'/Path(name).name;dest.parent.mkdir(exist_ok=True);shutil.copy2(name,dest);controller_hashes[name]=sha(dest)
command=[sys.executable,'/tmp/grope_round37_preflight_toy_child_20261101.py']
record=dict(attempt=number,started_unix=time.time(),command=command,source_hashes=hashes,controller_hashes=controller_hashes,
 source_snapshot=str(snapshot.relative_to(ROOT)),missing_joint_sources=sorted(missing),
 real_reference_replaced_by_precollection_raising_sentinel=True,formal_array_decode_guard=True,
 formal_inputs_read=False,real_reference_integrations=0,labels_read=False)
(OUT/(stem+'.json')).write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
with (OUT/(stem+'.log')).open('w') as log:result=subprocess.run(command,cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT)
record.update(finished_unix=time.time(),exit_code=result.returncode,log_sha256=sha(OUT/(stem+'.log')),source_hashes_after={name:sha(ROOT/name) for name in found})
record['owned_sources_unchanged']=all(record['source_hashes_after'][n]==hashes[n] for n in ('experiments/round37_component_round_real_preflight.py','tests/test_round37_component_round_real_preflight.py'))
(OUT/(stem+'.json')).write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print((OUT/(stem+'.log')).read_text());print(OUT/(stem+'.json'));sys.exit(result.returncode)
