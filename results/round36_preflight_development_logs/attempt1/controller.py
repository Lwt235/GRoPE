from pathlib import Path
import ast,hashlib,json,os,subprocess,time
ROOT=Path('/home/liwt/GRoPE')
LOGS=ROOT/'results/round36_preflight_development_logs'
OUT=LOGS/'attempt1'
assert not OUT.exists(),'preserve every attempt'
OUT.mkdir(parents=True)
SELF=Path(__file__)
(OUT/'controller.py').write_bytes(SELF.read_bytes())
contract=ROOT/'experiments/round36_component_round_contract.py'
tree=ast.parse(contract.read_text())
roots=[]
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in n.targets):
        roots.extend(ast.literal_eval(n.value))
roots+=['experiments/round34_log_narrow_preflight.py']
seen=set();missing=[];pending=list(roots)
while pending:
    name=pending.pop()
    if name in seen:continue
    path=ROOT/name
    if not path.is_file():
        missing.append(name);continue
    seen.add(name)
    if path.suffix!='.py':continue
    for node in ast.walk(ast.parse(path.read_text())):
        mods=[]
        if isinstance(node,ast.Import):mods=[n.name for n in node.names]
        elif isinstance(node,ast.ImportFrom) and node.module:
            mods=[node.module]+[node.module+'.'+n.name for n in node.names]
        for mod in mods:
            parts=mod.split('.')
            candidates=[Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py']
            candidates += [Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
            pending.extend(str(p) for p in candidates if (ROOT/p).is_file())
hashes={}
for name in sorted(seen):
    payload=(ROOT/name).read_bytes();p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
    hashes[name]=hashlib.sha256(payload).hexdigest()
required=('experiments/round36_component_round_reference.py','experiments/round36_component_round_preflight.py','tests/test_round36_component_round_preflight.py')
assert all(n in hashes for n in required),'wait for owned reference import to exist'
code='''import numpy
from experiments import round36_component_round_reference as reference
def forbidden_integrate(*args, **kwargs):
    raise AssertionError("mock-only runner tests must not execute any reference integral")
reference.integrate = forbidden_integrate
import pytest
raise SystemExit(pytest.main(["-q", "tests/test_round36_component_round_preflight.py"]))
'''
cmd=['/home/liwt/miniColmap/.venv/bin/python','-c',code]
env=os.environ.copy();threads={k:'4' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')};env.update(threads)
start=time.time()
before=dict(attempt=1,created_unix=start,command=cmd,cwd=str(ROOT),thread_environment=threads,
 source_hashes=hashes,source_files_not_yet_available=sorted(set(missing)),
 controller_sha256=hashlib.sha256(SELF.read_bytes()).hexdigest(),
 scope='Mock-only R36 runner tests; integrate replaced by raising sentinel before test collection',
 formal_integrations=0,real_inputs_read=False,labels_read=False)
(OUT/'before_execution.json').write_text(json.dumps(before,indent=2,sort_keys=True)+'\n')
with (OUT/'pytest.log').open('w') as log:
    result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
finish=time.time()
changed=[name for name,digest in hashes.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
receipt=dict(exit_code=result.returncode,started_unix=start,finished_unix=finish,
 source_changes_during_tests=changed,pytest_log_sha256=hashlib.sha256((OUT/'pytest.log').read_bytes()).hexdigest(),
 snapshot_before_test=True,formal_integrations=0,real_inputs_read=False,labels_read=False)
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print((OUT/'pytest.log').read_text())
print(json.dumps(receipt,sort_keys=True))
raise SystemExit(result.returncode)
