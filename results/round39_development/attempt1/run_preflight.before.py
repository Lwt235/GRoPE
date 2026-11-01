import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
ROOT=Path('/home/liwt/GRoPE')
HERE=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def main():
 if (HERE/'before_execution.json').exists(): raise RuntimeError('retain old attempt')
 todo=['ROUND39_DUAL_OVERLAP_PROTOCOL.md','models/round39_dual_overlap_pe.py','experiments/round39_dual_overlap_screen.py','experiments/round39_overlap_audit_numeric.py','tests/test_round39_dual_overlap.py','docs/gaussian_overlap_measure_pe_reserve_20261101.md','docs/research_gate_map_20261101.md']
 found=set()
 while todo:
  name=todo.pop()
  if name in found: continue
  p=ROOT/name
  assert p.is_file(),name
  found.add(name)
  if p.suffix!='.py': continue
  for node in ast.walk(ast.parse(p.read_text())):
   mods=[]
   if isinstance(node,ast.Import): mods=[x.name for x in node.names]
   elif isinstance(node,ast.ImportFrom):
    assert not node.level
    if node.module: mods=[node.module]+[node.module+'.'+x.name for x in node.names]
   for mod in mods:
    parts=mod.split('.')
    paths=[Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py']+[Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
    todo.extend(str(p) for p in paths if (ROOT/p).is_file() and str(p) not in found)
 hashes={n:sha(ROOT/n) for n in sorted(found)}
 for n,h in hashes.items():
  dst=HERE/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dst);assert sha(dst)==h
 shutil.copyfile(__file__,HERE/'run_preflight.before.py')
 write(HERE/'before_execution.json',dict(created_unix=time.time(),sources=hashes,controller_sha256=sha(__file__),protocol_sha256=sha(ROOT/'ROUND39_DUAL_OVERLAP_PROTOCOL.md'),arrays_read=False,labels_read=False))
 cmd=['/home/liwt/miniColmap/.venv/bin/python','-c',"from experiments import round39_dual_overlap_screen as S; S.C.configure_runtime(); import unittest; suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_round39_dual_overlap'); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"]
 started=time.perf_counter()
 with (HERE/'tests.log').open('w') as f:
  result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 unchanged=all(sha(ROOT/n)==h for n,h in hashes.items())
 write(HERE/'completion.json',dict(exit_code=result.returncode,passed=result.returncode==0 and unchanged,source_unchanged=unchanged,wall_seconds=time.perf_counter()-started,files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()},arrays_read=False,labels_read=False))
 print('R39_PREFLIGHT',result.returncode,'sources',len(hashes),'unchanged',unchanged,flush=True)
 raise SystemExit(result.returncode or (not unchanged))
if __name__=='__main__': main()
