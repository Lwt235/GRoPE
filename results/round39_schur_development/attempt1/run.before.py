import ast,hashlib,json,shutil,subprocess,time
from pathlib import Path
R=Path('/home/liwt/GRoPE');D=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
 assert not (D/'before_execution.json').exists()
 pending=['ROUND39_SCHUR_DIAGNOSTIC_PROTOCOL.md','experiments/round39_schur_diagnostic.py','experiments/round39_schur_components.py','experiments/round39_schur_components_audit.py','tests/test_round39_schur_diagnostic.py'];found=set()
 while pending:
  name=pending.pop()
  if name in found:continue
  p=R/name;assert p.is_file();found.add(name)
  if p.suffix!='.py':continue
  for node in ast.walk(ast.parse(p.read_text())):
   mods=[]
   if isinstance(node,ast.Import):mods=[n.name for n in node.names]
   elif isinstance(node,ast.ImportFrom):
    assert not node.level
    if node.module:mods=[node.module]+[node.module+'.'+n.name for n in node.names]
   for mod in mods:
    parts=mod.split('.');paths=[Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py']+[Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
    pending.extend(str(p) for p in paths if (R/p).is_file() and str(p) not in found)
 hashes={n:sha(R/n) for n in sorted(found)}
 for n,h in hashes.items():
  target=D/'source_snapshot'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,target);assert sha(target)==h
 shutil.copyfile(__file__,D/'run.before.py')
 write(D/'before_execution.json',dict(created_unix=time.time(),source_sha256=sha(__file__),sources=hashes,real_arrays_read=False,labels_read=False))
 command=['/home/liwt/miniColmap/.venv/bin/python','-c',"from experiments import round39_schur_diagnostic as D; D.OLD.C.configure_runtime(); import unittest; s=unittest.defaultTestLoader.loadTestsFromName('tests.test_round39_schur_diagnostic'); r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(not r.wasSuccessful())"]
 start=time.perf_counter()
 with (D/'tests.log').open('w') as f:proc=subprocess.run(command,cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 unchanged=all(sha(R/n)==h for n,h in hashes.items())
 write(D/'completion.json',dict(passed=proc.returncode==0 and unchanged,exit_code=proc.returncode,wall_seconds=time.perf_counter()-start,source_unchanged=unchanged,files={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file()},real_arrays_read=False,labels_read=False))
 print('R39_SCHUR_PREFLIGHT',proc.returncode,'sources',len(hashes),'unchanged',unchanged,flush=True)
 return proc.returncode or (not unchanged)
if __name__=='__main__':raise SystemExit(main())
