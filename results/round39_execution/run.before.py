import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
ROOT=Path('/home/liwt/GRoPE'); HERE=Path(__file__).resolve().parent
PYTHON='/home/liwt/miniColmap/.venv/bin/python'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def main():
 if (HERE/'before_execution.json').exists(): raise RuntimeError('preserve execution attempt')
 pre=ROOT/'results/round39_development/attempt1'
 done=read(pre/'completion.json'); assert done['passed']
 for n,h in done['files'].items(): assert sha(pre/n)==h,n
 sources=read(pre/'before_execution.json')['sources']
 for n,h in sources.items(): assert sha(ROOT/n)==h,n
 shutil.copyfile(__file__,HERE/'run.before.py')
 write(HERE/'before_execution.json',dict(created_unix=time.time(),controller_sha256=sha(__file__),sources=sources,preflight_completion_sha256=sha(pre/'completion.json'),stages=['freeze','forward','audit'],labels_authorized=False))
 results=[]; code=1
 try:
  for stage in ['freeze','forward','audit']:
   start=time.perf_counter()
   command=[PYTHON,'-m','experiments.round39_dual_overlap_screen',stage]
   path=ROOT/('results/round39_'+stage+'_20261101.log')
   with path.open('w') as f:
    proc=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1260)
   results.append(dict(stage=stage,exit_code=proc.returncode,wall_seconds=time.perf_counter()-start,log=str(path.relative_to(ROOT)),log_sha256=sha(path)))
   write(HERE/'stage_results.json',results)
   print('R39_STAGE',stage,proc.returncode,flush=True)
   if proc.returncode: code=proc.returncode; return code
  code=0
  return code
 finally:
  write(HERE/'completion.json',dict(passed=code==0,exit_code=code,stages=results,finished_unix=time.time(),files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()},labels_read=False,goal_complete=False))
if __name__=='__main__': raise SystemExit(main())
