import hashlib,json,shutil,subprocess,time
from pathlib import Path
R=Path('/home/liwt/GRoPE');D=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
 assert not (D/'before_execution.json').exists()
 pre=R/'results/round39_schur_development/attempt1';c=read(pre/'completion.json');assert c['passed']
 for n,h in c['files'].items():assert sha(pre/n)==h,n
 sources=read(pre/'before_execution.json')['sources']
 for n,h in sources.items():assert sha(R/n)==h,n
 shutil.copyfile(__file__,D/'run.before.py')
 write(D/'before_execution.json',dict(created_unix=time.time(),controller_sha256=sha(__file__),sources=sources,preflight_completion_sha256=sha(pre/'completion.json'),stages=['freeze','diagnostic','audit'],labels_authorized=False,new_probabilities_authorized=False))
 stages=[];code=1
 try:
  for stage in ['freeze','diagnostic','audit']:
   log=R/('results/round39_schur_'+stage+'_20261101.log');start=time.perf_counter()
   with log.open('w') as f:proc=subprocess.run(['/home/liwt/miniColmap/.venv/bin/python','-m','experiments.round39_schur_diagnostic',stage],cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=1260)
   stages.append(dict(stage=stage,exit_code=proc.returncode,wall_seconds=time.perf_counter()-start,log=str(log.relative_to(R)),log_sha256=sha(log)))
   write(D/'stages.json',stages);print('R39_SCHUR_STAGE',stage,proc.returncode,flush=True)
   if proc.returncode:code=proc.returncode;return code
  code=0;return code
 finally:
  write(D/'completion.json',dict(passed=code==0,exit_code=code,stages=stages,finished_unix=time.time(),files={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file()},labels_read=False,new_probabilities=False,goal_complete=False))
if __name__=='__main__':raise SystemExit(main())
