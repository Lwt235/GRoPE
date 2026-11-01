import hashlib,json,shutil,subprocess,time
from pathlib import Path
R=Path('/home/liwt/GRoPE');D=Path(__file__).resolve().parent;O=R/'results/round39_dual_overlap'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
 assert not (D/'before_execution.json').exists()
 receipt=read(O/'prelabel_root_acceptance.json');assert receipt['accepted_for_label_stage'] and receipt['numerical_audit_passed']
 for n,h in receipt['files'].items():assert sha(R/n)==h,n
 shutil.copyfile(__file__,D/'run.before.py')
 write(D/'before_execution.json',dict(created_unix=time.time(),source_sha256=sha(__file__),prelabel_receipt_sha256=sha(O/'prelabel_root_acceptance.json'),stages=['evaluate','metrics_audit'],first_r39_label_access_after_root_acceptance=True))
 stages=[];code=1
 try:
  for stage in ['evaluate','metrics_audit']:
   path=R/('results/round39_'+stage+'_20261101.log');start=time.perf_counter()
   with path.open('w') as f:proc=subprocess.run(['/home/liwt/miniColmap/.venv/bin/python','-m','experiments.round39_dual_overlap_screen',stage],cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=1260)
   stages.append(dict(stage=stage,exit_code=proc.returncode,wall_seconds=time.perf_counter()-start,log=str(path.relative_to(R)),log_sha256=sha(path)))
   write(D/'stages.json',stages);print('R39_LABEL_STAGE',stage,proc.returncode,flush=True)
   if proc.returncode:code=proc.returncode;return code
  code=0;return code
 finally:
  write(D/'completion.json',dict(passed=code==0,exit_code=code,finished_unix=time.time(),stages=stages,files={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file()},goal_complete=False))
if __name__=='__main__':raise SystemExit(main())
