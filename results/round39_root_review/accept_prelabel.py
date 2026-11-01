import hashlib,json,shutil,time
from pathlib import Path
R=Path('/home/liwt/GRoPE');D=Path(__file__).resolve().parent;O=R/'results/round39_dual_overlap'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
 assert not (D/'before_execution.json').exists()
 shutil.copyfile(__file__,D/'accept_prelabel.before.py')
 write(D/'before_execution.json',dict(created_unix=time.time(),source_sha256=sha(__file__),labels_read=False))
 bindings={}
 def bind(p,h=None):
  p=Path(p);digest=sha(p);assert h is None or digest==h,str(p);bindings[str(p.relative_to(R))]=digest
 def complete(p):
  c=read(p/'completion.json');assert c['passed'];bind(p/'completion.json')
  actual={str(x.relative_to(p)) for x in p.rglob('*') if x.is_file() and x.name!='completion.json'}
  assert actual==set(c['files']),str(p)
  for n,h in c['files'].items():bind(p/n,h)
  return c
 pre=R/'results/round39_development/attempt1';complete(pre)
 execute=R/'results/round39_execution';ec=complete(execute)
 assert [s['exit_code'] for s in ec['stages']]==[0,0,0]
 for s in ec['stages']:bind(R/s['log'],s['log_sha256'])
 lock=read(O/'lock.json');bind(O/'lock.json',(O/'lock.sha256').read_text().strip());bind(O/'lock.sha256')
 for n,h in lock['source_hashes'].items():bind(R/n,h);bind(O/'source_snapshot'/n,h)
 assert lock['source_hashes']==read(pre/'before_execution.json')['sources']
 forward=complete(O/'forward');audit=complete(O/'audit')
 assert forward['pairs']==audit['pairs']==48 and forward['queries']==audit['queries']==2304
 assert audit['role_pair_checks']==144 and not forward['labels_read'] and not audit['labels_read']
 records=read(O/'forward/records.json');checks=read(O/'audit/checks.json')
 ids=[e['pair_id'] for e in lock['entries']]
 assert [r['pair_id'] for r in records]==ids
 assert [(r['pair_id'],r['role']) for r in checks]==[(i,role) for i in ids for role in ['full','point','isotropic']]
 assert all(r['passed'] and r['queries']==48 and r['keys']==432 and r['winner_exact'] for r in checks)
 limits=dict(geometry=max(r['geometry_error'] for r in checks),probability=max(r['probability_error'] for r in checks),score=max(max(r['score_error'].values()) for r in checks),normalization=max(r['normalization_error'] for r in checks))
 assert limits['geometry']<=1e-10 and limits['probability']<=1e-10 and limits['score']<=1e-9 and limits['normalization']<=2e-12
 for n in ['accept_prelabel.py','accept_prelabel.before.py','before_execution.json']:bind(D/n)
 write(O/'prelabel_root_acceptance.json',dict(created_unix=time.time(),accepted_for_label_stage=True,pairs=48,queries=2304,role_pair_checks=144,numerical_audit_passed=True,labels_read=False,scientific_gate_passed=False,goal_complete=False,maximum_errors=limits,files=bindings))
 print('R39_PRELABEL_ACCEPTED',len(bindings),sha(O/'prelabel_root_acceptance.json'))
if __name__=='__main__':main()
