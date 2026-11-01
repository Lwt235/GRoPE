import hashlib,json,shutil,time
from pathlib import Path
R=Path('/home/liwt/GRoPE');D=Path(__file__).resolve().parent;O=R/'results/round39_schur_diagnostic'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
 assert not (D/'before_execution.json').exists()
 shutil.copyfile(__file__,D/'accept.before.py')
 write(D/'before_execution.json',dict(created_unix=time.time(),source_sha256=sha(__file__),labels_read=False,new_arrays_decoded=False))
 bindings={}
 def bind(p,h=None):
  p=Path(p);digest=sha(p);assert h is None or digest==h,str(p);bindings[str(p.relative_to(R))]=digest
 def complete(p):
  c=read(p/'completion.json');assert c['passed'];bind(p/'completion.json')
  files={str(x.relative_to(p)) for x in p.rglob('*') if x.is_file() and x.name!='completion.json'}
  assert files==set(c['files']),str(p)
  for n,h in c['files'].items():bind(p/n,h)
  return c
 pre=R/'results/round39_schur_development/attempt1';complete(pre)
 ex=R/'results/round39_schur_execution';ec=complete(ex);assert [s['exit_code'] for s in ec['stages']]==[0,0,0]
 for s in ec['stages']:bind(R/s['log'],s['log_sha256'])
 bind(O/'lock.json',(O/'lock.sha256').read_text().strip());bind(O/'lock.sha256');lock=read(O/'lock.json')
 assert lock['source_hashes']==read(pre/'before_execution.json')['sources']
 for n,h in lock['source_hashes'].items():bind(R/n,h);bind(O/'source_snapshot'/n,h)
 dc=complete(O/'diagnostic');ac=complete(O/'audit')
 for c in [dc,ac]:
  assert c['pairs']==48 and c['queries_per_role']==2304 and c['role_queries']==6912 and c['role_pair_checks']==144
  assert c['phase_wall_seconds']<=1200 and c['component_npz_bytes']<=512*2**20
  assert not c['labels_read'] and not c['new_probabilities']
 checks=read(O/'audit/checks.json');records=read(O/'diagnostic/records.json')
 assert [x['pair_id'] for x in records]==[e['pair_id'] for e in lock['entries']]
 assert [(x['pair_id'],x['role']) for x in checks]==[(e['pair_id'],role) for e in lock['entries'] for role in ['full','point','isotropic']]
 assert all(x['passed'] and x['queries']==48 and x['keys']==432 for x in checks)
 max_comp=max(max(x['component_errors'].values()) for x in checks);max_raw=max(x['raw_reconstruction_error'] for x in checks)
 assert max_comp<=1e-9 and max_raw<=1e-9
 v=read(O/'audit/verification.json');assert v['passed'] and v['shared_row_bookkeeping'] and v['independent_numerical_implementations']
 expected=[(e['pair_id'],role,i) for e in lock['entries'] for role in ['full','point','isotropic'] for i in range(48)]
 for stage in ['diagnostic','audit']:
  rows=read(O/stage/'rows.json');assert [(x['pair_id'],x['role'],x['query_index']) for x in rows]==expected
  summary=read(O/stage/'summary.json');assert all(summary[role]['all']['population']==2304 for role in ['full','point','isotropic'])
 for n in ['accept.py','accept.before.py','before_execution.json']:bind(D/n)
 p=R/'results/round39_schur_root_acceptance_20261101.json';assert not p.exists()
 write(p,dict(created_unix=time.time(),accepted_for_saved_output_interpretation=True,independent_numerical_implementations=True,shared_row_bookkeeping=True,role_queries=6912,queries_per_role=2304,maximum_component_error=max_comp,maximum_raw_reconstruction_error=max_raw,maximum_bookkeeping_discrepancy=v['maximum_discrepancy'],labels_read=False,new_probabilities=False,scientific_gate_passed=False,goal_complete=False,files=bindings))
 print('R39_SCHUR_ROOT_ACCEPTED',len(bindings),sha(p),flush=True)
if __name__=='__main__':main()
