import tempfile,json
from pathlib import Path
from experiments import native_render_parameter_resource_sidecar as m
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);m.CACHE=root/'cache';m.OUT=root/'out';m.RAM=root/'ram'
 for p in [m.CACHE,m.OUT,m.RAM]:p.mkdir()
 m.write(m.OUT/'budget.json',dict(formal_updates=12041,nontraining_forwards=4242))
 m.runtime=lambda:[]
 def available(p):return 100*m.GIB-sum(x.stat().st_blocks*512 for x in m.CACHE.rglob('*') if x.is_file()) if p==m.CACHE else 100*m.GIB
 m.free=available
 final=m.CACHE/'finals/P_1/step_2000.pt';pred=m.CACHE/'predictions/train_P_1.pt';final.parent.mkdir(parents=True);pred.parent.mkdir(parents=True)
 a=dict(baseline_artifact_accounting=dict(remaining_artifacts=[dict(kind='final_weight',arm='P',seed=1,path=str(final),expected_bytes=8192),dict(kind='predictions',arm='P',seed=1,cohort='train',path=str(pred),expected_bytes=8192)]));needs={};values=[]
 def check(label):
  r=m.ledger(a,needs);values.append((label,r['H'],r['R'],r['T_remaining']));assert r['H']==values[0][1];return r
 check('before')
 final.with_suffix('.tmp').write_bytes(b'x'*4096);check('partial_final')
 final.with_suffix('.tmp').write_bytes(b'x'*8192);check('full_final_temp')
 final.with_suffix('.tmp').replace(final);check('rename_without_marker')
 m.write(m.OUT/'P_1/status.json',dict(complete=True,bytes=8192,checkpoint_sha256='a'*64,checkpoint_path=str(final)));check('verified_final')
 partial=pred.with_name(pred.stem+'_partial.pt');partial.write_bytes(b'x'*4096);check('prediction_partial')
 pred.with_suffix('.tmp').write_bytes(b'x'*8192);check('publication_plus_partial')
 pred.with_suffix('.tmp').replace(pred);m.write(m.OUT/'evaluation/train_P_1.json',dict(complete=True,prediction_sha256='b'*64,prediction_path=str(pred)));check('verified_prediction_plus_partial')
 partial.unlink();check('transient_retired')
 print(json.dumps(dict(complete=True,checks=values,no_forward_calls=True,ledger_handles_allocation_rename_verified_marker_and_partial=True),indent=2))
