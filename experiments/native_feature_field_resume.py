"""Resume only incomplete C after J releases its single shared GPU lease."""
import time,sys,subprocess
from experiments import native_feature_field_data as d
while True:
    rows=[d.read(p) for p in (d.BASEOUT/'runtime').glob('row_J_*.json')]
    if rows and all(r.get('process_exited') for r in rows):
        assert rows[-1]['exit_code']==0,rows[-1];break
    time.sleep(5)
print('J actual process exited; resume C from existing exact checkpoint',flush=True)
code=subprocess.call([sys.executable,'-u','-m','experiments.native_feature_field_launch','--gpu','1','--mode','row','--row','C'],cwd=d.ROOT)
raise SystemExit(code)
