import faulthandler,os,sys,time
from pathlib import Path
sys.path.insert(0, '/home/liwt/GRoPE')
from experiments.region_confirmation_metric_runtime import RUNTIME, report
from experiments.region_confirmation_common import write, sha
identity=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],source_sha256=sha(__file__),started_unix=time.time(),plan_sha256=sha(RUNTIME/'plan.json'))
assert not (RUNTIME/'execution_identity.json').exists()
write(RUNTIME/'execution_identity.json',identity)
write(RUNTIME/'status.json',dict(identity,state='running',stage='report',goal_complete=False))
faulthandler.dump_traceback_later(45,repeat=True)
try:
    report()
    write(RUNTIME/'status.json',dict(identity,state='complete',stage='report',completed_unix=time.time(),scientific_review_pending=True,visual_review_pending=True,goal_complete=False))
except BaseException as error:
    write(RUNTIME/'status.json',dict(identity,state='failed',stage='report',error_type=type(error).__name__,error=str(error),observed_unix=time.time(),goal_complete=False))
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
