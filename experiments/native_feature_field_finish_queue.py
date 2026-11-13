"""Finite owned-process cost/CPU delivery queue; handoff waits for actual figure review."""
import os,time,sys,subprocess
from experiments import native_feature_field_data as d
while True:
    rows=[d.read(p) for p in (d.BASEOUT/'runtime').glob('row_*.json')]
    c=[r for r in rows if r['row']=='C' and r.get('process_exited') and r.get('exit_code')==0];j=[r for r in rows if r['row']=='J' and r.get('process_exited') and r.get('exit_code')==0]
    if c and j:break
    time.sleep(5)
# Model workers actually exited; select by current resource facts alone.
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);active={r.split(',')[0].strip() for r in apps.splitlines()};candidates=[]
for line in gpus.splitlines():
 i,uid,free,util=[s.strip() for s in line.split(',')]
 if int(free)>=16384:candidates.append((uid in active,int(util),-int(free),i))
assert candidates;gpu=min(candidates)[-1]
subprocess.run([sys.executable,'-u','-m','experiments.native_feature_field_launch','--gpu',gpu,'--mode','cost'],cwd=d.ROOT,check=True)
while not (d.BASEOUT/'factorial_summary.json').exists():time.sleep(5)
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
for module in ('experiments.native_feature_field_package','experiments.native_feature_field_delivery','experiments.native_feature_field_figures'):subprocess.run([sys.executable,'-u','-m',module],cwd=d.ROOT,env=env,check=True)
d.write(d.BASEOUT/'delivery_ready.json',dict(complete=True,actual_view_image_review_pending=True,implementation_handoff_not_yet_published=True,utc=d.now()));print('Full delivery ready for actual figure review and final handoff',flush=True)
