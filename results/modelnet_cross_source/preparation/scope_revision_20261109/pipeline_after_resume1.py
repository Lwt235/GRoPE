"""Finish the fixed experiment after the already-running authenticated download.

Never restart an unknown/failed process. Every stage preserves its output and
source identity. Machine results finish in review_pending, not scientific pass.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/liwt/GRoPE')
OUT = ROOT/'results/modelnet_cross_source/attempt1'
HERE = Path(__file__).resolve().parent
STATE = OUT/'pipeline_resume1_state.json'


def read(path):
    return json.loads(path.read_text())


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return dict(pid=pid,start_ticks=int(fields[19]),state=fields[0],command=command)
    except FileNotFoundError:
        return None


def write_state(**fields):
    state.update(fields,updated_unix=time.time())
    pending = STATE.with_suffix('.pending')
    pending.write_text(json.dumps(state,indent=2)+'\n')
    pending.replace(STATE)


if STATE.exists():
    raise ValueError('preserve prior pipeline execution; inspect before recovery')
download = read(OUT/'download_resume1_runtime.json')
identity = process(download['pid'])
if identity is None or identity['state']=='Z' or 'resume_transport1.py' not in identity['command']:
    raise ValueError('expected the specific download process to be live at attachment')
state = dict(pid=os.getpid(),started_unix=time.time(),download_process=identity,stages=[],
             lock_sha256=read(OUT/'download_resume1_runtime.json')['lock_sha256'],training_updates=0,
             scientific_review_complete=False,overall_goal_complete=False)
write_state(stage='wait_download',state='running')
print('PIPELINE_ATTACHED',identity['pid'],identity['start_ticks'],flush=True)
try:
    while True:
        current = read(OUT/'download_resume1_runtime.json')
        observed = process(identity['pid'])
        same = observed is not None and observed['start_ticks']==identity['start_ticks'] and observed['state']!='Z'
        if current['state']=='failed':
            raise RuntimeError('download reported failure; preserve partial data for review')
        if not same:
            if current['state']=='complete':
                break
            raise RuntimeError('original download process ended without completion record')
        time.sleep(15)
    commands = [
        ('prepare',[sys.executable,'-u','-m','experiments.modelnet_cross_source','prepare']),
        ('export',[sys.executable,'-u',str(HERE/'gpu_driver.py'),'export']),
        ('report',[sys.executable,'-u','-m','experiments.modelnet_cross_source','report']),
    ]
    environment=os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name,command in commands:
        log=OUT/(name+'_pipeline.log')
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            stage=dict(name=name,pid=child.pid,started_unix=time.time(),command=command,log=str(log))
            state['stages'].append(stage)
            write_state(stage=name,state='running')
            print('STAGE_STARTED',name,child.pid,flush=True)
            # Resource absence is terminal here. The goal worker can review and
            # resume explicitly; it is never treated as a successful evaluation.
            code=child.wait()
            stage.update(exit_code=code,finished_unix=time.time())
            write_state()
            if code!=0:
                raise RuntimeError(name+' stage failed; inspect original log and process')
    write_state(stage='review_pending',state='complete',finished_unix=time.time())
    print('FIXED_EVALUATION_COMPLETE_REVIEW_PENDING',flush=True)
except Exception as error:
    write_state(state='failed',finished_unix=time.time(),error_type=type(error).__name__,reason=str(error))
    print('PIPELINE_FAILED',type(error).__name__,str(error),flush=True)
    raise SystemExit(1) from None
