"""Finish fixed finite CPU summaries; actual scientific and visual review remain explicit."""
import fcntl
import os
import subprocess
import sys
import time
from experiments.source_geometry_data import *

def status(state,**kw):write_json(OUT/'postprocess_state.json',dict(state=state,pid=os.getpid(),epoch=time.time(),goal_complete=False,**kw))

def main():
    with (OUT/'postprocess.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            q=read(OUT/'queue_status.json')
            if q['failed']:status('queue_failure_requires_diagnosis',failures=q['failed']);return
            if q['state']=='all_fits_and_diagnostics_complete':break
            status('waiting_for_fixed_queue',completed=q['complete'],running=q['running']);time.sleep(30)
        for module in ['source_geometry_score','source_geometry_training_summary','source_geometry_final_checks','source_geometry_figures','source_geometry_report_tables']:
            status('cpu_postprocess_running',module=module)
            log=OUT/'logs'/f'post_{module}.log'
            with log.open('a',buffering=1) as stream:
                r=subprocess.run([sys.executable,'-u','-m','experiments.'+module],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(f'{module} failed; see{log}')
        status('fixed_results_ready_for_actual_review',remaining=['actual review of all3 figures','interpret quality manipulation and PE interaction','method decision and research docs','finite completion audit'])

if __name__=='__main__':
    try:main()
    except Exception as e:status('postprocess_error_requires_diagnosis',error=str(e));raise
