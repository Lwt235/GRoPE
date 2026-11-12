"""Finish the predeclared diagnostics after the finite training queue, then await scientific review."""
import fcntl
import json
import os
import subprocess
import sys
import time

from experiments.gaussian_scene_data import ROOT,OUT,read
from experiments.gaussian_scene_orchestrate import occupancy
from experiments.gaussian_memory_incremental_data import write_json


def status(state,**details):
    write_json(OUT/'postprocess_state.json',dict(state=state,pid=os.getpid(),epoch=time.time(),**details))


def gpu_task(module,args):
    while True:
        states=occupancy()
        idle=[gpu for gpu,used,free,util in states if used<=512 and free>=40000 and util<=5]
        shared=[gpu for gpu,used,free,util in sorted(states,key=lambda row:row[3]) if free>=24000]
        if not idle and not shared:
            status('diagnostic_waiting_for_gpu',module=module);time.sleep(30);continue
        gpu=(idle or shared)[0];launcher='experiments.gaussian_memory_launch' if idle else 'experiments.gaussian_scene_launch'
        command=[sys.executable,'-u','-m',launcher,'--gpu',str(gpu),module,*args]
        logpath=OUT/'logs'/('post_'+module.rsplit('.',1)[-1]+('_'+args[0] if args else '')+'.log')
        status('diagnostic_running',module=module,args=args,gpu=gpu,shared=not bool(idle),log=str(logpath))
        with logpath.open('a',buffering=1) as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            if 'no job launched' in logpath.read_text()[-5000:]:time.sleep(20);continue
            raise RuntimeError(f'Diagnostic failed: {module} {args}; see{logpath}')
        return


def main():
    with (OUT/'postprocess.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            queue=read(OUT/'queue_status.json')
            if queue['state']=='all_fits_references_interventions_exported':break
            if queue['failed']:
                status('queue_failure_requires_diagnosis',failures=queue['failed']);return
            status('waiting_for_finite_training_queue',complete=queue['complete'],running=queue['running'])
            time.sleep(30)
        gpu_task('experiments.gaussian_scene_carrier',[])
        gpu_task('experiments.gaussian_scene_cost',['construction'])
        gpu_task('experiments.gaussian_scene_cost',['query'])
        for module in ('experiments.gaussian_scene_score','experiments.gaussian_scene_figures'):
            status('cpu_analysis_running',module=module)
            logpath=OUT/'logs'/f'post_{module.rsplit(".",1)[-1]}.log'
            with logpath.open('a',buffering=1) as log:
                result=subprocess.run([sys.executable,'-u','-m',module],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Analysis failed: {module}; see{logpath}')
        status('exports_scoring_figures_complete_scientific_review_pending',goal_complete=False,
               remaining=['view3figures','interpret matched results and costs','method decision and documentation','finite completion audit'])


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('postprocess_failure_requires_diagnosis',error=str(error));raise
