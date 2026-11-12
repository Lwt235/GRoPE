"""After finite training ends, produce references, timing and a draft for human-readable review."""
import json
import fcntl
import os
import subprocess
import sys
import time
import numpy as np
from experiments.gaussian_memory_coarse_data import *
from experiments.gaussian_memory_coarse_queue import occupancy


def _gpu_action_unlocked(action):
    while True:
        occ=occupancy();gpu=next((g for g in ('2','3','4','5','7') if occ[g][0]<=512 and occ[g][1]>=40000 and occ[g][2]<=5),None)
        shared=False
        if gpu is None:
            # Existing one-worker 12GiB authorization; timing additionally requires
            # near-zero observed compute occupancy. Record that it is not exclusive.
            used,free,util=occ['6']
            if free>=20000 and util<=(5 if action=='timing' else 60):gpu='6';shared=True
        if gpu is None:time.sleep(15);continue
        write_json(OUT/f'{action}_resource_context.json',dict(gpu=gpu,shared=shared,cap_GiB=12 if shared else None,before=occ,scope='Resident other process may remain if shared; descriptive timing, no exclusive benchmark claim'))
        path=OUT/'logs'/f'finish_{action}.log';path.parent.mkdir(exist_ok=True)
        with path.open('a',buffering=1) as log:
            command=[sys.executable,'-u','-m','experiments.gaussian_memory_launch','--gpu',gpu,'--wait-for-gpu',*(['--allow-shared'] if shared else []),'experiments.gaussian_memory_coarse_runtime',action]
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        record=read(OUT/f'{action}_resource_context.json');record['after']=occupancy();write_json(OUT/f'{action}_resource_context.json',record)
        return


def gpu_action(action):
    """Allow independent fixed references/cost work without duplicate GPU runs."""
    target=OUT/'evaluation/references.json' if action=='references' else OUT/'runtime.json'
    with (OUT/f'postprocess_{action}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists() and read(target)['complete']:return
        _gpu_action_unlocked(action)


def main():
    while not (OUT/'train_queue_complete.json').exists():time.sleep(15)
    assert read(OUT/'train_queue_complete.json')['complete'],'Training incomplete/failed'
    p=plan();verify(p)
    write_json(OUT/'execution_status.json',dict(state='evaluation_and_cost',finite_goal=True,new_confirmation=False))
    if not (OUT/'evaluation/references.json').exists():gpu_action('references')
    if not (OUT/'runtime.json').exists():gpu_action('timing')
    subprocess.run([sys.executable,'-u','-m','experiments.gaussian_memory_coarse_analysis','analyze'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-u','-m','experiments.gaussian_memory_coarse_report'],cwd=ROOT,check=True)
    write_json(OUT/'phase_completion.json',dict(compute_complete=True,analysis_complete=True,visual_review_complete=False,method_decision_complete=False,overall_research_objective_achieved=False,new_confirmation=False))
    print(json.dumps({'compute_complete':True,'manual_result_and_figure_review_remaining':True}),flush=True)


if __name__=='__main__':main()
