"""Finish the fixed experiment; keep interpretation and visual review explicit."""
import argparse
import fcntl
import json
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT
from experiments.gaussian_memory_state_data import OUT, SEEDS, POLICIES, write_json
from experiments.gaussian_memory_state_analysis import (
    analyze_policy, conditional_decision, state_range, interventions, retention)


def queues():
    return [json.loads(p.read_text()) if p.exists() else {} for p in
            (OUT / 'queue_content.json', OUT / 'queue_soft.json')]


def run_center(gpu):
    assert json.loads((OUT / 'conditional_decision.json').read_text())['center_control_triggered']
    train = 'experiments.gaussian_memory_state_train'; evaluate = 'experiments.gaussian_memory_state_evaluate'
    tasks = []
    for seed in SEEDS:
        tasks += [[train, 'train', '--kind', 'soft_center', '--seed', str(seed)],
                  [evaluate, 'evaluate', '--kind', 'soft_center', '--seed', str(seed)]]
    path = OUT / 'center_queue.json'; old = json.loads(path.read_text()) if path.exists() else {}
    if old: assert old['tasks'] == tasks and old['gpu'] == gpu
    done = old.get('completed_tasks', [])
    state = {'tasks': tasks, 'gpu': gpu, 'completed_tasks': done, 'complete': False}
    with (OUT / 'center_queue.log').open('a', buffering=1) as log:
        for index, task in enumerate(tasks):
            if index in done: continue
            state.update(current_task=index, current_command=task); write_json(path, state)
            command = [sys.executable, '-u', '-m', 'experiments.gaussian_memory_launch', '--gpu', gpu,
                       '--allow-shared', '--wait-for-gpu', *task]
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(returncode=result.returncode, failed_task=index); write_json(path, state)
                raise RuntimeError('Conditional Center worker failed; inspect its preserved log.')
            done.append(index); state['completed_tasks'] = done; write_json(path, state)
    state.update(complete=True); write_json(path, state)


def final():
    for policy in POLICIES: analyze_policy(policy)
    decision = conditional_decision(); state_range(); interventions(); retention()
    if decision['center_control_triggered']:
        assert all(json.loads((OUT / f'read_soft_center_{s}_{p}_complete.json').read_text())['complete']
                   for s in SEEDS for p in ('static', 'corrective_clean'))
    from experiments.gaussian_memory_state_checks import main as independent_checks
    independent_checks()
    from experiments.gaussian_memory_state_report import main as report
    report()
    write_json(OUT / 'evaluation_pipeline_complete.json', {'complete': True,
               'center_control_triggered': decision['center_control_triggered']})
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'phase': 'state_read1',
        'stage': 'computed results; interpretation and actual visual review pending',
        'evaluation_complete': True, 'report_generated': True, 'phase_complete': False,
        'conclusion_review_pending': True, 'figure_review_pending': True})


def watch(gpu):
    lock = (OUT / 'finish_worker.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + 14400; primary_analyzed = (OUT / 'conditional_decision.json').exists()
    while True:
        states = queues()
        if any(s.get('returncode', 0) != 0 for s in states):
            raise RuntimeError('A GPU queue failed; inspect its preserved log.')
        ready = all((OUT / f'read_{k}_{s}_corrective_clean_complete.json').exists()
                    for k in ('content', 'soft') for s in SEEDS)
        if ready and not primary_analyzed:
            analyze_policy('corrective_clean'); decision = conditional_decision(); primary_analyzed = True
            write_json(OUT / 'execution_status.json', {'goal_active': True, 'phase': 'state_read1',
                'stage': 'Corrective primary comparison analyzed; remaining fixed evaluations running',
                'center_control_triggered': decision['center_control_triggered'], 'phase_complete': False})
        if primary_analyzed:
            decision = json.loads((OUT / 'conditional_decision.json').read_text())
            center_path = OUT / 'center_queue.json'
            center_done = center_path.exists() and json.loads(center_path.read_text()).get('complete', False)
            # The control depends on the primary result, not on evaluations on
            # another card. Reuse this card only after its own main queue ends.
            owners = [s for s in states if s.get('gpu') == gpu]
            card_released = bool(owners) and all(s.get('complete') for s in owners)
            if decision['center_control_triggered'] and not center_done and card_released:
                write_json(OUT / 'execution_status.json', {'goal_active': True, 'phase': 'state_read1',
                    'stage': 'conditional Center on released main-worker GPU; other fixed evaluations may continue',
                    'center_control_triggered': True, 'center_gpu': gpu, 'phase_complete': False})
                run_center(gpu)
        if all(s.get('complete') for s in states): break
        if time.monotonic() > deadline: raise TimeoutError('State-read queues have not completed within four hours.')
        time.sleep(30)
    if not primary_analyzed: analyze_policy('corrective_clean'); conditional_decision()
    center_path = OUT / 'center_queue.json'
    center_done = center_path.exists() and json.loads(center_path.read_text()).get('complete', False)
    if json.loads((OUT / 'conditional_decision.json').read_text())['center_control_triggered'] and not center_done:
        run_center(gpu)
    final()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('watch', 'final'))
    p.add_argument('--center-gpu', default='1'); args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'watch': watch(args.center_gpu)
    else: final()
