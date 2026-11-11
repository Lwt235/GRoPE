"""Finish fixed analyses, with an explicit boundary for conditional new fits."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_soft_data import (
    OUT, SEEDS, QUERIES, load_plan, center_path, prepare_center_priors, write_json)
from experiments.gaussian_memory_soft_analysis import (
    static_analysis, integrated_analysis, conditional_decision, intervention_analysis, retention_analysis)


def analyze():
    static_analysis(); integrated_analysis(); intervention_analysis(); retention_analysis()
    result = conditional_decision()
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'phase': 'soft_read1',
        'stage': 'primary comparisons analyzed; conditional control decision available',
        'primary_training_and_evaluation_complete': True, 'phase_complete': False,
        'center_control_triggered': result['center_control_triggered']})
    return result


def center_prepare():
    assert json.loads((OUT / 'conditional_decision.json').read_text())['center_control_triggered']
    # The inherited tensor writer expects its destination directory to exist.
    # Prepare directories here without changing frozen scientific sources.
    for meta in load_plan()['catalog']:
        center_path(meta['id'], QUERIES[0]).parent.mkdir(parents=True, exist_ok=True)
    prepare_center_priors()


def final():
    condition = json.loads((OUT / 'conditional_decision.json').read_text())
    if condition['center_control_triggered']:
        for seed in SEEDS:
            for name in (f'static_soft_center_{seed}_complete.json',
                         f'integrated_soft_center_{seed}_corrective_clean_complete.json'):
                assert json.loads((OUT / name).read_text())['complete']
        static_analysis(); integrated_analysis()
    from experiments.gaussian_memory_soft_checks import main as independent_checks
    independent_checks()
    from experiments.gaussian_memory_soft_report import main as report
    report()
    write_json(OUT / 'evaluation_pipeline_complete.json', {'complete': True,
        'center_control_triggered': condition['center_control_triggered']})
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'phase': 'soft_read1',
        'stage': 'results computed; interpretation and visual review pending',
        'evaluation_complete': True, 'report_generated': True, 'phase_complete': False,
        'conclusion_review_pending': True, 'figure_review_pending': True})


def watch():
    deadline = time.monotonic() + 10800
    while True:
        states = [json.loads(p.read_text()) if p.exists() else {} for p in
                  (OUT / 'queue_primary.json', OUT / 'queue_auxiliary.json')]
        if all(s.get('complete') for s in states): break
        if any(s.get('returncode', 0) != 0 for s in states):
            raise RuntimeError('A GPU queue failed; inspect its preserved log before continuing.')
        if time.monotonic() > deadline: raise TimeoutError('GPU queues have not completed within three hours.')
        time.sleep(30)
    result = analyze()
    if result['center_control_triggered']:
        print(json.dumps({'primary_complete': True, 'next': 'prepare and run the three matched Soft-center controls'}), flush=True)
    else: final()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('analyze', 'center_prepare', 'final', 'watch'))
    a = p.parse_args(); torch.set_num_threads(4); globals()[a.stage]()
