"""Complete fixed evaluations and the prospectively conditional branches."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_bound_data import OUT, SEEDS, KINDS, prepare_priors, write_json
from experiments.gaussian_memory_bound_train import train
from experiments.gaussian_memory_bound_evaluate import static, integrated, interventions, error_probe
from experiments.gaussian_memory_bound_analysis import static_analysis, integrated_analysis, intervention_analysis, error_analysis
from experiments.gaussian_memory_bound_checks import main as independent_checks
from experiments.gaussian_memory_bound_cost import run as native_costs


def seed_evaluation(seed):
    assert seed in SEEDS
    for kind in KINDS:
        assert json.loads((OUT / f'{kind}_{seed}/status.json').read_text())['complete']
    start = time.monotonic()
    static(seed=seed)
    for kind in ('coverage', 'bound'):
        integrated(kind=kind, seed=seed)
    if seed == SEEDS[0]:
        interventions()
    write_json(OUT / f'evaluation_seed_{seed}_complete.json', {
        'complete': True, 'seed': seed, 'seconds': time.monotonic() - start,
        'static_kinds': list(KINDS), 'integrated_kinds': ['coverage', 'bound'],
        'binding_interventions': seed == SEEDS[0]})


def final_evaluation():
    for seed in SEEDS:
        assert json.loads((OUT / f'evaluation_seed_{seed}_complete.json').read_text())['complete']
    start = time.monotonic()
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'stage': 'fixed analysis and conditional controls',
               'formal_training_complete': True, 'primary_evaluation_complete': True, 'phase_complete': False})
    result = static_analysis(); intervention_analysis(); integrated_analysis()
    if result['center_control_triggered']:
        prepare_priors(center=True)
        train(kind='center'); static(kind='center'); static_analysis()
    if result['error_probe_triggered']:
        error_probe()
    error_analysis(); native_costs(); independent_checks()
    write_json(OUT / 'evaluation_pipeline_complete.json', {
        'complete': True, 'seconds_final_stage': time.monotonic() - start,
        'center_control_triggered': result['center_control_triggered'],
        'error_probe_triggered': result['error_probe_triggered']})
    from experiments.gaussian_memory_bound_report import main as build_report
    build_report()
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'stage': 'results computed; interpretation and visual review pending',
               'formal_training_complete': True, 'evaluation_complete': True, 'report_generated': True,
               'phase_complete': False, 'conclusion_review_pending': True, 'figure_review_pending': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('seed', 'final'))
    p.add_argument('--seed', type=int); args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'seed': seed_evaluation(args.seed)
    else: final_evaluation()
