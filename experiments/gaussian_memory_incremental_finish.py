"""Finish the authorized fixed experiment, preserving all intermediate artifacts."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_incremental_data import OUT, SEEDS, KINDS, write_json
from experiments.gaussian_memory_incremental_train import train
from experiments.gaussian_memory_incremental_evaluate import baseline, static, integrated, error_probe
from experiments.gaussian_memory_incremental_analysis import static_analysis, integrated_analysis, state_analysis, independent_checks
from experiments.gaussian_memory_incremental_cost import run as native_costs


def evaluate():
    start = time.monotonic()
    assert (OUT / 'train_pipeline_complete.json').exists()
    barriers = ('partial_evaluation_complete.json', 'available_seed11_evaluation_complete.json')
    while not all((OUT / name).exists() for name in barriers):
        print(json.dumps({'waiting_for': 'auxiliary evaluations to finish before shared output stages'}), flush=True)
        time.sleep(30)
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'stage': 'native output evaluation',
               'formal_training_complete': True, 'evaluation_complete': False})
    baseline(); static()
    result = static_analysis()
    if result['center_control_triggered']:
        train('center'); static('center'); static_analysis()
    integrated(); error_probe(); integrated_analysis(); state_analysis()
    native_costs()
    independent_checks()
    write_json(OUT / 'evaluation_pipeline_complete.json', {'complete': True, 'seconds': time.monotonic() - start,
               'center_control_triggered': result['center_control_triggered']})
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'stage': 'evaluation complete; report and figure review pending',
               'formal_training_complete': True, 'evaluation_complete': True, 'report_complete': False})
    from experiments.gaussian_memory_incremental_report import main as build_report
    build_report()
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'stage': 'report generated; figure and conclusion review pending',
               'formal_training_complete': True, 'evaluation_complete': True, 'report_complete': True,
               'figure_review_pending': True, 'conclusions_review_pending': True})


def tail_seed():
    """Three distant queue entries delegated to one GPU worker, no method change."""
    assert json.loads((OUT / 'combined_20261112/status.json').read_text())['complete']
    for kind in ('geometry', 'content', 'local'):
        folder = OUT / f'{kind}_20261112'
        # The main worker skips completed cases; avoid a case it has already started.
        if (folder / 'initial.pt').exists() and not (folder / 'status.json').exists():
            raise RuntimeError(f'Main worker already reached {folder}; do not duplicate it.')
        train(kind, SEEDS[-1])
    static(seed=SEEDS[-1])
    write_json(OUT / 'auxiliary_seed12_complete.json', {'complete': True, 'training_cases': list(KINDS),
               'seed': SEEDS[-1], 'static_evaluation_complete': True})


def partial_evaluate():
    # These first-seed fits finish well before this queue; use freed time for fixed evaluation.
    assert all((OUT / f'{kind}_{SEEDS[0]}/status.json').exists() for kind in KINDS)
    static(seed=SEEDS[0])
    for seed in (SEEDS[0], SEEDS[-1]):
        for kind in ('content', 'combined'):
            integrated(kind, seed)
    error_probe()
    write_json(OUT / 'partial_evaluation_complete.json', {'complete': True,
               'additional_static_seeds': [SEEDS[0]], 'integrated_seeds': [SEEDS[0], SEEDS[-1]],
               'error_probe_complete': True})


def available_seed11():
    """Evaluate already-finished seed11 cases while the final CG fit runs."""
    assert (OUT / 'partial_evaluation_complete.json').exists()
    seed = SEEDS[1]
    for kind in ('local', 'content', 'geometry'):
        assert json.loads((OUT / f'{kind}_{seed}/status.json').read_text())['complete']
        static(kind, seed)
    integrated('content', seed)
    write_json(OUT / 'available_seed11_evaluation_complete.json', {'complete': True, 'seed': seed,
               'static_kinds': ['local', 'content', 'geometry'], 'integrated_kinds': ['content']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('evaluate', 'tail_seed', 'partial_evaluate', 'available_seed11'))
    args = p.parse_args(); torch.set_num_threads(4); globals()[args.stage]()
