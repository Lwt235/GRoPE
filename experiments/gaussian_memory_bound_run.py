"""Authorized stages with disjoint seed assignments for two GPU workers."""
import argparse
import json

import numpy as np
import torch

from experiments.gaussian_memory_bound_data import OUT, SEEDS, prepare_priors, write_json
from experiments.gaussian_memory_bound_train import preflight, train, source_identity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', nargs='?', default='preflight', choices=('preflight', 'train_primary', 'train_aux'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'preflight':
        write_json(OUT / 'execution_status.json', {'goal_active': True, 'stage': 'causal prior preparation and GPU pilot',
                   'formal_training_started': False})
        prepare_priors(); preflight()
        write_json(OUT / 'execution_status.json', {'goal_active': True, 'stage': 'pilot complete; matched formal training pending',
                   'formal_training_started': False, 'pilot_complete': True})
    else:
        pilot = json.loads((OUT / 'preflight/complete.json').read_text())
        assert pilot['complete'] and pilot['interface']['source_sha256'] == source_identity()
        seeds = SEEDS[:2] if args.stage == 'train_primary' else SEEDS[2:]
        write_json(OUT / f'{args.stage}_status.json', {'complete': False, 'seeds': list(seeds)})
        for seed in seeds:
            train(seed=seed)
        write_json(OUT / f'{args.stage}_status.json', {'complete': True, 'seeds': list(seeds)})
