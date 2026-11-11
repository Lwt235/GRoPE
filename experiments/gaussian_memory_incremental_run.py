"""Run authorized stages sequentially within one occupancy-checked GPU worker."""
import argparse
import time

import numpy as np
import torch

from experiments.gaussian_memory_incremental_data import OUT, extract, training_statistics, write_json
from experiments.gaussian_memory_incremental_train import train


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('train_pipeline',))
    p.parse_args()
    start = time.monotonic()
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'stage': 'causal extraction, then formal training',
               'pilot_complete': True, 'formal_training_started': False, 'worker_memory_limit_GiB': 12})
    extract('all')
    training_statistics()
    torch.cuda.empty_cache()
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'stage': 'matched formal training',
               'pilot_complete': True, 'formal_training_started': True, 'worker_memory_limit_GiB': 12})
    train()
    write_json(OUT / 'train_pipeline_complete.json', {'complete': True, 'seconds': time.monotonic() - start})
    write_json(OUT / 'execution_status.json', {'goal_active': True, 'stage': 'formal training complete; evaluation pending',
               'pilot_complete': True, 'formal_training_complete': True, 'evaluation_complete': False})


if __name__ == '__main__': main()
