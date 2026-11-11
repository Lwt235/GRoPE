"""One bounded GPU worker per seed, with disjoint output ownership."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_history_data import OUT, SEEDS, write_json
from experiments.gaussian_memory_history_evaluate import evaluate, retention, interventions


def main():
    p = argparse.ArgumentParser(); p.add_argument('--seed', required=True, type=int, choices=SEEDS)
    args = p.parse_args(); start = time.monotonic(); torch.set_num_threads(4)
    assert json.loads((OUT / 'states_complete.json').read_text())['complete']
    assert json.loads((OUT / 'baseline_reuse.json').read_text())['complete']
    try:
        evaluate(args.seed)
        retention(args.seed)
        if args.seed == SEEDS[0]: interventions()
    except Exception as error:
        write_json(OUT / f'worker_{args.seed}.json', {'complete': False, 'error': repr(error), 'seconds': time.monotonic() - start})
        raise
    write_json(OUT / f'worker_{args.seed}.json', {'complete': True, 'seed': args.seed,
        'seconds': time.monotonic() - start, 'new_predictions': 3228 if args.seed == SEEDS[0] else 2940})


if __name__ == '__main__': main()
