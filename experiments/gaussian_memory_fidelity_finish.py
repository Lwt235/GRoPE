"""Sequential evaluation of all fixed groups; never selects a model."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_fidelity_data import OUT, OLD_OUT, SEEDS, model_path, lock_models, extract
from experiments.gaussian_memory_fidelity_evaluate import evaluate
from experiments.gaussian_memory_view_read import run as standalone_read
from experiments.gaussian_memory_data import REPORT


def main():
    assert json.loads((OUT / 'training_status.json').read_text())['complete']
    stages = [
        ('development', lambda: evaluate(False)),
        ('lock_all_18_models', lock_models),
        ('extract_fixed_later_clips', extract),
        ('transfer', lambda: evaluate(True)),
        ('standalone_C', lambda: standalone_read(OLD_OUT / 'standalone/observed_input.npz',
            REPORT / 'learned_geometry/memories/val_12.pt', model_path('C', 'center', SEEDS[0]), OUT / 'standalone_C.npz'))]
    completed = []
    for name, action in stages:
        write_json(OUT / 'pipeline_status.json', {'complete': False, 'running': name, 'completed': completed})
        started = time.monotonic(); action(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        completed.append({'stage': name, 'seconds': time.monotonic()-started})
        print(json.dumps({'finished_stage': name, 'seconds': completed[-1]['seconds']}), flush=True)
    write_json(OUT / 'pipeline_status.json', {'complete': True, 'completed': completed,
               'analysis_and_visual_review_pending': True})


if __name__ == '__main__': main()
