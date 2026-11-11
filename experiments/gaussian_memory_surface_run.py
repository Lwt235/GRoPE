"""Bounded label preparation, training and evaluation; no model search."""
import gc
import json
import time
import traceback

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_surface_data import OUT, prepare
from experiments.gaussian_memory_surface_train import train
from experiments.gaussian_memory_surface_evaluate import evaluate


def main():
    completed = []
    for stage, function in [('target_preparation', prepare), ('training', train), ('evaluation', evaluate)]:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / 'pipeline_status.json', {'complete': False, 'stage': stage, 'completed': completed})
        start = time.monotonic()
        try:
            function()
        except Exception:
            write_json(OUT / f'failure_{stage}_{int(time.time())}.json',
                       {'stage': stage, 'traceback': traceback.format_exc(), 'completed': completed, 'scientific_result': False})
            raise
        gc.collect(); torch.cuda.empty_cache()
        row = {'stage': stage, 'seconds': time.monotonic() - start}; completed.append(row)
        print(json.dumps({'stage_complete': stage, 'seconds': row['seconds']}), flush=True)
    write_json(OUT / 'pipeline_status.json', {'complete': True, 'completed': completed, 'analysis_review_pending': True})


if __name__ == '__main__': main()
