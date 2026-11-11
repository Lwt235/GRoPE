"""One bounded teacher-extraction, training and evaluation pipeline."""
import gc
import time
import traceback

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_context_data import OUT, extract
from experiments.gaussian_memory_context_train import train
from experiments.gaussian_memory_context_evaluate import evaluate


def main():
    completed = []
    for stage, function in [('target_extraction', extract), ('training', train), ('evaluation', evaluate)]:
        write_json(OUT / 'pipeline_status.json', {'complete': False, 'stage': stage, 'completed': completed})
        started = time.monotonic()
        try:
            function()
        except Exception:
            failure = {'stage': stage, 'time': time.time(), 'traceback': traceback.format_exc(),
                       'completed': completed, 'scientific_result': False}
            write_json(OUT / f'failure_{stage}_{int(time.time())}.json', failure)
            raise
        gc.collect(); torch.cuda.empty_cache()
        row = {'stage': stage, 'seconds': time.monotonic() - started}; completed.append(row)
        print(__import__('json').dumps(dict(stage_complete=stage, seconds=row['seconds'])), flush=True)
    write_json(OUT / 'pipeline_status.json', {'complete': True, 'completed': completed, 'analysis_review_pending': True})


if __name__ == '__main__':
    main()
