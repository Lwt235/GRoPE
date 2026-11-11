"""Run the authorized scene-coverage experiment once, retaining failures."""
import json
import time
import traceback

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_coverage_data import OUT, extract
from experiments.gaussian_memory_coverage_train import train
from experiments.gaussian_memory_coverage_evaluate import evaluate


def main():
    stages = [('cache_extraction', extract), ('training', train), ('evaluation', evaluate)]
    completed = []
    for name, action in stages:
        write_json(OUT / 'pipeline_status.json', {'complete': False, 'running': name, 'completed': completed})
        started = time.monotonic()
        try:
            action(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        except Exception as exc:
            failure = {'stage': name, 'time_unix': time.time(), 'type': type(exc).__name__,
                       'message': str(exc), 'traceback': traceback.format_exc(), 'completed': completed}
            write_json(OUT / f'failure_{name}_{int(time.time())}.json', failure)
            raise
        completed.append({'stage': name, 'seconds': time.monotonic() - started})
        print(json.dumps({'stage_complete': name, 'seconds': completed[-1]['seconds']}), flush=True)
    write_json(OUT / 'pipeline_status.json', {'complete': True, 'completed': completed, 'analysis_review_pending': True})


if __name__ == '__main__':
    main()
