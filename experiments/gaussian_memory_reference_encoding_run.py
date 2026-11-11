"""Run the fixed reference encoding diagnostic and preserve any failures."""
import json
import time
import traceback

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_reference_encoding_data import OUT, extract
from experiments.gaussian_memory_reference_encoding_train import train
from experiments.gaussian_memory_reference_encoding_evaluate import evaluate


def main():
    completed = []
    for name, action in [('reference_extraction', extract), ('training', train), ('evaluation', evaluate)]:
        write_json(OUT / 'pipeline_status.json', {'complete': False, 'running': name, 'completed': completed})
        started = time.monotonic()
        try:
            action(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        except Exception as exc:
            write_json(OUT / f'failure_{name}_{int(time.time())}.json',
                       {'stage': name, 'type': type(exc).__name__, 'message': str(exc),
                        'traceback': traceback.format_exc(), 'completed': completed})
            raise
        completed.append({'stage': name, 'seconds': time.monotonic() - started})
        print(json.dumps({'stage_complete': name, 'seconds': completed[-1]['seconds']}), flush=True)
    write_json(OUT / 'pipeline_status.json', {'complete': True, 'completed': completed, 'analysis_review_pending': True})


if __name__ == '__main__':
    main()
