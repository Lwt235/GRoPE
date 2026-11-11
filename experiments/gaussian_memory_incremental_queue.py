"""Wait without a CUDA context, then apply the normal occupancy-checked launcher."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('tail_seed', 'evaluate', 'partial_evaluate', 'available_seed11')); p.add_argument('--gpu', required=True)
    args = p.parse_args(); root = Path(__file__).resolve().parents[1]
    out = root / 'results/gaussian_memory/incremental_vggt1'
    markers = {'tail_seed': 'combined_20261112/status.json', 'evaluate': 'train_pipeline_complete.json',
               'partial_evaluate': 'auxiliary_seed12_complete.json',
               'available_seed11': 'partial_evaluation_complete.json'}
    ready = out / markers[args.stage]
    start = time.monotonic(); last_print = -60
    while not ready.exists():
        elapsed = time.monotonic() - start
        if elapsed > 8 * 3600: raise TimeoutError(str(ready))
        if elapsed - last_print >= 60:
            print(json.dumps({'queued_stage': args.stage, 'waiting_for': str(ready), 'elapsed_s': round(elapsed)}), flush=True)
            last_print = elapsed
        time.sleep(10)
    assert json.loads(ready.read_text())['complete']
    command = [sys.executable, '-u', '-m', 'experiments.gaussian_memory_launch', '--gpu', args.gpu,
               '--allow-shared', '--wait-for-gpu', 'experiments.gaussian_memory_incremental_finish', args.stage]
    result = subprocess.run(command, cwd=root)
    (out / f'queue_{args.stage}_exit.json').write_text(json.dumps({'returncode': result.returncode,
       'wait_seconds': time.monotonic() - start, 'gpu': args.gpu}) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__': main()
