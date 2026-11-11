"""Finish only the authorized finite experiment; stop before interpretation review."""
import os
import subprocess
import sys
import time

import numpy as np

from experiments.gaussian_memory_native_pe_data import ROOT, OUT, read, write_json


def run(module, *args):
    result = subprocess.run([sys.executable, '-u', '-m', module, *args], cwd=ROOT)
    if result.returncode: raise RuntimeError(f'{module} failed with {result.returncode}')


def main():
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    while not (OUT / 'queue_complete.json').exists(): time.sleep(10)
    assert read(OUT / 'queue_complete.json')['complete']
    write_json(OUT / 'execution_status.json', {'state': 'independent_scoring', 'finite_goal': True})
    run('experiments.gaussian_memory_native_pe_checks', 'results')
    run('experiments.gaussian_memory_native_pe_analysis')
    gpu = None
    while gpu is None:
        for selected in ('2', '3', '4', '5', '7'):
            raw = subprocess.check_output(['nvidia-smi', '-i', selected, '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            used, util = map(int, raw.strip().split(', '))
            if used <= 512 and util <= 5: gpu = selected; break
        if gpu is None: time.sleep(30)
    run('experiments.gaussian_memory_launch', '--gpu', gpu, '--wait-for-gpu', 'experiments.gaussian_memory_native_pe_runtime')
    run('experiments.gaussian_memory_native_pe_report')
    write_json(OUT / 'execution_status.json', {'state': 'manual_interpretation_and_visual_review_pending', 'finite_goal': True,
        'all_fits_and_predictions_complete': True, 'independent_scores_complete': True, 'runtime_complete': True,
        'general_research_objective_achieved': False})


if __name__ == '__main__': main()
