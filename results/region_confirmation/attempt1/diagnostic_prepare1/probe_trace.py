import runpy
from pathlib import Path
import sys
sys.path.insert(0, '/home/liwt/GRoPE')
import numpy as np
from experiments import round60_data
original = round60_data.backproject
calls = 0
def traced(depth, k):
    global calls
    calls += 1
    print('BACKPROJECT_START', calls, depth.shape, int((depth > 0).sum()), flush=True)
    result = original(depth, k)
    print('BACKPROJECT_END', calls, len(result['points']), flush=True)
    return result
round60_data.backproject = traced
runpy.run_path(str(Path(__file__).with_name('probe.py')), run_name='__main__')
