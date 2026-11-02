"""Wait on the existing R56 controller; never restart research jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'results/round56_native_joint/attempt1'
initial=json.loads((OUT/'queue_status.json').read_text())
expected_pid=initial['pid']
last=None
while True:
    state=json.loads((OUT/'queue_status.json').read_text())
    if state['pid']!=expected_pid:
        raise RuntimeError('controller identity changed; inspect original handle')
    if state['state']=='complete':
        print('CONTROLLER_COMPLETE; start result reporting',flush=True)
        subprocess.run([sys.executable,'-m','experiments.round56_report'],cwd=ROOT,check=True)
        print('REPORT_READY; actual visual inspection and scientific interpretation pending',flush=True)
        break
    if state['state']=='stopped':
        raise RuntimeError('existing controller stopped: '+str(state.get('error')))
    try:
        os.kill(expected_pid,0)
        command=(Path('/proc')/str(expected_pid)/'cmdline').read_bytes()
    except (ProcessLookupError,FileNotFoundError):
        raise RuntimeError('controller missing; inspect terminal handle, do not relaunch')
    if b'experiments.round56_queue' not in command:
        raise RuntimeError('PID no longer belongs to R56 controller')
    observation=(state.get('stage'),len(state['completed']))
    if observation!=last:
        print('WAIT_EXISTING_CONTROLLER',expected_pid,observation,flush=True)
        last=observation
    time.sleep(20)
