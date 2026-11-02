"""Observe the original R57 controller; report only after its successful completion."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent.parent
ROOT=OUT.parents[2]
CONTROLLER_PID=96282
last=None
while True:
    status=json.loads((OUT/'queue_status.json').read_text())
    if status['pid']!=CONTROLLER_PID:
        raise RuntimeError('controller identity changed; do not follow or restart a replacement')
    if status['state']=='complete':
        if status['active'] or not status['all_children_exit_zero']:
            raise RuntimeError('inconsistent terminal state')
        print('CONTROLLER_COMPLETE; run reporting',flush=True)
        subprocess.run([sys.executable,'-m','experiments.round57_report'],cwd=ROOT,check=True)
        print('REPORT_READY; actual visual inspection and interpretation pending',flush=True)
        break
    if status['state']=='stopped':
        raise RuntimeError('original controller stopped: '+str(status.get('error')))
    os.kill(CONTROLLER_PID,0)
    current=(status.get('stage'),len(status['completed']))
    if current!=last:
        print('WAIT_EXISTING_CONTROLLER',CONTROLLER_PID,current,flush=True);last=current
    time.sleep(10)
