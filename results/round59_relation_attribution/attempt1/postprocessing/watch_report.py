from pathlib import Path
import subprocess
import sys
import time
from experiments.round59_common import ROOT,OUT,read,write
while True:
    status=read(OUT/'queue_status.json')
    if status['state']=='complete':break
    if status['state']=='stopped':raise RuntimeError('Original controller stopped: '+str(status.get('error')))
    command=Path('/proc')/str(status['pid'])/'cmdline'
    if not command.exists() or b'experiments.round59_queue' not in command.read_bytes():
        raise RuntimeError('Original controller handle missing; inspect rather than restart')
    time.sleep(5)
subprocess.run([sys.executable,'-m','experiments.round59_report'],cwd=ROOT,check=True)
write(OUT/'postprocessing/watcher_completed.json',dict(completed=True,unix=time.time(),goal_complete=False))
