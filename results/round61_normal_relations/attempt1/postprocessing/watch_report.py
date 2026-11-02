"""Observe only the original R61 controller; never launch/restart training."""
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiments.round61_common import ROOT,OUT,read,write,sha


def main():
    handle=read(OUT/'execution_handle.json');pid=handle['controller']['host_pid']
    start=time.monotonic()
    while True:
        q=read(OUT/'queue_status.json')
        if q['pid']!=pid:raise RuntimeError('original queue identity changed')
        if q['state']=='complete':
            if not q['all_children_exit_zero'] or q['active']:raise RuntimeError('incomplete queue')
            result=subprocess.run([sys.executable,'-m','experiments.round61_report'],cwd=ROOT,check=False)
            if result.returncode:raise RuntimeError('report generation failed')
            write(OUT/'postprocessing/watcher_completed.json',dict(completed=True,unix=time.time(),
                controller_pid=pid,report_source_sha256=sha(ROOT/'experiments/round61_report.py'),goal_complete=False))
            return
        if q['state']=='stopped':raise RuntimeError('original queue stopped: '+str(q.get('error')))
        proc=Path(f'/proc/{pid}/cmdline')
        try:command=proc.read_bytes()
        except FileNotFoundError:
            # One final authoritative queue read covers the completion/exit boundary.
            if read(OUT/'queue_status.json')['state']=='complete':continue
            raise RuntimeError('original controller missing without completion')
        if b'experiments.round61_queue' not in command:raise RuntimeError('controller PID identity mismatch')
        if time.monotonic()-start>10800:raise TimeoutError('watcher bound;inspect original handle,do not restart')
        time.sleep(5)


if __name__=='__main__':main()
