"""Observe the original R62 queue and generate reports; never restart fits."""
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiments.round62_common import ROOT, OUT, read, write, sha


def main():
    config = read(OUT/'postprocessing/watch_config.json')
    pid = config['controller_pid']
    report = ROOT/'experiments/round62_report.py'
    start = time.monotonic()
    print('WATCHING ORIGINAL R62', pid, 'watcher', os.getpid(), flush=True)
    while True:
        queue = read(OUT/'queue_status.json')
        if (queue['pid'], queue['started_unix'], queue['lock_sha256']) != (
                pid, config['controller_started_unix'], config['lock_sha256']):
            raise RuntimeError('original queue identity changed')
        if queue['state'] == 'complete':
            if (not queue['all_children_exit_zero'] or queue['active'] or
                    not all(r['exit_code'] == 0 for r in queue['completed'])):
                raise RuntimeError('incomplete queue')
            if sha(report) != config['report_source_sha256']:
                raise RuntimeError('postprocessing source changed')
            result = subprocess.run([sys.executable, '-m', 'experiments.round62_report'],
                                    cwd=ROOT, check=False)
            if result.returncode:
                raise RuntimeError('report generation failed; original fits remain intact')
            write(OUT/'postprocessing/watcher_completed.json', dict(
                completed=True, unix=time.time(), controller_pid=pid,
                report_source_sha256=sha(report), goal_complete=False,
                scientific_review_pending=True, visual_review_pending=True))
            print('REPORT GENERATED; scientific and visual review still pending', flush=True)
            return
        if queue['state'] == 'stopped':
            raise RuntimeError('original queue stopped: '+str(queue.get('error')))
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:
            if read(OUT/'queue_status.json')['state'] == 'complete':
                continue
            raise RuntimeError('original controller missing without completion')
        if b'experiments.round62_queue' not in command:
            raise RuntimeError('controller PID identity mismatch')
        if time.monotonic()-start > 10800:
            raise TimeoutError('observer time bound; inspect original handle, do not restart fits')
        time.sleep(5)


if __name__ == '__main__':
    main()
