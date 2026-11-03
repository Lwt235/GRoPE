"""Observe the original R65 controller and generate its completed results report."""
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.round65_common import ROOT, OUT, checked, read, write, sha


def process_identity(pid):
    try:
        proc = Path('/proc')/str(pid)
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
    except FileNotFoundError:
        return None
    if stat[0] == 'Z':
        return None
    return dict(pid=pid, start_ticks=stat[19], command=command)


def main():
    checked(); path = OUT/'report_watch.json'
    if path.exists():
        raise ValueError('inspect the existing watcher before restarting')
    queue = read(OUT/'queue_status.json'); original = process_identity(queue['pid'])
    if original is None or original['command'] != sys.executable+' -u -m experiments.round65_queue':
        raise ValueError('original R65 controller must be live at attachment')
    source = ROOT/'experiments/round65_report.py'
    identity = dict(pid=os.getpid(), controller=original, attached_unix=time.time(),
        lock_sha256=sha(OUT/'lock.json'), source_sha256=sha(__file__), report_source_sha256=sha(source))

    def record(state, **extra):
        write(path, dict(identity, state=state, observed_unix=time.time(), goal_complete=False, **extra))

    last = None
    try:
        record('waiting_for_original_controller')
        while True:
            current = process_identity(original['pid']); queue = read(OUT/'queue_status.json')
            if queue['pid'] != original['pid'] or queue['lock_sha256'] != identity['lock_sha256']:
                raise ValueError('original queue identity changed')
            if current is not None and current != original:
                raise ValueError('controller process identity changed')
            observed = (queue['state'], queue.get('stage'), len(queue['completed']))
            if observed != last:
                record('waiting_for_original_controller', queue_state=queue['state'],
                       stage=queue.get('stage'), completed_children=len(queue['completed']))
                print('OBSERVED', *observed, flush=True); last = observed
            if current is None:
                if queue['state'] != 'complete':
                    raise RuntimeError('original controller ended before successful completion')
                break
            time.sleep(10)
        checked()
        if sha(source) != identity['report_source_sha256']:
            raise ValueError('report source changed after attachment')
        env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   CUDA_VISIBLE_DEVICES='', MPLCONFIGDIR='/tmp/grope-round65-matplotlib')
        record('generating_report')
        with (OUT/'report_generation.log').open('x') as stream:
            result = subprocess.run([sys.executable, '-m', 'experiments.round65_report'],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('report generation exit '+str(result.returncode))
        record('report_generated_review_pending', report_exit_code=0,
            generated_record_sha256=sha(OUT/'report1/generated.json'),
            scientific_review_pending=True, visual_review_pending=True)
        print('REPORT GENERATED; actual scientific and image review still required', flush=True)
    except BaseException as error:
        record('stopped', error=repr(error)); raise


if __name__ == '__main__':
    main()
