"""Watch the original owned R64 process; generate reports only after completion.

This is postprocessing orchestration. It does not import training stages,
restart a controller, change scientific settings, or access evaluation early.
"""
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.round64_common import ROOT, OUT, checked, read, write, sha


def process_identity(pid):
    try:
        stat = (Path('/proc')/str(pid)/'stat').read_text()
        command = (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return None
    # Fields after the closing comm parenthesis start with field3 (state).
    fields = stat.rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        return None
    return dict(pid=pid, start_ticks=int(fields[19]), command=command)


def main():
    checked()
    status_path = OUT/'report_watch.json'
    if status_path.exists():
        raise ValueError('inspect the existing report watcher before restarting')
    queue = read(OUT/'queue_status.json')
    original = process_identity(queue['pid'])
    if original is None or '-m experiments.round64_queue' not in original['command']:
        raise ValueError('original R64 controller must be confirmed live at attachment')
    report_source = ROOT/'experiments/round64_report.py'
    identity = dict(pid=os.getpid(), controller=original, attached_unix=time.time(),
                    lock_sha256=sha(OUT/'lock.json'), source_sha256=sha(__file__),
                    report_source_sha256=sha(report_source))

    def record(state, **extra):
        write(status_path, dict(identity, state=state, observed_unix=time.time(),
                               goal_complete=False, **extra))

    last = None
    try:
        record('waiting_for_original_controller')
        while True:
            current = process_identity(original['pid'])
            queue = read(OUT/'queue_status.json')
            if queue['pid'] != original['pid'] or queue['lock_sha256'] != identity['lock_sha256']:
                raise ValueError('original queue identity changed')
            if current is not None and current != original:
                raise ValueError('controller PID identity changed')
            observed = (queue['state'], queue.get('stage'), len(queue['completed']))
            if observed != last:
                record('waiting_for_original_controller', queue_state=queue['state'],
                       stage=queue.get('stage'), completed_children=len(queue['completed']))
                print('OBSERVED', *observed, flush=True)
                last = observed
            if current is None:
                if queue['state'] != 'complete' or not all(x['exit_code'] == 0 for x in queue['completed']):
                    raise RuntimeError('original controller ended before successful experiment completion')
                break
            time.sleep(10)
        checked()
        if sha(report_source) != identity['report_source_sha256']:
            raise ValueError('report source changed after watcher attachment')
        env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   CUDA_VISIBLE_DEVICES='', MPLCONFIGDIR='/tmp/grope-round64-matplotlib')
        record('generating_report')
        with (OUT/'report_generation.log').open('x') as stream:
            result = subprocess.run([sys.executable, '-m', 'experiments.round64_report'],
                                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('report generation exit '+str(result.returncode))
        record('report_generated_review_pending', report_exit_code=0,
               generated_record_sha256=sha(OUT/'report1/generated.json'),
               scientific_review_pending=True, visual_review_pending=True)
        print('REPORT GENERATED; actual scientific and image review still required', flush=True)
    except BaseException as error:
        record('stopped', error=repr(error))
        raise


if __name__ == '__main__':
    main()
