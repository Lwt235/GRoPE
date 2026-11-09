"""Retry the frozen depth URLs after the original downloader terminates.

Only transport scheduling changes: one worker and two seconds before each
network request. Model, source files, cohort, paths and PNG verification remain
the frozen implementation's responsibility. Completed PNGs are reused.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
OUT = HERE.parent


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


original_identity = json.loads((OUT/'download_execution.json').read_text())
original_proc = Path('/proc')/str(original_identity['pid'])/'stat'
try:
    fields = original_proc.read_text().rsplit(')', 1)[1].split()
    assert fields[19] != original_identity['start_ticks'] or fields[0] == 'Z', 'original downloader still running'
except FileNotFoundError:
    pass
assert list(OUT.glob('download_failure_*.json')), 'terminal failure record required'
assert not (OUT/'download_complete.json').exists()
save(HERE/'previous_progress.json', json.loads((OUT/'download_progress.json').read_text()))
identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                started_unix=time.time(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                workers=1, request_spacing_seconds=2, previous_execution=original_identity,
                failure_records=sorted(str(p) for p in OUT.glob('download_failure_*.json')),
                frozen_scientific_source_unchanged=True, same_source_urls_only=True,
                completed_pngs_reused=True)
save(HERE/'execution.json', identity)


class Tee:
    def __init__(self, a, b):
        self.a, self.b = a, b
    def write(self, value):
        self.a.write(value)
        self.b.write(value)
        self.flush()
        return len(value)
    def flush(self):
        self.a.flush()
        self.b.flush()


with (HERE/'output.log').open('x') as log:
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
    try:
        # This module imports NumPy before Torch, as required by the runtime.
        from experiments import region_retokenization_confirmation as experiment
        experiment.torch.set_num_threads(4)
        executor = concurrent.futures.ThreadPoolExecutor
        class SingleWorker(executor):
            def __init__(self, max_workers=None, *args, **kwargs):
                super().__init__(1, *args, **kwargs)
        concurrent.futures.ThreadPoolExecutor = SingleWorker
        opener_open = urllib.request.OpenerDirector.open
        def spaced_open(self, fullurl, data=None, timeout=35):
            assert isinstance(fullurl, str) and fullurl.startswith('https://sun3d.cs.princeton.edu/data/')
            time.sleep(2)
            return opener_open(self, fullurl, data=data, timeout=timeout)
        urllib.request.OpenerDirector.open = spaced_open
        experiment.download()
    except BaseException as error:
        save(HERE/'terminal.json', dict(status='failed', error=repr(error), completed_unix=time.time()))
        raise
    else:
        save(HERE/'terminal.json', dict(status='complete', completed_unix=time.time()))
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
