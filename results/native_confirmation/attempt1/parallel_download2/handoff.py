"""One controlled resource handoff; never restart a job based on a poll timeout."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from experiments import native_confirmation_prefetch8 as extension
from experiments import native_confirmation_workflow as workflow
from experiments.native_stream_cache import atomic_json
from experiments.round65_common import ROOT, sha


REV = extension.REV


def live(pid, ticks):
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == str(ticks)
    except FileNotFoundError:
        return False


def prefix_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


extension.check()
assert not (REV / 'launch_identity.json').exists(), 'new controller already recorded'
assert not (REV / 'stop_record.json').exists(), 'inspect existing stop before any retry'
assert not (workflow.OUT / 'acquisition.json').exists(), 'acquisition already complete'
assert not (workflow.OUT / 'queue_status.json').exists(), 'inference has started'
old = json.loads((workflow.OUT / 'parallel_download1/launch_identity.json').read_text())
pid = old['pid']
proc = Path('/proc') / str(pid)
assert live(pid, old['start_ticks']), 'old controller is not verified live'
assert proc.stat().st_uid == os.getuid(), 'controller belongs to another user'
assert (proc / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0') == [s.encode() for s in old['command']]
before = json.loads((workflow.CACHE / 'progress.json').read_text())['completed']
identity_sha = sha(workflow.CACHE / 'identity.json')
access_sha = sha(workflow.OUT / 'data_access_started.json')
atomic_json(REV / 'stop_intent.json', dict(observed_unix=time.time(), process=old,
            completed_before=len(before), prefix_sha256=prefix_digest(before),
            cache_identity_sha256=identity_sha, data_access_sha256=access_sha))
os.kill(pid, signal.SIGTERM)
deadline = time.monotonic() + 180
while live(pid, old['start_ticks']):
    if time.monotonic() >= deadline:
        raise TimeoutError('original controller is still live; do not launch a second writer')
    time.sleep(.5)
after = json.loads((workflow.CACHE / 'progress.json').read_text())['completed']
assert after[:len(before)] == before, 'committed prefix changed while stopping'
assert sha(workflow.CACHE / 'identity.json') == identity_sha
assert sha(workflow.OUT / 'data_access_started.json') == access_sha
assert not (workflow.OUT / 'acquisition.json').exists()
assert not (workflow.OUT / 'queue_status.json').exists()
stopped = dict(stopped_unix=time.time(), process=old, completed_before=len(before),
               completed_after=len(after), prefix_sha256=prefix_digest(after),
               cache_identity_sha256=identity_sha, data_access_sha256=access_sha,
               signal='SIGTERM', only_verified_owned_controller_signaled=True)
atomic_json(REV / 'stop_record.json', stopped)
print('STOPPED_OLD_CONTROLLER', json.dumps(stopped), flush=True)
command = [sys.executable, '-u', '-m', 'experiments.native_confirmation_prefetch8', 'run']
env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
           PYTHONPATH=str(ROOT), CUBLAS_WORKSPACE_CONFIG=':4096:8')
log = REV / 'controller.log'
with log.open('x') as stream:
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
ticks = (Path('/proc') / str(child.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
launch = dict(pid=child.pid, start_ticks=ticks, started_unix=time.time(), command=command,
              log=str(log), detached=True, workers=8,
              runtime_record_sha256=sha(REV / 'runtime_record.json'),
              evaluation_lock_sha256=sha(workflow.OUT / 'evaluation_lock.json'),
              new_fits=0, goal_complete=False)
atomic_json(REV / 'launch_identity.json', launch)
print('LAUNCHED_NEW_CONTROLLER', json.dumps(launch), flush=True)
