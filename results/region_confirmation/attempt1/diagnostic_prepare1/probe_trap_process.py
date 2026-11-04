"""Use a separate timer process because the stalled NumPy call holds the GIL."""
import os
from pathlib import Path
import runpy
import signal
import time
parent = os.getpid()
timer = os.fork()
if timer == 0:
    time.sleep(10)
    os.kill(parent, signal.SIGTRAP)
    os._exit(0)
runpy.run_path(str(Path(__file__).with_name('probe.py')), run_name='__main__')
