"""A debugger-owned disposable child; never attaches to an existing job."""
import os
from pathlib import Path
import runpy
import signal
import threading
threading.Timer(10, lambda: os.kill(os.getpid(), signal.SIGTRAP)).start()
runpy.run_path(str(Path(__file__).with_name('probe.py')), run_name='__main__')
