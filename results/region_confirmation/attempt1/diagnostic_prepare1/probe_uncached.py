import runpy
from pathlib import Path
import sys
sys.path.insert(0, '/home/liwt/GRoPE')
import numpy as np
from experiments import round68_fixed_region_data as regional
original = regional.scene_reader
def uncached(entry, lock):
    reader, poses = original(entry, lock)
    def load(index):
        return reader.__wrapped__(index)
    load.cache_clear = reader.cache_clear
    return load, poses
regional.scene_reader = uncached
runpy.run_path(str(Path(__file__).with_name('probe.py')), run_name='__main__')
