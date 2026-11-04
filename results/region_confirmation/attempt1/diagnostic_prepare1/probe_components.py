import faulthandler
from pathlib import Path
import sys
import time
sys.path.insert(0, '/home/liwt/GRoPE')
faulthandler.dump_traceback_later(12, repeat=True)
import numpy as np
import torch
from experiments import round68_fixed_region_run as engine
from experiments.region_confirmation_common import ROOT, read
from experiments.round5_rgbd_data import load_sensor_depth, read_sun3d_intrinsics
engine.configure()
def mark(s): print(s, time.monotonic(), flush=True)
entry = next(e for e in read(ROOT / 'results/round60_real_scene/attempt1/lock.json')['entries'] if e['split'] == 'val')
mark('load')
depth = load_sensor_depth(entry['frames'][0]['path'])
k = read_sun3d_intrinsics(entry['intrinsics'])
mark(('loaded', depth.shape, depth.dtype, k.shape))
ids = np.flatnonzero(np.isfinite(depth) & (depth > 0))
mark(('ids', len(ids)))
y, x = np.unravel_index(ids, depth.shape)
mark(('xy', x.shape, x.strides, y.strides))
ones = np.ones(len(ids))
mark('ones')
columns = np.column_stack((x, y, ones))
mark(('columns', columns.shape, columns.strides))
inverse = np.linalg.inv(k).T
mark('inverse')
ray = columns @ inverse
mark('matmul')
points = ray * depth.ravel()[ids, None]
mark(('done', points.shape))
faulthandler.cancel_dump_traceback_later()
