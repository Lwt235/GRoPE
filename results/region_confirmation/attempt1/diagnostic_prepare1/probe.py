"""Bounded, input-only localization of the live CPU preparation stall."""
import copy
import faulthandler
from pathlib import Path
import sys
import time

sys.path.insert(0, '/home/liwt/GRoPE')
faulthandler.dump_traceback_later(15, repeat=True)
import numpy as np
import torch
from experiments import round68_fixed_region_run as engine
from experiments.region_confirmation_common import ROOT, OUT, PARENT, read, write, sha
from experiments.region_confirmation_data import build_arrays

print('CONFIGURE', flush=True)
engine.configure()
thresholds = read(PARENT / 'data/strata_thresholds.json')['thresholds']
records = []
for source, path, split in (
    ('old_validation', ROOT / 'results/round60_real_scene/attempt1/lock.json', 'val'),
    ('fixed_fresh_inputs', OUT / 'raw_data_lock.json', 'old_eval'),
):
    lock = read(path)
    entry = copy.deepcopy(next(e for e in lock['entries'] if e['split'] == split))
    entry['split'], entry['pairs'] = 'old_eval', entry['pairs'][:16]
    print('COLLECT', source, entry['scene'], flush=True)
    started = time.time()
    base, sparse, sample, meta = build_arrays(dict(lock, entries=[entry]), thresholds)
    record = dict(source=source, count=len(base['ids']), seconds=time.time() - started,
                  input_lock_sha256=sha(path))
    print('COMPLETED', record, flush=True)
    records.append(record)
write(Path(__file__).with_suffix('.json'), dict(records=records, no_model_scores=True, source_sha256=sha(__file__)))
faulthandler.cancel_dump_traceback_later()
