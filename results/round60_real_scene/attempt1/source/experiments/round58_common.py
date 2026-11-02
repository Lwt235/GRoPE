"""R58 fixed development scope; imports do not open data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/round58_observed_support/attempt1'
OLD_DATA = ROOT/'results/round48_modelnet/attempt1/data'
ROLES = ('gaussian', 'gaussian_distance', 'distance', 'distance_angle', 'invariant_mlp', 'nope')
SEEDS = (11, 22, 33)
CONDITIONS = ('base', 'noise02', 'sparse256', 'low_overlap')
CONFIG = dict(epochs=60, batch_size=32, eval_batch_size=16, validation_every=3,
              lr=.001, final_lr=.00001, warmup_epochs=3, weight_decay=.01, clip=1.,
              tokens=64, neighbors=32, points=512, resolution=.02,
              overlap_radius=.05, valid_overlap=.3, target_overlap=.1,
              center_threshold=.1, sigma_distance=.2, sigma_angle_degrees=15,
              dim=96, heads=2, fit_realizations=2, cache_batch_size=16,
              data_salt='R58-disjoint-support-v1', max_fit_seconds=28800)
SOURCES = ('ROUND58_OBSERVED_SUPPORT_PROTOCOL.md', 'models/round58_support_pe.py',
           'experiments/round58_common.py', 'experiments/round58_data.py',
           'experiments/round58_run.py', 'experiments/round58_audit.py',
           'experiments/round58_queue.py', 'tests/test_round58_support_pe.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def checked():
    lock = read(OUT/'lock.json')
    if sha(OUT/'lock.json') != (OUT/'lock.sha256').read_text().strip():
        raise ValueError('changed lock')
    if lock['config'] != CONFIG or lock['roles'] != list(ROLES) or lock['seeds'] != list(SEEDS):
        raise ValueError('changed scope')
    for name, digest in lock['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('changed source '+name)
    return lock
