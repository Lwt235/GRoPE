"""R64 prospective RGB-only scope and frozen source/data identities."""
from experiments.round51_data import ROOT, read, write, sha, pixels, load as rgb_load
from experiments.round62_common import (SOURCES as PARENT_SOURCES, OUT as PARENT_OUT,
    METRICS, run_path as parent_run, prediction_path as parent_prediction)
from models.round64_depth_field import ROLES, NEW_ROLES, REFERENCE_ROLES, validate_grid

OUT = ROOT/'results/round64_depth_field/attempt1'
SEEDS = (11, 22, 33)
PREPARATION = ROOT/'results/round64_depth_field/preparation/availability.json'
CONFIG = dict(steps=5000,batch=8,lr=.001,tail_lr=.0002,weight_decay=.0001,
              validation_every=100,gradient_clip=1.,primary='depth_surface',
              primary_metric='epe_soft',new_fits=9,reused_fits=24,
              depth_floor_units=1/30,covariance_floor_units=1/300,units_m=3.,
              grid=[18,24],stride_pixels=8,pixel_variance=5.25,
              derivative='central interior;one-sided boundary;no smoothing',
              confirmation_access=False)
SOURCES = tuple(sorted(set(PARENT_SOURCES)|{
    'ROUND64_DEPTH_FIELD_PROTOCOL.md','docs/round64_depth_field_plan.md',
    'models/round64_depth_field.py','experiments/round64_common.py',
    'experiments/round64_run.py','experiments/round64_audit.py',
    'experiments/round64_queue.py','tests/test_round64_depth_field.py'}))


def checked():
    lock=read(OUT/'lock.json')
    if sha(OUT/'lock.json')!=(OUT/'lock.sha256').read_text().strip():
        raise ValueError('changed R64 lock')
    if lock['config']!=CONFIG or lock['roles']!=list(ROLES) or lock['seeds']!=list(SEEDS):
        raise ValueError('R64 scope')
    for path,digest in lock['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('changed frozen source '+path)
    for path,digest in lock['input_records'].items():
        if sha(path)!=digest:raise ValueError('changed input record '+path)
    if sha(OUT/'data/manifest.json')!=lock['data_manifest_sha256']:
        raise ValueError('changed RGB manifest')
    return lock


def run_path(role,seed):
    return parent_run(role,seed) if role in REFERENCE_ROLES else OUT/'runs'/role/str(seed)


def prediction_path(role,seed,variant):
    return parent_prediction(role,seed,variant) if role in REFERENCE_ROLES else OUT/'evaluation'/role/f'{seed}_{variant}.npz'


def load(split,device):
    checked()
    if split=='old_eval' and not (OUT/'selected.json').exists():
        raise ValueError('all nine new selections must precede old_eval')
    # Validate known grid coordinates once on CPU, before transfer/training.
    data,ids=rgb_load(OUT,split,'cpu')
    validate_grid(data)
    return {k:v.to(device) for k,v in data.items()},ids
