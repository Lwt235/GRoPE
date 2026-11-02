"""Identity and exact R58 reference/cache reuse for R59."""
from pathlib import Path
from experiments.round58_common import (ROOT, OUT as REFERENCE_OUT, CONFIG, SEEDS,
    CONDITIONS, SOURCES as REFERENCE_SOURCES, read, write, sha)
from experiments.round58_data import load_cache as reference_cache
from models.round59_relation_controls import NEW_ROLES, REFERENCE_ROLES, ROLES

OUT = ROOT/'results/round59_relation_attribution/attempt1'
PRIMARY = 'invariant_mlp'
DIAGNOSTIC = 'base_reframe'
CACHE_NAMES = ('fit0', 'fit1', 'val')+tuple('eval_'+c for c in CONDITIONS)
SOURCES = tuple(sorted(set(REFERENCE_SOURCES) | {
    'ROUND59_RELATION_ATTRIBUTION_PROTOCOL.md','models/round59_relation_controls.py',
    'experiments/round59_common.py','experiments/round59_run.py',
    'experiments/round59_audit.py','experiments/round59_queue.py',
    'tests/test_round59_relation_controls.py'}))


def checked():
    lock = read(OUT/'lock.json')
    if sha(OUT/'lock.json') != (OUT/'lock.sha256').read_text().strip():
        raise ValueError('R59 lock identity')
    if lock['config'] != CONFIG or lock['roles'] != list(ROLES) or lock['seeds'] != list(SEEDS):
        raise ValueError('R59 scope')
    for name, digest in lock['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('changed frozen source '+name)
    for name, digest in lock['input_records'].items():
        if sha(name) != digest:
            raise ValueError('changed frozen input record '+name)
    return lock


def load_cache(name):
    lock = checked()
    if name not in CACHE_NAMES:
        raise ValueError('cache scope')
    if name.startswith('eval_') and not (OUT/'selected.json').exists():
        raise ValueError('new selections must precede new evaluation')
    record = read(REFERENCE_OUT/'data'/f'{name}.json')
    if record['sha256'] != lock['caches'][name]['sha256']:
        raise ValueError('frozen cache identity')
    return reference_cache(name)


def run_path(role, seed):
    return (REFERENCE_OUT if role in REFERENCE_ROLES else OUT)/'runs'/role/str(seed)


def prediction_path(role, seed, condition):
    root = REFERENCE_OUT if role in REFERENCE_ROLES and condition in CONDITIONS else OUT
    return root/'evaluation'/role/str(seed)/(condition+'.npz')


def reference_record(lock, role, seed):
    return next(x for x in lock['references'] if (x['role'], x['seed']) == (role, seed))
