"""R61 scope and identity-checked reuse; importing does not open data."""
from pathlib import Path
import numpy as np
from experiments.round58_common import ROOT, OUT as SYNTHETIC_OUT, CONFIG as SOURCE_CONFIG, SEEDS, CONDITIONS as SOURCE_CONDITIONS, read, write, sha
from experiments.round59_common import OUT as ATTRIBUTION_OUT, run_path as old_source_run, prediction_path as old_source_prediction
from experiments.round60_common import OUT as REAL_OUT, CONFIG as TARGET_CONFIG, SOURCES as PRIOR_SOURCES
from models.round61_normal_relations import ROLES, NEW_ROLES, REFERENCE_ROLES

OUT = ROOT/'results/round61_normal_relations/attempt1'
DOMAINS = ('synthetic', 'real')
PHASES = ('synthetic', 'real_source', 'real_adapted')
CONFIGS = dict(synthetic=SOURCE_CONFIG, real=TARGET_CONFIG)
CONDITIONS = dict(synthetic=SOURCE_CONDITIONS, real=('base', 'sparse256'))
DATA_ROOTS = dict(synthetic=SYNTHETIC_OUT, real=REAL_OUT)
SOURCES = tuple(sorted(set(PRIOR_SOURCES) | {
    'ROUND61_NORMAL_RELATION_PROTOCOL.md', 'docs/round61_normal_relation_plan.md',
    'models/round61_normal_relations.py', 'experiments/round61_common.py',
    'experiments/round61_run.py', 'experiments/round61_audit.py',
    'experiments/round61_queue.py', 'tests/test_round61_normal_relations.py'}))


def checked():
    lock = read(OUT/'lock.json')
    if sha(OUT/'lock.json') != (OUT/'lock.sha256').read_text().strip():
        raise ValueError('R61 lock changed')
    if lock['configs'] != CONFIGS or lock['roles'] != list(ROLES) or lock['seeds'] != list(SEEDS):
        raise ValueError('R61 scope')
    for path, digest in lock['sources'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('changed frozen source '+path)
    for path, digest in lock['input_records'].items():
        if sha(path) != digest:
            raise ValueError('changed frozen input record '+path)
    return lock


def cache_names(domain):
    return ('fit0', 'fit1', 'val')+tuple('eval_'+c for c in CONDITIONS[domain])


def load_cache(domain, name):
    lock = checked()
    if domain not in DOMAINS or name not in cache_names(domain):
        raise ValueError('cache scope')
    if name.startswith('eval_') and not (OUT/'selected.json').exists():
        raise ValueError('all twelve new selections must precede new evaluation')
    record = lock['caches'][domain][name]
    if sha(record['path']) != record['sha256']:
        raise ValueError('cache identity')
    with np.load(record['path'], allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def run_path(domain, role, seed):
    if role in NEW_ROLES:
        return OUT/domain/'runs'/role/str(seed)
    return old_source_run(role, seed) if domain == 'synthetic' else REAL_OUT/'runs'/role/str(seed)


def prediction_path(phase, role, seed, condition):
    if role in NEW_ROLES:
        return OUT/'evaluation'/phase/role/str(seed)/(condition+'.npz')
    if phase == 'synthetic':
        return old_source_prediction(role, seed, condition)
    return REAL_OUT/'evaluation'/phase.removeprefix('real_')/role/str(seed)/(condition+'.npz')


def reference_run(lock, domain, role, seed):
    return next(x for x in lock['reference_runs'] if (x['domain'],x['role'],x['seed'])==(domain,role,seed))


def reference_prediction(lock, phase, role, seed, condition):
    return next(x for x in lock['reference_predictions'] if (x['phase'],x['role'],x['seed'],x['condition'])==(phase,role,seed,condition))
