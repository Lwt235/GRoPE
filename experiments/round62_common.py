"""Fixed scope and checked identities for the R62 RGB bridge."""
from experiments.round51_data import ROOT, read, write, sha, pixels, load as rgb_load
from experiments.round55_train import SOURCES as OLD_SOURCES, OUT as OLD_OUT
from models.round62_rgb_relations import ROLES, NEW_ROLES, REFERENCE_ROLES

OUT = ROOT/'results/round62_rgb_relations/attempt1'
SEEDS = (11, 22, 33)
CONFIG = dict(steps=5000,batch=8,lr=.001,tail_lr=.0002,weight_decay=.0001,
              validation_every=100,gradient_clip=1.,primary='predicted_full',
              primary_metric='epe_soft',new_fits=18,reused_fits=6,
              observed_floor_m=.01,observed_depth_clip_m=[.1,20.],
              observed_missing_depth_m=3.,units_m=3.,confirmation_access=False)
METRICS = ('ce','pck4','pck8','auc8','epe_argmax','epe_soft','tail16','soft_tail16')
SOURCES = tuple(sorted(set(OLD_SOURCES)|{
    'ROUND62_RGB_RELATION_PROTOCOL.md','models/round62_rgb_relations.py',
    'models/round58_support_pe.py','experiments/round62_common.py',
    'experiments/round62_data.py','experiments/round62_run.py',
    'experiments/round62_audit.py','experiments/round62_queue.py',
    'experiments/round5_rgbd_data.py','tests/test_round62_rgb_relations.py'}))


def checked():
    lock=read(OUT/'lock.json')
    if sha(OUT/'lock.json')!=(OUT/'lock.sha256').read_text().strip():
        raise ValueError('changed R62 lock')
    if lock['config']!=CONFIG or lock['roles']!=list(ROLES) or lock['seeds']!=list(SEEDS):
        raise ValueError('R62 scope')
    for path,digest in lock['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('changed frozen source '+path)
    for path,digest in lock['input_records'].items():
        if sha(path)!=digest:raise ValueError('changed input record '+path)
    if sha(OUT/'data/manifest.json')!=lock['data_manifest_sha256']:
        raise ValueError('changed RGB manifest')
    return lock


def run_path(role,seed):
    return (OLD_OUT if role in REFERENCE_ROLES else OUT)/'runs'/role/str(seed)


def prediction_path(role,seed,variant):
    return (OLD_OUT if role in REFERENCE_ROLES else OUT)/'evaluation'/role/f'{seed}_{variant}.npz'


def load(split,device,observed=False):
    import numpy as np
    import torch
    checked()
    if split=='old_eval' and not (OUT/'selected.json').exists():
        raise ValueError('all new selections must precede old_eval')
    data,ids=rgb_load(OUT,split,device)
    if observed:
        path=OUT/'data'/('observed_'+split+'.npz')
        meta=read(path.with_suffix('.json'))
        if sha(path)!=meta['sha256'] or meta['lock_sha256']!=sha(OUT/'lock.json'):
            raise ValueError('observed cache identity')
        identity=[f"{x['scene']}:{x['pair_id']}" for x in ids]
        with np.load(path,allow_pickle=False) as z:
            if z['ids'].tolist()!=identity:raise ValueError('observed/RGB alignment')
            for key in ('qobserved_mean','qobserved_cov','kobserved_mean','kobserved_cov'):
                data[key]=torch.from_numpy(z[key]).to(device)
    return data,ids
