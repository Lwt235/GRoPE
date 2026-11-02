"""Check unchanged R51 controls; repeated seeds are not independent evidence."""
import numpy as np
import torch
from experiments import round52_train as P
from experiments.round52_data import OUT, PREVIOUS, read, write, sha


def main():
    P.check(); records=[]
    old_selection=read(PREVIOUS/'selection.json')['records']
    new_selection=read(OUT/'selection.json')['records']
    for role in ('nope','mlp','hrope'):
        for seed in P.SEEDS:
            old=next(r for r in old_selection if r['role']==role and r['seed']==seed)
            new=next(r for r in new_selection if r['role']==role and r['seed']==seed)
            paths=[folder/'runs'/role/str(seed)/'model.pt' for folder in (PREVIOUS,OUT)]
            for path,record in zip(paths,(old,new)):
                if sha(path)!=record['model_sha256']:raise ValueError('checkpoint identity')
            states=[torch.load(path,map_location='cpu',weights_only=True) for path in paths]
            if states[0].keys()!=states[1].keys():raise ValueError('control architecture changed')
            maximum=max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0])
            records.append(dict(role=role,seed=seed,tensor_max_difference=maximum,
                selected_step_equal=old['best_step']==new['best_step'],validation_ce_difference=new['val_ce']-old['val_ce'],
                common_init_equal=old['shared_init_sha256']==new['shared_init_sha256'],
                sampling_equal=old['sampling_sha256']==new['sampling_sha256']))
    write(OUT/'reference_replay.json',dict(records=records,source_sha256=sha(__file__),
        old_selection_sha256=sha(PREVIOUS/'selection.json'),new_selection_sha256=sha(OUT/'selection.json'),
        scope='Repeat of the same three seeds/inputs/budget, not nine new independent confirmations.',confirmation_access=False))
    print('CONTROL REPLAY',len(records),'max tensor difference',max(r['tensor_max_difference'] for r in records),flush=True)


if __name__=='__main__':main()
