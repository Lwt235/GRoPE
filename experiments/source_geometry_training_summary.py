"""Continuous update logs, common initialization and paired source/query sample identities."""
import hashlib
import json
import math

import numpy as np
import torch

from experiments.source_geometry_data import *
from experiments.gaussian_scene_config import schedule
from experiments import gaussian_scene_training_summary as timing

def main():
    torch.set_num_threads(4);p=plan();records=[];pending=[];time_records=[]
    for mode in MODES:
        timing.OUT=branch_out(mode);timing.KINDS=KINDS;timing.main()
        time_records+= [dict(r,geometry_mode=mode) for r in read(branch_out(mode)/'training_summary.json')['completed']]
    for seed in SEEDS:
        base=load(PARENT_OUT/f'L_{seed}'/'initial.pt')['model'];common=set(base)
        order=schedule(p,seed);orderhash=hashlib.sha256(json.dumps(order).encode()).hexdigest();extra_reference=None
        for mode in MODES:
            for kind in KINDS:
                folder=branch_out(mode)/f'{kind}_{seed}';name=f'{mode}_{kind}_{seed}'
                if not (folder/'initial.pt').exists():pending.append(name);continue
                initial=load(folder/'initial.pt');assert initial['schedule_sha256']==orderhash
                assert all(torch.equal(v,initial['model'][n]) for n,v in base.items())
                extra=set(initial['model'])-common;assert sum(initial['model'][n].numel() for n in extra)==(192 if kind=='P' else 0)
                if kind=='P':
                    if extra_reference is None:extra_reference={n:initial['model'][n] for n in extra}
                    else:assert all(torch.equal(v,initial['model'][n]) for n,v in extra_reference.items())
                log=folder/'training.jsonl';text=log.read_text() if log.exists() else '';lines=text.splitlines()
                if text and not text.endswith('\n'):lines=lines[:-1]
                rows=[json.loads(s) for s in lines if s];assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
                for r in rows:
                    assert r['samples']==[list(x) for x in order[(r['step']-1)*2:r['step']*2]]
                    assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
                complete=(folder/'status.json').exists()
                if complete:
                    status=read(folder/'status.json');assert len(rows)==2000 and status['checkpoint_sha256']==sha256(folder/'step_2000.pt')
                else:pending.append(name)
                records.append(dict(name=name,mode=mode,kind=kind,seed=seed,initial_state_sha256=sha256(folder/'initial.pt'),
                    common_parameters_exactly_match_original_L=True,paired_samples_match=True,actual_logged_updates=len(rows),fit_complete=complete))
    result=dict(complete=len(time_records)==12 and not pending,completed=time_records,pending=pending,paired_identity=records,
        retained_loop_worker_hours=sum(r['retained_loop_elapsed_seconds'] for r in time_records)/3600,
        discarded_loop_worker_hours=sum(r['discarded_loop_elapsed_seconds'] for r in time_records)/3600,
        scope='Initialization and sample pairing are inherited from unchanged native trainer; optimization-loop time excludes loads/pauses and is not exclusive performance.')
    write_json(OUT/'training_summary.json',result);print(dict(completed_fits=len(time_records),checked_initializations=len(records),pending=len(pending)),flush=True)

if __name__=='__main__':main()
