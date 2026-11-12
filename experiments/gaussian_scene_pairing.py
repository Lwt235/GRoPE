"""Check saved initial tensors and actual sampled trajectories against the fixed plan."""
import hashlib
import json

import numpy as np
import torch

from experiments.gaussian_scene_data import OUT, ROOT, KINDS, SEEDS, read
from experiments.gaussian_scene_config import schedule
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import write_json


def digest(state):
    value=hashlib.sha256()
    for name,tensor in state.items():
        value.update(name.encode());value.update(tensor.cpu().numpy().tobytes())
    return value.hexdigest()


def main():
    torch.set_num_threads(4)
    plan=read(OUT/'plan.json');plan_hash=sha256(OUT/'plan.json')
    lock=read(OUT/'training_lock.json')
    assert lock['revision']==2 and lock['plan_sha256']==plan_hash
    for path,expected in lock['input_manifest_sha256'].items():
        assert sha256(path)==expected,path
    records=[];all_fits_complete=True
    for seed in SEEDS:
        states={kind:torch.load(OUT/f'{kind}_{seed}'/'initial.pt',map_location='cpu',weights_only=True)
                for kind in KINDS}
        order=schedule(plan,seed)
        order_hash=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        base=states['L']['model'];common=set(base)
        for kind in KINDS:
            name=f'{kind}_{seed}';folder=OUT/name;initial=states[kind]
            assert initial['kind']==kind and initial['seed']==seed and initial['plan_sha256']==plan_hash
            assert initial['schedule_sha256']==order_hash
            assert initial['initial_state_sha256']==digest(initial['model'])
            assert common.issubset(initial['model'])
            assert all(torch.equal(value,initial['model'][key]) for key,value in base.items())
            extra=set(initial['model'])-common
            extra_count=sum(initial['model'][key].numel() for key in extra)
            assert extra_count==(192 if kind in ('T','P') else 0)
            if kind in ('T','P'):
                assert all(torch.equal(initial['model'][key],states['P']['model'][key]) for key in extra)
            # The last live worker may append a line while this snapshot is read.
            # Inspect only its completely written prefix, never call that a full fit.
            text=(folder/'training.jsonl').read_text()
            lines=text.splitlines()
            if text and not text.endswith('\n'):lines=lines[:-1]
            rows=[json.loads(line) for line in lines if line.strip()]
            assert [row['step'] for row in rows]==list(range(1,len(rows)+1))
            for row in rows:
                expected=[list(key) for key in order[(row['step']-1)*2:row['step']*2]]
                assert row['samples']==expected,(name,row['step'])
            marker=folder/'status.json';complete=marker.exists()
            if complete:
                status=read(marker)
                assert status['complete'] and status['steps']==len(rows)==2000
                assert status['schedule_sha256']==order_hash
                assert status['initial_state_sha256']==initial['initial_state_sha256']
                assert status['checkpoint_sha256']==sha256(folder/'step_2000.pt')
            else:all_fits_complete=False
            records.append(dict(name=name,seed=seed,kind=kind,
                initial_file_sha256=sha256(folder/'initial.pt'),
                common_initial_parameters_exactly_equal=True,extra_parameters=extra_count,
                actual_logged_updates=len(rows),logged_samples_match_fixed_schedule=True,
                schedule_sha256=order_hash,fit_complete=complete))
    write_json(OUT/'paired_controls_identity.json',dict(
        complete=all_fits_complete,plan_sha256=plan_hash,training_lock_revision=lock['revision'],
        checked_input_manifests=len(lock['input_manifest_sha256']),initializations_checked=len(records),
        all15_common_initial_parameters_match_within_seed=True,
        center_initial_parameters_T_and_P_identical=True,records=records,
        scope='Saved initial tensors, fixed source/query manifests and actual logged sample trajectories only; live prefix is not a completed fit. No efficacy claim or whole-goal completion.'))
    print(json.dumps(dict(initializations_checked=len(records),
        completed_fits=sum(record['fit_complete'] for record in records),
        all15_initial_and_observed_sample_pairing_verified=True,complete=all_fits_complete)),flush=True)


if __name__=='__main__':
    main()
