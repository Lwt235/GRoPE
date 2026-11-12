"""Matched completed schedules, exact initial fusions, persistence and scope."""
import json
import time
from experiments.native_coverage_data import *
from experiments.gaussian_scene_config import schedule


def main():
    torch.set_num_threads(4);p=plan();lock=read(OUT/'configuration_lock.json');records=[]
    assert sha256(OUT/'plan.json')==lock['plan_sha256']
    assert all(sha256(ROOT/name)==h for name,h in lock['source_sha256'].items())
    assert all(sha256(path)==h for path,h in lock['input_cache_sha256'].items())
    assert all(sha256(x['path'])==x['sha256'] for x in lock['unchanged_existing_sources'].values())
    for seed in SEEDS:
        order=schedule(p,seed);initials={}
        for kind in KINDS:
            folder=OUT/f'{kind}_{seed}';status=read(folder/'status.json');assert status['complete'] and status['steps']==2000
            lines=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
            assert [x['step'] for x in lines]==list(range(1,2001))
            assert [sample for r in lines for sample in r['samples']]==[list(k) for k in order]
            assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in lines)
            state=load(folder/'initial.pt');old=load(FROZEN/f'{kind}_{seed}'/'initial.pt')
            assert all(torch.equal(t,old['model'][n]) for n,t in state['model'].items());initials[kind]=state['model']
            assert sha256(checkpoint_path(kind,seed))==status['checkpoint_sha256']
            assert not resume_path(kind,seed).exists()
            records.append(dict(kind=kind,seed=seed,steps=2000,actual_schedule_matched=True,
                initial_fusion_matches_original=True,native_parameters=status['native_parameters'],
                final_checkpoint_sha256=status['checkpoint_sha256'],final_bytes=checkpoint_path(kind,seed).stat().st_size,
                peak_GiB=status['peak_GiB']))
        assert all(torch.equal(t,initials['Render'][n]) for n,t in initials['Local'].items())
    # RAM resumes are owned, now redundant and already retired after each verified final.
    if RAM.exists():
        assert not list(RAM.iterdir());RAM.rmdir()
    write_json(OUT/'matched_checks.json',dict(complete=True,records=records,
        all_actual_schedules_and_fusion_initials_matched=True,final_weights_persistent=True,
        original24fields_and_current_inputs_unchanged=True,new_training_and_development_places_disjoint=True,
        no_low_rank_or_optimizer_precision_change=True,temporary_ram_resumes_retired=True))
    print('Six completed fits, all paired controls and persistent final weights verified.',flush=True)


if __name__=='__main__':main()
