"""Bounded old-VAL intervention: restore point phases at selected weights.

Freeze this diagnostic before inspecting R45 old-evaluation results. No fitting,
selection change, hyperparameter scan or fresh data. This measures fixed-weight
dependence, not what a separately retrained model would achieve.
"""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from experiments import round45_phase_warp_train as R45
from experiments import round5_train as R5
from models.round45_phase_warp import make_model

OUT=R45.ROOT/'results/round45_phase_warp/diagnostic1'
ROLES=('warp_full','warp_fixed','fourier_warp','mlp_warp','affine_warp','free_frequency')


def freeze():
    R45.checked(R45.OUT)
    if OUT.exists():raise ValueError('retain diagnostic')
    OUT.mkdir(parents=True)
    source=Path(__file__);shutil.copyfile(source,OUT/'source.py')
    R45.write(OUT/'plan.json',dict(roles=ROLES,seeds=R45.SEEDS,split='val',pairs=64,
        hypotheses=['Does the selected model benefit from its learned phase change at fixed content weights?',
                    'Is any phase-change dependence specific to Gaussian maps or common to ordinary controls?'],
        conditions=['normal','point_phase'],training=False,tuning=False,fresh_data=False,
        decisions='Mechanism evidence only; neither changes R45 selection nor its success criteria.',
        source=str(source.resolve()),source_sha256=R45.sha(source),
        training_lock_sha256=R45.sha(R45.OUT/'lock.json'),unix=time.time(),
        scope='Same evaluation code, full old-VAL population; independent phase audit is in the main experiment.'))
    (OUT/'plan.sha256').write_text(R45.sha(OUT/'plan.json')+'\n')


@torch.no_grad()
def run():
    plan=R45.read(OUT/'plan.json')
    if R45.sha(OUT/'plan.json')!=(OUT/'plan.sha256').read_text().strip():raise ValueError('changed plan')
    if R45.sha(plan['source'])!=plan['source_sha256']:raise ValueError('changed diagnostic source')
    if R45.sha(R45.OUT/'lock.json')!=plan['training_lock_sha256']:raise ValueError('changed training lock')
    seal=R45.read(R45.OUT/'audit_training/completion.json')
    if len(seal['records'])!=70:raise ValueError('selected model seal required')
    if (OUT/'result.json').exists():raise ValueError('retain result')
    data,ids=R45.load('val',R45.OUT,'cuda:0');rows=[];summary=[];input_hashes={};max_ce_replay=0.
    for role in ROLES:
        for seed in R45.SEEDS:
            path=R45.OUT/'runs'/role/str(seed)
            rec=next(r for r in seal['records'] if (r['role'],r['seed'])==(role,seed))
            if R45.sha(path/'best.pt')!=rec['checkpoint_sha256']:raise ValueError('changed selected model')
            input_hashes[str(path/'best.pt')]=rec['checkpoint_sha256']
            model=make_model(role).cuda()
            model.load_state_dict(torch.load(path/'best.pt',map_location='cuda:0',weights_only=True)['state_dict'])
            model.eval();measure={}
            for condition in ('normal','point_phase'):
                if condition=='point_phase':
                    field=model.position_field
                    if hasattr(field,'projection'):field.projection.zero_()
                    elif role=='affine_warp':field.affine_offset.zero_()
                    else:field.frequency_offset.zero_()
                current=R5.evaluate(model,data,ids,role,seed,condition)
                if condition=='normal':
                    reference=R45.read(path/'validation.json')
                    if len(current)!=len(reference):raise ValueError('validation population')
                    for a,b in zip(current,reference):
                        for key in ('scene','pair_id','n_valid','pck8_count','pck4_count'):
                            if a[key]!=b[key]:raise ValueError('normal replay discrepancy '+key)
                        max_ce_replay=max(max_ce_replay,abs(a['ce_sum']-b['ce_sum']))
                rows.extend(dict(r,condition=condition) for r in current)
                measure[condition]=dict(ce=R5.macro(current),pck8=R5.macro(current,'pck8_count'))
            summary.append(dict(role=role,seed=seed,**measure,
                learned_phase_pck8_pp=100*(measure['normal']['pck8']-measure['point_phase']['pck8'])))
            print('DIAG',role,seed,summary[-1]['learned_phase_pck8_pp'],flush=True)
    R45.write(OUT/'frame_rows.json',rows)
    R45.write(OUT/'result.json',dict(summary=summary,rows=len(rows),input_hashes=input_hashes,
        max_normal_CE_frame_sum_replay_difference=max_ce_replay,
        normal_count_or_population_changes=0,fresh_data=False,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run'));a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    globals()[a.stage]()


if __name__=='__main__':main()
