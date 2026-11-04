"""Prospective old-val readout; unchanged R70 geometry metrics, no tuning."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch

from experiments.round65_common import read,write,sha
from experiments.round70_nvs_data import ROOT,NVSData
from experiments.round70_geometry_diagnostic import measurements,generate
from experiments.round72_rgb_projection import OUT,OLD,ROLES,SEEDS,check
from models.round72_rgb_projection import RGBTargetProjectionNVS

DEST=OUT/'geometry_diagnostic1'
PRIOR=OLD/'geometry_diagnostic1/result.json'


def plan():
    assert not (DEST/'plan.json').exists()
    write(DEST/'plan.json',dict(created_unix=time.time(),source_sha256=sha(Path(__file__)),
        helper_sha256=sha(ROOT/'experiments/round70_geometry_diagnostic.py'),
        training_lock_sha256=sha(OUT/'lock.json'),prior_geometry_result_sha256=sha(PRIOR),
        hypothesis='Target-reference projection may encourage the RGB Gaussian head to track observed geometry more closely than R70 reference-reference placement.',
        competing_explanation='Task utility may come from a latent Gaussian-form representation without improved physical depth/covariance consistency.',
        fixed_scope='All 12 selected R72 RGB models; all 32 old validation triples; base references only; no new pixels or fresh scenes.',
        fixed_comparison='Reuse every R70 selected RGB geometry readout and its initial constant generator by identity; same R70 measurements and aggregation.',
        measurements=['depth_abs_m','depth_abs_relative','log_cov_frobenius','log_cov_traceless_frobenius',
                      'depth/trace/condition-number 5/50/95 percentiles'],
        coverage_groups=['coverage>0 primary','coverage>=0.5 fixed sensitivity'],
        precision='CPU float32 generator, float64 metric arithmetic, not a replacement for official bf16 image metrics',
        decision='Only change interpretation of geometry and next diagnostic hypothesis; never replace image utility criteria, choose a checkpoint, or require this as another promotion gate.',
        caveats=['Sensor patch moments are observed region references, not optimized 3DGS ground truth.',
                 'Between-trained-model differences are descriptive, not unique causal identification.',
                 'No scan of scale, temperature, loss, epochs or seeds.'],
        fresh_confirmation=False,training_modified=False))
    print('GEOMETRY_PLAN_FIXED',sha(DEST/'plan.json'),flush=True)


def run():
    check();p=read(DEST/'plan.json')
    assert p['source_sha256']==sha(Path(__file__)) and p['prior_geometry_result_sha256']==sha(PRIOR)
    assert p['helper_sha256']==sha(ROOT/'experiments/round70_geometry_diagnostic.py')
    assert read(OUT/'selections.json')['all_12_selected_before_old_eval']
    assert not (DEST/'result.json').exists()
    torch.set_num_threads(4);ds=NVSData('val');refs=ds.arrays['triples'][:,[0,2]]
    tm,tc,coverage=[ds.arrays[k][refs] for k in ('mean','cov','coverage')]
    previous=read(PRIOR);rows=[]
    for role in ROLES:
        for seed in SEEDS:
            checkpoint=OUT/'runs'/role/str(seed)/'best.pt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=RGBTargetProjectionNVS(role);model.load_state_dict(saved['model'])
            pm,pc=generate(model,ds)
            assert np.isfinite(pm).all() and np.isfinite(pc).all()
            np.savez_compressed(DEST/f'{role}_{seed}.npz',mean=pm,cov=pc)
            row=dict(role=role,seed=seed,checkpoint_sha256=sha(checkpoint),selected_epoch=saved['epoch'],
                **measurements(pm,pc,tm,tc,coverage))
            rows.append(row);print('GEOMETRY',role,seed,row['any_observed']['mean'],flush=True)
    old_roles=tuple(dict.fromkeys(r['role'] for r in previous['runs']))
    combined=previous['runs']+rows
    keys=tuple(rows[0]['any_observed']['mean'])
    summary={role:{key:float(np.mean([r['any_observed']['mean'][key] for r in combined if r['role']==role]))
                   for key in keys} for role in old_roles+ROLES}
    result=dict(plan_sha256=sha(DEST/'plan.json'),selection_sha256=sha(OUT/'selections.json'),
        source_sha256=sha(Path(__file__)),prior_result_sha256=sha(PRIOR),runs=rows,
        reused_runs=previous['runs'],initial_fixed_generator=previous['initial_fixed_generator'],
        summary=summary,fresh_confirmation=False,goal_complete=False,
        scientific_review_pending=True,visual_review_pending=True)
    write(DEST/'result.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(18,5),layout='constrained')
    roles=old_roles+ROLES
    for ax,key in zip(axes,('depth_abs_relative','log_cov_frobenius','log_cov_traceless_frobenius')):
        ax.axhline(previous['initial_fixed_generator']['any_observed']['mean'][key],ls='--',c='gray')
        for i,role in enumerate(roles):
            y=[r['any_observed']['mean'][key] for r in combined if r['role']==role]
            ax.scatter([i]*3,y);ax.plot([i-.2,i+.2],[np.mean(y)]*2,c='black')
        ax.set_xticks(range(len(roles)),[('R70 ' if r in old_roles else 'R72 ')+r[4:] for r in roles],rotation=55,ha='right')
        ax.set_title(key+' (lower = closer to sensor moments)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Old-val reference geometry consistency; diagnostic only, no new promotion gate')
    fig.savefig(DEST/'geometry.png',dpi=150);plt.close(fig)
    lines=['# R72 预测几何的只读解释检查（待实际审阅）','',
        '旧验证集全部32三元组、12个新选定模型；复用R70全部18个模型及初始生成器的读数。',
        '没有新训练、几何选模或确认集读取。CPU描述统计不能替代正式bf16图像指标。','',
        '| 角色 | 深度 AbsRel ↓ | 深度绝对误差 m ↓ | log协方差距离 ↓ | 去迹 log协方差距离 ↓ |',
        '| --- | ---: | ---: | ---: | ---: |']
    for role,row in summary.items():
        lines.append(f"| {role} | {row['depth_abs_relative']:.5f} | {row['depth_abs_m']:.5f} | {row['log_cov_frobenius']:.5f} | {row['log_cov_traceless_frobenius']:.5f} |")
    lines+=['','逐种子、两种固定覆盖条件和分位数见 result.json。传感器矩不是最优 3DGS 真值；差异不能唯一归因。']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n')
    write(DEST/'completion.json',dict(completed=True,result_sha256=sha(DEST/'result.json'),
        scientific_review_pending=True,visual_review_pending=True,unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);args=parser.parse_args()
    globals()[args.action]()
