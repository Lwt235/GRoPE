"""Descriptive NVS evidence, independent primary-metric replay, fixed examples.

Not a training or model-selection script. All official evaluation views are old
development data. Review flags remain pending until plots are actually viewed.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.round65_common import read,write,sha
from experiments.round70_nvs_data import OUT,ROOT,NVSData
from experiments.round70_nvs_run import (ROLES,SEEDS,CONDITIONS,check,metrics,
                                       perceptual_metric,infer,configure)
from models.round70_nvs import CompactNVS

METRICS=('psnr','ssim','lpips','raw_mse','clipped_fraction')


def baselines():
    check();configure();assert read(OUT/'selections.json')['all_30_selected_before_old_eval']
    data=NVSData('old_eval');net=perceptual_metric();dest=OUT/'baselines';dest.mkdir(exist_ok=True)
    for condition in CONDITIONS:
        nearest=[];average=[];targets=[]
        for start in range(0,len(data),8):
            inp,target=data.batch(np.arange(start,min(start+8,len(data))),'rgb_raymap','cuda',condition)
            dist=(inp['c2w'][:,:2,:3,3]-inp['c2w'][:,2:,:3,3]).norm(dim=-1)
            best=dist.argmin(-1)
            nearest.append(inp['ref_rgb'][torch.arange(len(best),device='cuda'),best].cpu().numpy())
            average.append(inp['ref_rgb'].mean(1).cpu().numpy());targets.append(target.cpu().numpy())
        target=np.concatenate(targets)
        for name,images in [('nearest_camera_copy',nearest),('reference_average',average)]:
            pred=np.concatenate(images);val=metrics(pred,target,net)
            np.savez_compressed(dest/f'{name}_{condition}.npz',scene_id=data.arrays['scene_id'],**val)
        print('BASELINES',condition,flush=True)
    write(dest/'identity.json',dict(source_sha256=sha(Path(__file__)),selection_sha256=sha(OUT/'selections.json'),
                                 metrics='same fixed metric implementation as learned outputs',unix=time.time()))


def summarize():
    check();selected=read(OUT/'selections.json');manifest=read(OUT/'data/manifest.json')
    summary={};errors=[];file_ids={}
    for role in ROLES:
        summary[role]={}
        for condition in CONDITIONS:
            by_seed=[]
            for seed in SEEDS:
                f=OUT/'predictions'/role/str(seed)/f'{condition}.npz'
                meta=read(f.with_suffix('.json'))
                assert meta['prediction_sha256']==sha(f)
                assert meta['checkpoint_sha256']==sha(OUT/'runs'/role/str(seed)/'best.pt')
                assert meta['selections_sha256']==sha(OUT/'selections.json')
                file_ids[str(f)]=meta['prediction_sha256']
                with np.load(f) as z:
                    sid=z['scene_id'];p=z['prediction'].astype('float64');t=z['target_uint8'].astype('float32')/255
                    # Independent reconstruction from the saved images, never use
                    # the recorded PSNR/MSE as input to this computation.
                    sse=((np.minimum(np.maximum(p,0),1)-t)**2).reshape(len(p),-1).sum(-1)
                    psnr=10*np.log10(np.prod(p.shape[1:])/np.maximum(sse,1e-15*np.prod(p.shape[1:])))
                    raw=((p-t)**2).reshape(len(p),-1).sum(-1)/np.prod(p.shape[1:])
                    errors.append(dict(role=role,seed=seed,condition=condition,
                        psnr_max_abs=float(np.max(np.abs(psnr-z['psnr']))),
                        raw_mse_max_abs=float(np.max(np.abs(raw-z['raw_mse'])))))
                    assert np.max(np.abs(psnr-z['psnr']))<2e-5
                    assert np.max(np.abs(raw-z['raw_mse']))<1e-7
                    by_seed.append({m:[float(z[m][sid==s].mean()) for s in range(5)] for m in METRICS})
            summary[role][condition]={m:dict(seed_scene=[row[m] for row in by_seed],
                seed_mean=np.mean([row[m] for row in by_seed],axis=1).tolist(),
                scene_mean=np.mean([row[m] for row in by_seed],axis=0).tolist(),
                mean=float(np.mean([row[m] for row in by_seed]))) for m in METRICS}
    bas={}
    for name in ('nearest_camera_copy','reference_average'):
        bas[name]={}
        for condition in CONDITIONS:
            f=OUT/'baselines'/f'{name}_{condition}.npz'
            with np.load(f) as z:
                bas[name][condition]={m:float(np.mean([z[m][z['scene_id']==s].mean() for s in range(5)])) for m in METRICS}
    rng=np.random.default_rng(700071);bootstrap=rng.integers(0,5,size=(10000,5))
    differences={};decisions={}
    for candidate,controls in [('rgb_relative5',('rgb_distance','rgb_direct','rgb_prope','rgb_raymap','rgb_hrope')),
                               ('observed_relative5',('observed_distance','observed_direct','observed_prope'))]:
        differences[candidate]={};pass_core=[]
        for control in controls:
            comparisons={}
            for condition in CONDITIONS:
                comparisons[condition]={}
                for metric in METRICS:
                    a=np.array(summary[candidate][condition][metric]['seed_scene'])
                    b=np.array(summary[control][condition][metric]['seed_scene']);delta=a-b
                    ci=np.quantile(delta.mean(0)[bootstrap].mean(1),[.025,.975])
                    comparisons[condition][metric]=dict(mean=float(delta.mean()),
                        seed_mean=delta.mean(1).tolist(),scene_mean=delta.mean(0).tolist(),
                        exploratory_scene_bootstrap_interval=ci.tolist())
            differences[candidate][control]=comparisons
            if control.endswith(('distance','direct','prope')):
                d=comparisons['base']
                pass_core.append(d['psnr']['mean']>=.1 and min(d['psnr']['seed_mean'])>0 and d['lpips']['mean']<=0)
        decisions[candidate]=dict(fixed_base_utility_criterion_met=bool(all(pass_core)),
                                 limited_development_only=True,fresh_confirmation=False)
    result=dict(summary=summary,baselines=bas,differences=differences,decisions=decisions,
        selections=selected['rows'],scenes=manifest['scenes']['old_eval'],
        independent_primary_metrics=errors,prediction_files=file_ids,
        lock_sha256=sha(OUT/'lock.json'),source_sha256=sha(Path(__file__)),
        scientific_review_pending=True,visual_review_pending=True,goal_complete=False,
        caveats=['five old evaluation scenes; one old validation scene; three fit scenes',
                 'RGB-only geometry generator co-adapts, so final geometry values differ across methods',
                 'RGB-D reference input is extra sensor information and is reported separately',
                 'small LVSM adaptation and equation-based HRoPE, not original paper benchmark reproduction',
                 'shared content has coordinate-dependent geometry; no whole-network exact-invariance claim',
                 'relative term affects reference-reference attention only; quadratic memory remains',
                 'all earlier classification/RGB failures retained; no unused confirmation yet'])
    write(OUT/'result.json',result)
    plot(result)
    lines=['# R70：Transformer NVS 开发结果（待实际图表和科学复核）','',
           '全部30次固定预算拟合完成后选模；评估均为旧SUN3D场景，不是新数据确认。',
           'RGB-only与使用参考传感器深度的诊断组分开判断。指标对图像、场景和种子依次等权。','',
           '| 方法 | Base PSNR ↑ | SSIM ↑ | LPIPS-Squeeze ↓ | 外推 PSNR | 目标变焦 PSNR |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for role in ROLES:
        s=summary[role]
        lines.append(f"| {role} | {s['base']['psnr']['mean']:.4f} | {s['base']['ssim']['mean']:.4f} | {s['base']['lpips']['mean']:.4f} | {s['extrapolate']['psnr']['mean']:.4f} | {s['target_zoom']['psnr']['mean']:.4f} |")
    lines+=['','## 固定比较与边界','']
    for candidate,controls in differences.items():
        lines.append(f"{candidate} 固定用途标准：{decisions[candidate]['fixed_base_utility_criterion_met']}。")
        for control,conditions in controls.items():
            row=conditions['base'];d=row['psnr']
            lines.append(f"- 减 {control}：PSNR {d['mean']:+.4f} dB，种子 {d['seed_mean']}，场景区间 {d['exploratory_scene_bootstrap_interval']}；LPIPS {row['lpips']['mean']:+.5f}。")
        lines.append('')
    lines+=['所有固定条件、逐种子/逐场景、饱和比例、未裁剪MSE、资源和身份信息见 result.json。',
           '场景bootstrap仅探索性，不把5个场景当作大样本确认。HRoPE为本地论文公式适配。',
           '先核对输出是否学会任务，再解释编码差异；本报告生成不代表总体目标完成。','',
           '独立保存图像复算主指标最大误差：PSNR '+str(max(x['psnr_max_abs'] for x in errors))+
           '，未clip MSE '+str(max(x['raw_mse_max_abs'] for x in errors))+'。',
           'comparison.png、learning_curves.png、examples_*_seed11.png 需要实际查看。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('R70_SUMMARY_COMPLETE',decisions,flush=True)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for r,roles in enumerate([ROLES[:6],ROLES[6:]]):
        for c,metric in enumerate(('psnr','ssim','lpips')):
            ax=axes[r,c]
            for i,role in enumerate(roles):
                y=result['summary'][role]['base'][metric]['seed_mean']
                ax.scatter([i]*3,y,s=24);ax.plot([i-.2,i+.2],[np.mean(y)]*2,color='black')
            ax.set_xticks(range(len(roles)),[x.split('_',1)[1] for x in roles],rotation=30,ha='right')
            ax.set_title(('RGB only' if r==0 else 'Reference RGB-D')+' | '+metric)
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('R70 old-scene NVS: dots = seeds; bars = means (not fresh confirmation)')
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for r,roles in enumerate([ROLES[:6],ROLES[6:]]):
        for role in roles:
            histories=[read(OUT/'runs'/role/str(seed)/'history.json') for seed in SEEDS]
            fit=np.array([[x['mse'] for x in h['training']] for h in histories])
            val=np.array([[x['mse'] for x in h['validation']] for h in histories])
            for c,(array,x) in enumerate([(fit,np.arange(1,61)),(val,np.arange(3,61,3))]):
                axes[r,c].plot(x,array.mean(0),label=role.split('_',1)[1])
                axes[r,c].fill_between(x,array.min(0),array.max(0),alpha=.1)
        for c in range(2):
            axes[r,c].set_title(('RGB only' if r==0 else 'Reference RGB-D')+(' fit MSE' if c==0 else 'old val MSE'))
            axes[r,c].set_yscale('log');axes[r,c].set_xlabel('epoch');axes[r,c].legend(fontsize=8)
    fig.savefig(OUT/'learning_curves.png',dpi=150);plt.close(fig)
    data=NVSData('old_eval');a=data.arrays;indices=[int(np.flatnonzero(a['scene_id']==s)[0]) for s in range(5)]
    for group,roles in [('rgb',ROLES[:6]),('observed',ROLES[6:])]:
        fig,axes=plt.subplots(5,3+len(roles),figsize=(2*(3+len(roles)),8),layout='constrained')
        columns=['reference 0','target','reference 1']+[r.split('_',1)[1] for r in roles]
        for i,index in enumerate(indices):
            tri=a['triples'][index]
            for c,frame in enumerate(tri): axes[i,c].imshow(a['rgb'][frame])
            for c,role in enumerate(roles,3):
                with np.load(OUT/'predictions'/role/'11'/'base.npz') as z:
                    axes[i,c].imshow(z['prediction'][index].astype('float32').clip(0,1))
            for c in range(len(columns)):
                axes[i,c].axis('off')
                if i==0: axes[i,c].set_title(columns[c],fontsize=9)
        fig.suptitle('R70 '+group+': first triple from each old scene, seed 11; no selection by quality')
        fig.savefig(OUT/f'examples_{group}_seed11.png',dpi=130);plt.close(fig)


def audit():
    """Replay selected validation and sampled SSIM/LPIPS against saved predictions."""
    check();configure();result=read(OUT/'result.json');val=NVSData('val');rows=[]
    net=perceptual_metric()
    for role in ROLES:
        for seed in SEEDS:
            run=read(OUT/'runs'/role/str(seed)/'run.json')
            model=CompactNVS(role).cuda()
            saved=torch.load(OUT/'runs'/role/str(seed)/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);pred,target=infer(model,val,role)
            mse=float(np.square(pred.astype('float64')-target.astype('float64')).mean())
            assert abs(mse-run['selected_val_mse'])<1e-10
            f=OUT/'predictions'/role/str(seed)/'base.npz'
            with np.load(f) as z:
                ids=[0,48,96,144,192]
                pred=z['prediction'][ids].astype('float32');target=z['target_uint8'][ids].astype('float32')/255
                checked=metrics(pred,target,net)
                errors={m:float(np.max(np.abs(checked[m]-z[m][ids]))) for m in METRICS}
            assert max(errors.values())<2e-5
            rows.append(dict(role=role,seed=seed,val_mse_replay_error=abs(mse-run['selected_val_mse']),
                             sampled_metric_errors=errors))
            del model;torch.cuda.empty_cache()
    write(OUT/'audit.json',dict(rows=rows,passed=True,source_sha256=sha(Path(__file__)),
        result_sha256=sha(OUT/'result.json'),external_replication=False,
        note='independent primary arithmetic in summarize; SSIM/LPIPS replay checks implementation/data consistency'))
    print('R70_AUDIT_COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('baselines','summarize','audit'))
    args=p.parse_args();globals()[args.action]()
