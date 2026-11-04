"""All fixed R72 outcomes, archived R70 controls and independent image arithmetic."""
import argparse
from pathlib import Path

import numpy as np

from experiments.round72_rgb_projection import OUT, OLD, ROLES, SEEDS, CONDITIONS, check
from experiments.round65_common import read, write, sha
from experiments.round70_nvs_data import NVSData

METRICS=('psnr','ssim','lpips','raw_mse','clipped_fraction')
CONTROLS=('rgb_prope','rgb_relative5','rgb_raymap','rgb_hrope')


def summarize():
    check();sel=read(OUT/'selections.json');old=read(OLD/'result.json')
    summary={k:old['summary'][k] for k in CONTROLS};files={};errors=[]
    for role in ROLES:
        summary[role]={}
        for condition in CONDITIONS:
            seeds=[]
            for seed in SEEDS:
                f=OUT/'predictions'/role/str(seed)/f'{condition}.npz';meta=read(f.with_suffix('.json'))
                assert meta['prediction_sha256']==sha(f)
                assert meta['checkpoint_sha256']==sha(OUT/'runs'/role/str(seed)/'best.pt')
                assert meta['selections_sha256']==sha(OUT/'selections.json')
                files[str(f)]=sha(f)
                with np.load(f) as z:
                    p=z['prediction'].astype('float64');t=z['target_uint8'].astype('float32')/255
                    psnr=-10*np.log10(np.maximum(((p.clip(0,1)-t)**2).mean((1,2,3)),1e-15))
                    mse=((p-t)**2).mean((1,2,3))
                    e=dict(role=role,seed=seed,condition=condition,
                        psnr=float(np.max(np.abs(psnr-z['psnr']))),raw_mse=float(np.max(np.abs(mse-z['raw_mse']))))
                    assert e['psnr']<2e-5 and e['raw_mse']<1e-7
                    errors.append(e)
                    seeds.append({k:[float(z[k][z['scene_id']==s].mean()) for s in range(5)] for k in METRICS})
            summary[role][condition]={k:dict(seed_scene=[x[k] for x in seeds],
                seed_mean=np.mean([x[k] for x in seeds],axis=1).tolist(),
                scene_mean=np.mean([x[k] for x in seeds],axis=0).tolist(),mean=float(np.mean([x[k] for x in seeds]))) for k in METRICS}
    candidate='rgb_target_relative5';differences={};core=[]
    rng=np.random.default_rng(710071);indices=rng.integers(0,5,size=(10000,5))
    for control in CONTROLS+ROLES:
        if control==candidate:continue
        differences[control]={}
        for condition in CONDITIONS:
            row={}
            for k in METRICS:
                d=np.array(summary[candidate][condition][k]['seed_scene'])-np.array(summary[control][condition][k]['seed_scene'])
                row[k]=dict(mean=float(d.mean()),seed_mean=d.mean(1).tolist(),scene_mean=d.mean(0).tolist(),
                    exploratory_scene_bootstrap_interval=np.quantile(d.mean(0)[indices].mean(1),[.025,.975]).tolist())
            differences[control][condition]=row
        if control in ('rgb_prope','rgb_target_distance','rgb_target_direct','rgb_urope'):
            b=differences[control]['base']
            core.append(b['psnr']['mean']>=.1 and min(b['psnr']['seed_mean'])>0 and b['lpips']['mean']<=0)
    result=dict(summary=summary,differences=differences,
        decision=dict(fixed_utility_criterion_met=bool(all(core)),fresh_confirmation=False),
        new_training_count=12,reused_R70_controls=CONTROLS,selections=sel['rows'],
        independent_primary_metrics=errors,prediction_files=files,lock_sha256=sha(OUT/'lock.json'),
        reused_result_sha256=sha(OLD/'result.json'),source_sha256=sha(Path(__file__)),
        scientific_review_pending=True,visual_review_pending=True,goal_complete=False,
        caveats=['Old five evaluation scenes; RGB-only plus known cameras, jointly predicted geometry; not optimized native GS',
                 'Projection is local Gaussian moment approximation without visibility oracle',
                 'Shared compute timing is not fair speed comparison',
                 'URoPE coordinate/dtype/unit/head-count adaptation, not original benchmark reproduction'])
    write(OUT/'result.json',result);plot(result)
    lines=['# R72：目标查询—参考区域投影结果（待实际图表审阅）','',
        '12 次新训练；R70 PRoPE、参考—参考五维、raymap 与 HRoPE 复用，全部旧 SUN3D 开发数据；仅参考 RGB 与已知相机输入。','',
        '| 方法 | Base PSNR ↑ | SSIM ↑ | LPIPS ↓ | 外推 PSNR | 变焦 PSNR |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for role in CONTROLS+ROLES:
        d=summary[role]
        lines.append(f"| {role} | {d['base']['psnr']['mean']:.4f} | {d['base']['ssim']['mean']:.4f} | {d['base']['lpips']['mean']:.4f} | {d['extrapolate']['psnr']['mean']:.4f} | {d['target_zoom']['psnr']['mean']:.4f} |")
    lines+=['',f"固定用途标准：{all(core)}。",'']
    for control,conditions in differences.items():
        b=conditions['base'];p=b['psnr']
        lines.append(f"- 五维投影减 {control}：PSNR {p['mean']:+.4f} dB，种子 {p['seed_mean']}；LPIPS {b['lpips']['mean']:+.5f}。")
    lines+=['','全部条件/场景/种子和裁剪比例见 result.json。5 场景 bootstrap 不是大样本或独立确认。',
            '数值复算和实际图表审阅分别记录，结果生成不等于总体研究目标完成。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('R72_SUMMARIZED',result['decision'],flush=True)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    roles=CONTROLS+ROLES
    labels=['PRoPE (R70)','3D ref-ref (R70)','Raymap (R70)','HRoPE (R70)','Target distance','Target direct','Target relative5','URoPE']
    fig,axes=plt.subplots(1,3,figsize=(17,4),layout='constrained')
    for ax,k in zip(axes,('psnr','ssim','lpips')):
        for i,role in enumerate(roles):
            y=result['summary'][role]['base'][k]['seed_mean']
            ax.scatter([i]*3,y);ax.plot([i-.2,i+.2],[np.mean(y)]*2,color='black')
        ax.set_xticks(range(len(roles)),labels,rotation=30,ha='right');ax.set_title(k);ax.grid(axis='y',alpha=.2)
    fig.suptitle('R72 RGB-predicted geometry; old scenes; dots=seeds, bars=means')
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for role,label in zip(roles,labels):
        root=OLD if role in CONTROLS else OUT
        h=[read(root/'runs'/role/str(s)/'history.json') for s in SEEDS]
        for ax,key in zip(axes,('training','validation')):
            a=np.array([[x['mse'] for x in row[key]] for row in h]);x=[x['epoch'] for x in h[0][key]]
            ax.plot(x,a.mean(0),label=label);ax.fill_between(x,a.min(0),a.max(0),alpha=.1)
            ax.set_yscale('log');ax.set_xlabel('epoch');ax.set_title(key+' MSE')
    axes[1].legend(fontsize=8);fig.savefig(OUT/'learning_curves.png',dpi=150);plt.close(fig)
    data=NVSData('old_eval');a=data.arrays;ids=[int(np.flatnonzero(a['scene_id']==s)[0]) for s in range(5)]
    fig,axes=plt.subplots(5,len(roles)+1,figsize=(18,9),layout='constrained')
    for row,index in enumerate(ids):
        axes[row,0].imshow(a['rgb'][a['triples'][index,1]])
        for col,role in enumerate(roles,1):
            root=OLD if role in CONTROLS else OUT
            with np.load(root/'predictions'/role/'11'/'base.npz') as z:
                axes[row,col].imshow(z['prediction'][index].astype('float32').clip(0,1))
        for col,label in enumerate(['Target']+labels):
            axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(label,fontsize=9)
    fig.suptitle('Fixed first triple per old scene, seed 11; not selected by quality')
    fig.savefig(OUT/'examples_seed11.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['summarize']);p.parse_args();summarize()
