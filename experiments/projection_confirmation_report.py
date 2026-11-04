"""All predeclared new-place results together, with fixed examples and comparisons."""
import numpy as np
from experiments.round65_common import read,write,sha
from experiments.projection_confirmation import OUT,ROLES,SEEDS,CONDITIONS,check
from experiments.projection_confirmation_data import ConfirmationData

METRICS=('psnr','ssim','lpips','raw_mse','clipped_fraction')


def summarize():
    lock=check();manifest=read(OUT/'data_manifest.json');summary={};files={};errors=[]
    for role in ROLES:
        summary[role]={}
        for condition in CONDITIONS:
            seeds=[]
            for seed in SEEDS:
                path=OUT/'predictions'/role/str(seed)/f'{condition}.npz';meta=read(path.with_suffix('.json'))
                assert meta['prediction_sha256']==sha(path)
                assert meta['checkpoint_sha256']==lock['checkpoints'][f'{role}/{seed}']['sha256']
                assert meta['evaluation_lock_sha256']==sha(OUT/'evaluation_lock.json')
                assert meta['data_manifest_sha256']==sha(OUT/'data_manifest.json')
                files[str(path)]=sha(path)
                with np.load(path) as z:
                    assert np.array_equal(np.bincount(z['scene_id']),np.full(5,48))
                    p=z['prediction'].astype('float64');t=z['target_uint8'].astype('float32')/255
                    psnr=-10*np.log10(np.maximum(((p.clip(0,1)-t)**2).mean((1,2,3)),1e-15))
                    mse=((p-t)**2).mean((1,2,3))
                    error=dict(role=role,seed=seed,condition=condition,psnr=float(np.max(np.abs(psnr-z['psnr']))),
                        raw_mse=float(np.max(np.abs(mse-z['raw_mse']))))
                    assert error['psnr']<2e-5 and error['raw_mse']<1e-7
                    errors.append(error)
                    seeds.append({k:[float(z[k][z['scene_id']==s].mean()) for s in range(5)] for k in METRICS})
            summary[role][condition]={k:dict(seed_scene=[x[k] for x in seeds],
                seed_mean=np.mean([x[k] for x in seeds],axis=1).tolist(),
                scene_mean=np.mean([x[k] for x in seeds],axis=0).tolist(),mean=float(np.mean([x[k] for x in seeds]))) for k in METRICS}
    baseline_identity=read(OUT/'baselines/identity.json');baselines={}
    for name in ('nearest_camera_copy','reference_average'):
        baselines[name]={}
        for condition in CONDITIONS:
            path=OUT/'baselines'/f'{name}_{condition}.npz'
            assert baseline_identity['files'][str(path)]==sha(path);files[str(path)]=sha(path)
            with np.load(path) as z:
                baselines[name][condition]={k:dict(scene_mean=[float(z[k][z['scene_id']==s].mean()) for s in range(5)],
                    mean=float(np.mean([z[k][z['scene_id']==s].mean() for s in range(5)]))) for k in METRICS}
    candidate='observed_target_relative5';differences={};criteria={}
    rng=np.random.default_rng(710071);indices=rng.integers(0,5,size=(10000,5))
    for control in ROLES:
        if control==candidate:continue
        differences[control]={}
        for condition in CONDITIONS:
            differences[control][condition]={}
            for metric in METRICS:
                d=np.array(summary[candidate][condition][metric]['seed_scene'])-np.array(summary[control][condition][metric]['seed_scene'])
                differences[control][condition][metric]=dict(mean=float(d.mean()),seed_mean=d.mean(1).tolist(),scene_mean=d.mean(0).tolist(),
                    scene_bootstrap_interval=np.quantile(d.mean(0)[indices].mean(1),[.025,.975]).tolist())
        b=differences[control]['base']
        criteria[control]=dict(psnr_all_three_positive=min(b['psnr']['seed_mean'])>0,
            psnr_mean_at_least_0_10_db=b['psnr']['mean']>=.1,lpips_mean_not_worse=b['lpips']['mean']<=0)
    passed=all(all(row.values()) for row in criteria.values())
    result=dict(summary=summary,baselines=baselines,differences=differences,criteria=criteria,
        decision=dict(fixed_utility_criterion_met=passed,scope='R71 observed reference geometry, five unused places'),
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),data_manifest_sha256=sha(OUT/'data_manifest.json'),
        prediction_files=files,independent_primary_metrics=errors,scenes=manifest['scenes'],
        fresh_confirmation_executed=True,scientific_review_pending=True,visual_review_pending=True,goal_complete=False,
        caveats=['No new fitting or checkpoint selection; RGB group excluded by the prospective R72 outcome rule.',
            'Five place IDs do not establish broad real-world or native optimized-GS validity.',
            'Bootstrap intervals resample five scenes conditional on the fixed three seeds.',
            'Sensor reference geometry, local Gaussian projection; no visibility oracle.'])
    write(OUT/'result.json',result);plot(result)
    lines=['# R71 投影 PE：五个未使用场景确认（待实际审阅）','',
        '15 个已冻结模型，五场景 × 48 三元组；三条件、三种子，全部预定结果统一报告。无新训练、适配或选模。','',
        '| 方法 | Base PSNR ↑ | SSIM ↑ | LPIPS ↓ | 外推 PSNR | 变焦 PSNR |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for role in ROLES:
        d=summary[role]
        lines.append(f"| {role} | {d['base']['psnr']['mean']:.4f} | {d['base']['ssim']['mean']:.4f} | {d['base']['lpips']['mean']:.4f} | {d['extrapolate']['psnr']['mean']:.4f} | {d['target_zoom']['psnr']['mean']:.4f} |")
    for name,d in baselines.items():
        lines.append(f"| {name} | {d['base']['psnr']['mean']:.4f} | {d['base']['ssim']['mean']:.4f} | {d['base']['lpips']['mean']:.4f} | {d['extrapolate']['psnr']['mean']:.4f} | {d['target_zoom']['psnr']['mean']:.4f} |")
    lines+=['',f'固定用途条件：{passed}。','']
    for role,d in differences.items():
        b=d['base'];lines.append(f"- 五维减 {role}：PSNR {b['psnr']['mean']:+.4f} dB，种子 {b['psnr']['seed_mean']}；LPIPS {b['lpips']['mean']:+.5f}。")
    lines+=['','每场景/种子/条件、未裁剪误差、饱和比例和有限场景区间见 result.json。',
        '本次确认不能替代原生 3DGS、RGB-only 或更广泛用途验证；结果生成不等于研究目标完成。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('CONFIRMATION_SUMMARIZED',result['decision'],flush=True)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Target distance','Target direct','Target relative5','URoPE','PRoPE']
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,metric in zip(axes,('psnr','ssim','lpips')):
        for i,role in enumerate(ROLES):
            y=result['summary'][role]['base'][metric]['seed_mean']
            ax.scatter([i]*3,y);ax.plot([i-.2,i+.2],[np.mean(y)]*2,c='black')
        ax.set_xticks(range(len(ROLES)),labels,rotation=25,ha='right');ax.set_title(metric);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen R71 models on five new places; observed reference geometry')
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    data=ConfirmationData();a=data.arrays;ids=[int(np.flatnonzero(a['scene_id']==s)[0]) for s in range(5)]
    fig,axes=plt.subplots(5,len(ROLES)+1,figsize=(14,9),layout='constrained')
    for row,index in enumerate(ids):
        axes[row,0].imshow(a['rgb'][a['triples'][index,1]])
        for col,role in enumerate(ROLES,1):
            with np.load(OUT/'predictions'/role/'11'/'base.npz') as z:
                axes[row,col].imshow(z['prediction'][index].astype('float32').clip(0,1))
        for col,label in enumerate(['Target']+labels):
            axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(label,fontsize=9)
    fig.suptitle('First fixed triple per new scene, seed 11; not selected by quality')
    fig.savefig(OUT/'examples_seed11.png',dpi=150);plt.close(fig)
