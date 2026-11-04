"""Native five-input PE versus every frozen R65 control, all conditions and seeds."""
import numpy as np
from experiments.round65_common import read,write,sha,CONFIG
from experiments import round65_audit as audit
from experiments.round73_native_relative5 import OUT,OLD,ROLES,OLD_ROLES,SEEDS,VARIANTS,check

ALL_ROLES=ROLES+OLD_ROLES
PRIMARY=('distance_mlp','direct_mlp','center_rope')


def report():
    check();assert not (OUT/'result.json').exists()
    manifest=read(OUT/'data/manifest.json')
    partition=next(r for r in manifest['partitions'] if r['split']=='dev_eval')
    labels=np.load(OUT/'data/dev_eval/labels.npy',allow_pickle=False);ids=np.asarray(partition['ids'])
    rows=[];worst=0.;logits={}
    for root,roles in ((OUT,ROLES),(OLD,OLD_ROLES)):
        seal=read(root/'selected.json')
        for role in roles:
            for seed in SEEDS:
                selected=next(r for r in seal['records'] if (r['role'],r['seed'])==(role,seed))
                directory=root/'evaluation'/role/str(seed);meta=read(directory/'record.json')
                assert meta['checkpoint_sha256']==selected['checkpoint_sha256']
                assert meta['selection_sha256']==sha(root/'selected.json')
                assert meta['data_sha256']==sha(OUT/'data/manifest.json')
                assert [r['variant'] for r in meta['records']]==list(VARIANTS)
                for entry in meta['records']:
                    variant=entry['variant'];path=directory/f'{variant}.npz'
                    assert sha(path)==entry['sha256']
                    with np.load(path,allow_pickle=False) as z:
                        assert np.array_equal(z['labels'],labels) and np.array_equal(z['ids'],ids)
                        x=z['logits']
                    metrics=audit.independent_metrics(x,labels)
                    error=max(abs(metrics[k]-entry['metrics'][k]) for k in ('ce','oa','mca'))
                    error=max(error,float(np.max(np.abs(np.array(metrics['class_accuracy'])-entry['metrics']['class_accuracy']))))
                    assert error<1e-8;worst=max(worst,error)
                    domains={name:audit.independent_metrics(x[mask],labels[mask])
                        for name,mask in (('MACGS',labels<30),('ModelNet',labels>=30))}
                    rows.append(dict(role=role,seed=seed,variant=variant,metrics=metrics,domains=domains,
                        reused=root==OLD,prediction_sha256=entry['sha256']))
                    if role=='relative5' and variant in ('base','permuted'):logits[seed,variant]=x
    lookup={(r['role'],r['seed'],r['variant']):r for r in rows}
    assert len(lookup)==len(rows)==108
    differences=[]
    for variant in VARIANTS:
        for control in OLD_ROLES:
            left=np.array([lookup['relative5',s,variant]['metrics']['class_accuracy'] for s in SEEDS])
            right=np.array([lookup[control,s,variant]['metrics']['class_accuracy'] for s in SEEDS])
            delta=100*(left-right);paired=delta.mean(1);interval=audit.bootstrap(delta)
            differences.append(dict(control=control,variant=variant,mean_gain_pp=float(paired.mean()),
                paired_seed_gains_pp=paired.tolist(),paired_class_bootstrap_95_pp=interval,
                class_mean_gain_pp=delta.mean(0).tolist(),development_gain_criterion=bool(np.all(paired>0)
                    and paired.mean()>=CONFIG['minimum_gain_pp'] and interval[0]>0)))
    passed=all(r['development_gain_criterion'] for r in differences if r['variant']=='base' and r['control'] in PRIMARY)
    permutation=[]
    for seed in SEEDS:
        a,b=logits[seed,'base'],logits[seed,'permuted']
        permutation.append(dict(seed=seed,maximum_logit_error=float(np.max(np.abs(a-b))),
            prediction_changes=int((a.argmax(1)!=b.argmax(1)).sum())))
    result=dict(records=rows,comparisons=differences,permutation=permutation,
        primary_development_lead=passed,primary_controls=PRIMARY,groups=108,new_groups=18,reused_groups=90,
        independent_metric_max_error=worst,lock_sha256=sha(OUT/'lock.json'),
        selection_sha256=sha(OUT/'selected.json'),fresh_confirmation=False,goal_complete=False,
        scientific_review_pending=True,visual_review_pending=True)
    write(OUT/'result.json',result)
    lines=['# R73：原生优化 3DGS 的五维关系 PE（待实际审阅）','',
        '三个新 120-epoch 拟合，复用 R65 五组控制；共同完整 13 维内容、输入流、预算与 CE 选模不变。',
        'GSModel60 官方 TRAIN 的旧开发划分，不是官方 TEST 或独立确认。','',
        '| 方法 | Base mCA 三种子 % | 均值 % | OA % | CE |','| --- | --- | ---: | ---: | ---: |']
    for role in ALL_ROLES:
        m=[lookup[role,s,'base']['metrics'] for s in SEEDS]
        scores=[100*x['mca'] for x in m]
        lines.append(f"| {role} | {' / '.join(f'{x:.3f}' for x in scores)} | {np.mean(scores):.3f} | {100*np.mean([x['oa'] for x in m]):.3f} | {np.mean([x['ce'] for x in m]):.5f} |")
    lines+=['',f'固定开发用途条件：{passed}。总体目标未完成。','',
        '| 条件 | 五维减对照 | 三种子 pp | 均值 pp | 类别区间 pp |','| --- | --- | --- | ---: | --- |']
    for d in differences:
        lo,hi=d['paired_class_bootstrap_95_pp']
        lines.append(f"| {d['variant']} | {d['control']} | {' / '.join(f'{x:+.3f}' for x in d['paired_seed_gains_pp'])} | {d['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines+=['','所有类别与来源保留。类别区间条件于三个种子；不能由某个来源/CE 较好改写主准确率结果。',
        'Tilt 在 SO(3) 训练分布内；排列数值核查不替代实际鲁棒性。共享服务器耗时不是公平速度基准。',
        '新 SUN3D 确认不参与本轮设计选择，也不能覆盖本轮失败。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');plot(result)
    print('R73_REPORTED',passed,'metric error',worst,flush=True)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    contrasts=[r for r in result['comparisons'] if r['variant']=='base']
    for y,row in enumerate(contrasts):
        axes[0].plot(row['paired_class_bootstrap_95_pp'],[y,y],c='black')
        axes[0].scatter(row['paired_seed_gains_pp'],y+np.array([-.1,0,.1]),s=25)
        axes[0].scatter([row['mean_gain_pp']],[y],marker='D',c='black',s=25)
    axes[0].axvline(0,c='gray');axes[0].set(yticks=range(len(contrasts)),
        yticklabels=['relative5 - '+r['control'] for r in contrasts],xlabel='mCA difference (pp)',title='Base: paired seeds and class interval')
    axes[0].invert_yaxis();axes[0].grid(axis='x',alpha=.2)
    for role in ('relative5','gaussian_relation','direct_mlp','distance_mlp'):
        root=OUT if role=='relative5' else OLD
        histories=[read(root/'runs'/role/str(seed)/'history.json')['validation'] for seed in SEEDS]
        values=np.array([[r['ce'] for r in h] for h in histories]);epochs=[r['epoch'] for r in histories[0]]
        axes[1].plot(epochs,values.mean(0),label=role);axes[1].fill_between(epochs,values.min(0),values.max(0),alpha=.1)
    axes[1].set(xlabel='Epoch',ylabel='Validation CE',title='Fixed 120 epochs and minimum CE selection');axes[1].legend(fontsize=8)
    fig.suptitle('R73: current five-input PE on native optimized Gaussians; used development data')
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    controls=('distance_mlp','direct_mlp','gaussian_relation')
    matrix=np.array([[next(r['mean_gain_pp'] for r in result['comparisons'] if r['control']==c and r['variant']==v)
        for v in VARIANTS] for c in controls])
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');limit=max(.5,float(np.abs(matrix).max()))
    im=ax.imshow(matrix,cmap='coolwarm',vmin=-limit,vmax=limit)
    ax.set(xticks=range(len(VARIANTS)),xticklabels=VARIANTS,yticks=range(len(controls)),
        yticklabels=['relative5 - '+c for c in controls],title='All fixed conditions: mean mCA differences (pp)')
    for i in range(len(controls)):
        for j in range(len(VARIANTS)):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center')
    fig.colorbar(im,ax=ax);fig.savefig(OUT/'conditions.png',dpi=150);plt.close(fig)
