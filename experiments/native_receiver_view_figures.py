"""Diagnostic transfer plots and fixed/maximum-counterexample depth cases."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.native_receiver_view_data import *
from experiments.gaussian_scene_data import query_image


def main():
    torch.set_num_threads(4);p=plan();s=read(OUT/'scoring/summary.json')
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    stages=[('train','old','Fitted queries'),('train','new','Same-clip unseen queries'),
        ('development','old','Unseen places / old positions'),('development','new','Unseen places / new positions')]
    for prefix,color,shift,label in [('A_','#38866b',-.08,'Full receiver adapted'),('F_','#8471a1',.08,'Receiver frozen')]:
        for i,(cohort,view,_) in enumerate(stages):
            v=s['comparisons'][f'{cohort}/{view}/missing/roi/centered_log_mse/{prefix}Render-{prefix}Local']
            low,high=v['interval95_percent'];mean=v['relative_error_reduction_percent']
            axes[0].plot([low,high],[i+shift,i+shift],color=color,linewidth=2)
            axes[0].scatter([mean],[i+shift],c=color,s=35,label=label if i==0 else None,zorder=3)
    axes[0].axvline(0,color='black',linewidth=.8);axes[0].set_yticks(range(4),[x[2] for x in stages]);axes[0].invert_yaxis()
    axes[0].set_xlabel('Render vs Local error reduction (%)');axes[0].set_title('Missing ROI spatial log-MSE\nPaired place/seed 95% intervals');axes[0].legend(fontsize=8)
    for ax,cohort,title in [(axes[1],'train','Training places'),(axes[2],'development','Unseen places')]:
        old=s['comparisons'][f'{cohort}/old/missing/roi/centered_log_mse/A_Render-A_Local']['per_place_relative_percent']
        new=s['comparisons'][f'{cohort}/new/missing/roi/centered_log_mse/A_Render-A_Local']['per_place_relative_percent']
        for i,(a,b) in enumerate(zip(old,new)):
            ax.plot([0,1],[a,b],color='#888888',alpha=.55,linewidth=1)
        ax.scatter(np.zeros(12),old,c='#aaa077',s=22,label='Original queries',zorder=3)
        ax.scatter(np.ones(12),new,c='#38866b',s=22,label='New queries',zorder=3)
        ax.axhline(0,color='black',linewidth=.8);ax.set_xticks([0,1],['Original','New']);ax.set_ylabel('Render vs Local reduction (%)')
        ax.set_title(title+'\nEach line is one place, three-seed mean');ax.legend(fontsize=8)
    fig.suptitle('Fixed-weight view transfer | 12 existing models | zero training | old sensor-assisted clips',fontsize=12)
    fig.savefig(folder/'transfer.png',dpi=150);plt.close(fig)
    rows=read(OUT/'scoring/metrics.json')['rows'];candidates=[]
    for cohort,metas in [('train',p['training_catalog']),('development',p['catalog'])]:
        for meta in metas:
            for q in QUERIES:
                vals={k:float(np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows
                    if r['kind']==k and r['id']==meta['id'] and r['query']==q and r['condition']=='missing'])) for k in ['A_Local','A_Render']}
                candidates.append(dict(id=meta['id'],cohort=cohort,query=q,means=vals,difference=vals['A_Render']-vals['A_Local']))
    worst={cohort:max([x for x in candidates if x['cohort']==cohort],key=lambda x:(x['difference'],x['id'],x['query'])) for cohort in ['train','development']}
    cases=list(dict.fromkeys([(p['training_catalog'][0]['id'],1),(worst['train']['id'],worst['train']['query']),
        (p['catalog'][0]['id'],1),(worst['development']['id'],worst['development']['query'])]))
    predictions={}
    for kind in KINDS:
        record=read(OUT/'evaluation'/f'adapted_{kind}_{SEEDS[0]}.json');payload=load(record['prediction_path'])
        predictions[kind]={tuple(k):v for k,v in zip(payload['keys'],payload['depth_metric'])}
    fig,axes=plt.subplots(len(cases),4,figsize=(12,3*len(cases)),layout='constrained',squeeze=False)
    for i,(rid,q) in enumerate(cases):
        query=load(query_path(rid,q));target=query['target'].numpy();valid=query['valid'].numpy()
        low,high=np.percentile(target[valid],[2,98]);axes[i,0].imshow(query_image(query,'missing').permute(1,2,0).numpy())
        for j,value in enumerate([target]+[predictions[k][rid,q,'missing'].numpy() for k in KINDS],1):
            im=axes[i,j].imshow(np.where(valid,value,np.nan),cmap='viridis',vmin=low,vmax=high)
        fig.colorbar(im,ax=list(axes[i,1:]),shrink=.65,label='m, same row scale')
        axes[i,0].set_ylabel(f'{rid} / q{q}')
        for j in range(4):
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(['Missing input','Sensor GT','Adapted Local','Adapted Render'][j])
    fig.suptitle('Two fixed cases + largest new-query R-Local counterexample per cohort\nOnly seed20261110 shown; counterexamples selected by three-seed mean, all cases retained in scoring',fontsize=11)
    fig.savefig(folder/'depth_cases.png',dpi=150);plt.close(fig)
    write_json(folder/'selection.json',dict(cases=cases,worst_by_cohort=worst,
        selection='First training/development place at q1 plus largest new-query three-seed mean error difference within each cohort; deduplicate only identical panels.',
        files={str(x):sha256(x) for x in [folder/'transfer.png',folder/'depth_cases.png']},
        actual_visual_review_pending=True))
    print('Transfer and depth diagnostic figures generated.',flush=True)


if __name__=='__main__':main()
