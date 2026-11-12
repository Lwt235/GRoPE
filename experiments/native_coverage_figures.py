"""Predeclared utility/coverage contrasts and retained depth counterexamples."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.native_coverage_data import *
from experiments.gaussian_scene_data import query_image


def main():
    torch.set_num_threads(4);p=plan();s=read(OUT/'scoring/summary.json');folder=OUT/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    endpoint='development/original/missing/roi/centered_log_mse'
    methods=['N_Local','N_Render','W_Local','W_Render'];labels=['12-place Local','12-place Render','32-place Local','32-place Render']
    vals=[s['summaries'][endpoint+'/'+k] for k in methods]
    ax=axes[0,0];ax.bar(range(4),[v['mean'] for v in vals],color=['#aaaaaa','#8aaa9a','#777777','#358269'])
    for i,v in enumerate(vals):ax.scatter(np.full(3,i)+[-.1,0,.1],v['per_seed'],c='black',s=15)
    ax.set_xticks(range(4),labels,rotation=15);ax.set_ylabel('Lower is better');ax.set_title('Fixed unseen places, original queries\nMissing ROI spatial log-MSE')
    ax=axes[0,1]
    for i,view in enumerate(['original','heldout','pooled']):
        v=s['coverage_interactions'][f'development/{view}/missing/roi/centered_log_mse'];lo,hi=v['interval95_absolute']
        ax.plot([lo,hi],[i,i],c='#6d5a96',linewidth=2);ax.scatter([v['mean_absolute']],[i],c='#6d5a96',s=35)
    ax.axvline(0,c='black',linewidth=.8);ax.set_yticks(range(3),['Original queries (primary)','Other query positions','All8 queries (correlated)']);ax.invert_yaxis()
    ax.set_title('Coverage interaction, paired95% interval');ax.set_xlabel('(R-L)32 - (R-L)12; negative favors coverage')
    ax=axes[1,0]
    for i,(view,scope) in enumerate([('original','Original'),('heldout','Other'),('pooled','All8')]):
        for prefix,color,shift in [('N_','#8f83a7',-.08),('W_','#358269',.08)]:
            v=s['comparisons'][f'development/{view}/missing/roi/centered_log_mse/{prefix}Render-{prefix}Local'];lo,hi=v['interval95_percent']
            ax.plot([lo,hi],[i+shift,i+shift],c=color,linewidth=2);ax.scatter([v['relative_error_reduction_percent']],[i+shift],c=color,s=30,label=('12places' if prefix=='N_' else '32places') if i==0 else None)
    ax.axvline(0,c='black',linewidth=.8);ax.set_yticks(range(3),['Original','Other','All8']);ax.invert_yaxis();ax.legend(fontsize=8)
    ax.set_title('Net Render benefit vs matched Local');ax.set_xlabel('Error reduction (%); positive favors Render')
    ax=axes[1,1]
    old=s['comparisons'][endpoint+'/N_Render-N_Local']['per_place_relative_percent']
    new=s['comparisons'][endpoint+'/W_Render-W_Local']['per_place_relative_percent']
    for a,b in zip(old,new):ax.plot([0,1],[a,b],c='#999999',alpha=.65)
    ax.scatter(np.zeros(12),old,c='#8f83a7',s=25);ax.scatter(np.ones(12),new,c='#358269',s=25)
    ax.axhline(0,c='black',linewidth=.8);ax.set_xticks([0,1],['12training places','32training places'])
    ax.set_ylabel('Render vs Local error reduction (%)');ax.set_title('Each line = same unseen place\nThree-seed means; all places retained')
    fig.suptitle('Same model and2000updates | training coverage12→32 | old sensor-assisted data',fontsize=12)
    fig.savefig(folder/'coverage.png',dpi=150);plt.close(fig)
    rows=read(OUT/'scoring/metrics.json')['rows'];candidates=[]
    for meta in p['catalog']:
        for q in QUERIES:
            means={kind:float(np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows if r['id']==meta['id'] and r['query']==q and r['condition']=='missing' and r['kind']==kind])) for kind in ['W_Local','W_Render']}
            candidates.append(dict(id=meta['id'],query=q,difference=means['W_Render']-means['W_Local'],means=means))
    worst=max(candidates,key=lambda x:(x['difference'],x['id'],x['query']))
    cases=list(dict.fromkeys([(p['catalog'][0]['id'],2),(p['catalog'][0]['id'],1),(worst['id'],worst['query']),(p['added_training_catalog'][0]['id'],1)]))
    predictions={}
    for kind in KINDS:
        pred={}
        for cohort in ['development','train']:
            r=read(OUT/'evaluation'/f'{cohort}_{kind}_{SEEDS[0]}.json');payload=load(r['prediction_path']);pred.update({tuple(k):v for k,v in zip(payload['keys'],payload['depth_metric'])})
        predictions[kind]=pred
    fig,axes=plt.subplots(len(cases),4,figsize=(12,3*len(cases)),layout='constrained',squeeze=False)
    for i,(rid,q) in enumerate(cases):
        query=load(query_path(rid,q));target=query['target'].numpy();valid=query['valid'].numpy();lo,hi=np.percentile(target[valid],[2,98])
        axes[i,0].imshow(query_image(query,'missing').permute(1,2,0).numpy())
        for j,value in enumerate([target]+[predictions[k][rid,q,'missing'].numpy() for k in KINDS],1):im=axes[i,j].imshow(np.where(valid,value,np.nan),cmap='viridis',vmin=lo,vmax=hi)
        fig.colorbar(im,ax=list(axes[i,1:]),shrink=.65,label='m, same row scale');axes[i,0].set_ylabel(f'{rid}/q{q}')
        for j in range(4):
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(['Missing input','Sensor GT','32-place Local','32-place Render'][j])
    fig.suptitle('Fixed first development views + largest primary counterexample + first added-place held-out view\nSeed20261110 shown; worst selected by3seed mean; all cases retained',fontsize=11)
    fig.savefig(folder/'depth_cases.png',dpi=150);plt.close(fig)
    write_json(folder/'selection.json',dict(cases=cases,worst_primary_counterexample=worst,
        files={str(path):sha256(path) for path in [folder/'coverage.png',folder/'depth_cases.png']},actual_visual_review_pending=True))
    print('Coverage and depth diagnostic figures generated.',flush=True)


if __name__=='__main__':main()
