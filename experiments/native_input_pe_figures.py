"""Fixed PE/utility contrasts plus fixed, largest-counterexample and training-view cases."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.native_input_pe_data import *
from experiments.gaussian_scene_data import query_image


def main():
    torch.set_num_threads(4);p=plan();s=read(OUT/'scoring/summary.json');folder=OUT/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained');endpoint='development/original/missing/roi/centered_log_mse'
    labels=['Local','Center','Sphere','Full Gaussian'];colors=['#999999','#528ab5','#d6a550','#43866a']
    ax=axes[0,0];values=[s['summaries'][endpoint+'/'+k] for k in KINDS]
    ax.bar(range(4),[v['mean'] for v in values],color=colors)
    for i,v in enumerate(values):ax.scatter(np.full(3,i)+[-.1,0,.1],v['per_seed'],c='black',s=15)
    ax.set_xticks(range(4),labels);ax.set_title('Unseen places, original queries (primary)');ax.set_ylabel('Missing ROI spatial log-MSE; lower better')
    ax=axes[0,1]
    for i,pair in enumerate(['G-P','G-S','S-P','G-L','P-L']):
        v=s['comparisons'][endpoint+'/'+pair];lo,hi=v['interval95_percent']
        ax.plot([lo,hi],[i,i],c='#43866a',linewidth=2);ax.scatter(v['relative_error_reduction_percent'],i,c='#43866a',s=30)
        ax.scatter(v['per_seed_relative_percent'],np.full(3,i)+[-.09,0,.09],c='gray',s=12)
    ax.axvline(0,c='black',linewidth=.8);ax.set_yticks(range(5),['G vs P (primary)','G vs S','S vs P','G vs Local','P vs Local']);ax.invert_yaxis()
    ax.set_title('Paired place/seed95% intervals');ax.set_xlabel('Error reduction (%); positive favors first')
    ax=axes[1,0]
    for i,(cohort,view,label) in enumerate([('development','original','Dev/original'),('development','heldout','Dev/other'),('train_original','heldout','Original train/other'),('train_added','heldout','Added train/other')]):
        for pair,color,shift in [('G-P','#528ab5',-.08),('G-L','#43866a',.08)]:
            v=s['comparisons'][f'{cohort}/{view}/missing/roi/centered_log_mse/'+pair];lo,hi=v['interval95_percent']
            ax.plot([lo,hi],[i+shift,i+shift],c=color,linewidth=2);ax.scatter(v['relative_error_reduction_percent'],i+shift,c=color,s=25,label=pair if i==0 else None)
    ax.axvline(0,c='black',linewidth=.8);ax.set_yticks(range(4),['Dev/original','Dev/other','Original train/other','Added train/other']);ax.invert_yaxis();ax.legend(fontsize=8)
    ax.set_title('PE increment and source utility by scope');ax.set_xlabel('Error reduction (%)')
    ax=axes[1,1]
    for kind,color in [('G','#43866a'),('S','#d6a550')]:
        v=s['comparisons'][endpoint+'/'+kind+'-P'];ax.plot(range(12),v['per_place_relative_percent'],marker='o',label=kind+' vs P',color=color)
    ax.axhline(0,c='black',linewidth=.8);ax.set_xticks(range(12),s['development_places'],rotation=65,ha='right',fontsize=7);ax.legend(fontsize=8)
    low,high=ax.get_ylim();ax.set_ylim(min(-1.,low),max(1.,high))
    ax.set_ylabel('Error reduction (%)');ax.set_title('All12 unseen places, three-seed means\nAt least +/-1% display span; no effect threshold')
    fig.suptitle('One input attention | matched saved content |32 training places | old sensor-assisted data',fontsize=12)
    fig.savefig(folder/'outcomes.png',dpi=150);plt.close(fig)
    rows=read(OUT/'scoring/metrics.json')['rows'];candidates=[]
    for meta in p['catalog']:
        for q in QUERIES:
            means={kind:float(np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows if r['id']==meta['id'] and r['query']==q and r['condition']=='missing' and r['kind']==kind])) for kind in ['P','G']}
            candidates.append(dict(id=meta['id'],query=q,difference=means['G']-means['P'],means=means))
    worst=max(candidates,key=lambda x:(x['difference'],x['id'],x['query']))
    cases=list(dict.fromkeys([(p['catalog'][0]['id'],2),(p['catalog'][0]['id'],1),(worst['id'],worst['query']),(p['added_training_catalog'][0]['id'],1)]))
    predictions={}
    for kind in KINDS:
        pred={}
        for cohort in ['development','train']:
            record=read(OUT/'evaluation'/f'{cohort}_{kind}_{SEEDS[0]}.json');payload=load(record['prediction_path'])
            pred.update({tuple(k):v for k,v in zip(payload['keys'],payload['depth_metric'])})
        predictions[kind]=pred
    fig,axes=plt.subplots(len(cases),6,figsize=(18,3*len(cases)),layout='constrained',squeeze=False)
    for i,(rid,q) in enumerate(cases):
        query=load(query_path(rid,q));target=query['target'].numpy();valid=query['valid'].numpy();lo,hi=np.percentile(target[valid],[2,98])
        axes[i,0].imshow(query_image(query,'missing').permute(1,2,0).numpy())
        for j,value in enumerate([target]+[predictions[k][rid,q,'missing'].numpy() for k in KINDS],1):
            im=axes[i,j].imshow(np.where(valid,value,np.nan),cmap='viridis',vmin=lo,vmax=hi)
        fig.colorbar(im,ax=list(axes[i,1:]),shrink=.65,label='m, same row scale');axes[i,0].set_ylabel(f'{rid}/q{q}')
        for j in range(6):
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
    for j,label in enumerate(['Missing input','Sensor GT']+labels):axes[0,j].set_title(label)
    fig.suptitle('Fixed development views + largest G-P counterexample + first added training-place other view\nSeed20261110 shown; worst selected by3seed mean; all cases retained',fontsize=11)
    fig.savefig(folder/'depth_cases.png',dpi=150);plt.close(fig)
    write_json(folder/'selection.json',dict(cases=cases,worst_primary_counterexample=worst,
        files={str(path):sha256(path) for path in [folder/'outcomes.png',folder/'depth_cases.png']},actual_visual_review_pending=True))
    print('PE/utility and depth figures generated.',flush=True)


if __name__=='__main__':main()
