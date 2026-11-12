"""Fixed outcome plots and examples; actual image review remains separate."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.learned_projection_data import *
from experiments.gaussian_scene_data import query_image


def main():
    torch.set_num_threads(4);p=plan();s=read(OUT/'scoring/summary.json');folder=OUT/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['parent_L','P','S','G','old_fixed_G_sensor'];labels=['Local L','Center P','Learned sphere','Learned full','Old fixed G']
    colors=['#999999','#6695b6','#69aa8c','#9875af','#d4b18c']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,endpoint,title in [(axes[0,0],'missing/roi/centered_log_mse','Missing ROI spatial log-MSE'),
                              (axes[0,1],'full/roi/centered_log_mse','Full-input ROI spatial log-MSE')]:
        values=[s['summaries'][endpoint+'/'+k] for k in names]
        ax.bar(range(len(names)),[v['mean'] for v in values],color=colors)
        for i,v in enumerate(values):ax.scatter(np.full(3,i)+[-.12,0,.12],v['per_seed'],c='black',s=15)
        ax.set_xticks(range(len(names)),labels,rotation=20);ax.set_title(title);ax.set_ylabel('Lower is better')
    pairs=['G-P','G-S','S-P','P-parent_L','S-parent_L','G-parent_L'];ax=axes[1,0]
    for i,pair in enumerate(pairs):
        v=s['comparisons']['missing/roi/centered_log_mse/'+pair];mean=v['relative_error_reduction_percent'];lo,hi=v['interval95_percent']
        ax.plot([lo,hi],[i,i],color='#5e5479',linewidth=2);ax.scatter([mean],[i],c='#5e5479',s=30,zorder=3)
        ax.scatter(v['per_seed_relative_percent'],np.full(3,i)+[-.12,0,.12],color='gray',s=12)
    ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(range(len(pairs)),pairs);ax.invert_yaxis()
    ax.set_title('Missing ROI: paired error reduction');ax.set_xlabel('Positive favors first method (%); paired95% intervals')
    ax=axes[1,1];v=s['comparisons']['missing/roi/centered_log_mse/G-P']
    ax.barh(range(12),v['per_place_relative_percent'],color=['#69aa8c' if x>0 else '#bd7e76' for x in v['per_place_relative_percent']])
    ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(range(12),s['places'],fontsize=8);ax.invert_yaxis()
    ax.set_title('All12 places: learned full vs center');ax.set_xlabel('Error reduction (%), three-seed means')
    fig.suptitle('Learned projection PE | three paired seeds, old development data\nSource sensor/joint-validity diagnostic. Dots are seed means; intervals only in lower-left panel.',fontsize=12)
    fig.savefig(folder/'outcomes.png',dpi=150);plt.close(fig)
    rows=read(OUT/'scoring/metrics.json')['rows'];candidates=[]
    for meta in p['catalog']:
        for q in QUERIES:
            values={k:float(np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows
                if r['kind']==k and r['id']==meta['id'] and r['query']==q and r['condition']=='missing'])) for k in ('P','G')}
            candidates.append(dict(id=meta['id'],query=q,difference=values['G']-values['P'],means=values))
    worst=max(candidates,key=lambda x:(x['difference'],x['id'],x['query']))
    fixed=[(p['catalog'][0]['id'],2),(p['catalog'][0]['id'],11),(p['catalog'][1]['id'],5)]
    cases=list(dict.fromkeys(fixed+[(worst['id'],worst['query'])]));predictions={}
    for kind in KINDS:
        r=read(OUT/'evaluation'/f'{kind}_{SEEDS[0]}.json');x=load(r['prediction_path'])
        predictions[kind]={tuple(k):v for k,v in zip(x['keys'],x['depth_metric'])}
    r=read(SOURCE_OUT/'evaluation'/f'G_{SEEDS[0]}.json');x=load(r['prediction_path'])
    predictions['old_fixed_G_sensor']={tuple(k):v for k,v in zip(x['keys'],x['depth_metric'])}
    fig,axes=plt.subplots(len(cases),6,figsize=(15,3*len(cases)),layout='constrained',squeeze=False)
    titles=['Missing RGB','Sensor GT','Center P','Learned sphere','Learned full','Old fixed G']
    for row,(rid,q) in enumerate(cases):
        query=load(query_path(rid,q));valid=query['valid'].numpy();target=query['target'].numpy()
        limits=np.percentile(target[valid],[2,98]);axes[row,0].imshow(query_image(query,'missing').permute(1,2,0).numpy())
        images=[target]+[predictions[k][rid,q,'missing'].numpy() for k in ('P','S','G','old_fixed_G_sensor')]
        for col,depth in enumerate(images,1):
            im=axes[row,col].imshow(np.where(valid,depth,np.nan),cmap='viridis',vmin=limits[0],vmax=limits[1])
        fig.colorbar(im,ax=list(axes[row,1:]),shrink=.65,label='m, same row scale')
        for col in range(6):
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(titles[col])
        axes[row,0].set_ylabel(f'{rid}/q{q}'+('\nmax G-P' if (rid,q)==(worst['id'],worst['query']) else ''))
    fig.suptitle(f'Fixed cases plus largest G-P primary regression by three-seed means\nDisplayed seed{SEEDS[0]}; white is invalid GT; GT used for evaluation/display only.',fontsize=12)
    fig.savefig(folder/'depth_cases.png',dpi=150);plt.close(fig)
    write_json(folder/'case_selection.json',dict(fixed_cases=[list(x) for x in fixed],worst=worst,
        displayed_cases=[list(x) for x in cases],displayed_seed=SEEDS[0],unique_cases=len(cases)))
    write_json(folder/'manifest.json',dict(complete=True,files=['outcomes.png','depth_cases.png'],
        sha256={f.name:sha256(f) for f in folder.glob('*.png')},visual_review_pending=True))
    print('Two final figures generated; actual review pending.',flush=True)


if __name__=='__main__':main()
