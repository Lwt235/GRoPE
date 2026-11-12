"""Final finite diagnostic plots; generated figures still require actual visual review."""
import hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.source_geometry_data import *


def main():
    torch.set_num_threads(4);p=plan();summary=read(OUT/'scoring/summary.json');folder=OUT/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['parent_L','P_pred','G_pred','P_sensor','G_sensor'];labels=['Local L','Pred P','Pred G','Sensor P','Sensor G'];colors=['#999999','#7aa6c2','#27638f','#b59dc5','#794a94']
    endpoints=[('missing/roi/centered_log_mse','Missing ROI spatial log-MSE'),('missing/roi/log_mse','Missing ROI raw log-MSE'),('full/roi/centered_log_mse','Full-input ROI spatial log-MSE'),('full/whole/absrel','Full-input whole AbsRel')]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,(endpoint,title) in zip(axes.flat,endpoints):
        values=[summary['summaries'][endpoint+'/'+name] for name in names]
        ax.bar(np.arange(5),[v['mean'] for v in values],color=colors)
        for i,v in enumerate(values):ax.scatter(np.full(3,i)+np.array([-.12,0,.12]),v['per_seed'],c='black',s=13,zorder=3)
        ax.set_xticks(np.arange(5),labels,rotation=20);ax.set_title(title);ax.set_ylabel('Lower is better')
    ax=axes[1,1]
    for i,mode in enumerate(MODES):
        c=summary['comparisons']['missing/roi/centered_log_mse/G_'+mode+'-P_'+mode]
        mean=c['relative_error_reduction_percent'];lo,hi=c['interval95_percent']
        ax.errorbar(i,mean,yerr=np.array([[mean-lo],[hi-mean]]),fmt='o',color=colors[2+i*2],capsize=5)
        ax.scatter(np.full(3,i)+[-.12,0,.12],c['per_seed_relative_percent'],s=14,color='gray')
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks([0,1],['Pred geometry','Sensor geometry']);ax.set_title('G vs P: error reduction (%)');ax.set_ylabel('Positive favors G; paired95% interval')
    ax=axes[1,2];v=summary['interactions']['missing/roi/centered_log_mse'];mean=v['interaction_absolute'];lo,hi=v['interval95_absolute']
    ax.errorbar(0,mean,yerr=np.array([[mean-lo],[hi-mean]]),fmt='o',color='#794a94',capsize=6)
    ax.scatter([-.12,0,.12],v['per_seed_interaction_absolute'],color='gray',s=18);ax.axhline(0,color='black',linewidth=.8)
    ax.set_xticks([0],['Sensor PE increment - Pred PE increment']);ax.set_title('Primary interaction: absolute spatial error');ax.set_ylabel('Positive: more G benefit with sensor geometry')
    fig.suptitle('Source geometry x PE | all12 old development places, three paired seeds\nBlack/gray dots: seed means; only last two panels show paired confidence intervals. Sensor inputs are diagnostic.',fontsize=12)
    fig.savefig(folder/'outcomes.png',dpi=150);plt.close(fig)
    # Fixed scene for direct re-observation: source0 and two independent queries.
    rid=p['catalog'][0]['id'];data={m:load(branch_cache(m)/rid/'diagnostic_maps.pt') for m in MODES}
    fig,axes=plt.subplots(3,6,figsize=(16,8),layout='constrained');viewlist=[('source',0),('query',2),('query',11)]
    titles=['Observed RGB','Pred field RGB','Sensor field RGB','Sensor GT depth','Pred field depth','Sensor field depth']
    for row,(view,index) in enumerate(viewlist):
        prefix=f'{view}_{index}';target=data['sensor'][prefix+'_target'].numpy();valid=data['sensor'][prefix+'_valid'].numpy()
        limits=np.percentile(target[valid],[2,98])
        axes[row,0].imshow(data['sensor'][prefix+'_observed_rgb'].numpy().clip(0,1))
        for col,m in [(1,'pred'),(2,'sensor')]:axes[row,col].imshow(data[m][prefix+'_final_rgb'].numpy().clip(0,1))
        axes[row,3].imshow(np.where(valid,target,np.nan),cmap='viridis',vmin=limits[0],vmax=limits[1])
        for col,m in [(4,'pred'),(5,'sensor')]:
            d=data[m][prefix+'_final_depth'].numpy();a=data[m][prefix+'_final_alpha'].numpy()
            im=axes[row,col].imshow(np.where(a>1e-4,d,np.nan),cmap='viridis',vmin=limits[0],vmax=limits[1])
        fig.colorbar(im,ax=list(axes[row,3:]),shrink=.7,label='metres; same scale within row')
        for col in range(6):
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(titles[col])
        axes[row,0].set_ylabel(f'{rid}\n{view}{index}')
    fig.suptitle('One saved field per geometry, shared across views | white: invalid target or uncovered field\nGiven source/query cameras; query targets used only for evaluation.',fontsize=12)
    fig.savefig(folder/'scene_reobservation.png',dpi=150);plt.close(fig)
    # Worst G-P difference, all3seeds, missing ROI. Selection is fixed to primary error, no manual replacement.
    rows=read(OUT/'scoring/metrics.json')['rows'];candidates=[]
    for meta in p['catalog']:
        for q in QUERIES:
            for mode in MODES:
                means={kind:np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows if r['id']==meta['id'] and r['query']==q and r['condition']=='missing' and r['kind']==f'{kind}_{mode}']) for kind in KINDS}
                candidates.append(dict(id=meta['id'],query=q,mode=mode,difference=means['G']-means['P']))
    worst=max(candidates,key=lambda r:(r['difference'],r['id'],r['query'],r['mode']))
    cases=[(p['catalog'][0]['id'],2),(p['catalog'][0]['id'],11),(p['catalog'][1]['id'],5),(worst['id'],worst['query'])]
    predictions={}
    for mode in MODES:
        for kind in KINDS:
            r=read(branch_out(mode)/'evaluation'/f'{kind}_{SEEDS[0]}.json');x=load(r['prediction_path'])
            predictions[(mode,kind)]={tuple(k):d for k,d in zip(x['keys'],x['depth_metric'])}
    fig,axes=plt.subplots(4,8,figsize=(18,10),layout='constrained')
    for row,(rid,q) in enumerate(cases):
        query=load(query_path(rid,q));valid=query['valid'].numpy();target=query['target'].numpy();limits=np.percentile(target[valid],[2,98])
        rgb=query['rgb'].float()/255.;rgb[:,query['image_missing']]=torch.tensor([.485,.456,.406])[:,None]
        axes[row,0].imshow(rgb.permute(1,2,0).numpy());images=[target]
        images += [predictions[(mode,kind)][(rid,q,'missing')].numpy() for mode in MODES for kind in KINDS]
        images += [load(branch_cache(mode)/rid/'diagnostic_maps.pt')[f'query_{q}_final_depth'].numpy() for mode in MODES]
        for col,d in enumerate(images,1):im=axes[row,col].imshow(np.where(valid,d,np.nan),cmap='viridis',vmin=limits[0],vmax=limits[1])
        fig.colorbar(im,ax=list(axes[row,1:]),shrink=.65,label='m')
        for col in range(8):
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['Missing RGB','Sensor GT','Pred P','Pred G','Sensor P','Sensor G','Pred GS','Sensor GS'][col])
        axes[row,0].set_ylabel(f'{rid}/q{q}'+('\nmax G-P' if row==3 else ''))
    fig.suptitle(f'Fixed cases and largest primary G-P regression across either geometry | displayed seed{SEEDS[0]}\nWorst selected by all3seeds; duplicate fixed case is retained. Same depth scale in each row.',fontsize=12)
    fig.savefig(folder/'depth_cases.png',dpi=150);plt.close(fig)
    write_json(folder/'case_selection.json',dict(fixed_cases=[list(c) for c in cases[:3]],worst=worst,displayed_cases=[list(c) for c in cases],unique_cases=len(set(cases)),displayed_seed=SEEDS[0]))
    write_json(folder/'manifest.json',dict(complete=True,files=['outcomes.png','scene_reobservation.png','depth_cases.png'],visual_review_pending=True,
        sha256={f.name:sha256(f) for f in folder.glob('*.png')},scope='Generated old-development diagnostics, actual visual review still required. No new confirmation.'))
    print('Three final figures generated; actual review pending.',flush=True)

if __name__=='__main__':main()
