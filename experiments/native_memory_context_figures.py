"""Two prospective figures: endpoints/increments/cost and retained depth cases."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/liwt_native_memory_context_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.native_memory_context_data import *
from experiments.gaussian_scene_data import query_image


def main():
    torch.set_num_threads(4);p=plan();s=read(OUT/'scoring/summary.json');cost=read(OUT/'cost.json');folder=OUT/'figures';folder.mkdir(exist_ok=True)
    write(folder/'contract.json',dict(backend='existing Python/matplotlib stack',claim='Source writer relationship PE judged against common full geometry/center writer, downstream scope and write/query amortization.',panels=['Primary absolute endpoints and three seeds','G/P,G/S,S/P and inherited utility paired intervals','Training-clip versus development scope','Complete warm query timing with individual measured calls'],depth='Fixed fit_00/q1,confirm_12/q1,fidelity_00/q2,fidelity_02/q11 plus largest mean G-P development original-query difference; merge duplicates; same metric scale per row.',statistics='5000 paired seed/place resamples seed20261113; finite3seed/12dev-place scope; descriptive related comparisons',exports=['outcomes.png','depth_cases.png'],integrity='No smoothing, no exclusions or contrast adjustment by arm; GT valid mask only for display. All case queries remain scored.'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained');endpoint='development/original/missing/roi/centered_log_mse'
    methods=['Previous_L','Previous_R','P','S','G'];labels=['Local (inherited)','Render (inherited)','Center P','Sphere S','Full G'];colors=['#aaaaaa','#777777','#6488ad','#c6a86d','#4d8879'];ax=axes[0,0]
    for i,(kind,color) in enumerate(zip(methods,colors)):
        v=s['summaries'][endpoint+'/'+kind];ax.bar(i,v['mean'],color=color);ax.scatter(np.full(3,i)+[-.1,0,.1],v['per_seed'],s=15,c='#333333')
    ax.set_xticks(range(5),labels,rotation=15);ax.set_ylabel('Missing ROI spatial log-MSE; lower better');ax.set_title('12 old development places / original queries')
    ax=axes[0,1];pairs=['G-P','G-S','S-P','G-Previous_R','G-Previous_L']
    for i,pair in enumerate(pairs):
        v=s['comparisons'][endpoint+'/'+pair];lo,hi=v['interval95_percent'];ax.plot([lo,hi],[i,i],c='#4d8879',lw=2);ax.scatter(v['relative_error_reduction_percent'],i,c='#4d8879',s=30);ax.scatter(v['per_seed_relative_percent'],np.full(3,i)+[-.08,0,.08],s=12,c='#888888')
    ax.axvline(0,c='black',lw=.8);ax.set_yticks(range(5),['G vs P (primary)','G vs S','S vs P','G vs inherited Render','G vs inherited Local']);ax.invert_yaxis();ax.set_xlabel('Error reduction (%); positive favors first');ax.set_title('Paired place/seed95% intervals and seeds')
    ax=axes[1,0]
    scopes=[('development','original','Dev / original'),('development','heldout','Dev / other'),('train_original','heldout','Original train / other'),('train_added','heldout','Added train / other')]
    for i,(cohort,view,label) in enumerate(scopes):
        for pair,color,shift in [('G-P','#6488ad',-.08),('G-Previous_R','#4d8879',.08)]:
            v=s['comparisons'][f'{cohort}/{view}/missing/roi/centered_log_mse/'+pair];lo,hi=v['interval95_percent'];ax.plot([lo,hi],[i+shift,i+shift],c=color,lw=2);ax.scatter(v['relative_error_reduction_percent'],i+shift,c=color,s=25,label=pair.replace('Previous_','') if i==0 else None)
    ax.axvline(0,c='black',lw=.8);ax.set_yticks(range(4),[r[2] for r in scopes]);ax.invert_yaxis();ax.legend(fontsize=8);ax.set_xlabel('Error reduction (%)');ax.set_title('PE increment and retained Render utility by scope')
    ax=axes[1,1]
    for i,kind in enumerate(['Render','P','S','G']):
        values=[1000*v for r in cost['records'] if r['kind']==kind for v in r['seconds']];ax.scatter(np.full(len(values),i)+np.linspace(-.12,.12,len(values)),values,s=9,alpha=.4,c=colors[1+i] if i<4 else '#444444');ax.plot([i-.2,i+.2],[np.median(values)]*2,c='black',lw=2)
    ax.set_xticks(range(4),['Render','P','S','G']);ax.set_ylabel('Complete warm query (ms)');ax.set_title('Encoder + saved-state render + native depth\n2 fixed queries ×20 calls; shared GPU, no exclusivity claim')
    fig.suptitle('Source-only shared content writer | relation PE |32 trained /12 old development places',fontsize=12);fig.savefig(folder/'outcomes.png',dpi=150);plt.close(fig)
    rows=read(OUT/'scoring/metrics.json')['rows'];candidates=[]
    for meta in p['catalog']:
        for q in QUERIES:
            means={k:float(np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows if r['id']==meta['id'] and r['query']==q and r['condition']=='missing' and r['kind']==k])) for k in ['P','G']};candidates.append(dict(id=meta['id'],query=q,difference=means['G']-means['P'],means=means))
    worst=max(candidates,key=lambda r:(r['difference'],r['id'],r['query']));selected=list(dict.fromkeys([('fit_00',1),('confirm_12',1),('fidelity_00',2),('fidelity_02',11),(worst['id'],worst['query'])]));predictions={}
    for kind in ['Previous_L','Previous_R']+list(KINDS):
        pred={}
        for cohort in ['development','train']:
            path=parent.OUT/'evaluation'/f'{cohort}_{"Local" if kind=="Previous_L" else "Render"}_{SEEDS[0]}.json' if kind.startswith('Previous_') else OUT/'evaluation'/f'{cohort}_{kind}_{SEEDS[0]}.json'
            record=read(path);payload=load(record['prediction_path']);pred.update({tuple(k):v.numpy() for k,v in zip(payload['keys'],payload['depth_metric'])})
        predictions[kind]=pred
    fig,axes=plt.subplots(len(selected),7,figsize=(20,3*len(selected)),layout='constrained',squeeze=False);case_metrics=[]
    for i,(rid,q) in enumerate(selected):
        query=load(query_path(rid,q));target=query['target'].numpy();valid=query['valid'].numpy();lo,hi=np.percentile(target[valid],[2,98]);axes[i,0].imshow(query_image(query,'missing').permute(1,2,0).numpy())
        for j,value in enumerate([target]+[predictions[k][rid,q,'missing'] for k in methods],1):im=axes[i,j].imshow(np.where(valid,value,np.nan),cmap='viridis',vmin=lo,vmax=hi)
        fig.colorbar(im,ax=list(axes[i,1:]),shrink=.6,label='m, shared row scale');axes[i,0].set_ylabel(f'{rid}/q{q}')
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
        case_metrics.append(dict(id=rid,query=q,range_m=[float(lo),float(hi)],means={k:float(np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows if r['id']==rid and r['query']==q and r['condition']=='missing' and r['kind']==k])) for k in methods}))
    for ax,label in zip(axes[0],['Missing input','Sensor GT','Local inherited','Render inherited','Center P','Sphere S','Full G']):ax.set_title(label)
    fig.suptitle('Fixed training/development views + largest mean G-P development difference\nSeed20261110 shown; three-seed selection; all queries scored, same metric color scale per row',fontsize=12);fig.savefig(folder/'depth_cases.png',dpi=150);plt.close(fig)
    write(folder/'selection.json',dict(cases=selected,worst_development_G_minus_P=worst,case_metrics=case_metrics,files={str(x):sha256(x) for x in [folder/'outcomes.png',folder/'depth_cases.png']},actual_visual_review_pending=True));print('Two prospective figures generated.',flush=True)


if __name__=='__main__':main()
