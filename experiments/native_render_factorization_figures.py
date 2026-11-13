"""Fixed numeric evidence and predetermined same-scale depth case plate."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/liwt_native_render_factorization_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.native_render_factorization import OUT,ARMS,write
from experiments import native_coverage_data as d
from experiments.gaussian_scene_data import query_image


def main():
    torch.set_num_threads(4);s=d.read(OUT/'scoring/summary.json');rows=d.read(OUT/'scoring/metrics.json')['rows'];folder=OUT/'figures';folder.mkdir(exist_ok=True)
    write(folder/'contract.json',dict(backend='Python/matplotlib, existing project stack',claim='Local-content removal and attenuation removal can have different fixed-model costs; train-clip and old-development evidence must remain separate.',panels=['Absolute primary/secondary spatial errors and individual seeds','Six prospective contrasts with paired place/seed95% intervals in each scope','Fixed depth cases plus prespecified largest mean N-R development case'],exports=['outcomes.png','depth_cases.png'],statistics='5000 paired place/seed resamples; seed20261113; query/place equal weighting',image_integrity='Seed20261110 predictions; same metric depth scale per row, GT valid mask only for display; all cases scored; no cosmetic smoothing.',risks=['Fixed-weight distribution perturbation is not causal isolation or a newly trained method','A zero-crossing interval is not equivalence','Train-clip utility is not independent confirmation']))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    methods=['Local','R','N','C','U','M'];colors=['#999999','#3d846e','#8eb3a3','#8b95b8','#bac0d4','#d1af70']
    labels=['Local (trained control)','R local, coverage','N local, unit','C global, coverage','U global, unit','M norm matched']
    for col,(cohort,title) in enumerate([('train32','32 trained clips / other queries (primary)'),('development12','12 old development places / original queries')]):
        key=f'{cohort}/missing/roi/centered_log_mse';ax=axes[0,col]
        for i,(kind,color) in enumerate(zip(methods,colors)):
            v=s['summaries'][key+'/'+kind];ax.bar(i,v['mean'],color=color)
            ax.scatter(np.full(3,i)+[-.12,0,.12],v['per_seed'],s=13,c='#333333',zorder=3)
        ax.set_xticks(range(6),methods);ax.set_ylabel('Missing ROI spatial log-MSE (lower better)');ax.set_title(title)
        ax=axes[1,col];names=['D_address','D_coverage','interaction','U_minus_N','U_minus_C','D_matched']
        for i,name in enumerate(names):
            v=s['contrasts'][key][name];lo,hi=v['interval95_absolute'];ax.plot([lo,hi],[i,i],c='#597e86',lw=2);ax.scatter(v['mean_absolute'],i,c='#597e86',s=30)
            ax.scatter(v['per_seed_absolute'],np.full(3,i)+[-.09,0,.09],s=11,c='#999999')
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(6),['C-R: remove local content','N-R: remove attenuation','U-C-N+R: interaction','U-N: content cost, unit','U-C: attenuation cost, global','M-R: norm matched']);ax.invert_yaxis();ax.set_xlabel('Absolute error contrast; positive simple difference = cost')
        ax.set_title('Prospective contrasts, paired95% intervals')
    fig.suptitle('Fixed Render weights | 0 training | old sensor-assisted data\nR=original, N=local/unit, C=global/coverage, U=global/unit, M=one scalar norm match',fontsize=12)
    fig.savefig(folder/'outcomes.png',dpi=150);plt.close(fig)
    candidates=[]
    for meta in d.plan()['catalog']:
        for q in d.QUERIES:
            means={kind:float(np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows if r['id']==meta['id'] and r['query']==q and r['condition']=='missing' and r['kind']==kind])) for kind in ['R','N']}
            candidates.append(dict(id=meta['id'],query=q,difference=means['N']-means['R'],means=means))
    worst=max(candidates,key=lambda x:(x['difference'],x['id'],x['query']))
    selected=list(dict.fromkeys([('fit_00',1),('confirm_12',1),('fidelity_00',2),(worst['id'],worst['query'])]));original={}
    for cohort in ['train','development']:
        marker=d.read(d.OUT/'evaluation'/f'{cohort}_Render_{d.SEEDS[0]}.json');payload=d.load(marker['prediction_path']);original.update({tuple(k):v.numpy() for k,v in zip(payload['keys'],payload['depth_metric'])})
    fig,axes=plt.subplots(len(selected),7,figsize=(20,3*len(selected)),layout='constrained',squeeze=False)
    case_metrics=[]
    for i,(rid,q) in enumerate(selected):
        query=d.load(d.query_path(rid,q));target=query['target'].numpy();valid=query['valid'].numpy();lo,hi=np.percentile(target[valid],[2,98]);axes[i,0].imshow(query_image(query,'missing').permute(1,2,0).numpy())
        values=[target,original[rid,q,'missing']]+[np.load(OUT/'predictions'/f'{d.SEEDS[0]}_{rid}_{q}_missing_{arm}.npy') for arm in ARMS]
        for j,value in enumerate(values,1):im=axes[i,j].imshow(np.where(valid,value,np.nan),cmap='viridis',vmin=lo,vmax=hi)
        fig.colorbar(im,ax=list(axes[i,1:]),shrink=.6,label='m, shared row scale');axes[i,0].set_ylabel(f'{rid}/q{q}')
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
        case_metrics.append(dict(id=rid,query=q,display_range_m=[float(lo),float(hi)],mean_seed_metrics={kind:float(np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows if r['id']==rid and r['query']==q and r['condition']=='missing' and r['kind']==kind])) for kind in ['R']+list(ARMS)}))
    for ax,label in zip(axes[0],['Missing input','Sensor GT','R original','N local/unit','C global/coverage','U global/unit','M norm matched']):ax.set_title(label)
    fig.suptitle('Fixed old/new training views + fixed development view + largest mean N-R development counterexample\nSeed20261110 shown; selection uses3seed mean; every case remains in the full scores',fontsize=12)
    fig.savefig(folder/'depth_cases.png',dpi=150);plt.close(fig)
    write(folder/'selection.json',dict(cases=selected,worst_development_N_minus_R=worst,case_metrics=case_metrics,files={str(x):d.sha256(x) for x in [folder/'outcomes.png',folder/'depth_cases.png']},actual_visual_review_pending=True))
    print('Two fixed figures generated.',flush=True)


if __name__=='__main__':main()
