"""Measured outcome overview and fixed/max-counterexample panels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import native_scene_exchange_data as d
from experiments.native_scene_exchange_data import np

def main():
    summary=d.read(d.OUT/'scoring/summary.json');decision=d.read(d.OUT/'method_decision.json');rows=d.read(d.OUT/'scoring/metrics.json')['rows'];folder=d.OUT/'figures';folder.mkdir(exist_ok=True);primary=decision['primary']
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    comparisons=[(['U_G-U_P','R_G-R_P','U_G-U_S','U_S-U_P'],'PE on original development queries'),(['U_G-L','U_G-Native','R_G-L','R_G-Native','L-Native'],'Native net use'),(['U_P-R_P','U_S-R_S','U_G-R_G'],'Feedback versus restricted message')]
    for ax,(names,title) in zip(axes,comparisons):
        for i,name in enumerate(names):
            r=summary['comparisons'][primary+'/'+name];m=r['reduction_percent'];lo,hi=r['interval95_percent'];ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#226991',capsize=3);ax.scatter(r['per_seed_percent'],np.full(3,i)+np.array([-.1,0,.1]),s=12,color='#777777',zorder=3)
        ax.axvline(0,c='gray',lw=.7);ax.axvline(1,c='#9b6767',ls=':',lw=.8);ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Missing ROI centered log-MSE reduction (%)\nPaired place/seed95% interval; gray seed dots')
    I=decision['interaction'];fig.suptitle(f"Native scene exchange | used development data, not fresh\nFeedback x PE: {I['mean_absolute']:.3g} [{I['interval95_absolute'][0]:.3g}, {I['interval95_absolute'][1]:.3g}] absolute MSE",fontsize=12)
    fig.savefig(folder/'outcomes.png',dpi=170);fig.savefig(folder/'outcomes.pdf');plt.close(fig)
    dev=[r for r in rows if r['cohort']=='development' and r['condition']=='missing' and r['query_group']=='fit'];diffs=[]
    for meta in d.plan()['catalog']:
        for q in d.old.QUERIES:
            a=np.mean([r['metrics']['roi']['centered_log_mse'] for r in dev if r['id']==meta['id'] and r['query']==q and r['kind']=='U_G']);b=np.mean([r['metrics']['roi']['centered_log_mse'] for r in dev if r['id']==meta['id'] and r['query']==q and r['kind']=='U_P']);diffs.append(dict(id=meta['id'],query=q,U_G_minus_U_P=float(a-b)))
    worst=max(diffs,key=lambda x:x['U_G_minus_U_P']);fixed=[dict(id=d.plan()['catalog'][0]['id'],query=2),dict(id=d.plan()['catalog'][1]['id'],query=5)];cases=fixed+([worst] if not any(x['id']==worst['id'] and x['query']==worst['query'] for x in fixed) else [])
    fig,axes=plt.subplots(len(cases),6,figsize=(17,3.5*len(cases)),squeeze=False,constrained_layout=True)
    for i,case in enumerate(cases):
        rid,q=case['id'],case['query'];query=d.load(d.old.query_path(rid,q));target=query['target'].numpy();valid=query['valid'].numpy();vmax=np.nanpercentile(target[valid],98);image=d.load(d.old.query_path(rid,q))['rgb'].permute(1,2,0).numpy().copy();image[112:336,112:336]=[124,116,104];axes[i,0].imshow(image);axes[i,0].set_title(f'{rid} q{q} input')
        im=axes[i,1].imshow(np.where(valid,target,np.nan),vmin=0,vmax=vmax,cmap='viridis');axes[i,1].set_title('GT valid depth')
        for j,method in enumerate(['Native','L','U_P','U_G'],2):
            row=next(r for r in dev if r['id']==rid and r['query']==q and r['kind']==method and r['seed'] in (None,d.SEEDS[0]));pred=np.load(row['path']);axes[i,j].imshow(np.where(valid,pred,np.nan),vmin=0,vmax=vmax,cmap='viridis');axes[i,j].set_title(f"{method}: {row['metrics']['roi']['centered_log_mse']:.4f}")
        for ax in axes[i]:ax.axis('off')
        fig.colorbar(im,ax=list(axes[i,1:]),fraction=.018,pad=.01,label='Depth (m)')
    fig.suptitle('Fixed first two cases + maximum three-seed mean U/G minus U/P regression\nOnly seed20261110 displayed; panel values=ROI centered log-MSE. White=invalid GT; all queries retained.',fontsize=12);fig.savefig(folder/'cases.png',dpi=170);fig.savefig(folder/'cases.pdf');plt.close(fig)
    d.write(folder/'case_selection.json',dict(fixed=fixed,worst=worst,cases=cases,all_primary_case_differences=diffs,displayed_seed=d.SEEDS[0]))

if __name__=='__main__':main()
