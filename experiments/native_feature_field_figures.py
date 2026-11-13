"""Two final scientific figures: factorial outcomes and all fixed/maximum cases."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/native_feature_field_mpl')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from experiments import native_feature_field_data as d

def main():
    ss={r:d.read(d.BASEOUT/r/'scoring/summary.json') for r in ('C','J')};f=d.read(d.BASEOUT/'factorial_summary.json');folder=d.BASEOUT/'figures';folder.mkdir(exist_ok=True);colors={'C':'#3879a4','J':'#d78535'}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    panels=[(axes[0,0],None,['G-P','G-S','S-P'],'Native missing ROI spatial: PE'),(axes[0,1],None,['P-Native','S-Native','G-Native','RasterFill-Native'],'Native missing ROI spatial: net use'),(axes[1,0],'feature/development12/fit/full/raw_mse',['G-P','G-S','S-P'],'Full feature raw MSE (secondary)'),(axes[1,1],'native/development12/fit/whole/absrel',['P-RasterFill','S-RasterFill','G-RasterFill'],'Missing whole-image AbsRel')]
    for ax,prefix,names,title in panels:
        for off,row in enumerate(('C','J')):
            s=ss[row];p=prefix or s['primary'];rs=[s['comparisons'][p+'/'+name] for name in names];x=np.arange(len(rs))+(off-.5)*.15;y=np.array([r['reduction_percent'] for r in rs]);lo=np.array([r['interval95_percent'][0] for r in rs]);hi=np.array([r['interval95_percent'][1] for r in rs]);ax.errorbar(x,y,yerr=np.stack((np.maximum(0,y-lo),np.maximum(0,hi-y))),fmt='o',capsize=3,color=colors[row],label=row)
        ax.axhline(0,color='black',lw=.7);ax.axhline(1,color='gray',ls=':',lw=.7);ax.set_xticks(range(len(names)),names,rotation=12);ax.set(title=title,ylabel='Error reduction (%)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    rows=d.read(d.BASEOUT/'formation_statistics.json')['rows'];ax=axes[0,2]
    for row in ('C','J'):
        rs=[r for r in rows if r['row']==row];x=np.array([r['source_initial_raw_MSE'] for r in rs]);y=np.array([r['source_final_raw_MSE'] for r in rs]);ax.scatter(x,y,s=15,alpha=.65,label=row,color=colors[row])
    lim=ax.get_xlim();ax.plot(lim,lim,':',color='gray');ax.set(xlabel='Initial source raw MSE',ylabel='Saved-field source raw MSE',title='44 source-only scenes per formation');ax.legend(fontsize=8)
    I=f['interaction'];ax=axes[1,2];a=I['absolute'];lo,hi=I['absolute_CI95'];ax.errorbar([0],[a],yerr=[[max(0,a-lo)],[max(0,hi-a)]],fmt='o',capsize=4,color='#386d67');ax.scatter([0]*3,I['per_seed_absolute'],s=20,color='#b56f39');ax.axhline(0,color='black',lw=.8);ax.set(xticks=[0],xticklabels=['(JP−JG) − (CP−CG)'],ylabel='Absolute centered log-depth MSE gain',title='Paired formation × PE interaction');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.suptitle('Source feature-field formation | C: values only; J: values + covariance\nUsed development places, 3 paired seeds; source/feature improvement does not establish native PE utility',fontsize=12);fig.savefig(folder/'outcomes.png',dpi=160);fig.savefig(folder/'outcomes.pdf');plt.close(fig)
    cases=[]
    for row in ('C','J'):
        cases +=[dict(row=row,id=m['id'],place=m['place'],query=2,selection='fixed first two training scenes') for m in d.contract()['training_catalog'][:2]]
        cases.append(dict(row=row,**ss[row]['counterexamples'][0],selection='maximum three-seed mean development G-P primary error regression'))
    fig,axes=plt.subplots(6,7,figsize=(19,16),layout='constrained');titles=['Target feature RMS','RasterFill feature RMSE','P feature RMSE','G feature RMSE','P log-depth error','G log-depth error','Effective content coverage'];saved=[];seed=d.SEEDS[0]
    for ri,case in enumerate(cases):
        d.context(case['row']);rid,q=case['id'],case['query'];query=d.load(d.old.query_path(rid,q));target=d.load(d.old.patch_path(rid,q,'full'))['patches'][0].numpy().astype(np.float64);b,a=d.cached_anchor(rid,q);fp={arm:np.load(d.CACHE/'features'/f'{arm}_{seed}'/f'{rid}_{q}.npy') for arm in ('P','G')};dp={arm:np.load(d.CACHE/'depths'/f'{arm}_{seed}'/f'{rid}_{q}.npy') for arm in ('P','G')}
        ft=np.sqrt(np.mean(target**2,-1)).reshape(32,32);fe=[np.sqrt(np.mean((x.astype(np.float64)-target)**2,-1)).reshape(32,32) for x in (b.numpy(),fp['P'],fp['G'])];vmax=max(float(np.quantile(x,.99)) for x in fe);valid=query['valid'].numpy();targetd=query['target'].numpy();de={arm:np.where(valid,np.log(np.maximum(dp[arm],1e-6)/targetd),np.nan) for arm in ('P','G')};dv=max(float(np.nanquantile(abs(x),.99)) for x in de.values());images=[ft,*fe,de['P'],de['G'],a.numpy().reshape(32,32)]
        for ci,im in enumerate(images):
            ax=axes[ri,ci]
            if ci in (1,2,3):ax.imshow(im,cmap='magma',vmin=0,vmax=vmax)
            elif ci in (4,5):ax.imshow(im,cmap='coolwarm',vmin=-dv,vmax=dv)
            else:ax.imshow(im,cmap='viridis',vmin=0,vmax=max(1e-8,float(im.max())))
            if ci<=3 or ci==6:ax.add_patch(Rectangle((7.5,7.5),16,16,fill=False,edgecolor='white',lw=.7))
            else:ax.contour(query['hidden'].numpy().astype(float),levels=[.5],colors=['black'],linewidths=.4)
            ax.set_xticks([]);ax.set_yticks([])
            if ri==0:ax.set_title(titles[ci],fontsize=8)
            if ci==0:ax.set_ylabel(f"{case['row']} / {rid} / q{q}\n{case['place']}",fontsize=8)
        saved.append(dict(row=case['row'],id=rid,query=q,seed=seed,selection=case['selection']))
    fig.suptitle('Each row: fixed two source scenes and largest development G/P counterexample\nMatched color scales within each case; all queries remain scored; seed20261110 shown, no mechanism claim from pictures',fontsize=12);fig.savefig(folder/'cases.png',dpi=160);fig.savefig(folder/'cases.pdf');plt.close(fig);d.write(folder/'selection.json',dict(cases=saved,per_row_counterexamples={r:ss[r]['counterexamples'] for r in ('C','J')},pictures_do_not_replace_full_seed_place_metrics=True))
if __name__=='__main__':torch.set_num_threads(4);main()
