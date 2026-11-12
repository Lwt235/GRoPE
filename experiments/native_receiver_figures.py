"""Full matched outcomes, train/dev contrast and fixed/counterexample depth cases."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.native_receiver_data import *
from experiments.gaussian_scene_data import query_image


def main():
    torch.set_num_threads(4);p=plan();s=read(OUT/'scoring/summary.json');folder=OUT/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['F_Local','F_Global','F_Render','A_Local','A_Global','A_Render']
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for ax,cohort,title in [(axes[0,0],'development','Development: missing ROI spatial log-MSE'),(axes[0,1],'train','Fitted training inputs: same metric')]:
        vv=[s['summaries'][f'{cohort}/missing/roi/centered_log_mse/{k}'] for k in names]
        ax.bar(range(6),[v['mean'] for v in vv],color=['#aaa','#bdac94','#81ae9b','#888','#b78750','#368b68'])
        for i,v in enumerate(vv):ax.scatter(np.full(3,i)+[-.12,0,.12],v['per_seed'],c='black',s=15)
        ax.set_xticks(range(6),names,rotation=20);ax.set_title(title);ax.set_ylabel('Lower is better; seed dots, no intervals here')
    ax=axes[1,0];pairs=['A_Render-A_Local','A_Render-A_Global','F_Render-F_Local','F_Render-F_Global','A_Render-A_Render_shuffle_binding']
    for i,pair in enumerate(pairs):
        v=s['comparisons']['development/missing/roi/centered_log_mse/'+pair];lo,hi=v['interval95_percent']
        ax.plot([lo,hi],[i,i],color='#576b95');ax.scatter([v['relative_error_reduction_percent']],[i],c='#576b95')
    ax.axvline(0,c='black',lw=.8);ax.set_yticks(range(len(pairs)),pairs);ax.invert_yaxis();ax.set_xlabel('Error reduction %, positive favors first; paired95% intervals')
    ax.set_title('Utility and binding dependency are distinct')
    ax=axes[1,1]
    for i,control in enumerate(['Local','Global']):
        v=s['interactions']['development/missing/roi/centered_log_mse/Render-'+control];lo,hi=v['interval95_absolute']
        ax.plot([lo,hi],[i,i],color='#9c6375');ax.scatter([v['mean_absolute']],[i],c='#9c6375')
        ax.scatter(v['per_seed_absolute'],np.full(3,i)+[-.1,0,.1],s=15,c='gray')
    ax.axvline(0,c='black',lw=.8);ax.set_yticks([0,1],['R-Local interaction','R-Global interaction']);ax.invert_yaxis()
    ax.set_xlabel('Absolute log-MSE difference; negative favors adaptation');ax.set_title('(R-control) adapted minus frozen')
    fig.suptitle('Native receiver adaptation | 3 paired seeds | old sensor-assisted static data\nF=fusion only, A=fusion+all native interaction blocks+depth head. Training inputs are fitted, not fresh.',fontsize=12)
    fig.savefig(folder/'outcomes.png',dpi=150);plt.close(fig)
    rows=read(OUT/'scoring/metrics.json')['rows'];candidates=[]
    for meta in p['catalog']:
        for q in QUERIES:
            means={k:float(np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows if r['cohort']=='development' and r['kind']==k and r['id']==meta['id'] and r['query']==q and r['condition']=='missing'])) for k in ['A_Local','A_Render']}
            candidates.append(dict(id=meta['id'],query=q,difference=means['A_Render']-means['A_Local'],means=means))
    worst=max(candidates,key=lambda r:(r['difference'],r['id'],r['query']))
    fixed=[(p['catalog'][0]['id'],2),(p['catalog'][0]['id'],11),(p['catalog'][1]['id'],5)]
    cases=list(dict.fromkeys(fixed+[(worst['id'],worst['query'])]));pred={}
    for name in ['A_Local','A_Global','A_Render','F_Render']:
        kind=name[2:]
        marker=(OUT/'evaluation'/f'adapted_development_{kind}_{SEEDS[0]}.json' if name.startswith('A') else FROZEN/'evaluation'/f'{kind}_{SEEDS[0]}.json')
        payload=load(read(marker)['prediction_path']);pred[name]={tuple(k):v for k,v in zip(payload['keys'],payload['depth_metric'])}
    fig,axes=plt.subplots(len(cases),6,figsize=(17,3*len(cases)),layout='constrained',squeeze=False)
    for row,(rid,q) in enumerate(cases):
        query=load(query_path(rid,q));valid=query['valid'].numpy();gt=query['target'].numpy();lo,hi=np.percentile(gt[valid],[2,98])
        axes[row,0].imshow(query_image(query,'missing').permute(1,2,0).numpy())
        images=[gt]+[pred[k][rid,q,'missing'].numpy() for k in ['A_Local','A_Global','A_Render','F_Render']]
        for col,image in enumerate(images,1):im=axes[row,col].imshow(np.where(valid,image,np.nan),cmap='viridis',vmin=lo,vmax=hi)
        fig.colorbar(im,ax=list(axes[row,1:]),shrink=.65,label='m; shared row scale')
        for col in range(6):
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['Missing RGB','Sensor GT','Adapted Local','Adapted Global','Adapted Render','Frozen Render'][col])
        axes[row,0].set_ylabel(f'{rid}/q{q}'+('\nmax adapted R-L' if (rid,q)==(worst['id'],worst['query']) else ''))
    fig.suptitle('Fixed cases plus largest adapted R-Local regression by3seed means\nDisplayed seed20261110; white=invalid GT. Not a sample of all cases.',fontsize=12)
    fig.savefig(folder/'depth_cases.png',dpi=150);plt.close(fig)
    write_json(folder/'case_selection.json',dict(fixed_cases=[list(x) for x in fixed],worst=worst,displayed_cases=[list(x) for x in cases],displayed_seed=SEEDS[0]))
    write_json(folder/'manifest.json',dict(complete=True,sha256={f:sha256(folder/f) for f in ['outcomes.png','depth_cases.png']},visual_review_pending=True))
    print('Two figures generated; actual review remains.',flush=True)


if __name__=='__main__':main()
