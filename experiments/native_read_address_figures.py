"""Diagnostic fixed comparisons and explicitly ranked used-development cases."""
import os
os.environ['MPLCONFIGDIR']='/tmp/liwt_native_read_address_mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import native_read_address_data as d
from experiments.native_read_address_data import torch,np
def save(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(d.OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
def main():
    torch.set_num_threads(4);(d.OUT/'figures').mkdir(exist_ok=True);summary=d.read(d.OUT/'scoring/summary.json');rows=d.read(d.OUT/'scoring/metrics.json')['rows'];primary='development12/fit/missing/roi/centered_log_mse'
    # Prespecified per-seed/place counterexamples, independent of illustrative image selection.
    counterexamples={};pairs=[('A','P'),('A','S'),('P','S')]+[(a,b) for a in d.ARMS for b in ('B','Native','Local','Render')]
    for condition,region,metric in [('missing','roi','centered_log_mse'),('full','whole','absrel')]:
        chosen=[r for r in rows if r['cohort']=='development' and r['query_group']=='fit' and r['condition']==condition];cell={}
        for r in chosen:
            value=r['metrics'][region][metric]
            if value is not None:cell.setdefault((r['kind'],r['seed'],r['id']),[]).append(value)
        for a,b in pairs:
            records=[]
            for seed in d.SEEDS:
                for meta in d.plan()['catalog']:
                    rid=meta['id'];va=cell[(a,seed,rid)];vb=cell[(b,None if b=='Native' else seed,rid)];assert len(va)==len(vb)==4
                    x,y=float(np.mean(va)),float(np.mean(vb));records.append(dict(seed=seed,id=rid,place=meta['place'],test=x,reference=y,error_increase=x-y,reduction_percent=100*(1-x/y) if y else None))
            counterexamples[f'{condition}/{region}/{metric}/{a}-{b}']=dict(worst_absolute=max(records,key=lambda x:x['error_increase']),worst_relative=min(records,key=lambda x:x['reduction_percent']),per_seed_place=records)
    d.write(d.OUT/'scoring/counterexamples.json',dict(scope='All12useddevelopment places x3pairedseeds, four original queries averaged per cell; descriptive max counterexamples only, no tuning or new model calls.',comparisons=counterexamples))
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    d.write(d.OUT/'figures/figure_contract.json',dict(backend='Existing Python/matplotlib diagnostic workflow',purpose='Prespecified address-vs-center, shape-range and native net utility plus whole-image costs and counterexamples; no journal target',formats=['PNG','PDF','SVG'],interval='5000 paired seed/place bootstrap; query average within place',risks=['All used development','Same frozen B content/receiver, distinct learned scalar biases','A/S limits shape claim separately from generic A/P utility','Examples selected by three-seed centered error, displayed one seed; color clipping can hide raw bias']))
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    def forest(ax,prefix,pairs,title):
        for i,pair in enumerate(pairs):
            v=summary['comparisons'][prefix+'/'+pair];lo,hi=v['interval95_percent'];ax.plot([lo,hi],[i,i],color='#498d91',lw=1.7);ax.scatter(v['reduction_percent'],i,color='#498d91',s=28,zorder=3)
        ax.axvline(0,color='.6',lw=.7);ax.set_yticks(range(len(pairs)),pairs);ax.invert_yaxis();ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Error reduction (%) · paired 95% interval');ax.grid(axis='x',alpha=.15)
    forest(axes[0,0],primary,['A-P','A-S','P-S'],'a  Prespecified address and shape comparisons')
    forest(axes[0,1],primary,['A-B','P-B','S-B','A-Native','A-Local','A-Render'],'b  Adaptation and native net utility')
    forest(axes[1,0],'development12/fit/full/whole/absrel',['A-B','P-B','S-B','A-Native','A-Local','A-Render'],'c  Full-input whole-image AbsRel')
    colors={'A':'#498d91','P':'#877b9d','S':'#bb9060'}
    for arm in d.ARMS:
        values=np.array([d.load(d.final(arm,s))['gamma']['gamma'].numpy() for s in d.SEEDS])
        for row in values:axes[1,1].plot(range(1,25),row,color=colors[arm],lw=.7,alpha=.35)
        axes[1,1].plot(range(1,25),values.mean(0),color=colors[arm],label=arm,lw=1.8)
    axes[1,1].axhline(0,color='.6',lw=.7);axes[1,1].set_title('d  Learned per-block scalar (thin = seeds)',loc='left',fontweight='bold');axes[1,1].set_xlabel('Original global block');axes[1,1].set_ylabel('gamma');axes[1,1].legend(frameon=False);fig.suptitle('GS read-address PE · identical frozen states/receivers · all used development',fontsize=12);save(fig,'outcomes')
    cells={}
    for r in rows:
        if r['cohort']=='development' and r['query_group']=='fit' and r['condition']=='missing' and r['kind'] in ('A','P'):cells.setdefault((r['id'],r['query']),{}).setdefault(r['kind'],[]).append(r['metrics']['roi']['centered_log_mse'])
    effects=sorted((float(np.mean(v['A'])-np.mean(v['P'])),rid,q) for (rid,q),v in cells.items());zero=d.read(d.OUT/'scoring/independent_recomputation.json')['zero_coverage_queries'];dev={m['id'] for m in d.plan()['catalog']};z=next(x for x in zero if x['id'] in dev);selected=[(effects[-1][1],effects[-1][2],'Worst A/P'),(effects[0][1],effects[0][2],'Best A/P'),(d.plan()['catalog'][0]['id'],2,'Fixed first dev'),(z['id'],z['query'],'Original zero coverage')];used=set();distinct=[]
    for rid,q,label in selected:
        if (rid,q) in used:
            options=([(x['id'],x['query']) for x in zero if x['id'] in dev]+[(x['id'],x['query']) for x in zero]) if label=='Original zero coverage' else [(m['id'],qq) for m in d.plan()['catalog'] for qq in d.parent.old.QUERIES];rid,q=next(x for x in options if x not in used);label+=' (next distinct)'
        distinct.append((rid,q,label));used.add((rid,q))
    d.write(d.OUT/'figures/case_selection.json',dict(rule='Worst/best three-seed mean AminusP centered primary query error; fixed first development and first original-zero case; duplicates next deterministic distinct',selected=[dict(id=rid,query=q,label=label) for rid,q,label in distinct],all_effects=[dict(id=rid,query=q,A_minus_P=e) for e,rid,q in effects],display_seed=d.SEEDS[0]))
    fig,axes=plt.subplots(4,8,figsize=(17,9),constrained_layout=True);difference_images=[];difference_values=[]
    from experiments import camera_only_feature_data as oldnative
    for row,(rid,q,label) in enumerate(distinct):
        query=d.load(d.parent.old.query_path(rid,q));rgb=query['rgb'].permute(1,2,0).numpy().copy();rgb[query['image_missing'].numpy()]=0;target=query['target'].numpy();valid=query['valid'].numpy();lo,hi=np.percentile(target[valid],[2,98]);axes[row,0].imshow(rgb);axes[row,0].set_ylabel(f'{label}\n{rid},q{q}\nDepth{lo:.2f}–{hi:.2f}m',fontsize=8);axes[row,1].imshow(np.ma.masked_where(~valid,target),vmin=lo,vmax=hi,cmap='viridis')
        reference_log=None
        for col,method in enumerate(('Native','B','P','S','A'),2):
            if method=='Native':record=d.read(oldnative.OUT/'baselines'/f'{rid}_{q}.json')['methods']['Native'];pred=np.load(record['path'],allow_pickle=False)
            else:
                path=d.parent.CACHE/'predictions'/f'C_P_{d.SEEDS[0]}'/f'{rid}.pt' if method=='B' else d.predpath(method,d.SEEDS[0],rid);payload=d.load(path);index=[tuple(k) for k in payload['keys']].index((rid,q,'missing'));pred=payload['depth_metric'][index].numpy()
            axes[row,col].imshow(pred,vmin=lo,vmax=hi,cmap='viridis')
            if method=='P':reference_log=np.log(np.maximum(pred,1e-6)/np.maximum(target,1e-6))
            if method=='A':
                roi=valid&query['hidden'].numpy();res=np.log(np.maximum(pred,1e-6)/np.maximum(target,1e-6));difference=np.full_like(res,np.nan)
                if roi.any():difference[roi]=(res[roi]-res[roi].mean())**2-(reference_log[roi]-reference_log[roi].mean())**2;difference_values.append(difference[roi])
                difference_images.append(axes[row,7].imshow(difference,vmin=-1,vmax=1,cmap='coolwarm'))
        for col,ax in enumerate(axes[row]):
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(['Missing RGB','Target','Native','B','P','S','A','Spatial error A−P'][col],fontsize=9)
    limit=max(float(np.percentile(np.abs(np.concatenate(difference_values)),98)),1e-5) if difference_values else 1e-5
    for im in difference_images:im.set_clim(-limit,limit)
    selection=d.read(d.OUT/'figures/case_selection.json');selection.update(difference_map='ROI-only squared centered log-error(A) minus(P), red=worseA, blue=betterA; display seed only, distinct from three-seed selection.',shared_difference_color_limit=limit,difference_color_limit_rule='98th percentile absolute ROI-pixel delta over all four preselected images, clipped');d.write(d.OUT/'figures/case_selection.json',selection)
    fig.colorbar(difference_images[-1],ax=axes[:,7].tolist(),shrink=.6,label='Δ centered squared log error (A−P)');fig.suptitle('Relative centered-error examples · target2–98% row depth range (clipped) · seed20261110',fontsize=11);save(fig,'cases');print(dict(figures_created=2,actual_view_required=True),flush=True)
if __name__=='__main__':main()
