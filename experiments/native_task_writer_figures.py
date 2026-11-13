"""Python diagnostic evidence figures: paired primary/utility and selected cases."""
import os
os.environ['MPLCONFIGDIR']='/tmp/liwt_native_task_writer_mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import native_task_writer_data as d
from experiments.native_task_writer_data import np,torch

def save(fig,name):
    for suffix in ('png','pdf','svg'):fig.savefig(d.OUT/'figures'/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    torch.set_num_threads(4);(d.OUT/'figures').mkdir(exist_ok=True);s=d.read(d.OUT/'scoring/summary.json');primary='development12/fit/missing/roi/centered_log_mse';data=d.read(d.OUT/'scoring/metrics.json')['rows']
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    d.write(d.OUT/'figures/figure_contract.json',dict(backend='Existing Python/matplotlib experiment workflow',purpose='Diagnostic report, no journal submission target',claim='Display prespecified PE total effects, geometry interaction and native utility together with complete-image costs and counterexamples',panels=['C/J PE forest with95%paired intervals','Each arm vs Native primary error','Full input whole-image AbsRel effects','Per-place formation×PE absolute interaction'],cases='Worst and best development JG/JP query by three-seed average, plus fixed first development and original-zero query; duplicates replaced deterministically',review_risks=['All data used development','Three seeds and12places only','Field differences allowed: total training-chain effect','New support masks arm-dependent','Examples not representative aggregate'],formats=['PNG','editable PDF','SVG'],statistics='5000 paired seed/place bootstrap; query average within place;95%'))
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    def forest(ax,prefix,pairs,title):
        for i,pair in enumerate(pairs):
            v=s['comparisons'][prefix+'/'+pair];mean=v['reduction_percent'];lo,hi=v['interval95_percent'];color='#827b9a' if pair.startswith('C') else '#458e91'
            ax.plot([lo,hi],[i,i],color=color,lw=1.8);ax.scatter([mean],[i],s=30,color=color,zorder=3)
        ax.set_yticks(range(len(pairs)),pairs);ax.axvline(0,color='.65',lw=.7);ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Error reduction (%) · paired 95% interval');ax.grid(axis='x',alpha=.15);ax.invert_yaxis()
    forest(axes[0,0],primary,['C_G-C_P','C_G-C_S','C_S-C_P','J_G-J_P','J_G-J_S','J_S-J_P'],'a  Prespecified PE total effects')
    forest(axes[0,1],primary,[a+'-Native' for a in d.ARMS],'b  Native net utility (primary endpoint)')
    forest(axes[1,0],'development12/fit/full/whole/absrel',[a+'-Native' for a in d.ARMS],'c  Whole-image AbsRel with full input')
    I=s['interactions'][primary];ax=axes[1,1];values=I['per_place_absolute'];ax.bar(range(12),values,color='#7aa4a5');ax.axhline(0,color='.6',lw=.7);ax.set_xticks(range(12),range(1,13));ax.set_xlabel('Development place (catalog order)');ax.set_ylabel('(EJP−EJG)−(ECP−ECG)');ax.set_title('d  Geometry formation × PE interaction',loc='left',fontweight='bold');ax.text(.02,.98,f"Mean {I['mean_absolute']:.4g}\nPaired95% {I['interval95_absolute'][0]:.4g}, {I['interval95_absolute'][1]:.4g}",transform=ax.transAxes,va='top',fontsize=8)
    fig.suptitle('Native task writer: all used development · 12 places × 3 paired seeds',fontsize=12);save(fig,'outcomes')
    rr=[r for r in data if r['cohort']=='development' and r['query_group']=='fit' and r['condition']=='missing' and r['kind'] in ('J_G','J_P')];cells={}
    for row in rr:cells.setdefault((row['id'],row['query']),{}).setdefault(row['kind'],[]).append(row['metrics']['roi']['centered_log_mse'])
    effects=sorted([(float(np.mean(v['J_G'])-np.mean(v['J_P'])),rid,q) for (rid,q),v in cells.items()]);zero=d.read(d.OUT/'scoring/independent_recomputation.json')['zero_coverage_queries'];dev={m['id'] for m in d.plan()['catalog']};z=next((x for x in zero if x['id'] in dev),zero[0]);selected=[(effects[-1][1],effects[-1][2],'Worst JG/JP'),(effects[0][1],effects[0][2],'Best JG/JP'),(d.plan()['catalog'][0]['id'],2,'Fixed first dev'),(z['id'],z['query'],'Original zero coverage')]
    distinct=[];used=set()
    for rid,q,label in selected:
        if (rid,q) in used:
            if label=='Original zero coverage':options=[(x['id'],x['query']) for x in zero if x['id'] in dev]+[(x['id'],x['query']) for x in zero]
            elif label=='Fixed first dev':options=[(m['id'],qq) for m in d.plan()['catalog'] for qq in d.old.QUERIES]
            else:options=[(rr,qq) for _,rr,qq in effects]
            rid,q=next(x for x in options if x not in used);label+=' (next distinct)'
        distinct.append((rid,q,label));used.add((rid,q))
    selected=distinct
    d.write(d.OUT/'figures/case_selection.json',dict(rule='Three-seed average JG−JP primary query error; explicit worst/best, fixed first development and original-zero example; display seed20261110',selected=[dict(id=x,query=q,label=label) for x,q,label in selected],all_development_query_effects=[dict(id=rid,query=q,JG_minus_JP=value) for value,rid,q in effects]))
    fig,axes=plt.subplots(4,8,figsize=(17,9),constrained_layout=True);methods=['Native','C_P','C_G','J_P','J_G']
    from experiments import camera_only_feature_data as baseline
    for row,(rid,q,label) in enumerate(selected):
        query=d.load(d.old.query_path(rid,q));rgb=query['rgb'].permute(1,2,0).numpy().copy();rgb[query['image_missing'].numpy()]=0;axes[row,0].imshow(rgb);axes[row,0].set_ylabel(f'{label}\n{rid}, q{q}',fontsize=8)
        target=query['target'].numpy();valid=query['valid'].numpy();lo,hi=np.percentile(target[valid],[2,98]);axes[row,0].set_ylabel(f'{label}\n{rid}, q{q}\nDepth {lo:.2f}–{hi:.2f} m',fontsize=8);axes[row,1].imshow(np.ma.masked_where(~valid,target),vmin=lo,vmax=hi,cmap='viridis')
        for col,method in enumerate(methods,2):
            if method=='Native':record=d.read(baseline.OUT/'baselines'/f'{rid}_{q}.json')['methods']['Native'];pred=np.load(record['path'],allow_pickle=False)
            else:
                payload=d.load(d.CACHE/'predictions'/f'{method}_{d.SEEDS[0]}'/f'{rid}.pt');index=[tuple(k) for k in payload['keys']].index((rid,q,'missing'));pred=payload['depth_metric'][index].numpy()
            axes[row,col].imshow(pred,vmin=lo,vmax=hi,cmap='viridis')
            if method=='J_G':res=np.log(np.maximum(pred,1e-6)/np.maximum(target,1e-6));res[~valid]=np.nan;image=axes[row,7].imshow(res,vmin=-.7,vmax=.7,cmap='coolwarm')
        for col,ax in enumerate(axes[row]):
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(['Missing RGB','Target',*methods,'JG log residual'][col],fontsize=9)
    fig.colorbar(image,ax=axes[:,7].tolist(),shrink=.6,label='log(prediction/target)');fig.suptitle('Explicit examples and counterexamples · common per-row depth range · seed20261110',fontsize=12);save(fig,'cases')
    print(dict(figures_created=['outcomes','cases'],actual_view_required=True),flush=True)
if __name__=='__main__':main()
