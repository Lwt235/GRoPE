"""Finite report figures from already exported coordinates and score tables."""
import json,time,resource,collections
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import native_source_resolved_data as d

def training_figure(out):
    colors=['#4c78a8','#72b7b2','#e45756','#b279a2','#f2a541'];fig,ax=plt.subplots(figsize=(9,4.5));total=0
    for p in sorted((d.OUT/'calls').glob('fits*')):
        updates=[v for l in p.read_text().splitlines() if (v:=json.loads(l))['kind']=='formal_update']
        if not updates:continue
        total+=len(updates);kind,seed=updates[0]['arm'],updates[0]['seed'];x=np.array([v['step'] for v in updates]);y=np.array([v['loss'] for v in updates]);bins=np.arange(0,len(y),25);ax.plot(x[bins],np.add.reduceat(y,bins)/np.minimum(25,len(y)-bins),c=colors[list(d.KINDS).index(kind)],alpha=.5,label=f'{kind}/{seed}')
    assert total==15000
    ax.set(xlabel='Formal update',ylabel='SmoothL1 training loss (25-update means)',title='All paired fits; last-step evaluation only');ax.legend(ncol=4,fontsize=7);fig.tight_layout();fig.savefig(out/'training.png',dpi=180);plt.close(fig)

def main():
    start=time.monotonic();torch.set_num_threads(2);out=d.OUT/'figures';out.mkdir(exist_ok=True)
    summary=d.read(d.OUT/'scoring/summary.json');places=d.read(d.OUT/'scoring/place.json');coverage=d.read(d.OUT/'scoring/coverage.json');counter=d.read(d.OUT/'scoring/all_pairwise_counterexamples.json')
    score_index={(r['kind'],r['seed'],r['id'],r['anchor'],r['query']):r for r in d.read(d.OUT/'scoring/viewpairs.json')}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,ax=plt.subplots(1,3,figsize=(14,4.5));kinds=[*d.KINDS,'P','L','Native2','Native4'];colors=list(plt.get_cmap('tab10').colors)
    for i,kind in enumerate(kinds):
        r=summary['absolute']['dev12/original'][kind]['PCK8'];lo,hi=r['interval95'];ax[0].errorbar(i,r['mean'],yerr=[[r['mean']-lo],[hi-r['mean']]],fmt='o',capsize=4,color=colors[i])
    ax[0].tick_params(axis='x',rotation=60);ax[0].set(xticks=range(len(kinds)),xticklabels=kinds,ylabel='PCK@8 (%)',title='Used development: original queries')
    contrasts=['F_G-F_P','F_P-M_P','M_G-M_P','Interaction','F_G-F_S','F_G-P','F_G-L','F_G-Native4'];values=summary['comparisons_pp']['dev12/original']
    values=dict(values,Interaction=summary['representation_PE_interaction_pp']['dev12/original'])
    for i,key in enumerate(contrasts):
        r=values[key];lo,hi=r['interval95'];ax[1].errorbar(r['mean'],i,xerr=[[r['mean']-lo],[hi-r['mean']]],fmt='o',capsize=4,color='#e45756' if key=='F_G-F_P' else '#4c78a8')
    ax[1].axvline(0,c='gray',lw=1);ax[1].axvline(1,c='gray',lw=1,ls=':');ax[1].set(yticks=range(len(contrasts)),yticklabels=contrasts,xlabel='Paired PCK difference (percentage points)',title='95% paired place / seed intervals');ax[1].invert_yaxis()
    ids=[m['id'] for m in d.plan()['catalog']]
    for i,rid in enumerate(ids):
        for j,seed in enumerate(d.SEEDS):
            a=next(r['PCK8'] for r in places if r['id']==rid and r['seed']==seed and r['kind']=='F_G' and r['query_set']=='original');b=next(r['PCK8'] for r in places if r['id']==rid and r['seed']==seed and r['kind']=='F_P' and r['query_set']=='original')
            if a is not None and b is not None:ax[2].scatter(i+(j-1)*.17,(a-b)*100,s=18,c=['#4c78a8','#72b7b2','#e45756'][j],label=str(seed) if i==0 else None)
    ax[2].axhline(0,c='gray',lw=1);ax[2].set(xticks=range(12),xticklabels=ids,ylabel='F_G-F_P (percentage points)',title='Every development place and seed');ax[2].tick_params(axis='x',rotation=90);ax[2].legend(title='Training seed',fontsize=8,loc='upper right')
    fig.tight_layout();fig.savefig(out/'primary.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(13,5));stage=summary['coverage']['rejection_stage_totals'];ax[0].barh(list(stage),list(stage.values()),color='#4c78a8');ax[0].invert_yaxis();ax[0].set(xlabel='Points rejected at this ordered stage',title='Fixed sensor / pose visibility labels')
    allids=[m['id'] for m in d.metas()];v=[[r['valid_GT'] for r in coverage if r['id']==rid] for rid in allids];matrix=np.asarray(v);im=ax[1].imshow(matrix,aspect='auto',vmin=0,vmax=64,cmap='viridis');ax[1].set(yticks=np.arange(0,44,3),yticklabels=allids[::3],xlabel='Four anchors × eight fixed queries',title='All 44 places, including zero-label view pairs');fig.colorbar(im,ax=ax[1],label='Valid points / 64');fig.tight_layout()
    if not (out/'coverage.png').exists():fig.savefig(out/'coverage.png')
    plt.close(fig)
    cases=[dict(id=ids[0],anchor=0,query=2,seed=d.SEEDS[0],reason='Fixed first development case')]
    for A,B in [('F_G','F_P'),('F_P','M_P'),('F_G','L')]:
        r=next(r for r in counter if r['A']==A and r['B']==B and r['cohort']=='dev12' and r['query_set']=='original');cases.append(dict(id=r['id'],anchor=r['anchor'],query=r['query'],seed=r['seed'],reason=f'Largest development {A}-{B} pair deficit: {r["PCK_difference_pp"]:.1f} pp'))
    fig,ax=plt.subplots(len(cases),6,figsize=(18,13));selections=[]
    def image(x):
        v=x.detach().numpy().transpose(1,2,0);return np.clip(v/255 if v.max()>1.5 else v,0,1)
    for i,case in enumerate(cases):
        rid,a,q,seed=case['id'],case['anchor'],case['query'],case['seed'];obj=d.load(d.case(rid,a,q));valid=obj['labels']['valid'].numpy();gt=obj['labels']['target'].numpy();points=d.point_grid().numpy();sp=d.load(d.source(rid));qp=d.load(d.old.query_path(rid,q));ax[i,0].imshow(image(sp['rgb'][d.SOURCES.index(a)]));ax[i,0].scatter(points[:,0],points[:,1],c=np.where(valid,'#00d4ff','#bbbbbb'),s=12);ax[i,0].set_title(f'{rid} anchor{a}: {valid.sum()}/64 labelled')
        for j,kind in enumerate(('F_P','M_P','F_G','L','Native4'),1):
            pred=(obj['Native4']['coords'][-1] if kind=='Native4' else d.load((d.prior.CACHE if kind=='L' else d.CACHE)/'predictions'/f'{kind}_{seed}'/rid/f'{a}_{q}.pt')['coords'][-1]).numpy();err=np.linalg.norm(pred-gt,axis=-1);ax[i,j].imshow(image(qp['rgb']));ax[i,j].scatter(pred[~valid,0],pred[~valid,1],c='#bbbbbb',s=8,alpha=.45)
            for pp,gg,e in zip(pred[valid],gt[valid],err[valid]):ax[i,j].plot([pp[0],gg[0]],[pp[1],gg[1]],c='#00c878' if e<=8 else '#ff4055',lw=.8,alpha=.8)
            ax[i,j].scatter(gt[valid,0],gt[valid,1],s=10,c='#00d4ff',marker='+');ax[i,j].scatter(pred[valid,0],pred[valid,1],s=9,c=np.where(err[valid]<=8,'#00c878','#ff4055'));score=score_index[kind,None if kind=='Native4' else seed,rid,a,q]['PCK8'];score_text=f'{score*100:.1f}%' if score is not None else 'undefined (no GT)';ax[i,j].set_title(f'{kind}, query{q}: PCK {score_text}')
        for aa in ax[i]:aa.set(xlim=(-.5,447.5),ylim=(447.5,-.5),xticks=[],yticks=[])
        ax[i,0].set_ylabel(case['reason'],fontsize=8);selections.append(case)
    fig.suptitle('Sensor / pose labels in cyan; prediction-to-label error green ≤8px, red >8px; grey unlabelled predictions',fontsize=10);fig.tight_layout();fig.savefig(out/'tracking_cases.png');plt.close(fig)
    if not (out/'training.png').exists():training_figure(out)
    d.write(d.OUT/'figure_manifest.json',dict(complete=True,selection=selections,files=[str(p) for p in out.glob('*.png')],actual_viewing_record_required=True,no_new_predictions_or_scoring=True));d.write(d.OUT/'CPU_figure_runtime.json',dict(complete=True,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0));print('figure exports complete',flush=True)
if __name__=='__main__':main()
