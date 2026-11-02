"""Postprocess R61; generation is separate from scientific/visual review."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round61_common import OUT,DOMAINS,PHASES,CONDITIONS,ROLES,NEW_ROLES,SEEDS,checked,read,write,sha,run_path

PRIMARY='invariant_mlp'
KEYS=('region_recall','center_pck','soft_epe','ce')
LABELS=dict(invariant_mlp='Full Gaussian relations (candidate)',normal_ppf_mlp='Normal PPF + spectra',
    axial_mlp='Axial relations + original spectra',distance_mlp='Distance MLP',
    descriptor_mlp='Local descriptor MLP',angle_descriptor_mlp='Angle + descriptor MLP',matrix_mlp='Full log-matrix MLP')
PLOT_CONTROLS=('normal_ppf_mlp','axial_mlp','distance_mlp')


def main():
    checked();queue=read(OUT/'queue_status.json');summary=read(OUT/'metrics/summary.json')
    if queue['state']!='complete' or queue['active'] or not queue['all_children_exit_zero']:
        raise ValueError('wait for original controller')
    if summary['groups']!=168 or summary['new_groups']!=48 or len(read(OUT/'selected.json')['runs'])!=12:
        raise ValueError('incomplete experiment')
    dest=OUT/'report1'
    if dest.exists():raise ValueError('preserve generated report')
    dest.mkdir()
    def row(phase,condition,role,seed):
        return next(r for r in summary['rows'] if (r['phase'],r['condition'],r['role'],r['seed'])==(phase,condition,role,seed))
    def values(phase,condition,role,key):
        return np.array([row(phase,condition,role,s)['primary_metrics'][key] for s in SEEDS])
    lines=['# R61: normal/spectrum attribution','',
        '6 NEW synthetic fits + 6 NEW real adaptations. 15 source and 15 target reference models reused exactly.',
        '48 NEW independently recomputed prediction groups;120 old groups reused by identity. Total168.',
        'Both new controls use the same covariance inputs,backbone,actual data stream and fixed optimization as references.',
        'Target roles start from their OWN source models;this is a complete source-plus-target pipeline comparison.',
        'Synthetic primary: original query-weighted region recall. Real primary: scene-macro region recall.',
        'Development data only. Sparse populations change. No fresh confirmation or general-effectiveness claim.',
        'Normal PPF control is not a complete RoITr reproduction. Scientific/visual review pending at generation.','']
    comparisons=[]
    for phase in PHASES:
        domain='synthetic' if phase=='synthetic' else 'real'
        for condition in CONDITIONS[domain]:
            sample=row(phase,condition,PRIMARY,11)
            lines+=['## '+phase+' / '+condition,'',
                '| Method | Region R@1 % | Center PCK % | Soft EPE | CE |',
                '| --- | ---: | ---: | ---: | ---: |']
            for role in (PRIMARY,)+NEW_ROLES+tuple(r for r in ROLES if r not in (PRIMARY,)+NEW_ROLES):
                numbers=[values(phase,condition,role,k).mean() for k in KEYS]
                cells=[f'{v*100:.3f}' if i<2 else f'{v:.6f}' for i,v in enumerate(numbers)]
                lines.append('| '+LABELS[role]+' | '+' | '.join(cells)+' |')
            lines+=['','| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp |',
                    '| --- | ---: | --- | ---: |']
            for control in ROLES:
                if control==PRIMARY:continue
                rec=dict(phase=phase,condition=condition,control=control,metrics={})
                for key in KEYS:
                    delta=values(phase,condition,PRIMARY,key)-values(phase,condition,control,key)
                    better=delta>0 if key in ('region_recall','center_pck') else delta<0
                    metric=dict(mean=float(delta.mean()),paired_seeds=delta.tolist(),improving_seeds=int(better.sum()))
                    if domain=='real':
                        scenes=sorted(sample['by_scene']);rec['scene_order']=scenes
                        by_scene=np.array([[row(phase,condition,PRIMARY,s)['by_scene'][c][key]-
                            row(phase,condition,control,s)['by_scene'][c][key] for c in scenes] for s in SEEDS])
                        avg=by_scene.mean(0);better_scene=avg>0 if key in ('region_recall','center_pck') else avg<0
                        metric.update(seed_by_scene=by_scene.tolist(),scene_mean=avg.tolist(),improving_scenes=int(better_scene.sum()))
                    rec['metrics'][key]=metric
                comparisons.append(rec);r=rec['metrics']['region_recall'];c=rec['metrics']['center_pck']
                pairs='/'.join(f'{v*100:+.3f}' for v in r['paired_seeds'])
                lines.append(f'| {LABELS[control]} | {r["mean"]*100:+.3f} | {pairs} | {c["mean"]*100:+.3f} |')
            lines+=['',f'Eligible query fraction: {sample["micro"]["eligible_fraction"]:.6f}; valid counts: {sample["micro"]["valid_counts"]}.']
            if domain=='real':
                lines+=[f'Evaluable scenes: {sample["evaluable_scenes"]}/{sample["total_scenes"]};scene-order/differences are retained in comparisons.json.']
            counts=summary['multiplicities'][domain+'/'+condition]
            lines+=['Minimum-eigenspace multiplicities k1/k2/k3: '+str(counts['counts_k1_k2_k3'])+
                    f'; repeated fraction {counts["repeated_fraction"]:.6g}.','']
    lines+=['## New fitting and checks','',
        '| Domain | Role | PE parameters | Selected epochs | Updates per seed | Cumulative fit seconds |',
        '| --- | --- | ---: | --- | --- | ---: |']
    total=0.
    for domain in DOMAINS:
        for role in NEW_ROLES:
            runs=[read(run_path(domain,role,s)/'run.json') for s in SEEDS]
            seconds=sum(x['seconds'] for x in runs);total+=seconds
            lines.append(f'| {domain} | {LABELS[role]} | {runs[0]["pe_parameters"]} | '+
                '/'.join(str(x['selected_epoch']) for x in runs)+' | '+
                '/'.join(str(x['steps']) for x in runs)+f' | {seconds:.2f} |')
    lines+=['',f'NEW fitting/selection cumulative {total:.2f}s;shared-host timing,not a controlled speed benchmark.',
        f'Independent new metric maximum difference {summary["max_new_metric_error"]:.3g};same-author separate arithmetic.',
        'Old scalar/classification/sparse/source-transfer negatives remain. No global goal completion.',
        '',f'![Paired region differences]({dest}/comparison.png)','',f'![Validation curves]({dest}/learning_curves.png)','']
    (dest/'report.md').write_text('\n'.join(lines))
    write(dest/'comparisons.json',dict(rows=comparisons,goal_complete=False,uncertainty='descriptive paired seeds/scenes;no query IID confidence interval'))
    colors=plt.get_cmap('tab10').colors
    layouts=[('synthetic',c) for c in CONDITIONS['synthetic']]+[(p,c) for p in ('real_source','real_adapted') for c in CONDITIONS['real']]
    fig,axes=plt.subplots(2,4,figsize=(15,8),constrained_layout=True)
    for i,(phase,condition) in enumerate(layouts):
        ax=axes.flat[i]
        for j,control in enumerate(PLOT_CONTROLS):
            delta=100*(values(phase,condition,PRIMARY,'region_recall')-values(phase,condition,control,'region_recall'))
            ax.scatter(delta,j+np.array([-.1,0,.1]),color=colors[j],s=24)
            ax.scatter(delta.mean(),j,marker='|',s=150,color='black')
        ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(3),['Normal PPF','Axial relations','Distance MLP'] if i%4==0 else ['','',''])
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(phase+'\n'+condition,fontsize=10)
        ax.set_xlabel('Candidate minus control (pp)',fontsize=9)
    fig.savefig(dest/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,domain in enumerate(DOMAINS):
        for i,role in enumerate((PRIMARY,)+NEW_ROLES+('distance_mlp',)):
            histories=[read(run_path(domain,role,s)/'history.json')['validation'] for s in SEEDS]
            epochs=[x['epoch'] for x in histories[0]]
            for j,(key,scale) in enumerate((('selection_ce',1),('region_recall',100))):
                y=np.array([[x[key]*scale for x in h] for h in histories]);ax=axes[j,col]
                ax.plot(epochs,y.mean(0),label=LABELS[role],color=colors[i]);ax.fill_between(epochs,y.mean(0)-y.std(0),y.mean(0)+y.std(0),alpha=.12,color=colors[i])
                ax.set_xlabel('Epoch');ax.grid(alpha=.2)
                ax.set_ylabel('Validation CE' if j==0 else 'Validation region R@1 (%)')
                ax.set_title(domain+'; mean +/- seed SD (not CI)');ax.legend(fontsize=7)
    fig.savefig(dest/'learning_curves.png',dpi=160);plt.close(fig)
    write(dest/'generated.json',dict(generated_unix=time.time(),scientific_review_pending=True,visual_review_pending=True,
        source_sha256=sha(__file__),hashes={n:sha(dest/n) for n in ('report.md','comparisons.json','comparison.png','learning_curves.png')},goal_complete=False))
    print('R61 report generated; scientific/visual review pending',flush=True)


if __name__=='__main__':main()
