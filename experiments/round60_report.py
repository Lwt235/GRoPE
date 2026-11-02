"""Postprocess frozen R60 outcomes; scientific and visual review remain explicit."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round60_common import OUT,ROLES,SEEDS,PHASES,CONDITIONS,checked,read,write,sha
from experiments.round59_report import LABELS

PRIMARY='invariant_mlp'
CONTROLS=('distance_mlp','descriptor_mlp','angle_descriptor_mlp','matrix_mlp')
KEYS=('region_recall','center_pck','soft_epe','ce')


def main():
    checked();queue=read(OUT/'queue_status.json');summary=read(OUT/'metrics/summary.json')
    if queue['state']!='complete' or queue['active'] or not queue['all_children_exit_zero']:
        raise ValueError('original controller must finish')
    if summary['groups']!=120 or len(read(OUT/'selected.json')['runs'])!=30:
        raise ValueError('incomplete R60')
    dest=OUT/'report1'
    if dest.exists():raise ValueError('preserve generated report')
    dest.mkdir()
    def row(phase,condition,role,seed):
        return next(x for x in summary['rows'] if (x['phase'],x['condition'],x['role'],x['seed'])==
                    (phase,condition,role,seed))
    def values(phase,condition,role,key,kind='macro'):
        return np.array([row(phase,condition,role,s)[kind][key] for s in SEEDS])
    lines=['# R60 real-scene correspondence results','',
        '30 exact source models;30 NEW target adaptations;120 prediction groups. Same architecture and old scene pairs.',
        'Both source transfer and fixed target adaptation were planned before target outcomes. All target selections precede all old_eval exports.',
        'Native sensor depth is observed INPUT;relative pose is supervision/scoring only. No RGB-only,optimized-3DGS or full-registration claim.',
        'Metrics use pair-normalized scene units,not metres or pixels. Primary aggregate is scene-macro region recall.',
        'Old development scenes and reused source models are not fresh confirmation. The full research goal is unachieved.',
        'Scientific interpretation and visual review are pending when this report is generated.','']
    comparisons=[]
    for phase in PHASES:
        for condition in CONDITIONS:
            sample=row(phase,condition,PRIMARY,SEEDS[0]);scenes=sorted(sample['by_scene'])
            if sample['macro'] is None:raise ValueError('no evaluable scenes; inspect coverage rather than invent a score')
            lines += ['## '+phase+' / '+condition,'',
                f'Evaluable scenes {sample["evaluable_scenes"]}/{sample["total_scenes"]}; below uses scene macro and three-seed means.',
                '', '| Method | Region R@1 % | Center PCK % | Soft EPE | CE | Query-weighted R@1 % |',
                '| --- | ---: | ---: | ---: | ---: | ---: |']
            for role in (PRIMARY,)+tuple(r for r in ROLES if r!=PRIMARY):
                v=[values(phase,condition,role,k).mean() for k in KEYS]
                cells=[f'{x*100:.3f}' if i<2 else f'{x:.6f}' for i,x in enumerate(v)]
                lines.append('| '+LABELS[role]+' | '+' | '.join(cells)+
                    f' | {values(phase,condition,role,"region_recall","micro").mean()*100:.3f} |')
            lines += ['','| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp | Region scene means pp |',
                      '| --- | ---: | --- | ---: | --- |']
            for control in ROLES:
                if control==PRIMARY:continue
                rec=dict(phase=phase,condition=condition,control=control,metrics={},scene_order=scenes)
                for key in KEYS:
                    paired=values(phase,condition,PRIMARY,key)-values(phase,condition,control,key)
                    scene_pairs=np.array([[row(phase,condition,PRIMARY,s)['by_scene'][c][key]-
                        row(phase,condition,control,s)['by_scene'][c][key] for c in scenes] for s in SEEDS])
                    positive=paired>0 if key in ('region_recall','center_pck') else paired<0
                    scene_positive=scene_pairs.mean(0)>0 if key in ('region_recall','center_pck') else scene_pairs.mean(0)<0
                    rec['metrics'][key]=dict(mean=float(paired.mean()),paired_seeds=paired.tolist(),
                        improving_seeds=int(positive.sum()),scene_mean=scene_pairs.mean(0).tolist(),
                        improving_scenes=int(scene_positive.sum()),seed_by_scene=scene_pairs.tolist())
                comparisons.append(rec);r=rec['metrics']['region_recall'];c=rec['metrics']['center_pck']
                fmt=lambda a:'/'.join(f'{x*100:+.3f}' for x in a)
                lines.append(f'| {LABELS[control]} | {r["mean"]*100:+.3f} | {fmt(r["paired_seeds"])} | {c["mean"]*100:+.3f} | {fmt(r["scene_mean"])} |')
            lines += ['','Scene order: '+', '.join(scenes)+'.',
                      'Seed and scene differences are descriptive;nearby frames are correlated. No query-level confidence interval is presented.','',
                      '| Scene | Pairs | Valid queries | Eligible fraction |', '| --- | ---: | ---: | ---: |']
            cache=read(OUT/'data'/('eval_'+condition+'.json'))
            with np.load(OUT/'data'/('eval_'+condition+'.npz'),allow_pickle=False) as f:scene_ids=f['scenes']
            for scene in scenes:
                s=sample['by_scene'][scene]
                lines.append(f'| {scene} | {int((scene_ids==scene).sum())} | {sum(s["valid_counts"])} | {s["eligible_fraction"]:.6f} |')
            lines += ['',f'Excluded by native input availability: {len(cache["excluded"])}; zero-valid pairs: {cache["zero_valid_pair_count"]}.',
                'Sparse supports and eligible populations differ from base;compare matched methods within each condition.','']
    lines += ['## Target adaptation and checks','',
        'Each role starts from its OWN selected source checkpoint. Actual target initial backbone weights are not shared across roles.',
        'Same target data/order/budget/optimizer/selection;this compares complete source-plus-target pipelines,not an isolated reset of target PE.',
        '', '| Method | PE parameters | Selected target epochs | Target updates | Target fit seconds |',
        '| --- | ---: | --- | --- | ---: |']
    total=0.
    for role in ROLES:
        runs=[read(OUT/'runs'/role/str(s)/'run.json') for s in SEEDS];seconds=sum(x['seconds'] for x in runs);total+=seconds
        lines.append(f'| {LABELS[role]} | {runs[0]["pe_parameters"]} | '+
                     '/'.join(str(x['selected_epoch']) for x in runs)+' | '+
                     '/'.join(str(x['steps']) for x in runs)+f' | {seconds:.2f} |')
    lines += ['',f'NEW target fitting/selection cumulative {total:.2f}s. Shared-host timing is not a controlled speed benchmark.',
        f'Independent NumPy metric maximum difference {summary["max_metric_error"]:.3g}. Separate arithmetic by the same author,not another laboratory.',
        'R59 sparse uncertainty and older classification/scalar-form failures remain evidence. No global goal or fresh-confirmation claim.',
        '',f'![Region comparisons]({dest}/comparison.png)','',f'![Target validation curves]({dest}/learning_curves.png)','']
    (dest/'report.md').write_text('\n'.join(lines))
    write(dest/'comparisons.json',dict(rows=comparisons,uncertainty='paired seed and scene descriptions;no query IID interval',goal_complete=False))
    colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,phase in enumerate(PHASES):
        for j,condition in enumerate(CONDITIONS):
            ax=axes[i,j]
            for k,control in enumerate(CONTROLS):
                diff=100*(values(phase,condition,PRIMARY,'region_recall')-values(phase,condition,control,'region_recall'))
                ax.scatter(diff,k+np.array([-.1,0,.1]),color=colors[k],s=30)
                ax.scatter(diff.mean(),k,marker='|',s=180,color='black')
            ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(4),[LABELS[x] for x in CONTROLS])
            ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(phase+' / '+condition)
            ax.set_xlabel('Candidate minus control: scene-macro R@1 (pp)')
    fig.savefig(dest/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for i,role in enumerate((PRIMARY,)+CONTROLS+('nope',)):
        histories=[read(OUT/'runs'/role/str(s)/'history.json')['validation'] for s in SEEDS]
        epochs=[x['epoch'] for x in histories[0]]
        for ax,key,scale in zip(axes,('selection_ce','region_recall'),(1,100)):
            y=np.array([[x[key]*scale for x in h] for h in histories])
            ax.plot(epochs,y.mean(0),color=colors[i],label=LABELS[role])
            ax.fill_between(epochs,y.mean(0)-y.std(0),y.mean(0)+y.std(0),color=colors[i],alpha=.12)
    for ax in axes:ax.set_xlabel('Target epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set_ylabel('Validation soft CE');axes[1].set_ylabel('Validation region recall (%)')
    fig.savefig(dest/'learning_curves.png',dpi=160);plt.close(fig)
    write(dest/'generated.json',dict(generated_unix=time.time(),scientific_review_pending=True,visual_review_pending=True,
        goal_complete=False,report_source_sha256=sha(__file__),
        hashes={n:sha(dest/n) for n in ('report.md','comparisons.json','comparison.png','learning_curves.png')}))
    print('R60 report generated;scientific/visual review remain required',flush=True)


if __name__=='__main__':main()
