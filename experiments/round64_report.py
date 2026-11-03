"""Postprocessing only; the frozen experiment and predictions are unchanged."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round64_common import (OUT,ROLES,NEW_ROLES,REFERENCE_ROLES,SEEDS,METRICS,
    checked,read,write,sha,run_path)

LABELS=dict(camera_only='PRoPE reference',camera_plane='Calibrated plane',ray_point='Ray point',
    ray_affine='Ray affine',camera_mlp='Camera + MLP phase',free_relation='Direct learned relation',
    predicted_full='RGB Gaussian relations',predicted_point='RGB center only',
    depth_surface='Depth surface Gaussian',depth_footprint='Depth footprint Gaussian',
    depth_local_mlp='Same-information local MLP')
RGB_ROLES=tuple(r for r in ROLES if not r.startswith('observed_'))
LOWER=('ce','epe_argmax','epe_soft','tail16','soft_tail16')


def main():
    checked();queue=read(OUT/'queue_status.json')
    if queue['state']!='complete' or not all(r['exit_code']==0 for r in queue['completed']):
        raise ValueError('finish original queue before report')
    result=read(OUT/'metrics/summary.json');records=result['records'];selection=read(OUT/'selected.json')
    if len(records)!=66 or len(selection['runs'])!=9:raise ValueError('report scope')
    dest=OUT/'report1'
    if dest.exists():raise ValueError('preserve original report generation')
    dest.mkdir()
    lookup={(r['role'],r['seed'],r['variant']):r for r in records}
    aggregate={role:{variant:{metric:float(np.mean([lookup[role,s,variant]['metrics'][metric] for s in SEEDS]))
        for metric in METRICS} for variant in ('baseline','rgb_permuted')} for role in ROLES}
    pairs={}
    contrasts=[('depth_surface',r) for r in ROLES if r!='depth_surface']+[
        ('depth_footprint',r) for r in ('depth_local_mlp','camera_only','camera_plane','ray_point','ray_affine','predicted_point')]
    for candidate,control in contrasts:
        item={}
        for metric in METRICS:
            diffs=[lookup[candidate,s,'baseline']['metrics'][metric]-lookup[control,s,'baseline']['metrics'][metric] for s in SEEDS]
            scenes=sorted(lookup[candidate,SEEDS[0],'baseline']['by_scene'])
            scene_diffs={scene:float(np.mean([lookup[candidate,s,'baseline']['by_scene'][scene][metric]-
                lookup[control,s,'baseline']['by_scene'][scene][metric] for s in SEEDS])) for scene in scenes}
            better=lambda x:x<0 if metric in LOWER else x>0
            item[metric]=dict(mean=float(np.mean(diffs)),paired_seeds=diffs,
                favorable_seeds=sum(better(x) for x in diffs),by_scene=scene_diffs,
                favorable_scene_means=sum(better(x) for x in scene_diffs.values()))
        pairs[candidate+'__'+control]=item
    strata={}
    for role in ROLES:
        strata[role]={}
        for gap in (15,60):
            per_seed=[]
            for seed in SEEDS:
                rows=[r for r in lookup[role,seed,'baseline']['rows'] if r['frame_gap']==gap]
                scenes=[]
                for name in sorted({r['scene'] for r in rows}):
                    rr=[r for r in rows if r['scene']==name];denom=sum(r['n_valid'] for r in rr)
                    if denom:scenes.append({m:sum(r[m] for r in rr)/denom for m in METRICS})
                per_seed.append({m:float(np.mean([r[m] for r in scenes])) for m in METRICS})
            strata[role][str(gap)]={m:float(np.mean([r[m] for r in per_seed])) for m in METRICS}
    runs=[read(run_path(role,seed)/'run.json') for role in NEW_ROLES for seed in SEEDS]
    write(dest/'comparisons.json',dict(aggregate=aggregate,paired=pairs,frame_gap=strata,
        new_fit_seconds=sum(r['training_selection_seconds'] for r in runs),
        old_references_are_reused=True,source_sha256=sha(__file__),fresh_confirmation=False,goal_complete=False))
    lines=['# R64: depth-field Gaussian relations in RGB matching','',
        '9 new fits;24 exact R55/R62 RGB reference models.18 new+48 reused predictions;66 independent metric recomputations.',
        'RGB features remain frozen;parameter head,PE and matching network are trained jointly.',
        'All nine new selections precede new evaluation. Old development data;no fresh confirmation.',
        'Primary:depth_surface. Depth_footprint isolates the derivative term;depth_local_mlp receives the same local field information.',
        'Gaussian coupling and reachable range both change from R62;no uniquely isolated degrees-of-freedom claim.',
        'Scientific interpretation and actual image review pending at generation.','']
    for variant in ('baseline','rgb_permuted'):
        lines+=['## '+variant,'','| Method | Soft EPE px | PCK8 % | CE | Soft >16px % |','| --- | ---: | ---: | ---: | ---: |']
        for role in RGB_ROLES:
            r=aggregate[role][variant]
            lines.append(f"| {LABELS[role]} | {r['epe_soft']:.5f} | {100*r['pck8']:.3f} | {r['ce']:.5f} | {100*r['soft_tail16']:.3f} |")
        lines.append('')
    lines+=['## Fixed paired contrasts','','EPE deltas are candidate minus control;negative is favorable.','',
        'All listed methods use RGB/cameras only; reused references do not count as independent new evidence.','',
        '| Comparison | EPE mean px | EPE seeds11/22/33 | Favorable seeds/scenes | PCK8 mean pp |',
        '| --- | ---: | --- | --- | ---: |']
    for c,b in contrasts:
        x=pairs[c+'__'+b];e=x['epe_soft']
        signs='/'.join(f'{v:+.5f}' for v in e['paired_seeds'])
        lines.append(f"| {LABELS[c]} minus {LABELS[b]} | {e['mean']:+.5f} | {signs} | {e['favorable_seeds']}/3;{e['favorable_scene_means']}/5 | {100*x['pck8']['mean']:+.3f} |")
    lines+=['','## Frame gap','','| Method | 15frame EPE | 60frame EPE |','| --- | ---: | ---: |']
    for r in ROLES:lines.append(f"| {LABELS[r]} | {strata[r]['15']['epe_soft']:.5f} | {strata[r]['60']['epe_soft']:.5f} |")
    lines+=['','Frame spacing is not an isolated zoom/surface-tilt intervention.','']
    lines+=['','## New fits','','| Method | PE params | Selected steps | Fit/selection cumulative seconds |',
        '| --- | ---: | --- | ---: |']
    for role in NEW_ROLES:
        rr=[r for r in runs if r['role']==role]
        lines.append(f"| {LABELS[role]} | {rr[0]['pe_parameters']} | {'/'.join(str(r['best_step']) for r in rr)} | {sum(r['training_selection_seconds'] for r in rr):.2f} |")
    lines+=['',f"Independent metric max difference:{result['max_metric_error']:.3g}.",
        f"Selection CE max difference:{max(r['validation_ce_error'] for r in selection['runs']):.3g};CPU replay max:{max(r['replay_logit_error'] for r in selection['runs']):.3g}.",
        'Shared-host timing is not a controlled speed benchmark. No global goal completion.','',
        f'![Paired EPE]({dest}/comparison.png)','',f'![Validation curves]({dest}/learning_curves.png)']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(12,7));cs=[(c,b) for c,b in contrasts if c=='depth_surface']
    for y,(c,b) in enumerate(cs):
        v=pairs[c+'__'+b]['epe_soft'];ax.scatter(v['paired_seeds'],[y-.12,y,y+.12],s=35)
        ax.plot([v['mean'],v['mean']],[y-.22,y+.22],color='black',lw=2)
    ax.axvline(0,color='grey',lw=1);ax.set_yticks(range(len(cs)),[LABELS[b] for _,b in cs]);ax.invert_yaxis()
    ax.set_xlabel('Depth surface Gaussian minus control;continuous EPE (px);negative is better')
    ax.set_title('Paired seeds;black ticks are means;old development evaluation')
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,roles,title in [(axes[0],('depth_surface','depth_footprint','depth_local_mlp'),'New matched roles'),
                           (axes[1],('depth_surface','predicted_full','predicted_point','camera_only','ray_affine'),'Existing strong references')]:
        for role in roles:
            hh=[read(run_path(role,s)/'history.json') for s in SEEDS]
            values=np.array([[r['val_ce'] for r in h] for h in hh]);steps=[r['step'] for r in hh[0]]
            mean,sd=values.mean(0),values.std(0,ddof=1)
            ax.plot(steps,mean,label=LABELS[role]);ax.fill_between(steps,mean-sd,mean+sd,alpha=.12)
        ax.set_title(title+';mean +/- seed SD');ax.set_xlabel('Step');ax.set_ylabel('Validation CE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'learning_curves.png',dpi=160);plt.close(fig)
    files=('report.md','comparisons.json','comparison.png','learning_curves.png')
    write(dest/'generated.json',dict(generated_unix=time.time(),source_sha256=sha(__file__),
        hashes={n:sha(dest/n) for n in files},scientific_review_pending=True,visual_review_pending=True,goal_complete=False))
    print('R64 REPORT GENERATED;scientific/visual review pending',flush=True)


if __name__=='__main__':main()
