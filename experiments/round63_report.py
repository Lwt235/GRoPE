"""Postprocessing for the fixed R63 diagnostic; no fitting or source changes."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round63_diagnostic import OUT,SEEDS,METRICS,ROLES,cases,checked,read,write,sha

LABELS=dict(baseline='Predicted Gaussian baseline',observed_center='Replace center',
    observed_shape='Replace shape',observed_both='Replace both',initial_shape='Initial fixed shape',
    sphere='Fixed sphere',zero_relation='Zero added relation')
LOWER=('ce','epe_argmax','epe_soft','tail16','soft_tail16')


def main():
    checked();q=read(OUT/'queue_status.json')
    if q['state']!='complete' or q['active'] or not all(r['exit_code']==0 for r in q['completed']):
        raise ValueError('finish original diagnostic queue')
    result=read(OUT/'analysis/summary.json')
    if len(result['records'])!=54:raise ValueError('group scope')
    dest=OUT/'report1'
    if dest.exists():raise ValueError('preserve original report')
    dest.mkdir()
    lookup={(r['role'],r['seed'],r['split'],r['case']):r for r in result['records']}
    aggregate={};paired={}
    for split in ('fit','val'):
        aggregate[split]={};paired[split]={}
        for role in ROLES:
            for case in cases(role):
                aggregate[split][role+':'+case]={m:float(np.mean([lookup[role,s,split,case]['metrics'][m] for s in SEEDS])) for m in METRICS}
        for case in cases('predicted_full')[1:]:
            paired[split][case]={}
            for metric in METRICS:
                values=[lookup['predicted_full',s,split,case]['metrics'][metric]-lookup['predicted_full',s,split,'baseline']['metrics'][metric] for s in SEEDS]
                scenes=sorted(lookup['predicted_full',SEEDS[0],split,case]['by_scene'])
                differences={name:float(np.mean([lookup['predicted_full',s,split,case]['by_scene'][name][metric]-lookup['predicted_full',s,split,'baseline']['by_scene'][name][metric] for s in SEEDS])) for name in scenes}
                favorable=lambda x:x<0 if metric in LOWER else x>0
                paired[split][case][metric]=dict(mean=float(np.mean(values)),paired_seeds=values,
                    favorable_seeds=sum(favorable(v) for v in values),by_scene=differences,
                    favorable_scenes=sum(favorable(v) for v in differences.values()))
    geometry={}
    for split in ('fit','val'):
        geometry[split]={}
        for role in ('predicted_full','predicted_point'):
            geometry[split][role]={}
            for layer in range(2):
                rows=[next(g for g in result['geometry'] if (g['role'],g['seed'],g['split'])==(role,s,split))['layers'][layer]['strata'] for s in SEEDS]
                entry={}
                for key,stratum,metric,stat in (
                    ('center_median_m','native_center','center_distance_m','median'),
                    ('center_p90_m','native_center','center_distance_m','p90'),
                    ('depth_signed_mean_m','native_center','depth_signed_error_m','mean'),
                    ('predicted_depth_median_m','native_center','predicted_depth_m','median'),
                    ('observed_depth_median_m','native_center','observed_depth_m','median'),
                    ('shape_log_median','patch16','covariance_log_error','median')):
                    entry[key]=[r[stratum][metric][stat] for r in rows]
                entry['exposure_counts']={name:rows[0][name]['center_distance_m']['n'] for name in rows[0]}
                entry['diagonal_saturation_fraction']=[r['patch16']['diagonal_saturation_fraction'] for r in rows]
                entry['lower_saturation_fraction']=[r['patch16']['lower_saturation_fraction'] for r in rows]
                geometry[split][role][str(layer)]=entry
    write(dest/'comparisons.json',dict(aggregate=aggregate,paired=paired,geometry=geometry,
        observed_range=result['observed_range'],source_sha256=sha(__file__),
        distribution_shift_diagnostic=True,training_updates=0,fresh_confirmation=False,goal_complete=False))
    lines=['# R63: fixed-weight RGB geometry diagnostic','',
        '48 old fit pairs / 64 old val pairs;3seeds;54prediction groups;no new fitting.',
        'All original branches replay R62 selected weights. No old_eval or confirmation inputs.',
        'Observed substitutions add privileged information and cause distribution shifts.',
        'These are development diagnostics,not fair trained RGB competitors or causal proof.',
        'Scientific interpretation and actual image review pending at generation.','']
    for split in ('fit','val'):
        lines+=['## '+split,'','| Model / intervention | Soft EPE px | PCK8 % | CE |',
            '| --- | ---: | ---: | ---: |']
        for role in ROLES:
            for case in cases(role):
                label=LABELS[case] if role=='predicted_full' else ('Predicted center baseline' if role=='predicted_point' else 'Observed full baseline (privileged)')
                r=aggregate[split][role+':'+case]
                lines.append(f"| {label} | {r['epe_soft']:.5f} | {100*r['pck8']:.3f} | {r['ce']:.5f} |")
        lines+=['','Paired intervention minus unmodified predicted-full;negative EPE is favorable.','',
            '| Intervention | EPE mean px | Seeds11/22/33 | Favorable seeds/scenes | PCK8 delta pp |',
            '| --- | ---: | --- | --- | ---: |']
        for case,values in paired[split].items():
            r=values['epe_soft'];v='/'.join(f'{x:+.5f}' for x in r['paired_seeds'])
            lines.append(f"| {LABELS[case]} | {r['mean']:+.5f} | {v} | {r['favorable_seeds']}/3;{r['favorable_scenes']}/{len(r['by_scene'])} | {100*values['pck8']['mean']:+.3f} |")
        lines.append('')
    lines+=['## Geometry against sensor-derived support','',
        'Entries below average the three seed-specific medians;not a pooled median.',
        'Center errors use known native centers;shape errors use patches with>=16/64samples.',
        'Sensor support is not a ground-truth learned GS parameterization.','',
        '| Split | Model | Layer | Center median m | Shape log median | Predicted depth median m | Observed depth median m |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for split in geometry:
        for role in geometry[split]:
            for layer,r in geometry[split][role].items():
                values=[np.mean(r[k]) for k in ('center_median_m','shape_log_median','predicted_depth_median_m','observed_depth_median_m')]
                lines.append(f"| {split} | {role} | {int(layer)+1} | "+' | '.join(f'{v:.5f}' for v in values)+' |')
    lines+=['','## Observed covariance range','',
        'Cholesky coordinate clipping is not an optimal covariance projection. Tiny near-floor changes are not a material bottleneck.','',
        '| Split / stratum | Token exposures | Any coordinate outside % | Relative Frobenius median / p90 | Log change median |',
        '| --- | ---: | ---: | --- | ---: |']
    for split,groups in result['observed_range'].items():
        for stratum,r in groups.items():
            n=r['any_coordinate_outside']['n']
            if not n:continue
            f=r['relative_frobenius']
            lines.append(f"| {split}/{stratum} | {n} | {100*r['any_coordinate_outside']['mean']:.3f} | {f['median']:.6g} / {f['p90']:.6g} | {r['log_spectrum_change']['median']:.6g} |")
    lines+=['','Raw coordinate statistics,coverage strata,saturation and scene details are retained in analysis/summary.json.',
        f"Independent metric max difference:{result['max_metric_error']:.3g}.",
        f"Original val logit max difference:{max(r['max_logit_difference'] for r in result['baseline_val_replay']):.3g};CE max difference:{max(r['ce_error'] for r in result['baseline_val_replay']):.3g}.",
        'All model weights unchanged. No global goal completion.','',
        f'![Interventions]({dest}/interventions.png)','',f'![Geometry]({dest}/geometry.png)']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    cs=list(paired['fit']);fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for ax,split in zip(axes,('fit','val')):
        for y,case in enumerate(cs):
            r=paired[split][case]['epe_soft']
            ax.scatter(r['paired_seeds'],[y-.13,y,y+.13],s=28)
            ax.plot([r['mean']]*2,[y-.23,y+.23],color='black',lw=2)
        ax.axvline(0,color='grey',lw=1);ax.set_yticks(range(len(cs)),[LABELS[c] for c in cs]);ax.invert_yaxis()
        ax.set_xlabel('EPE intervention minus baseline (px);negative better');ax.set_title(split+';fixed weights;three paired seeds');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(dest/'interventions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));settings=[('fit','0'),('fit','1'),('val','0'),('val','1')]
    for ax,key,title in [(axes[0],'center_median_m','Known-center distance median (m)'),(axes[1],'shape_log_median','Patch support log-shape error median')]:
        for role,offset,color in [('predicted_full',-.13,'tab:blue'),('predicted_point',.13,'tab:orange')]:
            for i,(split,layer) in enumerate(settings):
                values=geometry[split][role][layer][key]
                ax.scatter([i+offset]*3,values,color=color,label=role if i==0 else None)
        ax.set_xticks(range(4),['fit L1','fit L2','val L1','val L2']);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'geometry.png',dpi=160);plt.close(fig)
    files=('comparisons.json','report.md','interventions.png','geometry.png')
    write(dest/'generated.json',dict(generated_unix=time.time(),source_sha256=sha(__file__),
        hashes={name:sha(dest/name) for name in files},scientific_review_pending=True,
        visual_review_pending=True,goal_complete=False))
    print('R63 REPORT GENERATED;interpretation and actual visual review pending',flush=True)


if __name__=='__main__':main()
