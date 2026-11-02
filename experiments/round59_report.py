"""Postprocessing of the frozen R59 comparison; scientific review stays explicit."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round59_common import (OUT, ROLES, NEW_ROLES, SEEDS, CONDITIONS, DIAGNOSTIC,
    PRIMARY, checked, read, write, sha, run_path)

LABELS = dict(gaussian='R58 scalar Gaussian', gaussian_distance='R58 Gaussian metric',
    distance='R58 sinusoidal distance', distance_angle='R58 distance + angle',
    invariant_mlp='Gaussian invariant MLP (candidate)', nope='No explicit PE',
    distance_mlp='Distance MLP', descriptor_mlp='Local descriptor MLP',
    angle_descriptor_mlp='Angle + descriptor MLP', matrix_mlp='Full log-matrix MLP')


def main():
    checked()
    queue = read(OUT/'queue_status.json')
    if queue['state'] != 'complete' or queue['active'] or not queue['all_children_exit_zero']:
        raise ValueError('original controller must complete')
    summary = read(OUT/'metrics/summary.json')
    selected = read(OUT/'selected.json')
    if summary['groups'] != 120 or summary['diagnostic_groups'] != 30 or len(selected['runs']) != 30:
        raise ValueError('incomplete comparison')
    destination = OUT/'report1'
    if destination.exists():
        raise ValueError('preserve existing report')
    destination.mkdir()
    def values(role, condition, key):
        source = summary['diagnostics'] if condition == DIAGNOSTIC else summary['rows']
        return np.array([next(x[key] for x in source if (x['role'], x['condition'], x['seed']) ==
                        (role, condition, seed)) for seed in SEEDS])
    lines = ['# R59 learned-relation attribution results', '',
        '12 new fits;18 EXACT R58 references;30 selected models;120 main condition groups and30 base-reframe diagnostics.',
        'The candidate was chosen after R58 development results. Reference fits/predictions are reused, not new independent evidence.',
        'All data remain previously used official-training ModelNet development data. Full research goal is unachieved.',
        'Scientific interpretation and visual review remain pending when this report is generated.', '']
    comparisons = []
    for condition in CONDITIONS:
        lines += ['## '+condition, '', '| Method | Region R@1 % | Center PCK % | Soft EPE | CE |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for role in (PRIMARY,)+NEW_ROLES+tuple(x for x in ROLES if x != PRIMARY and x not in NEW_ROLES):
            cells = [values(role, condition, k).mean() for k in ('region_recall', 'center_pck', 'soft_epe', 'ce')]
            lines.append('| '+LABELS[role]+' | '+' | '.join(f'{v*100:.3f}' if i < 2 else f'{v:.6f}' for i, v in enumerate(cells))+' |')
        with np.load(OUT/'metrics'/PRIMARY/str(SEEDS[0])/(condition+'_objects.npz'), allow_pickle=False) as f:
            counts = f['valid_count'].astype(np.float64)
        rng = np.random.default_rng(590800+CONDITIONS.index(condition))
        weights = rng.multinomial(len(counts), np.full(len(counts), 1/len(counts)), size=3000).astype(np.float64)
        denominator = weights@counts
        lines += ['', '| Candidate minus control | Region paired pp | Mean region pp [object-conditional95%] | Center paired pp | Soft EPE paired |',
                  '| --- | --- | --- | --- | --- |']
        for control in NEW_ROLES:
            comparison = dict(condition=condition, control=control, metrics={})
            for key in ('region_recall', 'center_pck', 'soft_epe', 'ce'):
                delta = values(PRIMARY, condition, key)-values(control, condition, key)
                objects = []
                for seed in SEEDS:
                    with np.load(OUT/'metrics'/PRIMARY/str(seed)/(condition+'_objects.npz'), allow_pickle=False) as a, \
                         np.load(OUT/'metrics'/control/str(seed)/(condition+'_objects.npz'), allow_pickle=False) as b:
                        if not np.array_equal(a['ids'], b['ids']) or not np.array_equal(a['valid_count'], b['valid_count']):
                            raise ValueError('unpaired object statistics')
                        objects.append(a[key]-b[key])
                boot = weights@np.stack(objects).mean(0)/denominator
                comparison['metrics'][key] = dict(mean=float(delta.mean()), paired=delta.tolist(),
                    improvements=int((delta > 0).sum()) if key in ('region_recall', 'center_pck') else int((delta < 0).sum()),
                    object_bootstrap_conditional_95=np.quantile(boot, [.025, .975]).tolist())
            comparisons.append(comparison)
            m = comparison['metrics'];region = m['region_recall'];lo, hi = np.array(region['object_bootstrap_conditional_95'])*100
            paired = lambda key, scale: '/'.join(f'{x*scale:+.4f}' for x in m[key]['paired'])
            lines.append('| '+LABELS[control]+' | '+paired('region_recall',100)+
                f' | {region["mean"]*100:+.4f} [{lo:+.4f},{hi:+.4f}] | '+paired('center_pck',100)+' | '+paired('soft_epe',1)+' |')
        sample = next(x for x in summary['rows'] if x['condition'] == condition)
        lines += ['', f'Eligible counts {sample["valid_counts"]}; fraction {sample["eligible_fraction"]:.6f}. '+
                  'Compare methods within the same condition; population changes across sampling conditions.', '']
    lines += ['## Fixed frame-change diagnostic', '',
        'Same physical base pairs and labels. Rotate view0 Z90,view1 X90,translate both(.7,-.3,.2),transform covariance;keep invariant local features.',
        'All30 models were actually evaluated. Matrix MLP is not invariant by construction; its sensitivity is model behavior,not a failed implementation.', '',
        '| Method | Region change pp | Center change pp | CE change | Max row-probability change | Changed A-to-B winners (all queries,3seeds) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for role in ROLES:
        delta = [float((values(role, DIAGNOSTIC, k)-values(role, 'base', k)).mean()) for k in ('region_recall','center_pck','ce')]
        lines.append(f'| {LABELS[role]} | {100*delta[0]:+.4f} | {100*delta[1]:+.4f} | {delta[2]:+.6f} | '+
                     f'{values(role, DIAGNOSTIC,"max_probability_difference").max():.6g} | {int(values(role,DIAGNOSTIC,"winner_changes").sum())} |')
    lines += ['', 'The probability/winner diagnostic is A-to-B and includes ineligible queries; task metrics remain bidirectional on eligible queries.',
        'One fixed reframing does not estimate an entire distribution of pose robustness. Mathematical invariance and measured task utility are distinct.', '',
        '## Fitting and checks', '', '| Method | PE parameters | Selected epochs | Fit seconds | Reused fit |',
        '| --- | ---: | --- | ---: | --- |']
    new_seconds = 0.
    for role in ROLES:
        runs = [read(run_path(role,s)/'run.json') for s in SEEDS]
        cost = sum(x['seconds'] for x in runs)
        reused = role not in NEW_ROLES
        if not reused:
            new_seconds += cost
        lines.append(f'| {LABELS[role]} | {runs[0]["pe_parameters"]} | '+
            '/'.join(str(x['selected_epoch']) for x in runs)+f' | {cost:.2f} | {reused} |')
    lines += ['', f'NEW fitting/selection cumulative time {new_seconds:.2f}s. Shared-host timing is not a controlled speed benchmark.',
        f'Independent metric maximum error {summary["max_metric_error"]:.3g}. Source/cache/model/prediction identities and actual sampling/init pairing checked.',
        'Independent arithmetic is a separate NumPy path by the same author,not a second-lab replication.',
        'Intervals are object bootstraps conditional on these three fitted models,not training-seed uncertainty or fresh confirmation.', '',
        '## Decision pending scientific review', '',
        'Evaluate directional-shape attribution against ALL fixed controls and conditions. Do not reinterpret R58 scalar-Gaussian failure as a success.',
        'The complete goal still requires appropriate cross-task,real-scene,generalization and untouched confirmation evidence.', '',
        f'![Comparisons]({destination}/comparison.png)', '', f'![Learning curves]({destination}/learning_curves.png)', '']
    (destination/'report.md').write_text('\n'.join(lines))
    write(destination/'comparisons.json', dict(rows=comparisons, bootstrap_replicates=3000,
        interpretation='conditional object uncertainty for three models;reference candidate is reused', goal_complete=False))
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, key, title, scale in zip(axes,('region_recall','center_pck','soft_epe'),
            ('Region recall difference (pp)','Center PCK difference (pp)','Soft EPE difference'),(100,100,1)):
        for j, control in enumerate(NEW_ROLES):
            delta = (values(PRIMARY,'base',key)-values(control,'base',key))*scale
            axis.scatter(delta,np.full(3,j)+np.array([-.1,0,.1]),color=colors[j],s=30)
            axis.scatter(delta.mean(),j,color='black',marker='|',s=200)
        axis.axvline(0,color='gray',lw=1);axis.grid(axis='x',alpha=.2)
        axis.set_yticks(range(4),[LABELS[x] for x in NEW_ROLES]);axis.invert_yaxis()
        axis.set_title(title);axis.set_xlabel('Gaussian invariant MLP minus control')
    fig.savefig(destination/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for j,role in enumerate((PRIMARY,)+NEW_ROLES):
        curves=[read(run_path(role,s)/'history.json')['validation'] for s in SEEDS]
        epochs=[x['epoch'] for x in curves[0]]
        for axis,key,scale in zip(axes,('selection_ce','region_recall'),(1,100)):
            y=np.array([[x[key]*scale for x in c] for c in curves])
            axis.plot(epochs,y.mean(0),color=colors[j],label=LABELS[role])
            axis.fill_between(epochs,y.mean(0)-y.std(0),y.mean(0)+y.std(0),color=colors[j],alpha=.12)
    for axis in axes:
        axis.set_xlabel('Epoch');axis.grid(alpha=.2);axis.legend(fontsize=8)
    axes[0].set_ylabel('Validation soft CE');axes[1].set_ylabel('Validation region recall (%)')
    fig.savefig(destination/'learning_curves.png',dpi=160);plt.close(fig)
    write(destination/'generated.json',dict(generated_unix=time.time(),scientific_review_pending=True,
        visual_review_pending=True,goal_complete=False,report_source_sha256=sha(__file__),
        hashes={name:sha(destination/name) for name in ('report.md','comparisons.json','comparison.png','learning_curves.png')}))
    print('R59 report generated; scientific/visual review still required.',flush=True)


if __name__=='__main__':
    main()
