"""Postprocessing only: fixed predictions, paired summaries and descriptive plots.

Not part of the frozen optimizer/model. Never declares the research goal complete.
"""
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round58_common import OUT, ROLES, SEEDS, CONDITIONS, checked, read, write, sha

LABELS = dict(gaussian='Gaussian relation', gaussian_distance='Gaussian metric only',
              distance='Euclidean distance', distance_angle='Distance + angle',
              invariant_mlp='Invariant MLP', nope='No explicit PE')


def main():
    checked()
    queue = read(OUT/'queue_status.json')
    if queue['state'] != 'complete' or queue['active'] or not queue['all_children_exit_zero']:
        raise ValueError('original controller must be terminal and successful')
    summary = read(OUT/'metrics/summary.json')
    selected = read(OUT/'selected.json')
    if summary['groups'] != 72 or len(selected['runs']) != 18:
        raise ValueError('incomplete experiment')
    destination = OUT/'report1'
    if destination.exists():
        raise ValueError('preserve existing report')
    destination.mkdir()
    rows = summary['rows']
    def values(role, condition, key):
        return np.asarray([next(x[key] for x in rows if x['role'] == role and
                               x['condition'] == condition and x['seed'] == seed) for seed in SEEDS])
    lines = ['# R58 observed support relations: development results', '',
             'All 18 fits and 72 condition groups completed. Scientific interpretation and visual review are still required.',
             'This is an old-ModelNet local-correspondence experiment, not fresh confirmation, optimized 3DGS, or full registration.',
             'R57 remains not promoted. The complete research goal remains unachieved.', '']
    comparisons = []
    for condition in CONDITIONS:
        lines += ['## '+condition, '', '| Method | Region R@1 % | Center PCK % | Selected overlap | Hard EPE | Soft EPE | CE |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for role in ROLES:
            means = [values(role, condition, k).mean() for k in ('region_recall', 'center_pck', 'overlap', 'hard_epe', 'soft_epe', 'ce')]
            lines.append('| '+LABELS[role]+' | '+' | '.join(f'{v*100:.3f}' if i < 2 else f'{v:.6f}' for i, v in enumerate(means))+' |')
        # Bootstrap objects conditional on these three fits. Keep seed signs separate.
        with np.load(OUT/'metrics'/'gaussian'/str(SEEDS[0])/(condition+'_objects.npz'), allow_pickle=False) as f:
            counts = f['valid_count'].astype(np.float64)
        rng = np.random.default_rng(580800+CONDITIONS.index(condition))
        weights = rng.multinomial(len(counts), np.full(len(counts), 1/len(counts)), size=3000).astype(np.float64)
        denominator = weights@counts
        for control in ROLES[1:]:
            comparison = dict(condition=condition, control=control, metrics={})
            for key in ('region_recall', 'center_pck', 'soft_epe', 'ce'):
                delta = values('gaussian', condition, key)-values(control, condition, key)
                objects = []
                for seed in SEEDS:
                    with np.load(OUT/'metrics'/'gaussian'/str(seed)/(condition+'_objects.npz'), allow_pickle=False) as a, \
                         np.load(OUT/'metrics'/control/str(seed)/(condition+'_objects.npz'), allow_pickle=False) as b:
                        if not np.array_equal(a['ids'], b['ids']) or not np.array_equal(a['valid_count'], b['valid_count']):
                            raise ValueError('unpaired object metric records')
                        objects.append(a[key]-b[key])
                object_delta = np.stack(objects).mean(0)
                boot = (weights@object_delta)/denominator
                comparison['metrics'][key] = dict(mean=float(delta.mean()), paired=delta.tolist(),
                    improvements=int((delta > 0).sum()) if key in ('region_recall', 'center_pck') else int((delta < 0).sum()),
                    object_bootstrap_conditional_95=np.quantile(boot, [.025, .975]).tolist())
            comparisons.append(comparison)
        lines += ['', 'Primary minus controls (paired seeds11/22/33; pp for recall/PCK):', '',
                  '| Control | Region differences pp | Center differences pp | Soft EPE differences |',
                  '| --- | --- | --- | --- |']
        for comp in comparisons[-5:]:
            cells = []
            for key in ('region_recall', 'center_pck', 'soft_epe'):
                scale = 100 if key != 'soft_epe' else 1
                cells.append('/'.join(f'{v*scale:+.4f}' for v in comp['metrics'][key]['paired']))
            lines.append('| '+LABELS[comp['control']]+' | '+' | '.join(cells)+' |')
        sample = next(x for x in rows if x['condition'] == condition)
        lines += ['', 'Eligible query counts: '+str(sample['valid_counts'])+
                  f'; fraction {sample["eligible_fraction"]:.6f}. Nearest-center oracle PCK {100*sample["oracle_center_pck"]:.3f}%; '
                  f'best overlap mean {sample["oracle_overlap"]:.6f}.',
                  'The saved best-overlap center oracle uses the first maximizing index; ties can affect it. '+
                  'The separate validation tie diagnostic measures this explicitly.', '']
    lines += ['## Optimization and checks', '', '| Method | PE parameters | Selected epochs | Cumulative fit seconds |',
              '| --- | ---: | --- | ---: |']
    seconds = 0.
    for role in ROLES:
        runs = [read(OUT/'runs'/role/str(s)/'run.json') for s in SEEDS]
        cost = sum(x['seconds'] for x in runs)
        seconds += cost
        lines.append(f'| {LABELS[role]} | {runs[0]["pe_parameters"]} | '+
                     '/'.join(str(x['selected_epoch']) for x in runs)+f' | {cost:.2f} |')
    lines += ['', f'Total fitting/selection time: {seconds:.2f}s. Shared-host timings are not a controlled speed benchmark.',
              f'Max independent metric error: {summary["max_metric_error"]:.3g}. '+
              f'Actual-input direct Gaussian scalar error: {selected["actual_gaussian_scalar_error"]:.3g}.',
              'Validation selection, source/cache identity, shared initialization and actual batch order were checked.',
              'Independent arithmetic here means a separate NumPy path by the same author, not independent laboratory replication.',
              '', 'Object-bootstrap intervals in comparisons.json are conditional on the three fitted models. '+
              'They do not estimate training-seed uncertainty or make reused objects fresh confirmation. '+
              'No multiplicity-adjusted hypothesis-test claim is made.', '']
    diagnostic_path = OUT/'diagnostics/repeatability/summary.json'
    if diagnostic_path.exists():
        diag = read(diagnostic_path)
        lines += ['## Input repeatability diagnostic', '',
                  f'On old validation data, {diag["selected_pairs"]} fixed sampled relations were compared. '+
                  f'{diag["empty_directions"]} of {diag["object_directions"]} directions had no eligible relation.', '',
                  '| Relation | Mean symmetric relative change | Mean sine-input RMSE |',
                  '| --- | ---: | ---: |']
        for role in ('distance', 'gaussian_distance', 'gaussian'):
            stat = diag['statistics'][role]
            lines.append(f'| {LABELS[role]} | {stat["relative"]["mean"]:.6f} | {stat["embedding_rmse"]["mean"]:.6f} |')
        lines += ['', 'This measures combined sampling/center/support sensitivity. It does not isolate covariance estimation '+
                  'error, predict task performance, or create an additional acceptance gate. The sine RMSE also depends '+
                  'on the fixed argument scales. The full Gaussian relation is slightly less variable than its metric-only '+
                  'control under relative error; the trace term is not uniquely implicated.',
                  f'[Diagnostic plan](/home/liwt/GRoPE/docs/round58_development_diagnostics.md); '+
                  f'[interpretation]({OUT}/diagnostics/repeatability/interpretation.md).', '']
    lines += [
              '## Interpretation still required', '',
              'Assess all paired strong-control effects, region/center agreement and robustness before any next decision. '+
              'A local pass cannot establish general effectiveness or erase previous classification failures.', '',
              f'![Paired comparisons]({destination}/comparison.png)', '',
              f'![Learning curves]({destination}/learning_curves.png)', '']
    (destination/'report.md').write_text('\n'.join(lines))
    write(destination/'comparisons.json', dict(rows=comparisons, bootstrap_replicates=3000,
          interpretation='object uncertainty conditional on three fitted models; not seed uncertainty', goal_complete=False))
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    controls = ROLES[1:]
    for axis, key, title, scale in zip(axes, ('region_recall', 'center_pck', 'soft_epe'),
            ('Region recall difference (pp)', 'Center PCK difference (pp)', 'Soft EPE difference'), (100, 100, 1)):
        for j, control in enumerate(controls):
            diff = (values('gaussian', 'base', key)-values(control, 'base', key))*scale
            axis.scatter(diff, np.full(3, j)+np.array([-.10, 0, .10]), s=30, color=colors[j], alpha=.8)
            axis.scatter(diff.mean(), j, marker='|', s=200, color='black')
        axis.axvline(0, color='gray', lw=1)
        axis.set_yticks(range(5), [LABELS[x] for x in controls])
        axis.invert_yaxis()
        axis.set_xlabel('Full Gaussian relation minus control')
        axis.set_title(title)
        axis.grid(axis='x', alpha=.2)
    fig.savefig(destination/'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for j, role in enumerate(ROLES):
        curves = [read(OUT/'runs'/role/str(s)/'history.json')['validation'] for s in SEEDS]
        epochs = [x['epoch'] for x in curves[0]]
        for axis, key, scale in zip(axes, ('selection_ce', 'region_recall'), (1, 100)):
            y = np.asarray([[x[key]*scale for x in c] for c in curves])
            axis.plot(epochs, y.mean(0), label=LABELS[role], color=colors[j])
            axis.fill_between(epochs, y.mean(0)-y.std(0), y.mean(0)+y.std(0), color=colors[j], alpha=.12)
    axes[0].set_ylabel('Validation soft CE')
    axes[1].set_ylabel('Validation region recall (%)')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(destination/'learning_curves.png', dpi=160)
    plt.close(fig)
    write(destination/'generated.json', dict(generated_unix=time.time(), scientific_review_pending=True,
          visual_review_pending=True, goal_complete=False, report_source_sha256=sha(__file__),
          hashes={name: sha(destination/name) for name in ('report.md', 'comparisons.json', 'comparison.png', 'learning_curves.png')}))
    print('R58 report generated; actual scientific and visual review remains.', flush=True)


if __name__ == '__main__':
    main()
