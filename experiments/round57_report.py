"""Postprocessing only: paired development results, all conditions and diagnostics.

Not imported by the frozen experiment. Does not fit/select models or change any
scientific source. Run only after the existing controller has completed.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round57_common import (OUT,NEW_ROLES,ROLES,SEEDS,PRIMARY,VARIANTS,
    read,write,checked,sha,model_path)

LABELS=dict(joint_conditional='RoPE + full Gaussian',joint_fixed='RoPE + fixed distance',
    joint_free_quadratic='RoPE + free quadratic',conditional='Full Gaussian alone',
    diagonal='Diagonal Gaussian alone',isotropic='Isotropic Gaussian alone',
    static='Static Gaussian alone',fixed='Fixed distance alone',free_quadratic='Free quadratic alone',
    rope='Center RoPE',fourier='Fourier + RoPE',mlp='MLP + RoPE',nope='No explicit PE')


LABELS.update(pair_full='RoPE + full Gaussian-pair',pair_distance='RoPE + adaptive pair distance',
    pair_center='RoPE + learned-center distance')

def report(out=OUT):
    checked(out)
    status=read(out/'queue_status.json')
    if status['state']!='complete' or status['active'] or not status['all_children_exit_zero']:
        raise ValueError('wait for terminal successful experiment; do not infer completion')
    summary=read(out/'metrics/summary.json');verification=read(out/'metrics/verification.json')
    if summary['total_models']!=48 or len(verification['metric_replays'])!=336 or len(summary['diagnostics'])!=3:
        raise ValueError('incomplete reporting population')
    dest=out/'report1'
    if dest.exists():raise ValueError('preserve previous report; version any reporting changes')
    dest.mkdir();table=summary['by_role_seed'];means=summary['means']
    costs={role:[read(model_path(role,s,out)/'run.json') for s in SEEDS] for role in ROLES}
    new_seconds=sum(r['seconds'] for role in NEW_ROLES for r in costs[role])
    comparison={r['control']:r for r in summary['comparisons']}
    fig,(left,right)=plt.subplots(1,2,figsize=(13,9),gridspec_kw={'width_ratios':[1.05,1]})
    y=np.arange(len(ROLES))
    for i,role in enumerate(ROLES):
        values=np.asarray([100*table[role][str(s)]['base']['oa'] for s in SEEDS])
        left.scatter(values,np.full(3,i),color='#a3a3a3',s=20,zorder=2)
        left.plot([values.min(),values.max()],[i,i],color='#b8b8b8',lw=1)
        left.scatter(values.mean(),i,color='#b33b24' if role==PRIMARY else '#235a82',s=48,marker='D',zorder=3)
    left.set_yticks(y,labels=[LABELS[r] for r in ROLES]);left.invert_yaxis()
    left.set_xlabel('Development OA (%)');left.set_title('Fixed budget, same three seeds')
    controls=[r for r in ROLES if r!=PRIMARY]
    for i,role in enumerate(controls):
        values=np.asarray(comparison[role]['paired_seed_oa_pp'])
        right.scatter(values,np.full(3,i),color='#999999',s=20,zorder=2)
        right.scatter(values.mean(),i,color='#b33b24',marker='D',s=48,zorder=3)
    right.axvline(0,color='black',lw=.8)
    right.set_yticks(np.arange(len(controls)),labels=[LABELS[r] for r in controls]);right.invert_yaxis()
    right.set_xlabel('Full Gaussian-pair minus control (percentage points)');right.set_title('Paired differences; positive favors Gaussian-pair')
    for ax in (left,right):ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('R57: Gaussian-pair positional relation — development only',fontsize=14)
    fig.text(.5,.015,'Small gray dots: individual seeds 11/22/33. Diamonds: means. R56/R48 controls reused; no confidence intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96));fig.savefig(dest/'comparison.png',dpi=170);fig.savefig(dest/'comparison.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    focus=NEW_ROLES+('joint_conditional','joint_free_quadratic','mlp')
    for role in focus:
        histories=[read(model_path(role,s,out)/'history.json') for s in SEEDS]
        validation=np.asarray([[r['ce'] for r in h['validation']] for h in histories])
        training=np.asarray([[r['loss'] for r in h['training']] for h in histories])
        axes[0].plot(np.arange(0,121,5),validation.mean(0),label=LABELS[role])
        axes[1].plot(np.arange(1,121),training.mean(0),label=LABELS[role])
    axes[0].set_title('Mean validation CE (selection metric)');axes[1].set_title('Mean training CE (label smoothing 0.1)')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'learning_curves.png',dpi=170);plt.close(fig)

    lines=['# R57: native center RoPE and Gaussian-pair similarity — development results','',
        'Nine new fits and39 reused R56/R48 references, seeds11/22/33. Same data/backbone120epoch budget.',
        'Official test and matching confirmation scenes remain untouched. No generality/novelty claim.',
        '','## Base utility','',
        '| Role | OA % | mCA % | CE | PE parameters |', '| --- | ---: | ---: | ---: | ---: |']
    for role in ROLES:
        m=means[role]['base']
        lines.append(f"| {LABELS[role]} | {100*m['oa']:.3f} | {100*m['mca']:.3f} | {m['ce']:.6f} | {costs[role][0]['positional_parameters']} |")
    lines+=['','## Paired primary differences','',
        '| Control | OA delta pp | Seed11/22/33 deltas pp | Wins / ties | mCA delta pp | CE delta |',
        '| --- | ---: | --- | --- | ---: | ---: |']
    for c in summary['comparisons']:
        values=', '.join(f'{v:+.3f}' for v in c['paired_seed_oa_pp'])
        lines.append(f"| {LABELS[c['control']]} | {c['oa_difference_pp']:+.3f} | {values} | {c['winning_seeds']}/3 ; {c['tied_seeds']} ties | {c['mca_difference_pp']:+.3f} | {c['ce_difference']:+.6f} |")
    lines+=['','## All fixed conditions (absolute OA %)','',
        '| Role | '+' | '.join(VARIANTS)+' |','| --- | '+' | '.join(['---:']*len(VARIANTS))+' |']
    for role in ROLES:
        lines.append('| '+LABELS[role]+' | '+' | '.join(f"{100*means[role][v]['oa']:.3f}" for v in VARIANTS)+' |')
    lines+=['','## Fixed-checkpoint shape interventions','',
        'Remove only the covariance-compatibility term at fixed checkpoints. Earlier layer interventions can alter later features and predicted centers.',
        'This does not isolate a physical covariance causal effect; compare with the separately trained equal-capacity distance control.',
        'Distribution shift also changes predictions; sensitivity alone does not establish useful physical geometry.',
        '','| Role / intervention | OA delta pp by seed | Mean OA delta pp | Mean CE delta |',
        '| --- | --- | ---: | ---: |']
    for role in ('pair_full',):
        for mode in ('shape_off',):
            rows=[r for r in summary['diagnostics'] if r['role']==role and r['mode']==mode]
            values=', '.join(f"{r['oa_difference_pp']:+.3f}" for r in rows)
            lines.append(f"| {LABELS[role]} / {mode} | {values} | {np.mean([r['oa_difference_pp'] for r in rows]):+.3f} | {np.mean([r['ce_difference'] for r in rows]):+.6f} |")
    errors=verification['metric_replays'];seal=read(out/'audit_training/completion.json')
    lines+=['','## Cost and bounded checks','',
        f'New training/selection cumulative seconds: {new_seconds:.2f}. Includes checkpoint overhead; shared-host cost is not a formal throughput benchmark.',
        f"48 independent formula samples: maxP error {max(r['numeric']['p_max_error'] for r in seal['records']):.8g}; winner changes {sum(r['numeric']['winner_changes'] for r in seal['records'])}; warnings {len(seal['warnings'])}.",
        f"336 saved-prediction metric replays: maximum absolute correct-count difference {max(abs(r['correct_difference']) for r in errors)}; maximum CE difference {max(r['ce_error'] for r in errors):.8g}.",
        'Same-author independent arithmetic, not an independent laboratory or full-network rewrite.',
        'The primary comparison needs substantive interpretation; there is no automatic1pp promotion threshold.',
        '','![Base comparisons]('+str(dest/'comparison.png')+')','',
        '![Learning curves]('+str(dest/'learning_curves.png')+')','']
    (dest/'report.md').write_text('\n'.join(lines))
    write(dest/'provenance.json',dict(summary_sha256=sha(out/'metrics/summary.json'),
        verification_sha256=sha(out/'metrics/verification.json'),reporter_sha256=sha(__file__),
        new_training_seconds=new_seconds,scientific_source_changed=False,
        full_goal_achieved=False,images_visually_inspected=False))
    print('REPORT',dest,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    report(parser.parse_args().out)
