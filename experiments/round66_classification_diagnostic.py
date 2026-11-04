"""Output decomposition and matched-evaluation diagnostics; no model fitting."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from experiments import round65_run as runner
from experiments.round65_common import ROOT, OUT as OLD, SEEDS, VARIANTS, read, write, sha
from experiments.round65_data import load
from experiments.round66_logshape import DEST as NEW, specification
from models.round65_actual_gs import NativeGSClassifier
from models.round66_logshape import LogShapeClassifier

DEST = ROOT/'results/round66_logshape/classification_diagnostic1'
ROLES = ('distance_mlp', 'log_shape_distance')
SOURCE = dict(zip(ROLES, (OLD, NEW)))
PLAN = ROOT/'docs/round66_classification_diagnostic_plan.md'
CODE = Path(__file__).resolve()


def observations(logits, labels):
    x, y = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    if x.shape != (len(y), 60) or not np.isfinite(x).all():
        raise ValueError('finite full60-class logits required')
    if not np.array_equal(np.unique(y), np.arange(60)):
        raise ValueError('all60 categories required')
    shifted = x-x.max(1, keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(1, keepdims=True))
    ce = -logp[np.arange(len(y)), y]
    correct = x.argmax(1) == y
    others = x.copy(); others[np.arange(len(y)), y] = -np.inf
    margin = x[np.arange(len(y)), y]-others.max(1)
    counts = np.bincount(y, minlength=60)
    weight = 1/(60*counts[y])
    return dict(ce=ce, correct=correct, true_probability=np.exp(-ce), margin=margin,
                weight=weight, labels=y)


def summarize(o):
    y = o['labels']
    classes = [dict(label=c, count=int((y == c).sum()),
                    ce=float(o['ce'][y == c].mean()),
                    accuracy=float(o['correct'][y == c].mean())) for c in range(60)]
    domains = {}
    for name, lo in (('MACGS', 0), ('ModelNet', 30)):
        mask = (y >= lo) & (y < lo+30)
        domains[name] = dict(count=int(mask.sum()), ce=float(o['ce'][mask].mean()),
            macro_ce=float(np.mean([v['ce'] for v in classes[lo:lo+30]])),
            oa=float(o['correct'][mask].mean()),
            mca=float(np.mean([v['accuracy'] for v in classes[lo:lo+30]])))
    return dict(ce=float(o['ce'].mean()), macro_ce=float(o['ce']@o['weight']),
        oa=float(o['correct'].mean()), mca=float(o['correct']@o['weight']),
        count=len(y), classes=classes, domains=domains)


def compare(a, b, labels):
    """Candidate a minus control b; signed contributions over full denominators."""
    g, d = observations(a, labels), observations(b, labels)
    sg, sd = summarize(g), summarize(d)
    delta = g['ce']-d['ce']
    change = g['correct'].astype(float)-d['correct'].astype(float)
    masks = dict(both_correct=g['correct'] & d['correct'],
        both_wrong=~g['correct'] & ~d['correct'],
        corrected=g['correct'] & ~d['correct'], harmed=~g['correct'] & d['correct'])
    groups = {}
    for name, mask in masks.items():
        row = dict(count=int(mask.sum()), ce_contribution=float(delta[mask].sum()/len(labels)),
            macro_ce_contribution=float(delta[mask]@g['weight'][mask]),
            oa_contribution_pp=float(100*change[mask].sum()/len(labels)),
            mca_contribution_pp=float(100*change[mask]@g['weight'][mask]),
            per_class_count=np.bincount(np.asarray(labels)[mask], minlength=60).tolist())
        for role, obj in (('candidate', g), ('control', d)):
            row[role] = {key: float(obj[key][mask].mean()) if mask.any() else None
                         for key in ('ce', 'true_probability', 'margin')}
        groups[name] = row
    gains = {key: sg[key]-sd[key] for key in ('ce', 'macro_ce', 'oa', 'mca')}
    for key in gains:
        suffix = key+'_contribution'+('_pp' if key in ('oa', 'mca') else '')
        expected = gains[key]*(100 if key in ('oa', 'mca') else 1)
        if abs(sum(r[suffix] for r in groups.values())-expected) > 1e-10:
            raise ValueError('nonadditive prediction decomposition')
    return dict(candidate=sg, control=sd, differences=gains, groups=groups)


def checkpoint(role, seed, policy):
    run = SOURCE[role]/'runs'/role/str(seed)
    return run/('best.pt' if policy == 'selected' else 'resume.pt')


def plan():
    specification()
    if (DEST/'plan.json').exists():
        raise ValueError('preserve previous diagnostic plan')
    artifacts = {}
    for role in ROLES:
        root = SOURCE[role]
        for seed in SEEDS:
            run = root/'runs'/role/str(seed)
            meta = read(run/'run.json')
            for name in ('best.pt', 'resume.pt', 'validation.npz', 'history.json'):
                path = run/name; digest = sha(path)
                if digest != meta['hashes'][name]:
                    raise ValueError('original run artifact changed '+str(path))
                artifacts[str(path.relative_to(ROOT))] = digest
            records = read(root/'evaluation'/role/str(seed)/'record.json')['records']
            for item in records:
                path = root/'evaluation'/role/str(seed)/(item['variant']+'.npz')
                digest = sha(path)
                if digest != item['sha256']:
                    raise ValueError('original prediction changed')
                artifacts[str(path.relative_to(ROOT))] = digest
    write(DEST/'plan.json', dict(created_unix=time.time(), plan_sha256=sha(PLAN),
        code_sha256=sha(CODE), manifest_sha256=sha(OLD/'data/manifest.json'), artifacts=artifacts,
        roles=ROLES, seeds=SEEDS, forward_policies=['selected', 'terminal120'],
        forward_splits=['fit', 'val'], used_development_only=True, goal_complete=False))
    print('DIAGNOSTIC_PLAN', sha(DEST/'plan.json'), flush=True)


def check():
    p = read(DEST/'plan.json')
    if (p['plan_sha256'] != sha(PLAN) or p['code_sha256'] != sha(CODE)
            or p['manifest_sha256'] != sha(OLD/'data/manifest.json')):
        raise ValueError('diagnostic plan/source/data identity changed')
    return p


def read_prediction(path, p):
    if sha(path) != p['artifacts'][str(path.relative_to(ROOT))]:
        raise ValueError('prediction bytes changed')
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in ('logits', 'labels', 'ids')}


def saved():
    p = check()
    if (DEST/'saved_result.json').exists():
        raise ValueError('preserve completed diagnostic')
    rows, curves = [], []
    for seed in SEEDS:
        for split, variants in (('val', ('base',)), ('dev_eval', VARIANTS)):
            for variant in variants:
                values = []
                for role in ROLES:
                    root = SOURCE[role]
                    path = (root/'runs'/role/str(seed)/'validation.npz' if split == 'val'
                            else root/'evaluation'/role/str(seed)/(variant+'.npz'))
                    values.append(read_prediction(path, p))
                d, g = values
                if any(not np.array_equal(d[key], g[key]) for key in ('ids', 'labels')):
                    raise ValueError('unpaired predictions')
                comparison = compare(g['logits'], d['logits'], d['labels'])
                for role, value in zip(ROLES, values):
                    independent = runner.metrics(value['logits'], value['labels'])
                    ours = comparison['control' if role == ROLES[0] else 'candidate']
                    if any(abs(ours[k]-independent[k]) > 1e-10 for k in ('ce', 'oa', 'mca')):
                        raise ValueError('independent metric disagreement')
                rows.append(dict(seed=seed, split=split, variant=variant, **comparison))
        histories = [read(SOURCE[role]/'runs'/role/str(seed)/'history.json') for role in ROLES]
        d, g = histories
        if [r['stream_sha256'] for r in d['training']] != [r['stream_sha256'] for r in g['training']]:
            raise ValueError('training streams were not paired')
        curves.append(dict(seed=seed, control=d, candidate=g,
            paired_training_loss_differences=[b['loss']-a['loss'] for a,b in zip(d['training'],g['training'])],
            paired_validation_ce_differences=[b['ce']-a['ce'] for a,b in zip(d['validation'],g['validation'])]))
    write(DEST/'saved_result.json', dict(plan_sha256=sha(DEST/'plan.json'), records=rows,
        curves=curves, analysis_scope='fixed original selected predictions', goal_complete=False))
    for split in ('val', 'dev_eval'):
        values = [r for r in rows if r['split'] == split and r['variant'] == 'base']
        print('SAVED', split, [(r['seed'], r['differences']) for r in values], flush=True)


def forward(seed):
    p = check(); runner.configure()
    if seed not in SEEDS or not torch.cuda.is_available():
        raise ValueError('declared seed and CUDA required')
    directory = DEST/'forward'/str(seed)
    if directory.exists():
        raise ValueError('preserve existing forward attempt')
    directory.mkdir(parents=True)
    start = time.time(); rows = []
    bundles = {split: load(split, OLD) for split in ('fit', 'val')}
    try:
        for role in ROLES:
            for policy in ('selected', 'terminal120'):
                path = checkpoint(role, seed, policy)
                if sha(path) != p['artifacts'][str(path.relative_to(ROOT))]:
                    raise ValueError('checkpoint bytes changed')
                state = torch.load(path, map_location='cpu', weights_only=False)
                epoch = state['epoch'] if policy == 'selected' else state['epoch_completed']
                if policy == 'terminal120' and epoch != 120:
                    raise ValueError('terminal checkpoint epoch changed')
                model = (NativeGSClassifier(role) if role == ROLES[0] else LogShapeClassifier(role)).cuda()
                model.load_state_dict(state['state_dict'])
                for split, data in bundles.items():
                    logits, _ = runner.evaluate(model, data)
                    stats = summarize(observations(logits, data['labels']))
                    prediction = directory/f'{role}_{policy}_{split}.npz'
                    np.savez_compressed(prediction, logits=logits, labels=data['labels'], ids=data['ids'])
                    replay = None
                    if policy == 'selected' and split == 'val':
                        prior = read_prediction(SOURCE[role]/'runs'/role/str(seed)/'validation.npz', p)
                        error = float(abs(logits-prior['logits']).max())
                        changes = int((logits.argmax(1) != prior['logits'].argmax(1)).sum())
                        replay = dict(maximum_logit_error=error, prediction_changes=changes)
                        if error > 2e-5 or changes:
                            raise ValueError('selected validation replay differs materially')
                    rows.append(dict(role=role, seed=seed, policy=policy, epoch=epoch, split=split,
                        metrics=stats, prediction=str(prediction.relative_to(ROOT)),
                        prediction_sha256=sha(prediction), checkpoint_sha256=sha(path), replay=replay))
                    write(directory/'progress.json', dict(records=rows, elapsed_seconds=time.time()-start))
                    print('FORWARD', role, seed, policy, split, stats['ce'], stats['mca'], flush=True)
                del model, state
                torch.cuda.empty_cache()
        write(directory/'result.json', dict(plan_sha256=sha(DEST/'plan.json'), records=rows,
            gpu=torch.cuda.get_device_name(), elapsed_seconds=time.time()-start, complete=True))
    except BaseException as error:
        write(directory/'failure.json', dict(type=type(error).__name__, error=str(error), records=rows))
        raise


def report():
    check(); saved_data = read(DEST/'saved_result.json')
    rows = [r for seed in SEEDS for r in read(DEST/'forward'/str(seed)/'result.json')['records']]
    lookup = {(r['role'],r['seed'],r['policy'],r['split']):r for r in rows}
    gaps = []
    for seed in SEEDS:
        for policy in ('selected','terminal120'):
            metrics = {role:{split:lookup[role,seed,policy,split]['metrics'] for split in ('fit','val')}
                       for role in ROLES}
            difference = {split:{key:metrics[ROLES[1]][split][key]-metrics[ROLES[0]][split][key]
                                 for key in ('ce','macro_ce','oa','mca')} for split in ('fit','val')}
            gaps.append(dict(seed=seed, policy=policy, differences=difference))
    result = dict(plan_sha256=sha(DEST/'plan.json'), saved_result_sha256=sha(DEST/'saved_result.json'),
        forward_records=rows, aligned_gaps=gaps, scientific_interpretation_pending=True,
        fresh_confirmation=False, goal_complete=False)
    write(DEST/'result.json',result)
    lines = ['# R66 classification diagnostic','',
        'Original selected predictions and fixed selected/terminal fit–val forward checks. '
        'Used development data; original R66 conclusion and model selection unchanged.','',
        '| Split | Seed | CE difference | Class-average CE difference | OA pp | mCA pp |',
        '|---|---:|---:|---:|---:|---:|']
    for row in saved_data['records']:
        if row['variant'] != 'base': continue
        d=row['differences']
        lines.append(f"| {row['split']} | {row['seed']} | {d['ce']:+.5f} | {d['macro_ce']:+.5f} | {100*d['oa']:+.3f} | {100*d['mca']:+.3f} |")
    lines += ['', 'Candidate minus distance; all contributions use the full sample/class denominator.','',
        '| Dev seed | Group | Count | CE contribution | Class-CE contribution | mCA contribution pp |',
        '|---|---|---:|---:|---:|---:|']
    for row in saved_data['records']:
        if row['split']!='dev_eval' or row['variant']!='base':continue
        for name,g in row['groups'].items():
            lines.append(f"| {row['seed']} | {name} | {g['count']} | {g['ce_contribution']:+.5f} | {g['macro_ce_contribution']:+.5f} | {g['mca_contribution_pp']:+.3f} |")
    lines += ['', 'Aligned eval mode, canonical512 tokens, no augmentation or label smoothing.','',
        '| Policy | Seed | Fit CE difference | Val CE difference | Fit mCA pp | Val mCA pp |',
        '|---|---:|---:|---:|---:|---:|']
    for row in gaps:
        f,v=row['differences']['fit'],row['differences']['val']
        lines.append(f"| {row['policy']} | {row['seed']} | {f['ce']:+.5f} | {v['ce']:+.5f} | {100*f['mca']:+.3f} | {100*v['mca']:+.3f} |")
    lines += ['', 'All seeds, classes, sources, six original conditions and full training/validation curves '
        'are retained in JSON. Interpretation pending. These descriptive checks do not identify a unique cause.', '']
    (DEST/'report.md').write_text('\n'.join(lines))
    print('DIAGNOSTIC_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('plan','saved','forward','report'))
    parser.add_argument('--seed',type=int); args=parser.parse_args()
    if args.action == 'forward':forward(args.seed)
    else:globals()[args.action]()
