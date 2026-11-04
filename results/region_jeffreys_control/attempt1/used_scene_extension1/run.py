"""Fixed inference on a previously used extension cohort, not fresh data."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from experiments.round65_common import ROOT, read, write, sha
from experiments.region_geometry_controls import dependencies
from experiments.region_confirmation import configure, infer
from experiments.region_confirmation_report import population_metrics, independent_error, compare
from experiments.region_confirmation_metric_runtime import BufferedNpz, batched_query
from experiments import round68_fixed_region_run as engine
from models.region_jeffreys_control import JeffreysMatcher

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parent
OLD = ROOT / 'results/region_confirmation/attempt1'
SEEDS = (11, 22, 33)
CONDITIONS = ('base', 'sparse256')


def freeze():
    assert not (OUT / 'lock.json').exists()
    plan = read(OUT / 'plan.json')
    sources = dict(plan['source_sha256'])
    sources.update({str(p): sha(p) for p in dependencies([Path(__file__)])})
    sources[str(OUT / 'plan.json')] = sha(OUT / 'plan.json')
    for name, digest in sources.items():
        assert sha(name) == digest, name
    for item in plan['checkpoints'].values():
        assert sha(item['path']) == item['sha256']
    for item in plan['cache_identity'].values():
        assert sha(item['path']) == item['sha256']
    write(OUT / 'lock.json', dict(source_sha256=sources, created_unix=time.time(),
        plan_sha256=sha(OUT / 'plan.json'), no_training=True, fresh_confirmation=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('USED_SCENE_EXTENSION_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    lock = read(OUT / 'lock.json')
    for name, digest in lock['source_sha256'].items():
        assert sha(name) == digest, name
    return read(OUT / 'plan.json')


def load(item):
    assert sha(item['path']) == item['sha256']
    with BufferedNpz(item['path']) as archive:
        return {k: archive[k] for k in archive.files}


@torch.inference_mode()
def export(seed):
    plan = check()
    configure()
    identity = plan['checkpoints'][str(seed)]
    assert sha(identity['path']) == identity['sha256']
    dest = OUT / 'predictions' / str(seed)
    dest.mkdir(parents=True, exist_ok=False)
    model = JeffreysMatcher().cuda().eval()
    payload = torch.load(identity['path'], map_location='cpu', weights_only=True)
    assert (payload['role'], payload['seed'], payload['epoch']) == ('jeffreys_mlp', seed, identity['selected_epoch'])
    model.load_state_dict(payload['state_dict'])
    valmeta = read(ROOT / 'results/round68_fixed_region/attempt1/data/manifest.json')['caches']['val']
    val = load(valmeta)
    logits = infer(model, val)
    with BufferedNpz(Path(identity['path']).parent / 'validation.npz') as archive:
        assert np.array_equal(archive['ids'], val['ids'])
        error = float(abs(logits - archive['logits']).max())
        changes = int((logits.argmax(-1) != archive['logits'].argmax(-1)).sum())
    assert error < 2e-4 and changes == 0
    records = []
    for condition in CONDITIONS:
        cache = load(plan['cache_identity'][condition])
        logits = infer(model, cache)
        path = dest / (condition + '.npz')
        np.savez_compressed(path, logits=logits, ids=cache['ids'])
        records.append(dict(role='jeffreys_mlp', seed=seed, condition=condition,
                            prediction_sha256=sha(path)))
    write(dest / 'identity.json', dict(records=records, checkpoint_sha256=identity['sha256'],
        lock_sha256=sha(OUT / 'lock.json'), old_validation_replay=dict(maximum_logit_error=error, prediction_changes=changes),
        gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        optimizer_updates=0, fresh_confirmation=False))
    print('USED_EXTENSION_EXPORTED', seed, flush=True)


def report():
    plan = check()
    assert not (OUT / 'result.json').exists()
    engine.configure()
    engine.query_values = batched_query
    caches = {c: load(plan['cache_identity'][c]) for c in CONDITIONS}
    rows = [r for r in read(OLD / 'result.json')['records']
            if r['role'] in ('relative5', 'distance_mlp', 'fixed_isotropic5')]
    assert len(rows) == 18
    for row in rows:
        path = OLD / 'predictions' / row['role'] / str(row['seed']) / (row['condition'] + '.npz')
        assert sha(path) == row['prediction_sha256']
    errors, replays = [], []
    for seed in SEEDS:
        dest = OUT / 'predictions' / str(seed)
        identity = read(dest / 'identity.json')
        assert identity['checkpoint_sha256'] == plan['checkpoints'][str(seed)]['sha256']
        assert identity['lock_sha256'] == sha(OUT / 'lock.json')
        replays.append(dict(seed=seed, **identity['old_validation_replay']))
        for record in identity['records']:
            path, cache = dest / (record['condition'] + '.npz'), caches[record['condition']]
            assert sha(path) == record['prediction_sha256']
            with BufferedNpz(path) as archive:
                assert np.array_equal(archive['ids'], cache['ids'])
                metrics = population_metrics(archive['logits'], cache, plan['planned_scenes'])
                error = independent_error(archive['logits'], cache, metrics)
            assert error < 1e-10 and metrics['micro']['valid_queries'] == 35859
            errors.append(error)
            rows.append(dict(record, metrics=metrics, independent_metric_error=error))
    assert len(rows) == 24
    pairs = (('relative5', 'jeffreys_mlp'), ('jeffreys_mlp', 'distance_mlp'), ('jeffreys_mlp', 'fixed_isotropic5'))
    comparisons = [compare(rows, a, b, c) for a, b in pairs for c in CONDITIONS]
    strata = [compare(rows, a, b, c, g) for a, b in pairs for c in CONDITIONS for g in range(4)]
    result = dict(records=rows, comparisons=comparisons, strata_comparisons=strata,
        replays=replays, independent_metric_max_error=max(errors),
        available_scenes=plan['available_scenes'], unavailable=plan['unavailable'],
        new_fits=0, new_prediction_groups=6, reused_prediction_groups=18,
        lock_sha256=sha(OUT / 'lock.json'), fresh_confirmation=False,
        population_previously_used=True, scientific_review_pending=True, goal_complete=False)
    write(OUT / 'result.json', result)
    lookup = {(r['role'], r['seed'], r['condition']): r['metrics']['macro'] for r in rows}
    lines = ['# 已使用九场景上的可学习散度补充比较', '',
        '三个旧验证集已选模型，零训练；这九个场景已经使用，本次不计为新的独立确认。预定第十个场景缺标定继续保留为缺失。', '',
        '| 条件 | 方法 | R@1 % | 中心 PCK % | Soft EPE | CE |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for condition in CONDITIONS:
        for role in ('distance_mlp', 'fixed_isotropic5', 'jeffreys_mlp', 'relative5'):
            values = [np.mean([lookup[role, s, condition][k] for s in SEEDS]) *
                      (100 if k in ('region_recall', 'center_pck') else 1)
                      for k in ('region_recall', 'center_pck', 'soft_epe', 'ce')]
            lines.append(f'| {condition} | {role} | ' + ' | '.join(f'{x:.5f}' for x in values) + ' |')
    lines += ['', '| 条件 / 左减右 | 三种子召回差 pp | 均值 pp | 场景区间 pp |', '| --- | --- | ---: | --- |']
    for row in comparisons:
        diffs = ' / '.join(f"{r['region_recall']:+.3f}" for r in row['paired'])
        lo, hi = row['paired_scene_bootstrap95_pp']
        lines.append(f"| {row['condition']} / {row['candidate']} − {row['control']} | {diffs} | "
                     f"{row['mean']['region_recall']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', '逐场景、四分组、中心指标与交叉种子/场景区间见 result.json。该补充比较不回写原确认报告，也不重新挑选模型。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print('USED_EXTENSION_REPORTED', max(errors), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'export', 'report'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == 'export':
        export(args.seed)
    else:
        globals()[args.action]()
