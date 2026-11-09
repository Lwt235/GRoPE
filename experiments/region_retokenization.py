"""One frozen, old-development inference test of reconstructed sparse regions."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.round65_common import ROOT, read, write, sha
from experiments.round65_run import configure
from experiments.round68_fixed_region_run import inputs, query_values, aggregate
from experiments.round68_fixed_region_data import shuffle_indices
from experiments.round58_run import metric_torch
from experiments.round60_audit import audit_sample
from experiments.region_geometry_controls import dependencies
from models.round68_fixed_region import FixedRegionMatcher
from models.round69_compact_relation import CompactMatcher
from models.region_geometry_controls import GeometryControlMatcher
from models.region_shape_attribution import FixedIsotropicMatcher

OUT = ROOT / 'results/region_retokenization/attempt1'
PLAN = ROOT / 'docs/region_retokenization_plan.md'
PARENT = ROOT / 'results/round68_fixed_region/attempt1'
R60 = ROOT / 'results/round60_real_scene/attempt1'
ROLES = ('relative5', 'distance_mlp', 'geotransformer_rpe', 'fixed_isotropic5')
SEEDS = (11, 22, 33)
CONDITIONS = ('base', 'fixed_sparse256', 'retokenized256')
ROOTS = dict(relative5=ROOT / 'results/round69_compact_relation/attempt1',
             distance_mlp=PARENT,
             geotransformer_rpe=ROOT / 'results/region_geometry_controls/attempt1',
             fixed_isotropic5=ROOT / 'results/region_shape_attribution/attempt1')
CACHES = dict(base=PARENT / 'data/eval_base.npz',
              fixed_sparse256=PARENT / 'data/eval_sparse256.npz',
              retokenized256=R60 / 'data/eval_sparse256.npz')
METRICS = ('region_recall', 'valid_and_correct_fraction', 'coverage',
           'center_pck', 'hard_epe', 'soft_epe', 'ce')


def make_model(role):
    if role == 'relative5':
        return CompactMatcher(role)
    if role == 'distance_mlp':
        return FixedRegionMatcher(role)
    if role == 'geotransformer_rpe':
        return GeometryControlMatcher(role)
    if role == 'fixed_isotropic5':
        return FixedIsotropicMatcher()
    raise ValueError(role)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_cache(condition):
    cache = load_arrays(CACHES[condition])
    if 'permutation' not in cache:
        cache['permutation'] = shuffle_indices(cache['ids'])
    return cache


def metrics(logits, cache):
    values = query_values(logits, cache)
    world = np.asarray(cache['world_centers'], dtype=np.float64)
    hard = []
    for direction in (0, 1):
        score = logits if direction == 0 else logits.swapaxes(-1, -2)
        target = world[:, 1-direction]
        chosen = target[np.arange(len(target))[:, None], score.argmax(-1)]
        hard.append(np.linalg.norm(chosen-world[:, direction], axis=-1))
    values['hard_epe'] = np.stack(hard, axis=1)
    scenes = {str(s): aggregate(values, (cache['scenes'] == s)[:, None, None])
              for s in sorted(set(cache['scenes']))}
    macro = {}
    for key in METRICS:
        eligible = [r[key] for r in scenes.values() if r[key] is not None]
        macro[key] = float(np.mean(eligible)) if eligible else None
    return dict(micro=aggregate(values), macro=macro, by_scene=scenes,
                zero_valid_pairs=int((values['valid'].sum((1, 2)) == 0).sum()),
                evaluable_scenes=sum(r['valid_queries'] > 0 for r in scenes.values()))


def paired_summary(delta, seed):
    delta = np.asarray(delta, dtype=np.float64)
    assert delta.ndim == 2 and np.isfinite(delta).all()
    rng = np.random.default_rng(seed)
    si = rng.integers(0, len(delta), (10000, len(delta)))
    ci = rng.integers(0, delta.shape[1], (10000, delta.shape[1]))
    boot = delta[si[:, :, None], ci[:, None, :]].mean((1, 2))
    return dict(mean_gain_pp=float(delta.mean()), paired_seed_gains_pp=delta.mean(1).tolist(),
                scene_mean_gain_pp=delta.mean(0).tolist(), seed_scene_gains_pp=delta.tolist(),
                crossed_seed_scene_bootstrap_95_pp=np.quantile(boot, [.025, .975]).tolist(),
                positive_seed_pairs=int((delta.mean(1) > 0).sum()), bootstrap_seed=seed)


def audit_data():
    """Bounded source/geometry check using original saved inputs, no scoring."""
    base, sparse = [load_arrays(R60 / 'data' / ('eval_'+c+'_sample.npz'))
                    for c in ('base', 'sparse256')]
    for key in ('points', 'sensor', 'source_ids', 'world', 'clean'):
        np.testing.assert_array_equal(sparse[key], base[key][:, :, :256])
    for key in ('origins', 'scale', 'poses'):
        np.testing.assert_array_equal(base[key], sparse[key])
    assert sparse['token_neighbors'].shape == (2, 2, 64, 32)
    assert sparse['token_neighbors'].max() < 256
    old, current, new = load_arrays(R60 / 'data/eval_base.npz'), load_cache('base'), load_cache('retokenized256')
    for key in ('ids', 'scenes', 'scales', 'frame_gaps'):
        np.testing.assert_array_equal(current[key], new[key])
        np.testing.assert_array_equal(old[key], current[key])
    np.testing.assert_array_equal(old['overlap'], current['overlap'])
    errors = {k: float(np.max(np.abs(old[k]-current[k])))
              for k in ('features', 'centers', 'covariance', 'world_centers')}
    assert max(errors.values()) < 2e-4, errors
    return dict(independent_sparse_geometry=audit_sample(R60 / 'data/eval_sparse256_sample.npz'),
                sparse_is_prefix=True, same_pairs_and_coordinate_normalization=True,
                base_cpu_gpu_rounding=errors, sparse_queries_not_assumed_identical=True,
                dense_sample_anchors_discarded=int((base['token_anchors'] >= 256).sum()),
                pair_count=len(new['ids']), scene_count=len(set(new['scenes'])))


def freeze():
    assert not OUT.exists(), 'preserve prior attempt'
    units = read(ROOT / 'results/region_retokenization/preparation/cpu_checks.json')
    assert units['passed']
    for path, digest in units['source_sha256'].items():
        assert sha(path) == digest
    sources = {str(p): sha(p) for p in dependencies([Path(__file__), ROOT/'tests/test_region_retokenization.py'])}
    sources[str(PLAN)] = sha(PLAN)
    sources[str(ROOT/'results/region_retokenization/preparation/cpu_checks.json')] = sha(ROOT/'results/region_retokenization/preparation/cpu_checks.json')
    old_lock = read(R60 / 'lock.json')
    for name in ('experiments/round60_data.py', 'experiments/round58_data.py', 'experiments/round60_common.py'):
        assert sha(ROOT/name) == old_lock['sources'][name], name
    provenance = {str(R60/'lock.json'): sha(R60/'lock.json')}
    for path in list(CACHES.values()) + [PARENT/'data/val.npz', R60/'data/eval_base.npz']:
        meta = read(path.with_suffix('.json'))
        assert sha(path) == meta['sha256']
        provenance[str(path)] = sha(path)
        provenance[str(path.with_suffix('.json'))] = sha(path.with_suffix('.json'))
    for name in ('eval_base_sample.npz', 'eval_sparse256_sample.npz'):
        path = R60/'data'/name
        provenance[str(path)] = sha(path)
    models, predictions = [], []
    for role in ROLES:
        root = ROOTS[role]
        assert read(root/'review.json')['scientific_review_complete']
        for name in ('selected.json', 'review.json', 'lock.json'):
            provenance[str(root/name)] = sha(root/name)
        selection = read(root/'selected.json')
        for seed in SEEDS:
            chosen = next(x for x in selection['records'] if (x['role'], x['seed']) == (role, seed))
            folder = root/'runs'/role/str(seed)
            for name, key in (('best.pt', 'checkpoint_sha256'), ('run.json', 'run_sha256')):
                assert sha(folder/name) == chosen[key]
                provenance[str(folder/name)] = chosen[key]
            provenance[str(folder/'validation.npz')] = sha(folder/'validation.npz')
            models.append(dict(chosen, checkpoint_path=str(folder/'best.pt'), validation_path=str(folder/'validation.npz')))
            directory = root/'evaluation'/role/str(seed)
            record = read(directory/'record.json')
            assert record['checkpoint_sha256'] == chosen['checkpoint_sha256']
            assert record['selection_sha256'] == sha(root/'selected.json')
            assert record['manifest_sha256'] == sha(PARENT/'data/manifest.json')
            provenance[str(directory/'record.json')] = sha(directory/'record.json')
            for condition in ('base', 'sparse256'):
                entry = next(x for x in record['records'] if x['condition'] == condition)
                path = directory/(condition+'.npz')
                assert sha(path) == entry['prediction_sha256']
                provenance[str(path)] = sha(path)
                predictions.append(dict(role=role, seed=seed,
                    condition='fixed_sparse256' if condition == 'sparse256' else condition,
                    path=str(path), sha256=entry['prediction_sha256']))
    check = audit_data()
    write(OUT/'lock.json', dict(created_unix=time.time(), source_sha256=sources,
        provenance_sha256=provenance, models=models, reused_predictions=predictions,
        data_audit=check, roles=ROLES, seeds=SEEDS, conditions=CONDITIONS,
        new_fits=0, new_prediction_groups=12, reused_prediction_groups=24,
        fresh_confirmation=False, official_test_access=False, goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('FROZEN', sha(OUT/'lock.json'), check, flush=True)


def check():
    lock = read(OUT/'lock.json')
    assert sha(OUT/'lock.json') == (OUT/'lock.sha256').read_text().strip()
    for group in ('source_sha256', 'provenance_sha256'):
        for path, digest in lock[group].items():
            assert sha(path) == digest, path
    return lock


@torch.inference_mode()
def infer(model, cache):
    result = np.concatenate([model(*inputs(cache, slice(i, i+16))).cpu().numpy()
                             for i in range(0, len(cache['ids']), 16)])
    assert np.isfinite(result).all()
    return result


def run():
    lock = check()
    configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert not (OUT/'execution_identity.json').exists(), 'do not restart an existing run blindly'
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(3*2**30/total)
    identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                    started_unix=time.time(), lock_sha256=sha(OUT/'lock.json'), gpu=torch.cuda.get_device_name())
    write(OUT/'execution_identity.json', identity)
    completed = []
    def status(stage, active=None, **extra):
        write(OUT/'status.json', dict(identity, stage=stage, active=active,
              completed=completed, observed_unix=time.time(), goal_complete=False, **extra))
    try:
        validation = load_arrays(PARENT/'data/val.npz')
        replay = []
        for item in lock['models']:
            role, seed = item['role'], item['seed']
            status('validation_replay', dict(role=role, seed=seed))
            payload = torch.load(item['checkpoint_path'], map_location='cpu', weights_only=True)
            assert (payload['role'], payload['seed'], payload['epoch']) == (role, seed, item['selected_epoch'])
            model = make_model(role).cuda().eval()
            model.load_state_dict(payload['state_dict'])
            actual = infer(model, validation)
            expected = load_arrays(item['validation_path'])
            assert np.array_equal(expected['ids'], validation['ids'])
            error = float(np.abs(actual-expected['logits']).max())
            changes = sum(int((a.argmax(-1) != b.argmax(-1)).sum()) for a,b in
                          ((actual, expected['logits']), (actual.swapaxes(-1,-2), expected['logits'].swapaxes(-1,-2))))
            assert error < 2e-4 and changes == 0, (role, seed, error, changes)
            replay.append(dict(role=role, seed=seed, maximum_logit_error=error, prediction_changes=changes))
            print('REPLAY', role, seed, error, changes, flush=True)
            del model
            torch.cuda.empty_cache()
        write(OUT/'preflight.json', dict(records=replay, passed=True, completed_unix=time.time(),
                                       lock_sha256=sha(OUT/'lock.json')))
        cache = load_cache('retokenized256')
        for item in lock['models']:
            role, seed = item['role'], item['seed']
            status('export', dict(role=role, seed=seed))
            start = time.time()
            model = make_model(role).cuda().eval()
            model.load_state_dict(torch.load(item['checkpoint_path'], map_location='cpu', weights_only=True)['state_dict'])
            torch.cuda.reset_peak_memory_stats()
            logits = infer(model, cache)
            path = OUT/'predictions'/role/str(seed)/'retokenized256.npz'
            path.parent.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(path, ids=cache['ids'], logits=logits)
            record = dict(role=role, seed=seed, condition='retokenized256', path=str(path),
                sha256=sha(path), checkpoint_sha256=item['checkpoint_sha256'],
                cache_sha256=sha(CACHES['retokenized256']), lock_sha256=sha(OUT/'lock.json'),
                seconds=time.time()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                completed_unix=time.time())
            write(path.with_suffix('.json'), record)
            completed.append(record)
            print('EXPORTED', role, seed, record['seconds'], flush=True)
            del model
            torch.cuda.empty_cache()
        check()
        status('complete', prediction_groups=12)
    except BaseException as error:
        status('failed', error=repr(error))
        raise


def report():
    lock = check()
    assert read(OUT/'status.json')['stage'] == 'complete'
    assert not (OUT/'result.json').exists()
    caches = {c: load_cache(c) for c in CONDITIONS}
    records = []
    sources = list(lock['reused_predictions']) + read(OUT/'status.json')['completed']
    for item in sources:
        assert sha(item['path']) == item['sha256']
        archive = load_arrays(item['path'])
        cache = caches[item['condition']]
        assert np.array_equal(cache['ids'], archive['ids'])
        actual = metrics(archive['logits'], cache)
        error = 0.
        for scene, scope in [(None, actual['micro'])] + list(actual['by_scene'].items()):
            index = slice(None) if scene is None else cache['scenes'] == scene
            independent = metric_torch(archive['logits'][index], cache['overlap'][index], cache['world_centers'][index])
            assert sum(independent['valid_counts']) == scope['valid_queries']
            for key in ('region_recall', 'center_pck', 'hard_epe', 'soft_epe', 'ce'):
                if scope[key] is not None:
                    error = max(error, abs(independent[key]-scope[key]))
            coverage = independent['eligible_fraction']
            error = max(error, abs(coverage-scope['coverage']),
                        abs(coverage*independent['region_recall']-scope['valid_and_correct_fraction']))
        assert error < 1e-10, error
        records.append(dict(item, metrics=actual, independent_metric_max_error=error,
                            reused=item['condition'] != 'retokenized256'))
    lookup = {(r['condition'], r['role'], r['seed']): r for r in records}
    assert len(lookup) == len(records) == 36
    comparisons = []
    for condition in CONDITIONS:
        for control in ROLES[1:]:
            a = [lookup[condition, 'relative5', seed]['metrics'] for seed in SEEDS]
            b = [lookup[condition, control, seed]['metrics'] for seed in SEEDS]
            pair = {k: [x['macro'][k]-y['macro'][k] for x,y in zip(a,b)] for k in METRICS}
            for metric in ('region_recall', 'valid_and_correct_fraction'):
                scenes = [s for s in sorted(a[0]['by_scene']) if a[0]['by_scene'][s][metric] is not None]
                delta = [[100*(x['by_scene'][s][metric]-y['by_scene'][s][metric]) for s in scenes] for x,y in zip(a,b)]
                comparisons.append(dict(condition=condition, left='relative5', right=control, metric=metric,
                    primary=condition=='retokenized256' and control != 'fixed_isotropic5' and metric=='region_recall',
                    scenes=scenes, paired_metrics=pair, **paired_summary(delta, 88001+len(comparisons))))
    result = dict(records=records, comparisons=comparisons, lock_sha256=sha(OUT/'lock.json'),
        new_fits=0, new_prediction_groups=12, reused_prediction_groups=24,
        independent_metric_max_error=max(r['independent_metric_max_error'] for r in records),
        candidate='relative5', fresh_confirmation=False, official_test_access=False,
        scientific_review_pending=True, visual_review_pending=True, goal_complete=False)
    write(OUT/'result.json', result)
    lines = ['# 稀疏区域重构：自动汇总，待实际审阅', '',
             '同一旧五场景、480 帧对，12 个冻结模型，无新增拟合。全部查询比例并不表示已训练拒识。', '',
             '| 条件 | 方法 | 条件 R@1 % | 全查询正确 % | 有效对应 % | 中心 PCK % | hard EPE | soft EPE | CE |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for condition in CONDITIONS:
        for role in ROLES:
            avg = {k: float(np.mean([lookup[condition,role,s]['metrics']['macro'][k] for s in SEEDS])) for k in METRICS}
            lines.append(f"| {condition} | {role} | {100*avg['region_recall']:.3f} | {100*avg['valid_and_correct_fraction']:.3f} | {100*avg['coverage']:.3f} | {100*avg['center_pck']:.3f} | {avg['hard_epe']:.5f} | {avg['soft_epe']:.5f} | {avg['ce']:.5f} |")
    lines += ['', '| 条件 | F 减对照 | 指标 | 三种子差 pp | 均值 pp | 联合区间 pp |',
              '| --- | --- | --- | --- | ---: | --- |']
    for c in comparisons:
        values=' / '.join(f'{x:+.3f}' for x in c['paired_seed_gains_pp'])
        lo,hi=c['crossed_seed_scene_bootstrap_95_pp']
        lines.append(f"| {c['condition']} | {c['right']} | {c['metric']} | {values} | {c['mean_gain_pp']:+.3f} | [{lo:+.3f},{hi:+.3f}] |")
    lines += ['', 'EPE 是配对归一化场景单位。三条件的查询身份与标签不相同；同条件方法间才共享评价人群。',
              '重构条件沿用抽样前视图均值及尺度，不是完全稀疏传感器流程。原生、RGB 和选择性输出负结果保持有效。', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax, metric, label in zip(axes, ('region_recall','valid_and_correct_fraction'),
                                ('Conditional R@1 gain (pp)','Correct / all queries gain (pp)')):
        cs=[c for c in comparisons if c['condition']=='retokenized256' and c['metric']==metric]
        for i,c in enumerate(cs):
            ax.scatter(np.repeat(i,3),c['paired_seed_gains_pp'],alpha=.75)
            ax.plot([i-.2,i+.2],[c['mean_gain_pp']]*2,color='black')
        ax.axhline(0,color='gray',lw=.8)
        ax.set_xticks(range(3),['F - Distance','F - GeoTransformer','F - fixed'],rotation=18)
        ax.set_ylabel(label)
    fig.savefig(OUT/'retokenized_comparison.png',dpi=160)
    plt.close(fig)
    print('REPORT', len(records), len(comparisons), result['independent_metric_max_error'], flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','run','report'))
    args=parser.parse_args()
    if args.action != 'run':
        torch.set_num_threads(4)
    globals()[args.action]()
