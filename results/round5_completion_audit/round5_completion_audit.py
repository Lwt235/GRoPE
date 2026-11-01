"""Independent offline inventory and count-based confirmation audit.

This reads frozen artifacts without importing the trainer, models or main
analyzer. Checkpoint payloads are read on CPU. It performs no inference and
does not inspect the separate probability audit's prediction archives.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ['point', 'expected', 'normalized', 'isotropic', 'fixed', 'resolvent']
SEEDS = [11, 22, 33, 44, 55]
METRICS = {'ce': 'ce_sum', 'pck8': 'pck8_count', 'pck4': 'pck4_count',
           'epe_argmax': 'epe_argmax_sum', 'epe_soft': 'epe_soft_sum'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def macro(rows, field):
    """Independently aggregate additive queries within physical place."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['scene'].split('/')[0]].append(row)
    return float(np.mean([math.fsum(r[field] for r in part)/sum(r['n_valid'] for r in part)
                          for part in grouped.values()]))


def audit(args):
    import torch
    torch.set_num_threads(1)
    issues, checked = [], []

    def check(condition, description):
        (checked if condition else issues).append(description)

    runs, data = Path(args.runs), Path(args.data)
    lock = read(runs/'confirmation_lock.json')
    selection = read(ROOT/'data/round5_sun3d/selection.json')
    expected_keys = {f'{m}_{s}' for m in METHODS for s in SEEDS}
    check(set(lock['runs']) == expected_keys, 'lock contains exactly six methods x five seeds')
    seals = {'selection_sha256': ROOT/'data/round5_sun3d/selection.json',
             'protocol_sha256': ROOT/'ROUND5_PROTOCOL.md',
             'analysis_sha256': ROOT/'experiments/round5_analysis.py'}
    for name, path in seals.items():
        check(sha(path) == lock[name], f'current {name} matches lock')
    for name, digest in lock['sourcehash'].items():
        check(sha(ROOT/name) == digest, f'current training source {name}')
        check(sha(runs/'source'/name) == digest, f'archived training source {name}')
    archived = read(runs/'source_manifest.json')
    for name, digest in archived.items():
        check(sha(ROOT/name) == digest, f'current manifest source {name}')
        check(sha(runs/'source'/name) == digest, f'archived manifest source {name}')

    expected, sources, validity, cache_summary, place_splits = {}, {}, {}, {}, {}
    for split, expected_count in [('train', 576), ('val', 64), ('test', 480)]:
        scenes = {s['scene']: s for s in selection['scenes'] if s['split'] == split}
        expected[split] = {(name, int(q), int(k)) for name, s in scenes.items() for q, k in s['pairs']}
        check(len(expected[split]) == expected_count, f'{split} selected pair count {expected_count}')
        sources[split], observed = [], set()
        total_valid = 0
        for directory in sorted((data/split).iterdir()):
            if not directory.is_dir():
                continue
            manifest_path = directory/'manifest.json'
            m = read(manifest_path)
            check(m['split'] == split and m['scene'] in scenes, f'{directory.name} split/scene')
            check(sha(directory/'inputs.npz') == m['input_sha256'], f'{directory.name} actual input hash')
            check(sha(directory/'labels.npz') == m['label_sha256'], f'{directory.name} actual label hash')
            with np.load(directory/'labels.npz') as labels:
                counts = labels['gt_valid'].sum(-1).tolist()
            check(len(counts) == len(m['pairs']), f'{directory.name} label pair axis')
            for (q, k), count in zip(m['pairs'], counts):
                identity = (m['scene'], int(q), int(k))
                check(identity not in observed, f'{directory.name} unique pair {q}:{k}')
                observed.add(identity)
                validity[(split, identity)] = int(count)
                total_valid += int(count)
            sources[split].append({key: m[key] for key in ['scene', 'input_sha256', 'label_sha256']} |
                                  {'manifest_sha256': sha(manifest_path)})
            place = m['scene'].split('/')[0]
            place_splits.setdefault(place, set()).add(split)
        check(observed == expected[split], f'{split} complete cache identity set')
        cache_summary[split] = dict(pairs=len(observed), places=len(scenes), valid_queries=total_valid,
                                    sources=sources[split])
    check(all(len(value) == 1 for value in place_splits.values()), 'physical places disjoint across splits')

    run_summary, all_rows = {}, {'val': [], 'test': []}
    for method in METHODS:
        for seed in SEEDS:
            key = f'{method}_{seed}'
            directory = runs/key
            config, done = read(directory/'config.json'), read(directory/'completed.json')
            history = read(directory/'history.json')
            frozen = lock['runs'][key]
            check(config['method'] == method and config['seed'] == seed, f'{key} config identity')
            check(config['parameters'] == 166611, f'{key} parameter count')
            arguments = config['arguments']
            for name, value in [('steps', 1000), ('eval_every', 100), ('batch_size', 8),
                                ('lr', .001), ('limit', None), ('test_only', False), ('zero_values', False)]:
                check(arguments[name] == value, f'{key} argument {name}')
            check(seed in arguments['seeds'] and method in arguments['methods'], f'{key} command group')
            check(config['sourcehash'] == done['sourcehash'] == lock['sourcehash'], f'{key} source equality')
            check(config['train_sources'] == sources['train'], f'{key} train cache provenance')
            check(config['val_sources'] == sources['val'], f'{key} validation cache provenance')
            check([row['step'] for row in history] == list(range(0, 1001, 100)), f'{key} complete validation history')
            check(all(math.isfinite(row['val_ce']) and 0 <= row['val_pck8'] <= 1 for row in history),
                  f'{key} finite history metrics')
            best = min(history, key=lambda row: row['val_ce'])  # stable earliest tie
            check(best['step'] == frozen['selected_step'], f'{key} earliest minimum CE step')
            check(best['val_ce'] == done['best_val_ce'] == frozen['val_ce'], f'{key} minimum CE equality')
            check(best['val_pck8'] == frozen['val_pck8'], f'{key} selected PCK equality')
            checkpoint = directory/'checkpoints/best.pt'
            digest = sha(checkpoint)
            check(digest == done['checkpoint_sha256'] == frozen['checkpoint_sha256'], f'{key} checkpoint SHA')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            check(payload['method'] == method and payload['seed'] == seed and payload['step'] == best['step']
                  and payload['sourcehash'] == lock['sourcehash'], f'{key} checkpoint payload identity/source')
            for split in ['val', 'test']:
                wrapper = read(directory/f'{split}.json')
                rows = wrapper['rows']
                check(wrapper['selected_step'] == best['step'], f'{key} {split} selected step')
                check(wrapper['input_sources'] == sources[split], f'{key} {split} source hashes')
                if split == 'test':
                    check(wrapper['sourcehash'] == lock['sourcehash'], f'{key} test source equality')
                identities = [(r['scene'], r['q_index'], r['k_index']) for r in rows]
                check(set(identities) == expected[split] and len(identities) == len(expected[split]),
                      f'{key} {split} complete unique pair set')
                check(all(r['method'] == method and r['seed'] == seed and r['split'] == split
                          and r['pair_id'] == f"{r['q_index']}:{r['k_index']}" for r in rows),
                      f'{key} {split} row identities')
                check(all(r['n_valid'] == validity[(split, identity)] for r, identity in zip(rows, identities)),
                      f'{key} {split} n_valid equals label counts')
                check(all(0 <= r['pck4_count'] <= r['pck8_count'] <= r['n_valid'] and
                          all(math.isfinite(r[field]) and r[field] >= 0 for field in METRICS.values())
                          for r in rows), f'{key} {split} finite admissible counts/sums')
                all_rows[split].extend(rows)
                if split == 'val':
                    check(abs(macro(rows, 'ce_sum')-best['val_ce']) < 1e-14, f'{key} validation CE from rows')
                    check(macro(rows, 'pck8_count') == best['val_pck8'], f'{key} validation PCK from counts')
            run_summary[key] = dict(selected_step=best['step'], checkpoint_sha256=digest,
                seconds=done['seconds'], max_gpu_memory=done['max_gpu_memory'], device=arguments['device'])

    validation_scores = {m: float(np.mean([macro([r for r in all_rows['val'] if r['method'] == m and
        r['seed'] == s], 'pck8_count') for s in SEEDS])) for m in ['point', 'fixed']}
    selected = max(['point', 'fixed'], key=validation_scores.get)
    check(validation_scores == lock['validation_baseline_scores'] and selected == lock['primary_baseline'],
          'validation-only baseline selection equals lock')
    accumulator = defaultdict(list)
    for row in all_rows['test']:
        accumulator[(row['method'], row['seed'], row['scene'].split('/')[0])].append(row)
    places = sorted({key[2] for key in accumulator})
    matrices = {m: {metric: np.array([[math.fsum(r[field] for r in accumulator[(m, s, p)]) /
        sum(r['n_valid'] for r in accumulator[(m, s, p)]) for p in places] for s in SEEDS], np.float64)
        for metric, field in METRICS.items()} for m in METHODS}
    difference = matrices['normalized']['pck8']-matrices[selected]['pck8']
    generator = np.random.default_rng(20261030)
    sd = generator.integers(0, 5, (20000, 5))
    pd = generator.integers(0, len(places), (20000, len(places)))
    distribution = difference[sd[:, :, None], pd[:, None, :]].mean((1, 2))
    mean, ci = float(difference.mean()), np.quantile(distribution, [.025, .975]).tolist()
    summary = read(Path(args.analysis)/'summary.json')
    primary = summary['primary_gate']
    check(primary['pck8_difference'] == mean and primary['ci95_crossed'] == ci,
          'independent count-based primary mean/CI exactly equals main summary')
    check(primary['positive_seed_count'] == int((difference.mean(1) > 0).sum()), 'primary seed direction count')
    check(summary['baseline_selection']['selected'] == selected, 'analysis baseline equals lock')
    for m in METHODS:
        for metric in METRICS:
            check(summary['test_macro'][m][metric] == float(matrices[m][metric].mean()),
                  f'{m} {metric} query-weighted within place / equal-place macro')
    analysis_config = read(Path(args.analysis)/'audit_config.json')
    check(analysis_config['source_sha256'] == lock['analysis_sha256'], 'analysis source matches sealed source')
    for path, digest in analysis_config['input_sha256'].items():
        check(sha(ROOT/path) == digest, f'analysis input unchanged {path}')
    log = Path(args.replay_log).read_text()
    replayed = re.findall(r'^VALIDATION_REPLAY_OK (\S+)$', log, re.M)
    check(set(replayed) == expected_keys and len(replayed) == 30, 'all 30 validation replay log entries unique')
    return dict(passed=not issues, issues=issues, checks_passed=len(checked), checks=checked,
        counts=cache_summary, run_count=len(run_summary), runs=run_summary,
        validation_baseline_scores=validation_scores, selected_baseline=selected,
        independently_recomputed_primary=dict(mean=mean, ci95_crossed=ci,
            positive_seed_count=int((difference.mean(1) > 0).sum()),
            per_seed=dict(zip(map(str, SEEDS), difference.mean(1).tolist())),
            per_place=dict(zip(places, difference.mean(0).tolist())),
            bootstrap_draws=20000, bootstrap_seed=20261030,
            aggregation='query-weighted within seed/place, equal-place mean, paired crossed seed x place'),
        macro_metrics={m: {metric: float(matrix.mean()) for metric, matrix in values.items()}
                       for m, values in matrices.items()},
        frozen_artifacts=dict(lock_sha256=sha(runs/'confirmation_lock.json'),
            analysis_summary_sha256=sha(Path(args.analysis)/'summary.json'),
            replay_log_sha256=sha(args.replay_log)),
        summed_training_seconds=sum(r['seconds'] for r in run_summary.values()),
        scope='offline integrity and independent row-count statistics; no inference/probability audit duplication')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=str(ROOT/'results/round5_confirmation'))
    parser.add_argument('--data', default=str(ROOT/'results/round5_inputs/pairs'))
    parser.add_argument('--analysis', default=str(ROOT/'results/round5_analysis'))
    parser.add_argument('--replay-log', default=str(ROOT/'results/round5_test_evaluation.log'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('choose a new audit output directory')
    result = audit(args)
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/Path(__file__).name)
    (out/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ['passed', 'issues', 'checks_passed', 'run_count',
                     'selected_baseline', 'independently_recomputed_primary']}), flush=True)
    if not result['passed']:
        raise SystemExit(1)
