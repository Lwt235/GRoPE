"""Model-independent, paired seed/place analysis of Round 5 evaluation JSON.

Input is one or more JSON lists of per-pair additive counts/sums. Queries are
weighted within each physical place, then places receive equal macro weight.
The validation baseline is chosen only among point/fixed by mean PCK8; no test
result chooses a baseline. No model, image, or geometry module is imported.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


DEFAULT_SEEDS = [11, 22, 33, 44, 55]
BOOTSTRAP_SEED = 20261030
DEFAULT_BOOTSTRAP = 20000
FIELDS = {'ce': 'ce_sum', 'pck8': 'pck8_count', 'pck4': 'pck4_count',
          'epe_argmax': 'epe_argmax_sum', 'epe_soft': 'epe_soft_sum'}
IDENTITY_FIELDS = ['method', 'seed', 'split', 'scene', 'pair_id', 'q_index', 'k_index']
REQUIRED = IDENTITY_FIELDS+['n_valid']+list(FIELDS.values())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_evaluation_file(path):
    """Read a bare row list or the trainer's rows-plus-metadata wrapper.

    Hash the exact bytes parsed, rather than rereading between parsing and
    hashing. Metadata is retained verbatim for provenance, never selection.
    """
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if isinstance(payload, list):
        rows, metadata, kind = payload, {}, 'bare_list'
    elif isinstance(payload, dict) and 'rows' in payload:
        rows = payload['rows']
        metadata = {key: value for key, value in payload.items() if key != 'rows'}
        kind = 'trainer_wrapper'
    else:
        raise ValueError('evaluation JSON must be a row list or a dictionary containing a rows list')
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('evaluation rows must be a list of dictionaries')
    return rows, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                     payload_kind=kind, row_count=len(rows), metadata=metadata)


def _canonical(values):
    return sorted(values, key=lambda value: (type(value).__name__, str(value)))


def bootstrap_draws(seeds, places, draws=DEFAULT_BOOTSTRAP, random_seed=BOOTSTRAP_SEED):
    if draws < 1 or seeds < 1 or places < 1:
        raise ValueError('positive bootstrap draws/seeds/places required')
    generator = np.random.default_rng(random_seed)
    return (generator.integers(0, seeds, (draws, seeds)),
            generator.integers(0, places, (draws, places)))


def crossed_bootstrap(differences, seed_ids, place_ids, draws=DEFAULT_BOOTSTRAP,
                      random_seed=BOOTSTRAP_SEED):
    """Paired crossed seed x place bootstrap, not independent token resampling.

    Sorting by identities makes row/column permutations exactly reproducible.
    The same sampled seeds and places are crossed in every bootstrap replicate.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (len(seed_ids), len(place_ids)) or not np.isfinite(values).all():
        raise ValueError('finite seed-by-place matrix required')
    if len(set(seed_ids)) != len(seed_ids) or len(set(place_ids)) != len(place_ids):
        raise ValueError('seed/place identities must be unique')
    seeds, places = _canonical(seed_ids), _canonical(place_ids)
    seed_lookup = {name: index for index, name in enumerate(seed_ids)}
    place_lookup = {name: index for index, name in enumerate(place_ids)}
    values = values[[seed_lookup[name] for name in seeds]][:, [place_lookup[name] for name in places]]
    sd, pd = bootstrap_draws(len(seeds), len(places), draws, random_seed)
    crossed = values[sd[:, :, None], pd[:, None, :]].mean((1, 2))
    seed_means, place_means = values.mean(-1), values.mean(0)
    return dict(mean=float(values.mean()),
        seed_differences={str(seed): float(value) for seed, value in zip(seeds, seed_means)},
        place_differences={str(place): float(value) for place, value in zip(places, place_means)},
        ci95_crossed=np.quantile(crossed, [.025, .975]).tolist(),
        ci95_seed_only=np.quantile(seed_means[sd].mean(-1), [.025, .975]).tolist(),
        ci95_place_only=np.quantile(place_means[pd].mean(-1), [.025, .975]).tolist(),
        positive_seed_count=int((seed_means > 0).sum()), seeds=len(seeds), places=len(places),
        bootstrap_draws=int(draws), bootstrap_seed=int(random_seed),
        estimand='mean training-seed difference of equal-place macro metrics; query-weighted within each place',
        scope='finite held-out place sample; no token-independence or arbitrary-building population claim')


def default_place(scene):
    return str(scene).strip('/').split('/')[0]


def default_institution(place):
    if place == 'harvard_c11':
        return 'Harvard_c11'
    if place.startswith('harvard'):
        return 'Harvard'
    if place.startswith('hotel'):
        return 'Hotel'
    if place.startswith('mit_'):
        return 'MIT'
    return place.split('_')[0]


def validate_rows(rows, seeds, place_map=None):
    if not rows:
        raise ValueError('evaluation input is empty')
    place_map = place_map or {}
    normalized, seen, pair_geometry, split_places = [], set(), {}, {}
    for index, original in enumerate(rows):
        missing = set(REQUIRED)-set(original)
        if missing:
            raise ValueError(f'row {index}: missing fields {sorted(missing)}')
        row = dict(original)
        if isinstance(row['seed'], bool) or int(row['seed']) != row['seed']:
            raise ValueError('seed must be an integer')
        row['seed'] = int(row['seed'])
        if row['seed'] not in seeds:
            raise ValueError(f'unexpected seed {row["seed"]}')
        row['method'], row['scene'], row['pair_id'] = map(str, [row['method'], row['scene'], row['pair_id']])
        row['split'] = 'val' if row['split'] == 'validation' else str(row['split'])
        row['place'] = str(place_map.get(row['scene'], default_place(row['scene'])))
        if not row['place'] or not row['method'] or not row['scene']:
            raise ValueError('method/scene/place must be nonempty')
        for name in ['q_index', 'k_index', 'n_valid', 'pck8_count', 'pck4_count']:
            value = row[name]
            if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value or value < 0:
                raise ValueError(f'row {index}: nonnegative integer required for {name}')
            row[name] = int(value)
        for name in FIELDS.values():
            if not math.isfinite(float(row[name])) or row[name] < 0:
                raise ValueError(f'row {index}: finite nonnegative {name} required')
            row[name] = float(row[name])
        if row['pck4_count'] > row['pck8_count'] or row['pck8_count'] > row['n_valid']:
            raise ValueError('PCK4 <= PCK8 <= n_valid is required')
        if row['n_valid'] == 0 and any(row[name] != 0 for name in FIELDS.values()):
            raise ValueError('zero-valid pair must have zero additive counts/sums')
        key = (row['method'], row['seed'], row['split'], row['scene'], row['pair_id'])
        if key in seen:
            raise ValueError(f'duplicate evaluation row: {key}')
        seen.add(key)
        pair_key = (row['split'], row['scene'], row['pair_id'])
        geometry = (row['q_index'], row['k_index'], row['n_valid'])
        if pair_key in pair_geometry and pair_geometry[pair_key] != geometry:
            raise ValueError(f'pair identities or n_valid differ across methods/seeds: {pair_key}')
        pair_geometry[pair_key] = geometry
        split_places.setdefault(row['place'], set()).add(row['split'])
        normalized.append(row)
    leaking = {place: sorted(splits) for place, splits in split_places.items() if len(splits) > 1}
    if leaking:
        raise ValueError(f'physical place occurs in multiple supplied splits: {leaking}')
    return sorted(normalized, key=lambda row: (row['split'], row['place'], row['scene'],
        row['pair_id'], row['method'], row['seed']))


def aggregate(rows, split, seeds, methods):
    picked = [row for row in rows if row['split'] == split and row['method'] in methods]
    if not picked:
        raise ValueError(f'no rows for split {split}')
    reference = None
    for method in methods:
        for seed in seeds:
            keys = {(r['scene'], r['pair_id'], r['q_index'], r['k_index'], r['n_valid'])
                    for r in picked if r['method'] == method and r['seed'] == seed}
            if not keys:
                raise ValueError(f'missing evaluation cell: {split}/{method}/{seed}')
            if reference is None:
                reference = keys
            elif keys != reference:
                raise ValueError(f'unpaired evaluation pairs/masks: {split}/{method}/{seed}')
    places = _canonical({row['place'] for row in picked})
    output, coverage = {}, {}
    for method in methods:
        output[method] = {metric: np.empty((len(seeds), len(places)), np.float64) for metric in FIELDS}
        coverage[method] = {}
        for si, seed in enumerate(seeds):
            coverage[method][str(seed)] = {}
            for pi, place in enumerate(places):
                part = [row for row in picked if row['method'] == method and row['seed'] == seed and row['place'] == place]
                count = sum(row['n_valid'] for row in part)
                if count == 0:
                    raise ValueError(f'no valid queries for {split}/{method}/{seed}/{place}')
                for metric, field in FIELDS.items():
                    output[method][metric][si, pi] = math.fsum(row[field] for row in part)/count
                coverage[method][str(seed)][place] = dict(n_valid=count, pairs=len(part),
                    zero_valid_pairs=sum(row['n_valid'] == 0 for row in part),
                    query_frames=len({(row['scene'], row['q_index']) for row in part}))
    return output, places, coverage


def leave_out(differences, seeds, places, groups):
    """Descriptive leave-group-out effects, with no new formal interval."""
    values = np.asarray(differences, np.float64)
    result = {}
    for group in _canonical(set(groups.values())):
        keep = [index for index, place in enumerate(places) if groups[place] != group]
        removed = [place for place in places if groups[place] == group]
        if not keep:
            result[str(group)] = dict(removed_places=removed, remaining_places=0, mean=None,
                                      note='all places removed; effect undefined')
            continue
        remaining = values[:, keep]
        result[str(group)] = dict(removed_places=removed, remaining_places=len(keep),
            mean=float(remaining.mean()),
            seed_differences={str(seed): float(value) for seed, value in zip(seeds, remaining.mean(-1))},
            descriptive_only=True)
    return result


def analyze_rows(rows, seeds=DEFAULT_SEEDS, candidate='normalized', baseline_candidates=('point', 'fixed'),
                 controls=None, place_map=None, institution_map=None, draws=DEFAULT_BOOTSTRAP,
                 random_seed=BOOTSTRAP_SEED, min_test_places=5):
    seeds = _canonical(list(seeds))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('the confirmation analysis requires exactly five unique training seeds')
    rows = validate_rows(rows, seeds, place_map)
    available = _canonical({row['method'] for row in rows})
    controls = _canonical(list(controls if controls is not None else set(available)-{candidate}))
    methods = [candidate]+controls
    if candidate not in available or len(set(methods)) != len(methods):
        raise ValueError('one candidate and distinct controls are required')
    if not set(baseline_candidates).issubset(controls):
        raise ValueError('validation baseline candidates must be included among controls')
    validation, val_places, val_coverage = aggregate(rows, 'val', seeds, methods)
    scores = {method: float(validation[method]['pck8'].mean()) for method in baseline_candidates}
    selected = max(baseline_candidates, key=scores.get)
    selection = dict(selected=selected, candidate=candidate, baseline_candidates=list(baseline_candidates),
        validation_macro_pck8=scores, rule='maximum equal-place macro validation PCK8 averaged over five seeds; ties use declared order',
        test_values_used=False, validation_places=val_places)
    test, places, test_coverage = aggregate(rows, 'test', seeds, methods)
    if len(places) < min_test_places:
        raise ValueError(f'expected at least {min_test_places} test places, got {len(places)}')
    institutions = {place: str((institution_map or {}).get(place, default_institution(place))) for place in places}
    means, per_place, comparisons = {}, {}, {}
    for method in methods:
        means[method] = {metric: float(matrix.mean()) for metric, matrix in test[method].items()}
        per_place[method] = {place: {metric: dict(mean=float(matrix[:, pi].mean()),
            seed_values={str(seed): float(value) for seed, value in zip(seeds, matrix[:, pi])})
            for metric, matrix in test[method].items()} for pi, place in enumerate(places)}
    for control in controls:
        comparisons[control] = {}
        for metric in FIELDS:
            difference = test[candidate][metric]-test[control][metric]
            statistics = crossed_bootstrap(difference, seeds, places, draws, random_seed)
            statistics['candidate'], statistics['control'] = candidate, control
            statistics['favorable_direction'] = 'positive' if metric.startswith('pck') else 'negative'
            statistics['leave_one_place_out'] = leave_out(difference, seeds, places, {place: place for place in places})
            statistics['leave_institution_out'] = leave_out(difference, seeds, places, institutions)
            comparisons[control][metric] = statistics
    primary = comparisons[selected]['pck8']
    baseline_ce = means[selected]['ce']
    ce_relative_change = ((means[candidate]['ce']-baseline_ce)/baseline_ce if baseline_ce > 0
                          else (0. if means[candidate]['ce'] == 0 else None))
    checks = dict(pck8_gain_at_least_1pp=primary['mean'] >= .01,
        crossed_ci_lower_above_zero=primary['ci95_crossed'][0] > 0,
        positive_in_at_least_4_of_5_seeds=primary['positive_seed_count'] >= 4,
        ce_relative_worsening_at_most_2percent=ce_relative_change is not None and ce_relative_change <= .02)
    gate = dict(selected_baseline=selected, candidate=candidate, passed=all(checks.values()), checks=checks,
        pck8_difference=primary['mean'], ci95_crossed=primary['ci95_crossed'],
        positive_seed_count=primary['positive_seed_count'], ce_relative_change=ce_relative_change,
        ce_guardrail='point-estimate tolerance only; not a formal statistical noninferiority claim',
        practical_thresholds=dict(pck8_absolute_gain=.01, ce_relative_worsening=.02),
        scope='five observed held-out places; small scene sample, no independent-token CI')
    return dict(baseline_selection=selection, primary_gate=gate, test_macro=means,
        test_per_place=per_place, comparisons=comparisons,
        validation_macro={method: {metric: float(matrix.mean()) for metric, matrix in validation[method].items()} for method in methods},
        coverage=dict(validation=val_coverage, test=test_coverage),
        identities=dict(seeds=seeds, test_places=places, validation_places=val_places, institutions=institutions),
        supplied_splits_checked=sorted({row['split'] for row in rows}),
        missing_training_split_note='train/test independence must also be checked in acquisition/training manifests when train rows are not supplied',
        row_count=len(rows))


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'audit_config.json').exists():
        raise FileExistsError('preserve completed analyses; choose a new output directory')
    inputs = [Path(path) for path in args.inputs]
    rows = []
    hashes, input_records = {}, []
    for path in inputs:
        parsed, audit = load_evaluation_file(path)
        rows.extend(parsed)
        hashes[str(path)] = audit['sha256']
        input_records.append(audit)
    place_map = json.loads(Path(args.place_map).read_text()) if args.place_map else None
    institution_map = json.loads(Path(args.institution_map).read_text()) if args.institution_map else None
    seeds = [int(seed) for seed in args.seeds.split(',')]
    baseline_candidates = args.baseline_candidates.split(',')
    controls = args.controls.split(',') if args.controls else None
    config = dict(config=vars(args), input_sha256=hashes, inputs=input_records,
        input_row_count=len(rows), source_sha256=sha256(__file__),
        place_map=place_map, institution_map=institution_map,
        default_place_rule='first scene path component',
        aggregation='sum pair numerators / sum valid queries within seed/place, then equal place macro, then equal seed mean',
        baseline_selection='validation PCK8 only; ordered candidates point,fixed by default; all test comparisons reported',
        bootstrap='paired crossed seed x place draws, identities sorted, common draws across all metrics/controls',
        intervals='percentile 95%; 20000 default draws; only five observed held-out places; no token bootstrap',
        sensitivity='leave-one-place and Harvard_c11/Hotel/MIT leave-group-out point estimates, no new formal CI',
        guardrail='PCK8 +1pp, crossed CI lower >0, >=4/5 seeds positive, CE point-estimate relative worsening <=2%',
        ignored_extra_row_fields='unused fields are never selection inputs')
    write_json(out/'audit_config.json', config)
    (out/'source').mkdir(exist_ok=True)
    (out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    started = time.time()
    result = analyze_rows(rows, seeds, args.candidate, baseline_candidates, controls,
        place_map, institution_map, args.bootstrap, args.random_seed, args.min_test_places)
    write_json(out/'baseline_selection.json', result['baseline_selection'])
    result.update(audit_config_sha256=sha256(out/'audit_config.json'), elapsed_seconds=time.time()-started)
    write_json(out/'summary.json', result)
    sd, pd = bootstrap_draws(len(seeds), len(result['identities']['test_places']), args.bootstrap, args.random_seed)
    np.savez_compressed(out/'bootstrap_draws.npz', seed_draws=sd.astype(np.int16), place_draws=pd.astype(np.int16))
    for path, checksum in hashes.items():
        if sha256(path) != checksum:
            raise ValueError(f'input changed during analysis: {path}')
    lines = ['# Round 5 paired seed/place analysis', '',
        f"Validation-selected position baseline: **{result['baseline_selection']['selected']}**. "
        f"Primary practical gate passed: **{result['primary_gate']['passed']}**.", '',
        'Each place is weighted equally after aggregating its valid queries. '
        'Intervals resample model seeds and physical places as crossed factors; '
        'tokens and overlapping pairs are not treated as independent datasets. '
        'Only five held-out places support the default interval, so population generalization remains limited.', '',
        '| Method | CE | PCK8 | PCK4 | Argmax EPE | Soft EPE |',
        '|---|---:|---:|---:|---:|---:|']
    for method, metrics in result['test_macro'].items():
        lines.append('| '+method+' | '+' | '.join(f'{metrics[metric]:.6f}' for metric in FIELDS)+' |')
    lines += ['', 'All seed/place metrics, candidate-minus-control differences, 95% crossed intervals, '
        'leave-place/group-out effects and coverage counts are in summary.json. '
        'Leave-out analyses are descriptive sensitivity checks, not additional confirmatory tests. '
        'CE worsening is an empirical 2% guardrail, not a formal noninferiority test.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['primary_gate']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out', required=True)
    parser.add_argument('--candidate', default='normalized')
    parser.add_argument('--baseline-candidates', default='point,fixed')
    parser.add_argument('--controls', default='')
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument('--bootstrap', type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument('--random-seed', type=int, default=BOOTSTRAP_SEED)
    parser.add_argument('--place-map')
    parser.add_argument('--institution-map')
    parser.add_argument('--min-test-places', type=int, default=5)
    main(parser.parse_args())
