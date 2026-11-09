"""A finite closing control using the existing paired-fit execution engine."""
import argparse
import ast
from pathlib import Path
import time

import numpy as np
import torch

from experiments import mean_precision as engine
from experiments.round65_common import ROOT, CONFIG, SEEDS, VARIANTS, read, write, sha
from models.normalized_precision_relation import (
    ROLES, NormalizedPrecisionMatcher, NormalizedPrecisionNative)

OUT = ROOT / 'results/normalized_precision/attempt1'
PLAN = ROOT / 'docs/normalized_precision_plan.md'
UNITS = ROOT / 'results/normalized_precision/preparation/cpu_checks.json'
PROVENANCE = ROOT / 'results/normalized_precision/preparation/rationale.json'
CASES = ('region', 'native512')
REGION_PRIOR = {k: engine.REGION_PRIOR[k] for k in ('relative5', 'distance_mlp', 'trace_isotropic5')}
REGION_PRIOR['mean_precision5'] = ROOT / 'results/mean_precision/attempt1/region'
NATIVE_PRIOR = {'native512': {k: engine.NATIVE_PRIOR['native512'][k]
                             for k in ('relative5', 'distance_mlp', 'trace_isotropic5')}}
NATIVE_PRIOR['native512']['mean_precision5'] = ROOT / 'results/mean_precision/attempt1/native512'
REGION_SETTINGS = dict(engine.REGION_SETTINGS, roles=ROLES)


def configure():
    """Configure a process-local engine; historical source and artifacts stay intact."""
    for name, value in dict(OUT=OUT, PLAN=PLAN, UNITS=UNITS, PROVENANCE=PROVENANCE,
            MODULE='experiments.normalized_precision', REGION_ROLES=ROLES,
            NATIVE_ROLES=ROLES, REGION_PRIOR=REGION_PRIOR, NATIVE_PRIOR=NATIVE_PRIOR,
            REGION_SETTINGS=REGION_SETTINGS, MeanPrecisionMatcher=NormalizedPrecisionMatcher,
            MeanPrecisionNative=NormalizedPrecisionNative).items():
        setattr(engine, name, value)


def source_files():
    pending = [Path(__file__), ROOT / 'experiments/normalized_precision_report.py',
               ROOT / 'tests/test_normalized_precision.py', ROOT / 'tests/test_normalized_precision_report.py']
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module] + [node.module + '.' + a.name for a in node.names]
            else:
                continue
            for name in modules:
                candidate = ROOT / (name.replace('.', '/') + '.py')
                if name.split('.')[0] in ('models', 'experiments') and candidate.is_file():
                    pending.append(candidate)
    return sorted(found)


def freeze():
    configure()
    assert not (OUT / 'lock.json').exists()
    units = read(UNITS)
    assert units['passed'] and units['plan_sha256'] == sha(PLAN)
    for path, digest in units['source_sha256'].items():
        assert sha(path) == digest, path
    prior = {}
    for case in CASES:
        mapping = REGION_PRIOR if case == 'region' else NATIVE_PRIOR[case]
        conditions = engine.region.CONDITIONS if case == 'region' else VARIANTS
        for role, root in mapping.items():
            review_root = root.parent if role in ('trace_isotropic5', 'mean_precision5') else root
            review = read(review_root / 'review.json')
            assert review.get('scientific_review_complete') or review.get('scientific_review_pending') is False
            evidence = 'result.json' if (review_root / 'result.json').exists() else 'metrics/summary.json'
            paths = [root / 'lock.json', root / 'selected.json', review_root / evidence, review_root / 'review.json']
            for seed in SEEDS:
                paths += [root / 'runs' / role / str(seed) / name for name in ('run.json', 'history.json', 'best.pt')]
                paths += [root / 'evaluation' / role / str(seed) / name
                          for name in ('record.json',) + tuple(c + '.npz' for c in conditions)]
            prior.update({str(p): sha(p) for p in paths})
    for root in (REGION_PRIOR['distance_mlp'], NATIVE_PRIOR['native512']['distance_mlp']):
        path = root / 'data/manifest.json'
        prior[str(path)] = sha(path)
    cache_lock = ROOT / 'results/region_shape_attribution/attempt1/lock.json'
    prior[str(cache_lock)] = sha(cache_lock)
    caches = read(cache_lock)['cache_sha256']
    for path, digest in caches.items():
        assert sha(path) == digest, path
    settings = {c: dict(roles=ROLES, seeds=SEEDS, settings=REGION_SETTINGS if c == 'region' else CONFIG,
                       model_train_tokens=64 if c == 'region' else 512, validation_variant='base') for c in CASES}
    paths = source_files() + [PLAN, UNITS, PROVENANCE]
    write(OUT / 'lock.json', dict(created_unix=time.time(), source_sha256={str(p): sha(p) for p in paths},
        prior_sha256=prior, cache_sha256=caches, cases=settings, new_fits=6,
        new_prediction_groups=24, reused_prediction_groups=96, comparisons=80,
        bootstrap_replicates=10000, bootstrap_base_seed=87001,
        maximum_gpu_jobs=6, maximum_shared_gpu_jobs=2, shared_minimum_free_mib=20480,
        allocator_cap_bytes=engine.CAP_BYTES, candidate='relative5', automatic_promotion=False,
        fresh_confirmation=False, official_test_access=False, goal_complete=False))
    digest = sha(OUT / 'lock.json')
    (OUT / 'lock.sha256').write_text(digest + '\n')
    for case in CASES:
        path = OUT / case
        write(path / 'lock.json', dict(root_lock_sha256=digest, case=case, **settings[case]))
        (path / 'lock.sha256').write_text(sha(path / 'lock.json') + '\n')
        if case != 'region':
            (path / 'data').symlink_to((NATIVE_PRIOR[case]['distance_mlp'] / 'data').resolve(), target_is_directory=True)
    print('NORMALIZED_PRECISION_FROZEN', digest, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'queue', 'preflight', 'train', 'selections', 'seal', 'export', 'report'))
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    configure()
    if args.action == 'freeze':
        freeze()
    elif args.action in ('queue', 'seal'):
        getattr(engine, args.action)()
    elif args.action == 'report':
        from experiments.normalized_precision_report import report
        report()
    elif args.action in ('preflight', 'selections'):
        getattr(engine, args.action)(args.case)
    else:
        engine.bind(args.case)
        engine.cuda_setup()
        if args.action == 'train':
            if args.case == 'region':
                engine.region.engine.train(args.role, args.seed)
            else:
                raise SystemExit(engine.native.fit(args.role, args.seed, OUT / args.case))
        else:
            engine.selected_globally()
            if args.case == 'region':
                engine.region_export(args.seed)
            else:
                engine.native.export(args.seed, OUT / args.case)


if __name__ == '__main__':
    main()
