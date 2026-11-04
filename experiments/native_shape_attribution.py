"""Three paired native-GS controls; reuse the existing fit/audit/scheduler."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from experiments import round65_run as runner, round65_audit as audit, round65_data as data_module
from experiments import round73_native_relative5 as parent
from experiments.round65_common import ROOT, CONFIG, SEEDS, VARIANTS, read, write, sha
from models.native_shape_attribution import NativeFixedIsotropic, ROLES

NATIVE = ROOT / 'results/round73_native_relative5/attempt1'
OLD = parent.OLD
REGION = ROOT / 'results/region_shape_attribution/attempt1'
OUT = ROOT / 'results/native_shape_attribution/attempt1'
PLAN = ROOT / 'results/native_shape_attribution/preparation/plan.json'
MODULE = 'experiments.native_shape_attribution'


def freeze():
    parent.check()
    assert not (OUT / 'lock.json').exists()
    assert read(OUT / 'units.json')['passed'] and read(REGION / 'review.json')['scientific_review_complete']
    original = read(NATIVE / 'lock.json')
    sources = dict(original['source_sha256'])
    files = parent.local_dependencies([Path(__file__), ROOT / 'experiments/native_shape_report.py',
                                      ROOT / 'tests/test_native_shape_attribution.py'])
    for path in files + [PLAN, OUT / 'units.json']:
        sources[str(path)] = sha(path)
    prior = {str(OLD / path): digest for path, digest in original['prior_sha256'].items()}
    for path in [NATIVE / name for name in ('lock.json', 'result.json', 'review.json', 'selected.json', 'data/manifest.json')]:
        prior[str(path)] = sha(path)
    for seed in SEEDS:
        for name in ('run.json', 'history.json', 'best.pt'):
            path = NATIVE / 'runs/relative5' / str(seed) / name
            prior[str(path)] = sha(path)
    for path in (REGION / 'review.json', REGION / 'result.json', REGION / 'lock.json'):
        prior[str(path)] = sha(path)
    for path, digest in read(OUT / 'units.json')['source_sha256'].items():
        assert sha(path) == digest
    write(OUT / 'lock.json', dict(created_unix=time.time(), config=CONFIG, roles=ROLES, seeds=SEEDS,
        variants=VARIANTS, source_sha256=sources, prior_sha256=prior,
        plan_sha256=sha(PLAN), new_fits=3, new_groups=18, reused_groups=108,
        candidate_changed=False, fresh_confirmation=False, official_test_access=False, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    (OUT / 'data').symlink_to((OLD / 'data').resolve(), target_is_directory=True)
    print('NATIVE_SHAPE_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check(out=OUT):
    assert Path(out) == OUT
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    assert lock['config'] == CONFIG and lock['roles'] == list(ROLES)
    assert lock['seeds'] == list(SEEDS) and lock['variants'] == list(VARIANTS)
    for group in ('source_sha256', 'prior_sha256'):
        for path, digest in lock[group].items():
            assert sha(path) == digest, path
    assert (OUT / 'data').resolve() == (OLD / 'data').resolve()
    return lock


def bind():
    check()
    for module in (runner, audit):
        module.NativeGSClassifier, module.ROLES, module.SEEDS = NativeFixedIsotropic, ROLES, SEEDS
        module.checked = check
    data_module.checked = check


def preflight():
    bind()
    runner.preflight(OUT / 'preflight1')
    result = read(OUT / 'preflight1/result.json')
    result.update(lock_sha256=sha(OUT / 'lock.json'), formal_fit_updates=0,
                  synthetic_optimizer_updates_per_role=1)
    write(OUT / 'cuda_preflight.json', result)


def selections():
    bind()
    audit.selections(OUT)
    selected = read(OUT / 'selected.json')
    selected.pop('all_fifteen_selected_before_development_evaluation')
    selected['all_three_selected_before_development_evaluation'] = True
    pairing, operators = [], []
    for seed in SEEDS:
        current, old = OUT / 'runs' / ROLES[0] / str(seed), NATIVE / 'runs/relative5' / str(seed)
        assert read(current / 'run.json')['initial_shared_sha256'] == read(old / 'run.json')['initial_shared_sha256']
        streams = lambda path: [r['stream_sha256'] for r in read(path / 'history.json')['training']]
        assert streams(current) == streams(old)
        pairing.append(dict(seed=seed,all120_epoch_streams_match_relative5=True))
        model = NativeFixedIsotropic()
        model.load_state_dict(torch.load(current / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
        c, s, _ = runner.synthetic_inputs(1, 9, 'cpu')
        distance = audit.independent_relation(c[0].numpy(), s[0].numpy())[..., 0]
        radial = np.log1p(distance ** 2)
        constant = np.full_like(distance, np.log(4.))
        features = np.stack((distance, radial, radial, constant, constant), -1)
        expected = audit.independent_projection(features, model.position)
        with torch.inference_mode():
            actual = model.position(c, s)[0].numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        operators.append(dict(seed=seed, maximum_error=float(abs(actual - expected).max())))
    selected.update(old_candidate_pairing=pairing, independent_radial_operator=operators,
                    original_R73_criterion_unchanged=True, goal_complete=False)
    write(OUT / 'selected.json', selected)


def queue():
    check()
    assert read(REGION / 'queue_status.json')['state'] == 'complete'
    assert not (OUT / 'queue_binding.json').exists()
    write(OUT / 'queue_binding.json', dict(adapter_sha256=sha(Path(__file__)),
        original_scheduler_sha256=sha(Path(parent.__file__)),
        scope='Reuse the unchanged R73 three-seed scheduler, idle-GPU checks and explicit exit75 resumption; only output root, child module and identity checker are bound to this experiment.'))
    parent.OUT, parent.MODULE, parent.check = OUT, MODULE, check
    parent.queue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'queue', 'preflight', 'train', 'selections', 'export', 'report'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action in ('train', 'export'):
        bind()
        if args.action == 'train':
            raise SystemExit(runner.fit(ROLES[0], args.seed, OUT))
        runner.export(args.seed, OUT)
    elif args.action == 'report':
        from experiments.native_shape_report import report
        report()
    else:
        globals()[args.action]()
