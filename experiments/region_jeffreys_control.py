"""One matched learned-divergence control; reuse completed regional machinery."""
import argparse
from pathlib import Path
import time
from experiments import region_shape_attribution as parent
from experiments.region_geometry_controls import dependencies
from experiments.round65_common import ROOT, SEEDS, read, write, sha
from models.region_jeffreys_control import JeffreysMatcher, ROLES

OUT = ROOT / 'results/region_jeffreys_control/attempt1'
PLAN = ROOT / 'results/region_jeffreys_control/preparation/plan.json'
MODULE = 'experiments.region_jeffreys_control'


def freeze():
    old = parent.check()
    assert read(parent.OUT / 'review.json')['scientific_review_complete']
    assert not (OUT / 'lock.json').exists()
    units = read(OUT / 'units.json')
    assert units['passed']
    sources = dict(old['source_sha256'])
    files = dependencies([Path(__file__), ROOT / 'experiments/region_jeffreys_report.py',
                          ROOT / 'tests/test_region_jeffreys_control.py'])
    for path in files + [PLAN, OUT / 'units.json']:
        sources[str(path)] = sha(path)
    for name, digest in units['source_sha256'].items():
        assert sha(name) == digest
    prior = dict(old['prior_sha256'])
    for name in ('lock.json', 'result.json', 'review.json'):
        path = parent.OUT / name
        prior[str(path)] = sha(path)
    write(OUT / 'lock.json', dict(created_unix=time.time(), source_sha256=sources,
        prior_sha256=prior, cache_sha256=old['cache_sha256'],
        settings=dict(parent.engine.SETTINGS, roles=ROLES), seeds=SEEDS, roles=ROLES,
        conditions=parent.CONDITIONS, plan_sha256=sha(PLAN), new_fits=3,
        candidate_changed=False, fresh_confirmation=False, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('REGION_JEFFREYS_FROZEN', sha(OUT / 'lock.json'), flush=True)


def bind():
    parent.OUT, parent.PLAN, parent.ROLES, parent.MODULE = OUT, PLAN, ROLES, MODULE
    parent.FixedIsotropicMatcher = JeffreysMatcher
    parent.bind()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'queue', 'preflight', 'train', 'selections', 'export', 'report'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--role', choices=ROLES)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        bind()
        if args.action == 'train':
            parent.engine.train(args.role, args.seed)
        elif args.action == 'export':
            parent.export(args.seed)
        elif args.action == 'report':
            from experiments.region_jeffreys_report import report
            report()
        else:
            getattr(parent, args.action)()
