"""Run the frozen seed replication with the validated memmap transfer path."""
import argparse
from pathlib import Path
import time

from experiments import native_seed_replication as experiment
from experiments.native_memmap_runtime import install
from experiments.round65_common import ROOT, read, sha, write

OUT = experiment.OUT / 'runtime1'
MODULE = 'experiments.native_seed_replication_runtime'


def freeze():
    experiment.check()
    assert not (OUT / 'lock.json').exists()
    assert not (experiment.OUT / 'execution_identity.json').exists()
    sources = [Path(__file__), ROOT / 'experiments/native_memmap_runtime.py',
        OUT / 'plan.json',
        ROOT / 'results/native_seed_replication/preparation/priority_update_20261105_03.json',
        ROOT / 'results/native_shape_attribution/attempt1/review.json',
        ROOT / 'results/native_shape_attribution/attempt1/resource_handoff1/validation.json',
        ROOT / 'results/native_confirmation_runtime_validation/attempt1/review.json']
    write(OUT / 'lock.json', dict(created_unix=time.time(),
        source_sha256={str(p.resolve()): sha(p) for p in sources},
        scientific_lock_sha256=sha(experiment.OUT / 'lock.json'),
        scientific_plan_sha256=sha(experiment.PLAN), science_changed=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('NATIVE_REPLICATION_RUNTIME_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    experiment.check()
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    assert lock['scientific_lock_sha256'] == sha(experiment.OUT / 'lock.json')
    assert lock['scientific_plan_sha256'] == sha(experiment.PLAN)
    for name, digest in lock['source_sha256'].items():
        assert sha(name) == digest, name


def run(action, role=None, seed=None):
    check()
    install(experiment.runner)
    if action == 'queue':
        assert not (OUT / 'execution_binding.json').exists()
        write(OUT / 'execution_binding.json', dict(created_unix=time.time(),
            runtime_module=MODULE, runtime_lock_sha256=sha(OUT / 'lock.json'),
            scientific_lock_sha256=sha(experiment.OUT / 'lock.json'),
            original_queue_source_sha256=sha(Path(experiment.__file__)),
            source_sha256=sha(__file__), maximum_gpu_jobs=4,
            model_optimizer_data_and_budget_unchanged=True,
            factored_relation_projection_enabled=False))
        experiment.MODULE = MODULE
        experiment.queue()
    elif action == 'train':
        experiment.bind()
        raise SystemExit(experiment.runner.fit(role, seed, experiment.OUT))
    elif action == 'export':
        experiment.bind()
        experiment.runner.export(seed, experiment.OUT)
    elif action == 'report':
        from experiments.native_seed_replication_report import report
        report()
    else:
        getattr(experiment, action)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'check', 'queue', 'preflight', 'train', 'selections', 'export', 'report'))
    parser.add_argument('--role', choices=experiment.ROLES)
    parser.add_argument('--seed', type=int, choices=experiment.SEEDS)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'check':
        check()
        print('NATIVE_REPLICATION_RUNTIME_CHECKED', flush=True)
    else:
        run(args.action, args.role, args.seed)
