"""Resume the same cohort after a validated raw-input runtime correction."""
import argparse
from pathlib import Path
import time

from experiments import region_confirmation as original
from experiments import region_confirmation_prepare_runtime as runtime
from experiments.region_confirmation_common import OUT, ROOT, ROLES, SEEDS, read, sha, preserved, check

RUNTIME = runtime.RUNTIME
MODULE = 'experiments.region_confirmation_resume'


def checked():
    lock = check()
    plan = RUNTIME / 'plan.json'
    assert sha(plan) == (RUNTIME / 'plan.sha256').read_text().strip()
    for path, digest in read(plan)['source_sha256'].items():
        assert sha(path) == digest, path
    validation = read(RUNTIME / 'old_validation_check.json')
    assert validation['passed'] and validation['source_sha256'] == sha(Path(runtime.__file__))
    return lock


def data_checked():
    checked()
    record = read(RUNTIME / 'data_runtime_identity.json')
    assert record['source_sha256'] == sha(Path(runtime.__file__))
    assert record['data_manifest_sha256'] == sha(OUT / 'data_manifest.json')
    assert record['old_validation_check_sha256'] == sha(RUNTIME / 'old_validation_check.json')


def queue():
    checked()
    state = read(OUT / 'queue_status.json')
    assert state['state'] == 'stopped' and state['stage'] == 'prepare'
    stat = Path(f"/proc/{state['pid']}/stat")
    if stat.exists():
        fields = stat.read_text().rsplit(')', 1)[1].split()
        assert fields[19] != state['start_ticks'] or fields[0] == 'Z'
    for item in state['active']:
        cmd = Path(f"/proc/{item['pid']}/cmdline")
        if cmd.exists():
            assert b'experiments.region_confirmation\x00prepare' not in cmd.read_bytes()
    assert not (RUNTIME / 'execution_identity.json').exists()
    preserved(RUNTIME / 'queue_binding.json', dict(adapter_sha256=sha(__file__),
        original_scheduler_sha256=sha(Path(original.__file__)),
        original_status_sha256=sha(OUT / 'queue_status.json'), created_unix=time.time(),
        scope='Unchanged two-GPU scheduler, new controller logs. Preflight/download identities are reused; only preparation uses the validated runtime adapter. Model export/report are unchanged.'))
    original.OUT, original.MODULE, original.check = RUNTIME, MODULE, checked
    original.queue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('queue', 'preflight', 'download', 'prepare', 'export', 'report'))
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    checked()
    if args.action == 'queue':
        queue()
    elif args.action == 'preflight':
        result = read(OUT / 'preflight.json')
        assert result['passed'] and len(result['records']) == 24
        assert result['evaluation_lock_sha256'] == sha(OUT / 'evaluation_lock.json')
        print('REUSED_UNCHANGED_24_MODEL_GPU_REPLAY', flush=True)
    elif args.action == 'download':
        done = read(OUT / 'download_complete.json')
        assert done['manifest_sha256'] == sha(OUT / 'download_manifest.json')
        assert done['evaluation_lock_sha256'] == sha(OUT / 'evaluation_lock.json')
        print('REUSED_COMPLETE_FIXED_DEPTH_DOWNLOAD', done['files'], flush=True)
    elif args.action == 'prepare':
        runtime.prepare()
    elif args.action == 'export':
        data_checked()
        original.export(args.role, args.seed)
    elif args.action == 'report':
        data_checked()
        from experiments.region_confirmation_report import report
        original.engine.configure()
        report()
