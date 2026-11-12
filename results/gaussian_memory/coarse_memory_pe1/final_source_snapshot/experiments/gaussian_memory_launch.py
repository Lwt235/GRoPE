"""Launch one explicitly selected idle GPU job without disturbing other users."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

from experiments.gaussian_memory_setup import DATA, ROOT, sha256


def state_center_lock(module, args):
    """Serialize a fit reached by both the queued and parallel controllers.

    The finite coarse comparison can also be reached by an owned-GPU peer
    dispatcher. Serialize that exact fit and reuse a fully validated completion.
    """
    if module == 'experiments.gaussian_memory_coarse_train' and args and args[0] == 'train':
        kind=args[args.index('--kind')+1];seed=int(args[args.index('--seed')+1])
        if kind not in ('center','sphere','gaussian') or seed not in (20261110,20261111,20261112):raise ValueError('Unknown finite coarse fit')
        root=ROOT/'results/gaussian_memory/coarse_memory_pe1';folder=root/f'{kind}_{seed}'
        lock=(root/f'{kind}_{seed}.lock').open('a')
        print(json.dumps({'waiting_for_fit_lock':str(root/f'{kind}_{seed}.lock')}),flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'status.json').exists() and (root/'evaluation'/f'{kind}_{seed}.json').exists():
            status=json.loads((folder/'status.json').read_text());ev=json.loads((root/'evaluation'/f'{kind}_{seed}.json').read_text());identity=sha256(root/'plan.json')
            if status['complete'] and ev['complete'] and status['steps']==2000 and status['plan_sha256']==ev['plan_sha256']==identity and sha256(folder/'step_2000.pt')==status['checkpoint_sha256']==ev['checkpoint_sha256'] and sha256(ev['prediction_path'])==ev['prediction_sha256']:
                print(json.dumps({'already_complete':True,'kind':kind,'seed':seed,'no_duplicate_training':True}),flush=True);raise SystemExit(0)
        return lock
    if module != 'experiments.gaussian_memory_state_train' or not args or args[0] != 'train':
        return None
    if '--kind' not in args or args[args.index('--kind') + 1] != 'soft_center':
        return None
    if '--seed' not in args:
        raise ValueError('Concurrent Center scheduling requires an explicit seed.')
    seed = int(args[args.index('--seed') + 1])
    path = ROOT / 'results/gaussian_memory/state_read1' / f'center_train_{seed}.lock'
    lock = path.open('a')
    print(json.dumps({'waiting_for_fit_lock': str(path)}), flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True)
    p.add_argument('--allow-shared', action='store_true', help='one previously authorized shared worker only')
    p.add_argument('--allow-owned-peer', type=int, help='coarse fit only: share with this verified same-user coarse-training PID, cap12GiB')
    p.add_argument('--wait-for-gpu', action='store_true', help='recheck every 30 seconds, up to 30 minutes')
    p.add_argument('module')
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.allow_owned_peer is not None:
        if a.allow_shared or a.module!='experiments.gaussian_memory_coarse_train':raise ValueError('Owned peer mode is limited to the finite coarse fits')
        peer=f'/proc/{a.allow_owned_peer}'
        if os.path.exists(peer):
            with open(peer+'/cmdline','rb') as f:peer_command=f.read()
            if os.stat(peer).st_uid!=os.getuid() or b'experiments.gaussian_memory_coarse_train' not in peer_command:raise ValueError('Peer is not our coarse-training process')
    fit_lock = state_center_lock(a.module, a.args)
    deadline = time.monotonic() + 1800
    while True:
        checks = []
        for _ in range(2):
            raw = subprocess.check_output(['nvidia-smi', '-i', a.gpu,
                '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                text=True, timeout=15).strip().split(', ')
            uuid, used, free, util = raw[0], *map(int, raw[1:])
            checks.append({'uuid': uuid, 'used_MiB': used, 'free_MiB': free, 'util_percent': util})
            if a.allow_owned_peer is not None:
                apps=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
                pids={int(parts[1].strip()) for line in apps.splitlines() if len(parts:=line.split(','))==2 and parts[0].strip()==uuid}
                checks[-1]['observed_compute_pids']=sorted(pids)
                acceptable=free>=20000 and pids.issubset({a.allow_owned_peer})
            else:acceptable = (free >= 20000 and util <= 60) if a.allow_shared else (used <= 512 and free >= 40000 and util <= 5)
            if not acceptable:
                break
            time.sleep(2)
        if acceptable and len(checks) == 2:
            break
        if not a.wait_for_gpu or time.monotonic() > deadline:
            raise SystemExit('Selected GPU no longer meets the occupancy rule; no job launched: ' + json.dumps(checks))
        print(json.dumps({'resource_wait': a.module, 'last_checks': checks}), flush=True)
        time.sleep(30)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MAX_JOBS='4', TORCH_CUDA_ARCH_LIST='8.6',
               TORCH_EXTENSIONS_DIR=str(DATA / 'torch_extensions'))
    env['PATH'] = '/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:' + env['PATH']
    print(json.dumps({'launch': a.module, 'args': a.args, 'gpu_checks': checks,
                      'shared_worker': a.allow_shared, 'owned_peer_pid':a.allow_owned_peer, 'memory_cap_GiB': 12 if a.allow_shared or a.allow_owned_peer is not None else None}), flush=True)
    if a.allow_shared or a.allow_owned_peer is not None:
        runner = '''import sys, runpy
import numpy as np
import torch
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(12 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__")
'''
        command = [sys.executable, '-u', '-c', runner, a.module, *a.args]
    else:
        command = [sys.executable, '-u', '-m', a.module, *a.args]
    result = subprocess.run(command, cwd=ROOT, env=env)
    raise SystemExit(result.returncode)


if __name__ == '__main__': main()
