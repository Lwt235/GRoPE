"""One12GiB source-geometry worker under the user's standing spare-VRAM authorization."""
import argparse
import json
import os
import subprocess
import sys
import time

from experiments.gaussian_scene_data import DATA,ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',required=True)
    parser.add_argument('--owned-peer',type=int,help='Only share with this verified same-goal CUDA training PID')
    parser.add_argument('--require-idle',action='store_true')
    parser.add_argument('module',choices=['experiments.source_geometry_build','experiments.source_geometry_prepare','experiments.source_geometry_train','experiments.source_geometry_diagnostics','experiments.source_geometry_cost'])
    parser.add_argument('args',nargs=argparse.REMAINDER);a=parser.parse_args()
    checks=[]
    for _ in range(2):
        row=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(', ')
        uuid=row[0];used,free,util=map(int,row[1:]);checks.append(dict(uuid=uuid,used_MiB=used,free_MiB=free,util_percent=util))
        if free<22528:raise SystemExit('Selected GPU no longer meets the occupancy rule; no job launched: '+json.dumps(checks))
        if a.require_idle and (used>512 or util>5):raise SystemExit('GPU no longer idle; no job launched')
        if a.owned_peer is not None:
            from pathlib import Path
            peer=Path('/proc')/str(a.owned_peer)
            if not peer.exists():raise SystemExit('Owned peer ended; no job launched')
            if peer.stat().st_uid!=os.getuid() or b'experiments.source_geometry_train' not in (peer/'cmdline').read_bytes() or peer.joinpath('cwd').resolve()!=ROOT:
                raise RuntimeError('Peer is not a verified same-goal training process')
            apps=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-compute-apps=gpu_uuid,pid',
                                          '--format=csv,noheader,nounits'],text=True,timeout=15)
            pids={int(parts[1].strip()) for line in apps.splitlines() if len(parts:=line.split(','))==2 and parts[0].strip()==uuid}
            checks[-1]['observed_compute_pids']=sorted(pids)
            if not pids.issubset({a.owned_peer}):raise SystemExit('External or additional worker appeared; no job launched')
        time.sleep(2)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(DATA/'torch_extensions'))
    env['PATH']='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+env['PATH']
    runner='''import sys,runpy
import numpy as np
import torch
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
sys.argv=sys.argv[1:]
runpy.run_module(sys.argv[0],run_name="__main__")
'''
    print(json.dumps(dict(launch=a.module,args=a.args,gpu_checks=checks,shared_worker=not a.require_idle,memory_cap_GiB=12,owned_peer_pid=a.owned_peer,
        authorization='Standing spare-VRAM authorization; at most two foreign-shared cards, each process capped12GiB; independently verified same-task peers on otherwise-owned GPUs',
        timing_scope='Shared GPU; not an exclusive speed benchmark')),flush=True)
    result=subprocess.run([sys.executable,'-u','-c',runner,a.module,*a.args],cwd=ROOT,env=env)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
