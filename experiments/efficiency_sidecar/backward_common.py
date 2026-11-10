"""Resources and identities for the bounded backward implementation study."""
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.efficiency_sidecar.benchmark_gpu import resource_check
from experiments.round65_common import ROOT, read, write, sha

OUT=ROOT/'results/efficiency_sidecar/backward1'


def initialize(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/f'{stage}_execution.json').exists():raise FileExistsError(stage)
    for filename in ('result.json','dependencies.json'):
        for p,d in read(ROOT/'results/efficiency_sidecar/cpu1'/filename)['sources'].items():
            assert sha(ROOT/p)==d,p
    queue=read(ROOT/'results/native_source_diversity/attempt1/queue_status.json')
    uuid=os.environ['CUDA_VISIBLE_DEVICES']
    peers={x['pid']:x['start_ticks'] for x in queue['active'] if x['gpu']['uuid']==uuid}
    assert peers
    snapshot=resource_check(uuid,shared_peers=peers)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.ones(1,device='cuda').sum().item()
    resource_check(uuid,initialized=True,shared_peers=peers)
    r=dict(stage=stage,pid=os.getpid(),started_unix=time.time(),gpu_uuid=uuid,peer_start_ticks=peers,
        gpu_name=torch.cuda.get_device_name(),torch_version=torch.__version__,allocator_cap_bytes=6*2**30,
        sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        plan_sha256=sha(Path(__file__).parent/'BACKWARD_PLAN.md'),shared_gpu=True,
        scientific_models_updated=False,resources=[snapshot],measurements=[])
    write(OUT/f'{stage}_execution.json',r)
    return r


def check(r):
    if time.time()-r['started_unix']>600:raise TimeoutError('Ten-minute stage budget.')
    r['resources'].append(resource_check(r['gpu_uuid'],initialized=True,shared_peers=r['peer_start_ticks']))


def finish(r):
    r.update(state='complete',elapsed_seconds=time.time()-r['started_unix'])
    write(OUT/f"{r['stage']}.json",r)


def failure(r,error):
    r.update(state='failed',error_type=type(error).__name__,error=str(error))
    write(OUT/f"{r['stage']}_failure.json",r)
