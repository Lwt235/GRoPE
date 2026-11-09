"""Resource-aware launcher for the unchanged frozen confirmation inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
assert (OUT/'download_complete.json').exists()
assert not (OUT/'execution_identity.json').exists()
assert not (OUT/'inference_launch.json').exists()
command = ['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu',
           '--format=csv,noheader,nounits']
snapshot = subprocess.check_output(command, text=True)
gpus = []
for line in snapshot.strip().splitlines():
    i, uuid, total, used, free, utilization = (v.strip() for v in line.split(','))
    gpus.append(dict(index=int(i), uuid=uuid, total_mib=int(total), used_mib=int(used),
                     free_mib=int(free), utilization_percent=int(utilization)))
# Leave at least 8 GiB even if the full 3 GiB allocator allowance is reached.
eligible = [g for g in gpus if g['free_mib'] >= 11*1024]
assert eligible, 'no GPU with the declared memory reserve; do not start'
selected = min(eligible, key=lambda g:(g['utilization_percent'], -g['free_mib']))
os.environ['CUDA_VISIBLE_DEVICES'] = selected['uuid']
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
record = dict(started_unix=time.time(), selected=selected, live_snapshot=gpus,
              allocator_limit_gib=3, minimum_remaining_reserve_gib=8,
              single_gpu_process=True, shared_gpu_authorized=True,
              launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              frozen_scientific_source_unchanged=True)
with (OUT/'inference_launch.json').open('x') as f:
    json.dump(record, f, indent=2)
    f.write('\n')


class Tee:
    def __init__(self, a, b): self.a, self.b = a, b
    def write(self, value):
        self.a.write(value)
        self.b.write(value)
        self.flush()
        return len(value)
    def flush(self):
        self.a.flush()
        self.b.flush()


with (OUT/'inference.log').open('x') as log:
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
    try:
        print('SELECTED_GPU', selected, flush=True)
        from experiments.region_retokenization_confirmation import infer_all
        infer_all()
    except BaseException as error:
        terminal = dict(status='failed', error=repr(error), completed_unix=time.time())
        raise
    else:
        terminal = dict(status='complete', completed_unix=time.time())
    finally:
        with (OUT/'inference_terminal.json').open('x') as f:
            json.dump(terminal, f, indent=2)
            f.write('\n')
        sys.stdout, sys.stderr = stdout, stderr
