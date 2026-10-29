"""Check compute occupancy immediately before a single-GPU pilot.

No jobs are killed or preempted. Selection is 7 -> 0, skipping occupied GPUs.
The probe is small; a memory threshold also excludes devices without visible PIDs.
"""
import os
import subprocess
import sys


def launch():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'
    ], text=True)
    occupied = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'
    ], text=True)
    busy = set(line.strip() for line in occupied.splitlines())
    cards = [line.strip().split(', ') for line in output.strip().splitlines()]
    for index, uuid, memory in sorted(cards, key=lambda row: int(row[0]), reverse=True):
        if uuid not in busy and int(memory) < 128:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = index
            env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            env['OMP_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            print(f'Using idle physical GPU {index}, uuid={uuid}, memory={memory}MiB', flush=True)
            os.execve(sys.executable, [sys.executable, '-m', 'experiments.matching_probe', '--device', 'cuda', *sys.argv[1:]], env)
    raise SystemExit('No idle GPU; pilot not started.')


if __name__ == '__main__':
    launch()
