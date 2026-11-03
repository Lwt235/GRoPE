"""Freeze an implemented experiment after data-free CPU/CUDA checks."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiments.round65_common import (ROOT,OUT,PREP,CONFIG,ROLES,SEEDS,VARIANTS,
                                       source_paths,read,write,sha,checked)


def freeze(acquisition_pid,acquisition_attempt,cuda_record):
    if OUT.exists():raise ValueError('preserve existing scientific attempt')
    cuda=read(cuda_record)
    if not cuda['passed'] or cuda['config']!=CONFIG or [r['role'] for r in cuda['rows']]!=list(ROLES):
        raise ValueError('complete matching CUDA preflight required')
    for name,digest in cuda['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('CUDA-tested source changed '+name)
    if len({r['initial_shared_sha256'] for r in cuda['rows']})!=1:
        raise ValueError('unpaired CUDA initializations')
    proc=Path('/proc')/str(acquisition_pid)
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
    expected='/home/liwt/miniColmap/.venv/bin/python -u -m experiments.round65_acquire acquire'
    if command!=expected or proc.stat().st_uid!=os.getuid():raise ValueError('acquisition process identity')
    source_plan=read(Path(acquisition_attempt)/'plan.json')
    if source_plan['source_sha256']!=sha(ROOT/'experiments/round65_acquire.py'):
        raise ValueError('running downloader uses a different implementation')
    catalog=PREP/'train_catalog.json'
    if sha(catalog)!=(PREP/'train_catalog.sha256').read_text().strip():raise ValueError('catalog changed')
    paths=source_paths()+['experiments/round65_freeze.py']
    sources={name:sha(ROOT/name) for name in sorted(set(paths))}
    test_command=[sys.executable,'-m','unittest','tests.test_round65_actual_gs',
                  'tests.test_round65_acquire','tests.test_round65_execution','-v']
    checks=PREP/f'freeze_checks_{time.time_ns()}';checks.mkdir()
    started=time.monotonic();result=subprocess.run(test_command,cwd=ROOT,capture_output=True,text=True)
    (checks/'tests.log').write_text(result.stdout+result.stderr)
    units=dict(passed=result.returncode==0,returncode=result.returncode,command=test_command,
               seconds=time.monotonic()-started,source_sha256=sources,real_vertex_arrays_decoded=False)
    write(checks/'result.json',units)
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:raise RuntimeError('implementation tests failed; record preserved')
    if any(sha(ROOT/name)!=digest for name,digest in sources.items()):raise ValueError('source changed during preflight')
    OUT.mkdir()
    write(OUT/'lock.json',dict(config=CONFIG,roles=ROLES,seeds=SEEDS,variants=VARIANTS,sources=sources,
        catalog_sha256=sha(catalog),cuda_preflight_sha256=sha(cuda_record),created_unix=time.time(),
        complete_scientific_objective_preserved=True,fresh_confirmation=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    write(OUT/'units.json',units)
    shutil.copyfile(cuda_record,OUT/'cuda_preflight.json')
    write(OUT/'execution.json',dict(lock_sha256=sha(OUT/'lock.json'),
        acquisition=dict(pid=acquisition_pid,start_ticks=(proc/'stat').read_text().split()[21],
            command=command,attempt=str(Path(acquisition_attempt).resolve())),
        automatic_steps=['await verified acquisition','prepare native train arrays','fit fifteen matched models',
            'selected checkpoint audit','development evaluation','independent metrics'],
        no_test_data_authorized=True,goal_complete=False))
    checked();print('R65_FROZEN',sha(OUT/'lock.json'),'sources',len(sources),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--acquisition-pid',type=int,required=True)
    parser.add_argument('--acquisition-attempt',type=Path,required=True)
    parser.add_argument('--cuda-record',type=Path,default=PREP/'cuda_preflight1/result.json')
    args=parser.parse_args();freeze(args.acquisition_pid,args.acquisition_attempt,args.cuda_record)
