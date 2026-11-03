"""Pinned TRAIN-only hfd helpers and a bounded, alternating download comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request

from experiments import round65_acquire as base
from experiments.round65_common import PREP, RAW, DATA_ROOT, checked, read, sha, write
from experiments.round65_mirror_acquire import DirectMirror, ENDPOINT

PYTHON = '/home/liwt/miniColmap/.venv/bin/python'
HFD = Path('/home/liwt/hfd.sh')
ROOT = Path(__file__).resolve().parents[1]


def clean_environment(bin_dir):
    env = {k:v for k,v in os.environ.items() if k.lower() not in
           ('http_proxy', 'https_proxy', 'all_proxy', 'hf_token', 'hf_username',
            'hugging_face_hub_token', 'huggingface_hub_token')}
    env.update(HF_ENDPOINT=ENDPOINT, NO_PROXY='*', no_proxy='*',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PATH=str(bin_dir)+os.pathsep+env.get('PATH', ''))
    return env


def prepare_hfd(destination, rows, bin_dir, threads=4, jobs=2):
    """Seed hfd's cache from the already pinned manifest, avoiding a repo-wide walk."""
    destination, bin_dir = Path(destination).resolve(), Path(bin_dir).resolve()
    cache = destination/'.hfd'
    cache.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    paths = [r['file_path'] for r in rows]
    if len(set(paths)) != len(paths) or not paths:
        raise ValueError('empty or duplicate hfd allowlist')
    if any(not re.fullmatch(r'[0-9]+/[A-Za-z0-9_-]+\.ply', p) for p in paths):
        raise ValueError('unexpected file path syntax')
    patterns = ['^'+p+'$' for p in paths]
    # hfd uses this exact cache fingerprint. Anchors prevent substring matches
    # across categories; no TEST paths are included in this local manifest.
    fingerprint = (f'REPO_ID={base.REPO} TOOL=aria2c INCLUDE={" ".join(patterns)} '
                   f'EXCLUDE= DATASET=1 HF_USERNAME= HF_TOKEN= HF_ENDPOINT={ENDPOINT} '
                   f'REVISION={base.REVISION}')
    (cache/'manifest').write_text(''.join(f'{r["size"]}\t{r["file_path"]}\n' for r in rows))
    (cache/'repo_info').write_text(f'{len(rows)} {sum(r["size"] for r in rows)}\n')
    (cache/'last_download_command').write_text(fingerprint)
    write(cache/'repo_metadata.json', dict(id=base.REPO, sha=base.REVISION, gated=False,
        local_train_manifest=True, catalog_sha256=sha(PREP/'train_catalog.json')))
    # Bypass user aria2 proxy/config settings too, not just environment proxies.
    wrapper = bin_dir/'aria2c'
    wrapper.write_text('#!/bin/sh\nexec /usr/bin/aria2c --no-conf=true --all-proxy= '
        '--http-proxy= --https-proxy= --auto-file-renaming=false '
        '--connect-timeout=30 --timeout=60 --max-tries=6 --retry-wait=60 "$@"\n')
    wrapper.chmod(0o700)
    command = ['/bin/bash', str(HFD), base.REPO, '--dataset', '--revision', base.REVISION,
               '--tool', 'aria2c', '-x', str(threads), '-j', str(jobs), '--local-dir', str(destination),
               '--include', *patterns]
    return command, clean_environment(bin_dir)


def python_trial(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    base.RAW = directory/'files'
    base.RAW.mkdir(exist_ok=True)
    transport = DirectMirror({r['file_path'] for r in rows}, directory)
    base.urllib = types.SimpleNamespace(error=urllib.error, request=types.SimpleNamespace(
        Request=urllib.request.Request, urlopen=transport.urlopen))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda row: base.download_with_retry(row, directory), rows))


def verify(directory, rows):
    for row in rows:
        path = directory/row['file_path']
        if (not path.exists() or path.stat().st_size != row['size'] or
                path.with_suffix(path.suffix+'.aria2').exists() or sha(path) != row['sha256']):
            raise ValueError('download not fully verified: '+row['file_path'])


def benchmark(threads=4, jobs=2, first_method='python'):
    checked()
    stamp = str(time.time_ns())
    record = PREP/('hfd_benchmark_'+stamp)
    data = DATA_ROOT/'GRoPE'/'hfd_benchmark'/stamp
    record.mkdir(); data.mkdir(parents=True)
    catalog = read(PREP/'train_catalog.json')
    groups = []
    for category, low, high in ((20, 20_000_000, 30_000_000), (40, 2_000_000, 10_000_000)):
        candidates = sorted((r for r in catalog['rows'] if r['taxonomy_id'] == category and
            low <= r['size'] <= high and not (RAW/r['file_path']).exists() and
            not (RAW/r['file_path']).with_suffix('.part').exists()), key=lambda r:r['file_path'])
        if len(candidates) < 4:
            raise ValueError('insufficient untouched benchmark files')
        groups.append(candidates[:4])
    batches = [groups[0][:2]+groups[1][:2], groups[0][2:]+groups[1][2:]]
    attempt = max((p for p in PREP.glob('acquisition_*') if (p/'plan.json').exists()),
                  key=lambda p:int(p.name.rsplit('_', 1)[1]))
    old = read(attempt/'plan.json'); pid = old['pid']; proc = Path('/proc')/str(pid)
    command = proc.joinpath('cmdline').read_bytes().split(b'\0')
    if b'experiments.round65_mirror_acquire' not in command:
        raise ValueError('unexpected production process; do not signal it')
    ticks = proc.joinpath('stat').read_text().split()[21]
    other = 'hfd' if first_method == 'python' else 'python'
    plan = dict(created_unix=time.time(), data_root=str(data), old_pid=pid,
        old_start_ticks=ticks, old_attempt=str(attempt), batches=batches,
        order=[[first_method, other], [other, first_method]], hfd_threads=threads, hfd_jobs=jobs,
        python_workers=4, minimum_speedup=1.25, require_both_batches_faster=True,
        trial_timeout_seconds=300, hfd_sha256=sha(HFD), driver_sha256=sha(__file__),
        catalog_sha256=sha(PREP/'train_catalog.json'), direct_only=True,
        vertices_decoded=False, test_access=False)
    write(record/'plan.json', plan)
    (record/'driver_source.py').write_bytes(Path(__file__).read_bytes())
    (record/'hfd_source.sh').write_bytes(HFD.read_bytes())
    print('HFD_BENCHMARK_START', str(record), 'pid', os.getpid(), flush=True)
    results = []
    paused = False
    def interrupt(signum, frame):
        raise KeyboardInterrupt('benchmark interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        # Suspend only our exact downloader. Its open partials stay untouched;
        # finally resumes it even when a trial times out or raises an error.
        if proc.joinpath('stat').read_text().split()[21] != ticks:
            raise ValueError('production identity changed')
        os.kill(pid, signal.SIGSTOP); paused = True
        time.sleep(3)
        for index, methods in enumerate(plan['order']):
            rows = batches[index]
            for method in methods:
                if sha(HFD) != plan['hfd_sha256']:
                    raise ValueError('hfd changed during benchmark')
                trial = data/f'batch{index}_{method}'
                trial.mkdir()
                rows_path = trial/'rows.json'; write(rows_path, rows)
                if method == 'python':
                    cmd = [PYTHON, '-u', '-m', 'experiments.round65_hfd_transfer', 'trial-python',
                           '--directory', str(trial), '--rows', str(rows_path)]
                    env = clean_environment(record/'bin')
                else:
                    cmd, env = prepare_hfd(trial/'files', rows, record/'bin', threads, jobs)
                print('TRIAL_START', index, method, sum(r['size'] for r in rows), flush=True)
                write(record/'status.json', dict(batch=index, method=method, started_unix=time.time()))
                started = time.monotonic()
                with (record/f'batch{index}_{method}.log').open('wb') as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = child.wait(timeout=plan['trial_timeout_seconds'])
                    except BaseException:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=10)
                        raise
                if code:
                    raise RuntimeError(f'{method} trial exited {code}; see its log')
                verify(trial/'files', rows)
                elapsed = time.monotonic()-started
                result = dict(batch=index, method=method, bytes=sum(r['size'] for r in rows),
                              seconds=elapsed, MB_s=sum(r['size'] for r in rows)/elapsed/1e6,
                              sha256_verified=True)
                results.append(result); write(record/'trials.json', results)
                print('TRIAL_VERIFIED', json.dumps(result), flush=True)
        totals = {method:sum(r['seconds'] for r in results if r['method']==method)
                  for method in ('python', 'hfd')}
        ratios = [next(r['seconds'] for r in results if r['batch']==i and r['method']=='python')/
                  next(r['seconds'] for r in results if r['batch']==i and r['method']=='hfd') for i in range(2)]
        speedup = totals['python']/totals['hfd']
        decision = dict(completed=True, speedup=speedup, batch_speedups=ratios,
                        switch_recommended=speedup>=1.25 and min(ratios)>1,
                        trials=results, vertices_decoded=False)
        write(record/'result.json', decision)
        print('BENCHMARK_RESULT', json.dumps(decision), flush=True)
        # Reuse the successful benchmark bytes in the real dataset as well.
        for index, rows in enumerate(batches):
            for row in rows:
                source=data/f'batch{index}_python'/'files'/row['file_path']
                target=RAW/row['file_path']; target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if sha(target)!=row['sha256']:raise ValueError('existing destination mismatch')
                else:
                    os.link(source, target)
    except BaseException as error:
        write(record/'failure.json', dict(error_type=type(error).__name__, message=str(error),
                                        trials=results, unix=time.time()))
        raise
    finally:
        if paused and proc.exists() and proc.joinpath('stat').read_text().split()[21] == ticks:
            os.kill(pid, signal.SIGCONT)
            write(record/'production_resumed.json', dict(pid=pid, unix=time.time()))
            print('PRODUCTION_RESUMED', pid, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('benchmark', 'trial-python'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--rows', type=Path)
    parser.add_argument('--threads', type=int, choices=range(1, 11), default=4)
    parser.add_argument('--jobs', type=int, choices=range(1, 11), default=2)
    parser.add_argument('--first', choices=('python', 'hfd'), default='python')
    args = parser.parse_args()
    if args.action == 'benchmark':
        benchmark(args.threads, args.jobs, args.first)
    else:
        python_trial(args.directory, read(args.rows))


if __name__ == '__main__':
    main()
