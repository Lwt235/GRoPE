"""Operational recovery of a verified short transfer; frozen science unchanged.

Only the seven exact locked archives are downloaded. Every response must match
the requested range and total size. Final LFS SHA256 is mandatory before rename.
Credentials are held in memory and never logged or forwarded to redirect hosts.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import errno
import fcntl
import http.client
import importlib.util
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.error

import numpy as np
from experiments import modelnet_cross_source as experiment

HERE = Path(__file__).resolve().parent
SEGMENT = 256 * 2**20
RETRIES = 4
WORKERS = 2


def check_range(status, content_range, content_length, start, end, total):
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range or '')
    if status != 206 or match is None or tuple(map(int,match.groups())) != (start,end,total):
        raise ValueError('response range or source size differs from fixed request')
    if content_length is not None and int(content_length) != end-start+1:
        raise ValueError('response length differs from declared range')


def credential():
    spec = importlib.util.spec_from_file_location('existing_cross_source_auth',HERE/'download_with_existing_auth.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.existing_user_credential()


def retryable(error):
    if isinstance(error,urllib.error.HTTPError):
        return error.code in (408,429,500,502,503,504)
    if isinstance(error,(TimeoutError,EOFError,ConnectionError,urllib.error.URLError,http.client.HTTPException)):
        return True
    return isinstance(error,OSError) and error.errno in (errno.ETIMEDOUT,errno.ECONNRESET,errno.ECONNABORTED,errno.EPIPE)


def run():
    lock = experiment.ready()
    failure = experiment.read(experiment.OUT/'transport_failure1.json')
    if not failure['partial_is_short'] or not failure['prefix_probe']['prefix_matches_existing_partial']:
        raise ValueError('this recovery is specifically for the recorded short transfer')
    for pid in failure['original_pids']:
        if Path(f'/proc/{pid}').exists():
            raise ValueError('original process handle still exists; review before recovery')
    root = experiment.DATA/'archives'
    exclusive = (root/'.resume_transport1.lock').open('a')
    fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token,origin = credential()
    mutex = threading.Lock()
    status_path = experiment.OUT/'download_resume1_runtime.json'
    if status_path.exists():
        raise ValueError('preserve prior recovery execution')
    status = dict(pid=os.getpid(),started_unix=time.time(),state='running',workers=WORKERS,
        range_bytes=SEGMENT,maximum_retries_per_segment=RETRIES,credential_available=bool(token),
        credential_origin=origin,credentials_logged=False,proxy_used=False,
        lock_sha256=experiment.sha(experiment.OUT/'lock.json'),scientific_plan_changed=False,
        recovery_source_sha256=experiment.sha(__file__),archives={},failures=[])

    def save():
        experiment.write(status_path,dict(status,updated_unix=time.time()))

    def progress(name,**fields):
        with mutex:
            row=status['archives'].setdefault(name,{})
            row.update(fields,updated_unix=time.time())
            save()

    save()
    print('RANGE_RECOVERY_STARTED',os.getpid(),'workers',WORKERS,flush=True)

    def archive(meta):
        name=meta['rfilename']
        if not re.fullmatch(r'[a-z_]+\.zip',name):
            raise ValueError('unexpected locked archive name')
        final=root/name;pending=final.with_suffix('.zip.partial')
        if final.is_symlink() or pending.is_symlink():
            raise ValueError('unexpected archive symlink')
        total=meta['size'];digest=meta['lfs']['sha256']
        if final.exists():
            if final.stat().st_size!=total or experiment.sha(final)!=digest:
                raise ValueError('completed archive identity changed')
            progress(name,state='verified',bytes=total,sha256=digest,reused_completed=True)
            return
        if not pending.exists():
            pending.touch(exist_ok=False)
        if pending.stat().st_size>total:
            raise ValueError('partial archive larger than fixed source')
        initial=pending.stat().st_size
        progress(name,state='downloading',bytes=initial,total_bytes=total,resumed_bytes=initial,retries=0)
        opener=experiment.mirror_opener()
        url=('https://hf-mirror.com/datasets/'+lock['candidate']['repository']+'/resolve/'+
             lock['candidate']['revision']+'/'+name)
        failures=0;attempts=[]
        while pending.stat().st_size<total:
            start=pending.stat().st_size;end=min(start+SEGMENT,total)-1
            if shutil.disk_usage(root).free<2*2**30+(end-start+1):
                raise OSError('insufficient disk reserve for range')
            request=experiment.mirror_request(url,token)
            request.add_header('Range',f'bytes={start}-{end}')
            started=time.time()
            try:
                with opener.open(request,timeout=30) as response:
                    check_range(response.status,response.headers.get('Content-Range'),
                                response.headers.get('Content-Length'),start,end,total)
                    with pending.open('ab') as sink:
                        if sink.tell()!=start:
                            raise ValueError('partial archive changed concurrently')
                        remaining=end-start+1
                        while remaining:
                            chunk=response.read(min(4*2**20,remaining))
                            if not chunk:
                                raise EOFError('range ended before its declared length')
                            sink.write(chunk);remaining-=len(chunk)
                        sink.flush();os.fsync(sink.fileno())
                failures=0
                attempts.append(dict(start=start,end=end,started_unix=started,finished_unix=time.time(),complete=True))
                progress(name,state='downloading',bytes=pending.stat().st_size,range_attempts=attempts)
                print('RANGE_COMPLETE',name,pending.stat().st_size,'/',total,flush=True)
            except Exception as error:
                failures+=1
                attempts.append(dict(start=start,end=end,started_unix=started,finished_unix=time.time(),
                    complete=False,error_type=type(error).__name__,bytes_after=pending.stat().st_size))
                progress(name,state='retrying',bytes=pending.stat().st_size,range_attempts=attempts,retries=failures)
                if not retryable(error) or failures>=RETRIES:
                    raise
                time.sleep(min(16,2**failures))
        actual=experiment.sha(pending)
        if actual!=digest:
            raise ValueError('completed archive does not match fixed LFS SHA256; preserve partial file')
        pending.replace(final)
        experiment.write(final.with_suffix('.verified.json'),dict(bytes=total,sha256=actual,
            revision=lock['candidate']['revision'],proxy_used=False,transport='bounded_http_ranges_recovery1'))
        progress(name,state='verified',bytes=total,sha256=actual)
        print('ARCHIVE_VERIFIED',name,total,flush=True)

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures={pool.submit(archive,meta):meta['rfilename'] for meta in lock['candidate']['archive_files']}
            for future in as_completed(futures):
                name=futures[future]
                try:
                    future.result()
                except Exception as error:
                    with mutex:
                        status['failures'].append(dict(archive=name,error_type=type(error).__name__))
                        save()
                    print('ARCHIVE_FAILED',name,type(error).__name__,flush=True)
        status.update(state='failed' if status['failures'] else 'complete',finished_unix=time.time())
        save()
    finally:
        token=None
        fcntl.flock(exclusive,fcntl.LOCK_UN)
        exclusive.close()
    if status['failures']:
        raise SystemExit(1)
    print('ALL_ARCHIVES_VERIFIED',flush=True)


if __name__=='__main__':
    run()
