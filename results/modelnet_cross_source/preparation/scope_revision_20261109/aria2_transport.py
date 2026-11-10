"""Resume only the fixed archives using the installed aria2 HTTP downloader.

Authenticated mirror resolution stays in Python. Short-lived redirect URLs are
given to aria2 via private input files, never command arguments or public logs.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse

import numpy as np
from experiments import modelnet_cross_source as experiment

HERE=Path(__file__).resolve().parent


def run():
    lock=experiment.ready()
    transition=experiment.read(experiment.OUT/'transport_transition2.json')
    for pid in transition['previous_pids']:
        if Path(f'/proc/{pid}').exists():
            raise ValueError('previous transfer process handle remains; inspect before replacement')
    root=experiment.DATA/'archives'
    exclusive=(root/'.aria2_transport.lock').open('a')
    fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=importlib.util.spec_from_file_location('existing_cross_source_auth',HERE/'download_with_existing_auth.py')
    auth=importlib.util.module_from_spec(spec);spec.loader.exec_module(auth)
    token,origin=auth.existing_user_credential()
    private=Path(tempfile.mkdtemp(prefix='grope_modelnet_aria2_',dir='/tmp'))
    os.chmod(private,0o700)
    mutex=threading.Lock()
    status_path=experiment.OUT/'download_aria2_runtime.json'
    if status_path.exists():
        raise ValueError('preserve prior execution record')
    status=dict(pid=os.getpid(),started_unix=time.time(),state='running',archive_workers=2,
        connections_per_archive=8,credential_available=bool(token),credential_origin=origin,
        credentials_logged=False,signed_urls_in_private_temporary_files_only=True,proxy_used=False,
        lock_sha256=experiment.sha(experiment.OUT/'lock.json'),transport_source_sha256=experiment.sha(__file__),
        scientific_plan_changed=False,archives={},failures=[])

    def save():
        experiment.write(status_path,dict(status,updated_unix=time.time()))

    def progress(name,**fields):
        with mutex:
            status['archives'].setdefault(name,{}).update(fields,updated_unix=time.time());save()

    save();print('ARIA2_TRANSPORT_STARTED',os.getpid(),flush=True)

    def archive(meta):
        name=meta['rfilename'];total=meta['size'];digest=meta['lfs']['sha256']
        if not re.fullmatch(r'[a-z_]+\.zip',name):raise ValueError('invalid locked filename')
        final=root/name;pending=final.with_suffix('.zip.partial')
        if final.is_symlink() or pending.is_symlink():raise ValueError('unexpected archive symlink')
        if final.exists():
            if final.stat().st_size!=total or experiment.sha(final)!=digest:raise ValueError('completed archive changed')
            progress(name,state='verified',bytes=total,sha256=digest,reused_completed=True);return
        initial=pending.stat().st_size if pending.exists() else 0
        if initial>total:raise ValueError('partial archive larger than fixed source')
        if shutil.disk_usage(root).free<total-initial+2*2**30:raise OSError('insufficient disk reserve')
        progress(name,state='resolving',sequential_prefix_bytes=initial,total_bytes=total)
        url=('https://hf-mirror.com/datasets/'+lock['candidate']['repository']+'/resolve/'+
             lock['candidate']['revision']+'/'+name)
        with experiment.mirror_opener().open(experiment.mirror_request(url,token,'HEAD'),timeout=30) as response:
            resolved=response.url
            if response.headers.get('Content-Length') is not None and int(response.headers['Content-Length'])!=total:
                raise ValueError('resolved source length changed')
        target=urllib.parse.urlsplit(resolved)
        if target.scheme!='https' or target.username or target.password or '\n' in resolved or '\r' in resolved:
            raise ValueError('unsafe resolved source URL')
        input_path=private/(name+'.input');log_path=private/(name+'.log')
        descriptor=os.open(input_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(descriptor,'w') as stream:stream.write(resolved+'\n')
        environment={key:value for key,value in os.environ.items() if key.lower() not in
                     ('http_proxy','https_proxy','all_proxy','ftp_proxy')}
        environment.update(NO_PROXY='*',no_proxy='*')
        command=['/usr/bin/aria2c','--no-conf=true','--all-proxy=','--http-proxy=','--https-proxy=',
                 '--continue=true','--auto-file-renaming=false','--allow-overwrite=false',
                 '--file-allocation=none','--split=8','--max-connection-per-server=8','--min-split-size=4M',
                 '--connect-timeout=20','--timeout=30','--max-tries=6','--retry-wait=5',
                 '--enable-color=false','--summary-interval=0','--show-console-readout=false',
                 '--download-result=hide','--console-log-level=warn','--input-file='+str(input_path),
                 '--dir='+str(root),'--out='+pending.name]
        try:
            descriptor=os.open(log_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(descriptor,'w') as output:
                child=subprocess.Popen(command,env=environment,stdout=output,stderr=subprocess.STDOUT)
                progress(name,state='downloading',aria2_pid=child.pid,final_host=target.hostname)
                while child.poll() is None:
                    if pending.exists():
                        st=pending.stat()
                        progress(name,allocated_bytes=st.st_blocks*512,logical_length=st.st_size,
                                 sparse_file_length_is_not_download_completion=True)
                    time.sleep(10)
                code=child.returncode
            if code!=0:
                # aria2 logs may contain signed URLs; publish only redacted text.
                safe=re.sub(r'https?://[^\s\]<>]+','[redacted_url]',log_path.read_text(errors='replace'))
                progress(name,state='failed',aria2_exit_code=code,redacted_error=safe[-4000:])
                raise RuntimeError('aria2 transfer failed with code '+str(code))
            if pending.stat().st_size!=total or experiment.sha(pending)!=digest:
                raise ValueError('completed transfer fails fixed archive size/SHA256; preserve it')
            pending.replace(final)
            experiment.write(final.with_suffix('.verified.json'),dict(bytes=total,sha256=digest,
                revision=lock['candidate']['revision'],proxy_used=False,transport='aria2_recovery2'))
            progress(name,state='verified',bytes=total,sha256=digest,aria2_exit_code=code)
            print('ARCHIVE_VERIFIED',name,total,flush=True)
        finally:
            input_path.unlink(missing_ok=True);log_path.unlink(missing_ok=True)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(archive,meta):meta['rfilename'] for meta in lock['candidate']['archive_files']}
            for future in as_completed(futures):
                try:future.result()
                except Exception as error:
                    with mutex:
                        status['failures'].append(dict(archive=futures[future],error_type=type(error).__name__));save()
                    print('ARCHIVE_FAILED',futures[future],type(error).__name__,flush=True)
        status.update(state='failed' if status['failures'] else 'complete',finished_unix=time.time());save()
    finally:
        token=None
        fcntl.flock(exclusive,fcntl.LOCK_UN);exclusive.close()
        if private.exists() and not any(private.iterdir()):private.rmdir()
    if status['failures']:raise SystemExit(1)
    print('ALL_ARCHIVES_VERIFIED',flush=True)


if __name__=='__main__':
    run()
