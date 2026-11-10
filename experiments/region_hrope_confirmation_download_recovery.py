"""Resume identical frozen depth requests with one connection and backoff."""
import argparse
import http.client
from pathlib import Path
import time
from urllib.parse import urlsplit
from PIL import Image

from experiments.round65_common import ROOT, read, write, sha
from experiments.region_hrope_confirmation import OUT, RAW, check as check_evaluation

RECOVERY=OUT/'download_connection_recovery1'
HOST='sun3d.cs.princeton.edu'


def jobs():
    result={}
    for row in read(RAW/'selection.json')['scenes']:
        for frame in row['frames']:
            path=RAW/row['scene']/'depth'/frame['depth']
            url=row['base_url']+'depth/'+frame['depth']
            parsed=urlsplit(url)
            assert parsed.scheme=='https' and parsed.netloc==HOST and parsed.path.endswith('.png')
            result[str(path)]=(url,path)
    assert len(result)==1440
    return list(result.values())


def freeze():
    check_evaluation()
    assert not (RECOVERY/'execution_lock.json').exists()
    state=read(OUT/'queue_status.json')
    assert state['state']=='stopped' and not state['active']
    assert read(OUT/'preflight.json')['passed']
    assert not (OUT/'download_complete.json').exists()
    source=[Path(__file__).resolve(),ROOT/'experiments/region_hrope_confirmation_recovery_queue.py']
    write(RECOVERY/'execution_lock.json',dict(created_unix=time.time(),
        source_sha256={str(p):sha(p) for p in source},evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
        original_terminal_queue_sha256=sha(OUT/'queue_status.json'),preflight_sha256=sha(OUT/'preflight.json'),
        selection_sha256=sha(RAW/'selection.json'),requests=len(jobs()),concurrent_requests=1,
        minimum_seconds_between_new_requests=1.,attempts_per_missing_file=6,
        retry_delays_seconds=[5,10,20,40,60],proxy_disabled=True,
        scientific_inputs_and_rules_changed=False,all_planned_files_required=True,goal_complete=False))
    print('DOWNLOAD_RECOVERY_FROZEN',sha(RECOVERY/'execution_lock.json'),flush=True)


def checked():
    check_evaluation()
    lock=read(RECOVERY/'execution_lock.json')
    assert lock['evaluation_lock_sha256']==sha(OUT/'evaluation_lock.json')
    assert lock['preflight_sha256']==sha(OUT/'preflight.json')
    assert lock['selection_sha256']==sha(RAW/'selection.json')
    for p,digest in lock['source_sha256'].items():assert sha(p)==digest,p
    return lock


def run():
    lock=checked()
    assert not (RECOVERY/'download_status.json').exists(),'Inspect previous recovery first.'
    assert not (OUT/'download_complete.json').exists()
    rows,pending=[],[]
    for url,path in jobs():
        if path.exists():
            with Image.open(path) as image:image.verify()
            rows.append(dict(path=str(path),url=url,bytes=path.stat().st_size,sha256=sha(path)))
        else:pending.append((url,path))
    retained=len(rows)
    state=dict(state='downloading',started_unix=time.time(),retained_existing_files=retained,
        completed=retained,total=len(jobs()),attempt_errors=[],active=None,
        execution_lock_sha256=sha(RECOVERY/'execution_lock.json'),goal_complete=False)

    def save():
        state['updated_unix']=time.time();write(RECOVERY/'download_status.json',state)

    save();print('RECOVERY_EXISTING_FILES',retained,'MISSING',len(pending),flush=True)
    connection=None
    try:
        for url,path in pending:
            path.parent.mkdir(parents=True,exist_ok=True)
            partial=path.with_suffix('.png.part')
            completed=False
            for attempt in range(6):
                state['active']=dict(url=url,attempt=attempt+1);state['state']='downloading';save()
                delay=None
                try:
                    if connection is None:connection=http.client.HTTPSConnection(HOST,timeout=35)
                    parsed=urlsplit(url)
                    connection.request('GET',parsed.path,headers={'User-Agent':'GRoPE-research-data/1.0','Connection':'keep-alive'})
                    response=connection.getresponse()
                    retry_after=response.getheader('Retry-After')
                    if response.status!=200:
                        response.read()
                        if retry_after and retry_after.isdigit():delay=int(retry_after)
                        raise RuntimeError('HTTP '+str(response.status))
                    with partial.open('wb') as stream:
                        while chunk:=response.read(2**20):stream.write(chunk)
                    with Image.open(partial) as image:image.verify()
                    partial.replace(path)
                    rows.append(dict(path=str(path),url=url,bytes=path.stat().st_size,sha256=sha(path)))
                    state['completed']=len(rows);state['active']=None;save()
                    if len(rows)%20==0:print('RECOVERY_DEPTHS',len(rows),'/',state['total'],flush=True)
                    completed=True
                    time.sleep(1.)
                    break
                except Exception as error:
                    if connection is not None:
                        connection.close();connection=None
                    record=dict(url=url,attempt=attempt+1,error=repr(error),unix=time.time())
                    state['attempt_errors'].append(record)
                    print('RECOVERY_RETRY',attempt+1,url,str(error),flush=True)
                    if attempt<5:
                        delay=max(lock['retry_delays_seconds'][attempt],delay or 0)
                        state.update(state='backoff',next_attempt_unix=time.time()+delay);save()
                        while delay>0:
                            seconds=min(60,delay);time.sleep(seconds);delay-=seconds
            if not completed:
                state.update(state='source_unavailable',finished_unix=time.time());save()
                write(RECOVERY/'retained_file_manifest.json',sorted(rows,key=lambda r:r['path']))
                raise RuntimeError('Same frozen source remains unavailable after bounded low-rate retries; files preserved.')
        assert len(rows)==lock['requests']==1440
        rows.sort(key=lambda r:r['path'])
        assert not (OUT/'download_manifest.json').exists()
        write(OUT/'download_manifest.json',rows)
        write(OUT/'download_complete.json',dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),
            manifest_sha256=sha(OUT/'download_manifest.json'),completed_unix=time.time(),
            rgb_downloaded=False,proxy_disabled=True,recovery_execution_lock_sha256=sha(RECOVERY/'execution_lock.json')))
        state.update(state='download_complete',active=None,finished_unix=time.time());save()
        print('RECOVERY_DOWNLOAD_COMPLETE',len(rows),flush=True)
    finally:
        if connection is not None:connection.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run'))
    globals()[parser.parse_args().action]()
