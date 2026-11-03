"""Acquire pinned, original-train GSModel60 bytes without decoding vertices.

The public hub metadata API supplies byte sizes and LFS SHA256 identities.
Only paths in the checked training manifest can be downloaded. Test objects
and test split content are never requested by this program.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/'results/round65_actual_gs/preparation'
RAW = Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GSModel60')
REPO = 'zxAi/GSModel60'
REVISION = 'e6407400f768fc7c0d8f8d1b2ed112739b38e9dc'
RESERVE = 50*2**30


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def training_rows():
    rows = [json.loads(s) for s in (PREP/'metadata/train.jsonl').read_text().splitlines()]
    paths = json.loads((PREP/'train.json').read_text())
    official = set((PREP/'modelnet40_official_train.txt').read_text().splitlines())
    if len(rows) != 7757 or len(set(paths)) != 7757 or set(paths) != {r['file_path'] for r in rows}:
        raise ValueError('training manifest identity')
    for row in rows:
        p = Path(row['file_path'])
        if p.is_absolute() or '..' in p.parts or len(p.parts) != 2 or p.suffix != '.ply':
            raise ValueError('unexpected source path')
        if row['split'] != 'train' or int(p.parts[0]) != row['taxonomy_id']:
            raise ValueError('unexpected source split')
        if row['taxonomy_id'] >= 30 and row['model_id'] not in official:
            raise ValueError('ModelNet identity outside original train')
    return rows


def category_tree(category):
    destination = PREP/'trees'/f'{category}.json'
    if destination.exists():
        return json.loads(destination.read_text())
    url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{category}?limit=1000'
    with urllib.request.urlopen(url, timeout=45) as response:
        if response.headers.get('Link'):
            raise ValueError('unexpected pagination: do not silently omit files')
        records = json.load(response)
    write(destination, records)
    return records


def catalog():
    destination = PREP/'train_catalog.json'
    if destination.exists():
        raise ValueError('catalog already exists; inspect it instead of replacing')
    rows = training_rows()
    (PREP/'trees').mkdir(exist_ok=True)
    entries = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for records in pool.map(category_tree, range(60)):
            entries.update({r['path']: r for r in records if r['type'] == 'file'})
    result = []
    for row in rows:
        item = entries[row['file_path']]
        lfs = item.get('lfs', {})
        digest = lfs.get('oid', '')
        if not re.fullmatch('[0-9a-f]{64}', digest) or lfs.get('size') != item['size']:
            raise ValueError('missing independently published byte identity')
        result.append(dict(row, size=item['size'], sha256=digest))
    total = sum(r['size'] for r in result)
    groups = {}
    for row in result:
        labels = groups.setdefault(row['sha256'], set())
        labels.add(row['taxonomy_id'])
    if any(len(v) > 1 for v in groups.values()):
        raise ValueError('identical bytes assigned inconsistent labels')
    record = dict(repository=REPO, revision=REVISION, rows=result, total_bytes=total,
                  unique_byte_groups=len(groups), vertices_decoded=False, test_access=False,
                  sources={str(p.relative_to(ROOT)):sha(p) for p in (
                      PREP/'train.json', PREP/'metadata/train.jsonl',
                      PREP/'modelnet40_official_train.txt')})
    write(destination, record)
    (PREP/'train_catalog.sha256').write_text(sha(destination)+'\n')
    print('CATALOG',len(result),'bytes',total,'unique_byte_groups',len(groups),
          'free_bytes',shutil.disk_usage(RAW.parent).free,flush=True)


def download_one(row):
    path = RAW/row['file_path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != row['size'] or sha(path) != row['sha256']:
            raise ValueError('existing raw file identity mismatch '+row['file_path'])
        return dict(path=row['file_path'], reused=True, bytes=row['size'])
    if shutil.disk_usage(RAW.parent).free < RESERVE+row['size']:
        raise OSError('preserve free-space reserve')
    partial = path.with_suffix('.part')
    digest, count = hashlib.sha256(), 0
    if partial.exists():
        with partial.open('rb') as stream:
            for chunk in iter(lambda:stream.read(2**20),b''):
                digest.update(chunk)
                count += len(chunk)
        if count > row['size']:
            raise ValueError('partial larger than declared source')
        if count == row['size']:
            if digest.hexdigest() != row['sha256']:
                raise ValueError('complete partial hash mismatch')
            partial.rename(path)
            return dict(path=row['file_path'],reused=True,bytes=count)
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row["file_path"]}'
    request = urllib.request.Request(url,headers={'Range':f'bytes={count}-'} if count else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        if count and response.status == 206:
            expected=f'bytes {count}-{row["size"]-1}/{row["size"]}'
            if response.headers.get('Content-Range') != expected:
                raise ValueError('resume range mismatch')
        elif count:
            if response.status != 200:
                raise ValueError('unsupported resume response')
            # Some servers ignore Range. Preserve the interrupted bytes, then
            # verify a complete response instead of appending duplicate bytes.
            partial.rename(partial.with_suffix(f'.interrupted_{time.time_ns()}'))
            digest,count=hashlib.sha256(),0
        with partial.open('ab') as f:
            for chunk in iter(lambda: response.read(2**20), b''):
                f.write(chunk)
                digest.update(chunk)
                count += len(chunk)
    if count < row['size']:
        # A clean-looking socket EOF can still truncate a large HTTP response.
        # Keep its prefix, retry the remaining range, and require the full hash.
        raise http.client.IncompleteRead(b'',row['size']-count)
    if count != row['size'] or digest.hexdigest() != row['sha256']:
        raise ValueError('download identity mismatch '+row['file_path'])
    partial.rename(path)
    return dict(path=row['file_path'], reused=False, bytes=count)


def download_with_retry(row,attempt):
    for retry in range(3):
        try:
            return download_one(row)
        except (urllib.error.URLError,TimeoutError,http.client.IncompleteRead) as error:
            key=row['file_path'].replace('/','_')
            write(attempt/f'network_{key}_{retry}.json',dict(path=row['file_path'],retry=retry,
                error_type=type(error).__name__,message=str(error),unix=time.time()))
            print('NETWORK_RETRY',row['file_path'],retry,type(error).__name__,flush=True)
            if retry==2:
                raise
            time.sleep(2**retry)


def acquire():
    path = PREP/'train_catalog.json'
    if sha(path) != (PREP/'train_catalog.sha256').read_text().strip():
        raise ValueError('catalog changed')
    source = json.loads(path.read_text())
    if source['revision'] != REVISION or source['repository'] != REPO:
        raise ValueError('pinned source changed')
    if {r['file_path'] for r in training_rows()} != {r['file_path'] for r in source['rows']}:
        raise ValueError('acquisition escapes training population')
    for name, digest in source['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('source metadata changed')
    remaining = sum(r['size'] for r in source['rows'] if not (RAW/r['file_path']).exists())
    if shutil.disk_usage(RAW.parent).free < remaining+RESERVE:
        raise OSError('insufficient storage for complete train population plus reserve')
    RAW.mkdir(parents=True, exist_ok=True)
    attempt = PREP/f'acquisition_{time.time_ns()}'
    attempt.mkdir()
    write(attempt/'plan.json',dict(source_sha256=sha(__file__),catalog_sha256=sha(path),
        expected_bytes=source['total_bytes'],raw_root=str(RAW),workers=4,
        minimum_free_bytes=RESERVE,vertices_decoded=False,test_access=False,
        pid=os.getpid(),started_unix=time.time()))
    print('ACQUISITION_START',os.getpid(),str(attempt),flush=True)
    completed, started = [], time.time()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            remaining_rows=iter(source['rows'])
            pending={pool.submit(download_with_retry,next(remaining_rows),attempt) for _ in range(4)}
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                # Resolve this bounded batch before scheduling more work. A
                # failure cannot hide behind thousands of already queued files.
                successful=[future.result() for future in done]
                pending-=done
                completed.extend(successful)
                write(attempt/'progress.json',dict(completed=len(completed),
                    verified_bytes=sum(r['bytes'] for r in completed),unix=time.time()))
                if len(completed)//20 != (len(completed)-len(successful))//20 or len(completed)==len(source['rows']):
                    print('ACQUIRED',len(completed),'/',len(source['rows']),flush=True)
                for _ in done:
                    row=next(remaining_rows,None)
                    if row is not None:
                        pending.add(pool.submit(download_with_retry,row,attempt))
        write(attempt/'completion.json',dict(completed=True,files=completed,
            seconds=time.time()-started,vertices_decoded=False,test_access=False))
        write(PREP/'acquisition_complete.json',dict(attempt=str(attempt),
            completion_sha256=sha(attempt/'completion.json'),catalog_sha256=sha(path)))
    except BaseException as error:
        write(attempt/'failure.json',dict(error_type=type(error).__name__,message=str(error),
            verified_files=completed,seconds=time.time()-started,vertices_decoded=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('catalog','acquire'))
    options = parser.parse_args()
    (catalog if options.action == 'catalog' else acquire)()
