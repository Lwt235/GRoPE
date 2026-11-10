"""Align active aria2 output inodes with the fixed archive staging names.

aria2 1.33 ignores global --out for this input-file invocation. Keep its active
files/control journals intact. Preserve any older sequential prefix, then bind
the expected staging name to the verified child process's actual output inode.
The original manager still requires complete size and LFS SHA256 before publish.
"""
from pathlib import Path
import os
import re
import time

import numpy as np
from experiments import modelnet_cross_source as experiment

OUT=experiment.OUT
ROOT=experiment.DATA/'archives'
STATUS=OUT/'path_alignment_runtime.json'


def identity(pid):
    base=Path('/proc')/str(pid)
    try:
        fields=(base/'stat').read_text().rsplit(')',1)[1].split()
        args=(base/'cmdline').read_bytes().decode().split('\0')
        return int(fields[19]),fields[0],args
    except FileNotFoundError:
        return None


if STATUS.exists():raise ValueError('preserve prior path-alignment run')
lock=experiment.ready()
manager=experiment.read(OUT/'download_aria2_runtime.json')
start=identity(manager['pid'])
assert start and start[1]!='Z' and any('aria2_transport.py' in x for x in start[2])
state=dict(pid=os.getpid(),started_unix=time.time(),manager_pid=manager['pid'],manager_start_ticks=start[0],
           state='running',records=[],scientific_plan_changed=False,lock_sha256=experiment.sha(OUT/'lock.json'))
experiment.write(STATUS,state)
backup=ROOT/'preserved_sequential_prefixes'
backup.mkdir(mode=0o700,exist_ok=True)
done=set()
while True:
    current=experiment.read(OUT/'download_aria2_runtime.json')
    for meta in lock['candidate']['archive_files']:
        name=meta['rfilename']
        if name in done or (ROOT/name).exists():continue
        row=current['archives'].get(name,{})
        pid=row.get('aria2_pid')
        child=identity(pid) if pid else None
        if not child or child[1]=='Z':continue
        args=child[2]
        assert '/usr/bin/aria2c' in args and '--out='+name+'.partial' in args and '--dir='+str(ROOT) in args
        outputs=[]
        try:
            for fd in (Path('/proc')/str(pid)/'fd').iterdir():
                try:target=fd.readlink()
                except FileNotFoundError:continue
                if target.parent==ROOT and re.fullmatch(r'[a-f0-9]{64}',target.name):outputs.append(target)
        except FileNotFoundError:continue
        outputs=sorted(set(outputs))
        if not outputs:continue
        if len(outputs)!=1:raise ValueError('ambiguous actual archive output')
        actual=outputs[0];pending=ROOT/(name+'.partial')
        if actual.is_symlink() or pending.is_symlink():raise ValueError('unexpected archive symlink')
        saved=None
        if pending.exists() and not os.path.samefile(pending,actual):
            saved=backup/(name+'.partial')
            if saved.exists():raise ValueError('preserve prior sequential prefix backup')
            os.link(pending,saved)
        if not pending.exists() or not os.path.samefile(pending,actual):
            temporary=ROOT/(name+'.alignment_pending')
            if temporary.exists():raise ValueError('preserve interrupted path alignment')
            os.link(actual,temporary)
            temporary.replace(pending)
        assert os.path.samefile(actual,pending)
        record=dict(archive=name,aria2_pid=pid,aria2_start_ticks=child[0],actual_path=str(actual),
                    staging_path=str(pending),inode=pending.stat().st_ino,
                    original_prefix_backup=str(saved) if saved else None,
                    original_prefix_bytes=saved.stat().st_size if saved else 0,
                    aligned_unix=time.time(),full_sha256_still_required=True)
        state['records'].append(record);done.add(name);experiment.write(STATUS,state)
        print('OUTPUT_PATH_ALIGNED',name,'preserved_prefix_bytes',record['original_prefix_bytes'],flush=True)
    observed=identity(manager['pid'])
    if not observed or observed[0]!=start[0] or observed[1]=='Z':
        state.update(state='complete',finished_unix=time.time(),manager_terminal_state=current['state'])
        experiment.write(STATUS,state)
        print('PATH_ALIGNMENT_COMPLETE',len(done),flush=True)
        break
    time.sleep(2)
