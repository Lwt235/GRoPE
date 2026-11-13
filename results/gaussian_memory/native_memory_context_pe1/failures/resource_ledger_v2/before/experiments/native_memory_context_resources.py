"""Round003 byte ledger; large-volume drops are conditional, never a science gate."""
import shutil
import time
from experiments.native_memory_context_data import *
GIB=2**30

def size(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0

def snapshot():
    # Each completed fixed output is credited once, partial files only consume
    # actual space. 24GiB is ONE concurrent two-resume+atomic-write peak.
    estimates={'finals':21.4,'memories':1.6,'predictions':1.1,'weights':2.8,'graphs':.5}
    categories={};remaining=0
    for k,g in estimates.items():
        used=size(CACHE/k);reserved=int(g*GIB);credit=min(used,reserved);remaining+=reserved-credit
        categories[k]=dict(used=used,fixed_estimate=reserved,remaining=reserved-credit)
    resumes=size(CACHE/'resumes');transient=max(0,24*GIB-resumes)
    used=size(CACHE);F=shutil.disk_usage(CACHE).free;project=size(OUT);pf=shutil.disk_usage(ROOT).free
    return dict(epoch=time.time(),F=F,H=F-remaining-transient,remaining_fixed=remaining,remaining_transient_peak=transient,categories=categories,data_bytes=used,data_hard=52*GIB,project_bytes=project,project_hard=2*GIB,project_free=pf,tmpfs_free=shutil.disk_usage(RAM).free,complete_resume_bytes=resumes)

def check(previous=None,admission=False):
    v=snapshot();reasons=[]
    if v['data_bytes']>=52*GIB:reasons.append('data_hard_budget')
    if v['project_bytes']>=2*GIB:reasons.append('project_hard_budget')
    if v['H']<(8 if admission else 6)*GIB:reasons.append('projected_peak_space')
    if v['F']<5*GIB:reasons.append('actual_data_space')
    if admission and v['project_free']<8*GIB:reasons.append('project_admission_space')
    if previous and v['epoch']-previous['epoch']>=29 and v['H']<8*GIB and previous['H']-v['H']>=.5*GIB:reasons.append('unexplained_drop_only_when_peak_below8GiB')
    v['stop_reasons']=reasons
    folder=OUT/'resources';folder.mkdir(exist_ok=True)
    with (folder/'ledger.jsonl').open('a') as f:f.write(__import__('json').dumps(v)+'\n')
    write(folder/'current.json',v);return v
