"""Round011 execution adapter. Scientific authority is the sole round011 contract."""
import os,json,time,fcntl
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from experiments import camera_only_feature_data as parent
from experiments.native_pose_memory_data import source as source_rgb,added
from experiments.source_geometry_data import CACHE as SOURCE_INPUT_CACHE
ROOT=parent.ROOT;BASEOUT=ROOT/'results/gaussian_memory/native_feature_field_pe1';BASECACHE=parent.old.DATA/'native_feature_field_pe_v1'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_011/science_plan.md';SCIENCE_SHA='8ef5ad0ee10ff96b261710a0b015cc21aa6748d9c3df954c57a41c3197598901'
ARMS=parent.ARMS;SEEDS=parent.SEEDS;ROW='C';OUT=BASEOUT/ROW;CACHE=BASECACHE/ROW
now=parent.now;read=parent.read;sha=parent.sha;load=parent.load;write=parent.write;journal=parent.journal;tensor_save=parent.tensor_save;numpy_save=parent.numpy_save;to_device=parent.to_device
source_geometry=parent.source_geometry;ordinary=parent.ordinary
@lru_cache(None)
def contract():return parent.contract()
@lru_cache(None)
def metas():return parent.metas()
_old_plan=parent.old.plan()
parent.old.plan=lambda:_old_plan
class Paths:
    def __getattr__(self,name):return getattr(parent.old,name)
    def scene_path(self,rid):return CACHE/'fields'/f'{rid}.pt'
    def feature_path(self,rid,q):return CACHE/'anchors'/f'{rid}_{q}.pt'
old=Paths()
def context(row):
    global ROW,OUT,CACHE
    assert row in ('C','J');ROW=row;OUT=BASEOUT/row;CACHE=BASECACHE/row;OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
def original_scene(rid):return parent.old.scene_path(rid)
def fields_path(rid):return CACHE/'geometry'/f'{rid}.pt'
def view_path(rid,q):return CACHE/'views'/f'{rid}_{q}.pt'
def final_path(arm,seed):return CACHE/'finals'/f'{arm}_{seed}.pt'
def memory_path(arm,seed,rid,permuted=False):return CACHE/'states'/f'{arm}_{seed}'/f'{rid}{"_permuted" if permuted else ""}.pt'
def target_path(rid):return BASECACHE/'source_targets'/f'{rid}.pt' if rid in added() else SOURCE_INPUT_CACHE/'inputs'/rid/'source_features.pt'
def cached_anchor(rid,q):
    x=load(old.feature_path(rid,q));return x['b'],x['effective_alpha']
def charge(purpose,**amounts):
    with (OUT/'budget.lock').open('a') as g:
        fcntl.flock(g,fcntl.LOCK_EX);p=OUT/'budget.json';b=read(p) if p.exists() else dict(counts={'pilot_updates':0},by_purpose={})
        for k,n in amounts.items():b['counts'][k]=b['counts'].get(k,0)+n;b['by_purpose'].setdefault(purpose,{})[k]=b['by_purpose'].get(purpose,{}).get(k,0)+n
        assert b['counts'].get('formal_updates',0)<=18000 and b['counts'].get('source_updates',0)<=13200
        b['updated_utc']=now();write(p,b)
def diskbytes(root):return parent.diskbytes(root)
def resources(previous=None,admission=False):
    used=diskbytes(BASECACHE);project=diskbytes(BASEOUT);s=os.statvfs(BASECACHE if BASECACHE.exists() else BASECACHE.parent);free=s.f_bavail*s.f_frsize;ps=os.statvfs(ROOT);pfree=ps.f_bavail*ps.f_frsize
    # Reserve only unwritten portion; 36GiB includes finals, complete source Adam states and two atomic transients.
    remaining=max(0,35.5*2**30-used);transient=.15*2**30;H=free-remaining-transient;reasons=[]
    if used+transient>36*2**30:reasons.append('DATA_peak_cap')
    if H<(8 if admission else 6)*2**30 or free<5*2**30:reasons.append('DATA_headroom')
    if project>2**30 or pfree-max(0,2**30-project)<2*2**30:reasons.append('project_headroom')
    runtime=[read(p) for p in (BASEOUT/'runtime').glob('*.json')] if (BASEOUT/'runtime').exists() else []
    seconds=sum(r.get('seconds',time.time()-r['start_epoch']) for r in runtime if r.get('GPU_session'))
    if seconds>=8*3600-60:reasons.append('GPU_worker_budget')
    unexplained=0
    if previous and time.time()-previous['epoch']>=29:
        unexplained=max(0,previous['free']-free-max(0,used-previous['used']))
        if H<8*2**30 and unexplained>=.5*2**30:reasons.append('unexplained_drop_tight_band')
    rec=dict(utc=now(),epoch=time.time(),used=used,free=free,H=H,remaining=remaining,transient=transient,project_bytes=project,project_free=pfree,GPU_seconds=seconds,unexplained=unexplained,stop_reasons=reasons);journal(BASEOUT/'resources.jsonl',rec);return rec
