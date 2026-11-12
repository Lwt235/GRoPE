"""Isolated paths and source-only paired depth payloads for source_geometry_control1."""
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_incremental_data import depth_source, sensor_target, save_tensor, write_json
from experiments.gaussian_scene_data import transformed_rgb, source_path as old_source_path
from experiments.gaussian_scene_data import query_path, read, SOURCES, QUERIES, CONDITIONS, SEEDS
from experiments.gaussian_scene_cache import patch_path

OUT = ROOT/'results/gaussian_memory/source_geometry_control1'
CACHE = DATA/'source_geometry_control_v1'
PARENT_OUT = ROOT/'results/gaussian_memory/gaussian_scene_memory1'
PARENT_CACHE = DATA/'gaussian_scene_memory_v1'
MODES = ('pred','sensor')
KINDS = ('P','G')

def branch_out(mode):return OUT/mode
def branch_cache(mode):return CACHE/mode
def geometry_path(mode,rid,q):return branch_cache(mode)/rid/f'geometry_{q}.pt'
def source_path(rid):return CACHE/'inputs'/rid/'source_pair.pt'
def load(path,device='cpu'):return torch.load(path,map_location=device,weights_only=True)
def parent_plan():return read(PARENT_OUT/'plan.json')
def plan(mode=None):return read((branch_out(mode) if mode else OUT)/'plan.json')

def warp_sensor(frame,transform):
    depth,valid=sensor_target(frame)
    nw,nh,left,top=transform
    def warp(x):
        im=Image.fromarray(x.numpy()).resize((nw,nh),Image.Resampling.NEAREST)
        return torch.from_numpy(np.asarray(im.crop((left,top,left+448,top+448))).copy())
    return warp(depth).float(),warp(valid).bool()

def prepare_source(meta,split):
    rid=meta['id'];marker=OUT/'inputs'/f'{rid}.json'
    if marker.exists():
        record=read(marker);assert sha256(source_path(rid))==record['source_pair_sha256'];return record
    source=load(old_source_path(rid));raw=load(PARENT_CACHE/rid/'anysplat_source_init.pt')
    registration=load(PARENT_CACHE/rid/'source_registration.pt')
    pred=raw['source_depth'].permute(0,3,1,2)[:,0]*registration['scale']
    sensor=[];valid=[];files={}
    for j,index in enumerate(SOURCES):
        frame=meta['row']['frames'][index];path=depth_source(frame);digest=sha256(path)
        if 'depth_sha256' in frame:assert digest==frame['depth_sha256']
        _,transform=transformed_rgb(frame['rgb_path'])
        d,v=warp_sensor(dict(frame,depth_path=str(path)),transform);sensor.append(d);valid.append(v)
        files[str(path)]=digest
    sensor=torch.stack(sensor);sensor_valid=torch.stack(valid)
    assert pred.shape==sensor.shape==(4,448,448)
    common=sensor_valid & torch.isfinite(pred)&(pred>.1)&(pred<10)
    unit=load(PARENT_CACHE/rid/'coarse_geometry.pt')['unit']
    paired=dict(source,depth_pred=pred.float(),depth_sensor=sensor,common_valid=common,unit=unit)
    assert set(paired)=={'rgb','c2w','K','source_ids','id','place','split','depth_pred','depth_sensor','common_valid','unit'}
    source_path(rid).parent.mkdir(parents=True,exist_ok=True);save_tensor(source_path(rid),paired)
    record=dict(complete=True,id=rid,split=split,source_ids=list(SOURCES),source_only=True,queries_used=False,
                source_pair_sha256=sha256(source_path(rid)),parent_source_sha256=sha256(old_source_path(rid)),
                parent_prediction_sha256=sha256(PARENT_CACHE/rid/'anysplat_source_init.pt'),source_sensor_files=files,
                common_valid_fraction=float(common.float().mean()),unit=float(unit),
                privilege='Both arms share sensor-and-prediction joint source validity for matched candidate support; neither is a deployable RGB-only score.')
    write_json(marker,record);return record
