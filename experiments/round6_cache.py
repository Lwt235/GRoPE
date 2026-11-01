"""Cache new scenes with the unchanged Round5 RGB-only pretrained predictors."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.round5_cache import frame_inputs, hash_file, ROOT
from experiments.round5_rgbd_data import read_sun3d_intrinsics, read_sun3d_extrinsics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=ROOT/'data/round6_sun3d')
    p.add_argument('--selection',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'results/round6_inputs')
    p.add_argument('--splits',nargs='+',choices=['val','test'],default=['val'])
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();torch.set_num_threads(4)
    from depth_anything_v2.dpt import DepthAnythingV2
    from experiments.real_frozen_features import official_backbone
    weights=ROOT/'results/round5_inputs/downloads/depth_anything_v2_metric_hypersim_vits.pth'
    depth=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20)
    depth.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    depth.eval().requires_grad_(False).to(args.device)
    backbone,bmeta=official_backbone(Path('/home/liwt/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'))
    backbone.to(args.device)
    producer=dict(wrapper_sha256=hash_file(__file__),
        source_sha256=hash_file(ROOT/'experiments/round5_cache.py'),
        depth_weights_sha256=hash_file(weights),backbone_weights_sha256=bmeta['weights_sha256'],geometry_version=2)
    path=args.selection or args.data_root/'selection.json';selection=json.loads(path.read_text())
    started=time.time();manifest=[]
    for scene in selection['scenes']:
        if scene['split'] not in args.splits:continue
        kpath=args.data_root/scene['scene']/'intrinsics.txt';ppath=args.data_root/scene['pose_file']
        if hash_file(kpath)!=scene['K_sha256'] or hash_file(ppath)!=scene['pose_sha256']:
            raise ValueError('changed camera metadata')
        K=read_sun3d_intrinsics(kpath);poses=read_sun3d_extrinsics(ppath)
        for f in scene['frames']:
            rgb=args.data_root/scene['scene']/'image'/f['rgb']
            dest=args.out/'cache'/scene['scene']/f"{f['index']:07d}.npz"
            dest.parent.mkdir(parents=True,exist_ok=True)
            provenance=dict(producer,rgb_sha256=hash_file(rgb),K_sha256=scene['K_sha256'],
                pose_sha256=scene['pose_sha256'],original_index=f['index'])
            sidecar=dest.with_suffix('.json');valid=False
            if sidecar.exists() and dest.exists():
                old=json.loads(sidecar.read_text())
                valid=old.get('producer')==provenance and old.get('cache_sha256')==hash_file(dest)
            if not valid:
                arrays=frame_inputs(rgb,K,poses[f['index']],depth,backbone,args.device)
                np.savez_compressed(dest,**arrays)
                sidecar.write_text(json.dumps(dict(producer=provenance,cache_sha256=hash_file(dest)),indent=2))
            manifest.append(dict(scene=scene['scene'],split=scene['split'],index=f['index'],
                cache_sha256=hash_file(dest),rgb_sha256=provenance['rgb_sha256']))
            if len(manifest)%20==0:print('CACHED',len(manifest),'seconds',round(time.time()-started,1),flush=True)
    (args.out/('cache_'+'_'.join(args.splits)+'.json')).write_text(json.dumps(dict(frames=manifest,
        selection_sha256=hash_file(path),producer=producer,seconds=time.time()-started,
        source='unchanged official frozen DepthAnythingV2 Hypersim vits and ImageNet ResNet18',
        inputs='RGB, supplied K/C2W only; no sensor-depth input'),indent=2))
    print('DONE',len(manifest),flush=True)


if __name__=='__main__':main()
