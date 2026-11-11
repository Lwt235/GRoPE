"""Standalone Direct read from only observed RGB, calibration and saved state."""
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_view_data import load_backbone
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_direct_reference import DirectReferenceAdapter
from models.gaussian_memory_view_token import inject_last_layer


@torch.inference_mode()
def run(input_path,memory_path,checkpoint_path,output_path):
    with np.load(input_path) as a:
        assert set(a.files)=={'image','K','c2w'}
        images=torch.from_numpy(a['image']).float().cuda()[None,None]
        K=torch.from_numpy(a['K']).float().cuda()[None]
        c2w=torch.from_numpy(a['c2w']).float().cuda()[None]
    ckpt=torch.load(checkpoint_path,weights_only=False,map_location='cuda')
    net=DirectReferenceAdapter('direct',ckpt['config']['center_variance']).cuda().eval()
    net.load_state_dict(ckpt['model'])
    memory={k:v.float() for k,v in torch.load(memory_path,weights_only=False,map_location='cuda')['memory'].items()}
    model=load_backbone()
    with torch.autocast('cuda',dtype=torch.bfloat16): tokens,start=model.aggregator(images)
    modified=inject_last_layer(tokens,start,net,projected_inputs(memory,c2w,K))
    base_depth,_=model.depth_head([None if x is None else x.float() for x in tokens],images=images,patch_start_idx=start)
    new_depth,_=model.depth_head([None if x is None else x.float() for x in modified],images=images,patch_start_idx=start)
    assert not Path(output_path).exists()
    np.savez_compressed(output_path,tokens=modified[-1][0,0,start:].cpu().numpy(),
                        original_tokens=tokens[-1][0,0,start:].float().cpu().numpy(),
                        depth_base=base_depth.float().cpu().numpy(),depth_modified=new_depth.float().cpu().numpy())
    print({'standalone':str(output_path),'finite':bool(torch.isfinite(new_depth).all()),
           'depth_change':float((new_depth-base_depth).abs().mean())},flush=True)
