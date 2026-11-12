"""Exact cached-encoder path; unused frozen modules stay on CPU during fitting."""
import torch
from experiments.gaussian_memory_native_run import model_load as original_model_load
from models.gaussian_scene_rendered import forward_read as original_forward_read


def model_load(p,offload_unused=True):
    model=original_model_load(p)
    if offload_unused:
        model.aggregator.patch_embed.cpu()
        for name in ('camera_head','point_head','track_head'):
            module=getattr(model,name,None)
            if module is not None:module.cpu()
        torch.cuda.empty_cache()
    return model


def forward_read(*args,**kwargs):
    # Each projection uses the same BF16 arithmetic without persistent FP32->BF16
    # weight-cast cache. No parameter/gradient/Adam state offload or quantization.
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        return original_forward_read(*args,**kwargs)
