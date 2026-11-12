"""Actual native integration, gradient and resource checks on two old fit clips."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.gaussian_scene_data import OUT, CACHE, input_plan, query_path, query_image, KINDS
from experiments.gaussian_scene_bind import load_tensor
from experiments.gaussian_memory_native_run import model_load
from experiments.gaussian_memory_incremental_data import write_json
from models.gaussian_scene_native import prepare_read, forward_read
from models.gaussian_scene_memory import SceneReader


def metric_depth(depth, hw=(210,210)):
    return F.interpolate(depth[None,None], hw, mode='bilinear', align_corners=False)[0,0]


def objective(depth, target, valid):
    residual = (metric_depth(depth)[valid].clamp_min(1e-6) / target[valid]).log()
    return .5 * (residual.square().mean() + residual.var(unbiased=False))


@torch.no_grad()
def official_current(model, image):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs, start = model.aggregator(image[None,None])
    with torch.autocast('cuda', enabled=False):
        depth, _ = model.depth_head(outputs, image[None,None], start)
    return depth[0,0,...,0]


def main():
    torch.set_num_threads(4)
    p = input_plan(); model = model_load(p); records=[]
    for meta in p['training_catalog'][:2]:
        rid = meta['id']; query = load_tensor(query_path(rid, 2), 'cuda')
        image = query_image(query, 'missing')
        reference = official_current(model, image)
        for kind in KINDS:
            torch.manual_seed(20261110)
            reader = SceneReader(center_pe=kind in ('T','P')).cuda()
            field = {} if kind == 'L' else load_tensor(CACHE/rid/('token_memory.pt' if kind=='T' else 'scene_memory.pt'), 'cuda')
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                sample=prepare_read(model, image, query['c2w'], query['K'], field, kind)
                with torch.no_grad():
                    initial=forward_read(model,sample,reader,kind)
            error=float((initial-reference).abs().max())
            assert error < 1e-4, (rid,kind,error)
            opt=torch.optim.AdamW(reader.parameters(),lr=1e-4,weight_decay=.01)
            steps=[]
            for step in range(2):
                trace={}; opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); begin=time.monotonic()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    depth=forward_read(model,sample,reader,kind,trace=trace)
                    loss=objective(depth,query['target'],query['valid'])
                assert torch.isfinite(loss)
                loss.backward()
                gradients={name:float(param.grad.float().norm()) for name,param in reader.named_parameters() if param.grad is not None}
                assert all(np.isfinite(value) for value in gradients.values())
                assert gradients['output.weight'] > 0 and all(trace[str(i)]>0 for i in (0,6,12,18))
                if step:
                    assert gradients['query.weight']>0
                    if kind in ('T','P'): assert gradients['center_pe.2.weight']>0
                torch.nn.utils.clip_grad_norm_(reader.parameters(),5.); opt.step()
                torch.cuda.synchronize()
                steps.append(dict(step=step,loss=float(loss),seconds=time.monotonic()-begin,gradients=gradients,read_layer_output_gradients=trace))
            record=dict(id=rid,kind=kind,complete=True,initial_native_depth_max_abs=error,
                        parameters=sum(x.numel() for x in reader.parameters()),steps=steps,
                        full_seconds=time.monotonic()-started,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
            records.append(record); print({key:record[key] for key in ('id','kind','initial_native_depth_max_abs','full_seconds','peak_allocated_GiB')},flush=True)
            write_json(OUT/'native_integration_progress.json',dict(complete=False,records=records,formal_training=False))
            del reader,opt,sample,field,depth,initial,loss
            torch.cuda.empty_cache()
        del query,reference
    write_json(OUT/'native_integration_checks.json',dict(complete=True,records=records,
        official_single_current_identity=True,all_four_read_layers_receive_gradient=True,
        pilot_weights_discarded=True,formal_training=False,scope='Old fit clips only; interface and cost, no downstream utility claim'))


if __name__=='__main__':main()
