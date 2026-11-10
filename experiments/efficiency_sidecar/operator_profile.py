"""Operator attribution only; profiler-instrumented time is not throughput."""
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.efficiency_sidecar.backward_common import OUT,initialize,check,finish,failure
from experiments.efficiency_sidecar.followup_gpu import base_model
from experiments.efficiency_sidecar.check_cpu import old_inputs,variant
from experiments.round65_common import write


def run():
    r=initialize('operators')
    try:
        data,labels,r['input_identity']=old_inputs(4,512)
        data=tuple(t.cuda() for t in data);labels=labels.cuda()
        r['scope']='Three profiled forward/backward steps after three warmups, no optimizer. Not throughput.'
        for role in ('relative5','distance_mlp'):
            check(r)
            base,ck=base_model(role);model=variant(base,'geometry_factored').cuda().train()
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                F.cross_entropy(model(*data),labels,label_smoothing=.1).backward()
            torch.cuda.synchronize();model.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,profile_memory=True,with_stack=False) as prof:
                for step in range(3):
                    model.zero_grad(set_to_none=True);torch.manual_seed(926000+step)
                    with torch.profiler.record_function('sidecar_forward'):
                        output=model(*data)
                        loss=F.cross_entropy(output,labels,label_smoothing=.1)
                    with torch.profiler.record_function('sidecar_backward'):
                        loss.backward()
                    torch.cuda.synchronize();prof.step()
            rows=[]
            for e in prof.key_averages(group_by_input_shape=True):
                rows.append(dict(name=e.key,count=e.count,input_shapes=e.input_shapes,
                    self_cpu_us=e.self_cpu_time_total,cpu_total_us=e.cpu_time_total,
                    self_device_us=e.self_device_time_total,device_total_us=e.device_time_total,
                    self_device_memory=e.self_device_memory_usage,device_memory=e.device_memory_usage))
            rows.sort(key=lambda x:x['self_device_us'],reverse=True)
            # Parent ranges identify which BMM/clone shapes occur in backward.
            backward_rows=[]
            for e in prof.events():
                parent=e.cpu_parent
                inside=False
                while parent is not None:
                    if parent.name=='sidecar_backward':inside=True;break
                    parent=parent.cpu_parent
                if inside and e.self_device_time_total>0:
                    backward_rows.append(dict(name=e.name,shapes=e.input_shapes,self_device_us=e.self_device_time_total))
            row=dict(role=role,checkpoint=ck,operators=rows,backward_events=backward_rows)
            r['measurements'].append(row)
            prof.export_chrome_trace(str(OUT/f'{role}_trace.json'))
            write(OUT/'operators_progress.json',r)
            print('PROFILE_TOP',role,[(x['name'],round(x['self_device_us']/1000,2),x['input_shapes']) for x in rows[:8]],flush=True)
            del model,output,loss,prof
            gc.collect();torch.cuda.empty_cache();time.sleep(.2)
        finish(r)
    except BaseException as e:failure(r,e);raise


if __name__=='__main__':run()
