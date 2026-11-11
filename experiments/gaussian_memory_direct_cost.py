"""Descriptive paired arithmetic timings on the shared GPU; no weight update."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_direct_data import OUT, SEEDS, VARIANTS, checked, load_record
from experiments.gaussian_memory_direct_evaluate import models
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_view_token import normalized_error


def measure():
    plan=checked(); nets=models()
    row=next(r for r in plan['catalog'] if r['row']['split']=='fit')
    d=load_record(row,'cuda'); rec,raw=d['rec'],d['raw']
    q=rec['query'][:2]; x=rec['input_tokens'][:2].cuda(); y=rec['teacher_tokens'][q].cuda()
    geometries={v:projected_inputs(d['direct_memory'] if v=='direct' else d['memory'],
                raw['c2w'][q].cuda(),raw['K'][q].cuda()) for v in VARIANTS}
    timings={v:{'forward':[],'forward_backward':[]} for v in VARIANTS}
    peaks={v:0. for v in VARIANTS}
    for mode in ('forward','forward_backward'):
        for repeat in range(35):
            for variant in VARIANTS[repeat%4:]+VARIANTS[:repeat%4]:
                net=nets[(variant,SEEDS[0])]; net.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
                with torch.set_grad_enabled(mode=='forward_backward'):
                    prediction=net(x,geometries[variant])
                    if mode=='forward_backward': normalized_error(prediction,y).mean().backward()
                torch.cuda.synchronize(); elapsed=(time.perf_counter()-started)*1000
                peaks[variant]=max(peaks[variant],torch.cuda.max_memory_allocated()/1024**3)
                if repeat>=5: timings[variant][mode].append(elapsed)
                del prediction
    record={'batch':2,'patches':300,'warmup':5,'repetitions':30,'optimizer_step_included':False,
            'shared_GPU':True,'exclusive_speed_claim':False,'memory_is_total_resident_process':True,
            'device':torch.cuda.get_device_name(0),'peak_GiB':peaks,
            'timing_ms':{v:{m:{'median':float(np.median(x)),'q25':float(np.quantile(x,.25)),
                               'q75':float(np.quantile(x,.75))} for m,x in a.items()} for v,a in timings.items()}}
    write_json(OUT / 'cost.json',record); print(json.dumps(record),flush=True)
