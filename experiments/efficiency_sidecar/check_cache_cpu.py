"""Single-thread correctness and cache accounting before any GPU cache test."""
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.efficiency_sidecar import cache_geometry as cached
from experiments.efficiency_sidecar.check_cpu import old_inputs, variant
from experiments.efficiency_sidecar.followup_gpu import OUT, base_model
from experiments.efficiency_sidecar.adapters import quadratic_descriptors
from experiments.round65_run import augment
from experiments.round65_common import write, sha, ROOT
from models.round65_actual_gs import shape_features, vectorize


def run():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    dest = OUT / 'cache_cpu.json'
    if dest.exists(): raise FileExistsError(dest)
    data, labels, identity = old_inputs(16, 2048)
    start = time.perf_counter()
    l = cached.prepare(data[1], precision=False)
    log_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    p = torch.linalg.inv(data[1])
    inv_ms = (time.perf_counter()-start)*1000
    both = dict(l, precision=p)
    record = dict(state='running', threads=1, nice=os.getpriority(os.PRIO_PROCESS, 0), input_identity=identity,
        prepare_cpu_ms=dict(log_matrix=log_ms, precision=inv_ms),
        additional_bytes={k: v.numel()*v.element_size() for k,v in both.items()},
        source_sha256={str(ROOT/'experiments/efficiency_sidecar/cache_geometry.py'):sha(ROOT/'experiments/efficiency_sidecar/cache_geometry.py')},
        checks=[], scientific_model_updated=False)
    try:
        rng = np.random.default_rng(924301)
        for round_id in range(4):
            indices = rng.choice(16, 2, replace=False)
            subset = np.stack([rng.choice(2048, 64, replace=False) for _ in indices])
            batch = tuple(t[indices[:, None], subset] for t in data)
            cache = {k: v[indices[:, None], subset] for k,v in both.items()}
            q = rng.normal(size=(2,4)); jitter = rng.normal(0,.01,(2,64,3)).clip(-.05,.05).astype(np.float32)
            c, s = augment(batch[0], batch[1], q, jitter)
            rc = cached.rotate(cache, q)
            common_error = float((shape_features(s)-vectorize(rc['log_matrix'])).abs().max())
            desc_error = float((quadratic_descriptors(c,s)-cached.cached_descriptor(c,s,rc['precision'])).abs().max())
            for role in ('relative5','distance_mlp'):
                base, ck = base_model(role)
                model = variant(base,'geometry_factored').train()
                outputs, gradients = [], []
                for use_cache in (False,True):
                    model.zero_grad(set_to_none=True); torch.manual_seed(924300)
                    output = (cached.forward(model,c,s,batch[2],log_matrix=rc['log_matrix'],
                        precision=rc['precision'] if role=='relative5' else None) if use_cache else model(c,s,batch[2]))
                    loss = F.cross_entropy(output,labels[indices],label_smoothing=.1);loss.backward()
                    outputs.append(output.detach())
                    gradients.append(torch.cat([t.grad.flatten() for t in model.parameters() if t.grad is not None]))
                err = float((outputs[0]-outputs[1]).abs().max())
                ge = float((gradients[0].double()-gradients[1].double()).norm()/gradients[0].double().norm())
                assert all(torch.isfinite(g).all() for g in gradients)
                assert err < 2e-4 and ge < 1e-4
                record['checks'].append(dict(round=round_id,role=role,checkpoint=ck,max_logit_abs=err,
                    gradient_relative_l2=ge, common_max_abs=common_error,descriptor_max_abs=desc_error))
        record['state']='complete'; write(dest,record)
        print('CACHE_CPU_PASS',max(x['max_logit_abs'] for x in record['checks']),
              max(x['gradient_relative_l2'] for x in record['checks']),flush=True)
    except BaseException as e:
        record.update(state='failed',error_type=type(e).__name__,error=str(e))
        write(OUT/'cache_cpu_failure.json',record);raise


if __name__=='__main__':run()
