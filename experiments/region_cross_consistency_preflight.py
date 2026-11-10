"""Discarded old-fit GPU checks before freezing a learned-reference study."""
import os
from pathlib import Path
import time
import numpy as np
import torch

from experiments.round65_common import ROOT,read,write,sha
from experiments.region_hrope_data import load_cache
from experiments.region_hrope_control import INPUT_FIELDS,configure
from experiments.round58_data import loss_function
from experiments.round58_run import shared_digest
from models.region_cross_consistency import ROLES,CrossConsistencyMatcher

OUT=ROOT/'results/region_cross_consistency/preparation'
SCALES=ROOT/'results/region_relation_consistency/diagnostic1/scales.json'


def main():
    assert not (OUT/'gpu_preflight.json').exists()
    assert not (OUT/'preflight_started.json').exists(),'Inspect an existing preflight before restarting.'
    cpu=read(OUT/'cpu_checks.json');assert cpu['passed'] and cpu['tests_passed']==3
    for path,digest in cpu['source_sha256'].items():assert sha(path)==digest
    inputs=read(ROOT/'results/region_hrope_control/preparation/data_lock.json')['caches']['fit0']
    assert sha(inputs['path'])==inputs['sha256']
    sources=[Path(__file__),ROOT/'models/region_cross_consistency.py',ROOT/'tests/test_region_cross_consistency.py',
        ROOT/'docs/region_cross_consistency_design.md',SCALES]
    write(OUT/'preflight_started.json',dict(started_unix=time.time(),pid=os.getpid(),
        source_sha256={str(p):sha(p) for p in sources},data=inputs,
        discarded_models=3,discarded_updates_per_model=2,selected_fit_indices=list(range(32)),
        new_eval_access=False,new_confirmation_access=False))
    configure();cache=load_cache('fit0')
    args=tuple(torch.as_tensor(cache[field][:32],device='cuda') for field in INPUT_FIELDS)
    target=torch.as_tensor(cache['overlap'][:32],device='cuda')
    scales=read(SCALES)['scales'];rows=[];common=None
    old=read(ROOT/'results/region_fixed_hrope/attempt1/runs/fixed_hrope/11/run.json')['initial_shared_sha256']
    for role in ROLES:
        torch.cuda.reset_peak_memory_stats();torch.manual_seed(11)
        scale=scales[role.removeprefix('cross_')]['scale']
        model=CrossConsistencyMatcher(role,scale).cuda().train()
        digest=shared_digest(model);common=digest if common is None else common
        assert digest==common==old
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        losses=[];gradients=[];started=time.time()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);logits=model(*args);loss=loss_function(logits,target)
            assert torch.isfinite(loss);loss.backward()
            g=model.position.coefficients.grad
            assert g is not None and torch.isfinite(g).all() and float(g.norm())>0
            gradients.append(g.detach().cpu().tolist())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();losses.append(float(loss.detach()))
        rows.append(dict(role=role,losses=losses,coefficient_gradients=gradients,
            final_coefficients=model.position.coefficients.detach().cpu().tolist(),
            parameters=sum(p.numel() for p in model.parameters()),pe_parameters=sum(p.numel() for p in model.position.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-started))
        print('CROSS_CONSISTENCY_PREFLIGHT',role,losses,rows[-1]['peak_allocated_bytes'],flush=True)
        del model,optimizer,loss,logits;torch.cuda.empty_cache()
    write(OUT/'gpu_preflight.json',dict(passed=True,completed_unix=time.time(),rows=rows,
        common_initialization_sha256=common,common_initialization_matches_fixedH_seed11=True,
        preflight_started_sha256=sha(OUT/'preflight_started.json'),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        all_weights_discarded=True,new_evaluation_outcomes=False,new_confirmation_access=False))


if __name__=='__main__':main()
