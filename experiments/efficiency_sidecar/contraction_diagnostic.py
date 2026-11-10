"""Old-input diagnosis of Adam amplification at softmax-invariant key bias.

Keep initial failure. No change to kernel, model formula, optimizer or gradients.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from experiments.efficiency_sidecar import contraction_gpu as runner
from experiments.efficiency_sidecar.check_cpu import old_inputs
from experiments.efficiency_sidecar.followup_gpu import base_model
from experiments.round65_common import write,sha


def run(args):
    runner.USE_MERGE=args.merge
    r,out=runner.initialize(args)
    try:
        cpu,y,r['input_identity']=old_inputs(args.batch,512)
        inputs=tuple(t.cuda() for t in cpu);y=y.cuda()
        for role in ('relative5','distance_mlp'):
            base,ck=base_model(role);states=[]
            for candidate in (False,True):
                runner.resource(r)
                model=runner.make(base,candidate);opt=runner.optimizer(model)
                torch.manual_seed(831);torch.cuda.reset_peak_memory_stats()
                logits=model(*inputs);loss=F.cross_entropy(logits,y,label_smoothing=.1);loss.backward()
                grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                updated={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
                model.eval()
                with torch.no_grad():
                    after=model(*inputs).cpu()
                    # Deterministic finite change only in key bias: tests exact-form softmax invariance.
                    for layer in model.layers:layer.qkv.bias[96:192].add_(torch.linspace(-.3,.3,96,device='cuda'))
                    shifted=model(*inputs).cpu()
                states.append(dict(before=logits.detach().cpu(),grad=grads,updated=updated,after=after,
                    key_shift_logit_max=float((after-shifted).abs().max()),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                del model,opt,logits,loss,grads,updated,after,shifted;runner.cleanup()
            a,b=states
            torch.testing.assert_close(a['before'],b['before'],atol=2e-6,rtol=2e-5)
            bad=[];gmax=0.;grel=0.;umax=0.
            for n in a['grad']:
                torch.testing.assert_close(a['grad'][n],b['grad'][n],atol=3e-6,rtol=3e-4)
                diff=(a['grad'][n]-b['grad'][n]).double()
                gmax=max(gmax,float(diff.abs().max()));grel=max(grel,float(diff.norm()/a['grad'][n].double().norm().clamp_min(1e-12)))
                delta=(a['updated'][n]-b['updated'][n]).abs();umax=max(umax,float(delta.max()))
                mask=delta>2e-5+2e-4*b['updated'][n].abs()
                if bool(mask.any()):
                    indices=torch.nonzero(mask).flatten()
                    bad.append(dict(name=n,indices=indices.tolist(),max_update_abs=float(delta.max()),
                        baseline_grad=a['grad'][n][mask].tolist(),candidate_grad=b['grad'][n][mask].tolist()))
                    assert n.endswith('qkv.bias') and bool(((indices>=96)&(indices<192)).all()),'Non-key-bias update mismatch'
            # Interpret update tolerance exceptions only if invariant coordinates and function stays close.
            torch.testing.assert_close(a['after'],b['after'],atol=2e-4,rtol=2e-4)
            assert max(a['key_shift_logit_max'],b['key_shift_logit_max'])<2e-4
            r['checks'].append(dict(role=role,checkpoint=ck,logit_max_abs=float((a['before']-b['before']).abs().max()),
                gradient_max_abs=gmax,gradient_max_tensor_relative_l2=grel,updated_max_abs=umax,
                after_update_logit_max_abs=float((a['after']-b['after']).abs().max()),update_tolerance_exceptions=bad,
                baseline_key_shift_logit_max=a['key_shift_logit_max'],candidate_key_shift_logit_max=b['key_shift_logit_max'],
                baseline_peak_bytes=a['peak_allocated_bytes'],candidate_peak_bytes=b['peak_allocated_bytes']))
            print('DIAGNOSTIC',r['checks'][-1],flush=True)
            del states,a,b;runner.cleanup()
        r['state']='complete';r['interpretation']='Original parameter-wise update gate fails only in softmax-invariant key biases. All gradients pass original tolerances; post-update outputs remain close. No clamping or code/formula change.'
        import time
        r['elapsed_seconds']=time.time()-r['started_unix']
        if (out/'correctness_failure.json').exists():r['original_failure_sha256']=sha(out/'correctness_failure.json')
        write(out/'diagnostic.json',r)
        assert not (out/'correctness.json').exists()
        write(out/'correctness.json',r)
    except BaseException as e:
        r.update(state='failed',error_type=type(e).__name__,error=str(e));write(out/'diagnostic_failure.json',r);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,choices=(4,16),default=4);p.add_argument('--merge',action='store_true')
    args=p.parse_args();args.stage='diagnostic';args.mode='shared';run(args)
