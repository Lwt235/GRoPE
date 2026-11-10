import numpy as np
import torch
from experiments.efficiency_sidecar.fused_hidden_grad import FanoutHidden, fuse_hidden_gradient
from experiments.efficiency_sidecar.check_cpu import model_for, variant, old_inputs, state_hash
from experiments.round65_common import ROOT, write

torch.set_num_threads(1)
r = {'checks': [], 'scope': 'CPU implementation checks only'}
x = torch.randn(1,3,4,5,dtype=torch.double,requires_grad=True)
assert torch.autograd.gradcheck(FanoutHidden.apply, (x,))
r['checks'].append('float64 gradcheck')
for used in ((0,), (1,3), (0,1,2,3)):
    for repeat in range(2):
        slots = FanoutHidden.apply(x)
        coeff = [torch.randn(1,3,5,4,dtype=torch.double).transpose(-1,-2) for _ in slots]
        loss = sum((slots[s]*coeff[s]).sum() for s in used)
        actual, = torch.autograd.grad(loss,x)
        torch.testing.assert_close(actual, sum(coeff[s] for s in used))
r['checks'].append('unused slots, strided cotangents, repeated backward')
inputs, y, r['inputs'] = old_inputs(2,16)
for role in ('relative5','distance_mlp'):
    for dtype in (torch.float32,torch.float64):
        base = model_for(role)
        a=variant(base,'geometry_factored').to(dtype=dtype).train()
        b=fuse_hidden_gradient(variant(base,'geometry_factored')).to(dtype=dtype).train()
        assert state_hash(a)==state_hash(b)
        values=tuple(t.to(dtype=dtype) for t in inputs)
        outputs=[]; gradients=[]
        for model in (a,b):
            torch.manual_seed(831)
            out=model(*values)
            torch.nn.functional.cross_entropy(out,y).backward()
            outputs.append(out.detach())
            gradients.append({n:p.grad.detach().clone() for n,p in model.named_parameters()})
        torch.testing.assert_close(*outputs,atol=1e-6,rtol=1e-5)
        maximum=0
        for n in gradients[0]:
            torch.testing.assert_close(gradients[0][n],gradients[1][n],atol=2e-6,rtol=2e-4)
            maximum=max(maximum,float((gradients[0][n]-gradients[1][n]).abs().max()))
        r['checks'].append(dict(role=role,dtype=str(dtype),max_gradient_abs=maximum))
write(ROOT/'results/efficiency_sidecar/backward1/cpu_checks.json',r)
print(r['checks'])
