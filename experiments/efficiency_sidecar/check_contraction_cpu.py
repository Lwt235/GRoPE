import numpy as np
import torch
from experiments.efficiency_sidecar.relation_contraction import Contract,adapt_contraction
from experiments.efficiency_sidecar.check_cpu import old_inputs,variant,state_hash,model_for
from experiments.round65_common import ROOT,write

torch.set_num_threads(1)
q=torch.randn(1,2,3,5,dtype=torch.double,requires_grad=True)
z=torch.randn(1,3,4,5,dtype=torch.double,requires_grad=True)
assert torch.autograd.gradcheck(Contract.apply,(q,z))
values,y,identity=old_inputs(2,16);rows=[]
for role in ('relative5','distance_mlp'):
    base=model_for(role)
    a=variant(base,'geometry_factored').double();b=adapt_contraction(variant(base,'geometry_factored')).double()
    assert state_hash(a)==state_hash(b)
    outputs=[];gradients=[]
    for model in (a,b):
        torch.manual_seed(839)
        out=model(*(t.double() for t in values));torch.nn.functional.cross_entropy(out,y).backward()
        outputs.append(out.detach());gradients.append({n:p.grad.detach().clone() for n,p in model.named_parameters()})
    torch.testing.assert_close(*outputs,atol=1e-10,rtol=1e-8)
    err=0
    for n in gradients[0]:
        torch.testing.assert_close(gradients[0][n],gradients[1][n],atol=1e-10,rtol=1e-8)
        err=max(err,float((gradients[0][n]-gradients[1][n]).abs().max()))
    rows.append(dict(role=role,max_gradient_abs=err))
write(ROOT/'results/efficiency_sidecar/contraction1/cpu.json',dict(gradcheck=True,models=rows,input_identity=identity))
print(rows)
