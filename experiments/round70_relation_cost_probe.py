"""Bounded algebra probe; does not change any R70 fit or reported prediction.

Question: can the final linear layer of the pairwise MLP be moved onto q,
avoiding materialization of the N x N x H x D relation tensor?
Alternative: different contraction order breaks the function or its gradient.
Fixed comparison: identical q, hidden pair features, W, and output cotangent;
float64, float32, and bfloat16, one fixed random seed, small CPU tensors.
Decision: a matching real/float64 function supports a future implementation
option, not a quality claim or an observed GPU speed/memory improvement.
Reduced-precision differences are recorded, never substituted into frozen R70.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'results/round70_nvs/attempt1/cost_diagnostic1'


def explicit(q, hidden, weight):
    heads, dim = q.shape[1], q.shape[-1]
    relation = torch.nn.functional.linear(hidden, weight).unflatten(-1, (heads, dim))
    return torch.einsum('bhid,bijhd->bhij', q, relation)/(dim**.5)


def factorized(q, hidden, weight):
    heads, dim = q.shape[1], q.shape[-1]
    projected = torch.einsum('bhid,hdk->bhik', q, weight.unflatten(0, (heads, dim)))
    return torch.einsum('bhik,bijk->bhij', projected, hidden)/(dim**.5)


def main():
    if DEST.exists():
        raise RuntimeError('preserve the existing probe; inspect before retrying')
    torch.set_num_threads(4)
    torch.manual_seed(700072)
    original = [torch.randn(2,4,24,48,dtype=torch.float64),
                torch.randn(2,24,24,96,dtype=torch.float64),
                torch.randn(192,96,dtype=torch.float64)/96**.5]
    cotangent = torch.randn(2,4,24,24,dtype=torch.float64)
    rows = []
    for dtype in (torch.float64, torch.float32, torch.bfloat16):
        outputs, derivatives = [], []
        for operation in (explicit, factorized):
            q, hidden, weight = [v.to(dtype).clone().requires_grad_() for v in original]
            out = operation(q, hidden, weight)
            grads = torch.autograd.grad(out, (q,hidden,weight), cotangent.to(dtype))
            outputs.append(out.detach().double().numpy())
            derivatives.append([x.double().numpy() for x in grads])
        differences = [outputs[0]-outputs[1]]+[a-b for a,b in zip(*derivatives)]
        references = [outputs[0]]+derivatives[0]
        row = {'dtype':str(dtype), 'errors':{}}
        for name, delta, reference in zip(('attention_bias','q_grad','hidden_grad','weight_grad'),
                                          differences,references):
            row['errors'][name] = dict(max_abs=float(np.abs(delta).max()),
                rms_relative=float(np.linalg.norm(delta)/np.linalg.norm(reference)))
        rows.append(row)
    assert max(v['max_abs'] for v in rows[0]['errors'].values()) < 1e-10
    batch, tokens, heads, dim, width, bytes_per_element = 8,384,4,48,96,2
    relation_elements = batch*tokens*tokens*heads*dim
    projected_elements = batch*heads*tokens*width
    result = dict(question='move final relation linear layer to queries',
        algebra='q_h dot (W_h hidden_ij) = (W_h^T q_h) dot hidden_ij',
        scope='CPU arithmetic only; no fit, checkpoint change, dataset access, or GPU benchmark',
        rows=rows, fixed_seed=700072,
        nvs_shape=dict(batch=batch,reference_tokens=tokens,heads=heads,head_dim=dim,hidden_width=width),
        tensor_storage_only=dict(bytes_per_element=bytes_per_element,
            explicit_relation_mib=relation_elements*bytes_per_element/2**20,
            projected_query_mib=projected_elements*bytes_per_element/2**20,
            pair_hidden_mib=batch*tokens*tokens*width*bytes_per_element/2**20,
            note='Not peak-memory savings: autograd, temporary tensors and six layer-specific projections remain.'),
        attention_contraction_macs=dict(explicit=batch*heads*tokens*tokens*dim,
            factorized=batch*heads*tokens*dim*width+batch*heads*tokens*tokens*width,
            note='Hidden width 96 exceeds head dim 48; this contraction can cost more arithmetic.'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unix=time.time(),goal_complete=False)
    DEST.mkdir(parents=True)
    (DEST/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
