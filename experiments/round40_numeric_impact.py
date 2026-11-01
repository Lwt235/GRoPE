"""Bounded follow-up of R40's two legacy float32 probability warnings.

No training, tolerance change, or replacement of recorded predictions. The
effect calculation covers the predetermined two validation pairs per run;
it is not a whole-population floating-point error certificate.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.round40_decoupled_data import read, write, sha
from experiments.round40_decoupled_train import load_split
from experiments.round7_depth_audit import numpy_attention
from models.round7_depth_attention import CONFIGS
from models.round40_decoupled_pe import DecoupledDepthAttention


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    out=Path('results/round40_decoupled_gpu/attempt1')
    dest=out/'audit_impact/numeric_impact.json'
    if dest.exists():raise ValueError('retain impact record')
    verification=read(out/'audit_impact/verification.json')
    summary=read(out/'audit_impact/summary.json')
    raw,_=load_split('val','cpu',out)
    rows=[]
    for warning in verification['numeric_warnings']:
        role,seed=warning['role'],warning['seed']
        run=out/'runs'/role/str(seed)
        state=torch.load(run/'best.pt',map_location='cpu',weights_only=True)['state_dict']
        model=DecoupledDepthAttention(role).double();model.load_state_dict(state)
        with np.load(run/'score_samples.npz',allow_pickle=False) as z:
            d={k:v[z['indices']].double() for k,v in raw.items()}
            saved=z['probability'].astype(np.float64)
        if role=='singleview':
            d.update(k=d['q'],kmean=d['qmean'],kcov=d['qcov'],q_to_k=d['qmean'],
                     q_to_kcov=d['qcov'],k_value_depth=d['q_value_depth'])
        method,h,temp=CONFIGS[role]
        ref=numpy_attention({k:v.numpy() for k,v in d.items()},state,method,h,temp)
        qbase=d['q_depth'].clamp(.1,20).log()
        qhidden=F.gelu(model.query(torch.cat((d['q'],qbase[...,None]),-1)))
        values=F.gelu(model.value(torch.cat((d['k'],d['k_value_depth'].clamp(.1,20).log()[...,None]),-1)))
        depths=[]
        for probability in (saved,ref):
            residual=model.head(torch.cat((qhidden,torch.from_numpy(probability)@values),-1)).squeeze(-1).clamp(-3,3)
            depths.append((qbase+residual).exp().clamp(.1,20))
        change=abs(depths[0]-depths[1])
        rows.append(dict(**warning,sampled_pairs=2,sampled_queries=864,
            p_max_error=float(abs(saved-ref).max()),winner_changes=int(np.sum(saved.argmax(-1)!=ref.argmax(-1))),
            depth_max_change_m=float(change.max()),depth_mean_change_m=float(change.mean())))
    affected={r['role'] for r in rows}
    decision_roles={'decoupled',*(r['control'] for r in summary['checks'])}
    record=dict(rows=rows,source_sha256=sha(__file__),
        warning_roles_intersect_primary_decision=bool(affected&decision_roles),
        original_tolerance_warnings_retained=True,predictions_replaced=False,training_repeated=False,
        scope='Two predetermined complete probability matrices per warned run; full saved depth metrics separately replayed for all65 runs',
        decision='Warnings do not invalidate the observed negative depth result: neither primary nor any controlling comparison has a warning. No universal numerical guarantee is inferred.')
    write(dest,record)
    print(record)


if __name__=='__main__':main()
