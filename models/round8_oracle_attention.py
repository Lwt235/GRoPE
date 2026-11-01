"""Privileged-information diagnostics on the sealed Round7 one-block model."""
import torch
import torch.nn.functional as F
from models.round7_depth_attention import DepthAttention, FIELDS
from models.gaussian_quadratic_pe import controlled_features

ROLES=('pred_pred','gt_route','gt_value','gt_both','singleview')
ORACLE_FIELDS=('oracle_indices','oracle_weights','oracle_valid','k_oracle_value_depth')


def replace_routing(predicted,indices,weights,eligible):
    """Bilinear route on eligible rows; identical predicted fallback elsewhere."""
    if indices.shape!=weights.shape or indices.shape[:-1]!=predicted.shape[:-1] or indices.shape[-1]!=4:
        raise ValueError('expected four-neighbor routing arrays')
    route=torch.zeros_like(predicted)
    route.scatter_add_(-1,indices.long(),weights.to(predicted.dtype))
    return torch.where(eligible[...,None],route,predicted)


class OracleDepthAttention(DepthAttention):
    def __init__(self,diagnostic_role='pred_pred'):
        if diagnostic_role not in ROLES:raise ValueError(diagnostic_role)
        super().__init__('singleview' if diagnostic_role=='singleview' else 'full')
        self.diagnostic_role=diagnostic_role

    def forward(self,data,intervention=None,return_attention=False):
        if self.diagnostic_role in ('pred_pred','singleview'):
            return super().forward(data,intervention,return_attention)
        d={key:data[key] for key in FIELDS}
        qg,kg=controlled_features(d,self.method,self.bandwidth)
        q=F.normalize(self.qk(d['q']),dim=-1);k=F.normalize(self.qk(d['k']),dim=-1)
        logits=self.temperature*(q@k.transpose(-1,-2))+qg@kg.transpose(-1,-2)
        attn=logits.softmax(-1)
        if self.diagnostic_role in ('gt_route','gt_both'):
            attn=replace_routing(attn,data['oracle_indices'],data['oracle_weights'],data['oracle_valid'])
        depth=data['k_oracle_value_depth'] if self.diagnostic_role in ('gt_value','gt_both') else d['k_value_depth']
        qbase=d['q_depth'].clamp(.1,20).log()
        qhidden=F.gelu(self.query(torch.cat([d['q'],qbase[...,None]],-1)))
        values=F.gelu(self.value(torch.cat([d['k'],depth.clamp(.1,20).log()[...,None]],-1)))
        if intervention=='zero_v':values=torch.zeros_like(values)
        elif intervention=='reverse_v':values=values.flip(-2)
        elif intervention is not None:raise ValueError(intervention)
        aggregate=attn@values
        residual=self.head(torch.cat([qhidden,aggregate],-1)).squeeze(-1).clamp(-3,3)
        output=(qbase+residual).exp().clamp(.1,20)
        return (output,attn,values) if return_attention else output
