"""Eight fixed diagnostic operators; four geometries and one shared content bias.

No labels, selection, training, file access, moment repair or adaptive rule.
All four geometries share R20's exact float32 tau(Ca)*content arithmetic.
The full4 mixture keeps its historical float32 components unchanged; its
double-precision aggregate moments and finite-precision residuals are exposed.
"""
from __future__ import annotations

import torch

from models import round15_validation_pe as OLD
from models import round17_trace_pe as TRACE
from models import round20_anchor_pe as ANCHOR

GEOMETRIES = ('mu_c', 'a_c', 'a_ca', 'full4')
FAMILIES = tuple(g+'_'+mode for g in GEOMETRIES for mode in ('zero','content'))
ROLE_CONFIGS = {g+'_'+mode: dict(geometry=g, content_policy=mode, h=.25, temperature=10.)
                for g in GEOMETRIES for mode in ('zero','content')}


def mixture_moments(nodes, covariance, weights):
    if (nodes.dtype != torch.float32 or covariance.dtype != torch.float32 or weights.dtype != torch.float32
            or nodes.ndim != 3 or nodes.shape[1:] != (4,2)
            or covariance.shape != (*nodes.shape[:2],2,2) or weights.shape != nodes.shape[:2]
            or any(x.device != nodes.device or not bool(torch.isfinite(x).all()) for x in (nodes,covariance,weights))
            or bool((weights <= 0).any()) or not torch.equal(weights.sum(-1),torch.ones_like(weights[:,0]))):
        raise ValueError('finite historical float32 four-component mixture required')
    TRACE._covariance(covariance)
    means, cov, w = nodes.double(), covariance.double(), weights.double()
    mean = (w[...,None]*means).sum(1)
    delta = means-mean[:,None]
    total = (w[...,None,None]*(cov+delta[..., :,None]*delta[...,None,:])).sum(1)
    return mean,total


def prepare(observations, query_partition, key_partition, anchor):
    """Prepare old moments once, unchanged old full4, and one common bias."""
    if any(not bool(torch.isfinite(x).all()) for x in observations.values()
           if isinstance(x,torch.Tensor)):
        raise ValueError('finite predicted observations required')
    primary = ANCHOR.prepare_query(observations,query_partition,'anchor_trace',anchor)
    full4 = OLD.prepare_query(observations,query_partition,'gaussian_full4')
    key = OLD.prepare_key(observations,key_partition,'gaussian_full1')
    if not torch.equal(primary['features'],full4['features']):
        raise ValueError('parent content changed between geometries')
    content = primary['features']@key['features'].T
    tau = TRACE.trace_temperature(primary['tau_cov'],.25,10.).float()
    bias = tau[:,None]*content
    if content.dtype != torch.float32 or not bool(torch.isfinite(bias).all()):
        raise ValueError('finite common float32 content arithmetic required')
    mean,cov = mixture_moments(full4['nodes'],full4['node_cov'],full4['node_weights'])
    return dict(mu32=primary['old_mean'],cov32=primary['original_cov'],a64=anchor,
        anchor_cov64=primary['tau_cov'],delta64=primary['anchor_delta'],
        q_features32=primary['features'],k_features32=key['features'],content32=content,tau32=tau,bias32=bias,
        full4_nodes32=full4['nodes'],full4_node_cov32=full4['node_cov'],full4_weights32=full4['node_weights'],
        mixture_mean64=mean,mixture_cov64=cov,mean_residual64=mean-primary['old_mean'].double(),
        cov_residual64=cov-primary['original_cov'].double(),key=key,full1=primary,full4=full4)


def moment_diagnostics(prepared):
    result = {}
    for name,reference in (('mean','mu32'),('cov','cov32')):
        residual=prepared[name+'_residual64']
        result[name+'_max_absolute_residual']=float(residual.abs().max())
        result[name+'_max_normalized_residual']=float((residual.abs()/(1+prepared[reference].double().abs())).max())
        result[name+'_bitwise_equal']=bool((residual==0).all())
    result['components_repaired']=False
    return result


def raw_mass(prepared,geometry):
    if geometry not in GEOMETRIES:raise ValueError('undeclared diagnostic geometry')
    key=prepared['key']
    if geometry=='full4':
        child=OLD._gaussian_mass(prepared['full4_nodes32'],prepared['full4_node_cov32'].double(),key,.25)
        raw=(child*prepared['full4_weights32'].double()[...,None]).sum(1)
    else:
        mean=prepared['mu32'] if geometry=='mu_c' else prepared['a64']
        cov=prepared['anchor_cov64'] if geometry=='a_ca' else prepared['cov32'].double()
        raw=OLD._gaussian_mass(mean,cov,key,.25)
    if (raw.dtype!=torch.float64 or raw.shape!=prepared['content32'].shape
            or not bool(torch.isfinite(raw).all()) or bool((raw<0).any())):
        raise ValueError('invalid unfiltered native mass')
    return raw


def probabilities(prepared,geometry,raw=None):
    if geometry not in GEOMETRIES:raise ValueError('undeclared diagnostic geometry')
    raw=raw_mass(prepared,geometry) if raw is None else raw
    if (raw.dtype!=torch.float64 or raw.shape!=prepared['content32'].shape
            or not bool(torch.isfinite(raw).all()) or bool((raw<0).any())):
        raise ValueError('invalid supplied raw mass')
    logits,diagnostics=ANCHOR._geometry(raw,dict(mean=prepared['a64']),prepared['key'])
    result={}
    for mode in ('zero','content'):
        bias=torch.zeros_like(prepared['bias32']) if mode=='zero' else prepared['bias32']
        probability=(logits+bias.double()).softmax(-1)
        normalization=float((probability.sum(-1)-1).abs().max())
        if not bool(torch.isfinite(probability).all()) or bool((probability<0).any()) or normalization>1.e-12:
            raise ValueError('invalid diagnostic probability')
        result[mode]=probability
    return result,dict(diagnostics,all_finite=True,
        max_normalization_error=max(float((p.sum(-1)-1).abs().max()) for p in result.values()))
