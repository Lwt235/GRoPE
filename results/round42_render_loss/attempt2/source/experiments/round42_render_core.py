"""Direct color fusion and its exact chain-rule gradient, without retained
cross-view attention graphs. The GS readout/PE architecture is unchanged.
"""
import torch
from experiments.round41_gs_data import TRAIN_VIEWS,query
from experiments import round9_gs_readout as R9


def sources_for(target):
    return tuple(i for i in TRAIN_VIEWS if i!=target)


def source_counts(data,target):
    return sum(data['views'][i]['visible'].long() for i in sources_for(target))


def training_target_views(data):
    """Common support for the two losses; no outcome-dependent filtering."""
    eligible=tuple(i for i in TRAIN_VIEWS
        if bool((data['views'][i]['visible']&(source_counts(data,i)>0)).any()))
    if not eligible:raise ValueError('no shared center supervision in any training view')
    return eligible


@torch.no_grad()
def fused_colors(model,data,target,chunk=256):
    counts=source_counts(data,target)
    original=data['state']['color_logits'].sigmoid()
    sums=torch.zeros_like(original,dtype=torch.float64)
    for source in sources_for(target):
        key=data['views'][source];ids=torch.where(key['visible'])[0]
        encoded=model.encode(key['feature'],key['anchor'],key['cov'])
        for offset in range(0,len(ids),chunk):
            ix=ids[offset:offset+chunk]
            q=query(data,source,ix,model,target=target)
            rgb=model.logits(q,key,key_encoded=encoded).softmax(-1)@key['rgb']
            sums[ix]+=rgb.double()
    return torch.where((counts>0)[:,None],sums/counts.clamp_min(1)[:,None],original.double()).to(original.dtype),counts


def accumulate_color_gradient(model,data,target,color_gradient,counts,chunk=256):
    """Caller zeros grads once and clips/steps once, outside this function."""
    observations=0;chunks=0
    active=(color_gradient!=0).any(-1)&(counts>0)
    for source in sources_for(target):
        key=data['views'][source];ids=torch.where(key['visible']&active)[0]
        if not len(ids):continue
        encoded=model.encode(key['feature'],key['anchor'],key['cov'])
        for offset in range(0,len(ids),chunk):
            ix=ids[offset:offset+chunk];q=query(data,source,ix,model,target=target)
            rgb=model.logits(q,key,key_encoded=encoded).softmax(-1)@key['rgb']
            coefficient=(color_gradient[ix].double()/counts[ix,None]).to(rgb.dtype).detach()
            (rgb*coefficient).sum().backward(retain_graph=offset+chunk<len(ids))
            observations+=len(ix);chunks+=1
    return dict(active_primitives=int(active.sum()),source_observations=observations,chunks=chunks)


def center_loss(colors,data,target,counts):
    valid=data['views'][target]['visible']&(counts>0)
    if not valid.any():raise ValueError('no context center supervision')
    return (colors[valid]-data['views'][target]['center_rgb'][valid]).square().mean()


def render_image(colors,data,target):
    camera=data['cameras'][target]
    return R9.render(data['state'],camera['K'],camera['R'],camera['t'],colors=colors)[0]


def image_loss(colors,data,target):
    predicted=render_image(colors,data,target)
    return (predicted-data['views'][target]['rgb'].reshape(144,192,3)).square().mean()


def color_gradient(colors,data,target,counts,objective):
    leaf=colors.detach().requires_grad_(True)
    if objective=='center':loss=center_loss(leaf,data,target,counts)
    elif objective=='image':loss=image_loss(leaf,data,target)
    else:raise ValueError(objective)
    gradient=torch.autograd.grad(loss,leaf)[0]
    return loss.detach(),gradient.detach()
