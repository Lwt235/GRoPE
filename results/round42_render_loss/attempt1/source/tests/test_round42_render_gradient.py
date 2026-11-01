"""Check the split chain rule against a separately assembled monolithic graph."""
import unittest
import numpy as np
import torch
from models.round41_gs_readout import GSReadout
from experiments.round41_gs_data import TRAIN_VIEWS,query
from experiments.round42_render_core import (fused_colors,source_counts,
    accumulate_color_gradient,center_loss,sources_for)


def fixture():
    generator=torch.Generator().manual_seed(4567)
    def rand(*shape):return torch.randn(*shape,generator=generator,dtype=torch.float64)
    views=[];n,m=5,7
    for i in range(24):
        c=rand(n,3,3)*.1;kc=rand(m,3,3)*.1
        visible=torch.tensor([True,i%2==0,i%3!=0,True,False])
        feature=rand(n,128)*visible[:,None]
        views.append(dict(feature=rand(m,128),anchor=rand(m,3),cov=kc@kc.transpose(-1,-2)+.01*torch.eye(3),
            xy=rand(m,2),rgb=rand(m,3).sigmoid(),visible=visible,center_feature=feature,center_rgb=rand(n,3).sigmoid(),
            q_anchor=rand(n,3),q_cov=c@c.transpose(-1,-2)+.01*torch.eye(3),q_xy=rand(n,2),
            q_projected_cov=torch.eye(2,dtype=torch.float64).repeat(n,1,1)*.3))
    data=dict(views=views,state=dict(color_logits=rand(n,3)),
        train_sum=sum(views[i]['center_feature'] for i in TRAIN_VIEWS),
        train_count=sum(views[i]['visible'].long() for i in TRAIN_VIEWS))
    weights=rand(6,n).softmax(-1);truth=rand(6,3).sigmoid()
    return data,weights,truth


def monolithic(model,data,target):
    """Explicit per-primitive stack/mean, separate from production fusion loop."""
    observations=[[] for _ in range(5)]
    for k in TRAIN_VIEWS:
        if k==target:continue
        ids=torch.where(data['views'][k]['visible'])[0]
        rgb=model(query(data,k,ids,model,target=target),data['views'][k])
        for index,primitive in enumerate(ids.tolist()):observations[primitive].append(rgb[index])
    return torch.stack([torch.stack(values).mean(0) if values else data['state']['color_logits'][i].sigmoid()
                        for i,values in enumerate(observations)])


class SplitColorGradientTests(unittest.TestCase):
    def test_all_roles_and_losses_match_monolithic(self):
        data,A,truth=fixture();target=0
        for role in ('point','full','decoupled'):
            for loss_name in ('center','image'):
                one=GSReadout(role,5,7).double();split=GSReadout(role,5,7).double();split.load_state_dict(one.state_dict())
                count=source_counts(data,target);reference=monolithic(one,data,target)
                loss=(A@reference-truth).square().mean() if loss_name=='image' else center_loss(reference,data,target,count)
                loss.backward()
                colors,counts=fused_colors(split,data,target,chunk=2)
                torch.testing.assert_close(colors,reference.detach(),atol=1e-12,rtol=1e-12)
                leaf=colors.detach().requires_grad_(True)
                other=(A@leaf-truth).square().mean() if loss_name=='image' else center_loss(leaf,data,target,counts)
                g=torch.autograd.grad(other,leaf)[0]
                accumulate_color_gradient(split,data,target,g,counts,chunk=2)
                for (name,p),(other_name,q) in zip(one.named_parameters(),split.named_parameters()):
                    self.assertEqual(name,other_name)
                    torch.testing.assert_close(p.grad,q.grad,atol=1e-11,rtol=1e-9,msg=f'{role}/{loss_name}/{name}')

    def test_target_view_exclusion_and_fallback(self):
        data,_,_=fixture();model=GSReadout('decoupled',5,7).double()
        before,count=fused_colors(model,data,0,chunk=2)
        self.assertEqual(int(count[-1]),0)
        torch.testing.assert_close(before[-1],data['state']['color_logits'][-1].sigmoid())
        change=torch.ones_like(data['views'][0]['center_feature'])*100
        data['views'][0]['center_feature']+=change;data['train_sum']+=change
        data['views'][0]['rgb']=torch.zeros_like(data['views'][0]['rgb'])
        after,_=fused_colors(model,data,0,chunk=3)
        torch.testing.assert_close(after,before,rtol=1e-11,atol=1e-11)

    def test_zero_color_derivative_has_no_model_gradient(self):
        data,_,_=fixture();model=GSReadout('decoupled',5,7).double()
        colors,counts=fused_colors(model,data,0)
        stat=accumulate_color_gradient(model,data,0,torch.zeros_like(colors),counts,chunk=2)
        self.assertEqual(stat['source_observations'],0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
