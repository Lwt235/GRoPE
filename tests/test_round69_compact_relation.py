import numpy as np
import torch
from models.round58_support_pe import SupportPosition
from models.round68_fixed_region import FixedRegionMatcher
from models.round69_compact_relation import CompactMatcher,descriptors,ROLES


def test_exact_existing_descriptor_columns():
    torch.manual_seed(69)
    centers=torch.randn(2,2,7,3)
    a=torch.randn(2,2,7,3,3,dtype=torch.float64)
    covariance=a@a.transpose(-1,-2)+.1*torch.eye(3,dtype=torch.float64)
    model=SupportPosition('invariant_mlp');captured=[]
    handle=model.mlp.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    model(centers,covariance);handle.remove()
    for role,columns in [('relative5',[0,7,8,9,10]),('metric3',[0,7,8]),('spectrum7',list(range(7)))]:
        assert torch.equal(descriptors(role,centers,covariance),captured[0][...,columns])


def test_common_initialization_identical_to_r68():
    for seed in (11,22,33):
        torch.manual_seed(seed);parent=FixedRegionMatcher('gaussian_relation').state_dict()
        for role in ROLES:
            torch.manual_seed(seed);child=CompactMatcher(role).state_dict()
            assert {k for k in child if not k.startswith('position.')}=={k for k in parent if not k.startswith('position.')}
            for key in parent:
                if not key.startswith('position.'):assert torch.equal(parent[key],child[key]),key
