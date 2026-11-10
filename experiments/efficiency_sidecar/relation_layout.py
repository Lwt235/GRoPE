"""One private shared-hidden layout copy; same PyTorch contraction and formula."""
from types import MethodType


def projection(self,hidden):
    packed=hidden.transpose(-1,-2).contiguous().transpose(-1,-2)
    return self._layout_previous_forward(packed)


def adapt_layout(model):
    assert getattr(model,'_efficiency_sidecar',False) and not getattr(model,'_relation_layout',False)
    last=model.position.mlp[-1]
    last._layout_previous_forward=last.forward
    last.forward=MethodType(projection,last)
    model._relation_layout=True
    return model
