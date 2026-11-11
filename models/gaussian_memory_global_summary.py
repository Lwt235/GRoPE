"""Parameter-matched query-conditioned scene summary, shared across patches."""
import torch
from torch.nn import functional as F

from models.gaussian_memory_transport import GeometryTransport


class GlobalSummary(GeometryTransport):
    def __init__(self, center_variance=.7146816253662109):
        super().__init__('latent', 'gaussian', center_variance)

    def forward(self, tokens, data, condition='normal', return_parts=False):
        q = F.gelu(self.query(torch.cat([self.norm(tokens.float()), data['camera_query']], -1)))
        batch, patches = q.shape[:2]
        count = data['key'].shape[1]
        global_query = q.mean(1, keepdim=True).reshape(batch, 1, 4, 16).transpose(1, 2)
        key = self.key(data['key']).reshape(batch, count, 4, 16).transpose(1, 2)
        score = global_query @ key.transpose(-1, -2) / 4
        score = score * F.softplus(self.position_scale)[None, :, None, None]
        score = score + data['log_weight'][:, None, None]
        if condition == 'removed':
            score = torch.full_like(score, -torch.inf)
        background = torch.where(torch.isfinite(score).any(-1, keepdim=True),
                                 score.new_full((), -torch.inf), score.new_zeros(()))
        weights = torch.cat([score, background], -1).softmax(-1)[..., :-1]
        value = self.value(torch.cat([data['value'], data['log_depth'][..., None]], -1))
        value = value.reshape(batch, count, 4, 16).transpose(1, 2)
        summary = (weights @ value).transpose(1, 2).reshape(batch, 1, 64)
        prediction = self.output(F.gelu(self.local(q) + summary)).squeeze(-1)
        if return_parts:
            return prediction, {'summary': summary.expand(-1, patches, -1), 'weights': weights}
        return prediction
