"""Query-conditioned Gaussian positional channels in the complete Transformer.

One shared linear predictor produces a head's relative center and precision.
The positional log score has an exact quadratic Q/K lift. Neither a renderer
nor a geometric shortcut in the final matching head is used.
"""
import torch
from torch import nn
import torch.nn.functional as F

from models.round5_cross_attention import (
    Round5CrossAttentionMatcher, MODEL_DIM, HEADS, _temperature,
)

KINDS = ('conditional', 'diagonal', 'isotropic', 'static', 'fixed', 'free_quadratic')
ROLES = tuple(kind+str(dim)+'d' for dim in (3, 2) for kind in KINDS)+('nope',)


def dimension(role):
    return 2 if role.endswith('2d') or role == 'nope' else 3


def split_role(role):
    if role not in ROLES:
        raise ValueError(role)
    return ('nope', 2) if role == 'nope' else (role[:-2], dimension(role))


class GaussianPosition(nn.Module):
    def __init__(self, role):
        super().__init__()
        self.kind, self.dim = split_role(role)
        self.width = 4.0
        self.size = self.dim + (1 if self.kind == 'isotropic' else self.dim)
        if self.kind in ('conditional', 'static', 'free_quadratic'):
            self.size += self.dim*(self.dim-1)//2
        if self.kind == 'static':
            self.raw = nn.Parameter(torch.zeros(HEADS, self.size))
        elif self.kind not in ('fixed', 'nope'):
            self.predictor = nn.Linear(MODEL_DIM, HEADS*self.size)
            nn.init.zeros_(self.predictor.weight)
            nn.init.zeros_(self.predictor.bias)

    def coefficients(self, features):
        """Return P and b for -.5*delta^T P delta + b^T delta.

        Gaussian b=P*m; the equally sized free-quadratic control has no SPD
        constraint, but matches the Gaussian value and first derivative at zero.
        """
        batch, count, _ = features.shape
        if self.kind in ('fixed', 'nope'):
            raw = features.new_zeros(batch, HEADS, count, self.size).double()
        elif self.kind == 'static':
            raw = self.raw.double()[None, :, None].expand(batch, -1, count, -1)
        else:
            raw = self.predictor(features).reshape(batch, count, HEADS, self.size)
            raw = raw.permute(0, 2, 1, 3).double()
        mean = 4*torch.tanh(raw[..., :self.dim]/4)
        shape = 3*torch.tanh(raw[..., self.dim:]/3)
        diagonal = shape[..., :1].expand(*shape.shape[:-1], self.dim) if self.kind == 'isotropic' else shape[..., :self.dim]
        lower = torch.zeros(*shape.shape[:-1], self.dim, self.dim, dtype=raw.dtype, device=raw.device)
        if self.kind in ('conditional', 'static', 'free_quadratic'):
            i, j = torch.tril_indices(self.dim, self.dim, offset=-1, device=raw.device)
            lower[..., i, j] = shape[..., self.dim:]
        eye = torch.eye(self.dim, dtype=raw.dtype, device=raw.device)
        if self.kind == 'free_quadratic':
            u = torch.diag_embed(diagonal)+lower
            precision = (eye+u+u.transpose(-1, -2))/self.width**2
            linear = mean/self.width**2
        else:
            factor = torch.diag_embed(diagonal.exp())+lower
            precision = (factor@factor.transpose(-1, -2))/self.width**2
            linear = (precision@mean.unsqueeze(-1)).squeeze(-1)
        return precision, linear

    def vectors(self, features, query_mean, key_mean):
        """Five positional channels in 2D, nine in 3D; omit row constants."""
        precision, linear = self.coefficients(features)
        origin = query_mean.double().mean(-2, keepdim=True)
        x = (query_mean.double()-origin).unsqueeze(1)
        y = (key_mean.double()-origin).unsqueeze(1)
        first = (precision@x.unsqueeze(-1)).squeeze(-1)+linear
        i, j = torch.triu_indices(self.dim, self.dim, device=x.device)
        weights = torch.where(i == j, -.5, -1.).to(x.dtype)
        second = precision[..., i, j]*weights
        query = torch.cat((first, second), -1)
        key = torch.cat((y, y[..., i]*y[..., j]), -1)
        return query, key

    def bias(self, features, query_mean, key_mean):
        if self.kind == 'nope':
            return None
        q, k = self.vectors(features, query_mean, key_mean)
        return q@k.transpose(-1, -2)


class ConditionalGaussianMatcher(Round5CrossAttentionMatcher):
    def __init__(self, role):
        super().__init__('point')
        self.method = role
        self.position_field = GaussianPosition(role)

    def attention_scores(self, layer, query, key, features, query_mean, key_mean):
        score = _temperature(layer.log_temperature)*(query@key.transpose(-1, -2))
        bias = self.position_field.bias(features, query_mean, key_mean)
        return score if bias is None else score+bias.to(score.dtype)

    def encode(self, data, zero_values=False):
        q = self.input_projection(data['q'])
        k = self.input_projection(data['k'])
        for layer in self.layers:
            qq, qk, qv = layer._project(q)
            kq, kk, kv = layer._project(k)
            qq, qk = F.normalize(qq, dim=-1), F.normalize(qk, dim=-1)
            kq, kk = F.normalize(kq, dim=-1), F.normalize(kk, dim=-1)
            sq = self.attention_scores(layer, qq, kk, layer.attention_norm(q), data['qmean'], data['k_to_q'])
            sk = self.attention_scores(layer, kq, qk, layer.attention_norm(k), data['kmean'], data['q_to_k'])

            def aggregate(score, value):
                if zero_values:
                    value = value*0
                a = score.softmax(-1)@value
                return layer.output(a.transpose(1, 2).reshape(a.shape[0], a.shape[2], MODEL_DIM))

            q, k = q+aggregate(sq, kv), k+aggregate(sk, qv)
            q = q+layer.ffn(layer.ffn_norm(q))
            k = k+layer.ffn(layer.ffn_norm(k))
        return q, k


def make_model(role):
    return ConditionalGaussianMatcher(role)
