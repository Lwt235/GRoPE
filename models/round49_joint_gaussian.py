"""Center RoPE and a conditional Gaussian prior in one attention score.

The existing quadratic positional lift is retained. There is no new matching
head, renderer, loss, trainable mixing coefficient, or physical covariance input.
"""
from models.geometry import rotary
from models.round5_cross_attention import _temperature
from models.round46_native3d_pe import sphere_frequencies
from models.round47_conditional_gaussian import (
    KINDS, ConditionalGaussianMatcher, dimension,
)

ROLES = tuple('joint_'+kind+str(dim)+'d' for dim in (3, 2) for kind in KINDS)


def base_role(role):
    if role not in ROLES:
        raise ValueError(role)
    return role.removeprefix('joint_')


class JointGaussianMatcher(ConditionalGaussianMatcher):
    def __init__(self, role):
        super().__init__(base_role(role))
        self.method = role
        if dimension(role) == 3:
            self.frequency = sphere_frequencies(24)

    def rotate_content(self, content, mean):
        return rotary(content.double(), mean.double().unsqueeze(1),
                      self.frequency.double()).to(content.dtype)

    def attention_scores(self, layer, query, key, features, query_mean, key_mean):
        q = self.rotate_content(query, query_mean)
        k = self.rotate_content(key, key_mean)
        content = _temperature(layer.log_temperature)*(q@k.transpose(-1, -2))
        position = self.position_field.bias(features, query_mean, key_mean)
        return content+position.to(content.dtype)


def make_model(role):
    return JointGaussianMatcher(role)
