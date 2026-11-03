"""Frozen-weight diagnostic routes around the unchanged R62 matcher."""
import torch
from models.round53_ray_frame import center_rays
from models.round62_rgb_relations import (RGBRelationMatcher, OBSERVED_FIELDS,
    common_frame, gaussian_from_raw, FLOOR, POINT_VARIANCE)

INTERVENTIONS = ('baseline', 'observed_center', 'observed_shape', 'observed_both',
                 'initial_shape', 'sphere', 'zero_relation')
OBSERVED_ROUTES = ('observed_center', 'observed_shape', 'observed_both')


class InterventionMatcher(RGBRelationMatcher):
    def __init__(self, role, intervention='baseline', capture=False):
        if role not in ('predicted_full', 'predicted_point', 'observed_full'):
            raise ValueError(role)
        if intervention not in INTERVENTIONS or (role != 'predicted_full' and intervention != 'baseline'):
            raise ValueError((role, intervention))
        if capture and (role == 'observed_full' or intervention != 'baseline'):
            raise ValueError('capture only unmodified predicted geometry')
        super().__init__(role)
        self.intervention, self.capture = intervention, capture
        self.captured, self._observed = [], None

    def encode(self, data, regime='oracle', zero_values=False, zero_relation=False):
        self.captured = []
        self._observed = ({k:data[k] for k in OBSERVED_FIELDS}
                          if self.intervention in OBSERVED_ROUTES else None)
        return super().encode(data, regime, zero_values,
                              zero_relation or self.intervention == 'zero_relation')

    def geometry(self, features, data, prefix):
        mean, covariance = super().geometry(features, data, prefix)
        if self.capture:
            raw = self.field['predictor'](features)
            local_mean, local_cov = gaussian_from_raw(raw,
                center_rays(data[prefix+'pos'], data[prefix+'K']), self.variant == 'predicted_full')
            self.captured.append(dict(layer=len(self.captured)//2, prefix=prefix,
                raw=raw.detach().cpu().numpy(), mean=local_mean.detach().cpu().numpy(),
                cov=local_cov.detach().cpu().numpy()))
        if self.intervention in OBSERVED_ROUTES:
            observed_mean, observed_cov = common_frame(self._observed[prefix+'observed_mean'],
                self._observed[prefix+'observed_cov'], data[prefix+'geom'])
            if self.intervention in ('observed_center', 'observed_both'):
                mean = observed_mean
            if self.intervention in ('observed_shape', 'observed_both'):
                covariance = observed_cov
        elif self.intervention in ('initial_shape', 'sphere'):
            diagonal = (features.new_tensor([.025, .030, .010]).square()+FLOOR**2
                        if self.intervention == 'initial_shape'
                        else features.new_full((3,), POINT_VARIANCE))
            local_cov = torch.diag(diagonal).expand_as(covariance)
            rotation = data[prefix+'geom'][:, 0, :9].reshape(-1, 3, 3)
            covariance = rotation[:, None]@local_cov@rotation[:, None].transpose(-1, -2)
        return mean, covariance
