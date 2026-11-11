"""A Gaussian spatial preference that preserves access to every memory slot."""
import math

import torch


def soft_attention_prior(projected):
    """Use the same alpha-composited weights, with a uniform content base.

    Empty projected coverage is meaningful for an existing nonempty memory:
    it gives the exact uniform prior used by the parent content control.
    A genuinely absent memory is outside this experiment's fixed-slot setup.
    """
    weights = projected['weights']
    slots = weights.shape[-1]
    if slots == 0:
        raise ValueError('The fixed-slot experiment requires a nonempty memory.')
    real = torch.log1p(slots * weights) - math.log(slots)
    null = weights.new_full((*weights.shape[:-1], 1), -torch.inf)
    return torch.cat((real, null), -1)
