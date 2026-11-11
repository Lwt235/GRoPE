"""The same residual read, with richer reference values or a matched local MLP."""
import torch
from torch import nn

from models.gaussian_memory_view_token import ViewTokenAdapter


class DirectReferenceAdapter(ViewTokenAdapter):
    def __init__(self, variant, center_variance=.7146816253662109):
        assert variant in ('direct', 'wide')
        super().__init__('center' if variant == 'direct' else 'local', center_variance)
        self.family = variant
        if variant == 'direct':
            self.key = nn.Linear(2048 + 10, 64)
            self.value = nn.Linear(2048, 64)
        else:
            self.width = 125
            self.query = nn.Linear(2048 + 5, self.width)
            self.local = nn.Linear(self.width, self.width)
            self.output = nn.Linear(self.width, 2048)
            nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def copy_common_initialization(self, original):
        assert self.family == 'direct'
        state = self.state_dict()
        with torch.no_grad():
            for key in state:
                if state[key].shape == original[key].shape:
                    state[key].copy_(original[key])
            self.key.weight[:, -10:].copy_(original['key.weight'][:, -10:])

