"""Round028 source-resolved and matched-mean shared decoder initialization."""
import torch
from models.native_track_memory import TrackMemory
class SourceResolvedMemory(TrackMemory):
    def __init__(self,basis,mean):
        super().__init__()
        with torch.no_grad():
            self.source.weight.zero_()
            for v in range(4):self.source.weight[:,v*256:v*256+255].copy_(basis.float())
            self.source.bias.copy_(mean.float())
