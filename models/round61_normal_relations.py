"""Two prospective controls over the SAME observed Gaussian supports."""
import torch
from torch import nn
from models.round58_support_pe import DIM, SupportMatcher, gaussian_terms
from models.round59_relation_controls import make_model as previous_model

NEW_ROLES = ('normal_ppf_mlp', 'axial_mlp')
REFERENCE_ROLES = ('invariant_mlp', 'distance_mlp', 'descriptor_mlp',
                   'angle_descriptor_mlp', 'matrix_mlp')
ROLES = REFERENCE_ROLES + NEW_ROLES
ROOT_TOLERANCE = 1e-6


def spectral_geometry(covariance):
    eigen, vectors = torch.linalg.eigh(covariance.double())
    selected = eigen-eigen[..., :1] <= ROOT_TOLERANCE*eigen[..., -1:]
    count = selected.sum(-1)
    projector = (vectors*selected.unsqueeze(-2))@vectors.transpose(-1, -2)
    normal = projector/count[..., None, None]
    low = (eigen*selected).sum(-1)/count
    high = (eigen*(~selected)).sum(-1)/(3-count).clamp_min(1)
    identity = torch.eye(3, device=eigen.device, dtype=eigen.dtype)
    axial = low[..., None, None]*projector+high[..., None, None]*(identity-projector)
    # At a triple root, use an isotropic matrix independent of the eigenbasis.
    axial = torch.where((count == 3)[..., None, None], eigen.mean(-1)[..., None, None]*identity, axial)
    return eigen, normal, axial, count


def pair_inputs(role, centers, covariance):
    if role not in NEW_ROLES:
        raise ValueError(role)
    eigen, normal, axial, _ = spectral_geometry(covariance)
    c = centers.double()
    delta = c.unsqueeze(-2)-c.unsqueeze(-3)
    distance = delta.norm(dim=-1)
    spectrum = .5*eigen.div(.04).log()
    n = centers.shape[-2]
    first = spectrum.unsqueeze(-2).expand(*spectrum.shape[:-1], n, 3)
    second = spectrum.unsqueeze(-3).expand(*spectrum.shape[:-2], n, n, 3)
    if role == 'normal_ppf_mlp':
        unit = delta/distance.clamp_min(1e-30).unsqueeze(-1)
        a = torch.einsum('...ijd,...ide,...ije->...ij', unit, normal, unit)
        b = a.transpose(-1, -2)
        cross = torch.einsum('...ide,...jed->...ij', normal, normal)
        relations = torch.stack((a, b, cross), -1)
    else:
        _, mi, mj, ti, tj = gaussian_terms(centers, axial)
        relations = torch.stack((mi, mj, ti, tj), -1).clamp_min(0).log1p()
    return torch.cat((distance.unsqueeze(-1)/.2, first, second, relations), -1).to(centers.dtype)


class NormalPosition(nn.Module):
    def __init__(self, role):
        super().__init__()
        self.role = role
        self.mlp = nn.Sequential(nn.Linear(10 if role == 'normal_ppf_mlp' else 11, DIM),
                                 nn.GELU(), nn.Linear(DIM, DIM, bias=False))

    def forward(self, centers, covariance):
        return self.mlp(pair_inputs(self.role, centers, covariance))


class NormalMatcher(SupportMatcher):
    def __init__(self, role):
        if role not in NEW_ROLES:
            raise ValueError(role)
        super().__init__('nope')
        self.role = role
        self.position = NormalPosition(role)


def make_model(role):
    return NormalMatcher(role) if role in NEW_ROLES else previous_model(role)
