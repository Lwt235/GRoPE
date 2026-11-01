"""Shared constructor for the explicitly named experimental PE families."""
from models.gaussian_pe import GaussianPEMatcher
from models.gaussian_world_pe import GaussianWorldMatcher, METHODS as WORLD_METHODS
from models.gaussian_unitary_pe import GaussianUnitaryMatcher, METHODS as UNITARY_METHODS


def make_matcher(method, features=256, resolution=0.):
    if method in WORLD_METHODS:
        return GaussianWorldMatcher(method)
    if method in UNITARY_METHODS:
        return GaussianUnitaryMatcher(method,resolution)
    return GaussianPEMatcher(method,features,resolution)
