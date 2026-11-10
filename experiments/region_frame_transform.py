"""Deterministic coordinate re-expression, with no labels or poses accepted."""
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

SALT = 'region-frame-gauge-v1:99701'


def frames(ids):
    rotation = np.empty((len(ids),2,3,3),np.float64)
    translation = np.empty((len(ids),2,3),np.float64)
    for index,identity in enumerate(ids):
        for view in range(2):
            digest = hashlib.sha256(f'{SALT}:{identity}:{view}'.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8],'little'))
            rotation[index,view] = Rotation.from_quat(rng.normal(size=4)).as_matrix()
            translation[index,view] = rng.uniform(-.5,.5,size=3)
    return rotation,translation


def transform(centers,covariance,rotation,translation):
    moved = np.einsum('bvij,bvnj->bvni',rotation,centers.astype(np.float64))+translation[:,:,None,:]
    matrix = rotation[:,:,None]
    shaped = matrix@covariance.astype(np.float64)@matrix.swapaxes(-1,-2)
    return moved.astype(centers.dtype),shaped


def reframe(cache):
    r,t = frames(cache['ids'])
    centers,covariance = transform(cache['centers'],cache['covariance'],r,t)
    return dict(cache,centers=centers,covariance=covariance),r,t
