"""Known Gaussian kernel-mean position bias; fixed R39 exploratory design."""
from __future__ import annotations
import math
import numpy as np
from scipy.special import logsumexp

ROLES = ('full', 'point', 'isotropic')
OBS_FIELDS = ('q_predicted_depth', 'k_predicted_depth', 'q_K', 'k_K', 'q_to_k_transform')


def bounds(size):
    if size not in (8, 24):
        raise ValueError('fixed native partitions only')
    return np.asarray([[x, y, x+size, y+size] for y in range(0, 144, size)
                       for x in range(0, 192, size)], np.int64)


def dense(depth, K):
    depth = np.asarray(depth, np.float64)
    K = np.asarray(K, np.float64)
    if (depth.shape != (144, 192) or K.shape != (3, 3)
            or not np.isfinite(K).all() or not np.array_equal(K[2], [0., 0., 1.])):
        raise ValueError('standard finite camera and full depth raster required')
    z = np.nan_to_num(depth, nan=10., posinf=20., neginf=.1).clip(.1, 20.)
    inverse = np.linalg.inv(K)
    yy, xx = np.mgrid[:144, :192]
    rays = np.stack((xx, yy, np.ones_like(xx)), -1) @ inverse.T
    points = rays*z[..., None]
    basis = inverse[:, :2]
    within = z[..., None, None]**2*(basis @ basis.T)/12.
    pixel_length2 = z*z*np.square(basis).sum()/2.
    return z, points, within, pixel_length2


def support(depth, K, size):
    z, points, within, lengths = dense(depth, K)
    cells = bounds(size)
    center = (cells[:, :2]+cells[:, 2:]-1.)/2.
    xy = np.floor(center).astype(np.int64)
    f = center-xy
    zc = np.zeros(len(cells), np.float64)
    for dy in (0, 1):
        for dx in (0, 1):
            zc += z[xy[:, 1]+dy, xy[:, 0]+dx]*(f[:, 0] if dx else 1-f[:, 0])*(f[:, 1] if dy else 1-f[:, 1])
    anchor = (np.column_stack((center, np.ones(len(center)))) @ np.linalg.inv(K).T)*zc[:, None]
    covariance = []
    for a, (x0, y0, x1, y1) in zip(anchor, cells):
        offsets = (points[y0:y1, x0:x1]-a).reshape(-1, 3)
        c = offsets.T @ offsets/len(offsets)+within[y0:y1, x0:x1].mean((0, 1))
        covariance.append(c)
    return anchor, np.asarray(covariance), lengths


def prepare(obs):
    if set(obs) != set(OBS_FIELDS):
        raise ValueError('exact predicted-observation allowlist required')
    qa, qc, ql = support(obs['q_predicted_depth'], obs['q_K'], 24)
    ka, kc, kl = support(obs['k_predicted_depth'], obs['k_K'], 8)
    T = np.asarray(obs['q_to_k_transform'], np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.array_equal(T[3], [0., 0., 0., 1.]):
        raise ValueError('finite homogeneous camera transform required')
    R = T[:3, :3]
    if np.max(abs(R @ R.T-np.eye(3))) > 1e-6 or abs(np.linalg.det(R)-1) > 1e-6:
        raise ValueError('supplied transform is not a rigid camera transform')
    gamma = np.asarray(4.*np.median(np.concatenate((ql.ravel(), kl.ravel()))))
    result = dict(q_anchor=qa @ R.T+T[:3, 3], q_cov=R @ qc @ R.T,
                  k_anchor=ka, k_cov=kc, gamma=gamma)
    if not all(np.isfinite(v).all() for v in result.values()) or gamma <= 0:
        raise ValueError('nonfinite prepared geometry or nonpositive base metric')
    return result


def covariance_role(covariance, role):
    c = np.asarray(covariance, np.float64)
    if c.ndim != 3 or c.shape[1:] != (3, 3) or not np.isfinite(c).all():
        raise ValueError('finite 3D covariance bank required')
    scale = np.maximum(1., np.max(abs(c), axis=(1, 2)))
    if (np.max(abs(c-c.transpose(0, 2, 1)), axis=(1, 2)) > 1e-12*scale).any():
        raise ValueError('asymmetric support')
    if (np.linalg.eigvalsh(c)[:, 0] < -1e-12*scale).any():
        raise ValueError('non-PSD support; no repair')
    if role == 'full': return c
    if role == 'point': return np.zeros_like(c)
    if role == 'isotropic': return np.trace(c, axis1=1, axis2=2)[:, None, None]*np.eye(3)/3.
    raise ValueError('undeclared role')


def score(q_anchor, q_cov, k_anchor, k_cov, gamma, content, role='full'):
    q, k = np.asarray(q_anchor, np.float64), np.asarray(k_anchor, np.float64)
    qc, kc = covariance_role(q_cov, role), covariance_role(k_cov, role)
    gamma = float(gamma)
    content = np.asarray(content, np.float64)
    if (q.shape != (len(qc), 3) or k.shape != (len(kc), 3)
            or content.shape != (len(q), len(k)) or gamma <= 0
            or not math.isfinite(gamma) or not all(np.isfinite(v).all() for v in (q, k, content))):
        raise ValueError('finite consistent pair inputs required')
    S = qc[:, None]+kc[None]+gamma*np.eye(3)
    S = .5*(S+S.swapaxes(-1, -2))
    L = np.linalg.cholesky(S)
    delta = q[:, None]-k[None]
    white = np.linalg.solve(L, delta[..., None])[..., 0]
    logdet = 2.*np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1)
    lograw = math.log(9.)+.5*(3.*math.log(gamma)-logdet-np.square(white).sum(-1))
    logfloored = np.logaddexp(lograw, math.log(9.e-30))
    logits = logfloored+content
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    p = np.exp(logp)
    if not all(np.isfinite(v).all() for v in (lograw, logfloored, logp, p)) or np.max(abs(p.sum(-1)-1.)) > 2e-12:
        raise ValueError('nonfinite output or normalization failure')
    return dict(lograw=lograw, logfloored=logfloored, logp=logp, p=p,
                winner=p.argmax(-1).astype(np.int32))
