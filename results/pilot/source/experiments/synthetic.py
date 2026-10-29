"""Controlled paired views, NOT a rendered image or a 3DGS benchmark.

Objects have shared latent centers, appearance and support covariances. Each view
gets independent noisy observations. This deliberately tests when *consistent
support* contains extra correspondence information. Independent support is a
negative control, since image patches need not have identical physical support.
Correspondence labels are used only in the loss/evaluation, never descriptors.
"""
import torch
import torch.nn.functional as F
from models.geometry import (covariance, logcov_descriptor, pairwise_squared,
                             project, project_covariance)


SCENARIOS = {
    'id': {},
    'depth_noise': {'depth_noise': .45},
    'shape_noise': {'shape_noise': .9},
    'shape_shuffled': {'shuffle_shape': True},
    'independent_support': {'independent_support': True},
    'shared_scale': {'shared_scale': True},
    'shared_normal': {'shared_normal': True},
    'wide_view': {'angle': 60.},
}


def rotations(shape, generator):
    q, _ = torch.linalg.qr(torch.randn(*shape, 3, 3, generator=generator))
    # QR sign is irrelevant for covariance; enforce SO(3) for affine diagnostics.
    q[..., :, 2] *= torch.linalg.det(q).unsqueeze(-1)
    return q


def make_dataset(seed, scenes, tokens=32, scenario='id'):
    cfg = {'depth_noise': .12, 'shape_noise': .15, 'angle': 30.,
           **SCENARIOS.get(scenario, {})}
    gen = torch.Generator().manual_seed(seed)
    randn = lambda *s: torch.randn(*s, generator=gen)
    # Four nearby points per cluster make coarse center matching ambiguous.
    clusters = randn(scenes, tokens // 4, 3) * torch.tensor([.65, .4, .5])
    clusters[..., 2] += 5
    mu = clusters.repeat_interleave(4, 1) + randn(scenes, tokens, 3) * .065
    prototype = randn(scenes, 4, 16)
    ids = torch.randint(0, 4, (scenes, tokens), generator=gen)
    appearance = prototype.gather(1, ids[..., None].expand(-1, -1, 16)) + .15 * randn(scenes, tokens, 16)
    # Shapes drawn independently of class, center and candidate order.
    rot = rotations((scenes, tokens), gen)
    logs = torch.tensor([-1.9, -2.7, -4.3]) + randn(scenes, tokens, 3) * .35
    if cfg.get('shared_scale'):
        logs = torch.tensor([-1.9, -2.7, -4.3]).expand_as(logs).clone()
    if cfg.get('shared_normal'):
        # Common complete frame is a stronger normal-sharing negative control.
        rot = torch.eye(3).expand_as(rot).clone()

    camera_q = torch.eye(3).repeat(scenes, 1, 1)
    a = torch.deg2rad(torch.tensor(cfg['angle']))
    camera_k = torch.tensor([[a.cos(), 0., a.sin()], [0., 1., 0.], [-a.sin(), 0., a.cos()]]).repeat(scenes, 1, 1)
    # Rotate both cameras about scene center, keeping all points in front.
    pivot = torch.tensor([0., 0., 5.]).expand(scenes, 3)
    tq = pivot - (camera_q @ pivot[..., None]).squeeze(-1)
    tk = pivot - (camera_k @ pivot[..., None]).squeeze(-1)

    observed_mu, observed_cov, err_cov, frames, normals, sigmas = [], [], [], [], [], []
    for view, (cam, trans) in enumerate([(camera_q, tq), (camera_k, tk)]):
        # Camera ray converted to world; only depth is noisy, observed u/v exact.
        center_cam = (cam @ mu.transpose(1, 2)).transpose(1, 2) + trans[:, None]
        ray_cam = center_cam / center_cam[..., 2:]
        ray_world = (cam.transpose(-1, -2) @ ray_cam.transpose(1, 2)).transpose(1, 2)
        depth_error = cfg['depth_noise'] * randn(scenes, tokens, 1)
        observed_mu.append(mu + ray_world * depth_error)
        err_cov.append(ray_world[..., :, None] @ ray_world[..., None, :] * cfg['depth_noise']**2 + torch.eye(3)*1e-6)
        nr = rot.clone()
        nl = logs.clone()
        if cfg.get('independent_support') and view == 1:
            independent_gen = torch.Generator().manual_seed(seed+99173)
            nr = rotations((scenes, tokens), independent_gen)
            nl = torch.tensor([-1.9, -2.7, -4.3]) + torch.randn(scenes, tokens, 3, generator=independent_gen) * .35
        # Noisy near-identity rotation and log-scales. QR introduces column signs,
        # intentionally not used for the matching descriptors.
        perturb = torch.eye(3) + cfg['shape_noise'] * randn(scenes, tokens, 3, 3)
        perturb = torch.linalg.qr(perturb)[0]
        nr = perturb @ nr
        nl = nl + cfg['shape_noise'] * randn(scenes, tokens, 3)
        cov = covariance(nr, nl.exp())
        if cfg.get('shuffle_shape') and view == 1:
            # Within-scene permutation preserves the covariance marginal exactly.
            shuffle_gen = torch.Generator().manual_seed(seed+88173)
            perm = torch.rand(scenes, tokens, generator=shuffle_gen).argsort(-1)
            cov = cov.gather(1, perm[..., None, None].expand(-1, -1, 3, 3))
        observed_cov.append(cov)
        frames.append(nr)
        ev, vec = torch.linalg.eigh(cov)
        normal = vec[..., :, 0]
        normals.append((normal[..., :, None] @ normal[..., None, :]).flatten(-2))
        sigmas.append(ev.log())

    fq = appearance + .35 * randn(scenes, tokens, 16)
    fk = appearance + .35 * randn(scenes, tokens, 16)
    mq, mk = observed_mu
    cq, ck = observed_cov
    # Every candidate is projected into the same query camera before comparison.
    uq, jq, zq = project(mq, camera_q, tq)
    uk_to_q, jk_to_q, zk = project(mk, camera_q, tq)
    assert min(zq.min(), zk.min()) > 0
    pq = project_covariance(jq, cq)
    pk = project_covariance(jk_to_q, ck)
    output = {
        'q': fq, 'k': fk, 'mq': mq, 'mk': mk, 'cq': cq, 'ck': ck,
        'uqcov': err_cov[0], 'ukcov': err_cov[1],
        'center': pairwise_squared(mq, mk) / .15**2,
        'reprojection': pairwise_squared(uq, uk_to_q) / .03**2,
        'normal': pairwise_squared(normals[0], normals[1]),
        'scale': pairwise_squared(sigmas[0], sigmas[1]) / 3,
        'shape': pairwise_squared(logcov_descriptor(cq), logcov_descriptor(ck)) / 6,
        'projected': pairwise_squared(logcov_descriptor(pq), logcov_descriptor(pk)) / 3,
    }
    # Capacity/control: same scalar shape weight, independent Gaussian descriptors.
    output['random'] = pairwise_squared(randn(scenes, tokens, 6), randn(scenes, tokens, 6)) / 6
    # Randomize candidate order for every scene. labels never enter model inputs.
    perm = torch.rand(scenes, tokens, generator=gen).argsort(-1)
    output['labels'] = perm.argsort(-1)
    for key in ['k', 'mk', 'ck', 'ukcov']:
        x = output[key]
        index = perm.reshape(scenes, tokens, *([1] * (x.ndim-2))).expand_as(x)
        output[key] = x.gather(1, index)
    for key in ['center', 'reprojection', 'normal', 'scale', 'shape', 'projected', 'random']:
        output[key] = output[key].gather(2, perm[:, None].expand(-1, tokens, -1))
    return output
