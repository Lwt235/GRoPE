"""Deterministic float64 falsification probes; no training/data downloads."""
import argparse
import json
from pathlib import Path
import torch
from models.geometry import (affine_frame, covariance, expected_phase,
                             logcov_descriptor, project, project_covariance,
                             tangent_transfer)


def run(seed=20261029):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    n = 256
    rot = torch.linalg.qr(torch.randn(n, 3, 3))[0]
    scales = torch.exp(torch.randn(n, 3) * .8 - 2)
    mean = torch.randn(n, 3)
    mean[:, 2] = mean[:, 2].abs() + 4
    cov = covariance(rot, scales)
    f = affine_frame(mean, rot, scales)
    signs = torch.diag(torch.tensor([-1., -1., 1.]))
    fg = affine_frame(mean, rot @ signs, scales)
    original = torch.linalg.solve(f[::2], f[1::2])
    changed = torch.linalg.solve(fg[::2], f[1::2])
    cov_g = covariance(rot @ signs, scales)
    q, k = torch.randn(128, 4), torch.randn(128, 4)
    score = torch.einsum('ni,nij,nj->n', q, original, k)
    score_g = torch.einsum('ni,nij,nj->n', q, changed, k)
    world = torch.eye(4)
    world[:3, :3] = torch.linalg.qr(torch.randn(3, 3))[0]
    world[:3, 3] = torch.randn(3)
    global_relative = torch.linalg.solve(world @ f[::2], world @ f[1::2])
    gauge = {
        'same_covariance_max_abs': (cov - cov_g).abs().max().item(),
        'relative_matrix_gauge_change_median_fro': torch.linalg.matrix_norm(original - changed).median().item(),
        'fixed_features_logit_gauge_change_median_abs': (score - score_g).abs().median().item(),
        'global_rigid_relative_max_abs': (original - global_relative).abs().max().item(),
        'logcov_descriptor_gauge_max_abs': (logcov_descriptor(cov) - logcov_descriptor(cov_g)).abs().max().item(),
    }
    ra = torch.eye(3).expand(n, 3, 3)
    angle = .6
    rb0 = torch.tensor([[torch.cos(torch.tensor(angle)), 0., torch.sin(torch.tensor(angle))],
                        [0., 1., 0.], [-torch.sin(torch.tensor(angle)), 0., torch.cos(torch.tensor(angle))]])
    rb = rb0.expand(n, 3, 3)
    ta = torch.zeros(n, 3)
    tb = torch.tensor([-1., 0., .2]).expand(n, 3)
    _, ja, _ = project(mean[:, None], ra, ta)
    _, jb, _ = project(mean[:, None], rb, tb)
    tangent = rot[:, :, :2] * scales[:, None, :2]
    transfer, cond = tangent_transfer(ja[:, 0], jb[:, 0], tangent)
    modified = tangent * torch.exp(torch.randn(n, 1, 2) * 2)
    transfer2, _ = tangent_transfer(ja[:, 0], jb[:, 0], modified)
    proj1 = project_covariance(ja[:, 0], cov)
    cov2 = covariance(rot, scales * torch.tensor([3., .4, 1.]))
    proj2 = project_covariance(ja[:, 0], cov2)
    tangent_result = {
        'scale_cancellation_max_relative': ((transfer-transfer2).flatten(1).norm(dim=1) / transfer.flatten(1).norm(dim=1)).max().item(),
        'projected_covariance_change_median_relative': ((proj2-proj1).flatten(1).norm(dim=1)/proj1.flatten(1).norm(dim=1)).median().item(),
        'random_tangent_condition_max': cond.max().item(),
        'grazing_sweep': [],
    }
    for degrees in [0, 60, 80, 89, 89.9, 89.99]:
        a = torch.deg2rad(torch.tensor(float(degrees)))
        basis = torch.tensor([[a.cos(), 0.], [0., 1.], [a.sin(), 0.]])
        _, j0, _ = project(torch.tensor([[[0., 0., 5.]]]), torch.eye(3)[None], torch.zeros(1, 3))
        tangent_result['grazing_sweep'].append({'degrees': degrees, 'condition': torch.linalg.cond(j0[0, 0] @ basis).item()})

    # Closed-form mean rotary blocks checked against independent Monte Carlo.
    m = torch.tensor([.4, -.2, .7])
    c = torch.tensor([[.04, .01, 0.], [.01, .02, 0.], [0., 0., .06]])
    freq = torch.randn(12, 3) * 3
    real, imag = expected_phase(m, c, freq)
    sample = m + torch.randn(200000, 3) @ torch.linalg.cholesky(c).T
    empirical = torch.exp(1j * (sample @ freq.T)).mean(0)
    mc_error = (empirical - torch.complex(real, imag)).abs().max().item()
    x = torch.tensor([[.2, .3, -.1]])
    cx = c[None]
    # Axis-frequency RoPE cannot observe off-diagonal covariances at all.
    cy = cx.clone()
    cy[:, 0, 1] *= -1
    cy[:, 1, 0] *= -1
    axis = torch.eye(3) * 4
    oblique = torch.tensor([[3., 3., 0.]])
    ea = torch.complex(*expected_phase(x, cx, axis))
    eb = torch.complex(*expected_phase(x, cy, axis))
    oa = torch.complex(*expected_phase(x, cx, oblique))
    ob = torch.complex(*expected_phase(x, cy, oblique))
    probability = {
        'monte_carlo_n': 200000, 'expected_phase_mc_max_error': mc_error,
        'axis_frequency_offdiagonal_difference': (ea-eb).abs().max().item(),
        'oblique_frequency_offdiagonal_difference': (oa-ob).abs().max().item(),
        'shape_as_uncertainty_attenuation': [
            {'sigma': s, 'frequency': 20, 'amplitude': float(torch.exp(torch.tensor(-.5 * (20*s)**2)))}
            for s in [.01, .05, .1, .2, .4]],
    }
    projection = []
    # Relative support size controls the validity of pinhole linearization.
    for rel in [.01, .05, .1, .2, .35]:
        center = torch.tensor([.5, .2, 5.])
        sigma = 5 * rel
        pts = center + torch.randn(100000, 3) * sigma
        valid = pts[:, 2] > 0
        # Reject near-camera rays too for finite moments; report the rejection.
        safe = pts[:, 2] > .5
        uv = pts[safe, :2] / pts[safe, 2:]
        _, jj, _ = project(center[None, None], torch.eye(3)[None], torch.zeros(1, 3))
        linear = jj[0, 0] @ (torch.eye(3)*sigma**2) @ jj[0, 0].T
        empirical_cov = torch.cov(uv.T)
        projection.append({'sigma_over_depth': rel,
                           'nonpositive_depth_fraction': (~valid).double().mean().item(),
                           'z_le_0_5_fraction': (~safe).double().mean().item(),
                           'conditional_cov_relative_error': ((linear-empirical_cov).norm()/empirical_cov.norm()).item()})
    checks = {
        'affine_global_rigid': gauge['global_rigid_relative_max_abs'] < 1e-9,
        'gauge_counterexample': gauge['same_covariance_max_abs'] < 1e-12 and gauge['fixed_features_logit_gauge_change_median_abs'] > .1,
        'covariance_gauge_safe': gauge['logcov_descriptor_gauge_max_abs'] < 1e-10,
        'tangent_scale_cancellation': tangent_result['scale_cancellation_max_relative'] < 1e-9,
        'characteristic_function_mc': mc_error < .012,
        'axis_blind_spot': probability['axis_frequency_offdiagonal_difference'] < 1e-12 and probability['oblique_frequency_offdiagonal_difference'] > .01,
    }
    assert all(checks.values()), checks
    return {'seed': seed, 'dtype': 'float64', 'checks': checks, 'affine_gauge': gauge,
            'tangent': tangent_result, 'probability': probability, 'projection': projection}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/pilot/math.json')
    args = parser.parse_args()
    result = run()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
