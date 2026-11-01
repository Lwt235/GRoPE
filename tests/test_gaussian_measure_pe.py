"""Independent identities for Gaussian overlap; no training/data labels."""
import math

import pytest
import torch

from models.gaussian_measure_pe import gaussian_measure_log_bias


def data(dtype=torch.float64):
    g = torch.Generator().manual_seed(830)
    qm = torch.randn(2, 3, 2, generator=g, dtype=dtype)
    km = torch.randn(2, 4, 2, generator=g, dtype=dtype)
    qa = torch.randn(2, 3, 2, 2, generator=g, dtype=dtype)
    ka = torch.randn(2, 4, 2, 2, generator=g, dtype=dtype)
    eye = torch.eye(2, dtype=dtype)
    return qm, qa @ qa.transpose(-1, -2) + .02 * eye, km, ka @ ka.transpose(-1, -2) + .03 * eye


def reference(qm, qc, km, kc, mass, width, strength=1):
    s = qc[..., :, None, :, :] + kc[..., None, :, :, :] + width**2 * torch.eye(2, dtype=qm.dtype)
    d = qm[..., :, None, :] - km[..., None, :, :]
    mahal = (d * torch.linalg.solve(s, d.unsqueeze(-1)).squeeze(-1)).sum(-1)
    logdet = torch.linalg.slogdet(s).logabsdet
    return mass.log()[..., None, :] + strength * (-.5 * (mahal + logdet) - math.log(2 * math.pi))


def test_float64_reference_and_float32():
    qm, qc, km, kc = data()
    mass = torch.tensor([[.5, 1, 3, 7], [2, 1, 4, .2]], dtype=torch.float64)
    expected = reference(qm, qc, km, kc, mass, .4, .7)
    result = gaussian_measure_log_bias(qm, qc, km, kc, key_mass=mass, bandwidth=.4, density_strength=.7)
    torch.testing.assert_close(result, expected, rtol=2e-13, atol=2e-13)
    result32 = gaussian_measure_log_bias(*(x.float() for x in (qm, qc, km, kc)),
                                         key_mass=mass.float(), bandwidth=.4, density_strength=.7)
    torch.testing.assert_close(result32.double(), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('strength', [.3, 1., 2.7])
def test_duplicate_mass_splitting_preserves_attention(strength):
    qm, qc, km, kc = data()
    mass = torch.tensor([.4, 3., .8, 2.], dtype=qm.dtype)
    content = torch.arange(12, dtype=qm.dtype).reshape(1, 3, 4) / 5
    values = torch.arange(8, dtype=qm.dtype).reshape(4, 2) / 3
    original = (content + gaussian_measure_log_bias(qm, qc, km, kc,
        key_mass=mass, density_strength=strength)).softmax(-1) @ values
    indices = torch.tensor([0, 1, 1, 2, 3])
    split_mass = mass[indices].clone(); split_mass[1:3] /= 2
    split = (content[..., indices] + gaussian_measure_log_bias(qm, qc, km[:, indices], kc[:, indices],
        key_mass=split_mass, density_strength=strength)).softmax(-1) @ values[indices]
    torch.testing.assert_close(original, split, rtol=2e-13, atol=2e-13)


def test_controls_and_mass_use_original_covariance():
    qm, qc, km, kc = data()
    mass = 12 * torch.linalg.det(kc).sqrt()
    eye = torch.eye(2, dtype=qm.dtype)
    for method in ('point', 'fixed', 'isotropic', 'full'):
        qctrl, kctrl = qc, kc
        if method == 'point':
            qctrl, kctrl = qc * 0, kc * 0
        elif method == 'fixed':
            qctrl, kctrl = torch.ones_like(qc[..., :1, :1]) * .3 * eye, torch.ones_like(kc[..., :1, :1]) * .7 * eye
        elif method == 'isotropic':
            qctrl = qc.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye
            kctrl = kc.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye
        actual = gaussian_measure_log_bias(qm, qc, km, kc, method=method, fixed_variance=(.3, .7))
        expected = reference(qm, qctrl, km, kctrl, mass, .25)
        torch.testing.assert_close(actual, expected, rtol=5e-13, atol=5e-13)


def test_fixed_variance_batch_and_per_token():
    qm, qc, km, kc = data()
    qvar = torch.tensor([.2, .5], dtype=qm.dtype)
    kvar = torch.tensor([[.3, .5, .8, .7], [.2, .7, .9, 1.]], dtype=qm.dtype)
    result = gaussian_measure_log_bias(qm, qc, km, kc, key_mass=1., method='fixed', fixed_variance=(qvar, kvar))
    eye = torch.eye(2, dtype=qm.dtype)
    expected = reference(qm, qvar[:, None, None, None].expand(-1, 3, 2, 2) * eye,
        km, kvar[..., None, None] * eye, torch.ones(2, 4, dtype=qm.dtype), .25)
    torch.testing.assert_close(result, expected, rtol=2e-13, atol=2e-13)


def test_consistent_unit_scaling_and_common_translation_rotation():
    qm, qc, km, kc = data()
    mass = 12 * torch.linalg.det(kc).sqrt()
    original = gaussian_measure_log_bias(qm, qc, km, kc, key_mass=mass, bandwidth=.6)
    angle = torch.tensor(.73, dtype=qm.dtype)
    c, s = angle.cos(), angle.sin()
    r = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))
    scale = 3.7
    t = torch.tensor([11., -8.], dtype=qm.dtype)
    actual = gaussian_measure_log_bias(scale * (qm @ r.T) + t, scale**2 * (r @ qc @ r.T),
        scale * (km @ r.T) + t, scale**2 * (r @ kc @ r.T), key_mass=mass * scale**2, bandwidth=.6 * scale)
    torch.testing.assert_close(original, actual, rtol=2e-12, atol=2e-12)


def test_uniform_key_covariance_row_constants_cancel():
    qm, qc, km, kc = data()
    kc = torch.eye(2, dtype=qm.dtype).expand_as(kc) * .3
    mass = torch.full((2, 4), 3.6, dtype=qm.dtype)
    bias = gaussian_measure_log_bias(qm, qc, km, kc, key_mass=mass, bandwidth=.6)
    s = qc + .66 * torch.eye(2, dtype=qm.dtype)
    d = qm[..., :, None, :] - km[..., None, :, :]
    pure_mahal = -.5 * (d * torch.linalg.solve(s[..., None, :, :], d.unsqueeze(-1)).squeeze(-1)).sum(-1)
    torch.testing.assert_close(bias.softmax(-1), pure_mahal.softmax(-1), rtol=2e-13, atol=2e-13)


def test_finite_gradients_thin_spd_and_bandwidth():
    qm, qc, km, kc = data()
    # A thin valid SPD support; pair blur remains physically explicit.
    qc = torch.diag(torch.tensor([1e-9, 1e3], dtype=qm.dtype)).expand_as(qc).clone()
    tensors = [x.clone().requires_grad_() for x in (qm, qc, km, kc)]
    width = torch.tensor(.2, dtype=qm.dtype, requires_grad=True)
    strength = torch.tensor(.8, dtype=qm.dtype, requires_grad=True)
    mass = torch.ones(2, 4, dtype=qm.dtype, requires_grad=True)
    out = gaussian_measure_log_bias(*tensors, key_mass=mass, bandwidth=width, density_strength=strength)
    grads = torch.autograd.grad(out.square().mean(), tensors + [width, strength, mass])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    torch.testing.assert_close(out, reference(*tensors, mass, width, strength), rtol=1e-12, atol=1e-12)


def test_point_limit_single_pair_floor_and_no_mutation():
    qm, qc, km, kc = data()
    before = [x.clone() for x in (qm, qc, km, kc)]
    point = gaussian_measure_log_bias(qm, qc * 0, km, kc * 0, key_mass=1., bandwidth=.2)
    d2 = (qm[..., :, None, :] - km[..., None, :, :]).square().sum(-1)
    torch.testing.assert_close(point, -d2 / (.08) - math.log(2 * math.pi * .04))
    for x, old in zip((qm, qc, km, kc), before):
        assert torch.equal(x, old)


def test_bad_scalar_arguments_rejected():
    args = data()
    with pytest.raises(ValueError):
        gaussian_measure_log_bias(*args, bandwidth=-1)
    with pytest.raises(ValueError):
        gaussian_measure_log_bias(*args, bandwidth=float('nan'))
    with pytest.raises(ValueError):
        gaussian_measure_log_bias(*args, method='unknown')
