"""Algebra/domain audit independent of any quadratic-PE implementation."""
import torch

from models.gaussian_measure_pe import gaussian_measure_log_bias


def examples():
    g = torch.Generator().manual_seed(617)
    qm = torch.randn(2, 7, 2, generator=g, dtype=torch.float64)
    km = torch.randn(2, 8, 2, generator=g, dtype=torch.float64)
    qa = torch.randn(2, 7, 2, 2, generator=g, dtype=torch.float64)
    ka = torch.randn(2, 1, 2, 2, generator=g, dtype=torch.float64)
    qc = qa @ qa.transpose(-1, -2) + .1 * torch.eye(2)
    kc = ka @ ka.transpose(-1, -2) + .2 * torch.eye(2)
    return qm, qc, km, kc


def lifted_score(qm, qc, km, common_kc, width, origin=None):
    if origin is not None:
        qm, km = qm - origin, km - origin
    covariance = qc + common_kc + width**2 * torch.eye(2, dtype=qm.dtype)
    # Both formulas are defined on symmetric covariance matrices. Use the
    # same symmetric extension when checking unconstrained matrix gradients.
    covariance = (covariance + covariance.transpose(-1, -2)) / 2
    a = torch.linalg.inv(covariance)
    am = (a @ qm.unsqueeze(-1)).squeeze(-1)
    qg = torch.stack((am[..., 0], am[..., 1], -.5 * a[..., 0, 0],
                      -a[..., 0, 1], -.5 * a[..., 1, 1]), -1)
    x, y = km.unbind(-1)
    kg = torch.stack((x, y, x.square(), x*y, y.square()), -1)
    return qg @ kg.transpose(-1, -2)


def test_five_channel_lift_exact_probabilities_and_value_aggregation():
    qm, qc, km, kc = examples()
    bias = gaussian_measure_log_bias(qm, qc, km, kc.expand(2, 8, 2, 2),
                                     key_mass=1., bandwidth=.37)
    content = torch.arange(56, dtype=qm.dtype).reshape(1, 7, 8) / 100
    values = torch.arange(24, dtype=qm.dtype).reshape(8, 3) / 7
    expected = (bias + content).softmax(-1)
    origin = torch.tensor([9., -7.], dtype=qm.dtype)
    actual = (lifted_score(qm, qc, km, kc, .37, origin) + content).softmax(-1)
    torch.testing.assert_close(actual, expected, rtol=4e-13, atol=4e-13)
    torch.testing.assert_close(actual @ values, expected @ values, rtol=4e-13, atol=4e-13)


def test_five_channel_lift_same_geometry_bandwidth_gradients():
    leaves = [x.requires_grad_() for x in examples()]
    qm, qc, km, kc = leaves
    width = torch.tensor(.37, dtype=qm.dtype, requires_grad=True)
    bias = gaussian_measure_log_bias(qm, qc, km, kc.expand(2, 8, 2, 2),
                                     key_mass=1., bandwidth=width)
    lifted = lifted_score(qm, qc, km, kc, width)
    weight = torch.arange(112, dtype=qm.dtype).reshape(2, 7, 8) / 112
    direct_grad = torch.autograd.grad((bias.log_softmax(-1) * weight).sum(), leaves+[width], retain_graph=True)
    lifted_grad = torch.autograd.grad((lifted.log_softmax(-1) * weight).sum(), leaves+[width])
    for actual, expected in zip(lifted_grad, direct_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def test_generic_unequal_key_covariances_cannot_use_exact_five_channel_lift():
    qm, qc, km, kc = examples()
    kscale = torch.linspace(.05, 5., 8, dtype=qm.dtype)
    variable_kc = kc.expand(2, 8, 2, 2) * kscale[..., None, None]
    actual = gaussian_measure_log_bias(qm, qc, km, variable_kc, key_mass=1., bandwidth=.37)
    false_lift = lifted_score(qm, qc, km, variable_kc[:, :1], .37)
    assert (actual.softmax(-1) - false_lift.softmax(-1)).abs().max() > .05
    # Subtracting row constants preserves any putative rank<=5 Qg Kg^T
    # factorization. This ordinary Gaussian-overlap example has larger rank.
    centered = actual - actual.mean(-1, keepdim=True)
    assert torch.linalg.matrix_rank(centered, atol=1e-9).min() > 5


def test_bures_gaussian_rbf_not_automatically_psd_even_for_spd_covariance():
    angle = torch.arange(4, dtype=torch.float64) * torch.pi / 4
    u = torch.stack((angle.cos(), angle.sin()), -1)
    covariance = u[..., :, None] * u[..., None, :] + 1e-5 * torch.eye(2)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    root = (eigenvectors * eigenvalues.sqrt()[:, None, :]) @ eigenvectors.transpose(-1, -2)
    middle = root[:, None] @ covariance[None] @ root[:, None]
    trace_root = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum(-1)
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    distance2 = trace[:, None] + trace[None] - 2 * trace_root
    gram = (-.1 * distance2).exp()
    assert torch.linalg.eigvalsh(covariance).min() > 0
    assert torch.linalg.eigvalsh(gram).min() < -.06
