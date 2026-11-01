"""Independent analytic identities and fairness checks for the scale mixture."""
import torch
from models.gaussian_pe import GaussianPEMatcher
from models.gaussian_spectral_pe import GaussianSpectralMatcher, resolvent_rotary
from models.geometry import rotary
from tests.test_normalized_pe import ordinary_fixture, model_fixture


def test_resolvent_norm_point_limit_and_translation():
    x, m, c, f = ordinary_fixture()
    out = resolvent_rotary(x, m, c, f)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(resolvent_rotary(x, m, c*0, f), rotary(x, m, f))
    shift = torch.tensor([.7, -.4], dtype=x.dtype).expand_as(m)
    shifted = resolvent_rotary(x, m+shift, c, f)
    torch.testing.assert_close(shifted @ shifted.T, out @ out.T)


def test_direct_formula_and_gradients():
    x, m, c, f = ordinary_fixture()
    x, m, c = [v[:2].detach().requires_grad_() for v in (x, m, c)]
    a = 1/(1+torch.einsum('fd,...de,fe->...f', f, c, f)/2)
    direct = (rotary(x, m, f).reshape(2, 24, 2)*a[..., None]).flatten(-2)
    direct = direct / direct.norm(dim=-1, keepdim=True)*x.norm(dim=-1, keepdim=True)
    actual = resolvent_rotary(x, m, c, f)
    torch.testing.assert_close(actual, direct)
    weight = torch.arange(48, dtype=x.dtype)[None]/48
    da = torch.autograd.grad((actual*weight).sum(), (x, m, c), retain_graph=True)
    db = torch.autograd.grad((direct*weight).sum(), (x, m, c))
    for a, b in zip(da, db):
        torch.testing.assert_close(a, b)


def test_scale_mixture_integral():
    # Deterministic midpoint integration of E_exp(1) exp(-T v/2).
    t = (torch.arange(40000, dtype=torch.float64)+.5)*.0005
    v = torch.tensor([0., .1, 1., 5., 20.], dtype=t.dtype)
    integrated = (torch.exp(-t[:, None]*(1+v[None]/2))).sum(0)*.0005
    torch.testing.assert_close(integrated, 1/(1+v/2), atol=2e-7, rtol=2e-6)


def test_existing_gaussian_full_and_key_fixed_match_exactly():
    data = model_fixture()
    for old, new in [('normalized_shape', 'gaussian_full'), ('normalized_fixed', 'gaussian_qonly')]:
        a = GaussianPEMatcher(old, 16, .25)
        b = GaussianSpectralMatcher(new, 16, .25)
        b.load_state_dict(a.state_dict(), strict=True)
        torch.testing.assert_close(a(data), b(data), atol=0, rtol=0)
        assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_symmetric_controls_remove_both_covariances():
    data = model_fixture()
    a = GaussianSpectralMatcher('gaussian_none', 16)
    altered = dict(data, projected_qcov=data['projected_qcov']*17,
                   projected_kcov=data['projected_kcov']*31)
    torch.testing.assert_close(a(data), a(altered), atol=0, rtol=0)
    for control in ('trace', 'area'):
        a = GaussianSpectralMatcher('resolvent_'+control, 16)
        _, cq, _, ck = a.positional_inputs(data)
        for c in (cq, ck):
            assert torch.equal(c[..., 0, 0], c[..., 1, 1])
            assert (c[..., 0, 1] == 0).all()


def test_extreme_sparse_support_stays_finite_with_finite_gradient():
    x = torch.tensor([[0., 0., 1., 0.]], requires_grad=True)
    m = torch.zeros(1, 2)
    c = torch.eye(2)[None]*1e20
    f = torch.tensor([[1., 0.], [2., 0.]])
    result = resolvent_rotary(x, m, c, f)
    torch.testing.assert_close(result, x)
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
