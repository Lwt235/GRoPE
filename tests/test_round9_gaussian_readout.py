import pytest
import torch

from models.round9_gaussian_readout import METHODS, bilinear_readout, gaussian_readout


torch.set_num_threads(1)


def example(dtype=torch.float64):
    generator = torch.Generator().manual_seed(923)
    image = torch.rand((5, 7, 3), dtype=dtype, generator=generator)
    means = torch.tensor([[0., 0.], [6., 4.], [2.3, 1.7], [-1.5, 2.], [4., 3.]], dtype=dtype)
    factor = torch.randn((5, 2, 2), dtype=dtype, generator=generator)
    covariance = factor @ factor.transpose(-1, -2) + torch.eye(2, dtype=dtype) * .03
    mask = torch.rand((5, 5, 7), generator=generator) > .25
    mask[1] = False
    visible = torch.tensor([True, True, True, True, False])
    return image, means, covariance, mask, visible


def independent_reference(image, means, covariance, *, method, bandwidth, fixed_variance,
                          key_mask=None, query_valid=None):
    """Deliberately dense direct displacement formula, independent of PE code.

    This implementation does not import the quadratic lift or any production
    covariance/masking/normalization helper. Tiny fixtures keep it inexpensive.
    """
    height, width, _ = image.shape
    coords = torch.tensor([[x, y] for y in range(height) for x in range(width)],
                          dtype=means.dtype, device=means.device)
    rgb, valid = [], []
    for i in range(len(means)):
        if method == "point":
            c = torch.zeros((2, 2), dtype=means.dtype, device=means.device)
        elif method == "fixed":
            c = torch.eye(2, dtype=means.dtype, device=means.device) * fixed_variance
        elif method == "isotropic":
            c = torch.eye(2, dtype=means.dtype, device=means.device) * covariance[i].trace() / 2
        else:
            c = (covariance[i] + covariance[i].T) / 2
        c = c + torch.eye(2, dtype=means.dtype, device=means.device) * (1/12 + bandwidth**2)
        delta = coords - means[i]
        energy = -.5 * (delta * torch.linalg.solve(c, delta.T).T).sum(-1)
        keep = torch.ones(len(coords), dtype=torch.bool, device=means.device)
        if key_mask is not None:
            keep = (key_mask if key_mask.ndim == 2 else key_mask[i]).reshape(-1)
        if query_valid is not None and not query_valid[i]:
            keep = torch.zeros_like(keep)
        if keep.any():
            rgb.append(energy[keep].softmax(0) @ image.reshape(-1, 3)[keep])
            valid.append(True)
        else:
            rgb.append(image.new_zeros(3))
            valid.append(False)
    return torch.stack(rgb), torch.tensor(valid, device=means.device)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("chunks", [(1, 1), (2, 8), (17, 100)])
def test_five_dimensional_exactness_for_every_control_and_chunking(method, chunks):
    image, means, covariance, mask, visible = example()
    actual = gaussian_readout(image, means, covariance, method=method, bandwidth=.37,
                              fixed_variance=1.7, key_mask=mask, query_valid=visible,
                              query_chunk_size=chunks[0], key_chunk_size=chunks[1])
    rgb, valid = independent_reference(image, means, covariance, method=method, bandwidth=.37,
                                       fixed_variance=1.7, key_mask=mask, query_valid=visible)
    torch.testing.assert_close(actual.rgb, rgb, atol=2e-13, rtol=2e-13)
    assert torch.equal(actual.valid, valid)
    assert torch.equal(actual.rgb[~valid], torch.zeros_like(actual.rgb[~valid]))


@pytest.mark.parametrize("method", METHODS)
def test_shared_mask_and_per_query_mask_agree(method):
    image, means, covariance, mask, visible = example()
    shared = mask[0]
    a = gaussian_readout(image, means, covariance, method=method, key_mask=shared, query_valid=visible)
    b = gaussian_readout(image, means, covariance, method=method,
                         key_mask=shared.expand(len(means), -1, -1), query_valid=visible)
    torch.testing.assert_close(a.rgb, b.rgb, rtol=0, atol=0)
    assert torch.equal(a.valid, b.valid)


def test_all_pixels_participate_and_rgb_gradients_reach_far_corners():
    image = torch.zeros((11, 13, 3), dtype=torch.float64, requires_grad=True)
    means = torch.tensor([[0., 0.]], dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64)[None] * 100
    result = gaussian_readout(image, means, covariance, key_chunk_size=7)
    result.rgb.sum().backward()
    assert result.valid.item()
    assert torch.all(image.grad > 0)
    assert image.grad[-1, -1, 0] > .001


def test_empty_masks_and_invisible_queries_have_finite_zero_gradients():
    image, means, covariance, mask, visible = example()
    image.requires_grad_(); means.requires_grad_(); covariance.requires_grad_()
    mask.zero_()
    actual = gaussian_readout(image, means, covariance, key_mask=mask,
                              query_valid=visible, query_chunk_size=2, key_chunk_size=4)
    assert not actual.valid.any()
    assert not actual.rgb.any()
    actual.rgb.sum().backward()
    for tensor in (image, means, covariance):
        assert torch.isfinite(tensor.grad).all()
        assert not tensor.grad.any()


def test_masked_pixels_do_not_affect_rgb_or_receive_gradients():
    image, means, covariance, _, _ = example()
    image.requires_grad_()
    mask = torch.zeros(image.shape[:2], dtype=torch.bool)
    mask[2, 3] = True
    output = gaussian_readout(image, means, covariance, key_mask=mask, key_chunk_size=4)
    torch.testing.assert_close(output.rgb, image[2, 3].expand(5, 3), rtol=0, atol=1e-15)
    output.rgb.sum().backward()
    assert not image.grad[~mask].any()
    torch.testing.assert_close(image.grad[2, 3], torch.full((3,), 5., dtype=image.dtype))


def test_five_dimensional_gradients_match_dense_mahalanobis_reference():
    image, means, covariance, mask, visible = example()
    image.requires_grad_(); means.requires_grad_(); covariance.requires_grad_()
    h = torch.tensor(.53, dtype=torch.float64, requires_grad=True)
    actual = gaussian_readout(image, means, covariance, bandwidth=h, key_mask=mask,
                              query_valid=visible, query_chunk_size=2, key_chunk_size=6).rgb
    expected, _ = independent_reference(image, means, covariance, method="full", bandwidth=h,
                                         fixed_variance=1, key_mask=mask, query_valid=visible)
    loss_weights = torch.arange(15, dtype=torch.float64).reshape(5, 3) / 17
    aa = torch.autograd.grad((actual * loss_weights).sum(), (image, means, covariance, h))
    bb = torch.autograd.grad((expected * loss_weights).sum(), (image, means, covariance, h))
    for a, b in zip(aa, bb):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-11)


def test_gradient_finite_difference_through_covariance_parameterization():
    image, means, covariance, _, _ = example()
    image = image[:3, :4].contiguous().requires_grad_()
    means = means[2:3].clone().requires_grad_()
    factor = torch.linalg.cholesky(covariance[2:3]).requires_grad_()
    h = torch.tensor(.7, dtype=torch.float64, requires_grad=True)

    def forward(values, location, shape, bandwidth):
        return gaussian_readout(values, location, shape @ shape.transpose(-1, -2),
                                bandwidth=bandwidth, key_chunk_size=5).rgb

    assert torch.autograd.gradcheck(forward, (image, means, factor, h), atol=1e-6, rtol=1e-4)


def test_isotropic_retains_trace_and_point_retains_common_pixel_covariance():
    image, means, covariance, _, _ = example()
    iso = covariance.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(2)
    a = gaussian_readout(image, means, covariance, method="isotropic")
    b = gaussian_readout(image, means, iso, method="full")
    torch.testing.assert_close(a.rgb, b.rgb, rtol=0, atol=0)
    zeros = torch.zeros_like(covariance)
    a = gaussian_readout(image, means, covariance, method="point", bandwidth=.01)
    b = gaussian_readout(image, means, zeros, method="full", bandwidth=.01)
    torch.testing.assert_close(a.rgb, b.rgb, rtol=0, atol=0)
    # A finite shared cell makes point a smooth Gaussian, not a nearest lookup.
    assert not torch.equal(a.rgb[0], image[0, 0])


def test_bilinear_corners_edges_interior_and_outside():
    yy, xx = torch.meshgrid(torch.arange(3., dtype=torch.float64),
                            torch.arange(4., dtype=torch.float64), indexing="ij")
    image = torch.stack((xx, yy, xx+2*yy), -1)
    means = torch.tensor([[0., 0.], [3., 2.], [1.2, .6], [-.01, 1.], [2., 2.01]], dtype=torch.float64)
    actual = bilinear_readout(image, means)
    expected = torch.tensor([[0., 0., 0.], [3., 2., 7.], [1.2, .6, 2.4], [0., 0., 0.], [0., 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(actual.rgb, expected, rtol=0, atol=1e-15)
    assert actual.valid.tolist() == [True, True, True, False, False]


def test_bilinear_masks_renormalize_and_zero_weight_neighbors_are_not_support():
    image, _, _, _, _ = example()
    means = torch.tensor([[1.5, 2.5], [1., 2.], [1.5, 2.5]], dtype=torch.float64)
    mask = torch.zeros((3, 5, 7), dtype=torch.bool)
    mask[:, 3, 2] = True
    result = bilinear_readout(image, means, key_mask=mask,
                              query_valid=torch.tensor([True, True, False]))
    torch.testing.assert_close(result.rgb[0], image[3, 2], rtol=0, atol=0)
    assert result.valid.tolist() == [True, False, False]
    assert not result.rgb[1:].any()


def test_bilinear_single_pixel_and_empty_query_set():
    image = torch.tensor([[[.1, .2, .3]]], dtype=torch.float64)
    means = torch.tensor([[0., 0.], [.1, 0.]], dtype=torch.float64)
    result = bilinear_readout(image, means)
    torch.testing.assert_close(result.rgb[0], image[0, 0])
    assert result.valid.tolist() == [True, False]
    empty = means[:0]
    covariance = torch.empty((0, 2, 2), dtype=torch.float64)
    for output in (bilinear_readout(image, empty), gaussian_readout(image, empty, covariance)):
        assert output.rgb.shape == (0, 3)
        assert output.valid.shape == (0,)


def test_float32_tracks_float64_reference_and_dtype_override():
    image, means, covariance, mask, visible = example(torch.float32)
    result = gaussian_readout(image, means, covariance, key_mask=mask, query_valid=visible,
                              query_chunk_size=2, key_chunk_size=6)
    accurate = gaussian_readout(image, means, covariance, key_mask=mask, query_valid=visible,
                                compute_dtype=torch.float64)
    assert result.rgb.dtype == torch.float32
    assert accurate.rgb.dtype == torch.float64
    torch.testing.assert_close(result.rgb.double(), accurate.rgb, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("kwargs", [{"bandwidth": 0}, {"bandwidth": float("nan")},
                                    {"method": "unknown"}, {"key_chunk_size": 0},
                                    {"query_chunk_size": 1.5}, {"compute_dtype": torch.int64},
                                    {"method": "fixed", "fixed_variance": -1}])
def test_invalid_configuration_rejected(kwargs):
    image, means, covariance, _, _ = example()
    with pytest.raises(ValueError):
        gaussian_readout(image, means, covariance, **kwargs)


def test_invalid_covariance_and_masks_rejected():
    image, means, covariance, _, _ = example()
    covariance[0, 0, 0] = -100
    with pytest.raises(ValueError, match="positive semidefinite"):
        gaussian_readout(image, means, covariance)
    covariance = torch.eye(2).double().expand(5, 2, 2).clone()
    covariance[0, 0, 1] = 1
    with pytest.raises(ValueError, match="symmetric"):
        gaussian_readout(image, means, covariance)
    with pytest.raises(ValueError, match="key_mask"):
        bilinear_readout(image, means, key_mask=torch.ones((5, 7)))
    with pytest.raises(ValueError, match="query_valid"):
        bilinear_readout(image, means, query_valid=torch.ones(4, dtype=torch.bool))
