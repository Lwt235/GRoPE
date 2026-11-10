import math
import numpy as np
import pytest
import torch

from experiments.region_hrope_data import camera_fields
from models.region_hrope_control import CameraRegionMatcher, ROLES, homographies, transport, hrope_layer
from models.round58_support_pe import AttentionBlock
from experiments.region_retokenization import make_model

torch.set_num_threads(4)


def sample(dtype=torch.float64):
    torch.manual_seed(915)
    features = torch.randn(1, 2, 7, 11, dtype=dtype)
    centers = torch.randn(1, 2, 7, 3, dtype=dtype)
    a = torch.randn(1, 2, 7, 3, 3, dtype=dtype)
    covariance = .1*(a@a.transpose(-1, -2))+.01*torch.eye(3, dtype=dtype)
    permutation = torch.arange(7).expand(1, 2, 7)
    intrinsics = torch.tensor([[[.9, .02, -.04], [0., 1.2, .03], [0., 0., 1.]],
                              [[1.1, -.01, .06], [0., .8, -.05], [0., 0., 1.]]], dtype=dtype)[None]
    xy = torch.rand(1, 2, 7, 2, dtype=dtype)-.5
    return features, centers, covariance, permutation, intrinsics, xy


def full_matrices(matrix, xy):
    h = matrix.detach().numpy()
    position = xy.detach().numpy()
    result = np.zeros((*h.shape[:-2], 48, 48))
    for idx in np.ndindex(h.shape[:-2]):
        for block in range(8):
            result[idx][3*block:3*block+3, 3*block:3*block+3] = h[idx]
        phases = (8*position[idx][:, None]*10000.**(-np.arange(1, 7)/12)).ravel()
        for block, phi in enumerate(phases):
            result[idx][24+2*block:26+2*block, 24+2*block:26+2*block] = (
                (np.cos(phi), -np.sin(phi)), (np.sin(phi), np.cos(phi)))
    return result


def test_paper_matrices_inverse_and_complete_cross_attention():
    torch.manual_seed(14)
    _, _, _, _, intrinsic, xy = sample()
    raw = .12*torch.randn(1, 2, 7, 13, dtype=torch.float64)
    raw[..., 0] += 1.
    raw[..., 4] += 1.
    raw[..., 11] += 1.
    h, hi = homographies(raw, intrinsic.unsqueeze(-3))
    expected = np.empty((1, 2, 7, 3, 3))
    for idx in np.ndindex((1, 2, 7)):
        x = raw.numpy()[idx]
        e1 = x[:3]/np.linalg.norm(x[:3])
        e2 = x[3:6]-e1*np.dot(e1, x[3:6])
        e2 /= np.linalg.norm(e2)
        r = np.column_stack((e1, e2, np.cross(e1, e2)))
        n = x[9:12]/np.linalg.norm(x[9:12])
        expected[idx] = intrinsic.numpy()[idx[:2]]@(r+np.clip(x[12], -.5, .5)*np.outer(np.clip(x[6:9], -.5, .5), n))
    np.testing.assert_allclose(h.numpy(), expected, atol=1e-12)
    np.testing.assert_allclose(hi.numpy(), np.linalg.inv(expected), atol=1e-12)
    full = full_matrices(h, xy)
    values = torch.randn(1, 2, 2, 7, 48, dtype=torch.float64)
    for op, inverse_rope, reference in ((h, False, full), (h.transpose(-1, -2), True, full.swapaxes(-1, -2)),
                                       (hi, True, np.linalg.inv(full))):
        actual = transport(values, op, xy, inverse_rope).numpy()
        independent = np.einsum('bvnij,bvhnj->bvhni', reference, values.numpy())
        np.testing.assert_allclose(actual, independent, atol=3e-12)
    layer = AttentionBlock().double()
    x = torch.randn(1, 2, 7, 96, dtype=torch.float64)
    q, k, v = layer.qkv(layer.norm(x)).reshape(1, 2, 7, 3, 2, 48).permute(3, 0, 1, 4, 2, 5).unbind(0)
    q = np.einsum('bvnij,bvhnj->bvhni', full.swapaxes(-1, -2), q.detach().numpy())
    k = np.einsum('bvnij,bvhnj->bvhni', np.linalg.inv(full), k.detach().numpy())[:, ::-1]
    v = np.einsum('bvnij,bvhnj->bvhni', np.linalg.inv(full), v.detach().numpy())[:, ::-1]
    logits = q@k.swapaxes(-1, -2)/math.sqrt(48)
    exp = np.exp(logits-logits.max(-1, keepdims=True))
    aggregate = (exp/exp.sum(-1, keepdims=True))@v
    aggregate = np.einsum('bvnij,bvhnj->bvhni', full, aggregate)
    aggregate = torch.from_numpy(aggregate).transpose(-3, -2).reshape(1, 2, 7, 96)
    reference = x+layer.output(aggregate)
    reference = reference+layer.ffn(layer.ffn_norm(reference))
    torch.testing.assert_close(hrope_layer(layer, x, h, hi, xy, True), reference, atol=1e-11, rtol=1e-11)


def test_common_initialization_and_original_controls():
    inputs = sample()
    common = None
    for role in ROLES:
        torch.manual_seed(11)
        model = CameraRegionMatcher(role).double()
        state = {k: v for k, v in model.state_dict().items() if not k.startswith('position.')}
        if common is None:
            common = state
        else:
            for key in state:
                assert torch.equal(state[key], common[key]), (role, key)
        assert torch.count_nonzero(model.stem[0].weight[:, 20:]) == 0
        if role != 'predicted_hrope':
            torch.manual_seed(11)
            original = make_model(role).double()
            for key, value in original.state_dict().items():
                current = model.state_dict()[key]
                if key == 'stem.0.weight':
                    current = current[:, :20]
                assert torch.equal(value, current), (role, key)
            torch.testing.assert_close(model(*inputs), original(*inputs[:4]), atol=1e-11, rtol=1e-11)


def test_predicted_hrope_permutation_and_learning_gradients():
    inputs = list(sample())
    torch.manual_seed(22)
    model = CameraRegionMatcher('predicted_hrope').double()
    with torch.no_grad():
        model.position.weight.normal_(0, .01)
    original = model(*inputs)
    order = torch.tensor([4, 1, 6, 0, 2, 5, 3])
    changed = [x[:, :, order] if k in (0, 1, 2, 3, 5) else x for k, x in enumerate(inputs)]
    torch.testing.assert_close(model(*changed), original[:, order][:, :, order], atol=1e-10, rtol=1e-10)
    with pytest.raises(TypeError):
        model(*inputs, true_pose=torch.eye(4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss = original.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.position.weight.grad.norm() > 0
    assert model.stem[0].weight.grad[:, 20:].norm() > 0
    optimizer.step()
    assert not torch.equal(model(*inputs), original)


def test_camera_positions_use_original_member_pixels():
    ids = np.array([[[0, 5, 17, 23], [1, 6, 16, 22]]])
    neighbors = np.array([[[[0, 2], [1, 3]], [[0, 1], [2, 3]]]])
    sizes = np.array([[[4, 6], [4, 6]]])
    k = np.broadcast_to(np.array([[8., .1, 2.3], [0., 9., 1.7], [0., 0., 1.]]), (1, 2, 3, 3))
    norm_k, xy = camera_fields(ids, neighbors, k, sizes)
    reference = np.empty_like(xy)
    for v in range(2):
        for n in range(2):
            pixels = [divmod(int(ids[0, v, i]), 6) for i in neighbors[0, v, n]]
            reference[0, v, n] = [np.mean([a[1] for a in pixels])/6-.5,
                                   np.mean([a[0] for a in pixels])/4-.5]
    np.testing.assert_allclose(xy, reference)
    transformation = np.array([[1/6, 0, -.5], [0, 1/4, -.5], [0, 0, 1.]])
    np.testing.assert_allclose(norm_k, transformation@k, atol=1e-7)
