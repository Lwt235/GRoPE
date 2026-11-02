import unittest
import numpy as np
import torch
from scipy.linalg import expm
from experiments.round51_data import camera_fields
from models.round51_visual_alignment import VisualAlignmentMatcher
from models.round52_camera_frame import CameraFrameMatcher, ROLES, BASE, camera_lift


def numpy_operator(model, features, intrinsics, geometry):
    """Separate float64 construction: no production frame or inverse helpers."""
    count = len(features)
    if model.experiment_role == 'camera_only':
        frames = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    else:
        raw = features @ model.field.weight.detach().numpy().T + model.field.bias.detach().numpy()
        frames = []
        for x in raw:
            if model.role == 'affine':
                linear = expm(.5*np.tanh(x[:9]).reshape(3, 3))
                mu = np.tanh(x[9:12])
            elif model.role == 'gs_iso':
                linear = np.eye(3)*np.exp(.5*np.tanh(x[0]))
                mu = np.tanh(x[1:4])
            else:
                u = x[:3]/np.linalg.norm(x[:3])
                v = x[3:6]-u*np.dot(u, x[3:6]); v = v/np.linalg.norm(v)
                r = np.column_stack((u, v, np.cross(u, v)))
                linear = r @ np.diag(np.exp(.5*np.tanh(x[6:9]))) @ r.T
                mu = np.tanh(x[9:12])
            g = np.eye(4); g[:3, :3] = linear; g[:3, 3] = mu
            frames.append(g)
        frames = np.array(frames)
    if model.experiment_role == 'local_gs':
        return frames
    t = np.eye(4); t[:3, :3] = geometry[0, :9].reshape(3, 3); t[:3, 3] = geometry[0, 9:12]
    k = np.eye(4); k[:3, :3] = intrinsics
    return (k @ np.linalg.inv(t))[None] @ frames


def full_operator(a, xy):
    full = np.zeros((len(a), 48, 48))
    for j in range(0, 24, 4):
        full[:, j:j+4, j:j+4] = a
    for i, point in enumerate(xy):
        phases = (point[:, None]*np.logspace(-1, .6, 6)).ravel()
        for j, angle in enumerate(phases):
            c, s = np.cos(angle), np.sin(angle)
            full[i, 24+2*j:26+2*j, 24+2*j:26+2*j] = [[c, s], [-s, c]]
    return full


class CameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def sample(self):
        torch.manual_seed(815)
        t = np.eye(4); theta = .4
        t[:3, :3] = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        t[:3, 3] = [.3, -.2, .1]
        K = np.array([[170., 0, 94.7], [0, 172., 73.1], [0, 0, 1.]])
        d = {key:torch.randn(1, 7, 128, dtype=torch.float64) for key in ('q', 'k')}
        for prefix, pose in (('q', np.eye(4)), ('k', t)):
            for key, value in camera_fields(dict(K=K), pose).items():
                d[prefix+key] = torch.from_numpy(value if key=='K' else value[:7]).double()[None]
        return d

    def test_physical_chart_and_world_gauge(self):
        d = self.sample(); geom = d['kgeom']; intr = d['kK']
        p, pi = camera_lift(geom, intr)
        g = geom.numpy()[0]; K = intr.numpy()[0]
        R, origin = g[0, :9].reshape(3, 3), g[0, 9:12]*3
        camera_xyz = np.array([[.6, .4, 2.], [-.3, .5, 3.]])
        world_xyz = camera_xyz @ R.T+origin
        h = np.column_stack((world_xyz/3, np.ones(2)))
        image = h @ p.numpy()[0].T
        expected = camera_xyz @ K.T
        np.testing.assert_allclose(image[:, :2]/image[:, 2:3], expected[:, :2]/expected[:, 2:3], atol=1e-12)
        np.testing.assert_allclose(image @ pi.numpy()[0].T, h, atol=1e-12)
        # World-coordinate change and the canonical-root right gauge cancel.
        w = np.eye(4); w[:3, :3] = expm(np.array([[0., -.3, .2], [.3, 0, -.1], [-.2, .1, 0]])); w[:3, 3] = [.2, -.4, .1]
        b = np.eye(4); b[:3, :3] = w[:3, :3].T
        rng = np.random.default_rng(771); frames = []
        for _ in range(2):
            x = rng.normal(0, .2, (3, 3)); G = np.eye(4)
            G[:3, :3] = expm(x+x.T); G[:3, 3] = rng.normal(0, .2, 3); frames.append(G)
        p0 = camera_lift(d['qgeom'], d['qK'])[0].numpy()[0]; p1 = p.numpy()[0]
        relative = p0 @ frames[0] @ np.linalg.inv(p1 @ frames[1])
        changed = (p0 @ np.linalg.inv(w)) @ (w @ frames[0] @ b) @ np.linalg.inv((p1 @ np.linalg.inv(w)) @ (w @ frames[1] @ b))
        np.testing.assert_allclose(relative, changed, atol=1e-12)

    def test_nonzero_operator_transport_and_gradient(self):
        d = self.sample(); feat = torch.randn(1, 7, 96, dtype=torch.float64)
        for role in BASE:
            torch.manual_seed(72); model = CameraFrameMatcher(role).double()
            if hasattr(model, 'field'):
                with torch.no_grad():
                    model.field.weight.normal_(0, .015)
                    model.field.bias.add_(.1*torch.randn_like(model.field.bias))
            camera = camera_lift(d['kgeom'], d['kK'])
            a, ai = model.matrices(feat, d['kK'], camera)
            ref = numpy_operator(model, feat.numpy()[0], d['kK'].numpy()[0], d['kgeom'].numpy()[0])
            np.testing.assert_allclose(a.detach().numpy()[0], ref, atol=1e-11)
            np.testing.assert_allclose(ai.detach().numpy()[0], np.linalg.inv(ref), atol=1e-11)
            full = full_operator(ref, d['kpos'].numpy()[0]); value = torch.randn(1, 2, 7, 48, dtype=torch.float64)
            for operation, matrix, reference in (('query', a.transpose(-1,-2), full.transpose(0,2,1)),
                    ('key', ai, np.linalg.inv(full)), ('value', ai, np.linalg.inv(full)), ('output', a, full)):
                actual = model.transport(value, matrix, d['kpos'], operation).detach().numpy()[0]
                np.testing.assert_allclose(actual, np.einsum('nij,hnj->hni', reference, value.numpy()[0]), atol=2e-11)
            if role == 'camera_only':
                continue
            probe = torch.randn(1,7,7,dtype=torch.float64)
            loss = (model(d)*probe).sum(); loss.backward(); param = model.field.weight
            idx = np.unravel_index(int(param.grad.abs().argmax()), param.shape)
            analytic = float(param.grad[idx]); original = float(param.detach()[idx]); eps = 1e-5
            with torch.no_grad():
                param[idx] = original+eps; plus = float((model(d)*probe).sum())
                param[idx] = original-eps; minus = float((model(d)*probe).sum()); param[idx] = original
            numeric = (plus-minus)/(2*eps)
            self.assertGreater(abs(analytic), 1e-8)
            self.assertLess(abs(analytic-numeric)/max(1, abs(analytic), abs(numeric)), 2e-5, role)
        # The Gaussian operator cannot depend on a choice of axis signs.
        model = CameraFrameMatcher('camera_gs').double()
        with torch.no_grad(): model.field.bias[6:9].copy_(torch.tensor([.2,-.3,.1]))
        a = model.matrices(feat, d['kK'], camera)[0]
        with torch.no_grad(): model.field.bias[:6].mul_(-1)
        b = model.matrices(feat, d['kK'], camera)[0]
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_shared_inputs_and_reference_controls(self):
        d = self.sample(); common = None; perm = torch.tensor([3,1,0,6,5,2,4])
        for role in ROLES:
            torch.manual_seed(11); model = CameraFrameMatcher(role).double().eval()
            state = {k:v for k,v in model.state_dict().items() if not k.startswith('field.')}
            if common is None: common = state
            else:
                self.assertEqual(common.keys(), state.keys())
                for key in state: self.assertTrue(torch.equal(state[key], common[key]), (role, key))
            pred = model(d)
            forbidden = dict(d, gt_valid=torch.randn(1,7), pixcontinuous=torch.randn(1,7,2),
                             predicted_depth=torch.randn(1,7), mean3=torch.randn(1,7,3))
            torch.testing.assert_close(model(forbidden), pred, rtol=0, atol=0)
            changed = {k:(v[:,perm] if k in ('k','kgeom','kpos') else v) for k,v in d.items()}
            torch.testing.assert_close(model(changed), pred[:,:,perm], rtol=1e-10, atol=1e-10)
            pred.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            if role in ('nope', 'mlp', 'hrope'):
                torch.manual_seed(11); old = VisualAlignmentMatcher(role).double().eval()
                torch.testing.assert_close(old(d), pred, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
