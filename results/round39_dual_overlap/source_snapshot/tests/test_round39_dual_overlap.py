"""Prospective synthetic checks. No real archives, labels or model fitting."""
import unittest
import numpy as np
from numpy.polynomial.hermite import hermgauss
from models import round39_dual_overlap_pe as P
from experiments import round39_overlap_audit_numeric as A


class TestOverlap(unittest.TestCase):
    def geometry(self):
        rng = np.random.default_rng(20261101)
        q, k = rng.normal(size=(4, 3))*.3, rng.normal(size=(6, 3))*.3
        F, G = rng.normal(size=(4, 3, 2))*.09, rng.normal(size=(6, 3, 2))*.09
        return q, F @ F.swapaxes(-1, -2), k, G @ G.swapaxes(-1, -2)

    def test_dirac(self):
        q = np.array([[0., 0., 0.]])
        k = np.array([[0., 0., 0.], [1., 2., 3.]])
        s = P.score(q, np.zeros((1, 3, 3)), k, np.zeros((2, 3, 3)), 2., np.zeros((1, 2)))
        np.testing.assert_allclose(s['lograw'], [[np.log(9.), np.log(9.)-3.5]], atol=2e-14)

    def test_psd_independent_three_roles(self):
        args = self.geometry()
        content = np.arange(24).reshape(4, 6)*.01
        for role in P.ROLES:
            p = P.score(*args, .7, content, role)
            a = A.audit_scores(*args, .7, content, role)
            for field in p:
                np.testing.assert_allclose(p[field], a[field], rtol=1e-11, atol=1e-12)

    def test_expectation_quadrature(self):
        q, qc, k, kc = self.geometry()
        raw = P.score(q, qc, k, kc, .7, np.zeros((4, 6)))['lograw']
        expectations = []
        for order in (20, 32):
            x, w = hermgauss(order)
            points = np.stack(np.meshgrid(x, x, x, indexing='ij'), -1).reshape(-1, 3)*np.sqrt(2.)
            weight = (w[:, None, None]*w[None, :, None]*w[None, None, :]).ravel()/np.pi**1.5
            vals = np.empty((4, 6))
            for i in range(4):
                for j in range(6):
                    eig, vec = np.linalg.eigh(qc[i]+kc[j])
                    self.assertGreaterEqual(eig.min(), -1e-15)
                    # Positive definiteness holds for these fixed synthetic pairs.
                    self.assertGreater(eig.min(), 0.)
                    samples = points @ (vec*np.sqrt(eig)).T+q[i]-k[j]
                    vals[i, j] = np.sum(weight*np.exp(-np.square(samples).sum(-1)/(2*.7)))
            expectations.append(vals)
        np.testing.assert_allclose(expectations[0], expectations[1], rtol=2e-11, atol=1e-12)
        np.testing.assert_allclose(np.exp(raw)/9., expectations[1], rtol=2e-11, atol=1e-12)

    def test_prepared_rigid_and_similarity(self):
        q, qc, k, kc = self.geometry()
        angle = .7
        R = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        t = np.array([.7, -.4, 2.])
        original = P.score(q, qc, k, kc, .7, np.zeros((4, 6)))
        for scale in (.001, 1., 1000.):
            transformed = P.score((q @ R.T+t)*scale, (R @ qc @ R.T)*scale**2,
                                  (k @ R.T+t)*scale, (R @ kc @ R.T)*scale**2,
                                  .7*scale**2, np.zeros((4, 6)))
            for field in original:
                np.testing.assert_allclose(original[field], transformed[field], rtol=1e-11, atol=1e-11)

    def test_support_preparation(self):
        yy, xx = np.mgrid[:144, :192]
        depth = 1.+xx*.001+yy*.002
        depth[0, :3] = [np.nan, np.inf, -np.inf]
        K = np.array([[150., .2, 95.5], [0., 148., 71.5], [0., 0., 1.]])
        T = np.eye(4); T[:3, 3] = [.1, -.2, .3]
        obs = dict(q_predicted_depth=depth, k_predicted_depth=depth[::-1].copy(), q_K=K, k_K=K.copy(), q_to_k_transform=T)
        primary, other = P.prepare(obs), A.audit_prepare(obs)
        self.assertEqual(set(primary), set(other))
        for field in primary:
            np.testing.assert_allclose(primary[field], other[field], rtol=1e-10, atol=1e-12)
        self.assertGreater(float(primary['gamma']), 0.)

    def test_mass_and_extreme_floor(self):
        q = np.array([[1e6, 1e6, 1e6]])
        k = np.zeros((2, 3)); k[1, 0] = 1.
        p = P.score(q, np.zeros((1, 3, 3)), k, np.zeros((2, 3, 3)), .01, np.array([[0., 1.]]))
        np.testing.assert_allclose(p['logfloored'], np.log(9.e-30), atol=1e-12)
        np.testing.assert_allclose(p['p'], [[1/(1+np.e), np.e/(1+np.e)]], atol=1e-14)
        self.assertEqual(p['winner'].tolist(), [1])
        other = A.audit_scores(q, np.zeros((1, 3, 3)), k, np.zeros((2, 3, 3)), .01, np.array([[0., 1.]]))
        for name in p:
            np.testing.assert_allclose(p[name], other[name], rtol=1e-11, atol=1e-12)
        for score in (P.score, A.audit_scores):
            tied = score(q, np.zeros((1, 3, 3)), k, np.zeros((2, 3, 3)), .01, np.zeros((1, 2)))
            self.assertEqual(tied['winner'].tolist(), [0])

    def test_reject_indefinite(self):
        q, qc, k, kc = self.geometry(); qc[0] = -np.eye(3)
        with self.assertRaises(ValueError): P.score(q, qc, k, kc, .7, np.zeros((4, 6)))

    def test_native_population(self):
        self.assertEqual(P.bounds(24).shape, (48, 4))
        self.assertEqual(P.bounds(8).shape, (432, 4))
        np.testing.assert_array_equal(P.bounds(24)[-1], [168, 120, 192, 144])

    def test_constant_plane_support(self):
        K = np.array([[150., 0., 95.5], [0., 150., 71.5], [0., 0., 1.]])
        B = np.linalg.inv(K)[:, :2]
        for size in (8, 24):
            _, cov, _ = P.support(np.full((144, 192), 2.), K, size)
            expected = 4.*B @ (size**2/12.*np.eye(2)) @ B.T
            np.testing.assert_allclose(cov, np.broadcast_to(expected, cov.shape), atol=1e-13)

    def test_depth_exchange_is_new_information(self):
        q = np.array([[0., 0., 2.]])
        k = np.array([[-.1, 0., 2.], [.1, 0., 3.]])
        covq = np.eye(3)[None]*.01; covk = np.tile(covq, (2, 1, 1))
        before = P.score(q, covq, k, covk, .01, np.zeros((1, 2)))
        k[:, 2] = k[::-1, 2].copy()
        after = P.score(q, covq, k, covk, .01, np.zeros((1, 2)))
        self.assertEqual(before['winner'].tolist(), [0])
        self.assertEqual(after['winner'].tolist(), [1])


if __name__ == '__main__': unittest.main()
