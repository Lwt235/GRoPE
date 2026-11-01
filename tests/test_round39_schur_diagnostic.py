"""Synthetic Schur checks; no real input or target decoding."""
import math
import unittest
import numpy as np
from experiments.round39_schur_components import schur_components as primary
from experiments.round39_schur_components_audit import audit_components as audit
from experiments import round39_schur_diagnostic as D


class TestSchur(unittest.TestCase):
    def setup_geometry(self):
        rng = np.random.default_rng(20261101)
        q = rng.normal(size=(5, 3))
        k = np.array([[0., 0., 2.], [1e-14, -1e-14, 2.], [0., 0., -2.],
                      [.7, -.2, .4], [1., 1., 1.], [1., 0., 0.]], np.float64)
        F, G = rng.normal(size=(5, 3, 2))*.3, rng.normal(size=(6, 3, 3))*.2
        return q, F @ F.swapaxes(-1, -2), k, G @ G.swapaxes(-1, -2), np.asarray(.07)

    def test_point_analytic(self):
        q = np.array([[1., 2., 3.]], np.float64)
        k = np.array([[0., 0., 1.]], np.float64)
        zeros = np.zeros((1, 3, 3))
        for fn in (primary, audit):
            v = fn(q, zeros, k, zeros, np.asarray(2.), 'point')
            for name, expected in dict(td=-1.25, tv=0., rd=-1., rv=0., s=2., residual=2., radial_delta=2.).items():
                np.testing.assert_allclose(v[name], expected, atol=2e-14)

    def test_two_bases_all_roles(self):
        args = self.setup_geometry()
        for role in D.ROLES:
            a, b = primary(*args, role), audit(*args, role)
            self.assertEqual(set(a), set(D.FIELDS))
            for name in D.FIELDS:
                np.testing.assert_allclose(a[name], b[name], rtol=1e-10, atol=1e-11)

    def test_direct_3d_reconstruction(self):
        q, qc, k, kc, gamma = self.setup_geometry()
        for role in D.ROLES:
            cq, ck = qc.copy(), kc.copy()
            if role == 'point': cq[:]=0.; ck[:]=0.
            if role == 'isotropic':
                cq = np.trace(cq, axis1=1, axis2=2)[:, None, None]*np.eye(3)/3.
                ck = np.trace(ck, axis1=1, axis2=2)[:, None, None]*np.eye(3)/3.
            S = gamma*np.eye(3)+cq[:, None]+ck[None]
            delta = q[:, None]-k[None]
            quad = (delta[..., None, :] @ np.linalg.solve(S, delta[..., None]))[..., 0, 0]
            sign, logdet = np.linalg.slogdet(S)
            self.assertTrue(np.all(sign == 1))
            direct = math.log(9.)+.5*(3*np.log(gamma)-logdet-quad)
            for fn in (primary, audit):
                v = fn(q, qc, k, kc, gamma, role)
                np.testing.assert_allclose(math.log(9.)+v['td']+v['tv']+v['rd']+v['rv'], direct, rtol=1e-10, atol=1e-11)

    def test_rotation_and_scale(self):
        q, qc, k, kc, gamma = self.setup_geometry()
        theta = .71
        R = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
        # Camera center is the ray origin: rotate and scale about that origin.
        # Translation would require translating the camera center too.
        for fn in (primary, audit):
            base = fn(q, qc, k, kc, gamma)
            for scale in (.001, 1., 1000.):
                v = fn(scale*q @ R.T, scale**2*(R @ qc @ R.T), scale*k @ R.T,
                       scale**2*(R @ kc @ R.T), np.asarray(scale**2*gamma))
                for name in D.FIELDS:
                    factor = scale**2 if name == 's' else scale if name in ('residual', 'radial_delta') else 1.
                    np.testing.assert_allclose(v[name], factor*base[name], rtol=1e-9, atol=1e-8)

    def test_covariance_cross_term(self):
        q = np.array([[1., 0., 2.]])
        k = np.array([[0., 0., 1.]])
        cov = np.array([[[1., 0., .8], [0., 1., 0.], [.8, 0., 1.]]])
        for fn in (primary, audit):
            v = fn(q, cov, k, np.zeros((1, 3, 3)), np.asarray(.2))
            np.testing.assert_allclose(v['residual'], 1-.8/1.2, atol=1e-13)
            np.testing.assert_allclose(v['s'], 1.2-.8*.8/1.2, atol=1e-13)

    def test_invalid_ray_and_covariance(self):
        q, qc, k, kc, g = self.setup_geometry()
        for fn in (primary, audit):
            bad = k.copy(); bad[0] = 0
            with self.assertRaises(ValueError): fn(q, qc, bad, kc, g)
            badcov = qc.copy(); badcov[0] = -np.eye(3)
            with self.assertRaises(ValueError): fn(q, badcov, k, kc, g)

    def test_anchor_boundaries(self):
        K = np.eye(3)
        q = np.array([[-.5, -.5, 1.], [191.5, 0., 1.], [7.5, 7.5, 1.], [0., 0., -1.], [0., 0., 0.]])
        rows = D.anchors(q, K)
        self.assertEqual([r['anchor_eligible'] for r in rows], [True, False, True, False, False])
        self.assertEqual([r['anchor_index'] for r in rows], [0, None, 25, None, None])
        self.assertTrue(rows[0]['on_cell_boundary'])

    def test_radial_reversal_gap(self):
        terms = {k: np.zeros(2) for k in D.FIELDS}
        terms['td'] = np.array([-3., 0.]); terms['rd'] = np.array([0., -5.])
        raw = math.log(9.)+terms['td']+terms['rd']
        v = D.gap(terms, raw, raw.copy(), np.zeros(2), 0, 1)
        self.assertTrue(v['radial_distance_reversal']); self.assertTrue(v['radial_total_reversal'])
        self.assertFalse(v['volume_reversal'])
        self.assertAlmostEqual(v['rawgap'], 2.)

    def test_volume_reversal_gap(self):
        terms = {k: np.zeros(2) for k in D.FIELDS}
        terms['td'] = np.array([-1., 0.]); terms['tv'] = np.array([0., -3.])
        raw = math.log(9.)+terms['td']+terms['tv']
        v = D.gap(terms, raw, raw.copy(), np.zeros(2), 0, 1)
        self.assertTrue(v['volume_reversal']); self.assertFalse(v['radial_total_reversal'])

    def test_floor_content_ties_and_all_rows(self):
        shape=(3,432); zeros=np.zeros(shape)
        raw=np.full(shape,-1000.); floor=np.full(shape,np.log(9.e-30)); content=zeros.copy(); content[1,2]=1.
        # Stored synthetic probabilities only. Diagnostic never constructs replacements.
        p=np.full(shape,1/432); p[1]=0.; p[1,2]=1.
        data=dict(q_anchor=np.array([[0.,0.,1.],[0.,0.,1.],[0.,0.,-1.]]),content=content,
                  full__lograw=raw,full__logfloored=floor,full__logp=zeros.copy(),full__p=p,full__winner=p.argmax(-1).astype(np.int32))
        data['full__logp'][1,2]=1.
        terms={k:zeros.copy() for k in D.FIELDS};terms['td']=raw-math.log(9.);terms['s'][:]=1.
        rows,error=D.pair_rows(data,np.eye(3),terms,dict(scene=D.OLD.C.SCENES[0],pair_id='synthetic'),'full')
        self.assertEqual(len(rows),3);self.assertLess(error,1e-12)
        self.assertEqual(rows[0]['raw_winner'],0);self.assertFalse(rows[0]['raw_anchor_below_max'])
        self.assertEqual(rows[0]['stage_tie_counts']['raw'],432)
        self.assertTrue(rows[0]['stage_anchor_is_max']['raw'])
        self.assertFalse(rows[1]['stage_anchor_is_max']['logits'])
        self.assertTrue(rows[1]['content_changes_first']);self.assertTrue(rows[1]['final_first_miss'])
        self.assertIsNone(rows[2]['raw_comparison'])


if __name__ == '__main__': unittest.main()
