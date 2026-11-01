"""Pure synthetic reserve probe; never a production operator or effect study."""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
from scipy.special import erf, erfc
import torch

from experiments import round15_quartic_numerical_audit as R

ROOT = Path(__file__).resolve().parent


def save(name, value):
    with (ROOT/name).open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)


def flux(mean, cov, bounds, h, order, kernel='erf', segment=False):
    """Signed raw flux, deliberately no clamp/floor; all inputs synthetic."""
    s = np.asarray(cov)+h*h*np.eye(2)
    inv = np.linalg.inv(s)
    norm = 2*math.pi*math.sqrt(np.linalg.det(s))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    contributions = []
    for axis in range(2):
        other = 1-axis
        subdivisions = max(1, math.ceil(bounds[other+2]-bounds[other])) if segment else 1
        cuts = np.linspace(bounds[other], bounds[other+2], subdivisions+1)
        for edge, sign in ((bounds[axis], -1), (bounds[axis+2], 1)):
            for low, high in zip(cuts[:-1], cuts[1:]):
                p = np.empty((order, 2))
                p[:, axis] = edge
                p[:, other] = (high+low)/2+(high-low)/2*nodes
                d = p-mean
                q = np.einsum('ni,ij,nj->n', d, inv, d)
                if kernel == 'erf':
                    small = q < 1e-3
                    safe = np.where(small, 1., q)
                    f = erf(q/(2*math.sqrt(math.pi)))/safe
                    series = (1-q*q/(12*math.pi)+q**4/(160*math.pi**2))/math.pi
                    f = np.where(small, series, f)
                else:
                    f = erfc(q/(2*math.sqrt(math.pi)))/q
                contributions.append(float(sign*(high-low)/2*np.dot(weights, d[:, axis]*f)/norm))
    total = math.fsum(contributions)
    return dict(mass=total, l1_flux=math.fsum(abs(v) for v in contributions), edges=contributions)


def torch_flux(mean, cov, bounds, h, order=32):
    s = cov+h.square()*torch.eye(2, dtype=torch.float64)
    inv = torch.linalg.inv(s)
    norm = 2*math.pi*torch.linalg.det(s).sqrt()
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = torch.from_numpy(x), torch.from_numpy(w)
    total = torch.zeros((), dtype=torch.float64)
    for axis in range(2):
        other = 1-axis
        for edge, sign in ((bounds[axis], -1), (bounds[axis+2], 1)):
            vector = [None, None]
            vector[axis] = torch.ones_like(x)*edge
            vector[other] = (bounds[other+2]+bounds[other])/2+(bounds[other+2]-bounds[other])/2*x
            d = torch.stack(vector, -1)-mean
            q = torch.einsum('ni,ij,nj->n', d, inv, d)
            small = q < 1e-3
            safe = torch.where(small, torch.ones_like(q), q)
            f = torch.where(small, (1-q.square()/(12*math.pi)+q**4/(160*math.pi**2))/math.pi,
                            torch.erf(q/(2*math.sqrt(math.pi)))/safe)
            total = total+sign*(bounds[other+2]-bounds[other])/2*torch.dot(w, d[:, axis]*f)/norm
    return total


def main():
    source = Path(__file__).read_bytes()
    save('started.json', dict(created_unix=time.time(), source_sha256=hashlib.sha256(source).hexdigest(),
        independent_reference_sha256=hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(),
        real_observations_read=False, labels_read=False, effect_data_read=False,
        orders=[16,24,32,48], raw_tolerance=1e-8, normalized_tolerance=1e-4,
        purpose='synthetic mathematical reserve only, no replacement authorization'))
    cases, grids = R.cases()
    records = []
    for c in cases:
        ref = R.adaptive_rectangle(c['mean'], c['cov'], c['bounds'], c['h'])
        for order in (16,24,32,48):
            for segment in (False, True):
                actual = flux(c['mean'], c['cov'], c['bounds'], c['h'], order, segment=segment)
                children = R.subdivide(c['bounds'], 2)
                split = math.fsum(flux(c['mean'], c['cov'], b, c['h'], order, segment=segment)['mass'] for b in children)
                records.append(dict(name=c['name'], order=order, segment=segment, reference=ref['mass'],
                    actual=actual, error=abs(actual['mass']-ref['mass']), split_error=abs(split-actual['mass'])))
    save('rectangles.json', records)
    frecords = []
    for name, mean, cov, h in grids:
        bounds = R.native_grid(8)
        expected = np.array([R.adaptive_rectangle(mean, cov, b, h)['mass'] for b in bounds])
        for order in (16,24,32,48):
            actual = np.array([flux(mean, cov, b, h, order)['mass'] for b in bounds])
            floored = actual+1e-30*np.prod(bounds[:,2:]-bounds[:,:2], axis=-1)
            normalized = floored/floored.sum() if floored.sum()!=0 else np.zeros_like(floored)
            # Clipping is diagnostic only; even that does not fix the tail error.
            clipped = R.normalize(np.maximum(actual, 0.), bounds)
            row = dict(name=name, order=order, true_fov=float(expected.sum()), actual_fov=float(actual.sum()),
                raw_max_error=float(np.abs(actual-expected).max()), negative_cells=int((actual<0).sum()),
                minimum_raw=float(actual.min()), maximum_probability_error=float(np.abs(normalized-R.normalize(expected,bounds)).max()),
                diagnostic_clipped_probability_error=float(np.abs(clipped-R.normalize(expected,bounds)).max()))
            frecords.append(row)
            np.savez_compressed(ROOT/(name+f'_order{order}.npz'), bounds=bounds, reference=expected, flux=actual)
    save('fov.json', frecords)
    tailrecords = []
    cov = np.zeros((2,2)); h=.25; bounds=np.array([-.0625,-.0625,.9375,.9375])
    for mx in (-1.25,-1.35,-1.4,-1.5,-2.,-128.):
        mean = np.array([mx,.4]); ref=R.adaptive_rectangle(mean,cov,bounds,h)
        for order in (16,24,32,48):
            positive=flux(mean,cov,bounds,h,order)
            # Only outside-mu synthetic cases: winding exactly zero.
            complement=flux(mean,cov,bounds,h,order,kernel='erfc')
            tailrecords.append(dict(mean=mean.tolist(),order=order,reference=ref,
                smooth_erf=positive, outside_only_erfc_mass=-complement['mass']))
    save('tail_examples.json', tailrecords)
    gradients=[];bounds=np.array([-.0625,-.0625,.9375,.9375]);h=.5
    for label,mean,cov in [('generic',np.array([.23,-.19]),np.array([[.7,.18],[.18,.4]])),
        ('edge',np.array([-.0625,.4]),np.array([[.7,.18],[.18,.4]])),
        ('corner',np.array([-.0625,-.0625]),np.array([[.7,.18],[.18,.4]])),
        ('near_edge_inside',np.array([-.0625+1e-10,.4]),np.array([[.7,.18],[.18,.4]])),
        ('near_edge_outside',np.array([-.0625-1e-10,.4]),np.array([[.7,.18],[.18,.4]])),
        ('thin',np.array([.4,.4]),R.rotated_covariance((.02,64.),45))]:
        reference=R.gradient_reference(mean,cov,bounds,h)
        tm=torch.tensor(mean,dtype=torch.float64,requires_grad=True)
        tc=torch.tensor(cov,dtype=torch.float64,requires_grad=True)
        th=torch.tensor(h,dtype=torch.float64,requires_grad=True)
        value=torch_flux(tm,tc,bounds,th)
        gm,gc,gh=torch.autograd.grad(value,(tm,tc,th))
        actual=np.r_[gm.detach().numpy(),gc[0,0].item(),(gc[0,1]+gc[1,0]).item(),gc[1,1].item(),gh.item()]
        gradients.append(dict(name=label,actual=actual.tolist(),reference=reference.tolist(),max_error=float(np.abs(actual-reference).max())))
    save('gradients.json',gradients)
    save('completion.json', dict(source_sha256=hashlib.sha256(source).hexdigest(), rectangles=len(records),fov=len(frecords),
        tails=len(tailrecords),gradient_cases=len(gradients),labels_read=False,real_observations_read=False,
        no_production_or_validation_sources_changed=True))
    print(json.dumps(dict(rectangle_max=max(r['error'] for r in records), fov_records=frecords,
        gradient_max=max(r['max_error'] for r in gradients)),indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
