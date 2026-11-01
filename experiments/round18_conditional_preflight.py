"""Frozen synthetic-only numerical falsification of conditional Gaussian PE."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.special import roots_hermitenorm

from models.round18_conditional_pe import integrate, rectangle_masses
from experiments.round18_conditional_reference import (
    conditional_gl_reference, gaussian_marginal_reference,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'ROUND18_CONDITIONAL_PREFLIGHT_PROTOCOL.md',
    'models/round18_conditional_pe.py',
    'experiments/round18_conditional_preflight.py',
    'experiments/round18_conditional_reference.py',
    'docs/round18_alternative_operator_review.md',
    'docs/round18_conditional_normalization_math_review.md',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def grid(nx, ny):
    return np.asarray([[x-nx/2, y-ny/2, x+1-nx/2, y+1-ny/2]
                       for y in range(ny) for x in range(nx)], np.float64)


def rotated_covariance(a, b, angle):
    c, s = np.cos(angle), np.sin(angle)
    return [[a*c*c+b*s*s, (a-b)*c*s],
            [(a-b)*c*s, a*s*s+b*c*c]]


def fixtures():
    bounds = grid(5, 4)
    centers = .5 * (bounds[:, :2] + bounds[:, 2:])
    x, y = centers.T
    content = .45*np.sin(1.7*x) + .35*np.cos(.8*y) + .2*np.sin(x-y)
    specs = [
        ('zero', [.2, -.3], [[0, 0], [0, 0]], .25, 5.),
        ('nominal8', [.2, -.3], [[1/12, 0], [0, 1/12]], .25, 10.),
        ('nominal16', [.2, -.3], [[1/3, 0], [0, 1/3]], .25, 10.),
        ('nominal24', [.2, -.3], [[.75, 0], [0, .75]], .25, 10.),
        ('rotated', [.2, -.3], rotated_covariance(1.5, .1, np.pi/6), .25, 10.),
        ('rank1', [.2, -.3], [[.5, .5], [.5, .5]], .5, 5.),
        ('wide', [.2, -.3], [[4., 0], [0, 1.]], .25, 10.),
        ('boundary', [2.4, 1.8], rotated_covariance(.75, .25, np.pi/6), .25, 10.),
        ('outside', [5., 0], [[.75, 0], [0, .25]], .5, 5.),
        ('tiny_visible', [8., 0], [[.25, 0], [0, .25]], .25, 10.),
        ('floor_dominated', [100., 100.], [[.25, 0], [0, .25]], .5, 10.),
    ]
    rows = []
    for name, mean, cov, h, temperature in specs:
        for t in (0., temperature):
            rows.append(dict(name=name+'_t'+str(int(t)), mean=mean, covariance=cov,
                h=h, temperature=t, bounds=bounds.tolist(), content=content.tolist()))
    peak = np.full(len(bounds), -.8)
    peak[np.argmin(((centers-np.asarray([1., .5]))**2).sum(-1))] = .9
    repeated = np.full(len(bounds), -.8)
    for target in ([-1., .5], [1., .5]):
        repeated[np.argmin(((centers-np.asarray(target))**2).sum(-1))] = .9
    target = np.asarray([1., .5])
    distance2 = ((centers-target)**2).sum(-1)
    correction = -distance2/distance2.max()
    for name, scores, mean in [('content_spike', peak, [.2, -.3]),
                              ('repeated_peaks', repeated, [.2, -.3]),
                              ('content_correction', correction, [-.75, -.5])]:
        rows.append(dict(name=name, mean=mean, covariance=[[1/3, 0], [0, 1/3]],
            h=.25, temperature=10., bounds=bounds.tolist(), content=scores.tolist(),
            intended_content_location=(target.tolist() if name=='content_correction' else None)))
    return rows


def environment():
    builds = {}
    for name, show in (('numpy', np.show_config), ('scipy', scipy.show_config)):
        stream=io.StringIO()
        with redirect_stdout(stream):
            show()
        builds[name+'_build_sha256']=hashlib.sha256(stream.getvalue().encode()).hexdigest()
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                scipy=scipy.__version__, platform=platform.platform(), **builds,
                threads={name:os.environ.get(name) for name in
                         ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})


def freeze(out):
    if out.exists():
        raise ValueError('refuse existing output directory')
    sources = {name: sha(ROOT/name) for name in SOURCES}
    out.mkdir(parents=True)
    write(out/'fixtures.json', fixtures())
    lock = dict(created_unix=time.time(), source_hashes=sources,
        fixture_sha256=sha(out/'fixtures.json'), environment=environment(),
        source_decision_sha256=sha(ROOT/'results/round17_validation/decision.json'),
        production_order=64, diagnostic_order=96, reference_orders=[384, 512],
        reference_standard_normal_extent=14., numerical_probability_tolerance=1e-4,
        reference_tolerance=1e-8, identity_tolerance=2e-12,
        synthetic_only=True, real_inputs_read=False, scientific_gate_passed=False)
    write(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    for name in SOURCES:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    verify(out)
    print('FROZEN', sha(out/'lock.json'), flush=True)


def verify(out):
    lock = read(out/'lock.json')
    assert sha(out/'lock.json') == (out/'lock.sha256').read_text().strip()
    assert sha(out/'fixtures.json') == lock['fixture_sha256']
    assert environment() == lock['environment'], 'runtime changed'
    assert sha(ROOT/'results/round17_validation/decision.json') == lock['source_decision_sha256']
    for name, digest in lock['source_hashes'].items():
        assert sha(ROOT/name) == digest, name
        assert sha(out/'source_snapshot'/name) == digest, name
    return lock


def difference(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def identities():
    b = grid(4, 4)
    coarse = np.asarray([[-2, -2, 0, 0], [0, -2, 2, 0],
                         [-2, 0, 0, 2], [0, 0, 2, 2]], np.float64)
    group = np.asarray([(y//2)*2+x//2 for y in range(4) for x in range(4)])
    scores = np.asarray([-.8, .3, .6, -.2])
    mu = np.asarray([.15, -.1]); cov = np.asarray([[.8, .25], [.25, .4]])
    args = (mu, cov, b, scores[group], .5, 5.)
    base = integrate(*args)['probability']
    coarse_p = integrate(mu, cov, coarse, scores, .5, 5.)['probability']
    merged = np.asarray([base[group == i].sum() for i in range(4)])
    perm = np.arange(15, -1, -1)
    permutation = integrate(mu, cov, b[perm], scores[group][perm], .5, 5.)['probability']
    offset = np.asarray([1.25, -.75])
    translated = integrate(mu+offset, cov, b+np.tile(offset, 2), scores[group], .5, 5.)['probability']
    scale = 3.7
    scaled = integrate(mu*scale, cov*scale**2, b*scale, scores[group], .5*scale, 5.,
                       floor_density=1e-30/scale**2)['probability']
    constant = integrate(mu, cov, b, np.full(16, .37), .5, 10.)['probability']
    t0 = integrate(mu, cov, b, scores[group], .5, 0.)['probability']
    m = rectangle_masses(mu[None], b, .5)[0]
    weighted = m * np.exp(5*(scores[group]-scores.max()))
    point = integrate(mu, np.zeros((2, 2)), b, scores[group], .5, 5.)['probability']
    errors = dict(fixed_content_merge=difference(merged, coarse_p),
        key_permutation=difference(base[perm], permutation),
        joint_translation=difference(base, translated), joint_units=difference(base, scaled),
        constant_content=difference(t0, constant),
        point_limit=difference(point, weighted/weighted.sum()))
    # Whole-plane, no floor, genuinely continuous LINEAR content. These
    # moment identities are deliberately not imposed on native-cell scores.
    z, w = roots_hermitenorm(64); w = w/np.sqrt(2*np.pi)
    weights = np.multiply.outer(w, w).reshape(-1)
    standard = np.stack(np.meshgrid(z, z, indexing='ij'), axis=-1).reshape(-1, 2)
    e, v = np.linalg.eigh(cov)
    locations = mu + standard @ (v*np.sqrt(e)[None, :]).T
    g = np.asarray([.2, -.15]); temperature=5.; h=.5
    local_centers = locations + temperature*h*h*g
    new_mean = np.einsum('n,nd->d', weights, local_centers)
    delta = local_centers-new_mean
    new_cov = np.einsum('n,ni,nj->ij', weights, delta, delta) + h*h*np.eye(2)
    old_weights = weights*np.exp(temperature*(locations@g))
    old_weights /= old_weights.sum()
    old_mean = np.einsum('n,nd->d', old_weights, local_centers)
    errors.update(linear_new_mean=difference(new_mean, mu+temperature*h*h*g),
        linear_new_covariance=difference(new_cov, cov+h*h*np.eye(2)),
        linear_old_mean=difference(old_mean, mu+temperature*(cov+h*h*np.eye(2))@g))
    rejected = {}
    for name, wrong in [('negative_covariance', np.diag([-1., 1.])),
                        ('asymmetric_covariance', np.asarray([[1., .2], [0., 1.]])),
                        ('nonfinite_covariance', np.asarray([[np.nan, 0.], [0., 1.]]))]:
        try:
            integrate(mu, wrong, b, scores[group], .5, 5.)
        except ValueError:
            rejected[name] = True
        else:
            rejected[name] = False
    return dict(errors=errors, rejected_inputs=rejected,
        passed=all(v <= 2e-12 for v in errors.values()) and all(rejected.values()),
        linear_scope='whole plane, no floor, continuous linear field only')


def run(out):
    lock = verify(out)
    if (out/'started.json').exists():
        raise ValueError('refuse to overwrite started or completed run')
    write(out/'started.json', dict(started_unix=time.time(), lock_sha256=sha(out/'lock.json')))
    rows = []
    for fixture in read(out/'fixtures.json'):
        args = tuple(fixture[k] for k in ('mean','covariance','bounds','content','h','temperature'))
        outputs = {}; durations = {}
        for name, function, order in (
            ('production64', integrate, 64), ('diagnostic96', integrate, 96),
            ('reference384', conditional_gl_reference, 384),
            ('reference512', conditional_gl_reference, 512)):
            start=time.perf_counter(); outputs[name]=function(*args, order=order)
            durations[name]=time.perf_counter()-start
        p = outputs['production64']['probability']; d=outputs['diagnostic96']['probability']
        r=outputs['reference512']; prior=outputs['reference384']
        rp=np.asarray(r['P']); pp=np.asarray(prior['P'])
        errors = dict(production_probability=difference(p, rp),
            diagnostic96_probability=difference(d, rp), reference_probability=difference(pp, rp),
            reference_z_relative=float(abs(prior['Zg']/r['Zg']-1)),
            production_z_relative=float(abs(outputs['production64']['z_geometry']/r['Zg']-1)),
            reference_amplified_tail=float(r['amplified_tail_bound']),
            certified_amplified_tail=float(r['certified_amplified_tail_bound']))
        arrays = {}
        for name, result in outputs.items():
            for key, value in result.items():
                if isinstance(value, (np.ndarray, np.number, float, int)):
                    arrays[name+'__'+key]=np.asarray(value)
        probability_rows=[p,d,pp,rp]
        checks=dict(probabilities_valid=all(np.isfinite(v).all() and (v >= 0).all()
            and abs(v.sum()-1) <= 2e-12 for v in probability_rows),
            production_probability=errors['production_probability'] <= 1e-4,
            reference_probability=errors['reference_probability'] <= 1e-8,
            reference_z=errors['reference_z_relative'] <= 1e-8,
            reference_tail=errors['certified_amplified_tail'] <= 1e-8)
        if fixture['temperature'] == 0:
            analytic=gaussian_marginal_reference(fixture['mean'],fixture['covariance'],
                fixture['bounds'],fixture['h'])
            errors['production_t0_marginal']=difference(p, analytic['P'])
            errors['reference_t0_marginal']=difference(rp, analytic['P'])
            checks['production_t0_marginal']=errors['production_t0_marginal'] <= 1e-4
            checks['reference_t0_marginal']=errors['reference_t0_marginal'] <= 1e-8
            for key,value in analytic.items():
                if isinstance(value,(np.ndarray,np.number,float,int)):
                    arrays['marginal__'+key]=np.asarray(value)
        path=out/(fixture['name']+'.npz');np.savez_compressed(path,**arrays)
        row=dict(name=fixture['name'], errors=errors, checks=checks, passed=all(checks.values()),
            z_geometry=float(r['Zg']), timings_seconds=durations,
            bare_vs_conditioned_max_difference=difference(outputs['production64']['bare_expectation'],p),
            old_vs_conditioned_max_difference=difference(outputs['production64']['old_mix_then_normalize'],p),
            artifact=path.name,sha256=sha(path))
        if fixture.get('intended_content_location') is not None:
            # A declared synthetic correction scenario, never a selection gate.
            centers=.5*(np.asarray(fixture['bounds'])[:,:2]+np.asarray(fixture['bounds'])[:,2:])
            target=np.asarray(fixture['intended_content_location'])
            scores=np.asarray(fixture['content'])
            reference_old=r['geometry_mass']*np.exp(fixture['temperature']*(scores-scores.max()))
            reference_old/=reference_old.sum()
            row['synthetic_content_correction']=dict(target=target.tolist(),
                conditioned_mean_distance=float(np.linalg.norm(rp@centers-target)),
                old_mean_distance=float(np.linalg.norm(reference_old@centers-target)),
                both_use_same_reference512=True,
                descriptive_only=True)
        rows.append(row);write(out/(fixture['name']+'.json'),row)
        print('SYNTHETIC_CASE',row['name'],'passed',row['passed'],'error',errors['production_probability'],flush=True)
    identity=identities();write(out/'identities.json',identity)
    verify(out)
    result=dict(rows=rows,identities=identity,synthetic_numerical_passed=all(r['passed'] for r in rows) and identity['passed'],
        scientific_gate_passed=False,real_inputs_read=False,fresh_inputs_read=False,
        actual_value_run=False,optimized_gs_run=False,general_goal_achieved=False,
        production_order=64,diagnostic_order_promoted=False,
        lock_sha256=sha(out/'lock.json'),finished_unix=time.time())
    result['next_action']=('separate_old_train_protocol_required' if result['synthetic_numerical_passed']
                           else 'stop_order64_preserve_all_numerical_failures')
    write(out/'summary.json',result)
    write(out/'completion.json',dict(finished_unix=time.time(),files={str(p.relative_to(out)):sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()},
        synthetic_numerical_passed=result['synthetic_numerical_passed'],scientific_gate_passed=False))
    print('SYNTHETIC_COMPLETE',result['synthetic_numerical_passed'],result['next_action'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run'))
    parser.add_argument('--out',type=Path,default=Path('results/round18_conditional_preflight'))
    args=parser.parse_args()
    (freeze if args.command=='freeze' else run)(args.out.resolve())


if __name__=='__main__':
    main()
