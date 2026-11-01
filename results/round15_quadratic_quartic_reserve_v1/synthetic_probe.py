"""Exploratory fixed-a mathematical reserve; no real inputs or selection.

Standalone reproducible calculations after preliminary analytic/scratch work.
This is not a preregistered real-data experiment or a production PE operator.
"""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfcx


OUT = Path(__file__).resolve().parent
A = .25
CONTENT_STRENGTHS = (.001, .01, .05, .1)


def write(name, value):
    with (OUT/name).open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)


def radial_z(a, b):
    if b == 0:
        return 1/a
    return math.sqrt(math.pi)/(2*math.sqrt(b))*erfcx(a/(2*math.sqrt(b)))


def radial_mean(a, b):
    return (1/radial_z(a,b)-a)/(2*b) if b else 1/a


def solve_b():
    # A fixed bracket for the unique covariance equation, not task scoring.
    return brentq(lambda b: radial_mean(A,b)-2., 1e-5, .1, xtol=1e-15)


def box_statistics(a, b, z, x_center, y_center, half_width, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = x_center+half_width*nodes[:,None]
    y = y_center+half_width*nodes[None,:]
    q = x*x+y*y
    weighted = weights[:,None]*weights[None,:]*np.exp(-a*q-b*q*q)
    denominator = weighted.sum()
    score_x = -(2*a+4*b*q)*x
    gradient = float((weighted*score_x).sum()/denominator)
    second = -2*a-4*b*(q+2*x*x)
    curvature = float((weighted*(second+score_x**2)).sum()/denominator-gradient**2)
    return dict(mass=float(denominator*half_width**2/(math.pi*z)),
        mean_x=float((weighted*x).sum()/denominator), mean_y=float((weighted*y).sum()/denominator),
        second_x=float((weighted*x*x).sum()/denominator),
        second_y=float((weighted*y*y).sum()/denominator),
        cross=float((weighted*x*y).sum()/denominator),
        log_mass_x_derivative=gradient, log_mass_x_curvature=curvature)


def main():
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write('started.json', dict(created_unix=time.time(), source_sha256=digest,
        exploratory=True, fixed_a=A, covariance_target=2., content_strengths=CONTENT_STRENGTHS,
        a_grid_search=False, real_observations_read=False, labels_read=False,
        preexisting_effects_read=False, production_changes=False,
        declaration='Archive follows preliminary math/scratch calculations; no false preregistration claim.'))
    b = solve_b(); z = radial_z(A,b)
    moments = {n:quad(lambda q:q**n*math.exp(-A*q-b*q*q),0,math.inf,
        epsabs=1e-12,epsrel=1e-12)[0]/z for n in range(5)}
    variance = moments[2]-moments[1]**2
    covariance = moments[3]-moments[1]*moments[2]
    coefficient = dict(a=A,b=b,radial_normalizer=z,radial_normalizer_from_identity=1/(A+4*b),
        spatial_normalizer_at_det_one=1/(math.pi*z),radial_moments=moments,
        derivative_mean_wrt_b=-covariance, derivative_b_wrt_a=-variance/covariance,
        covariance_equation_residual=radial_mean(A,b)-2.)
    records = []
    for family,a,bb in (('gaussian',.5,0.),('quartic',0.,1/(4*math.pi)),('fixed_quadratic_quartic',A,b)):
        iz=radial_z(a,bb)
        domain64=box_statistics(a,bb,iz,0.,0.,10.,128)
        domain96=box_statistics(a,bb,iz,0.,0.,10.,192)
        # Uniform square [-sqrt(3),sqrt(3)]² has Cov=I, Eq=2, Eq²=28/5.
        kl=math.log(math.pi*iz/12)+2*a+(28/5)*bb
        nodes,weights=np.polynomial.legendre.leggauss(32)
        x=math.sqrt(3)*nodes[:,None]; y=math.sqrt(3)*nodes[None,:];q=x*x+y*y
        kl_quad=float((weights[:,None]*weights[None,:]*(-math.log(12)+math.log(math.pi*iz)+a*q+bb*q*q)).sum()/4)
        native0=box_statistics(a,bb,iz,0.,0.,.5,96)
        native1=box_statistics(a,bb,iz,1.,0.,.5,96)
        modes=[]
        for strength in CONTENT_STRENGTHS:
            point=brentq(lambda r:2*a*r+4*bb*r**3-strength,0.,4.,xtol=1e-14)
            integrated=[]
            for order in (48,96):
                integrated.append(brentq(lambda r:box_statistics(a,bb,iz,r,0.,.5,order)
                    ['log_mass_x_derivative']+strength,0.,4.,xtol=1e-14))
            modes.append(dict(strength=strength,continuous_point_mode=point,
                sliding_unit_cell_mode=integrated[1],cell_mode_order48_96_difference=abs(integrated[0]-integrated[1])))
        records.append(dict(family=family,a=a,b=bb,I0=iz,continuous_uniform_square_kl=kl,
            uniform_square_kl_quadrature_error=abs(kl_quad-kl),
            large_square_moments_order128=domain64,large_square_moments_order192=domain96,
            point_center_log_hessian_xx=-2*a,unit_cell_center_log_hessian_xx=native0['log_mass_x_curvature'],
            unit_cell_center_mass=native0['mass'],unit_cell_neighbor_mass=native1['mass'],
            point_neighbor_tilt_threshold=a+bb,
            unit_cell_neighbor_tilt_threshold=math.log(native0['mass']/native1['mass']),
            content_modes=modes))
    result=dict(exploratory=True,coefficient=coefficient,comparisons=records,
        parameter_selection=False,production_operator_implemented=False,
        native_integration_certified=False,scientific_gate_passed=False,
        interpretation='Intermediate weak-tilt stability with a uniform-square KL tradeoff; no empirical benefit claim.')
    write('result.json',result)
    write('completion.json',dict(source_sha256=digest,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()},
        exploratory=True,real_observations_read=False,labels_read=False,finished_unix=time.time()))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
