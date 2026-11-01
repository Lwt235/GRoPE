"""Independent NumPy/SciPy R17 numerical reference and frozen synthetic audit.

Reference functions never import production. Only run_synthetic's actual path
imports the Torch model and frozen native Gaussian integrator. This module has
no dataset/label loader, selection or real-effect evaluation.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr, softmax

WIDTHS = (.25, .5, 1., 2.)
TEMPERATURES = (0., 5., 10.)
PARTITIONS = ('g8', 'g16', 'mixed', 'g24')
REFERENCE_ORDERS = (64, 96)
PROBABILITY_TOLERANCE = 1e-4
REFERENCE_TOLERANCE = 1e-8
SAME_MASS_TOLERANCE = 1e-12
FLOOR = 1e-30
SOURCE_NAMES = ('docs/round17_trace_theory.md',
    'experiments/round17_trace_numerical_audit.py',
    'tests/test_round17_trace_numerical_audit.py',
    'models/round17_trace_pe.py', 'models/round12_stress_pe.py',
    'models/round10_cell_integral_fast.py', 'models/round10_cell_integral.py')


def _covariance(cov):
    value=np.asarray(cov,np.float64)
    if value.ndim<2 or value.shape[-2:]!=(2,2) or not value.size or not np.isfinite(value).all():
        raise ValueError('finite covariance matrices ending in 2x2 required')
    scale=np.maximum(1.,np.max(np.abs(value),axis=(-1,-2)))
    if np.any(np.abs(value-value.swapaxes(-1,-2))>1e-12*scale[...,None,None]):
        raise ValueError('covariance must be symmetric')
    if np.any(np.linalg.eigvalsh(value)<-1e-12*scale[...,None]):
        raise ValueError('covariance must be PSD')
    if np.any(value[...,0,0]/scale+value[...,1,1]/scale<0):
        raise ValueError('covariance trace must be nonnegative')
    return value


def _scalars(h,T):
    h=float(h);T=float(T)
    if not np.isfinite(h) or h<=0 or not np.isfinite(h*h) or h*h<=0 or not np.isfinite(T) or T<0:
        raise ValueError('finite positive h and nonnegative content strength required')
    return h,T


def independent_temperature(cov,h,T):
    """Independent NumPy evaluation of the locked scaled float64 ratio.

    The evaluation order is an implementation contract, not a shared helper:
    s=max(h^2,C00,C11); r=h^2/s; c=C00/s+C11/s; tau=T*(r/(r+c)).
    Tests compare this against the unscaled mathematical formula separately.
    """
    covariance=_covariance(cov);h,T=_scalars(h,T)
    h2=h*h;scale=np.maximum(h2,np.maximum(covariance[...,0,0],covariance[...,1,1]))
    relative_h2=h2/scale
    scaled_trace=covariance[...,0,0]/scale+covariance[...,1,1]/scale
    return T*(relative_h2/(relative_h2+scaled_trace))


def independent_bias(content,cov,h,T,*,float32_bias=True):
    content=np.asarray(content)
    tau=independent_temperature(cov,h,T)
    if content.ndim!=tau.ndim+1 or content.shape[:-1]!=tau.shape or not np.isfinite(content).all():
        raise ValueError('content and covariance query populations differ')
    if float32_bias:
        if content.dtype!=np.float32:raise ValueError('the actual content path requires float32')
        return (tau.astype(np.float32)[...,None]*content).astype(np.float64)
    return tau[...,None]*content.astype(np.float64)


def _bounds(bounds):
    b=np.asarray(bounds,np.float64)
    if b.ndim!=2 or b.shape[1]!=4 or len(b)==0 or not np.isfinite(b).all() or np.any(b[:,2:]<=b[:,:2]):
        raise ValueError('finite nonempty positive native rectangles required')
    return b


@lru_cache(None)
def _rule(order):
    if type(order) is not int or order<8:raise ValueError('invalid quadrature order')
    return np.polynomial.legendre.leggauss(order)


def _interval(lo,hi):
    # SciPy survival-CDF subtraction for the positive tail. No Torch erfc or
    # production interval helper is used by this reference.
    return np.maximum(np.where(lo>=0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo)),0.)


def independent_mass(mean,cov,bounds,h,order=64):
    """Raw Gaussian rectangle mass, whole native cells, no floor or tiling.

    mean/cov use reference8; bounds=[x0,y0,x1,y1] also use reference8.
    GL64/96 are independent positive conditional-CDF quadratures. Large native
    24px cells are integrated as their original width3 rectangle.
    """
    mean=np.asarray(mean,np.float64);cov=_covariance(cov);b=_bounds(bounds)
    h,_=_scalars(h,0.)
    if mean.ndim<2 or mean.shape[-1]!=2 or cov.shape!=mean.shape[:-1]+(2,2) or not np.isfinite(mean).all():
        raise ValueError('mean/covariance query shapes differ')
    shape=mean.shape[:-1];m=mean.reshape(-1,2)
    S=cov.reshape(-1,2,2)+h*h*np.eye(2)
    var_y=S[:,1,1]-S[:,1,0]**2/S[:,0,0]
    if np.any(var_y<=0):raise ValueError('regularized covariance numerically singular')
    nodes,weights=_rule(order)
    mid=(b[:,0]+b[:,2])*.5;half=(b[:,2]-b[:,0])*.5
    x=mid[:,None]+half[:,None]*nodes
    out=np.empty((len(m),len(b)),np.float64)
    for start in range(0,len(m),8):
        mu=m[start:start+8];s=S[start:start+8]
        sx=np.sqrt(s[:,0,0])[:,None,None]
        slope=(s[:,1,0]/s[:,0,0])[:,None,None]
        sy=np.sqrt(var_y[start:start+8])[:,None,None]
        delta=x[None]-mu[:,None,0,None]
        ymean=mu[:,None,1,None]+slope*delta
        yprob=_interval((b[None,:,1,None]-ymean)/sy,(b[None,:,3,None]-ymean)/sy)
        density=np.exp(-.5*(delta/sx)**2)/(np.sqrt(2*np.pi)*sx)
        out[start:start+8]=half[None]*np.sum(yprob*density*weights,axis=-1)
    return out.reshape(*shape,len(b))


def independent_probability(raw,content,cov,h,T,bounds,*,float32_bias=True):
    """Area floor once, independent double tau -> float32 product -> softmax."""
    raw=np.asarray(raw,np.float64);b=_bounds(bounds)
    if raw.shape!=np.asarray(content).shape or raw.shape[-1]!=len(b) or not np.isfinite(raw).all() or np.any(raw<0):
        raise ValueError('raw mass/content/native key shape or values invalid')
    area=np.prod(b[:,2:]-b[:,:2],axis=1)
    logits=np.log(raw+FLOOR*area)+independent_bias(content,cov,h,T,float32_bias=float32_bias)
    return softmax(logits,axis=-1)


def probability_check(actual,reference64,reference96,raw96):
    actual=np.asarray(actual);p64=np.asarray(reference64);p96=np.asarray(reference96)
    if any(x.dtype!=np.float64 or x.ndim!=2 or x.shape!=p96.shape or not np.isfinite(x).all()
           or np.any(x<0) or np.any(np.abs(x.sum(1)-1)>1e-12) for x in (actual,p64,p96)):
        raise ValueError('invalid comparable normalized probability arrays')
    error=float(np.max(np.abs(actual-p96)));convergence=float(np.max(np.abs(p64-p96)))
    a=actual.argmax(1);r=p96.argmax(1);changes=[]
    for q in np.flatnonzero(a!=r):
        top=np.sort(p96[q])[-2:];gap=float(top[1]-top[0])
        changes.append(dict(query_index=int(q),actual_argmax=int(a[q]),reference_argmax=int(r[q]),
            reference_top_gap=gap,exact_reference_tie=gap==0.,near_reference_tie=gap<=1e-8,
            reference_advantage=float(p96[q,r[q]]-p96[q,a[q]])))
    fov=np.sum(raw96,axis=1)
    return dict(passed=error<=PROBABILITY_TOLERANCE and convergence<=REFERENCE_TOLERANCE,
        max_probability_error=error,max_reference_convergence_error=convergence,
        argmax_change_count=len(changes),argmax_changes=changes,
        raw_fov_mass=fov.tolist(),zero_fov_rows=int(np.count_nonzero(fov==0)),
        tiny_fov_rows=int(np.count_nonzero(fov<1e-20)),zero_raw_cells=int(np.count_nonzero(raw96==0)))


def synthetic_cases():
    """Fixed before the audit run; no true-image data or sampled case search."""
    def rotated(large,small,angle):
        c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s],[s,c]])
        return R@np.diag([large,small])@R.T
    cases=[
        ('point_center',[11.9375,8.9375],np.zeros((2,2))),
        ('pixel_footprint',[4.137,7.223],np.eye(2)/(12*64)),
        ('rank1_positive',[9.2,6.1],[[4,2],[2,1]]),
        ('rank1_negative',[9.2,6.1],[[4,-2],[-2,1]]),
        ('rotated_anisotropic',[11.1,8.4],rotated(4.,.02,np.pi/6)),
        ('large_rotated',[7.1,5.4],rotated(16384.,.02,np.pi/4)),
        ('large_isotropic',[11.9375,8.9375],np.eye(2)*16384),
        ('left_edge',[-.0625,8.9375],np.zeros((2,2))),
        ('lower_left_corner',[-.0625,-.0625],np.zeros((2,2))),
        ('right_edge',[23.9375,8.9375],[[.08,.03],[.03,.11]]),
        ('outside_left',[-.5625,9.],[[.08,.03],[.03,.11]]),
        ('tiny_tail',[-2.5,8.9375],np.zeros((2,2))),
        ('far_tail',[-30.,8.9375],np.zeros((2,2))),
        ('all_underflow',[1000.,1000.],np.zeros((2,2))),
        ('outside_corner',[26.4375,20.4375],np.zeros((2,2))),
        ('float32_trace_overflow',[11.9375,8.9375],np.eye(2)*2e38),
        ('near_edge',[-.0625+1e-6,9.],rotated(.2,.01,np.pi/5)),
        ('four_cell_tie',[.9375,.9375],np.zeros((2,2))),
        ('isotropic_nonzero',[.4375,.4375],np.eye(2)*.0625),
    ]
    return dict(names=[x[0] for x in cases],mean=np.asarray([x[1] for x in cases],np.float32),
                covariance=np.asarray([x[2] for x in cases],np.float32))


def synthetic_partitions():
    """Literal fixed192x144 native bounds; mixed has both8/16 cells."""
    output={}
    for name in PARTITIONS:
        step=8 if name=='g8' else 24 if name=='g24' else 16
        bounds=[]
        for y in range(0,144,step):
            for x in range(0,192,step):
                if name=='mixed' and (x//16+y//16)%2:
                    bounds.extend([[a,b,a+8,b+8] for b in (y,y+8) for a in (x,x+8)])
                else:bounds.append([x,y,x+step,y+step])
        native=np.asarray(bounds,np.int64)
        output[name]=dict(native_bounds=native,bounds=(native-.5)/8)
    return output


def synthetic_content(mean,bounds):
    centers=(bounds[:,:2]+bounds[:,2:])*.5
    return (.6*np.sin((mean[:,0,None]+1)*(centers[None,:,0]+.3)*.13)
            +.3*np.cos((mean[:,1,None]-centers[None,:,1])*.21)).astype(np.float32)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_synthetic(out):
    """Run only after the outer orchestrator's joint source freeze.

    Refuses overwrite. Writes every fixed probability/reference/raw/tau/bias
    array plus all failed checks; outer code owns the global lock/completion.
    This is not a standalone freeze or a scientific effect test.
    """
    import torch
    from models import round12_stress_pe as G
    from models import round17_trace_pe as P
    out=Path(out)
    if out.exists():raise FileExistsError('synthetic audit output already exists')
    out.mkdir(parents=True)
    began=time.time();cases=synthetic_cases();parts=synthetic_partitions()
    mean,cov=cases['mean'],cases['covariance'];records=[];arrays={'mean32':mean,'covariance32':cov}
    mt=torch.from_numpy(mean);ct=torch.from_numpy(cov)
    with torch.no_grad():
        for name,part in parts.items():
            bounds=part['bounds'];native=part['native_bounds'];content=synthetic_content(mean,bounds)
            children,owner=G.split_cell_bounds(native)
            key=dict(cell_bounds=torch.from_numpy(children),cell_owner=torch.from_numpy(owner),
                     mass=torch.from_numpy(np.prod(native[:,2:]-native[:,:2],axis=1).astype(np.float64)))
            arrays[name+'__bounds']=bounds;arrays[name+'__content32']=content
            for hi,h in enumerate(WIDTHS):
                raw=G.tiled_cell_mass(mt,ct,key,h)
                raw_np=raw.numpy();r64=independent_mass(mean,cov,bounds,h,64);r96=independent_mass(mean,cov,bounds,h,96)
                geometry=(raw+FLOOR*(key['mass']/64)[None]).log()
                prefix=f'{name}__h{hi}'
                arrays[prefix+'__raw32']=raw_np;arrays[prefix+'__raw64']=r64;arrays[prefix+'__raw96']=r96
                for ti,T in enumerate(TEMPERATURES):
                    tau=P.trace_temperature(ct,h,T).numpy();bias=P.content_bias(ct,h,T,torch.from_numpy(content)).numpy()
                    actual=P.probabilities(geometry,torch.from_numpy(content),ct,h,T).numpy()
                    tau_ref=independent_temperature(cov,h,T).astype(np.float32)
                    bias_ref=independent_bias(content,cov,h,T).astype(np.float32)
                    p64=independent_probability(r64,content,cov,h,T,bounds)
                    p96=independent_probability(r96,content,cov,h,T,bounds)
                    same=independent_probability(raw_np,content,cov,h,T,bounds)
                    check=probability_check(actual,p64,p96,r96)
                    same_error=float(np.max(np.abs(actual-same)))
                    tau_equal=np.array_equal(tau,tau_ref);bias_equal=np.array_equal(bias,bias_ref)
                    point=np.all(cov==0,axis=(1,2))
                    point_expected=(geometry[point]+T*torch.from_numpy(content[point])).softmax(-1).numpy()
                    point_equal=np.array_equal(actual[point],point_expected)
                    zero_equal=T!=0 or np.array_equal(actual,geometry.softmax(-1).numpy())
                    check.update(partition=name,h=h,temperature=T,queries=len(mean),keys=len(bounds),
                        tau32_exact=bool(tau_equal),bias32_exact=bool(bias_equal),
                        point_identity_exact=bool(point_equal),zero_temperature_identity_exact=bool(zero_equal),
                        same_mass_probability_error=same_error)
                    check['passed']=bool(check['passed'] and tau_equal and bias_equal and point_equal and zero_equal
                                         and same_error<=SAME_MASS_TOLERANCE)
                    records.append(check)
                    for field,value in (('actual',actual),('reference64',p64),('reference96',p96),('tau32',tau),('bias32',bias)):
                        arrays[f'{prefix}__t{ti}__{field}']=value
    np.savez_compressed(out/'arrays.npz',**arrays)
    report=dict(passed=all(x['passed'] for x in records),synthetic_only=True,labels_read=False,
        new_real_predictions=False,scientific_gate_passed=False,case_names=cases['names'],
        cases=len(mean),matrix_count=len(records),probability_rows=len(mean)*len(records),
        reference_orders=list(REFERENCE_ORDERS),production_order=32,
        probability_tolerance=PROBABILITY_TOLERANCE,reference_tolerance=REFERENCE_TOLERANCE,
        same_mass_tolerance=SAME_MASS_TOLERANCE,argmax_is_reported_not_an_independent_gate=True,
        max_probability_error=max(x['max_probability_error'] for x in records),
        max_reference_convergence_error=max(x['max_reference_convergence_error'] for x in records),
        max_same_mass_probability_error=max(x['same_mass_probability_error'] for x in records),
        argmax_change_count=sum(x['argmax_change_count'] for x in records),checks=records,
        arrays_sha256=_sha(out/'arrays.npz'),started_unix=began,finished_unix=time.time())
    (out/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
