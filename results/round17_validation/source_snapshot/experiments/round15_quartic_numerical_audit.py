"""Independent Cartesian adaptive integration for covariance-matched quartics.

This file reads no experiment observations or labels. The reference does not
use ray angles, erf radial antiderivatives, or the actual integration model.
Every actual-model run writes an immutable record, including numerical failures.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import time
import traceback
import warnings
import numpy as np
from scipy.integrate import IntegrationWarning, quad

ROOT=Path(__file__).resolve().parents[1]
PI=math.pi
PROBABILITY_TOLERANCE=1e-4
RAW_ABSOLUTE_TOLERANCE=1e-8
SPLIT_ABSOLUTE_TOLERANCE=1e-8
GRADIENT_TOLERANCE=2e-5


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def effective_covariance(covariance,bandwidth):
    c=np.asarray(covariance,np.float64)
    if c.shape!=(2,2) or not np.isfinite(c).all() or not np.allclose(c,c.T,atol=1e-12,rtol=0):
        raise ValueError('finite symmetric 2D covariance required')
    if np.linalg.eigvalsh(c).min()<-1e-10 or not np.isfinite(bandwidth) or bandwidth<=0:
        raise ValueError('PSD covariance and positive bandwidth required')
    return c+float(bandwidth)**2*np.eye(2)


def density(point,mean,covariance,bandwidth):
    c=effective_covariance(covariance,bandwidth);delta=np.asarray(point)-mean
    q=float(delta@np.linalg.solve(c,delta))
    return math.exp(-q*q/(4*PI))/(PI**2*math.sqrt(np.linalg.det(c)))


def rectangle_minimum(mean,covariance,bounds):
    """Exact constrained quadratic minimum via interior and four clipped edges."""
    mean=np.asarray(mean,float);bounds=np.asarray(bounds,float)
    if bounds.shape!=(4,) or not np.isfinite(bounds).all() or np.any(bounds[2:]<=bounds[:2]):
        raise ValueError('finite nonempty rectangle required')
    precision=np.linalg.inv(covariance);candidates=[]
    if np.all(mean>=bounds[:2]) and np.all(mean<=bounds[2:]):return 0.,mean.copy()
    for x in (bounds[0],bounds[2]):
        y=np.clip(mean[1]-precision[1,0]/precision[1,1]*(x-mean[0]),bounds[1],bounds[3])
        candidates.append(np.array([x,y]))
    for y in (bounds[1],bounds[3]):
        x=np.clip(mean[0]-precision[0,1]/precision[0,0]*(y-mean[1]),bounds[0],bounds[2])
        candidates.append(np.array([x,y]))
    values=[float((x-mean)@precision@(x-mean)) for x in candidates]
    i=int(np.argmin(values));return max(0.,values[i]),candidates[i]


def adaptive_rectangle(mean,covariance,bounds,bandwidth=.25,epsrel=2e-10):
    """Independent positive conditional integral with shifted tail exponent.

    x=mu_x+L00*u; y=mu_y+L10*u+L11*v. Its Jacobian cancels
    sqrt(det(C_eff)), leaving exp[-(u²+v²)²/(4*pi)]/pi².
    Cartesian breakpoints resolve the mode and conditional zero crossings.
    Extremely tiny mass may underflow; this is explicitly recorded, not a
    claim that the distribution has compact support.
    """
    mean=np.asarray(mean,np.float64);bounds=np.asarray(bounds,np.float64)
    c=effective_covariance(covariance,bandwidth);L=np.linalg.cholesky(c)
    qmin,closest=rectangle_minimum(mean,c,bounds);shift=-qmin*qmin/(4*PI)
    area=float(np.prod(bounds[2:]-bounds[:2]))
    upper_log=shift+math.log(area)-2*math.log(PI)-math.log(np.linalg.det(L))
    if upper_log<-744:
        return dict(mass=0.,log_mass=None,estimated_absolute_error=0.,underflow_bound=True,
                    log_mass_upper_bound=upper_log,warnings=[])
    lo=(bounds[0]-mean[0])/L[0,0];hi=(bounds[2]-mean[0])/L[0,0]
    points=[0.,(closest[0]-mean[0])/L[0,0]]
    if L[1,0]!=0:points.extend((bounds[[1,3]]-mean[1])/L[1,0])
    points=sorted({float(p) for p in points if lo<p<hi})
    inner_errors=[]
    def integrand(u):
        vl=(bounds[1]-mean[1]-L[1,0]*u)/L[1,1]
        vh=(bounds[3]-mean[1]-L[1,0]*u)/L[1,1]
        def vertical(v):
            exponent=-(u*u+v*v)**2/(4*PI)-shift
            return math.exp(min(exponent,700.))
        vpoints=[0.] if vl<0<vh else None
        value,error=quad(vertical,vl,vh,epsabs=1e-12,epsrel=epsrel,points=vpoints,limit=200)
        inner_errors.append(error);return value
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',IntegrationWarning)
        value,error=quad(integrand,lo,hi,epsabs=1e-12,epsrel=epsrel,points=points,limit=250)
    scale=math.exp(shift)/PI**2
    return dict(mass=max(value*scale,0.),log_mass=math.log(value/PI**2)+shift if value>0 else None,
        estimated_absolute_error=(error+(hi-lo)*max(inner_errors,default=0.))*scale,
        underflow_bound=False,log_mass_upper_bound=upper_log,warnings=[str(w.message) for w in caught])


def cartesian_moments():
    """Numerical full-plane moments on [-8,8]^2; omitted radial tail <1e-140."""
    result={}
    for name,powers in dict(mass=(0,0),mean_x=(1,0),mean_y=(0,1),cov_xx=(2,0),cov_xy=(1,1),cov_yy=(0,2),fourth_x=(4,0)).items():
        def outer(x):
            return quad(lambda y:x**powers[0]*y**powers[1]*math.exp(-(x*x+y*y)**2/(4*PI))/PI**2,
                        -8,8,epsabs=2e-12,epsrel=2e-11,points=[0.],limit=150)[0]
        value,error=quad(outer,-8,8,epsabs=2e-12,epsrel=2e-11,points=[0.],limit=150)
        result[name]=dict(value=value,estimated_error=error)
    expected=dict(mass=1.,mean_x=0.,mean_y=0.,cov_xx=1.,cov_xy=0.,cov_yy=1.,fourth_x=3*PI/4)
    result['maximum_identity_error']=max(abs(result[k]['value']-v) for k,v in expected.items())
    return result


def rotated_covariance(eigenvalues,angle):
    t=np.deg2rad(angle);rotation=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    return rotation@np.diag(eigenvalues)@rotation.T


def native_grid(stride):
    return np.array([[x-.5,y-.5,x+stride-.5,y+stride-.5]
                     for y in range(0,144,stride) for x in range(0,192,stride)],float)/8


def subdivide(bounds,n):
    xs=np.linspace(bounds[0],bounds[2],n+1);ys=np.linspace(bounds[1],bounds[3],n+1)
    return np.array([[xs[x],ys[y],xs[x+1],ys[y+1]] for y in range(n) for x in range(n)])


def cases():
    """Fixed synthetic coverage, independent of any experimental outcome."""
    families=[('point',np.zeros((2,2)),.25),('point_h05',np.zeros((2,2)),.5),
        ('isotropic',np.eye(2)*.1,.25),('ellipse45',rotated_covariance((.02,2),45),.25),
        ('thin15',rotated_covariance((0,64),15),.25),('thin45',rotated_covariance((0,256),45),.25),
        ('thin80_extreme',rotated_covariance((0,16384),80),.25),
        ('broad',np.diag([256.,4096.]),.25)]
    means=[('middle',[.4375,.4375]),('edge',[-.0625,.4375]),('corner',[-.0625,-.0625]),
           ('outside',[1.4375,.4375]),('far',[-5.,-5.])]
    result=[]
    for name,c,h in families:
        for placement,mean in means:
            for side in (1.,2.,3.):
                result.append(dict(name=f'{name}_{placement}_side{side:g}',mean=np.array(mean),cov=c,h=h,
                                   bounds=np.array([-.0625,-.0625,side-.0625,side-.0625])))
    # Near-floor FOV cases stress relative/normalized accuracy, not just raw absolutes.
    grids=[('inside_point',np.array([10.4375,7.4375]),np.zeros((2,2)),.25),
           ('tail_fov',np.array([-1.25,8.]),np.zeros((2,2)),.25),
           ('underflow_fov',np.array([-128.,8.]),np.zeros((2,2)),.25),
           ('thin_extreme',np.array([12.,9.]),rotated_covariance((0,16384),45),.25)]
    return result,grids


def normalize(mass,bounds):
    area=np.prod(bounds[:,2:]-bounds[:,:2],axis=1)
    corrected=np.asarray(mass,float)+1e-30*area
    return corrected/corrected.sum()


def gradient_reference(mean,cov,bounds,h,step=2e-5):
    result=[]
    for axis in range(2):
        delta=np.eye(2)[axis]*step
        result.append((adaptive_rectangle(mean+delta,cov,bounds,h)['mass']-adaptive_rectangle(mean-delta,cov,bounds,h)['mass'])/(2*step))
    for matrix in (np.array([[1.,0],[0,0]]),np.array([[0.,1],[1.,0]]),np.array([[0.,0],[0,1.]])):
        result.append((adaptive_rectangle(mean,cov+step*matrix,bounds,h)['mass']-adaptive_rectangle(mean,cov-step*matrix,bounds,h)['mass'])/(2*step))
    result.append((adaptive_rectangle(mean,cov,bounds,h+step)['mass']-adaptive_rectangle(mean,cov,bounds,h-step)['mass'])/(2*step))
    return np.asarray(result)


def audit(out,module,order,device):
    if out.exists():raise ValueError('refuse overwrite numerical run, including failures')
    import torch
    model=importlib.import_module(module);operator=model.quartic_cell_mass
    out.mkdir(parents=True);started=time.time();torch.set_num_threads(4)
    source_paths=[Path(__file__).resolve(),ROOT/'tests/test_round15_quartic_numerical_audit.py',Path(model.__file__).resolve()]
    source_hashes={str(p):sha(p) for p in source_paths}
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in source_paths:
        with (snapshot/path.name).open('xb') as f:f.write(path.read_bytes())
    write(out/'started.json',dict(source_hashes=source_hashes,created_unix=started,module=module,order=order,device=device,
        probability_tolerance=PROBABILITY_TOLERANCE,raw_absolute_tolerance=RAW_ABSOLUTE_TOLERANCE,
        split_absolute_tolerance=SPLIT_ABSOLUTE_TOLERANCE,gradient_tolerance=GRADIENT_TOLERANCE,
        source_can_iterate_before_effects=True,labels_read=False,real_observations_read=False))
    def actual(mean,cov,bounds,h):
        with torch.no_grad():
            return operator(torch.tensor(np.asarray(mean)[None],device=device,dtype=torch.float64),
                torch.tensor(np.asarray(cov)[None],device=device,dtype=torch.float64),
                torch.tensor(np.asarray(bounds).reshape(-1,4),device=device,dtype=torch.float64),
                bandwidth=h,quadrature_order=order).detach().cpu().numpy()[0]
    moments=cartesian_moments();records=[];splits=[];grid_records=[];failures=[];warning_count=0
    if moments['maximum_identity_error']>1e-10:failures.append('cartesian normalization/covariance')
    individual,grids=cases()
    for case in individual:
        ref=adaptive_rectangle(case['mean'],case['cov'],case['bounds'],case['h']);warning_count+=len(ref['warnings'])
        value=float(actual(case['mean'],case['cov'],case['bounds'],case['h'])[0]);error=abs(value-ref['mass'])
        row=dict(name=case['name'],mean=case['mean'].tolist(),covariance=case['cov'].tolist(),h=case['h'],
            bounds=case['bounds'].tolist(),reference=ref,actual=value,absolute_error=error,
            passed=bool(np.isfinite(value) and value>=0 and error<=RAW_ABSOLUTE_TOLERANCE))
        records.append(row)
        if not row['passed']:failures.append(case['name'])
        for split in (2,3):
            children=subdivide(case['bounds'],split);children_mass=actual(case['mean'],case['cov'],children,case['h'])
            defect=abs(float(children_mass.sum())-value)
            splitrow=dict(name=case['name'],children=split**2,absolute_defect=defect,
                passed=bool(np.isfinite(children_mass).all() and np.all(children_mass>=0) and defect<=SPLIT_ABSOLUTE_TOLERANCE))
            splits.append(splitrow)
            if not splitrow['passed']:failures.append(case['name']+f'_split{split}')
    for name,mean,cov,h in grids:
        for stride in (8,16,24):
            bounds=native_grid(stride);refs=[adaptive_rectangle(mean,cov,b,h) for b in bounds]
            warning_count+=sum(len(r['warnings']) for r in refs);mass=np.array([r['mass'] for r in refs])
            predicted=actual(mean,cov,bounds,h);p=normalize(predicted,bounds);expected=normalize(mass,bounds)
            error=float(abs(p-expected).max());absolute=float(abs(predicted-mass).max())
            row=dict(name=name,stride=stride,mean=mean.tolist(),covariance=cov.tolist(),h=h,
                raw_fov_reference=float(mass.sum()),raw_fov_actual=float(predicted.sum()),
                floor_total=float(1e-30*np.prod(bounds[:,2:]-bounds[:,:2],axis=1).sum()),
                maximum_probability_error=error,maximum_raw_error=absolute,
                underflow_bound_cells=sum(r['underflow_bound'] for r in refs),
                passed=bool(np.isfinite(predicted).all() and np.all(predicted>=0) and error<=PROBABILITY_TOLERANCE and absolute<=RAW_ABSOLUTE_TOLERANCE))
            grid_records.append(row)
            np.savez_compressed(out/(name+f'_g{stride}.npz'),reference_mass=mass,actual_mass=predicted,
                                reference_probability=expected,actual_probability=p,bounds=bounds)
            if not row['passed']:failures.append(name+f'_g{stride}')
            print('QUARTIC_GRID',name,stride,'error',error,flush=True)
    # Physical density is smooth even when mu hits a cell edge/corner; an angular
    # topology branch must not silently introduce NaN or wrong derivatives there.
    gradient_records=[];gb=np.array([-.0625,-.0625,.9375,.9375]);gh=.5
    for label,gm,gc in [('generic',np.array([.23,-.19]),np.array([[.7,.18],[.18,.4]])),
                       ('edge',np.array([-.0625,.4]),np.array([[.7,.18],[.18,.4]])),
                       ('corner',np.array([-.0625,-.0625]),np.array([[.7,.18],[.18,.4]])),
                       ('thin',np.array([.4,.4]),rotated_covariance((.02,64.),45))]:
        gradient=gradient_reference(gm,gc,gb,gh)
        tm=torch.tensor(gm[None],dtype=torch.float64,device=device,requires_grad=True)
        tc=torch.tensor(gc[None],dtype=torch.float64,device=device,requires_grad=True)
        th=torch.tensor(gh,dtype=torch.float64,device=device,requires_grad=True)
        try:
            val=operator(tm,tc,torch.tensor(gb[None],dtype=torch.float64,device=device),bandwidth=th,quadrature_order=order).sum()
            dm,dc,dh=torch.autograd.grad(val,(tm,tc,th))
            actual_gradient=np.r_[dm.detach().cpu().numpy().ravel(),
                float(dc[0,0,0]),float(dc[0,0,1]+dc[0,1,0]),float(dc[0,1,1]),float(dh)]
            finite=bool(np.isfinite(actual_gradient).all())
            gradient_error=float(abs(actual_gradient-gradient).max()) if finite else None
            gradient_record=dict(name=label,reference=gradient.tolist(),actual=actual_gradient.tolist() if finite else None,
                maximum_error=gradient_error,passed=bool(finite and gradient_error<=GRADIENT_TOLERANCE))
        except Exception as exc:
            gradient_record=dict(name=label,reference=gradient.tolist(),passed=False,error=str(exc))
        gradient_records.append(gradient_record)
        if not gradient_record['passed']:failures.append(label+' autograd versus adaptive finite difference')
    write(out/'individual.json',records);write(out/'split.json',splits);write(out/'grids.json',grid_records)
    write(out/'moments.json',moments);write(out/'gradients.json',gradient_records)
    result=dict(passed=not failures,failures=failures,individual_rectangles=len(records),split_checks=len(splits),
        grid_conditions=len(grid_records),maximum_raw_error=max(r['absolute_error'] for r in records),
        maximum_grid_probability_error=max(r['maximum_probability_error'] for r in grid_records),
        maximum_split_absolute_defect=max(r['absolute_defect'] for r in splits),reference_integration_warnings=warning_count,
        source_hashes=source_hashes,order=order,device=device,seconds=time.time()-started,
        real_observations_read=False,labels_read=False,validation_or_test_read=False,
        scientific_gate_passed=False,scope='Finite declared synthetic domain only; no universal guarantee for arbitrary ill-conditioned covariance, tails, gradients, or real-task effectiveness.')
    write(out/'verification.json',result)
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'completion.json',dict(files=files,finished_unix=time.time(),passed=result['passed']))
    print(json.dumps(result,indent=2),flush=True)


def freeze(out,module,report):
    if out.exists():raise ValueError('refuse final freeze overwrite')
    model=importlib.import_module(module);verification=json.loads((report/'verification.json').read_text())
    if not verification['passed']:raise ValueError('cannot final-freeze a failed numerical preflight')
    sources={str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'tests/test_round15_quartic_numerical_audit.py',Path(model.__file__).resolve())}
    if sources!=verification['source_hashes']:raise ValueError('source changed since final numerical run')
    out.mkdir(parents=True)
    for path in sources:
        with (out/Path(path).name).open('xb') as f:f.write(Path(path).read_bytes())
    lock=dict(created_unix=time.time(),source_hashes=sources,report=str(report.resolve()),
        report_sha256=sha(report/'verification.json'),completion_sha256=sha(report/'completion.json'),
        labels_read=False,real_observations_read=False,scientific_gate_passed=False)
    write(out/'lock.json',lock);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FINAL_NUMERICAL_FREEZE',sha(out/'lock.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('audit','freeze'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model',default='models.round15_quartic_pe')
    p.add_argument('--order',type=int,default=32);p.add_argument('--device',default='cpu');p.add_argument('--report',type=Path)
    args=p.parse_args()
    if args.stage=='audit':
        try:audit(args.out,args.model,args.order,args.device)
        except Exception as exc:
            if args.out.exists() and not (args.out/'verification.json').exists() and not (args.out/'unhandled_failure.json').exists():
                write(args.out/'unhandled_failure.json',dict(passed=False,error=repr(exc),traceback=traceback.format_exc(),
                    finished_unix=time.time(),source_sha256=sha(__file__),labels_read=False,real_observations_read=False))
            raise
    else:freeze(args.out,args.model,args.report)


if __name__=='__main__':main()
