"""Prospectively frozen minimum-X geometry count; no probability integration."""
from __future__ import annotations
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import scipy
import scipy.special  # Match frozen numerical libraries; no special function is called.
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'results/round35_log_real_reference'
REF = ROOT/'experiments/round36_component_round_reference.py'
QUAD = ROOT/'experiments/round36_component_round_quadrature.py'
DESIGN = ROOT/'docs/round36_real_initial_cost_lowerbound_design.md'
FIELDS = ('input__mu','input__factor','input__bounds','input__content_bias')
THREADS = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')
MAX_SECONDS = 300.
NODE_LIMIT = 250_000_000
SCOPE = dict(labels_read=False, reference_integrations=0, model_integrations=0,
             Gaussian_pdf_evaluations=0, Gaussian_cdf_evaluations=0,
             quadrature_weighted_sums=0, independent_real_reference=False,
             general_numerical_accuracy_established=False, efficacy_tested=False,
             candidate_selected=False, scientific_gate_passed=False, goal_complete=False)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def append(path, value):
    with Path(path).open('a') as stream: stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
def array_sha(value):
    value=np.ascontiguousarray(value)
    header=json.dumps(dict(dtype=str(value.dtype),shape=list(value.shape)),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(header+b'\n'+value.tobytes()).hexdigest()


def runtime():
    builds={}
    for name, function in (('numpy',np.show_config),('scipy',scipy.show_config)):
        stream=io.StringIO()
        with redirect_stdout(stream):function()
        builds[name+'_build_sha256']=hashlib.sha256(stream.getvalue().encode()).hexdigest()
    value=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),threads={k:os.environ.get(k) for k in THREADS},threadpools=threadpool_info(),**builds)
    if any(value['threads'][k]!='4' for k in THREADS) or any(p['num_threads']!=4 for p in value['threadpools']):
        raise ValueError('frozen four-thread numerical runtime required')
    return value


def evidence():
    bindings={}
    for name in ('results/round35_log_real_root_acceptance_20261101.json',
                 'results/round36_component_round_root_acceptance_20261101.json'):
        receipt_path=ROOT/name;receipt=read(receipt_path);bindings[str(receipt_path)]=sha(receipt_path)
        if not receipt['delivery_integrity_passed'] or not receipt['all_current_and_dual_snapshot_sources_verified']:
            raise ValueError('accepted delivery is required')
        if 'round36_' in name and not receipt['numerical_preflight_passed']:
            raise ValueError('accepted R36 numerical pass required')
        for field in ('files','source_hashes'):
            for relative,digest in receipt[field].items():
                path=ROOT/relative
                if sha(path)!=digest:raise ValueError('accepted provenance changed: '+str(path))
                bindings[str(path)]=digest
    records=read(RUN/'prepared/records.json');specs=read(RUN/'query_specs.json')
    if len(records)!=72 or len(specs)!=72 or [x['id'] for x in records]!=[x['id'] for x in specs]:
        raise ValueError('fixed ordered all72 population required')
    if len({x['id'] for x in records})!=72:raise ValueError('duplicate input id')
    identity=('id','scene','pair','pair_name','query_size','query_index','query_bounds')
    for row,spec in zip(records,specs):
        if not row['passed'] or any(row[k]!=spec[k] for k in identity) or set(row['array_sha256'])!=set(FIELDS):
            raise ValueError('accepted input identity differs')
        path=RUN/row['archive_path']
        if not path.resolve().is_relative_to((RUN/'prepared/arrays').resolve()):raise ValueError('input path outside allowlist')
        if sha(path)!=row['archive_sha256']:raise ValueError('prepared archive hash differs')
        bindings[str(path)]=row['archive_sha256']
    return bindings,records


def freeze():
    if (OUT/'lock.json').exists():raise ValueError('refusing diagnostic refreeze')
    bindings,records=evidence()
    local=[Path(__file__).resolve(),DESIGN,OUT/'test_minimum_cost.py',REF,QUAD]
    local_hashes={str(p):sha(p) for p in local}
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    for path,digest in local_hashes.items():
        dest=snapshot/Path(path).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
        if sha(dest)!=digest:raise ValueError('snapshot differs')
    oldlock=read(RUN/'lock.json')
    accepted=read(ROOT/'results/round36_component_round_preflight/lock.json')
    if accepted['policy']!=oldlock['policy']:raise ValueError('accepted geometry law changed')
    refpolicy=accepted['reference_policy']
    if (refpolicy['max_call_cell_node_evaluations']!=NODE_LIMIT or refpolicy['standard_x_panel_width']!=2.
            or refpolicy['orders']!=[32,48]):raise ValueError('frozen necessary-cost contract changed')
    value=dict(created_unix=time.time(),bindings=bindings,local_source_hashes=local_hashes,
        runtime=runtime(),records=records,case_ids=[r['id'] for r in records],case_count=72,
        orders=[32,48],policy=oldlock['policy'],reference_policy=refpolicy,
        max_seconds=MAX_SECONDS,per_call_node_limit=NODE_LIMIT,real_inputs_read=False,
        command=[sys.executable,str(Path(__file__).resolve()),'run'],**SCOPE)
    write(OUT/'lock.json',value);(OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')


def verify():
    lock=read(OUT/'lock.json')
    if sha(OUT/'lock.json')!=(OUT/'lock.sha256').read_text().strip():raise ValueError('lock hash differs')
    for field in ('bindings','local_source_hashes'):
        for path,digest in lock[field].items():
            if sha(path)!=digest:raise ValueError('frozen file changed: '+path)
    for path,digest in lock['local_source_hashes'].items():
        if sha(OUT/'source_snapshot'/Path(path).relative_to(ROOT))!=digest:raise ValueError('snapshot changed')
    if runtime()!=lock['runtime']:raise ValueError('frozen runtime changed')
    if lock['case_count']!=72 or lock['orders']!=[32,48] or lock['max_seconds']!=300. or lock['per_call_node_limit']!=250_000_000:
        raise ValueError('frozen diagnostic limits differ')
    return lock


def pure_geometry_functions():
    """AST extract only the two frozen geometry functions and GK abscissae."""
    tree=ast.parse(REF.read_text());names={'outer_breaks','coalesce_depth_edges'}
    definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {n.name for n in definitions}!=names:raise ValueError('geometry source whitelist differs')
    namespace={'np':np,'math':math}
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(REF),'exec'),namespace)
    qt=ast.parse(QUAD.read_text())
    assignment=next(n for n in qt.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='X' for t in n.targets))
    nodes=np.asarray(ast.literal_eval(assignment.value),np.float64)
    if nodes.shape!=(21,) or not np.isfinite(nodes).all():raise ValueError('frozen GK21 abscissae differ')
    return namespace['outer_breaks'],namespace['coalesce_depth_edges'],nodes


def geometry(mu,factor,h):
    """Same RMS/SVD geometry as REF36; intentionally no front probability."""
    if (mu.dtype!=np.float64 or mu.shape!=(3,) or factor.dtype!=np.float64 or factor.shape!=(3,3)
            or not np.isfinite(mu).all() or not np.isfinite(factor).all()):raise ValueError('finite float64 geometry required')
    depth_scale=max(abs(float(mu[2])),float(np.max(np.abs(factor[2]))))
    stochastic=bool(np.any(factor[2]!=0))
    if depth_scale==0:return dict(branch='zero_depth',structural_zero=True)
    if not stochastic and mu[2]<0:return dict(branch='nonpositive_constant_depth',structural_zero=True)
    rr=math.hypot(*map(float,np.r_[mu[2]/depth_scale,factor[2]/depth_scale]))
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        nm=(mu/depth_scale)/rr;nf=(factor/depth_scale)/rr
    rms=depth_scale*rr
    if not np.isfinite(nm).all() or not np.isfinite(nf).all() or not math.isfinite(rms):raise FloatingPointError('unrepresentable RMS geometry')
    m=float(nm[2]);sigma=math.hypot(*map(float,nf[2]))
    if stochastic and sigma==0:raise FloatingPointError('stochastic depth lost in RMS normalization')
    if not stochastic and m<=0:return dict(branch='nonpositive_constant_depth',structural_zero=True)
    if stochastic:
        direction=nf[2]/sigma
        _,_,vh=np.linalg.svd(nf[2:3],full_matrices=True)
        base=nf[:2]@vh[1:].T;beta=nf[:2]@direction
    else:base,beta=nf[:2],np.zeros(2)
    cf=np.column_stack((base,h*np.eye(2)))
    sx=math.hypot(*map(float,cf[0]));marginal_sy=math.hypot(*map(float,cf[1]))
    dx=cf[0]/sx;k=float(np.dot(cf[1],dx));sy=math.hypot(*map(float,cf[1]-k*dx))
    if not all(math.isfinite(v) and v>0 for v in (sx,marginal_sy,sy)) or not np.isfinite(beta).all() or not math.isfinite(k):
        raise FloatingPointError('nonfinite conditional Gaussian geometry')
    return dict(branch='stochastic_depth' if stochastic else 'positive_constant_depth',structural_zero=False,
        normalized_depth_mean=m,normalized_depth_std=sigma,xy_mean=nm[:2],beta=beta,
        sx=sx,marginal_sy=marginal_sy,conditional_y_residual_std=sy,conditional_y_coefficient=k)


def minimum_panels(location,z,boxes,sx,width):
    """Frozen pre-Y-root X admission count; no grid or inner nodes allocated."""
    if not math.isfinite(z) or z<=0:raise FloatingPointError('strict positive homogeneous depth required')
    mid=(z*((boxes[:,0]+boxes[:,2])/2)-location[0])/sx
    half=z*((boxes[:,2]-boxes[:,0])/2)/sx
    low,high=mid-half,mid+half
    if (not np.isfinite(mid).all() or not np.isfinite(half).all() or np.any(half<=0)
            or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high<low)):
        raise FloatingPointError('positive full-X interval is not representable')
    regular=high>low
    if np.any((~regular)&(2*half>width)):raise FloatingPointError('rounded interval cannot represent width2 subdivision')
    total=len(boxes)
    for a,b,valid in zip(low,high,regular):
        n=max(0,math.ceil(float(b)/width)-math.floor(float(a)/width)-1) if valid else 0
        if n and np.spacing(max(abs(float(a)),abs(float(b))))>width:
            raise FloatingPointError('global width2 grid spacing is not representable')
        total+=n
    return int(total)


def count_case(bank,policy,reference_policy,deadline,functions):
    arrays={};report=dict(branch=None,completed_required_callbacks=0,required_callbacks=None,
        zero_inner_callbacks=0,minimum_panels_prefix=0,full_initial_geometry_counted=False,
        status='geometry_failure',failure_reason=None,front_probability_not_evaluated=True)
    def check_time():
        if time.perf_counter()>=deadline:raise TimeoutError('fixed 300-second diagnostic deadline')
    try:
        check_time();bounds=bank['input__bounds']
        if bounds.dtype!=np.float64 or bounds.shape!=(432,4) or not np.isfinite(bounds).all() or np.any(bounds[:,2:]<=bounds[:,:2]):
            raise ValueError('fixed432 positive native rectangles required')
        fov=np.r_[bounds[:,:2].min(0),bounds[:,2:].max(0)];boxes=np.vstack((bounds,fov))
        g=geometry(bank['input__mu'],bank['input__factor'],policy['h']);report['branch']=g['branch']
        if g['structural_zero']:
            report.update(status='structural_zero',required_callbacks=0,full_initial_geometry_counted=True)
            return report,arrays
        def count(zeta,constant=False):
            check_time()
            z=1. if constant else g['normalized_depth_mean']+g['normalized_depth_std']*zeta
            if not constant and z<=0:
                if zeta<=threshold:
                    report['zero_inner_callbacks']+=1;report['completed_required_callbacks']+=1;return
                raise FloatingPointError('strict positive interior depth lost by cancellation/underflow')
            location=g['xy_mean'] if constant else g['xy_mean']+g['beta']*zeta
            report['minimum_panels_prefix']+=minimum_panels(location,z,boxes,g['sx'],reference_policy['standard_x_panel_width'])
            report['completed_required_callbacks']+=1
        if g['normalized_depth_std']==0:
            report['required_callbacks']=1;count(0.,constant=True)
        else:
            threshold=-g['normalized_depth_mean']/g['normalized_depth_std']
            lower=max(-reference_policy['depth_cutoff'],threshold);upper=reference_policy['depth_cutoff']
            if lower>=upper:raise FloatingPointError('positive depth event lies outside fixed physical cutoff')
            outer_breaks,coalesce,nodes=functions
            points=outer_breaks(g,boxes,lower,upper,reference_policy)
            metadata={}
            try:edges=coalesce(np.asarray([lower]+points+[upper],np.float64),available=arrays,metadata=metadata)
            finally:
                if 'depth_edge_coalescing' in metadata:report['depth_edge_coalescing']=metadata['depth_edge_coalescing']
            intervals=len(edges)-1;report['initial_interval_count']=intervals
            if intervals>reference_policy['maximum_depth_intervals']:raise RuntimeError('fixed initial depth-interval budget exceeded')
            pilots=(edges[:-1]+edges[1:])/2
            if np.any(pilots<=edges[:-1]) or np.any(pilots>=edges[1:]):raise FloatingPointError('strict pilot midpoint not representable')
            arrays['pilot_zeta']=pilots;report['required_callbacks']=22*intervals
            for zeta in pilots:count(float(zeta))
            for left,right in zip(edges[:-1],edges[1:]):
                c=.5*(float(left)+float(right));h=.5*(float(right)-float(left))
                if not math.isfinite(c) or not math.isfinite(h) or not left<c<right or h<=0:
                    raise FloatingPointError('strict GK21 interval midpoint not representable')
                for x in nodes:count(c+h*float(x))
        report.update(status='initial_lower_bound_only',full_initial_geometry_counted=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='timed_out' if isinstance(error,TimeoutError) else 'geometry_failure',
                      failure_reason=type(error).__name__+': '+str(error))
    return report,arrays


def run():
    lock=verify()
    if (OUT/'started.json').exists():raise ValueError('refusing diagnostic rerun')
    functions=pure_geometry_functions();tick=time.perf_counter();deadline=tick+MAX_SECONDS
    write(OUT/'started.json',dict(started_unix=time.time(),lock_sha256=sha(OUT/'lock.json'),real_inputs_read=False,**SCOPE))
    (OUT/'geometry_arrays').mkdir();records=[];read_any=False;halt=None
    for index,spec in enumerate(lock['records']):
        item={k:spec[k] for k in ('id','scene','pair','pair_name','query_size','query_index','query_bounds')}
        item.update(case_index=index,input_archive_path=str(RUN/spec['archive_path']),input_archive_sha256=spec['archive_sha256'],real_inputs_read=False)
        arrays={};decoded=False
        if halt is not None or time.perf_counter()>=deadline:
            halt=halt or 'fixed_diagnostic_deadline'
            result=dict(status='not_processed',failure_reason=halt,minimum_panels_prefix=0,completed_required_callbacks=0,
                        required_callbacks=None,full_initial_geometry_counted=False,front_probability_not_evaluated=True)
        else:
            try:
                path=RUN/spec['archive_path']
                if sha(path)!=spec['archive_sha256']:raise ValueError('frozen input archive changed')
                bank={}
                with np.load(path,allow_pickle=False) as archive:
                    if set(archive.files)!=set(FIELDS):raise ValueError('prepared fields changed')
                    for key in FIELDS:
                        bank[key]=archive[key].copy();decoded=True;read_any=True
                if {k:array_sha(v) for k,v in bank.items()}!=spec['array_sha256']:raise ValueError('canonical input identity changed')
                result,arrays=count_case(bank,lock['policy'],lock['reference_policy'],deadline,functions)
                if result['status']=='timed_out':halt='fixed_diagnostic_deadline'
            except (Exception,KeyboardInterrupt) as error:
                result=dict(status='input_failure',failure_reason=type(error).__name__+': '+str(error),minimum_panels_prefix=0,
                    completed_required_callbacks=0,required_callbacks=None,full_initial_geometry_counted=False,front_probability_not_evaluated=True)
                halt='input_delivery_failure'
        item.update(result,real_inputs_read=decoded,**SCOPE)
        item['orders']={str(order):dict(minimum_x_nodes_prefix=order*int(item['minimum_panels_prefix']),
            guaranteed_initial_node_budget_failure=order*int(item['minimum_panels_prefix'])>NODE_LIMIT,
            absence_of_counterexample_is_unknown=True) for order in (32,48)}
        if arrays:
            path=OUT/'geometry_arrays'/(spec['id']+'.npz');np.savez_compressed(path,**arrays)
            item.update(geometry_archive_path=str(path.relative_to(OUT)),geometry_archive_sha256=sha(path),
                        geometry_array_sha256={k:array_sha(v) for k,v in arrays.items()})
        records.append(item);append(OUT/'progress.jsonl',item)
    write(OUT/'records.json',records)
    phase_seconds=time.perf_counter()-tick
    verify()
    counterexamples=[dict(id=r['id'],order=int(order),minimum_x_nodes_prefix=x['minimum_x_nodes_prefix'])
        for r in records for order,x in r['orders'].items() if x['guaranteed_initial_node_budget_failure']]
    report=dict(finished_unix=time.time(),phase_wall_seconds=phase_seconds,case_count=len(records),scheduled_order_count=144,
        within_diagnostic_wall_budget=phase_seconds<=MAX_SECONDS,
        phase_timing_scope='before started.json through records.json write; final source checks and inventory separate',
        all72_accounted_for=len(records)==72,real_inputs_read=read_any,
        full_geometry_case_count=sum(r['full_initial_geometry_counted'] for r in records),
        status_counts={s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})},
        guaranteed_node_counterexamples=counterexamples,
        conclusion='necessary_initial_node_budget_counterexample_found' if counterexamples else 'unknown_no_node_counterexample_established',
        no_reference_or_scientific_pass_claim=True,**SCOPE)
    write(OUT/'diagnostic.json',report)
    inventory={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()
               and p.name not in ('completion.json','execution.log') and 'development' not in p.parts}
    delivery_passed=not any(r['status']=='input_failure' for r in records)
    write(OUT/'completion.json',dict(passed=delivery_passed,delivery_integrity_passed=delivery_passed,
        all72_accounted_for=True,real_inputs_read=read_any,diagnostic_sha256=sha(OUT/'diagnostic.json'),
        finished_unix=time.time(),files=inventory,**SCOPE))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'))
    getattr(sys.modules[__name__],parser.parse_args().stage)()
