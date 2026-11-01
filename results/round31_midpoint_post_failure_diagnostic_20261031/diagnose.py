"""Post-failure metadata-only R31 diagnosis, not a reference integration.

Only authenticated prepare and outer_breaks function ASTs are executed.
No NPZ is decoded and no integrator/producer is imported or called.
"""
import ast
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.special import log_ndtr

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def array_sha(value):
    value=np.ascontiguousarray(value)
    header=json.dumps(dict(dtype=str(value.dtype),shape=list(value.shape)),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(header+b'\n'+value.tobytes()).hexdigest()
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value
def number(value):
    value=float(value);n,d=value.as_integer_ratio()
    return dict(value=value,repr=repr(value),hex=value.hex(),numerator=n,denominator=d)


def main():
    report_path=OUT/'diagnostic.json'
    if report_path.exists():raise RuntimeError('preserve original diagnostic')
    started=time.time()
    receipt_path=ROOT/'results/round31_log_conditional_root_acceptance_20261031.json'
    receipt=read(receipt_path)
    if receipt['numerical_preflight_passed'] is not False:raise ValueError('accepted R31 failure required')
    bindings={str(receipt_path.relative_to(ROOT)):sha(receipt_path)}
    for name,digest in receipt['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('accepted metadata changed: '+name)
        bindings[name]=digest
    for name,digest in receipt['source_hashes'].items():
        for path in (ROOT/name,ROOT/'results/round31_log_conditional_preflight/source_snapshot'/name,
                     ROOT/'results/round31_log_conditional_independent_audit/source_snapshot'/name):
            if sha(path)!=digest:raise ValueError('accepted source changed: '+str(path))
            bindings[str(path.relative_to(ROOT))]=digest
    run=ROOT/'results/round31_log_conditional_preflight'
    lock=read(run/'lock.json');specs=read(run/'query_specs.json');rows=read(run/'forward/records.json')
    spec=next(s for s in specs if s['id']=='07_radial_noncentral')
    failed=[r for r in rows if r['id']==spec['id']]
    if len(failed)!=2 or any(r['passed'] or r['work']['pilot_depth_evaluations'] for r in failed):
        raise ValueError('expected two pre-pilot failures')
    source_path=ROOT/'experiments/round31_log_conditional_reference.py'
    text=source_path.read_text();tree=ast.parse(text)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('prepare','outer_breaks')]
    if {n.name for n in functions}!={'prepare','outer_breaks'}:raise ValueError('function extraction differs')
    namespace=dict(np=np,math=math,log_ndtr=log_ndtr)
    module=ast.Module(body=functions,type_ignores=[])
    exec(compile(module,str(source_path),'exec'),namespace)
    def forbidden(*args,**kwargs):raise RuntimeError('diagnostic may not decode NPZ')
    np.load=forbidden
    fixture=spec['fixture'];inputs={n:np.asarray(fixture[n.removeprefix('input__')],np.float64) for n in lock['input_keys']} if 'input_keys' in lock else {
        'input__'+n:np.asarray(fixture[n],np.float64) for n in ('mu','factor','bounds','content_bias')}
    hashes={n:array_sha(v) for n,v in inputs.items()}
    if hashes!=spec['array_sha256']:raise ValueError('descriptor canonical input hashes differ')
    if sha(spec['source_archive_path'])!=spec['source_archive_sha256']:
        raise ValueError('source archive bytes differ')
    bindings[spec['source_archive_path']]=spec['source_archive_sha256']
    policy=dict(lock['policy'],**spec['policy_overrides']);rp=lock['reference_policy']
    prepared=namespace['prepare'](inputs['input__mu'],inputs['input__factor'],policy['h'])
    bounds=inputs['input__bounds'];fov=np.r_[bounds[:,:2].min(0),bounds[:,2:].max(0)];boxes=np.vstack((bounds,fov))
    m,sigma=prepared['normalized_depth_mean'],prepared['normalized_depth_std']
    lower=max(-rp['depth_cutoff'],-m/sigma);upper=rp['depth_cutoff']
    points=namespace['outer_breaks'](prepared,boxes,lower,upper,rp)
    edges=np.asarray([lower]+points+[upper]);midpoints=(edges[:-1]+edges[1:])/2
    # Trace identical frozen root arithmetic to associate each breakpoint with
    # its axis/edge/geometry-level origin. This is not a quadrature evaluation.
    origins=[]
    for axis,sd in ((0,prepared['sx']),(1,prepared['marginal_sy'])):
        for edge in np.unique(boxes[:,[axis,axis+2]]):
            mu,beta=float(prepared['xy_mean'][axis]),float(prepared['beta'][axis])
            scale=max(1.,abs(float(edge)),abs(mu),abs(beta),sd)
            denominator=sigma*(edge/scale)-beta/scale
            if denominator==0:continue
            for c in rp['outer_geometry_breaks']:
                root=(mu/scale-m*(edge/scale)+c*(sd/scale))/denominator
                if math.isfinite(root) and lower<root<upper:
                    origins.append(dict(axis=axis,edge=number(edge),geometry_level=c,sd=number(sd),
                        arithmetic_scale=number(scale),denominator=number(denominator),root=number(root)))
    if sorted({o['root']['value'] for o in origins})!=points:raise ValueError('root origin trace differs')
    bad=[]
    for index,(a,b,midpoint) in enumerate(zip(edges[:-1],edges[1:],midpoints)):
        if a<midpoint<b:continue
        gap=Fraction(float(b))-Fraction(float(a))
        bad.append(dict(interval_index=index,left=number(a),right=number(b),legacy_midpoint=number(midpoint),
            stable_formula_midpoint=number(float(a)+(float(b)-float(a))/2),
            nextafter_left_toward_right=number(np.nextafter(a,b)),
            nextafter_right_toward_left=number(np.nextafter(b,a)),
            no_representable_float_strictly_between=bool(np.nextafter(a,b)==b),
            exact_gap=dict(numerator=gap.numerator,denominator=gap.denominator),
            left_origins=[o for o in origins if o['root']['value']==a],right_origins=[o for o in origins if o['root']['value']==b]))
    report=dict(kind='post_execution_metadata_and_breakpoint_diagnostic_not_preregistration',
        started_unix=started,finished_unix=time.time(),root_acceptance_sha256=sha(receipt_path),bindings=bindings,
        code_sha256=sha(__file__),source_hash=sha(source_path),
        executed_frozen_functions=['prepare','outer_breaks'],executed_diagnostic_arithmetic=['same root-origin trace','float midpoint/nextafter/hex/exact rational gap'],
        integration_calls=0,production_model_calls=0,npz_decodes=0,real_inputs_read=False,labels_read=False,
        complete_input_spec=spec,canonical_input_hashes=hashes,effective_policy=policy,reference_policy=rp,
        prepared=serial(prepared),lower=number(lower),upper=number(upper),breakpoint_count=len(points),
        all_breakpoints=[number(v) for v in points],all_initial_edges=[number(v) for v in edges],
        all_legacy_midpoints=[number(v) for v in midpoints],bad_intervals=bad,root_origins=origins,
        accepted_failed_records=failed,proposal_only_R32_not_executed=True,
        proposed_rule='Keep endpoints. Greedily retain an interior cut only when its midpoint with the last retained edge is strict; otherwise omit that cut as a partition hint. Pop trailing interior cuts until the last-to-upper midpoint is strict; if even the whole endpoint interval has no strict midpoint, fail. Use exactly retained edges for both pilots and quad points. No integration domain is removed.',
        scope='Diagnoses partition representation only; does not revise R31, improve a numerical gate or establish R32 accuracy.')
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(report_path.relative_to(ROOT)),sha256=sha(report_path),
        breakpoint_count=len(points),bad_intervals=bad),indent=2))


if __name__=='__main__':main()
