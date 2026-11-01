"""Frozen synthetic-only actual-V precision preparation; never task training."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'docs/round17_value_interface_preflight_protocol.md',
    'models/__init__.py', 'experiments/__init__.py',
    'models/gaussian_quadratic_pe.py', 'models/round7_depth_attention.py',
    'models/round12_depth_attention.py', 'models/round17_trace_pe.py',
    'models/round17_depth_attention.py', 'tests/test_round17_depth_attention.py',
    'experiments/round17_value_interface_preflight.py',
)
LIMITS = dict(same_content_probability=1e-12, high_precision_probability=1e-5,
              high_precision_relative_depth=1e-5, gradient_relative_l2=1e-3,
              value_product_scaled_absolute=2e-5)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode()+a.tobytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def runtime():
    return dict(device='cpu', torch_version=torch.__version__, numpy_version=np.__version__,
                cpu_threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest())


def fixture(batch, queries, keys):
    generator = torch.Generator().manual_seed(171710)
    q = F.normalize(torch.randn(batch, queries, 128, generator=generator), dim=-1)
    k = F.normalize(torch.randn(batch, keys, 128, generator=generator), dim=-1)
    qindex, kindex = torch.arange(queries), torch.arange(keys)
    qm = torch.stack((qindex % 24, qindex // 24), -1).double()
    km = torch.stack((kindex % 24, kindex // 24), -1).double()
    geometry = (-.5*((qm[:, None]-km[None])**2).sum(-1)/1.5**2).repeat(batch, 1, 1)
    index = torch.arange(queries, dtype=torch.float64)
    c, s = ((index+1)*.371).cos(), ((index+1)*.371).sin()
    a, d = .01+(index % 17)/9, .02+(index % 11)/13
    covariance = torch.empty(queries, 2, 2, dtype=torch.float64)
    covariance[:, 0, 0] = a*c*c+d*s*s
    covariance[:, 1, 1] = a*s*s+d*c*c
    covariance[:, 0, 1] = covariance[:, 1, 0] = (a-d)*c*s
    covariance[0] = 0
    return dict(q=q, k=k, q_depth=1+4*torch.rand(batch, queries, generator=generator),
                k_value_depth=1+5*torch.rand(batch, keys, generator=generator),
                geometry_logits=geometry, support_covariance=covariance.float().repeat(batch, 1, 1, 1))


def reference_tau(covariance, mode, bandwidth, temperature):
    covariance = np.asarray(covariance, dtype=np.float64)
    if mode == 'constant':
        return np.full(covariance.shape[:-2], temperature, dtype=np.float32)
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    h2 = bandwidth*bandwidth
    scale = np.maximum(diagonal.max(-1), h2)
    relative_h2 = h2/scale
    return (temperature*(relative_h2/(relative_h2+(diagonal/scale[..., None]).sum(-1)))).astype(np.float32)


def np_softmax(x):
    p = np.exp(x-x.max(-1, keepdims=True))
    return p/p.sum(-1, keepdims=True)


def np_forward(data, parameters, tau, intervention):
    """Independent double affine/normalize/GELU/value/head forward."""
    d = {n:t.detach().numpy().astype(np.float64) for n, t in data.items()}
    p = {n:t.detach().numpy().astype(np.float64) for n, t in parameters.items()}
    def linear(x, name, bias=True):
        value = x@p[name+'.weight'].T
        return value+p[name+'.bias'] if bias else value
    def gelu(x):
        return .5*x*(1+np.vectorize(math.erf, otypes=[np.float64])(x/math.sqrt(2)))
    def normalize(x):
        return x/np.maximum(np.sqrt((x*x).sum(-1, keepdims=True)), 1e-12)
    q, k = normalize(linear(d['q'], 'qk', False)), normalize(linear(d['k'], 'qk', False))
    probability = np_softmax(d['geometry_logits']+tau.astype(np.float64)[..., None]*(q@k.swapaxes(-1, -2)))
    qbase = np.log(np.clip(d['q_depth'], .1, 20))
    hidden = gelu(linear(np.concatenate((d['q'], qbase[..., None]), -1), 'query'))
    values = gelu(linear(np.concatenate((d['k'], np.log(np.clip(d['k_value_depth'], .1, 20))[..., None]), -1), 'value'))
    if intervention == 'zero_v': values = np.zeros_like(values)
    elif intervention == 'reverse_v': values = values[..., ::-1, :]
    residual = linear(gelu(linear(np.concatenate((hidden, probability@values), -1), 'head.0')), 'head.2')[..., 0]
    depth = np.clip(np.exp(qbase+np.clip(residual, -3, 3)), .1, 20)
    return depth, probability


def double_gradients(data, parameters, tau, intervention):
    """Separately assembled double autograd reference; no adapter calls."""
    d = {n:t.detach().double() for n, t in data.items()}
    p = {n:t.detach().double().requires_grad_() for n, t in parameters.items()}
    def linear(x, name, bias=True):
        return F.linear(x, p[name+'.weight'], p[name+'.bias'] if bias else None)
    q, k = F.normalize(linear(d['q'], 'qk', False), dim=-1), F.normalize(linear(d['k'], 'qk', False), dim=-1)
    attention = (d['geometry_logits']+torch.from_numpy(tau).double()[..., None]*(q@k.transpose(-1, -2))).softmax(-1)
    qbase = d['q_depth'].clamp(.1, 20).log()
    hidden = F.gelu(linear(torch.cat((d['q'], qbase[..., None]), -1), 'query'))
    values = F.gelu(linear(torch.cat((d['k'], d['k_value_depth'].clamp(.1, 20).log()[..., None]), -1), 'value'))
    if intervention == 'zero_v': values = torch.zeros_like(values)
    elif intervention == 'reverse_v': values = values.flip(-2)
    residual = linear(F.gelu(linear(torch.cat((hidden, attention@values), -1), 'head.0')), 'head.2').squeeze(-1).clamp(-3, 3)
    depth = (qbase+residual).exp().clamp(.1, 20)
    target = (data['q_depth']*.91).double()
    ((depth.log()-target.log())**2).mean().backward()
    return {n:t.grad for n, t in p.items()}


def case(data, name, mode, temperature, intervention, archive_path):
    from models.round17_depth_attention import DepthAttention
    from models.round17_trace_pe import trace_temperature
    torch.manual_seed(171711)
    model = DepthAttention(mode, bandwidth=.25, temperature=temperature)
    parameters = dict(model.named_parameters())
    depth, attention, values = model(data, intervention, return_attention=True)
    tau = reference_tau(data['support_covariance'].numpy(), mode, .25, temperature)
    actual_tau = (trace_temperature(data['support_covariance'], .25, temperature).detach().numpy()
                  if mode == 'trace' else np.full(tau.shape, temperature, dtype=np.float32))
    tau_exact = np.array_equal(tau, actual_tau)
    with torch.no_grad():
        q, k = F.normalize(model.qk(data['q']), dim=-1), F.normalize(model.qk(data['k']), dim=-1)
        content = (q@k.transpose(-1, -2)).numpy()
        same = np_softmax(data['geometry_logits'].numpy()+(tau[..., None]*content).astype(np.float64))
        aggregate32 = (attention.float()@values).numpy()
    probabilities = attention.detach().numpy()
    ref_depth, ref_probability = np_forward(data, parameters, tau, intervention)
    value64 = values.detach().numpy().astype(np.float64)
    ref_aggregate = probabilities@value64
    scaled_product_error = float(abs(aggregate32-ref_aggregate).max()/max(1., float(abs(value64).max())))
    target = data['q_depth']*.91
    ((depth.log()-target.log())**2).mean().backward()
    reference = double_gradients(data, parameters, tau, intervention)
    gradients = {}
    retained = dict(probability=probabilities, same_content_probability=same,
        high_precision_probability=ref_probability, depth=depth.detach().numpy(),
        high_precision_depth=ref_depth, tau32=tau, actual_tau32=actual_tau,
        values32=values.detach().numpy(), aggregate32=aggregate32, reference_aggregate64=ref_aggregate)
    for parameter_name, parameter in parameters.items():
        actual, expected = parameter.grad, reference[parameter_name]
        retained['parameter__'+parameter_name] = parameter.detach().numpy()
        if actual is not None: retained['actual_gradient__'+parameter_name] = actual.numpy()
        if expected is not None: retained['reference_gradient__'+parameter_name] = expected.numpy()
        if actual is None or expected is None:
            allowed = intervention=='zero_v' and parameter_name in ('value.weight','value.bias')
            gradients[parameter_name] = dict(disconnected=actual is None and expected is None,
                passed=allowed and actual is None and expected is None)
            continue
        actual = actual.double()
        expected_norm = float(torch.linalg.vector_norm(expected))
        actual_norm = float(torch.linalg.vector_norm(actual))
        absolute_l2 = float(torch.linalg.vector_norm(actual-expected))
        relative_l2 = absolute_l2/expected_norm if expected_norm else None
        gradients[parameter_name] = dict(reference_norm=expected_norm, actual_norm=actual_norm,
            absolute_l2_error=absolute_l2, relative_l2_error=relative_l2,
            maximum_absolute_error=float((actual-expected).abs().max()),
            passed=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
            and (relative_l2<=LIMITS['gradient_relative_l2'] if expected_norm else actual_norm==0))
    errors = dict(same_content_probability=float(abs(probabilities-same).max()),
        high_precision_probability=float(abs(probabilities-ref_probability).max()),
        high_precision_relative_depth=float((abs(depth.detach().numpy()-ref_depth)/ref_depth).max()),
        value_product_scaled_absolute=scaled_product_error)
    zero_qk_exact = temperature!=0 or bool((model.qk.weight.grad==0).all())
    passed = (tau_exact and zero_qk_exact and all(errors[n]<=LIMITS[n] for n in errors)
              and all(v['passed'] for v in gradients.values()))
    if archive_path.exists(): raise ValueError('preserve original numerical archive')
    np.savez_compressed(archive_path, **retained)
    return dict(fixture=name, mode=mode, temperature=temperature, intervention=intervention,
        passed=passed, tau32_exact=tau_exact, t0_qk_exact_zero=zero_qk_exact if temperature==0 else None,
        errors=errors, gradients=gradients, probability_sha256=array_digest(probabilities),
        depth_sha256=array_digest(depth.detach().numpy()), minimum_probability=float(probabilities.min()),
        probability_zero_count=int((probabilities==0).sum()), parameters=sum(p.numel() for p in model.parameters()),
        retained_archive=archive_path.name, retained_archive_sha256=digest(archive_path),
        retained_probability_shape=list(probabilities.shape))


def verify_source_closure():
    python = {name for name in SOURCES if name.endswith('.py')}
    found, pending = set(), list(python)
    while pending:
        name = pending.pop()
        if name in found: continue
        found.add(name)
        tree = ast.parse((ROOT/name).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import): names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level: raise ValueError('relative import needs explicit closure review: '+name)
                if node.module: names = [node.module]+[node.module+'.'+n.name for n in node.names]
            for module in names:
                parts = module.split('.')
                candidates = [Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py']
                candidates += [Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
                pending.extend(str(p) for p in candidates if (ROOT/p).is_file() and str(p) not in found)
    if found != python: raise ValueError('source closure incomplete: '+str(sorted(found-python)))
    return len(found)


def freeze(out):
    if out.exists(): raise ValueError('refusing preflight overwrite')
    closure_count = verify_source_closure()
    timing = ROOT/'results/round17_validation/timing/completion.json'
    if not timing.is_file(): raise ValueError('main paired timing must be completed')
    out.mkdir(parents=True)
    sources = {name:digest(ROOT/name) for name in SOURCES}
    for name in sources:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    write(out/'lock.json', dict(created_unix=time.time(), source_hashes=sources, runtime=runtime(),
        limits=LIMITS, fixtures=[[1,432,432], [2,6,7]], fixture_seed=171710, model_seed=171711,
        synthetic_only=True, real_inputs_read=False, actual_task_training_authorized=False,
        scientific_gate_passed=False, main_validation_effects_read=False,
        recursive_python_sources=closure_count, main_timing_completion_sha256=digest(timing)))
    (out/'lock.sha256').write_text(digest(out/'lock.json')+'\n')


def run(out):
    if not out.is_dir(): raise ValueError('pre-existing frozen directory required')
    if (out/'started.json').exists(): raise ValueError('preserve each preflight attempt')
    started = time.time()
    write(out/'started.json', dict(created_unix=started,
        lock_sha256=digest(out/'lock.json') if (out/'lock.json').is_file() else None))
    try:
        lock = json.loads((out/'lock.json').read_text())
        if digest(out/'lock.json')!=(out/'lock.sha256').read_text().strip() or lock['runtime']!=runtime():
            raise ValueError('preflight runtime/lock changed')
        for name, expected in lock['source_hashes'].items():
            if digest(ROOT/name)!=expected or digest(out/'source_snapshot'/name)!=expected:
                raise ValueError('preflight source changed: '+name)
        if (lock['source_hashes'].keys()!=set(SOURCES) or lock['limits']!=LIMITS
                or lock['recursive_python_sources']!=verify_source_closure()
                or lock['main_timing_completion_sha256']!=digest(ROOT/'results/round17_validation/timing/completion.json')):
            raise ValueError('preflight fixed source/policy/timing changed')
        fixtures = {'native':fixture(1,432,432), 'batched_rectangular':fixture(2,6,7)}
        fixture_hashes = {name:{k:array_digest(v.numpy()) for k,v in data.items()} for name,data in fixtures.items()}
        for name,data in fixtures.items():
            np.savez_compressed(out/('fixture_'+name+'.npz'), **{k:v.numpy() for k,v in data.items()})
        write(out/'fixtures.json', dict(created_unix=time.time(), fields=fixture_hashes, synthetic_only=True))
        with (out/'unit_tests.log').open('x') as log:
            unit = subprocess.run([sys.executable,'-c',
                "import numpy; import torch,pytest; torch.set_num_threads(1); torch.use_deterministic_algorithms(True); "
                "torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; "
                "raise SystemExit(pytest.main(['-q','tests/test_round17_depth_attention.py']))"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if unit.returncode: raise RuntimeError('unit preflight failed; see preserved log')
        checks = []
        for name, data in fixtures.items():
            for mode in ('trace','constant'):
                for intervention in (None,'zero_v','reverse_v'):
                    check = case(data,name,mode,10.,intervention,out/f'arrays_{len(checks)+1:02d}.npz')
                    checks.append(check)
                    write(out/f'check_{len(checks):02d}.json', check)
            checks.append(case(data,name,'trace',0.,None,out/f'arrays_{len(checks)+1:02d}.npz'))
            write(out/f'check_{len(checks):02d}.json', checks[-1])
            point = dict(data, support_covariance=torch.zeros_like(data['support_covariance']))
            checks.append(case(point,name+'_zero_covariance','trace',10.,None,out/f'arrays_{len(checks)+1:02d}.npz'))
            write(out/f'check_{len(checks):02d}.json', checks[-1])
        passed = all(c['passed'] for c in checks)
        write(out/'verification.json', dict(passed=passed, synthetic_only=True, checks=checks,
            check_count=len(checks), unit_tests_passed=True, scientific_gate_passed=False,
            actual_task_training_run=False, real_inputs_or_labels_read=False,
            same_content_gemm_shared=True, numpy_full_precision_reference_independent=True,
            torch_double_gradient_assembly_separate=True, started_unix=started, finished_unix=time.time()))
    except Exception as error:
        write(out/'failure.json', dict(passed=False, error_type=type(error).__name__, error=str(error),
                                      started_unix=started, failed_unix=time.time()))
        write(out/'completion.json', dict(passed=False, files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}))
        raise
    write(out/'completion.json', dict(passed=passed, scientific_gate_passed=False,
        files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}, finished_unix=time.time()))
    if not passed: raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze','run'))
    parser.add_argument('--out', type=Path, default=ROOT/'results/round17_value_interface_preflight')
    args=parser.parse_args(); setup()
    (freeze if args.stage=='freeze' else run)(args.out.resolve())


if __name__=='__main__': main()
