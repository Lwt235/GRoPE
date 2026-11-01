"""Jointly frozen, label-free R17 numerical orchestration.

Only this adapter calls production preparation. Reference moments/content,
integration and trace arithmetic are separately implemented and retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import round15_feasibility as A
from experiments import round15_regional_audit_numeric as OLD
from experiments import round17_trace_numerical_audit as N
from models import round17_trace_pe as P

ROOT = Path(__file__).resolve().parents[1]
sha, read, write = A.sha, A.read, A.write
SCENES = A.SCENES
GRID = A.GRID
CONDITIONS = OLD.CONDITIONS
FIELDS = A.FIELDS
SOURCE_NAMES = (
    'ROUND17_CANDIDATE_PROTOCOL.md', 'docs/round17_design_decision.md',
    'docs/round17_literature_scope.md',
    'docs/round16_trace_temperature_review.md',
    'ROUND17_PREFLIGHT_PROTOCOL.md', 'models/round17_trace_pe.py',
    'tests/test_round17_trace_pe.py', 'experiments/round17_preflight.py',
    'tests/test_round17_preflight.py', 'experiments/round17_trace_numerical_audit.py',
    'tests/test_round17_trace_numerical_audit.py', 'docs/round17_trace_theory.md',
)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sources():
    result = A.sources()
    for name in tuple(OLD.SOURCE_NAMES) + SOURCE_NAMES:
        result[name] = sha(ROOT / name)
    return result


def cohort(data):
    records = A.T.data_records(data)
    rows = []
    for scene in SCENES:
        chosen = min((r for r in records if r['scene'] == scene), key=lambda r: r['path'])
        path = absolute(chosen['path'])
        meta = read(path)
        if sha(path) != chosen['sha256'] or meta['split'] != 'train' or meta['forward_uses_sensor']:
            raise ValueError('invalid frozen first TRAIN pair')
        inp = path.parent / 'inputs.npz'
        if sha(inp) != meta['input_sha256']:
            raise ValueError('changed old forward observation archive')
        rows.append(dict(scene=scene, pair=chosen['pair'], manifest_path=str(path),
                         manifest_sha256=chosen['sha256'], input_path=str(inp), input_sha256=meta['input_sha256']))
    return rows


def evidence():
    paths = ('results/round15_regional_post_result_decision.json',
             'results/round16_mechanism/lock.json', 'results/round16_mechanism/summary.json',
             'results/round16_mechanism/completion.json',
             'results/round16_mechanism_independent_audit/verification.json',
             'results/round16_delivery_manifest.json')
    verification = read(ROOT / paths[-2])
    if not verification['passed']:
        raise ValueError('R16 independent audit must pass')
    for name in ('lock','summary','completion'):
        if verification['mechanism_'+name+'_sha256'] != sha(ROOT/f'results/round16_mechanism/{name}.json'):
            raise ValueError('R16 independent audit binding differs')
    return {name: sha(ROOT / name) for name in paths}


def freeze(out, data):
    if out.exists():
        raise ValueError('refuse existing numerical run')
    data = data.resolve()
    records, hashes = cohort(data), sources()
    out.mkdir(parents=True)
    for name in hashes:
        dest = out / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    parts = A.S.M.partitions()
    np.savez_compressed(out / 'partitions.npz', **{n+'__'+k:v for n,p in parts.items() for k,v in p.items()})
    lock = dict(created_unix=time.time(), source_hashes=hashes, cohort=records,
        data_root=str(data), data_completion_sha256=sha(data/'completion.json'),
        evidence=evidence(), fields=list(FIELDS), conditions=[list(x) for x in CONDITIONS],
        grid=list(GRID), query_indices='linspace(0,n-1,8,dtype=int64)',
        partitions_sha256=sha(out/'partitions.npz'), probability_rows=4032, geometry_rows=1344,
        reference_orders=[64,96], probability_tolerance=1e-4, reference_tolerance=1e-8,
        support_relative_tolerance=1e-4, content_absolute_tolerance=1e-5,
        production_order=32, floor=1e-30, labels_read=False, validation_or_test_read=False,
        numpy_version=np.__version__, torch_version=torch.__version__, cpu_threads=4)
    write(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN', sha(out/'lock.json'), flush=True)


def verify_lock(out):
    lock = read(out/'lock.json')
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip() or lock['source_hashes'] != sources():
        raise ValueError('numerical source/lock changed')
    for name, digest in lock['source_hashes'].items():
        if sha(out/'source_snapshot'/name) != digest:
            raise ValueError('numerical source snapshot changed')
    if (lock['cohort'] != cohort(Path(lock['data_root'])) or lock['evidence'] != evidence()
            or sha(Path(lock['data_root'])/'completion.json') != lock['data_completion_sha256']
            or sha(out/'partitions.npz') != lock['partitions_sha256']
            or lock['grid'] != list(GRID) or lock['conditions'] != [list(x) for x in CONDITIONS]
            or lock['fields'] != list(FIELDS) or lock['reference_orders'] != [64,96]
            or lock['numpy_version'] != np.__version__ or lock['torch_version'] != torch.__version__):
        raise ValueError('numerical input/policy changed')
    return lock


def finish(directory, out, result, started):
    verify_lock(out)
    write(directory/'summary.json', dict(result, lock_sha256=sha(out/'lock.json'),
        seconds=time.time()-started, labels_read=False, validation_or_test_read=False,
        scientific_gate_passed=False))
    files = {str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
    write(directory/'completion.json', dict(passed=result['passed'], lock_sha256=sha(out/'lock.json'),
        files=files, started_unix=started, finished_unix=time.time()))
    print(directory.name, 'PASS' if result['passed'] else 'FAIL', flush=True)


def begin(out, stage):
    lock = verify_lock(out)
    directory = out/stage
    directory.mkdir()  # deliberately fail on partial or completed stage
    A.configure_runtime('cpu')
    started = time.time()
    write(directory/'started.json', dict(started_unix=started, lock_sha256=sha(out/'lock.json')))
    return lock, directory, started


def synthetic(out):
    _, directory, started = begin(out, 'synthetic')
    result = N.run_synthetic(directory/'independent')
    finish(directory, out, result, started)


def completed_stage(out, stage):
    directory = out/stage
    value = read(directory/'completion.json')
    if value['lock_sha256'] != sha(out/'lock.json') or value['passed'] is not True:
        raise ValueError('passing preceding numerical stage required')
    if set(value['files']) != {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'completion.json'}:
        raise ValueError('numerical artifact population changed')
    A.verify_files({str(directory/p):h for p,h in value['files'].items()})
    return value


def row_check(actual, reference64, reference96, hybrid, index):
    for p in (actual, reference64, reference96, hybrid):
        if p.dtype != np.float64 or p.shape != actual.shape or not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>1e-12:
            raise ValueError('invalid normalized probability row')
    error = float(abs(actual-reference96).max())
    integration = float(abs(actual-hybrid).max())
    convergence = float(abs(reference64-reference96).max())
    winner, reference_winner = int(actual.argmax()), int(reference96.argmax())
    return dict(query_index=int(index), max_probability_error=error,
        integration_only_error=integration, reference_convergence_error=convergence,
        actual_argmax=winner, reference_argmax=reference_winner,
        argmax_changed=winner!=reference_winner,
        reference_winner_advantage=float(reference96[reference_winner]-reference96[winner]),
        passed=error<=1e-4 and integration<=1e-4 and convergence<=1e-8)


@torch.no_grad()
def train(out):
    completed_stage(out, 'synthetic')
    lock, directory, started = begin(out, 'train')
    parts = A.S.read_parts(out)
    tensors = A.S.M.partition_tensors(parts, 'cpu')
    schema = OLD.reference_partition_schema(parts)
    rows, geometry, supports = [], [], []
    for entry in lock['cohort']:
        inputs = A.load_forward_inputs(Path(entry['input_path']))
        obs = A.S.A.observation_tensors(inputs, 'cpu')
        actual_queries = {q:A.prepare_query(obs, tensors[q], True) for q in OLD.PARTITIONS}
        actual_keys = {k:A.S.M.prepare_key(obs, tensors[k]) for k in OLD.PARTITIONS}
        indices = {q:OLD.selected_indices(len(parts[q]['bounds'])) for q in OLD.PARTITIONS}
        ref_queries = {q:OLD.slice_support(OLD.AREF.reference_support(inputs,schema,q),indices[q]) for q in OLD.PARTITIONS}
        ref_keys = {k:OLD.AREF.reference_key(inputs,schema,k) for k in OLD.PARTITIONS}
        dest = directory/'arrays'/entry['scene'].replace('/','__')
        dest.mkdir(parents=True)
        for q,k in CONDITIONS:
            idx = indices[q]
            aq = actual_queries[q]
            amean, acov = aq['mean'][idx], aq['cov'][idx]
            ac = (aq['features']@actual_keys[k]['features'].T)[idx]
            rq = ref_queries[q]
            rc = np.asarray(rq['features']@ref_keys[k]['features'].T, np.float32)
            bounds = (np.asarray(parts[k]['bounds'],np.float64)-.5)/8
            support = dict(scene=entry['scene'], query=q,key=k,
                mean_error=float(np.max(abs(amean.numpy()-rq['mean'])/(1+abs(rq['mean'])))),
                covariance_error=float(np.max(abs(acov.numpy()-rq['cov'])/(1+abs(rq['cov'])))),
                content_error=float(np.max(abs(ac.numpy()-rc))))
            support['passed'] = max(support['mean_error'],support['covariance_error'])<=1e-4 and support['content_error']<=1e-5
            supports.append(support)
            np.savez_compressed(dest/f'{q}_{k}_support.npz', indices=idx, actual_mean=amean.numpy(),
                actual_cov=acov.numpy(), actual_content=ac.numpy(), reference_mean=rq['mean'],
                reference_cov=rq['cov'], reference_content=rc, bounds=bounds)
            for hi,h in enumerate((.25,.5,1.,2.)):
                raw = A.S.M.tiled_cell_mass(amean,acov.double(),actual_keys[k],h)
                r64 = N.independent_mass(rq['mean'],rq['cov'],bounds,h,64)
                r96 = N.independent_mass(rq['mean'],rq['cov'],bounds,h,96)
                rh = N.independent_mass(amean.numpy(),acov.numpy(),bounds,h,96)
                geo = (raw+1e-30*(actual_keys[k]['mass'].double()/64)[None]).log()
                arrays = dict(actual_raw=raw.numpy(), reference64_raw=r64, reference96_raw=r96, hybrid_raw=rh)
                for j,i in enumerate(idx):
                    cell = dict(scene=entry['scene'],query=q,key=k,h=h,query_index=int(i))
                    for name,mass in (('actual',raw.numpy()[j]),('reference',r96[j])):
                        fov = float(mass.sum())
                        floor = float(np.prod(bounds[:,2:]-bounds[:,:2],axis=1).sum()*1e-30)
                        cell.update({name+'_fov':fov,name+'_zero_fov':fov==0,name+'_tiny_fov':fov<1e-20,
                                     name+'_zero_cells':int((mass==0).sum()),name+'_floor_fraction':floor/(fov+floor)})
                    geometry.append(cell)
                for ti,t in enumerate((0.,5.,10.)):
                    p = P.probabilities(geo,ac,acov,h,t).numpy()
                    p64 = N.independent_probability(r64,rc,rq['cov'],h,t,bounds)
                    p96 = N.independent_probability(r96,rc,rq['cov'],h,t,bounds)
                    ph = N.independent_probability(rh,ac.numpy(),acov.numpy(),h,t,bounds)
                    arrays.update({f'actual_T{t:g}':p,f'reference64_T{t:g}':p64,f'reference96_T{t:g}':p96,f'hybrid_T{t:g}':ph})
                    for j,i in enumerate(idx):
                        rows.append(dict(scene=entry['scene'],pair=entry['pair'],query=q,key=k,h=h,temperature=t,
                            **row_check(p[j],p64[j],p96[j],ph[j],i)))
                np.savez_compressed(dest/f'{q}_{k}_h{h:g}.npz',**arrays)
            print('R17_NUMERIC',entry['scene'],q,k,len(rows),flush=True)
    if len(rows)!=4032 or len(geometry)!=1344 or len(supports)!=42:
        raise ValueError('incomplete predeclared numerical population')
    write(directory/'probability_rows.json',rows)
    write(directory/'geometry_rows.json',geometry)
    write(directory/'support_rows.json',supports)
    result = dict(passed=all(r['passed'] for r in rows+supports), probability_rows=len(rows),
        geometry_rows=len(geometry),support_sets=len(supports),
        max_probability_error=max(r['max_probability_error'] for r in rows),
        max_integration_only_error=max(r['integration_only_error'] for r in rows),
        max_reference_convergence_error=max(r['reference_convergence_error'] for r in rows),
        argmax_changes=sum(r['argmax_changed'] for r in rows),
        raw_zero_fov_rows=sum(r['actual_zero_fov'] for r in geometry),
        raw_tiny_fov_rows=sum(r['actual_tiny_fov'] for r in geometry),
        failures=[r for r in rows+supports if not r['passed']], real_old_train_observations=True)
    finish(directory,out,result,started)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','synthetic','train'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round17_preflight')
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    args=parser.parse_args()
    if args.stage=='freeze': freeze(args.out,args.data)
    elif args.stage=='synthetic': synthetic(args.out)
    else: train(args.out)


if __name__=='__main__': main()
