"""R17 Stage A: only the new trace Gaussian is predicted; sealed point controls reused.

Complete new forward precedes target decoding. All original point guards,
exact integer/Fraction decisions and all new native probability arrays remain.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from experiments import round15_feasibility as A
from experiments import round15_regional as B
from experiments import round17_preflight as N
from experiments import round17_arithmetic_audit as AR
from models import round17_trace_pe as P
ROOT,S,GRID,SCENES=A.ROOT,A.S,A.GRID,A.SCENES
FIELDS=A.FIELDS
QUERIES=A.QUERIES
QUERY_COUNTS=A.QUERY_COUNTS
FAMILIES=('trace_full',)
POINT_FAMILIES=('quartic_point','gaussian_point')
ALL_FAMILIES=FAMILIES+POINT_FAMILIES
LEGACY_INDEX=4
PCK_EXACT_TOLERANCE=Fraction(1,200)
RAW_PROBABILITY_BYTES=12*48*sum(QUERY_COUNTS.values())*432*8
OLD_A=ROOT/'results/round15_feasibility'
OLD_B=ROOT/'results/round15_regional'
sha,read,write,array_sha=A.sha,A.read,A.write,A.array_sha
verify_files=A.verify_files
absolute=N.absolute
configure_runtime=A.configure_runtime
load_forward_inputs=A.load_forward_inputs
prepare_query=A.prepare_query
load_center_targets=A.load_center_targets
center_metric=A.center_metric
complete_stage=A.complete_stage
verify_stage=A.verify_stage
SOURCE_NAMES=('ROUND17_CANDIDATE_PROTOCOL.md','docs/round17_design_decision.md',
 'experiments/round17_feasibility.py','tests/test_round17_feasibility.py',
 'experiments/round17_feasibility_audit.py','tests/test_round17_feasibility_audit.py')


def sources():
    output=N.sources()
    output.update(B.sources())
    for name in SOURCE_NAMES+AR.SOURCE_NAMES: output[name]=sha(ROOT/name)
    return output


def baseline_contract():
    bindings={}
    for base,names,prefix in ((OLD_A,('lock','forward_complete','summary','completion','decision'),'feasibility'),
                              (OLD_B,('lock','forward_complete','summary','completion','selection'),'regional')):
        for name in names: bindings[str((base/(name+'.json')).relative_to(ROOT))]=sha(base/(name+'.json'))
        complete=read(base/'completion.json')
        if complete['lock_sha256']!=sha(base/'lock.json') or complete['summary_sha256']!=sha(base/'summary.json'):
            raise ValueError('changed baseline completion chain')
        for path,digest in complete['files'].items():
            if path.endswith('.json'):
                if sha(base/path)!=digest: raise ValueError('changed old baseline table/lock')
                bindings[str((base/path).relative_to(ROOT))]=digest
    for name in ('results/round15_feasibility_independent_audit_path_repair/verification.json',
                 'results/round15_regional_independent_audit/verification.json',
                 'results/round15_regional_post_result_decision.json'):
        bindings[name]=sha(ROOT/name)
    a=read(ROOT/'results/round15_feasibility_independent_audit_path_repair/verification.json')
    b=read(ROOT/'results/round15_regional_independent_audit/verification.json')
    if a['passed'] is not True or b['passed'] is not True:
        raise ValueError('old independently audited controls required')
    if a['feasibility_summary_sha256']!=sha(OLD_A/'summary.json') or b['regional_summary_sha256']!=sha(OLD_B/'summary.json'):
        raise ValueError('old independent audit summary binding differs')
    return bindings


def numerical_contract(path):
    N.verify_lock(path)
    for stage in ('synthetic','train'): N.completed_stage(path,stage)
    arithmetic_root=ROOT/'results/round17_arithmetic_audit'
    alock,_=AR.verify_lock(arithmetic_root)
    audit=read(arithmetic_root/'verification.json')
    complete=read(arithmetic_root/'completion.json')
    A.verify_files({str(arithmetic_root/p):h for p,h in complete['files'].items()})
    if (audit['passed'] is not True or complete['passed'] is not True
        or alock['base_numeric_lock_sha256']!=sha(path/'lock.json')
        or audit['arithmetic_lock_sha256']!=sha(arithmetic_root/'lock.json')
        or complete['arithmetic_lock_sha256']!=sha(arithmetic_root/'lock.json')
        or audit['base_numeric_lock_sha256']!=sha(path/'lock.json')
        or audit['train_summary_sha256']!=sha(path/'train/summary.json')
        or audit['train_completion_sha256']!=sha(path/'train/completion.json')
        or audit['probability_rows']!=4032 or audit['probability_matrices']!=504
        or not audit['tau32_exact'] or not audit['bias32_exact']
        or audit['max_same_mass_probability_error']>1e-12
        or audit['labels_read'] or audit['validation_or_test_read'] or audit['scientific_gate_passed']):
        raise ValueError('passing independent same-mass arithmetic audit required')
    return dict(root=str(path.resolve()),files={name:sha(path/name) for name in
      ('lock.json','synthetic/summary.json','synthetic/completion.json','train/summary.json','train/completion.json')},
      arithmetic=dict(root=str(arithmetic_root),files={name:sha(arithmetic_root/name)
                      for name in ('lock.json','verification.json','completion.json')}))


def freeze(out,preflight):
    if out.exists(): raise ValueError('refusing Stage A overwrite')
    previous=A.verify_lock(OLD_A)
    controls=baseline_contract(); numeric=numerical_contract(preflight)
    entries=previous['target_contract']['entries']
    if len(entries)!=48: raise ValueError('unexpected old cohort')
    hashes=sources()
    out.mkdir(parents=True)
    shutil.copyfile(OLD_A/'partitions.npz',out/'partitions.npz')
    for name in hashes:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    lock=dict(created_unix=time.time(),source_hashes=hashes,baseline_bindings=controls,
      numerical_contract=numeric,target_contract=previous['target_contract'],inputs=previous['inputs'],
      manifests=previous['manifests'],training_manifest_records=previous['training_manifest_records'],
      data_root=previous['data_root'],data_completion_sha256=previous['data_completion_sha256'],
      partitions_sha256=sha(out/'partitions.npz'),grid=list(GRID),families=list(FAMILIES),
      reused_point_families=list(POINT_FAMILIES),queries=list(QUERIES),probability_arrays=1728,
      probability_rows=48*12*sum(QUERY_COUNTS.values()),raw_probability_bytes=RAW_PROBABILITY_BYTES,
      all_final_probabilities_retained=True,all_forward_before_target_decode=True,
      pck_tolerance_exact=[1,200],field_allowlist=list(A.FIELDS),legacy_point_index=4,
      numpy_version=np.__version__,torch_version=torch.__version__,
      center_labels_decoded=False,regional_labels_decoded=False,fresh_test_read=False)
    write(out/'lock.json',lock);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN',sha(out/'lock.json'),flush=True)


def verify_lock(out):
    lock=read(out/'lock.json')
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip() or lock['source_hashes']!=sources():
        raise ValueError('changed Stage A source/lock')
    for name,digest in lock['source_hashes'].items():
        if sha(out/'source_snapshot'/name)!=digest: raise ValueError('changed source snapshot')
    if (lock['baseline_bindings']!=baseline_contract() or
        lock['numerical_contract']!=numerical_contract(Path(lock['numerical_contract']['root'])) or
        lock['grid']!=list(GRID) or lock['families']!=list(FAMILIES) or
        lock['reused_point_families']!=list(POINT_FAMILIES) or lock['queries']!=list(QUERIES) or
        sha(out/'partitions.npz')!=lock['partitions_sha256'] or
        lock['numpy_version']!=np.__version__ or lock['torch_version']!=torch.__version__):
        raise ValueError('changed Stage A controls/numerics/configuration')
    A.verify_files({str(absolute(p)):h for p,h in lock['inputs'].items()})
    A.verify_files({str(absolute(p)):h for p,h in lock['manifests'].items()})
    target=lock['target_contract']
    if A.target_contract(Path(target['root']),Path(target['audit_path']),lock['training_manifest_records'])!=target:
        raise ValueError('old center metadata changed')
    return lock


def condition_predictions(full,key):
    content=full['features']@key['features'].T
    output,diagnostics={},[]
    for hi,h in enumerate((.25,.5,1.,2.)):
        raw=S.M.tiled_cell_mass(full['mean'],full['cov'].double(),key,h)
        fov=raw.sum(-1)
        if not bool(torch.isfinite(raw).all()) or bool((raw<0).any()): raise ValueError('invalid raw mass')
        geometry=(raw+1e-30*(key['mass'].double()/64)[None]).log()
        diagnostics.append(dict(family='trace_full',h=h,n_query=len(raw),raw_zero_cells=int((raw==0).sum()),
          raw_zero_fov_rows=int((fov==0).sum()),raw_tiny_fov_rows=int((fov<1e-20).sum()),
          raw_fov_min=float(fov.min()),raw_fov_max=float(fov.max())))
        for ti,t in enumerate((0.,5.,10.)):
            output['trace_full',hi*3+ti]=P.probabilities(geometry,content,full['cov'],h,t)
    return output,diagnostics


def pair_predictions(inputs,partition,device):
    obs=S.A.observation_tensors(inputs,device)
    key=S.M.prepare_key(obs,partition['g8'])
    arrays,diagnostics={},[]
    for q in QUERIES:
        full=prepare_query(obs,partition[q],True)
        predictions,numeric=condition_predictions(full,key)
        for (family,index),value in predictions.items(): arrays[named_probability(family,index,q)]=value.cpu().numpy()
        diagnostics.extend(dict(query=q,**r) for r in numeric)
    return arrays,diagnostics


def old_t0_commitments(scene):
    rows=read(OLD_B/'forward'/scene.replace('/','__')/'records.json')
    result={(tuple(r['pair']),r['grid_index'],r['query']):r['array_sha256'] for r in rows
            if r['family']=='gaussian_full1' and r['grid_index']%3==0 and r['key']=='g8' and r['query'] in QUERIES}
    if len(result)!=16*4*3: raise ValueError('incomplete old Gaussian T0 population')
    return result


def old_point_rows():
    return [r for s in SCENES for r in read(OLD_A/'evaluation'/s.replace('/','__')/'rows.json') if r['family'] in POINT_FAMILIES]


def named_probability(family, index, query):
    return f'{family}__c{index:02d}__{query}'


def expected_arrays():
    return {named_probability(f, i, q): (QUERY_COUNTS[q], 432)
            for f in FAMILIES for i in range(len(GRID)) for q in QUERIES}


def scene_entries(lock, scene):
    if scene not in SCENES:
        raise ValueError('only fixed old-training scenes are authorized')
    entries = sorted((e for e in lock['target_contract']['entries'] if e['scene'] == scene),
                     key=lambda e: Path(e['path']).name)
    if len(entries) != 16:
        raise ValueError('expected 16 old-training pairs')
    return entries


def validate_arrays(arrays):
    if set(arrays) != set(expected_arrays()):
        raise ValueError('missing or extra native probability arrays')
    for name, shape in expected_arrays().items():
        probability = arrays[name]
        if probability.shape != shape or probability.dtype != np.float64:
            raise ValueError('probability shape/dtype mismatch')
        if not np.isfinite(probability).all() or (probability < 0).any():
            raise ValueError('invalid probability')
        if np.max(np.abs(probability.sum(-1) - 1.)) > 1.e-12:
            raise ValueError('unnormalized probability')


def verify_population(rows, entries, families=FAMILIES):
    expected = {(e['scene'], tuple(e['pair']), f, i, q)
                for e in entries for f in families for i in range(12) for q in QUERIES}
    identities = [(r['scene'], tuple(r['pair']), r['family'], r['grid_index'], r['query']) for r in rows]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError('missing, duplicate or unauthorized probability population')
    populations = {}
    for row in rows:
        if row['config'] != GRID[row['grid_index']]:
            raise ValueError('configuration identity mismatch')
        if 'n_point' in row:
            identity = (row['scene'], tuple(row['pair']), row['query'])
            current = (row['n_query'], row['n_point'])
            if row['n_query'] != QUERY_COUNTS[row['query']] or not 0 <= row['pck_count'] <= row['n_point'] <= row['n_query']:
                raise ValueError('invalid center metric population')
            if identity in populations and populations[identity] != current:
                raise ValueError('center population differs between configurations')
            populations[identity] = current


@torch.no_grad()
def forward(out, scene, device):
    lock = verify_lock(out)
    entries = scene_entries(lock, scene)
    t0 = old_t0_commitments(scene)
    directory = out / 'forward' / scene.replace('/', '__')
    if directory.exists():
        raise ValueError('refusing overwrite/partial forward')
    if shutil.disk_usage(out).free < RAW_PROBABILITY_BYTES + 512 * 1024**2:
        raise ValueError('insufficient free space for full retained Stage A probabilities')
    runtime = configure_runtime(device)
    directory.mkdir(parents=True)
    started = time.time()
    write(directory / 'started.json', dict(created_unix=started, lock_sha256=sha(out / 'lock.json'), **runtime))
    partition = S.M.partition_tensors(S.read_parts(out), device)
    records, numerical = [], []
    for pair_index, entry in enumerate(entries):
        path = Path(entry['path'])
        inputs = load_forward_inputs(absolute(path) / 'inputs.npz')
        arrays, diagnostics = pair_predictions(inputs, partition, device)
        validate_arrays(arrays)
        for i in range(0,12,3):
            for q in QUERIES:
                if array_sha(arrays[named_probability("trace_full",i,q)]) != t0[tuple(entry["pair"]),i,q]:
                    raise ValueError("new T0 must exactly equal old Gaussian full1")
        if pair_index == 0:
            repeated, _ = pair_predictions(inputs, partition, device)
            if any(array_sha(arrays[k]) != array_sha(repeated[k]) for k in arrays):
                raise ValueError('actual-input repeated forward is not bitwise deterministic')
            write(directory / 'determinism.json', dict(scene=scene, pair=entry['pair'],
                array_hashes={k: array_sha(v) for k, v in arrays.items()}, passed=True,
                repeated_array_count=len(arrays), labels_decoded=False))
            del repeated
        name = path.name + '.npz'
        np.savez_compressed(directory / name, **arrays)
        archive_hash = sha(directory / name)
        for family in FAMILIES:
            for i, cfg in enumerate(GRID):
                for q in QUERIES:
                    field = named_probability(family, i, q)
                    records.append(dict(scene=scene, pair=entry['pair'], family=family, grid_index=i,
                        query=q, config=cfg, archive=name, archive_sha256=archive_hash, field=field,
                        shape=list(arrays[field].shape), dtype=str(arrays[field].dtype),
                        array_sha256=array_sha(arrays[field]), input_sha256=lock['inputs'][str(path / 'inputs.npz')]))
        numerical.extend(dict(scene=scene, pair=entry['pair'], **row) for row in diagnostics)
        print('FORWARD', scene, pair_index + 1, '/16', flush=True)
    write(directory / 'records.json', records)
    write(directory / 'numerical.json', numerical)
    complete_stage(directory, out, 'forward', scene, started, record_count=len(records), pair_count=16,
                   labels_decoded=False, all_probabilities_retained=True, runtime=runtime)


def forward_chain(out):
    result = {}
    for scene in SCENES:
        directory = out / 'forward' / scene.replace('/', '__')
        complete = verify_stage(directory, out, 'forward', scene)
        if complete['record_count'] != 16 * 36 or complete['labels_decoded']:
            raise ValueError('incomplete forward or prior label read')
        result[scene] = dict(completion_sha256=sha(directory / 'completion.json'),
                             finished_unix=complete['finished_unix'])
    return result


def seal_forward(out):
    lock = verify_lock(out)
    if (out / 'forward_complete.json').exists():
        raise ValueError('refusing forward seal overwrite')
    chain = forward_chain(out)
    all_records = []
    for scene in SCENES:
        directory = out / 'forward' / scene.replace('/', '__')
        records = read(directory / 'records.json')
        for entry in scene_entries(lock, scene):
            name = Path(entry['path']).name + '.npz'
            archive_hash = sha(directory / name)
            with np.load(directory / name, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            validate_arrays(arrays)
            matching = [r for r in records if r['pair'] == entry['pair']]
            for row in matching:
                if (row['archive'] != name or row['archive_sha256'] != archive_hash
                        or row['field'] != named_probability(row['family'], row['grid_index'], row['query'])
                        or row['array_sha256'] != array_sha(arrays[row['field']])
                        or row['shape'] != list(arrays[row['field']].shape) or row['dtype'] != 'float64'
                        or row['input_sha256'] != lock['inputs'][str(Path(entry['path']) / 'inputs.npz')]):
                    raise ValueError('forward array/record provenance mismatch')
        repeated = read(directory / 'determinism.json')
        first_pair = scene_entries(lock, scene)[0]['pair']
        first = {r['field']: r['array_sha256'] for r in records if r['pair'] == first_pair}
        if repeated['passed'] is not True or repeated['pair'] != first_pair or repeated['array_hashes'] != first:
            raise ValueError('first-pair determinism evidence mismatch')
        all_records.extend(records)
    verify_population(all_records, lock['target_contract']['entries'])
    write(out / 'forward_complete.json', dict(lock_sha256=sha(out / 'lock.json'),
        forward_chain=chain, array_count=len(all_records), raw_probability_bytes=RAW_PROBABILITY_BYTES,
        sealed_unix=time.time(), labels_decoded=False, all_probabilities_retained=True))


def verify_forward_complete(out):
    value = read(out / 'forward_complete.json')
    if (value['lock_sha256'] != sha(out / 'lock.json') or value['forward_chain'] != forward_chain(out)
            or value['array_count'] != 1728 or value['labels_decoded']):
        raise ValueError('global forward seal mismatch')
    return value


def evaluate(out, scene):
    lock = verify_lock(out)
    entries = scene_entries(lock, scene)
    seal = verify_forward_complete(out)  # Must precede every target hash/decode.
    directory = out / 'evaluation' / scene.replace('/', '__')
    if directory.exists():
        raise ValueError('refusing evaluation overwrite')
    started = time.time()
    if seal['sealed_unix'] > started:
        raise ValueError('forward-before-label chronology violated')
    directory.mkdir(parents=True)
    key_centers = S.read_parts(out)['g8']['mean'] * 8
    rows, label_sources = [], {}
    forward_dir = out / 'forward' / scene.replace('/', '__')
    records = read(forward_dir / 'records.json')
    for entry in entries:
        target = Path(entry['target_path'])
        if sha(target) != entry['target_sha256']:
            raise ValueError('independent center target changed')
        labels = load_center_targets(target)
        label_sources[str(target)] = entry['target_sha256']
        name = Path(entry['path']).name + '.npz'
        with np.load(forward_dir / name, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        validate_arrays(arrays)
        for record in (r for r in records if r['pair'] == entry['pair']):
            q = record['query']; probability = arrays[record['field']]
            if array_sha(probability) != record['array_sha256']:
                raise ValueError('probability changed before center evaluation')
            row = center_metric(probability, key_centers, labels[q + '_point_xy'], labels[q + '_point_visible'])
            rows.append(dict(**record, **row, target_sha256=entry['target_sha256']))
    verify_population(rows, entries)
    write(directory / 'rows.json', rows)
    complete_stage(directory, out, 'evaluation', scene, started, record_count=len(rows),
        forward_complete_sha256=sha(out / 'forward_complete.json'), label_sources=label_sources,
        center_labels_decoded=True, regional_labels_decoded=False)


def summarize_rows(rows, entries):
    verify_population(rows, entries, ALL_FAMILIES)
    metrics, exact = {}, {}
    for family in ALL_FAMILIES:
        values, exact_values = [], []
        for i in range(12):
            scene_values, scene_fractions = {}, {}
            for scene in SCENES:
                scene_values[scene], scene_fractions[scene] = {}, {}
                for q in QUERIES:
                    subset = [r for r in rows if (r['family'], r['grid_index'], r['scene'], r['query']) == (family, i, scene, q)]
                    denominator = sum(r['n_point'] for r in subset)
                    if denominator <= 0:
                        raise ValueError('empty scene/stride center population')
                    fraction = Fraction(sum(r['pck_count'] for r in subset), denominator)
                    scene_fractions[scene][q] = fraction
                    scene_values[scene][q] = dict(pck8=float(fraction),
                                                 n_point=denominator, n_query=sum(r['n_query'] for r in subset))
            query_fractions = {q: sum(scene_fractions[s][q] for s in SCENES) / len(SCENES) for q in QUERIES}
            macro = sum(query_fractions.values()) / len(QUERIES)
            by_query = {q: float(value) for q, value in query_fractions.items()}
            values.append(dict(grid_index=i, config=GRID[i], by_scene_query=scene_values,
                               by_query=by_query, equal_scene_equal_scale_pck8=float(macro)))
            exact_values.append(dict(by_query=query_fractions, macro=macro))
        metrics[family] = values
        exact[family] = exact_values
    selections = {family: max(range(12), key=lambda i: exact[family][i]['macro'])
                  for family in ('gaussian_point', 'quartic_point')}
    references = dict(gaussian_point=( 'gaussian_point', selections['gaussian_point']),
                      quartic_point=('quartic_point', selections['quartic_point']),
                      legacy_point_pck=('gaussian_point', LEGACY_INDEX))
    feasible, constraints = [], []
    for i in range(12):
        checks = []
        for q in QUERIES:
            candidate = exact['trace_full'][i]['by_query'][q]
            for name, (family, index) in references.items():
                reference = exact[family][index]['by_query'][q]
                checks.append(dict(query=q, reference=name, reference_family=family, reference_grid_index=index,
                    candidate_pck8=float(candidate), reference_pck8=float(reference),
                    difference_pp=float(100 * (candidate-reference)),
                    passed=candidate + PCK_EXACT_TOLERANCE >= reference))
        passed = all(row['passed'] for row in checks)
        constraints.append(dict(grid_index=i, config=GRID[i], feasible=passed, checks=checks))
        if passed:
            feasible.append(i)
    return dict(metrics=metrics, point_localization_selections={f: dict(grid_index=i, config=GRID[i])
                for f, i in selections.items()}, legacy_point=dict(grid_index=LEGACY_INDEX, config=GRID[LEGACY_INDEX]),
                feasibility=constraints, feasible_grid_indices=feasible, feasible=bool(feasible),
                decision_pending_independent_audit='complete_full_regional_calibration' if feasible else 'stop_unchanged_candidate',
                final_primary_selected=False, regional_calibration_done=False,
                scientific_gate_passed=False, fresh_test_read=False)


def summarize(out):
    lock = verify_lock(out)
    verify_forward_complete(out)
    if (out / 'summary.json').exists() or (out / 'completion.json').exists():
        raise ValueError('refusing summary overwrite')
    rows = []
    for scene in SCENES:
        directory = out / 'evaluation' / scene.replace('/', '__')
        completion = verify_stage(directory, out, 'evaluation', scene)
        if completion['forward_complete_sha256'] != sha(out / 'forward_complete.json'):
            raise ValueError('evaluation forward seal changed')
        rows.extend(read(directory / 'rows.json'))
    new_row_count=len(rows)
    rows.extend(old_point_rows())
    summary = summarize_rows(rows, lock['target_contract']['entries'])
    summary.update(lock_sha256=sha(out / 'lock.json'), forward_complete_sha256=sha(out / 'forward_complete.json'),
                   pair_count=48, row_count=len(rows), new_row_count=new_row_count, reused_point_row_count=len(rows)-new_row_count, created_unix=time.time())
    write(out / 'summary.json', summary)
    files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    write(out / 'completion.json', dict(lock_sha256=sha(out / 'lock.json'), files=files,
        summary_sha256=sha(out / 'summary.json'), forward_complete_sha256=sha(out / 'forward_complete.json'),
        finished_unix=time.time(), center_labels_decoded=True, regional_labels_decoded=False,
        fresh_test_read=False, independent_audit_required=True))
    print('FEASIBLE CONFIGURATIONS', summary['feasible_grid_indices'], 'AUDIT PENDING', flush=True)


def audit_decision(out, verification_path):
    verify_lock(out)
    complete, summary, audit = read(out / 'completion.json'), read(out / 'summary.json'), read(verification_path)
    verify_files({str(out / path): digest for path, digest in complete['files'].items()})
    expected = dict(feasibility_lock_sha256=sha(out / 'lock.json'),
                    feasibility_forward_complete_sha256=sha(out / 'forward_complete.json'),
                    feasibility_summary_sha256=sha(out / 'summary.json'),
                    feasibility_completion_sha256=sha(out / 'completion.json'))
    if audit['passed'] is not True or any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError('independent feasibility audit/hash binding required')
    if audit['feasible_grid_indices'] != summary['feasible_grid_indices'] or audit['scientific_gate_passed'] is not False:
        raise ValueError('independent feasibility conclusion mismatch')
    result = dict(**expected, audit_path=str(verification_path.resolve()), audit_sha256=sha(verification_path),
        feasible_grid_indices=summary['feasible_grid_indices'],
        action='complete_full_regional_calibration' if summary['feasible'] else 'stop_unchanged_candidate',
        primary_selected=False, validation_authorized=False, scientific_gate_passed=False,
        regional_design=dict(new_families=1, reused_families=8, configurations_per_family=12, conditions=14, unchanged_J=True),
        created_unix=time.time())
    write(out / 'decision.json', result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','forward','seal-forward','evaluate','summarize','accept-audit'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round17_feasibility')
    parser.add_argument('--preflight',type=Path,default=ROOT/'results/round17_preflight')
    parser.add_argument('--scene',choices=SCENES)
    parser.add_argument('--audit',type=Path)
    args=parser.parse_args()
    if args.stage=='freeze': freeze(args.out,args.preflight)
    elif args.stage=='forward': forward(args.out,args.scene,'cpu')
    elif args.stage=='seal-forward': seal_forward(args.out)
    elif args.stage=='evaluate': evaluate(args.out,args.scene)
    elif args.stage=='summarize': summarize(args.out)
    else: audit_decision(args.out,args.audit)

if __name__=='__main__': main()
