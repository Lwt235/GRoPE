"""Describe 96 already saved TRAIN supports; no prediction, label, or tuning.

The selected configuration is immutable accepted TRAIN metadata. Three fixed
first pairs and eight linspace queries per partition are not the full TRAIN
population. Across-key duplicates are removed only after byte identity checks.
Only this external descriptive output is written; frozen artifacts are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SCENES = ('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
COUNTS = dict(g8=432,g16=108,mixed=243,g24=48)
CONDITIONS = tuple((q,k) for q in ('g8','g16','mixed') for k in ('g8','g16','mixed')) + (
    ('g8','g24'),('g24','g8'),('g16','g24'),('g24','g16'),('g24','g24'))
FIELDS = {'indices','actual_mean','actual_cov','actual_content','reference_mean','reference_cov','reference_content','bounds'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def absolute(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT/value


def check_idle():
    timing = ROOT/'results/round17_validation/timing'
    if timing.exists() and not (timing/'completion.json').exists():
        raise RuntimeError('benchmark is active; wait without reading its results')


def bind_metadata(preflight, regional):
    paths = set()
    def bind(path, expected):
        path = absolute(path)
        if sha(path) != expected:
            raise ValueError('immutable source/metadata hash mismatch: '+str(path))
        paths.add(path)
    lock = read(preflight/'lock.json'); complete = read(preflight/'train/completion.json')
    bind(preflight/'lock.json',(preflight/'lock.sha256').read_text().strip())
    if complete['lock_sha256'] != sha(preflight/'lock.json') or complete['passed'] is not True:
        raise ValueError('completed numerical TRAIN preflight required')
    paths.add(preflight/'train/completion.json')
    for name,digest in lock['source_hashes'].items():
        bind(ROOT/name,digest); bind(preflight/'source_snapshot'/name,digest)
    bind(preflight/'partitions.npz',lock['partitions_sha256'])
    bind(preflight/'train/summary.json',complete['files']['summary.json'])
    summary = read(preflight/'train/summary.json')
    if (summary['passed'] is not True or summary['support_sets'] != 42 or summary['labels_read']
            or summary['validation_or_test_read'] or lock['labels_read'] or lock['validation_or_test_read']
            or lock['conditions'] != [list(c) for c in CONDITIONS]):
        raise ValueError('TRAIN-only support contract changed')
    data = absolute(lock['data_root']); bind(data/'completion.json',lock['data_completion_sha256'])
    population = read(data/'completion.json')['manifests']
    if len(lock['cohort']) != 3 or tuple(e['scene'] for e in lock['cohort']) != SCENES:
        raise ValueError('exactly three original TRAIN first pairs required')
    for entry in lock['cohort']:
        first = min((r for r in population if r['scene']==entry['scene'] and r['split']=='train'),key=lambda r:r['path'])
        if (absolute(first['path']) != absolute(entry['manifest_path']) or first['pair'] != entry['pair']
                or first['sha256'] != entry['manifest_sha256']):
            raise ValueError('cohort is not the original metadata-first TRAIN pair')
        bind(entry['manifest_path'],entry['manifest_sha256']); bind(entry['input_path'],entry['input_sha256'])
        meta = read(absolute(entry['manifest_path']))
        if (meta['split']!='train' or meta['forward_uses_sensor'] or meta['pair']!=entry['pair']
                or meta['scene']!=entry['scene'] or meta['input_sha256']!=entry['input_sha256']):
            raise ValueError('TRAIN forward observation identity changed')
    selected = read(regional/'selection.json')
    bind(regional/'selection.json',(regional/'selection.sha256').read_text().strip())
    audit_path = absolute(selected['audit_path']); bind(audit_path,selected['audit_sha256'])
    audit = read(audit_path); audited = read(audit_path.parent/'completion.json')
    paths.add(audit_path.parent/'completion.json')
    if audited['passed'] is not True or audited['files'].get(audit_path.name) != sha(audit_path) or audit['passed'] is not True:
        raise ValueError('accepted regional audit missing')
    for name in ('lock','forward_complete','summary','completion'):
        field = 'regional_'+name+'_sha256'; bind(regional/(name+'.json'),selected[field])
        if audit[field] != selected[field]:
            raise ValueError('accepted TRAIN selection/audit belongs to a different run')
    selected_cfg = selected['selection']['primary']
    if (selected_cfg['family']!='trace_full' or selected_cfg['grid_index']!=2
            or selected_cfg['config']!={'h':.25,'temperature':10.}
            or selected['train_screen']['passed'] is not True or selected['validation_run']
            or selected['selection']!=audit['selection']):
        raise ValueError('immutable accepted primary configuration changed')
    regional_source = read(regional/'lock.json')['source_hashes']
    if regional_source['models/round17_trace_pe.py'] != lock['source_hashes']['models/round17_trace_pe.py']:
        raise ValueError('accepted candidate uses a different temperature operator')
    expected = {str(Path('arrays')/e['scene'].replace('/','__')/f'{q}_{k}_support.npz')
                for e in lock['cohort'] for q,k in CONDITIONS}
    if {p for p in complete['files'] if p.endswith('_support.npz')} != expected:
        raise ValueError('support archive population changed')
    for name in sorted(expected):
        bind(preflight/'train'/name,complete['files'][name])
    config = dict(family=selected_cfg['family'],grid_index=selected_cfg['grid_index'],**selected_cfg['config'])
    return lock,complete,config,{str(p):sha(p) for p in sorted(paths)}


def describe(out, preflight, regional):
    check_idle()
    if out.exists():
        raise ValueError('refusing descriptive output overwrite')
    lock, complete, config, inputs = bind_metadata(preflight,regional)
    check_idle()
    os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    import numpy as np
    import torch
    from models import round17_trace_pe as production
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    source = Path(__file__)
    (out/'source.py').write_bytes(source.read_bytes())
    write(out/'lock.json',dict(created_unix=time.time(),source_sha256=sha(source),input_sha256=inputs,
        selected_configuration=config,cohort=lock['cohort'],descriptive_only=True,
        changes_selection=False,changes_gates=False,labels_decoded=False,validation_results_read=False,
        original_observation_archives_hashed_only=True,expected_archives=42,expected_query_occurrences=336,
        expected_unique_supports=96,expected_unique_support_sets=12,
        tau_formula='T*h²/(h²+trace(C)); scaled float64 evaluation, rounded once to float32',
        ratio_formula='float64(saved-exact tau32)/10; statistics use linear quantiles',
        covariance_coordinates='reference8 squared',scope='3 fixed metadata-first TRAIN pairs; 8 linspace queries per partition, not the full 48-pair TRAIN population'))
    try:
        rows=[]; duplicate_checks=[]; occurrences=0
        for entry in lock['cohort']:
            groups={}
            for q,k in CONDITIONS:
                path=preflight/'train/arrays'/entry['scene'].replace('/','__')/f'{q}_{k}_support.npz'
                with np.load(path,allow_pickle=False) as archive:
                    if set(archive.files)!=FIELDS:
                        raise ValueError('saved support field set changed')
                    indices,cov,mean=(archive[n] for n in ('indices','actual_cov','actual_mean'))
                if (indices.dtype!=np.int64 or indices.shape!=(8,) or cov.dtype!=np.float32 or cov.shape!=(8,2,2)
                        or mean.dtype!=np.float32 or mean.shape!=(8,2) or not np.isfinite(cov).all()
                        or not np.array_equal(indices,np.linspace(0,COUNTS[q]-1,8,dtype=np.int64))):
                    raise ValueError('actual float32 fixed-eight support contract changed')
                occurrences+=8
                identity=(indices.tobytes(),cov.tobytes(),mean.tobytes())
                if q in groups:
                    if identity!=groups[q]['identity']:
                        raise ValueError('query support differs across key partitions; refuse deduplication')
                    groups[q]['keys'].append(k)
                else:
                    groups[q]=dict(identity=identity,indices=indices,cov=cov,keys=[k])
            for q,group in groups.items():
                cov=group['cov']; diagonal=np.diagonal(cov.astype(np.float64),axis1=-2,axis2=-1)
                trace=diagonal.sum(-1); h2=np.float64(config['h'])**2; T=np.float64(config['temperature'])
                scale=np.maximum(diagonal.max(-1),h2); relative=h2/scale
                tau64=T*(relative/(relative+(diagonal/scale[:,None]).sum(-1)))
                tau32=tau64.astype(np.float32)
                with torch.no_grad():
                    actual=production.trace_temperature(torch.from_numpy(cov),config['h'],config['temperature']).numpy()
                if actual.dtype!=np.float32 or actual.tobytes()!=tau32.tobytes():
                    raise ValueError('independent NumPy temperature differs from actual production float32')
                duplicate_checks.append(dict(scene=entry['scene'],pair=entry['pair'],query=q,keys=group['keys'],
                    indices=group['indices'].tolist(),cross_key_covariance_bytes_identical=True,
                    covariance_bytes_sha256=hashlib.sha256(cov.tobytes()).hexdigest(),tau32_bytes_sha256=hashlib.sha256(tau32.tobytes()).hexdigest(),
                    production_numpy_tau32_bytes_exact=True,maximum_direct_formula_error=float(abs(tau64-T*h2/(h2+trace)).max())))
                rows.extend(dict(scene=entry['scene'],pair=entry['pair'],query=q,query_index=int(index),
                    covariance_reference8=c.tolist(),trace_reference8=float(tr),tau32=float(tau),tau_over_T=float(np.float64(tau)/T))
                    for index,c,tr,tau in zip(group['indices'],cov,trace,tau32))
        if occurrences!=336 or len(rows)!=96 or len(duplicate_checks)!=12:
            raise ValueError('descriptive unique support population incomplete')
        def statistics(values):
            values=np.asarray(values,np.float64)
            return dict(zip(('min','p05','median','p95','max'),np.quantile(values,[0,.05,.5,.95,1],method='linear').tolist()))
        distributions={}
        for q in COUNTS:
            selected=[r for r in rows if r['query']==q]
            if len(selected)!=24:
                raise ValueError('partition sample population changed')
            distributions[q]=dict(count=24,**{name:statistics([r[name] for r in selected])
                for name in ('trace_reference8','tau32','tau_over_T')})
        write(out/'rows.json',rows); write(out/'cross_key_identity.json',duplicate_checks)
        write(out/'summary.json',dict(descriptive_only=True,configuration=config,archives=42,query_occurrences=occurrences,
            unique_supports=len(rows),unique_support_sets=12,duplicates_removed=occurrences-len(rows),
            all_cross_key_bytes_exact=True,all_production_numpy_tau32_bytes_exact=True,
            distributions=distributions,quantile_method='linear',labels_decoded=False,new_predictions=False,
            validation_results_read=False,selection_or_gate_changed=False,
            scope='Only 3 metadata-first TRAIN pairs × 4 query partitions × 8 fixed queries; no full-TRAIN or validation population claim.'))
        write(out/'completion.json',dict(completed=True,finished_unix=time.time(),
            files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        return distributions
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),finished_unix=time.time()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/round17_train_temperature_description')
    parser.add_argument('--preflight',type=Path,default=ROOT/'results/round17_preflight')
    parser.add_argument('--regional',type=Path,default=ROOT/'results/round17_regional')
    args=parser.parse_args()
    print(json.dumps(describe(args.out,args.preflight,args.regional),indent=2))


if __name__=='__main__':
    main()
