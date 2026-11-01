"""Frozen deterministic Round6 analysis: paired named-scan bootstrap, no seeds.

The CLI checks a root-written protocol lock BEFORE reading test result JSONs.
Pure functions are independently testable with synthetic rows. Primary units
are five named scan/place groups, not guaranteed independent physical rooms.
All central 95% intervals share 20,000 paired scan draws (RNG 20261030).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

GRIDS = ('q8_k8','q8_k16','q16_k8','q16_k16')
CROSS_GRIDS = ('q8_k16','q16_k8')
SAME_GRIDS = ('q8_k8','q16_k16')
ROLES = ('full','fixed','isotropic','point','point_pck')
BOOTSTRAP = 20000
RNG_SEED = 20261030
COVERAGE = ('0_0.25','0.25_0.5','0.5_0.75','0.75_1.01')
SPECS = {'kl':('kl_sum','visible_mass'), 'ce':('ce_sum','visible_mass'),
    'entropy':('entropy_sum','visible_mass'), 'js':('js_sum','visible_mass'),
    'query_kl':('row_kl_sum','n_valid'), 'pck8':('pck8_count','n_point'),
    'pck4':('pck4_count','n_point'), 'pck_stride':('pck_stride_count','n_point'),
    'point_ce':('point_ce_sum','n_point'), 'point_epe':('point_epe_sum','n_point'),
    'point_soft_epe':('point_soft_epe_sum','n_point')}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    if 'q_index' in row and 'k_index' in row:
        q,k=int(row['q_index']),int(row['k_index'])
        if 'pair_id' in row and row['pair_id'] != f'{q}:{k}':
            raise ValueError('pair_id disagrees with original frame indices')
    elif 'pair_id' in row:
        q,k=map(int,row['pair_id'].split(':'))
    else:
        raise ValueError('rows require original pair identity')
    grid=row['grid']
    if grid not in GRIDS:
        raise ValueError('unexpected grid')
    qs,ks=map(int,grid.replace('q','').replace('k','').split('_'))
    if int(row['q_stride']) != qs or int(row['k_stride']) != ks:
        raise ValueError('row stride metadata disagrees with grid')
    return str(row['scene']),grid,q,k


def align_rows(rows_by_role, expected_scenes):
    if len(expected_scenes)!=5 or len(set(expected_scenes))!=5:
        raise ValueError('exactly five unique frozen named scan/place groups required')
    if not set(ROLES).issubset(rows_by_role):
        raise ValueError('all five primary roles required')
    if set(rows_by_role)-set(ROLES)-{'full_no_content'}:
        raise ValueError('unfrozen extra role')
    indexed={}
    required=set(field for pair in SPECS.values() for field in pair)
    required.update(f'coverage_{c}_{suffix}' for c in COVERAGE for suffix in ['mass','kl'])
    for role,rows in rows_by_role.items():
        table={}
        for row in rows:
            if row.get('method',role)!=role:
                raise ValueError('row method role disagrees with result file')
            key=identity(row)
            if key in table:raise ValueError('duplicate scene/grid/frame-pair identity')
            if key[0] not in expected_scenes:raise ValueError('unexpected named scan/place group')
            if not required.issubset(row):raise ValueError('incomplete metric row')
            if not all(np.isfinite(float(row[field])) for field in required):
                raise ValueError('nonfinite metric row')
            for field in ['visible_mass','n_valid','n_point']:
                if row[field]<0:raise ValueError('negative measurement count')
            if (row['visible_mass']>0)!=(row['n_valid']>0):
                raise ValueError('visible mass/query validity mismatch')
            for field in ['pck8_count','pck4_count','pck_stride_count']:
                if not 0<=row[field]<=row['n_point']:raise ValueError('invalid PCK counts')
            if abs(sum(row[f'coverage_{c}_mass'] for c in COVERAGE)-row['visible_mass'])>1e-5:
                raise ValueError('coverage bins do not partition visible source mass')
            table[key]=row
        indexed[role]=table
    keys=sorted(indexed['full'])
    if not keys:raise ValueError('empty result rows')
    for role,table in indexed.items():
        if set(table)!=set(keys):raise ValueError('method row identities differ')
        for key in keys:
            a,b=indexed['full'][key],table[key]
            for field in ['n_valid','visible_mass','n_point']+[f'coverage_{c}_mass' for c in COVERAGE]:
                if a[field]!=b[field]:raise ValueError('method supervision counts/masks differ: '+field)
            if not np.isclose(a['entropy_sum'],b['entropy_sum'],rtol=1e-7,atol=1e-6):
                raise ValueError('method target entropy differs')
    for scene in expected_scenes:
        sets=[{(q,k) for s,g,q,k in keys if s==scene and g==grid} for grid in GRIDS]
        if not sets[0] or any(value!=sets[0] for value in sets[1:]):
            raise ValueError('missing or mismatched pair coverage across four grids')
    return indexed,keys


def aggregate(indexed, scenes):
    matrices={}
    for role,table in indexed.items():
        matrices[role]={}
        for metric,(numerator,denominator) in SPECS.items():
            values=np.full((len(scenes),len(GRIDS)),np.nan)
            for i,scene in enumerate(scenes):
                for j,grid in enumerate(GRIDS):
                    rows=[row for (s,g,_,_),row in table.items() if s==scene and g==grid]
                    n=sum(row[denominator] for row in rows)
                    if n>0:values[i,j]=sum(row[numerator] for row in rows)/n
            matrices[role][metric]=values
    return matrices


def _finite_mean(values):
    return float(np.mean(values)) if np.isfinite(values).all() else None


def paired_scan_statistics(full, baseline, draws, scenes, acquisition_groups=None, pck=False):
    """Equal named-scan macro; paired scan draws only, no manufactured seeds."""
    full,baseline=np.asarray(full,float),np.asarray(baseline,float)
    if full.shape!=baseline.shape or full.shape!=(len(scenes),):
        raise ValueError('expected one paired value per named scan')
    if not np.isfinite(full).all() or not np.isfinite(baseline).all():
        return {'evaluable':False,'reason':'a frozen scene/condition denominator is zero; no silent dropping'}
    delta=full-baseline
    a,b=full[draws].mean(-1),baseline[draws].mean(-1)
    bootstrap=a-b
    ci=np.quantile(bootstrap,[.025,.975]).tolist()
    denominator=float(baseline.mean())
    relative=float(delta.mean()/denominator) if denominator>0 else None
    relative_ci=np.quantile(bootstrap/b,[.025,.975]).tolist() if np.all(b>0) else None
    output={'evaluable':True,'full':float(full.mean()),'baseline':denominator,
        'difference':float(delta.mean()),'ci95':ci,'relative_difference':relative,
        'relative_ci95':relative_ci,'improved_scan_count':int((delta>0).sum() if pck else (delta<0).sum()),
        'per_scan':{s:{'full':float(a),'baseline':float(b),'difference':float(a-b)} for s,a,b in zip(scenes,full,baseline)},
        'leave_one_scan_out':{s:{'difference':float(np.delete(delta,i).mean()),
            'relative_difference':float(np.delete(delta,i).mean()/np.delete(baseline,i).mean()) if np.delete(baseline,i).mean()>0 else None}
            for i,s in enumerate(scenes)},'bootstrap':len(draws)}
    if pck:
        output.update(difference_pp=float(delta.mean()*100),ci95_pp=[float(x*100) for x in ci],
                      pck_unit='fractions internally; _pp fields are percentage points')
    groups=acquisition_groups or {}
    sensitivity={}
    for name,members in groups.items():
        selected=np.array([s in members for s in scenes])
        if not selected.any():continue
        sensitivity[name]={'named_scans':[s for s in scenes if s in members],
            'difference':float(delta[selected].mean()),
            'relative_difference':float(delta[selected].mean()/baseline[selected].mean()) if baseline[selected].mean()>0 else None,
            'leave_group_out_difference':float(delta[~selected].mean()) if (~selected).any() else None}
    output['acquisition_group_sensitivity']=sensitivity  # Deliberately no 3-group CI.
    return output


def analyze_rows(rows_by_role, expected_scenes, area_baseline='fixed',
                 acquisition_groups=None, bootstrap=BOOTSTRAP, rng_seed=RNG_SEED):
    if area_baseline not in ['point','fixed']:
        raise ValueError('area baseline must be validation-selected point or fixed')
    scenes=list(expected_scenes)
    if acquisition_groups is None:
        acquisition_groups={name:[s for s in scenes if s.lower().startswith(prefix)]
            for name,prefix in [('Brown','brown'),('MIT','mit'),('home','home')]}
        remaining=[s for s in scenes if not any(s in values for values in acquisition_groups.values())]
        if remaining:acquisition_groups['other_named_scans']=remaining
        acquisition_groups={k:v for k,v in acquisition_groups.items() if v}
    primary_rows={role:rows_by_role[role] for role in ROLES if role in rows_by_role}
    indexed,keys=align_rows(primary_rows,scenes)
    optional_status={}
    if 'full_no_content' in rows_by_role:
        try:
            indexed,keys=align_rows(rows_by_role,scenes)
            optional_status['full_no_content']='complete diagnostic; excluded from all primary gates'
        except ValueError as error:
            optional_status['full_no_content']='not analyzed: '+str(error)
    matrices=aggregate(indexed,scenes)
    draws=np.random.default_rng(rng_seed).integers(0,len(scenes),(bootstrap,len(scenes)))
    draw_hash=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    summaries={}
    for role,values in matrices.items():
        summaries[role]={}
        for metric,value in values.items():
            numerator,denominator=SPECS[metric]
            all_rows=indexed[role].values()
            total_denominator=sum(row[denominator] for row in all_rows)
            summaries[role][metric]={'macro':_finite_mean(value),
                'by_grid':{g:_finite_mean(value[:,j]) for j,g in enumerate(GRIDS)},
                'by_scan':{s:_finite_mean(value[i]) for i,s in enumerate(scenes)},
                'by_scan_grid':{s:{g:float(value[i,j]) if np.isfinite(value[i,j]) else None for j,g in enumerate(GRIDS)} for i,s in enumerate(scenes)},
                'pooled_ratio':sum(row[numerator] for row in all_rows)/total_denominator if total_denominator>0 else None}
    def compare(role,metric,grids):
        indices=[GRIDS.index(grid) for grid in grids]
        return paired_scan_statistics(matrices['full'][metric][:,indices].mean(-1),
            matrices[role][metric][:,indices].mean(-1),draws,scenes,acquisition_groups,metric.startswith('pck'))
    comparisons={}
    slices={'all':GRIDS,'cross':CROSS_GRIDS,'same':SAME_GRIDS,**{g:(g,) for g in GRIDS}}
    for role in indexed:
        if role=='full':continue
        result={'kl':{name:compare(role,'kl',grids) for name,grids in slices.items()}}
        for metric in ['ce','js','query_kl']:
            result[metric]=compare(role,metric,GRIDS)
        for metric in ['pck8','pck4','pck_stride','point_ce','point_epe','point_soft_epe']:
            result[metric]=compare(role,metric,('q8_k8',))
        result['status']='primary preselected baseline' if role==area_baseline else 'PCK noninferiority baseline' if role=='point_pck' else 'exploratory or diagnostic; no simultaneous coverage claim'
        comparisons['full_minus_'+role]=result
    primary=comparisons['full_minus_'+area_baseline]
    improvements={}
    for condition in ['all',*CROSS_GRIDS]:
        d=primary['kl'][condition]
        improvements[condition]=bool(d['evaluable'] and d['relative_difference'] is not None
            and d['relative_difference']<=-.02 and d['ci95'][1]<0)
    stable=bool(primary['kl']['all']['evaluable'] and primary['kl']['all']['improved_scan_count']>=4)
    same={}
    for grid in SAME_GRIDS:
        d=primary['kl'][grid]
        same[grid]=bool(d['evaluable'] and d['relative_ci95'] is not None and d['relative_ci95'][1]<.02)
    pck={}
    for role in dict.fromkeys([area_baseline,'point_pck']):
        d=comparisons['full_minus_'+role]['pck8']
        pck[role]=bool(d['evaluable'] and d['ci95'][0]>-.005)
    coverage={}
    for role,table in indexed.items():
        coverage[role]={}
        for c in COVERAGE:
            groups={}
            for scene in scenes:
                for grid in GRIDS:
                    rows=[row for (s,g,_,_),row in table.items() if s==scene and g==grid]
                    mass=sum(row[f'coverage_{c}_mass'] for row in rows)
                    numerator=sum(row[f'coverage_{c}_kl'] for row in rows)
                    groups[f'{scene}/{grid}']={'visible_mass':mass,'kl':numerator/mass if mass>0 else None}
            mass=sum(row[f'coverage_{c}_mass'] for row in table.values())
            coverage[role][c]={'groups':groups,'pooled_visible_mass':mass,
                'pooled_kl':sum(row[f'coverage_{c}_kl'] for row in table.values())/mass if mass>0 else None,
                'interpretation':'descriptive coverage sensitivity; never filters primary rows'}
    gates={'candidate':'full only; isotropic cannot replace a failed candidate after test',
        'area_baseline':area_baseline,'kl_at_least_2pct_and_ci_upper_below_zero':improvements,
        'at_least_four_of_five_scans_improve_total_kl':stable,
        'same_scale_relative_ci_upper_below_2pct':same,'q8_k8_pck8_ci_lower_above_minus_0_5pp':pck}
    gates['pass']=all(improvements.values()) and stable and all(same.values()) and all(pck.values())
    return {'statistical_protocol':{'unit':'five named scan/place groups; physical-room independence not guaranteed',
        'named_scans':scenes,'grids':list(GRIDS),'macro':'visible-pixel weighted within scene/grid, then equal named-scan/grid',
        'bootstrap':bootstrap,'rng_seed':rng_seed,'paired_draw_sha256':draw_hash,
        'interval':'central percentile 95% [.025,.975] for all gates','training_seed_replicates':0,
        'group_ci':'none: Brown/MIT/home sensitivity is descriptive, not a three-cluster CI'},
        'gates':gates,'summaries':summaries,'comparisons':comparisons,'coverage':coverage,
        'optional_diagnostics':optional_status,
        'pair_count_per_scan':{s:sum(key[0]==s and key[1]==GRIDS[0] for key in keys) for s in scenes},
        'limits':['Only five named scan/place groups; bootstrap precision is limited.',
            'All nonprimary and isotropic comparisons are exploratory; intervals are not multiplicity-adjusted.',
            'Inference is deterministic; repeated seeds would not be independent replicates.',
            'Provided reconstructed cameras and RGB-predicted Gaussian support are not optimized 3DGS primitive R/S.']}


def normalize_config(config):
    method=config.get('method')
    method={'point_pck':'point','full_no_content':'full'}.get(method,method)
    width=config.get('width',config.get('bandwidth'))
    temperature=config.get('temperature',config.get('content_temperature'))
    if method not in ['full','fixed','isotropic','point'] or width is None or temperature is None:
        raise ValueError('config must freeze method, width/bandwidth and temperature')
    if not np.isfinite([width,temperature]).all() or width<=0 or temperature<0:
        raise ValueError('invalid frozen configuration')
    return {'method':method,'width':float(width),'temperature':float(temperature)}


def validate_lock(lock_path):
    """No test file is opened until this source/selection contract passes."""
    lock_path=Path(lock_path)
    lock=json.loads(lock_path.read_text())
    if lock.get('analysis_source_sha256')!=sha256(__file__):
        raise ValueError('root protocol lock does not freeze this exact analysis source')
    scenes=lock.get('expected_scenes',[])
    if len(scenes)!=5 or len(set(scenes))!=5:
        raise ValueError('lock must name exactly five scan/place groups')
    if lock.get('area_baseline') not in ['point','fixed']:
        raise ValueError('lock must freeze validation-selected area_baseline')
    configs=lock.get('method_configs')
    if configs is None and isinstance(lock.get('roles'),dict):
        configs={role:value['config'] for role,value in lock['roles'].items()}
    if not isinstance(configs,dict) or not set(ROLES).issubset(configs):
        raise ValueError('lock must freeze method_configs for all five primary roles')
    if set(configs)-set(ROLES)-{'full_no_content'}:
        raise ValueError('unexpected locked role')
    for role,config in configs.items():
        normalized=normalize_config(config)
        expected={'point_pck':'point','full_no_content':'full'}.get(role,role)
        if normalized['method']!=expected:raise ValueError('role does not match locked model method')
    if 'full_no_content' in configs:
        full,diagnostic=normalize_config(configs['full']),normalize_config(configs['full_no_content'])
        if diagnostic['width']!=full['width'] or diagnostic['temperature']!=0:
            raise ValueError('full_no_content must use frozen full bandwidth with T=0')
    if 'selection_path' in lock:
        expected=lock.get('selection_v2_sha256',lock.get('selection_sha256'))
        if sha256(lock['selection_path'])!=expected:raise ValueError('frozen selection metadata changed')
    if 'expected_grids' in lock and set(lock['expected_grids'])!=set(GRIDS):
        raise ValueError('lock must freeze the four expected grids')
    lock['method_configs']=configs  # Normalize either supported root lock schema.
    return lock


def _source_contract(payload):
    if 'input_sources' not in payload:return None
    return sorted((str(row['scene']),str(row['grid']),str(row['input_sha256']),str(row['label_sha256']))
                  for row in payload['input_sources'])


def load_locked_results(results_root, lock_path):
    lock=validate_lock(lock_path)  # Must precede every test-result read.
    root=Path(results_root)
    rows,provenance,contracts={},{},{}
    lock_sha=sha256(lock_path)
    files=lock.get('result_files',{})
    for role,expected in lock['method_configs'].items():
        if role in files:
            path=Path(files[role])
            if not path.is_absolute():path=root/path
        else:
            candidates=[path for path in [root/role/'test.json',root/f'{role}_test.json'] if path.is_file()]
            if role=='full_no_content' and not candidates:
                provenance[role]={'status':'optional diagnostic not produced'}
                continue
            if len(candidates)!=1:raise ValueError('expected one unambiguous test result for role '+role)
            path=candidates[0]
        payload=json.loads(path.read_text())
        if payload.get('complete') is False:raise ValueError('incomplete evaluation result')
        if 'lock_sha256' in payload and payload['lock_sha256']!=lock_sha:
            raise ValueError('test result was produced under a different root lock')
        if 'sourcehash' in payload and 'sourcehash' in lock and payload['sourcehash']!=lock['sourcehash']:
            raise ValueError('test evaluator source hashes differ from root lock')
        actual=normalize_config(payload['config'])
        if actual!=normalize_config(expected):raise ValueError('test config differs from frozen validation selection')
        rows[role]=payload['rows']
        contracts[role]=_source_contract(payload)
        provenance[role]={'file':str(path),'sha256':sha256(path),'executed_config':actual}
    if any(value is not None for value in contracts.values()):
        if any(value is None for value in contracts.values()) or any(value!=contracts['full'] for value in contracts.values()):
            raise ValueError('methods do not share the same input/label source hashes')
    provenance['shared_input_hash_contract']='verified' if contracts['full'] is not None else 'unavailable; row count/entropy invariants checked, not a complete proof of equal labels'
    return rows,lock,provenance


def main(args):
    rows,lock,provenance=load_locked_results(args.results_root,args.lock)
    lock_hash=sha256(args.lock)
    result=analyze_rows(rows,lock['expected_scenes'],lock['area_baseline'],lock.get('acquisition_groups'))
    expected=lock.get('expected_pairs_per_scene')
    if expected is not None:
        expected={scene:expected for scene in lock['expected_scenes']} if isinstance(expected,int) else expected
        if expected!=result['pair_count_per_scan']:raise ValueError('unexpected frozen pair coverage')
    if sha256(args.lock)!=lock_hash or sha256(__file__)!=lock['analysis_source_sha256']:
        raise ValueError('lock/analysis source changed during analysis')
    for role in rows:
        if sha256(provenance[role]['file'])!=provenance[role]['sha256']:
            raise ValueError('test result changed during analysis')
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('refusing to overwrite analysis')
    args.out.mkdir(parents=True,exist_ok=True)
    result['provenance']={'lock':str(args.lock),'lock_sha256':lock_hash,'analysis_source_sha256':sha256(__file__),
        'result_files':provenance,'validation_selected_configs':lock['method_configs'],
        'selection_v2_sha256':lock.get('selection_v2_sha256',lock.get('selection_sha256'))}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    shutil.copy2(__file__,args.out/'source_snapshot.py')
    baseline=lock['area_baseline']
    primary=result['comparisons']['full_minus_'+baseline]
    lines=['# Frozen deterministic Gaussian position analysis','',
        f"Candidate full {'passes' if result['gates']['pass'] else 'does not pass'} all frozen criteria against validation-selected {baseline} and the PCK-optimal point guard. No alternative candidate is selected from test.",'',
        '| KL condition | full | baseline | difference | relative change | paired named-scan central95% CI |',
        '|---|---:|---:|---:|---:|---|']
    for name in ['all',*CROSS_GRIDS,*SAME_GRIDS]:
        d=primary['kl'][name]
        if not d['evaluable']:
            lines.append(f'| {name} | not evaluable | | | | |');continue
        rel=f"{d['relative_difference']*100:+.2f}%" if d['relative_difference'] is not None else 'undefined'
        lines.append(f"| {name} | {d['full']:.6f} | {d['baseline']:.6f} | {d['difference']:+.6f} | {rel} | [{d['ci95'][0]:+.6f},{d['ci95'][1]:+.6f}] |")
    lines+=['','PCK@8px guards use q8_k8 only. Differences below are percentage points, not percent relative changes.','']
    for role in dict.fromkeys([baseline,'point_pck']):
        d=result['comparisons']['full_minus_'+role]['pck8']
        if d['evaluable']:lines.append(f"- Full minus {role}: {d['difference_pp']:+.3f} pp, central95% CI [{d['ci95_pp'][0]:+.3f},{d['ci95_pp'][1]:+.3f}] pp.")
        else:lines.append(f'- Full minus {role}: not evaluable; gate fails.')
    lines+=['','The sampling units are five named scan/place groups, with limited bootstrap precision and no guarantee that all are independent physical rooms. Inference has no trained parameters and no training-seed replication. Isotropic and other comparisons are exploratory, without simultaneous interval coverage. Brown/MIT/home and leave-one-scan sensitivities are descriptive. CE, JS, point EPE, equal-query KL, pooled ratios and coverage diagnostics are retained in summary.json.','']
    (args.out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(result['gates'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
