"""Post-acceptance R20 delivery tables; not part of frozen scientific execution.

Standard library only. Never imports a model/runner, opens labels or decodes
NPZ arrays. Accepted failures remain failures; there is no new selection/gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT=ROOT/'results/round20_anchor_root_acceptance_20261031.json'
DEFAULT_OUT=ROOT/'results/round20_anchor_report_tables'
RUN=ROOT/'results/round20_anchor_fixed'
AUDIT=ROOT/'results/round20_anchor_independent_audit'
SCENES=('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
FAMILIES=('anchor_trace','mean_only_matched_tau','anchor_constant','anchor_isotropic_matched',
    'anchor_point_regional','anchor_point_localizer','anchor_point_legacy','anchor_point_matched',
    'anchor_fixed_regional','anchor_fixed_matched','anchor_quartic_regional','anchor_quartic_localizer')
CONDITIONS=tuple(q+'_'+k for q in ('g8','g16','mixed') for k in ('g8','g16','mixed'))+(
    'g8_g24','g24_g8','g16_g24','g24_g16','g24_g24')
CENTER_CONDITIONS=tuple(c for c in CONDITIONS if not c.startswith('mixed_'))
REQUIRED_ACCEPTANCE=('accepted_for_development_interpretation',
    'full_forward_and_final_independent_audits_passed','strict_forward_before_label_chronology_checked')
COUNT_NOTE=('Counts across conditions are query-condition occurrences, not independent samples. '
            'Different key partitions reuse query identities; roles also reuse the same source queries.')


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def resolve_bound(path,repo):
    p=Path(path);p=p if p.is_absolute() else repo/p
    require(not p.is_symlink(),'symlink is not a bound report input: '+str(p))
    p=p.resolve();require(p.is_relative_to(repo.resolve()),'receipt binding outside repository')
    return p


def verify_receipt(path,repo=ROOT):
    """This is the only read before all receipt hash bindings are checked."""
    receipt=read(path)
    require(all(receipt.get(k) is True for k in REQUIRED_ACCEPTANCE),'completed root development acceptance required')
    files=receipt.get('files')
    require(isinstance(files,dict) and bool(files),'root receipt must bind accepted files')
    verified={}
    for name,expected in files.items():
        p=resolve_bound(name,repo)
        require(isinstance(expected,str) and len(expected)==64,'invalid SHA256 in root receipt')
        require(digest(p)==expected,'root-accepted artifact changed: '+str(p))
        verified[str(p)]=expected
    return receipt,verified


def check_inventory(directory,completion):
    files=completion['files']
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p!=directory/'completion.json'}
    require(actual==set(files),'accepted inventory changed: '+str(directory))
    for relative,expected in files.items():
        p=Path(relative)
        require(not p.is_absolute() and '..' not in p.parts,'unsafe inventory path')
        require(not (directory/p).is_symlink() and digest(directory/p)==expected,'accepted inventory artifact changed: '+relative)


def accepted_metadata(receipt_path,run=RUN,audit=AUDIT,repo=ROOT):
    receipt,bindings=verify_receipt(receipt_path,repo)
    # Requiring these specific receipt bindings prevents a valid unrelated
    # receipt from authorizing the summary about to be opened.
    required=(run/'lock.json',run/'forward_complete.json',run/'summary.json',run/'completion.json',
              audit/'lock.json',audit/'forward_verification.json',audit/'verification.json',audit/'completion.json')
    require(all(str(p.resolve()) in bindings for p in required),'receipt omits this producer/audit evidence chain')
    pc,ac=read(run/'completion.json'),read(audit/'completion.json')
    require(pc.get('passed') is True and ac.get('passed') is True,'completed producer and final audit required')
    check_inventory(run,pc);check_inventory(audit,ac)
    verification=read(audit/'verification.json');forward=read(audit/'forward_verification.json')
    require(verification.get('passed') is True and forward.get('passed') is True,'both independent audits must pass')
    for field,path in (
        ('producer_lock_sha256',run/'lock.json'),('producer_forward_complete_sha256',run/'forward_complete.json'),
        ('producer_summary_sha256',run/'summary.json'),('producer_completion_sha256',run/'completion.json'),
        ('audit_lock_sha256',audit/'lock.json'),('forward_verification_sha256',audit/'forward_verification.json'),
        ('independent_summary_sha256',audit/'independent_summary.json')):
        require(verification[field]==digest(path),'final audit evidence binding changed: '+field)
    require(forward['producer_lock_sha256']==digest(run/'lock.json')
            and forward['producer_forward_complete_sha256']==digest(run/'forward_complete.json')
            and forward['audit_lock_sha256']==digest(audit/'lock.json'),'forward audit chain changed')
    lock,design=read(run/'lock.json'),read(audit/'lock.json')
    require(design['producer_lock_sha256']==digest(run/'lock.json') and Path(design['run']).resolve()==run.resolve(),
            'independent audit belongs to another run')
    for directory,l in ((run,lock),(audit,design)):
        require(digest(directory/'lock.json')==(directory/'lock.sha256').read_text().strip(),'lock checksum changed')
        for name,expected in l['source_hashes'].items():
            p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'unsafe source path')
            require(digest(repo/p)==expected and digest(directory/'source_snapshot'/p)==expected,'frozen science source changed')
    require(lock['families']==list(FAMILIES) and lock['matrices']==8064,'fixed operator population changed')
    # Effect JSON is parsed only after root acceptance, all its bindings,
    # complete inventories and the independent audit chain above.
    summary=read(run/'summary.json')
    require(summary['producer_lock_sha256']==digest(run/'lock.json')
            and summary['forward_audit_sha256']==digest(audit/'forward_verification.json')
            and summary['scientific_gate_passed'] is False and summary['validation_or_fresh_read'] is False
            and summary['inherited_baselines'] is True and summary['new_baselines_optimized_for_a'] is False,
            'report scope or summary source changed')
    require(verification['screen_passed']==summary['screen']['passed'],'accepted screen differs')
    numeric=read(audit/'forward_audit/numerical_checks.json')
    forward_rows=read(run/'forward/records.json')
    return dict(receipt=receipt,receipt_sha256=digest(receipt_path),receipt_bindings=bindings,lock=lock,
                verification=verification,forward_verification=forward,summary=summary,numeric=numeric,
                forward_rows=forward_rows,run=str(run),audit=str(audit))


def number(value):
    require(type(value) in (int,float) and math.isfinite(value),'finite scalar required')
    return value


def exact_fraction(value):
    require(isinstance(value,dict) and set(value)=={'numerator','denominator','value'},'complete exact Fraction required')
    require(type(value['numerator']) is int and type(value['denominator']) is int and value['denominator']>0,'invalid Fraction integers')
    f=Fraction(value['numerator'],value['denominator'])
    require(f.numerator==value['numerator'] and f.denominator==value['denominator'] and float(f)==value['value'],
            'noncanonical or changed exact Fraction')
    return f


def fraction_columns(prefix,value):
    f=exact_fraction(value)
    return {prefix+'_numerator':f.numerator,prefix+'_denominator':f.denominator,prefix+'_value':float(f)}


def scalar_columns(prefix,value):
    if isinstance(value,dict):return dict({prefix:None},**fraction_columns(prefix,value))
    return {prefix:number(value),prefix+'_numerator':None,prefix+'_denominator':None,prefix+'_value':None}


def role_identity(role,inherited,lock):
    spec=lock['old_controls'][role] if inherited else lock['role_configs'][role]
    cfg=spec['config'] if inherited else spec
    return dict(role=role,evidence='inherited' if inherited else 'new_forward',
                source_family=spec['family'] if inherited else role,h=cfg['h'],temperature=cfg['temperature'],
                grid_index=spec.get('grid_index'),new_settings_optimized_for_a=False)


def summary_tables(summary,lock):
    require(set(summary['by_role'])==set(FAMILIES),'all twelve new roles required')
    require(set(summary['old_by_role'])==set(lock['old_controls']),'all inherited aliases required')
    tables={n:[] for n in ('regional_scene_conditions','regional_equal_scene','regional_role_macros',
                            'center_scene_conditions','center_equal_scene','screen_all_checks','screen_failed_checks',
                            'primary_regional_comparisons','primary_center_comparisons')}
    banks=((False,summary['by_role']),(True,summary['old_by_role']))
    for inherited,bank in banks:
        for role,v in bank.items():
            identity=role_identity(role,inherited,lock)
            require(set(v['by_scene_condition'])==set(SCENES) and set(v['by_condition'])==set(CONDITIONS),
                    'complete regional role summary required')
            for scene in SCENES:
                require(set(v['by_scene_condition'][scene])==set(CONDITIONS),'omitted scene condition')
                for condition in CONDITIONS:
                    q,k=condition.split('_');values=v['by_scene_condition'][scene][condition]
                    require(type(values['mass']) is int and values['mass']>0,'integer visible-pixel mass required')
                    tables['regional_scene_conditions'].append(dict(identity,scene=scene,condition=condition,query=q,key=k,**values))
                tables['regional_role_macros'].append(dict(identity,scene=scene,aggregation='within_scene_equal_conditions',**v['by_scene'][scene]))
            tables['regional_role_macros'].append(dict(identity,scene='equal_scene_macro',aggregation='equal_conditions_within_scene_then_equal_scenes',**v['macro']))
            for condition in CONDITIONS:
                q,k=condition.split('_')
                tables['regional_equal_scene'].append(dict(identity,condition=condition,query=q,key=k,aggregation='equal_three_scenes',**v['by_condition'][condition]))
            center=v['center'];conditions=tuple(center['by_condition'])
            if not inherited:require(set(conditions)==set(CENTER_CONDITIONS),'new center condition omitted')
            if conditions:
                require(set(center['by_scene_condition'])==set(SCENES),'center scene omitted')
                for scene in SCENES:
                    require(set(center['by_scene_condition'][scene])==set(conditions),'center condition omitted')
                    for condition in conditions:
                        q,k=condition.split('_');c=center['by_scene_condition'][scene][condition];f=exact_fraction(c['pck'])
                        require(f==Fraction(c['pck_count'],c['n_point']),'center Fraction/count mismatch')
                        tables['center_scene_conditions'].append(dict(identity,scene=scene,condition=condition,query=q,key=k,
                            pck_count=c['pck_count'],n_point=c['n_point'],n_query=c['n_query'],**fraction_columns('pck',c['pck']),
                            distance_sum=c.get('distance_sum'),distance_squared_sum=c.get('distance_squared_sum'),
                            mean=c.get('mean'),rmse=c.get('rmse'),continuous_available='mean' in c))
                for condition in conditions:
                    q,k=condition.split('_');c=center['by_condition'][condition]
                    expected=sum((exact_fraction(center['by_scene_condition'][s][condition]['pck']) for s in SCENES),Fraction())/3
                    require(exact_fraction(c['pck'])==expected,'center macro is not equal-scene Fraction')
                    tables['center_equal_scene'].append(dict(identity,condition=condition,query=q,key=k,
                        **fraction_columns('pck',c['pck']),mean=c.get('mean'),rmse=c.get('rmse'),
                        aggregation='equal_scene_PCK_and_mean_of_scene_mean_RMSE',continuous_available='mean' in c))
    checks=summary['screen']['checks']
    require(len(checks)==60 and len({(r['category'],r['name']) for r in checks})==60,'all sixty distinct checks required')
    require(Counter(r['category'] for r in checks)==dict(old_macro=8,old_condition=28,same_a_macro=6,old_center=9,same_a_center=9),
            'fixed screen comparison groups changed')
    for i,c in enumerate(checks):
        require(type(c['passed']) is bool,'exact screen boolean required')
        row={k:c[k] for k in ('category','name','metric','reference','passed')}
        row.update(check_index=i)
        for key in ('candidate_value','reference_value','bound'):row.update(scalar_columns(key,c[key]))
        # Exact originals are retained alongside convenient flattened CSV fields.
        row['exact_check_json']=json.dumps(c,sort_keys=True,separators=(',',':'))
        tables['screen_all_checks'].append(row)
        if c['passed'] is False:tables['screen_failed_checks'].append(dict(row))
    require(summary['screen']['passed']==all(c['passed'] for c in checks),'screen aggregate changed')
    primary=summary['by_role']['anchor_trace']
    for inherited,bank in banks:
        for role,reference in bank.items():
            if not inherited and role=='anchor_trace':continue
            identity=role_identity(role,inherited,lock)
            for scene in (*SCENES,'equal_scene_macro'):
                p=primary['macro'] if scene=='equal_scene_macro' else primary['by_scene'][scene]
                r=reference['macro'] if scene=='equal_scene_macro' else reference['by_scene'][scene]
                for metric in ('common9_kl','direct5_kl','J'):
                    tables['primary_regional_comparisons'].append(dict(identity,scene=scene,condition='macro',metric=metric,
                        primary_value=p[metric],reference_value=r[metric],primary_minus_reference=p[metric]-r[metric],
                        relative_primary_improvement=None if r[metric]==0 else 1-p[metric]/r[metric]))
                for condition in CONDITIONS:
                    p=primary['by_condition'][condition] if scene=='equal_scene_macro' else primary['by_scene_condition'][scene][condition]
                    r=reference['by_condition'][condition] if scene=='equal_scene_macro' else reference['by_scene_condition'][scene][condition]
                    for metric in ('kl','js','direct_kl','direct_js'):
                        tables['primary_regional_comparisons'].append(dict(identity,scene=scene,condition=condition,metric=metric,
                            primary_value=p[metric],reference_value=r[metric],primary_minus_reference=p[metric]-r[metric],
                            relative_primary_improvement=None if r[metric]==0 else 1-p[metric]/r[metric]))
                for condition in reference['center']['by_condition']:
                    p=primary['center']['by_condition'][condition] if scene=='equal_scene_macro' else primary['center']['by_scene_condition'][scene][condition]
                    r=reference['center']['by_condition'][condition] if scene=='equal_scene_macro' else reference['center']['by_scene_condition'][scene][condition]
                    delta=exact_fraction(p['pck'])-exact_fraction(r['pck'])
                    tables['primary_center_comparisons'].append(dict(identity,scene=scene,condition=condition,
                        **fraction_columns('primary_pck',p['pck']),**fraction_columns('reference_pck',r['pck']),
                        delta_numerator=delta.numerator,delta_denominator=delta.denominator,delta_pp=float(delta*100),
                        primary_mean=p.get('mean'),reference_mean=r.get('mean'),primary_rmse=p.get('rmse'),reference_rmse=r.get('rmse')))
    require(sum(r['evidence']=='new_forward' for r in tables['regional_scene_conditions'])==504,'new region table must contain 12x14x3 rows')
    return tables


def numerical_tables(rows,lock,forward_rows):
    first={s:next(e['pair'] for e in lock['entries'] if e['scene']==s) for s in SCENES}
    expected={(s,tuple(first[s]),c,f) for s in SCENES for c in CONDITIONS for f in FAMILIES}
    keys=[(r['scene'],tuple(r['pair']),r['condition'],r['family']) for r in rows]
    require(len(keys)==len(set(keys))==504 and set(keys)==expected,'fixed complete numerical population required')
    tables=dict(numerical_checks=[],numerical_argmax_changes=[],numerical_tail_rows=[],numerical_by_role=[],
                full_forward_numerics_by_role=[])
    for row in rows:
        identity={n:row[n] for n in ('scene','condition','query','key','family')};identity['pair']=json.dumps(row['pair'],separators=(',',':'))
        require(len(row['query_indices'])==8 and len(set(row['query_indices']))==8,'fixed eight query rows required')
        require(row['passed'] is True and row['tau32_exact'] is True and row['bias32_exact'] is True,'accepted numerical check failed')
        item=dict(identity,passed=row['passed'],query_indices_json=json.dumps(row['query_indices']),
            production_order=row['production_order'],reference_orders_json=json.dumps(row['reference_orders']),
            reference_probability_error=row['max_probability_error'],reference_convergence_error=row['max_reference_convergence_error'],
            hybrid_probability_error=row['hybrid']['max_probability_error'],hybrid_convergence_error=row['hybrid']['max_reference_convergence_error'],
            same_mass_probability_error=row['same_mass_probability_error'],content_error=row['content_error'],
            maximum_anchor_error_px=row['maximum_anchor_error_px'],tau32_exact=row['tau32_exact'],bias32_exact=row['bias32_exact'],
            reference_argmax_changes=row['argmax_change_count'],hybrid_argmax_changes=row['hybrid']['argmax_change_count'],
            archive_path=row['archive_path'],archive_sha256=row['archive_sha256'])
        item.update({n+'_normalized_error':v for n,v in row['moment_errors'].items()})
        tables['numerical_checks'].append(item)
        for kind,detail in (('reference',row),('hybrid',row['hybrid'])):
            require(detail['argmax_change_count']==len(detail['argmax_changes']),'argmax change population differs')
            for change in detail['argmax_changes']:
                require(change['query_index'] in row['query_indices'],'argmax query must be original native index')
                tables['numerical_argmax_changes'].append(dict(identity,reference_kind=kind,**change))
        for kind,masses in (('actual',row['actual_raw_fov_mass']),('reference',row['raw_fov_mass']),('hybrid',row['hybrid']['raw_fov_mass'])):
            require(len(masses)==8,'raw tail population differs')
            for local,(query,mass) in enumerate(zip(row['query_indices'],masses)):
                require(number(mass)>=0,'negative raw FOV mass')
                tables['numerical_tail_rows'].append(dict(identity,reference_kind=kind,sampled_row_index=local,
                    query_index=query,raw_fov_mass=mass,zero_fov=mass==0,tiny_fov=mass<1.e-20,counting_unit='query_condition_occurrence'))
    for family in FAMILIES:
        selected=[r for r in rows if r['family']==family];flat=[r for r in tables['numerical_checks'] if r['family']==family]
        unique={(r['scene'],tuple(r['pair']),r['query'],i) for r in selected for i in r['query_indices']}
        item=dict(family=family,matrices=len(selected),query_condition_occurrences=sum(len(r['query_indices']) for r in selected),
                  unique_query_identities=len(unique),all_passed=all(r['passed'] for r in selected),counts_are_independent_samples=False)
        for name in ('reference_probability_error','reference_convergence_error','hybrid_probability_error',
                     'hybrid_convergence_error','same_mass_probability_error','content_error','maximum_anchor_error_px'):
            item['max_'+name]=max(r[name] for r in flat)
        for name in ('old_mean','original_cov','anchor_cov','executed_cov'):
            item['max_'+name+'_normalized_error']=max(r['moment_errors'][name] for r in selected)
        item['all_tau32_exact']=all(r['tau32_exact'] for r in selected)
        item['all_bias32_exact']=all(r['bias32_exact'] for r in selected)
        for kind in ('reference','hybrid'):
            changes=[r for r in tables['numerical_argmax_changes'] if r['family']==family and r['reference_kind']==kind]
            item[kind+'_argmax_change_occurrences']=len(changes)
            item[kind+'_argmax_unique_query_identities']=len({(r['scene'],r['pair'],r['query'],r['query_index']) for r in changes})
            item[kind+'_argmax_exact_tie_occurrences']=sum(r['exact_reference_tie'] for r in changes)
            item[kind+'_argmax_near_tie_occurrences']=sum(r['near_reference_tie'] for r in changes)
        for kind in ('actual','reference','hybrid'):
            tails=[r for r in tables['numerical_tail_rows'] if r['family']==family and r['reference_kind']==kind]
            item[kind+'_zero_fov_occurrences']=sum(r['zero_fov'] for r in tails)
            item[kind+'_tiny_fov_occurrences']=sum(r['tiny_fov'] for r in tails)
            item[kind+'_minimum_raw_fov_mass']=min(r['raw_fov_mass'] for r in tails)
            item[kind+'_maximum_raw_fov_mass']=max(r['raw_fov_mass'] for r in tails)
        item['actual_zero_raw_cells']=sum(r['actual_zero_raw_cells'] for r in selected)
        item['reference_zero_raw_cells']=sum(r['zero_raw_cells'] for r in selected)
        item['hybrid_zero_raw_cells']=sum(r['hybrid']['zero_raw_cells'] for r in selected)
        tables['numerical_by_role'].append(item)
    cohort={(e['scene'],tuple(e['pair'])) for e in lock['entries']}
    expected={(s,p,c,f) for s,p in cohort for c in CONDITIONS for f in FAMILIES}
    keys=[(r['scene'],tuple(r['pair']),r['condition'],r['family']) for r in forward_rows]
    require(len(keys)==len(set(keys))==8064 and set(keys)==expected,'all full-forward numerical records required')
    for family in FAMILIES:
        selected=[r for r in forward_rows if r['family']==family]
        tables['full_forward_numerics_by_role'].append(dict(family=family,matrices=len(selected),
            query_condition_occurrences=sum(r['shape'][0] for r in selected),
            unique_query_identities=sum({'g8':432,'g16':108,'mixed':243,'g24':48}[q] for _ in cohort for q in ('g8','g16','mixed','g24')),
            raw_zero_cell_occurrences=sum(r['numerical']['raw_zero_cells'] for r in selected),
            raw_zero_fov_occurrences=sum(r['numerical']['raw_zero_fov_rows'] for r in selected),
            raw_tiny_fov_occurrences=sum(r['numerical']['raw_tiny_fov_rows'] for r in selected),
            minimum_raw_fov_mass=min(r['numerical']['raw_fov_min'] for r in selected),
            maximum_raw_fov_mass=max(r['numerical']['raw_fov_max'] for r in selected),
            max_normalization_error=max(r['numerical']['max_normalization_error'] for r in selected),
            all_finite=all(r['numerical']['all_finite'] for r in selected),counts_are_independent_samples=False))
    return tables


def csv_write(path,rows,fields=None):
    if fields is None:
        fields=list(dict.fromkeys(k for row in rows for k in row))
    require(bool(fields),'empty CSV needs explicit columns')
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def extract(receipt_path=DEFAULT_RECEIPT,out=DEFAULT_OUT,run=RUN,audit=AUDIT):
    out=out.resolve();run=run.resolve();audit=audit.resolve()
    require(not out.exists(),'refusing existing report output')
    require(not any(out.is_relative_to(p) or p.is_relative_to(out) for p in (run,audit)),
            'report must be outside frozen producer/audit directories')
    accepted=accepted_metadata(receipt_path,run,audit)
    tables=summary_tables(accepted['summary'],accepted['lock'])
    tables.update(numerical_tables(accepted['numeric'],accepted['lock'],accepted['forward_rows']))
    report=dict(post_acceptance_delivery_only=True,independent_new_audit=False,new_parameter_selection=False,
        scientific_gate_passed=False,new_baselines_optimized_for_a=False,inherited_baselines_labeled=True,
        count_interpretation=COUNT_NOTE,center_rmse_interpretation='equal-scene mean of scene RMSE is not pooled RMSE',
        center_error_units='pixels',pck_radius_pixels=8,
        receipt_path=str(receipt_path.resolve()),receipt_sha256=accepted['receipt_sha256'],
        producer=str(run),audit=str(audit),source_sha256=digest(Path(__file__)),
        fixed_screen=accepted['summary']['screen'],role_declarations=accepted['lock']['role_configs'],
        inherited_role_declarations=accepted['lock']['old_controls'],
        reference_policy=accepted['lock']['numerical_references'],
        audited_numerical_checks=accepted['numeric'],tables=tables,created_unix=time.time())
    # All computations above are metadata extraction. Failed acceptance leaves
    # no output directory; interrupted writes below are retained and never reused.
    out.mkdir(parents=True)
    for name,rows in tables.items():
        if rows:csv_write(out/(name+'.csv'),rows)
        elif name=='screen_failed_checks':csv_write(out/(name+'.csv'),rows,list(tables['screen_all_checks'][0]))
        elif name=='numerical_argmax_changes':
            csv_write(out/(name+'.csv'),rows,('scene','condition','query','key','family','pair','reference_kind',
                'query_index','sampled_row_index','actual_argmax','reference_argmax','reference_top_gap',
                'exact_reference_tie','near_reference_tie','reference_advantage'))
        else:raise ValueError('unexpected empty required report table: '+name)
    write(out/'report.json',report)
    write(out/'completion.json',dict(passed=True,post_acceptance_delivery_only=True,scientific_gate_passed=False,
        receipt_sha256=accepted['receipt_sha256'],source_sha256=digest(Path(__file__)),
        producer_summary_sha256=digest(run/'summary.json'),audit_verification_sha256=digest(audit/'verification.json'),
        table_row_counts={n:len(v) for n,v in tables.items()},files={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()},
        finished_unix=time.time()))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance',type=Path,default=DEFAULT_RECEIPT)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();extract(args.acceptance,args.out)


if __name__=='__main__':main()
