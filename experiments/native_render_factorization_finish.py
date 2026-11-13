"""Finite endpoint integrity, full fixed tables and cost accounting; no new experiment."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from experiments.native_render_factorization import OUT,PLAN,ARMS,cases,write,verify,now
from experiments import native_coverage_data as d


def main():
    lock=verify();s=d.read(OUT/'scoring/summary.json');audit=d.read(OUT/'independent_recomputation.json');assert audit['complete']
    logs=[];starts=[];completes=[];gpu_seconds=0.;peak=0.;weights_seconds=0.;input_seconds=0.;events={}
    for path in sorted((OUT/'runtime').glob('*.jsonl')):
        lines=[json.loads(x) for x in path.read_text().splitlines()];assert lines[-1]['event']=='complete'
        gpu_seconds+=lines[-1]['elapsed_seconds'];peak=max(peak,max(r.get('peak_GiB',0) for r in lines))
        weights_seconds+=sum(r.get('seconds',0) for r in lines if r['event']=='weights_loaded')
        input_seconds+=sum(r.get('seconds',0) for r in lines if r['event']=='inputs_constructed')
        starts += [r for r in lines if r['event']=='forward_start'];completes += [r for r in lines if r['event']=='forward_complete']
        logs.append(dict(path=str(path),sha256=d.sha256(path),pid=lines[0]['pid'],gpu=lines[0]['cuda_visible_devices'],seconds=lines[-1]['elapsed_seconds'],seeds=lines[0]['seeds'],exited_successfully=True))
    assert len(starts)==len(completes)==4248 and len({(r['purpose'],r['key']) for r in starts})==4248
    assert len([r for r in starts if r['purpose']=='formal'])==4224
    assert d.read(OUT/'forward_budget.json')['charged']==4248
    assert gpu_seconds<=7200
    replay=[d.read(x) for x in (OUT/'replay').glob('*.json')];assert len(replay)==24 and max(r['max_abs'] for r in replay)<=1e-5
    assert len(list((OUT/'completed').glob('*.json')))==4224
    stats=[d.read(OUT/'input_statistics'/f'{m["id"]}_{q}.json') for m,q,_ in cases()];assert len(stats)==176
    expected={(m['id'],q) for m,q,_ in cases()};assert {(x['id'],x['query']) for x in stats}==expected
    assert all(x['alpha_quantile_levels']==[0,.25,.5,.75,.9,.99,1.] for x in stats)
    all_files=[x for x in OUT.rglob('*') if x.is_file()];size=sum(x.stat().st_size for x in all_files);assert size<2*2**30
    marker_seconds=sum(d.read(x)['inference_seconds'] for x in (OUT/'completed').glob('*.json'))
    groups={}
    for cohort in ('train','development'):
        vals=[v for v in stats if v['cohort']==cohort]
        ratio_vals=[v for v in vals if v['r_norm']>0]
        groups[cohort]=dict(queries=len(vals),alpha_zero_fraction_mean=float(np.mean([v['alpha_zero_fraction'] for v in vals])),alpha_zero_fraction_range=[min(v['alpha_zero_fraction'] for v in vals),max(v['alpha_zero_fraction'] for v in vals)],s_mean=float(np.mean([v['s'] for v in vals])),s_range=[min(v['s'] for v in vals),max(v['s'] for v in vals)],norm_ratio_defined_queries=len(ratio_vals),zero_r_norm_queries_retained=len(vals)-len(ratio_vals),norm_ratio_mean={k:float(np.mean([v['norm_ratio'][k] for v in ratio_vals])) for k in ('R',)+ARMS},relative_difference_mean={k:float(np.mean([v['relative_difference_from_r'][k] for v in ratio_vals])) for k in ('R',)+ARMS})
    write(OUT/'input_statistics/summary.json',dict(groups=groups,queries=176,zero_valid_slot_queries=sum(v['valid_slots']==0 for v in stats),zero_r_norm_queries=sum(v['r_norm']==0 for v in stats),norm_match_max_abs_ratio_error=max(abs(v['norm_ratio']['M']-1) for v in stats if v['r_norm']>0)))
    cost=dict(complete=True,training_fits=0,optimizer_updates=0,source_builds=0,new_model_weights=0,formal_new_predictions=4224,baseline_replay_forwards=24,extra_preflight_forwards=0,total_native_forwards=4248,maximum_native_forwards=4272,worker_seconds=gpu_seconds,worker_hours=gpu_seconds/3600,worker_hard_cap_hours=2,worker_time_scope='From worker admission to complete, including model loading, weights loading, input construction, inference, export and recovery; no resource waits occurred. CPU identity preparation and scoring are separate.',workers=logs,peak_allocated_GiB=peak,formal_forward_and_CPU_transfer_seconds=marker_seconds,weights_load_seconds=weights_seconds,input_construction_seconds=input_seconds,CPU_prepare_seconds=lock['seconds'],scoring_seconds=d.read(OUT/'scoring/runtime.json')['seconds'],new_disk_bytes_before_final_report=size,disk_hard_cap_bytes=2*2**30,foreign_shared_cards=0,max_concurrent_workers=2,GPUs=[3,4],other_users_jobs_untouched=True,extra_speed_benchmarks=0,no_download=True,no_weight_copy=True,source_asset_stats_unchanged=True,baseline_replay_max_abs=max(r['max_abs'] for r in replay),source_build_and_deployment_cost_reference=str(d.OUT/'cost_description.json'),not_end_to_end_RGB_benchmark=True)
    write(OUT/'cost_description.json',cost)
    write(OUT/'matched_checks.json',dict(complete=True,formal_completed_keys=4224,expected_complete_set=True,unique_native_forward_starts=4248,duplicate_native_forward_starts=0,replay_count=24,replay_max_abs=max(r['max_abs'] for r in replay),unchanged_source_asset_stats=True,input_assets_verified_SHA_before_run=len(lock['input_sha256']),models_verified_SHA_before_run=3,configuration_lock_sha256=d.sha256(OUT/'configuration_lock.json'),all_workers_recorded_complete=True,no_training=True,new_confirmation=False))
    # All regions and all six metrics are available in this standalone table.
    with (OUT/'scoring/all_summary_cells.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['cohort','condition','region','metric','arm','mean','seed20261110','seed20261111','seed20261112','per_place_in_summary_order'])
        for key,v in s['summaries'].items():writer.writerow(key.split('/')+[v['mean']]+v['per_seed']+[json.dumps(v['per_place'])])
    text=['# Fixed Render input factorization: complete tables','', 'All rows use the same fixed Render weights; Local is an inherited separately trained control. Positive error differences mean intervention cost. All data are already used; this is not confirmation.','','[All regions, six metrics, conditions, arms and seed/place means](scoring/all_summary_cells.csv) · [All paired contrasts and intervals](scoring/summary.json) · [All individual-query scores](scoring/metrics.json)','','## Absolute endpoints','','|Scope|Arm|Missing ROI centered log-MSE|Missing ROI raw log-MSE|Missing ROI bias²|Full whole AbsRel|','|---|---|---:|---:|---:|---:|']
    for cohort in ['train32','train_original','train_added','development12']:
        for arm in ['Local','R']+list(ARMS):
            vals=[s['summaries'][f'{cohort}/{cond}/{region}/{metric}/{arm}']['mean'] for cond,region,metric in [('missing','roi','centered_log_mse'),('missing','roi','log_mse'),('missing','roi','bias_squared'),('full','whole','absrel')]]
            text.append('|'+cohort+'|'+arm+'|'+ '|'.join(f'{x:.9f}' for x in vals)+'|')
    text+=['','## Prospective contrasts, missing ROI spatial error','','|Scope|Contrast|Mean absolute|95% paired interval|Seed differences|Positive / negative places|','|---|---|---:|---|---|---|']
    for cohort in ['train32','train_original','train_added','development12']:
        for name,v in s['contrasts'][f'{cohort}/missing/roi/centered_log_mse'].items():text.append(f"|{cohort}|{name}|{v['mean_absolute']:.9f}|[{v['interval95_absolute'][0]:.9f}, {v['interval95_absolute'][1]:.9f}]|"+', '.join(f'{x:.9f}' for x in v['per_seed_absolute'])+f"|{v['positive_place_count']} / {v['negative_place_count']}|")
    text+=['','## Relative comparisons','','Positive reduction favors the first arm. Paired intervals cross zero where stated; that does not establish equivalence.','','|Scope|Comparison|Error reduction %|95% paired interval %|','|---|---|---:|---|']
    for cohort in ['train32','train_original','train_added','development12']:
        for pair in [k+'-R' for k in ARMS]+[k+'-Local' for k in ('R',)+ARMS]:
            v=s['comparisons'][f'{cohort}/missing/roi/centered_log_mse/{pair}'];text.append(f"|{cohort}|{pair}|{v['relative_error_reduction_percent']:.4f}|[{v['interval95_percent'][0]:.4f}, {v['interval95_percent'][1]:.4f}]|")
    for cohort in ['train32','development12']:
        text+=['',f'## {cohort}: every place, three-seed means','','|Place|Local|R|N|C|U|M|C−R|N−R|M−R|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for pi,place in enumerate(s['place_order'][cohort]):
            v={arm:s['summaries'][f'{cohort}/missing/roi/centered_log_mse/{arm}']['per_place'][pi] for arm in ['Local','R']+list(ARMS)}
            vals=[v[arm] for arm in ['Local','R']+list(ARMS)]+[v['C']-v['R'],v['N']-v['R'],v['M']-v['R']]
            text.append('|'+place+'|'+'|'.join(f'{x:.9f}' for x in vals)+'|')
    text+=['','Bootstrap: 5000 paired resamples of three seeds and places, seed20261113; queries equal within place. Main scope has32 places; original/added splits12/20; development12 places. Correlated contrasts are descriptive, not independent tests. Input statistics and causal limitations remain part of the report.']
    (OUT/'report_tables.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(complete=True,cost=cost,input_summary=groups)),flush=True)


if __name__=='__main__':main()
