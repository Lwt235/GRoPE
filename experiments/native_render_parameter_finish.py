"""Finite endpoint checks, complete tables and prospective decision flags."""
import csv
import hashlib
import json
import os
import time
import numpy as np
from experiments.native_render_parameter_data import *
from experiments.native_render_parameter_queue import definitions,done
from experiments.native_render_parameter_train import verify_lock
from experiments.gaussian_scene_config import schedule


def material_harm(v):return v['relative_error_reduction_percent']<=-1 and v['interval95_percent'][1]<0
def screen(v):return v['relative_error_reduction_percent']>=1 and v['interval95_percent'][0]>0 and all(x>0 for x in v['per_seed_relative_percent'])


def main():
    p=plan();lock=verify_lock();s=read(OUT/'scoring/summary.json');cost=read(OUT/'cost.json');assert s['complete'] and cost['complete'];checks=[];paired={}
    for seed in SEEDS:
        order=schedule(p,seed);initial=[]
        for kind in KINDS:
            j=dict(name=f'{kind}_{seed}',kind=kind,seed=seed);assert done(j);folder=OUT/j['name'];status=read(folder/'status.json');identity=read(folder/'initial_identity.json');initial.append(identity)
            lines=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
            assert [r['step'] for r in lines]==list(range(1,2001));assert [sample for r in lines for sample in r['samples']]==[list(k) for k in order]
            assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in lines)
            assert status['native_parameters']+status['fusion_parameters']+status['PE_parameters']==638240546
            assert status['steps']==2000 and checkpoint_path(kind,seed).stat().st_size==status['bytes']
            assert not resume_path(kind,seed).exists() and read(folder/'resume_retirement.json')['after_verified_final']
            checkpoints=[json.loads(x) for x in (folder/'checkpoints.jsonl').read_text().splitlines()]
            checks.append(dict(kind=kind,seed=seed,steps=2000,actual_samples_matched=True,initial_fusion_sha256=identity['initial_fusion_sha256'],initial_PE_sha256=identity['initial_PE_sha256'],checkpoint_sha256=status['checkpoint_sha256'],checkpoint_bytes=status['bytes'],peak_GiB=status['peak_GiB'],last_segment_log_seconds=lines[-1]['elapsed_seconds'],checkpoint_io_seconds=sum(x['save_seconds'] for x in checkpoints),persistence_seconds=status['persistence_seconds']))
        assert len({r['initial_fusion_sha256'] for r in initial})==1 and len({r['initial_PE_sha256'] for r in initial})==1
        assert all(r['schedule_sha256']==lock['schedule_sha256'][str(seed)] for r in initial)
        first_losses=[json.loads((OUT/f'{kind}_{seed}'/'training.jsonl').read_text().splitlines()[0])['loss'] for kind in KINDS];assert len(set(first_losses))==1
        paired[str(seed)]=dict(identical_initial_fusion=True,identical_initial_PE=True,actual_schedules_matched=True,zero_PE_first_step_losses=first_losses)
    runtime=[];worker_seconds=0.;still_live=[]
    for path in sorted((OUT/'runtime').glob('*.json')):
        r=read(path)
        if not r.get('runtime_session'):continue
        assert r.get('process_exited') and 'seconds' in r
        worker_seconds+=r['seconds'];runtime.append(dict(path=str(path),sha256=sha256(path),**r))
        if Path('/proc',str(r['pid'])).exists():still_live.append(r['pid'])
    assert not still_live and worker_seconds<=8*3600
    b=read(OUT/'budget.json');assert b['formal_updates']==18000 and b['preflight_updates']==6 and b['nontraining_forwards']<=6600
    discarded=sum(read(OUT/j['name']/'discarded_work.json')['updates'] for j in definitions() if (OUT/j['name']/'discarded_work.json').exists())
    assert discarded==b.get('discarded_formal_updates',0)
    write(OUT/'training_attempt_accounting.json',dict(formal_retained_updates=18000,formal_discarded_updates=discarded,formal_actual_attempts=18000+discarded,retained_sample_presentations=36000,discarded_sample_presentations=2*discarded,actual_sample_presentations=36000+2*discarded,pilot_discarded_updates=6,pilot_forward_calls=6,retained_trajectory_limit=18000,hard_GPU_worker_hours=8,reconciliation=str(OUT/'failures/runtime_observation_v3/budget_reconciliation.json'),note='Discarded attempts did not enter final weights. Exact all-state/RNG recovery repeats only the same fixed schedule after the last checkpoint; root explicitly interpreted18000 as retained final trajectories. All actual work remains charged to8worker-hours.'))
    if RAM.exists():
        assert RAM.stat().st_uid==os.getuid()
        if not list(RAM.iterdir()):RAM.rmdir()
    data_bytes=sum(x.stat().st_size for x in CACHE.rglob('*') if x.is_file());project_bytes=sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file());assert data_bytes<26*2**30 and project_bytes<2*2**30
    write(OUT/'matched_checks.json',dict(complete=True,records=checks,paired=paired,all_nine_final_steps=2000,formal_retained_updates=18000,retained_sample_presentations=36000,all_actual_sample_presentations=36000+2*discarded,formal_actual_attempts=18000+discarded,formal_discarded_updates=discarded,source_scene_and_cached_r_reused_read_only=True,P_inactive_coefficients=384,pilot_updates_discarded=6,all_own_resume_files_retired=True,new_confirmation=False))
    accounting=read(OUT/'preflight_accounting.json')
    write(OUT/'cost_description.json',dict(complete=True,training_fits=9,formal_retained_updates=18000,formal_retained_sample_presentations=36000,formal_actual_sample_presentations=36000+2*discarded,formal_actual_attempts=18000+discarded,formal_discarded_updates=discarded,pilot_training_updates=6,pilot_training_forwards=6,formal_new_predictions=6336,inherited_scored_predictions=4224,cost_native_forwards=200,nontraining_budget_counter_including_pilot_training=b['nontraining_forwards'],semantic_nontraining_charged=b['nontraining_forwards']-6,semantic_nontraining_executed_if_no_other_failure=b['nontraining_forwards']-7,preflight_accounting=accounting,GPU_worker_seconds=worker_seconds,GPU_worker_hours=worker_seconds/3600,hard_GPU_worker_hours=8,runtime=runtime,training_records=checks,full_query_cost=str(OUT/'cost.json'),data_bytes_at_check=data_bytes,project_bytes_at_check=project_bytes,data_cap_bytes=26*2**30,project_cap_bytes=2*2**30,no_old_result_cleanup=True,new_downloads=0,source_builds=0,source_cost_reference=str(parent.OUT/'cost_description.json'),checkpoint_io_note='Training elapsed includes preceding checkpoints; do not add checkpoint seconds to worker or training elapsed again. Worker total includes preparation/preflight/loads/recovery/evaluation/cost; resource waiting excluded.'))
    endpoint='development/original/missing/roi/centered_log_mse';flags={};harm_endpoints=[('main',endpoint,['Previous_R','Previous_L']),('raw_roi','development/original/missing/roi/log_mse',['P','Previous_R','Previous_L']),('full_absrel','development/original/full/whole/absrel',['P','Previous_R','Previous_L'])]
    for kind in ['S','G']:
        primary=s['comparisons'][endpoint+'/'+kind+'-P'];harms={}
        for name,key,controls in harm_endpoints:
            for control in controls:
                v=s['comparisons'][key+'/'+kind+'-'+control];harms[name+'/'+control]=dict(supported_material_harm=material_harm(v),relative_error_increase_percent=-v['relative_error_reduction_percent'],interval95_error_reduction_percent=v['interval95_percent'],absolute_difference=v['absolute_difference'])
        time_increase=100*(cost['summary'][kind]['median_ms']/cost['summary']['P']['median_ms']-1)
        flags[kind]=dict(primary_development_screen=screen(primary),primary_relative_error_reduction_percent=primary['relative_error_reduction_percent'],primary_interval95_percent=primary['interval95_percent'],all_three_primary_seeds_positive=all(x>0 for x in primary['per_seed_relative_percent']),supported_material_development_harms=harms,any_supported_material_development_harm=any(v['supported_material_harm'] for v in harms.values()),full_warm_query_increase_over_P_percent=time_increase,within_prespecified_10percent_time_condition=time_increase<=10)
    flags['G']['G_over_S_interval_supported']=s['comparisons'][endpoint+'/G-S']['interval95_percent'][0]>0
    write(OUT/'method_decision.json',dict(complete=True,status='fixed_development_complete_return_to_Astra',empirical_prespecified_flags=flags,rules='Primary PE screen >=1% mean reduction, interval lower>0 and all3seeds positive; material harm >=1% increase plus paired CI support; inherited utility means are soft ranking, training-clip tradeoffs bound scope and never automatic veto. No equivalence/noninferiority claim.',scientific_successor_selected=False,method_adopted=False,new_confirmation=False,overall_goal_achieved=False,automatic_training=False,remaining_analysis='Same Astra task chooses candidate/closure and any substantively different next plan; no additional fitting or mechanism scan started.'))
    with (OUT/'scoring/all_summary_cells.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['cohort','view','condition','region','metric','method','mean','seed20261110','seed20261111','seed20261112','per_place_in_catalog_order'])
        for key,v in s['summaries'].items():writer.writerow(key.split('/')+[v['mean']]+v['per_seed']+[json.dumps(v['per_place'])])
    text=['# Complete fixed parameter-PE tables','','All data are already used. P/S/G retain identical Render content/coverage; Previous_R/Previous_L are inherited matching32-place/2000step references.','','[All metrics/regions/conditions and seed/place cells](scoring/all_summary_cells.csv) · [All query metrics](scoring/metrics.json) · [All paired relative intervals](scoring/summary.json)','','## Absolute fixed endpoints','','|Scope / views|Method|Missing ROI spatial|Missing ROI raw|Missing ROI bias²|Full whole AbsRel|','|---|---|---:|---:|---:|---:|']
    scopes=[('development','original'),('development','heldout'),('development','pooled'),('train_all','original'),('train_all','heldout'),('train_original','heldout'),('train_added','heldout')]
    methods=['Previous_L','Previous_R','P','S','G']
    for cohort,view in scopes:
        for method in methods:
            values=[s['summaries'][f'{cohort}/{view}/{condition}/{region}/{metric}/{method}']['mean'] for condition,region,metric in [('missing','roi','centered_log_mse'),('missing','roi','log_mse'),('missing','roi','bias_squared'),('full','whole','absrel')]]
            text.append('|'+cohort+'/'+view+'|'+method+'|'+'|'.join(f'{v:.9f}' for v in values)+'|')
    text+=['','## Fixed comparisons, missing ROI spatial error','','|Scope / views|Comparison|Absolute candidate−control|Relative reduction %|95% paired interval %|Three seed reductions %|','|---|---|---:|---:|---|---|']
    pairs=['G-P','G-S','S-P','P-Previous_R','S-Previous_R','G-Previous_R','P-Previous_L','S-Previous_L','G-Previous_L']
    for cohort,view in scopes:
        for pair in pairs:
            v=s['comparisons'][f'{cohort}/{view}/missing/roi/centered_log_mse/{pair}'];text.append(f"|{cohort}/{view}|{pair}|{v['absolute_difference']:.9f}|{v['relative_error_reduction_percent']:.4f}|[{v['interval95_percent'][0]:.4f}, {v['interval95_percent'][1]:.4f}]|"+', '.join(f'{x:.4f}' for x in v['per_seed_relative_percent'])+'|')
    text+=['','## Every development place, original queries / missing ROI spatial','','|Place|Inherited Local|Inherited Render|P|S|G|G−P|G−S|S−P|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for i,meta in enumerate(p['catalog']):
        vals={k:s['summaries'][endpoint+'/'+k]['per_place'][i] for k in methods};values=[vals[k] for k in methods]+[vals['G']-vals['P'],vals['G']-vals['S'],vals['S']-vals['P']];text.append('|'+meta['place']+'|'+'|'.join(f'{x:.9f}' for x in values)+'|')
    text+=['','## Complete warm-query cost','','|Method|Median ms|Peak GiB|','|---|---:|---:|']
    for k in ['Render','P','S','G']:v=cost['summary'][k];text.append(f"|{k}|{v['median_ms']:.3f}|{v['peak_GiB']:.3f}|")
    text+=['','Two fixed query conditions,5warmup+20measured each, seed20261110; exactly200 native calls total. Includes encoder, actual original content render, required PE fields/projection and native output. Excludes disk/JPEG/source construction/camera estimation; shared GPU timings are not exclusive deployment performance.','','Bootstrap5000 paired seed/place resamples, seed20261113; equal queries/places; finite3seed/12development-place scope. Auxiliary comparisons remain correlated descriptive evidence. Material-harm rule is >=1% and supported CI; means/clip-generalization tradeoffs are reported, not zero-tolerance gates.']
    (OUT/'report_tables.md').write_text('\n'.join(text)+'\n')
    guard=read(OUT/'resource_sidecar/current.json');assert read(OUT/'resource_sidecar/complete.json')['complete']
    assert guard['H']>=6*2**30 and guard['F']>=5*2**30
    details=read(OUT/'cost_description.json')
    details.update(resource_addendum=dict(path=str(ROOT/'results/gaussian_memory/autonomous_research_loop/round_002/resource_addendum_v1.json'),sha256=sha256(ROOT/'results/gaussian_memory/autonomous_research_loop/round_002/resource_addendum_v1.json'),scientific_plan_and_checkpoint_identities_unchanged=True),resource_sidecar=dict(config=str(OUT/'resource_sidecar/config.json'),admission=str(OUT/'resource_sidecar/admission.json'),ledger=str(OUT/'resource_sidecar/ledger.jsonl'),complete=str(OUT/'resource_sidecar/complete.json'),last=guard),resource_pause=str(OUT/'disk_resource_pause.json'),all_actual_native_forward_calls=36000+2*discarded+6+b['nontraining_forwards']-7,final_data_available_bytes=__import__('shutil').disk_usage(CACHE).free,final_project_available_bytes=__import__('shutil').disk_usage(OUT).free)
    write(OUT/'cost_description.json',details)
    print(dict(finite_checks_complete=True,worker_hours=worker_seconds/3600,data_GiB=data_bytes/2**30,project_GiB=project_bytes/2**30,flags=flags),flush=True)


if __name__=='__main__':main()
