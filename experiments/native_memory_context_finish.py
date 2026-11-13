"""CPU endpoint accounting, complete tables and prespecified empirical flags."""
import csv
import json
import shutil
import numpy as np
from experiments.native_memory_context_data import *
from experiments.native_memory_context_train import verify_lock
from experiments.native_memory_context_queue import definitions,done
from experiments.gaussian_scene_config import schedule

def screen(v):return v['relative_error_reduction_percent']>=1 and v['interval95_percent'][0]>0 and all(x>0 for x in v['per_seed_relative_percent'])
def harm(v):return v['relative_error_reduction_percent']<=-1 and v['interval95_percent'][1]<0

def main():
    p=plan();lock=verify_lock();s=read(OUT/'scoring/summary.json');cost=read(OUT/'cost.json');assert s['complete'] and cost['complete'];checks=[];paired={};memories={}
    for seed in SEEDS:
        order=schedule(p,seed);initial=[]
        for kind in KINDS:
            j=dict(name=f'{kind}_{seed}',kind=kind,seed=seed);assert done(j);folder=OUT/j['name'];status=read(folder/'status.json');identity=read(folder/'initial_identity.json');initial.append(identity);rows=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
            assert [x['step'] for x in rows]==list(range(1,2001));assert [key for x in rows for key in x['samples']]==[list(k) for k in order];assert all(np.isfinite(x['loss']) and np.isfinite(x['gradient_norm']) for x in rows)
            assert status['native_parameters']==637383906 and status['fusion_parameters']==787712 and status['writer_parameters']==408768
            assert checkpoint_path(kind,seed).stat().st_size==status['bytes'] and read(folder/'resume_retirement.json')['after_verified_final'] and not resume_path(kind,seed).exists()
            checks.append(dict(kind=kind,seed=seed,steps=2000,actual_samples_matched=True,identity=identity,checkpoint_sha256=status['checkpoint_sha256'],checkpoint_bytes=status['bytes'],peak_GiB=status['peak_GiB'],last_segment_log_seconds=rows[-1]['elapsed_seconds']))
            for meta in p['training_catalog']+p['catalog']:
                rid=meta['id'];record=read(OUT/'memories'/f'{kind}_{seed}_{rid}.json');assert record['complete'] and memory_path(kind,seed,rid).stat().st_size==record['bytes'];memories[f'{kind}_{seed}_{rid}']=record
            for cohort in ['train','development']:
                rec=read(OUT/'evaluation'/f'{cohort}_{kind}_{seed}.json');assert rec['writer_calls_during_evaluation']==0 and rec['source_RGB_calls_during_evaluation']==0
                assert all(h==memories[f'{kind}_{seed}_{rid}']['sha256'] for rid,h in rec['memory_sha256'].items())
        assert len({i['initial_fusion_sha256'] for i in initial})==1 and len({i['initial_writer_sha256'] for i in initial})==1
        assert all(i['schedule_sha256']==lock['schedule_sha256'][str(seed)] for i in initial)
        losses=[json.loads((OUT/f'{k}_{seed}'/'training.jsonl').read_text().splitlines()[0])['loss'] for k in KINDS];assert len(set(losses))==1;paired[str(seed)]=dict(identical_initial_fusion_writer=True,actual_schedules_matched=True,zero_residual_first_step_losses=losses)
    assert len(memories)==396;write(OUT/'memory_manifest.json',dict(complete=True,states=memories,count=396,source_only=True,one_state_each_model_scene=True,all8queries2conditions_same_identity=True,geometry_fixed_original=True,writer_calls_during_evaluation=0))
    runtime=[];seconds=0.
    for path in sorted((OUT/'runtime').glob('*.json')):
        rec=read(path)
        if not rec.get('runtime_session'):continue
        assert rec['process_exited'] and 'seconds' in rec;seconds+=rec['seconds'];runtime.append(dict(path=str(path),**rec))
    assert seconds<=8*3600
    b=read(OUT/'budget.json');assert b['formal_updates']==18000 and b['preflight_updates']==6 and b['nontraining_forwards']<=6600 and b['independent_writer_forwards']<=600
    discarded=sum(read(OUT/j['name']/'discarded_work.json')['updates'] for j in definitions() if (OUT/j['name']/'discarded_work.json').exists());assert discarded==b.get('discarded_formal_updates',0)
    write(OUT/'matched_checks.json',dict(complete=True,records=checks,paired=paired,all_nine_steps=2000,total_trainable=638580386,writer_parameters=408768,P_inactive_coefficients=256,original_full_shape_content_common=True,original_renderer_weights_coverage_common=True,learned_contents_can_differ=True,query_native_parameter_PE_absent=True,all396_permanent_states_verified_on_save_reload=True,new_confirmation=False))
    write(OUT/'training_attempt_accounting.json',dict(formal_retained_updates=18000,formal_attempted_updates=18000+discarded,formal_discarded_updates=discarded,retained_sample_presentations=36000,actual_formal_samples=36000+discarded*2,pilot_discarded_updates=6,pilot_training_forward_calls=6,training_writer_calls=b['training_writer_forwards'],retained_formal_writer_calls=36000))
    data_bytes=sum(x.stat().st_size for x in CACHE.rglob('*') if x.is_file());project_bytes=sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file());assert data_bytes<52*2**30 and project_bytes<2*2**30
    # Deployment/scoring CPU timing is separate from the GPU worker ledger.
    import ast
    score_seconds=None
    for line in (OUT/'postqueue/score.log').read_text().splitlines():
        try:record=ast.literal_eval(line)
        except (ValueError,SyntaxError):continue
        if isinstance(record,dict) and record.get('scoring_complete'):score_seconds=record['seconds']
    write(OUT/'scoring/runtime.json',dict(complete=True,CPU_score_seconds=score_seconds,GPU_forward_calls=0,source='Completed scoring process printed its own monotonic elapsed seconds.'))
    write(OUT/'cost_description.json',dict(complete=True,GPU_worker_hours=seconds/3600,GPU_worker_seconds=seconds,CPU_scoring_seconds=score_seconds,runtime=runtime,hard_GPU_worker_hours=8,formal_retained_updates=18000,formal_attempted_updates=18000+discarded,formal_discarded_updates=discarded,formal_sample_presentations=36000+2*discarded,pilot_discarded_updates=6,new_predictions=6336,inherited_scored_predictions=4224,persistent_encoded_states=396,nontraining_conservative_counter_including_pilot=b['nontraining_forwards'],semantic_nontraining_native=b['nontraining_forwards']-6,independent_writer_forwards=b['independent_writer_forwards'],formal_training_writer_forwards=b['training_writer_forwards'],cost_native_forwards=200,cost_writer_forwards=150,all_actual_native_forwards=36000+2*discarded+b['nontraining_forwards'],data_bytes=data_bytes,project_bytes_at_check=project_bytes,hard_data_bytes=52*2**30,hard_project_bytes=2*2**30,source_builds=0,new_downloads=0,old_assets_deleted=0,preflight_accounting=read(OUT/'preflight_accounting.json'),checkpoint_IO='Already included in runtime and training elapsed; never add twice.',timing_scope='Includes all GPU workers prep/preflight/loading/formalfit/recovery/export/evaluation/cost. CPU scoring/report and resource wait separate.',source_cost_reference=str(ROOT/'results/gaussian_memory/source_geometry_control1/cost_description.json'),resources=str(OUT/'resources/ledger.jsonl'),final_data_available_bytes=shutil.disk_usage(CACHE).free,final_project_available_bytes=shutil.disk_usage(ROOT).free))
    endpoint='development/original/missing/roi/centered_log_mse';flags={}
    for k in ['S','G']:
        primary=s['comparisons'][endpoint+'/'+k+'-P'];harms={}
        for tag,key,controls in [('main',endpoint,['Previous_R','Previous_L']),('raw','development/original/missing/roi/log_mse',['P','Previous_R','Previous_L']),('full_absrel','development/original/full/whole/absrel',['P','Previous_R','Previous_L'])]:
            for control in controls:
                v=s['comparisons'][key+'/'+k+'-'+control];harms[tag+'/'+control]=dict(supported_material_harm=harm(v),relative_error_reduction_percent=v['relative_error_reduction_percent'],interval95_percent=v['interval95_percent'])
        n8=cost['amortization'][k]['totals']['8']['new_seconds'];p8=cost['amortization']['P']['totals']['8']['new_seconds'];ratio=cost['amortization'][k]['totals']['8']['ratio_to_R']
        flags[k]=dict(primary_development_screen=screen(primary),primary_relative_error_reduction_percent=primary['relative_error_reduction_percent'],primary_interval95_percent=primary['interval95_percent'],three_seeds_positive=all(v>0 for v in primary['per_seed_relative_percent']),supported_material_development_harms=harms,any_supported_material_development_harm=any(v['supported_material_harm'] for v in harms.values()),N8_over_R_increase_percent=100*(ratio-1),N8_over_P_increase_percent=100*(n8/p8-1),N8_cost_over_R_within10percent=ratio<=1.1,N8_cost_over_P_within10percent=n8/p8<=1.1)
    gs=s['comparisons'][endpoint+'/G-S'];flags['G']['G_over_S_shape_supported']=gs['relative_error_reduction_percent']>0 and gs['interval95_percent'][0]>0 and all(v>0 for v in gs['per_seed_relative_percent'])
    write(OUT/'method_decision.json',dict(complete=True,status='fixed_implementation_complete_return_Astra',empirical_prespecified_flags=flags,rules='Primary>=1% relative error reduction, paired95%lower>0 and all3seedspositive; material harm>=1% supported increase. N8 incremental write+query cost >10% practical counterevidence. Uncertainty is not equivalence/noninferiority.',common_writer_gain_not_O6=True,native_query_parameter_PE_tested=False,writer_Transformer_PE_tested=True,scientific_successor_selected=False,method_adopted=False,new_confirmation=False,overall_goal_achieved=False))
    # Retain nonzero historical construction with its actual boundary; no fake
    # RGB-end-to-end or per-fidelity timing is inferred from old training cases.
    src=read(ROOT/'results/gaussian_memory/source_geometry_control1/cost_description.json');legacy=float(np.mean(src['modes']['sensor']['prepared_source_to_field_seconds']));ext=read(parent.OUT/'cost_description.json')['source_construction'];extension=sum(ext[k] for k in ['source_prediction_worker_seconds','source_fit_worker_seconds','binding_encoding_render_cache_worker_seconds'])/20
    totals={}
    for kind in ['Render','P','S','G']:
        tq=cost['summary'][kind]['median_ms']/1000;tw=0 if kind=='Render' else cost['amortization'][kind]['write_seconds'];totals[kind]={str(n):dict(T_source_reference=legacy,T_new_write=tw,T_queries=n*tq,total=legacy+tw+n*tq,extension_pipeline_reference_total=extension+tw+n*tq) for n in [1,4,8,32]}
    write(OUT/'cost_amortized_totals.json',dict(complete=True,legacy_prepared_source_seconds_reference=legacy,legacy_scope=src['construction_scope'],legacy_reference_cases='Two prior training source cases, not individual fidelity_00 timing; excludes AnySplat generation/sensor acquisition/raw preprocessing.',extension_seconds_reference=extension,extension_scope='Prior20additional scenes aggregate prediction+fit+binding+rendercache /20; contains cache prep, separately labeled, not directly interchangeable with legacy source timing.',totals=totals,source_cost_not_zero=True,cold_model_load_excluded=True,serialization_reload_separately=cost['writer_records'],N32_is_amortization_illustration_not32observedqueries=True))
    with (OUT/'scoring/all_summary_cells.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['cohort','view','condition','region','metric','method','mean','seed20261110','seed20261111','seed20261112','per_place'])
        for key,v in s['summaries'].items():w.writerow(key.split('/')+[v['mean']]+v['per_seed']+[json.dumps(v['per_place'])])
    lines=['# 完整固定数表','','全部旧数据；共同writer用途不算PE增量。原R/Local为继承匹配32地点/2000步参照。','','[全部细胞](scoring/all_summary_cells.csv) · [全查询](scoring/metrics.json) · [全部区间](scoring/summary.json)','','|范围/视角|方法|Missing ROI spatial|Missing ROI raw|Missing ROI bias²|Full whole AbsRel|','|---|---|---:|---:|---:|---:|'];scopes=[('development','original'),('development','heldout'),('development','pooled'),('train_all','original'),('train_all','heldout'),('train_original','heldout'),('train_added','heldout')]
    for cohort,view in scopes:
        for k in ['Previous_L','Previous_R','P','S','G']:
            vals=[s['summaries'][f'{cohort}/{view}/{ep}/{k}']['mean'] for ep in ['missing/roi/centered_log_mse','missing/roi/log_mse','missing/roi/bias_squared','full/whole/absrel']];lines.append('|'+cohort+'/'+view+'|'+k+'|'+'|'.join(f'{v:.9f}' for v in vals)+'|')
    lines+=['','|范围/视角|比较|绝对candidate−control|相对降误差%|配对95%区间%|三种子%|','|---|---|---:|---:|---|---|']
    for cohort,view in scopes:
        for pair in ['G-P','G-S','S-P','P-Previous_R','S-Previous_R','G-Previous_R','P-Previous_L','S-Previous_L','G-Previous_L']:
            v=s['comparisons'][f'{cohort}/{view}/missing/roi/centered_log_mse/{pair}'];lines.append(f"|{cohort}/{view}|{pair}|{v['absolute_difference']:.9f}|{v['relative_error_reduction_percent']:+.4f}|{v['interval95_percent']}|{v['per_seed_relative_percent']}|")
    lines+=['','5000配对seed/place重采样；3种子/12开发地点有限范围，无fresh确认。全部region/条件/支持分层/raw/bias/full输出/逐地点和query见完整CSV/JSON。','','|方法|完整暖query中位ms|峰值GiB|N8增量总成本/原R|','|---|---:|---:|---:|']
    for k in ['Render','P','S','G']:lines.append(f"|{k}|{cost['summary'][k]['median_ms']:.3f}|{cost['summary'][k]['peak_GiB']:.3f}|{1 if k=='Render' else cost['amortization'][k]['totals']['8']['ratio_to_R']:.4f}|")
    lines+=['','|方法|N|一次图/关系+writer及查询s|相对N次原R|含非零继承来源构建参考s|','|---|---:|---:|---:|---:|']
    for k in ['P','S','G']:
        for n in [1,4,8,32]:v=cost['amortization'][k]['totals'][str(n)];lines.append(f"|{k}|{n}|{v['new_seconds']:.6f}|{v['ratio_to_R']:.6f}|{totals[k][str(n)]['total']:.6f}|")
    lines+=['','来源参考与排除项见[cost_amortized_totals](cost_amortized_totals.json)；一次图/关系、writer、序列化/重载分开记录，来源成本未设零。共享卡计时不支持独占/跨轮速度结论。']
    (OUT/'report_tables.md').write_text('\n'.join(lines)+'\n');print(dict(finite_checks_complete=True,worker_hours=seconds/3600,data_GiB=data_bytes/2**30,flags=flags),flush=True)
if __name__=='__main__':main()
