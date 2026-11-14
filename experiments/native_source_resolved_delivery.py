"""Round028 delivery from terminal workers and exported evidence; no model calls."""
import json,collections,time,resource
from pathlib import Path
from experiments import native_source_resolved_data as d
from experiments.native_track_memory_delivery import fmt,estimate,table

def ledger():
    workers=[d.read(p) for p in sorted((d.OUT/'workers').glob('*.json'))]
    assert all(w.get('actual_wait_returned') and w.get('exit_code') is not None for w in workers)
    events=[dict(worker=p.stem,**json.loads(l)) for p in sorted((d.OUT/'calls').glob('*.jsonl')) for l in p.read_text().splitlines()]
    counts=collections.Counter(e['kind'] for e in events);updates=[e for e in events if e['kind']=='formal_update'];unique={(e['arm'],e['seed'],e['step']) for e in updates};byfit=collections.Counter((e['arm'],e['seed']) for e in updates)
    assert len(unique)==len(updates)==15000 and len(byfit)==15 and all(n==1000 for n in byfit.values())
    assert counts['evaluation_query']==21120 and len(list((d.OUT/'completed').glob('*.json')))==21120
    z=[e for e in updates if not e['GT_points']];zg=[e for e in updates if e['GT_points'] and e['loss']==0]
    unsup=dict(zero_GT_batches=len(z),per_fit={f'{k}_{s}':sum(e['arm']==k and e['seed']==s for e in z) for k in d.KINDS for s in d.SEEDS},all_zero_GT_loss_and_gradient_norm_zero=all(e['loss']==0 and e['gradient_norm']==0 for e in z),zero_loss_with_positive_GT=len(zg),interpretation='Fixed no-GT batches retain graph-connected zero supervision and the original AdamW momentum/decay step; no resampling and no zero-track-error claim')
    d.write(d.OUT/'unsupervised_training_batches.json',dict(summary=unsup,all_zero_GT_records=z,all_zero_loss_positive_GT_records=zg))
    initials={}
    for seed in d.SEEDS:
        ii=[d.read(d.OUT/'fits'/f'{k}_{seed}_initial.json') for k in d.KINDS];assert len({v['initial_sha256'] for v in ii})==len({v['schedule_sha256'] for v in ii})==1
        initials[str(seed)]=dict(initial_sha256=ii[0]['initial_sha256'],schedule_sha256=ii[0]['schedule_sha256'],all_five_exact=True,parameters=ii[0]['parameters'])
    finals=[d.read(p) for p in sorted((d.OUT/'fits').glob('*_complete.json'))];assert len(finals)==15
    assert len(list((d.OUT/'evaluation').glob('*.json')))==15
    fieldstats=[]
    for item in d.read(d.OUT/'input_identities.json')['scenes']:
        p=Path(item['field_path']);v=item['field_published_identity'];st=p.stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'];fieldstats.append(dict(id=item['id'],path=str(p),bytes_and_mtime_unchanged=True,inherited_sha256=v['sha256_inherited']))
    d.write(d.OUT/'source_field_readonly_verification.json',dict(complete=True,all44unchanged=True,fields=fieldstats,old_large_hash_not_recomputed=True,scope='Published byte size/mtime plus readonly code and inherited identities'))
    source_features={v['id']:v['source_features_path'] for v in d.read(d.OUT/'input_identities.json')['scenes']}
    for index in (d.OUT/'public_states').glob('*.json'):
        v=d.read(index)
        # IDs contain underscores; recover through the explicit field filename.
        state_name=Path(v['field_reference']).stem;rid=state_name.rsplit('_',1)[0];v.update(native_anchor_source_DINO_reference=source_features[rid],native_anchor_prefix_DPT_are_ephemeral_not_new_serialized_state=True);d.write(index,v)
    deferral_names={p.stem for p in (d.OUT/'admission_deferrals').glob('*.json')}
    resources=[d.read(p) for p in (d.OUT/'worker_resources').glob('*.json')];cost=d.read(d.OUT/'actual_cost.json');stages=[e for e in events if e['kind']=='cost_stage'];pre=d.read(d.OUT/'preflight.json')
    calls=dict(global_source_only_PCA=counts['global_source_only_PCA'],source_alpha_raster_batches=counts['actual_source_alpha_raster'],source_alpha_raster_views=4*counts['actual_source_alpha_raster'],source_bindings=counts['source_binding'],formal_train_queries=counts['formal_update'],formal_backwards=counts['formal_update'],formal_Adam_updates=counts['formal_update'],formal_evaluation_queries=counts['evaluation_query'],preflight_queries=pre['extra_native_queries'],preflight_backwards=pre['extra_backwards'],discarded_preflight_Adam_updates=pre['restore']['discarded_preflight_updates'],frozen_anchor_reconstructions=counts['frozen_anchor_reconstruction'],cost_initial_native_queries=sum(e['stage']=='initial_native_query' for e in stages),cost_adapted_queries=counts['cost_adapted_query'],cost_fresh_DINO=cost['actual_calls']['fresh_DINO'],cost_native_source_frames=cost['actual_calls']['native_anchor_or_source_frames'],cost_GS_rasters=cost['actual_calls']['GS_raster'],new_Native_reference_predictions=0,old_predictions_or_metrics_rerun=0,activation_checkpoint_recomputation='All24 native frame/global pairs are checkpointed in each backward; recomputation and cache misses included in worker wallclock')
    calls['full_native_query_chains']=calls['formal_train_queries']+calls['formal_evaluation_queries']+calls['preflight_queries']+calls['cost_initial_native_queries']+calls['cost_adapted_queries'];calls['official_TrackHead_calls_4iterations_each']=calls['full_native_query_chains']-calls['cost_initial_native_queries'];calls['full_native_query_chains_scope']='Each chain means one complete24pair query pass; train/eval reuse immutable original initializer geometry and run the actual adapted native/DPT/Track path. End-to-end deployment cost includes a separate initial24pair camera pass, measured explicitly, not hidden as a cached full query.'
    data=sum(p.stat().st_size for p in d.CACHE.rglob('*') if p.is_file());sources=[*d.ROOT.glob('experiments/native_source_resolved*.py'),d.ROOT/'models/native_source_resolved_memory.py'];project=sum(p.stat().st_size for p in d.OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in sources);hours=sum(w['elapsed_seconds'] for w in workers)/3600
    result=dict(complete=True,workers=workers,worker_count=len(workers),all_owned_GPU_children_actual_waited=True,GPU_worker_hours=hours,GPU_worker_hour_cap=18,worker_lifetime_includes_loading_IO_failures_recovery_PCA_binding_training_evaluation_cost=True,DATA_final_bytes=data,DATA_peak_conservative_bytes=data+256*2**20,project_final_bytes=project,project_peak_conservative_bytes=project+64*2**20,peak_scope='Concurrent complete Adam latest/final and atomic temporary checkpoints bounded by256MiB reserve; no owned RAM filesystem or new source-prefix files; project64MiB export/manifest reserve',full_Adam_checkpoint_max_bytes=max(Path(x['final']).stat().st_size for x in finals),peak_GPU_allocated_bytes=max([x.get('peak_GPU_allocated_bytes',0) for x in resources]+[cost['peak_GPU_allocated_bytes']]),peak_RSS_single_worker_bytes=max([x.get('peak_RSS_bytes',0) for x in resources]+[cost['peak_RSS_bytes']]),worker_resource_records=resources,failed_worker_peaks_unknown=True,pre_CUDA_admission_deferral_count=len(list((d.OUT/'admission_deferrals').glob('*.json'))),pre_CUDA_deferral_GPU_calls=0,pre_CUDA_deferral_pool_wall_conservatively_charged=True,failed_GPU_worker_count=sum(w['exit_code'] not in (0,75) for w in workers),checkpoint_or_yield_exit75_count=sum(w['exit_code']==75 and w['name'] not in deferral_names for w in workers),total_exit75_count=sum(w['exit_code']==75 for w in workers),startup_CPU_failure_GPU_seconds=0,call_counts=calls,event_counts=dict(counts),paired_initial_and_sampler_identities=initials,finals=finals,formal_journal_unique_steps=len(unique),formal_scientific_steps_repeated_or_discarded=0,new_downloads=0,new_physical_GS_fits=0,new_source_content_states=88)
    if (d.OUT/'resource_observations.jsonl').exists():
        observations=[json.loads(l) for l in (d.OUT/'resource_observations.jsonl').read_text().splitlines()]
        cohost=[g['used_MiB'] for v in observations for g in v['GPUs'] if sum(w['GPU']==g['index'] for w in v['active'])>=2]
        result['cohost_peak_sampled_whole_card_used_MiB']=max(cohost) if cohost else None
        result['cohost_peak_scope']='Read-only NVML samples, not a certified unobserved absolute peak; full per-process allocated peaks in worker resources'
    assert hours<=18 and result['DATA_peak_conservative_bytes']<=8*2**30 and result['project_peak_conservative_bytes']<=2**30
    d.write(d.OUT/'runtime_budget_calls.json',result);return result

def living(s):
    p=s['comparisons_pp']['dev12/original'];flags=s['flags'];links='[完整报告](../results/gaussian_memory/native_source_resolved_pe1/report.md) · [实现交接](../results/gaussian_memory/native_source_resolved_pe1/implementation_handoff.json) · [唯一合同](../results/gaussian_memory/autonomous_research_loop/round_028/science_plan.md)'
    result=f'主F/G−F/P {estimate(p["F_G-F_P"])} pp；表示F/P−M/P {estimate(p["F_P-M_P"])} pp；交互 {estimate(s["representation_PE_interaction_pp"]["dev12/original"])} pp。'
    for name in ('research_current.md','research_outline_alignment.md','gaussian_memory_evidence_scope.md'):
        path=d.ROOT/'docs'/name;t=path.read_text();start=t.index('**round028');end=t.index('\n\n',start)
        line=f'**round028来源分解实施交付（2026-11-14）。** 15×1000及21120新预测、旧完整均值P/G、L、Native继承对照与全部辅助／反例／成本完成。{result} F/G开发筛={flags["F_G"]["development_candidate"]}、F/S次要筛={flags["F_S"]["development_candidate"]}；机械标记不替代尾部及成本科学取舍。{links}。仅C2表示实质改变，O2/O3公共粗GS、O4/O5真实多query读取及O6同信息强中心保持；旧关闭与正负证据不改，全部used、无fresh或总体成功，交原Astra评价。'
        t=t[:start]+line+t[end:]
        if name=='research_outline_alignment.md':
            t=t.replace('实施中，尚无效果结论。','完整结果见[本轮报告](../results/gaussian_memory/native_source_resolved_pe1/report.md)，主PE／表示／交互及净用途分列。')
            t=t.replace('当前PCA／44场绑定及真实梯度恢复检查完成，唯一15fit队列运行；尚无完整效果结果。','当前15fit与全量评价／成本已完成，科学取舍交原Astra；[完整结果](../results/gaussian_memory/native_source_resolved_pe1/report.md)。')
            t=t.replace('[round028来源分解合同](../results/gaussian_memory/autonomous_research_loop/round_028/science_plan.md)，实施中','[round028来源分解结果](../results/gaussian_memory/native_source_resolved_pe1/report.md)，实施完成')
            t=t.replace('同预算F/M表示及同信息P/S/G，旧完整均值／L／Native不削弱；暂无结果。','同预算F/M与同信息PE已完整测量，旧完整均值／L／Native强参照、尾部与全部反例保留；科学取舍待Astra。')
        path.write_text(t)

def main():
    start=time.monotonic();s=d.read(d.OUT/'scoring/summary.json');r=ledger();cost=d.read(d.OUT/'actual_cost.json');pre=d.read(d.OUT/'preflight.json');P=s['comparisons_pp']['dev12/original'];I=s['representation_PE_interaction_pp']['dev12/original'];cov=s['coverage'];view=d.read(d.OUT/'figure_review.json');assert view['actually_viewed'];places=d.read(d.OUT/'scoring/place.json');counter=d.read(d.OUT/'scoring/all_pairwise_counterexamples.json')
    tables=[]
    for cohort in ('dev12','train32','train_original12','train_added20'):
        for qs in ('original','other','all8'):
            k=cohort+'/'+qs;tables.append(f'## {k}\n\n'+table(['Method','PCK8 % [95%]','EPE px','Pair-tail95 px','Failed labelled points'],[[a,estimate(v['PCK8']),fmt(v['EPE_place_mean']),fmt(v['EPE95_place_mean']),v['failed_valid_points']] for a,v in s['absolute'][k].items()]))
            tables.append(table(['A−B','PCK pp [95%]','EPE px [95%]','Pair-tail95 px [95%]'],[[a,estimate(v),estimate(s['comparisons_continuous_pixels'][k][a]['EPE']),estimate(s['comparisons_continuous_pixels'][k][a]['EPE95'])] for a,v in s['comparisons_pp'][k].items()]))
            tables.append('PE interaction pp: '+estimate(s['representation_PE_interaction_pp'][k])+'\n')
    tables.append('## All places and seeds\n\n'+table(['Place','Cohort','Origin','Method','Seed','Queries','PCK8 %','EPE px','Pair-tail95 px','Defined queries/pairs'],[[v['id'],v['cohort'],v['training_origin'],v['kind'],v['seed'],v['query_set'],fmt(v['PCK8']*100 if v['PCK8'] is not None else None),fmt(v['EPE']),fmt(v['EPE95']),f'{v["defined_queries"]}/{v["defined_viewpairs"]}'] for v in places if v['query_set']!='all8']))
    (d.OUT/'report_tables.md').write_text('\n'.join(tables))
    primary=table(['Used-dev original contrast','PCK8 difference pp [paired95%]'],[[k,estimate(v)] for k,v in P.items()]+[['(F_G−F_P)−(M_G−M_P)',estimate(I)]])
    absolute=table(['Method','PCK8 % [paired95%]','EPE px','Pair-tail95 px'],[[k,estimate(v['PCK8']),fmt(v['EPE_place_mean']),fmt(v['EPE95_place_mean'])] for k,v in s['absolute']['dev12/original'].items()])
    ct=table(['Arm','Once/source binding s','Native anchor+calib s','Complete query s','At8queries s/query'],[[x['kind'],fmt(x['once_per_place_source_content_formation_seconds'],4),fmt(x['source_dependency_seconds'],4),fmt(x['fresh_full_query_seconds'],4),fmt(x['amortized'][-1]['seconds_per_query'],4)] for x in cost['records']])
    worst=[]
    for a,b in [('F_G','F_P'),('F_P','M_P'),('F_G','L'),('F_G','P'),('F_G','Native4')]:worst.extend([v for v in counter if v['A']==a and v['B']==b and v['cohort']=='dev12' and v['query_set']=='original'][:3])
    wt=table(['Contrast','Place','Seed','Anchor→query','PCK difference pp','GT points'],[[x['A']+'−'+x['B'],x['id'],x['seed'],f'{x["anchor"]}→{x["query"]}',fmt(x['PCK_difference_pp']),x['valid_GT']] for x in worst]);uns=d.read(d.OUT/'unsupervised_training_batches.json')['summary'];binding=[d.read(p) for p in (d.OUT/'binding').glob('*.json')];b=cost['state_tensor_bytes'];pe=s['comparisons_continuous_pixels']['dev12/original']['F_G-F_P'];flags=s['flags']
    coverage_table=table(['Cohort','Queries','Defined/all pairs','Labelled/candidate points','Zero-GT pairs'],[[v['cohort'],v['queries'],f'{v["defined_pairs"]}/{v["pairs"]}',f'{v["valid_GT_points"]}/{v["candidate_points"]}',v['zero_GT_pairs']] for v in d.read(d.OUT/'scoring/label_coverage_cohorts.json')])
    report=f'''# native_source_resolved_pe1：固定预算的来源分量与原生对应

2026-11-14，round028实现交付。唯一[科学合同](../autonomous_research_loop/round_028/science_plan.md)，SHA256 `{d.SCIENCE_SHA}`。15 fits×1000更新、21120新末步预测完成；旧完整均值P/G、当前图像L共12672预测及Native2/4共2816预测／评分原身份继承，没有重训、重跑或新数据。全部数据均used development。

假设是同预算保留逐来源残差比提前均值更适合原生对应，并可能使footprint PE有强中心以外的增量；竞争解释是残差主要为view上下文／遮挡／偏移，损失通道分辨率不值得，或共同普通Σ与中心已足够。固定F/P、F/S、F/G、M/P、M/G及继承强参照，不调秩、权重、来源数、位置层、seed、步数、阈值或数据。

## 主要事实及范围

{primary}
{absolute}
主PE F_G−F_P为{estimate(P['F_G-F_P'])}pp；表示F_P−M_P为{estimate(P['F_P-M_P'])}pp；表示×PE交互为{estimate(I)}pp。原合同开发筛：F/G PE={flags['F_G']['PE_increment']}、对旧P/L/Native4净用途={flags['F_G']['net_over_old_full_mean_P_L_Native4']}、候选={flags['F_G']['development_candidate']}；F/S次要候选={flags['F_S']['development_candidate']}。预定规则是PE均值≥1pp且95%下界>0，并对旧完整P、L、Native4净差区间均为正才形成开发候选；F/S同规则仅属次要，不能挽救G主比较。仅表示或连续误差改善保留内容用途，不替代O6；候选未立时结束此固定预算接法，后续科学取舍仍交原Astra，不追加扫描。标记不替代材料性尾部／成本权衡；没有fresh确认或总体目标达成。

共同内容表示、同表示PE及原生净用途分列。F/M是同信息均值／质量匹配下的学习链总效果，不是固定同一接收器的直接因果干预，也不能唯一归为同物理表面关联；F/G与M/G各自学习。旧P/G/L使用完整1024维内容而非新PCA初值，是保留较强完整信息的研究参照，不冒称严格新初始化配对。Native4没有GS来源sensor/pose特权，净用途只限该条件。

![主结果](figures/primary.png)

主估计每锚/query对有效点先算PCK@8（448px），再锚、query、地点等权；无标签端点undefined，模型/初始化失败保留PCK0、EPE∞。5000次配对交叉重采样地点与种子，seed20261114，共同索引；不把点或view对当独立地点。主连续EPE差{estimate(pe['EPE'])}px、每view对95%尾部差{estimate(pe['EPE95'])}px，负值有利F/G，不能替代主PCK。所有dev/train、原/other/all8、原train12/added20、逐place/seed/query/anchor和尾部见[完整数表](report_tables.md)、[summary](scoring/summary.json)、[viewpairs](scoring/viewpairs.json)、[query](scoring/query.json)、[source](scoring/source.json)、[pooled tails](scoring/pooled_point_tails.json)。other/added不是fresh。

## 实际表示与原生链

原44份2048基元alpha绑定场只读，绝非C/J writer场；来源顺序0/3/6/9。只用32训练地点131072个原始source DINO token计算共同μ与正交PCA255，无whitening/query/标签/秩扫描；保留训练来源方差{d.read(d.OUT/'PCA.json')['retained_variance_fraction']*100:.4f}%。按原4view alpha权重定义总质量m_i、p_iv=m_iv/m_i、z_iv=Σ_u w_ivu Uᵀ(x_vu−μ)/m_i。F=(z_iv,p_iv)四块，M=(p_ivΣ_vz_iv,p_iv)四块，同4×256=1024；原has_content保持，零质量不补来源。88份FP16内容状态均source-only保存、重载和自身小文件身份校验；全局PCA共享常量也计依赖。

1045→1024映射共同初始化四个255列块各为U并相加、bias=μ；四质量列及其余几何列初始0。FP32学习adapter与Adam，原生BF16；数学初值相同，FP16保存／BF16运算可造成有限精度差。固定预检FP32映射F−M max/RMS={pre['initial_mapping_finite_precision']['FP32']}，BF16={pre['initial_mapping_finite_precision']['BF16']}；四次输出坐标差max/RMS={pre['initial_prediction_finite_precision']['max_abs_pixels']:.6f}/{pre['initial_prediction_finite_precision']['RMS_pixels']:.6f}px，loss差{pre['initial_prediction_finite_precision']['loss_difference_F_minus_M']:.8f}。全部44场FP16解码差最大{max(x['FP16_decoded_F_minus_M_max_abs'] for x in binding):.8f}、场级RMS最大{max(x['FP16_decoded_F_minus_M_RMS'] for x in binding):.8f}，逐场在[binding](binding)。实数匹配不等于位级输出相等；未另设数值门禁、未调初值或挑样本。该边界妨碍把F/M差异唯一解释成纯来源残差机制。

每组{next(iter(r['paired_initial_and_sampler_identities'].values()))['parameters']:,}参数：同映射、29→64→1024零输出位置MLP、query首frame/global rank8 QKV/proj LoRA。所有原native24对块、DPT、TrackHead、camera/depth冻结；first existing global读取只读GS，无新增层／reader／反馈。源锚prefix冻结，所有八query／四锚复用相同内容状态及全局adapter。Σ表示原基元支撑而非定位不确定性。普通完整logΣ六通道、opacity、强中心25维、原预测query相机、原2DRoPE／GS连续中心RoPE共享；仅四个K位置槽P强中心density／S同迹球／G完整footprint不同，不给G独占形状，V无位置项，零K/V背景保持。

官方两帧TrackHead真实四次坐标监督，等权SmoothL1 beta8；AdamW1e−4/wd.01/clip5，batch1，三个原seed20261110–12与配对1000sampler，固定step1000。{uns['zero_GT_batches']}个固定batch无有效GT，loss及梯度范数0，仍沿原AdamW动量／decay计step，不重采样、不称零跟踪误差；有效GT而loss0有{uns['zero_loss_with_positive_GT']}个，[全部记录](unsupervised_training_batches.json)。每seed五组初始化及sampler摘要完全相同，正式15000唯一更新无重做／丢弃；预检一次F/P Adam更新明确丢弃。

每个连续评价进程对模型/place内容仅加载一次并供四锚八query复用；恢复时必要的只读重载另计，不重建来源状态，查询不输入形成器、不写场；660份[依赖索引](public_states)引用88共同状态、15最终adapter、原几何和独立native源锚混合状态，而非660份独立场或逐scene参数。训练原128锚缓存与评价8锚LRU只保留不可变原生前缀，全部真实cache miss和加载进worker时间，未将cache命中冒称完整query成本。原相机、标签、GS普通字段与PE只读复用，不重新预测或渲染已有1408case。源sensor/pose标定／query预测相机、静态四源、有限标签与冻结范围都是本次实验选择，非用户核心要求。

## 覆盖、零支持和全部反例

原8×8网格64点不因conf/vis删选；source原448最近邻到210、原query210 sensor/pose与共同valid；双向深度容差max(.02m,.02depth)、两cycle≤1px，沿原合同不调整。1408view对共90112点，GT有效{cov['valid_GT']}；零GT对{cov['zero_GT_pairs']}、raster零支持{cov['zero_GS_raster_pairs']}、attention零键{cov['zero_GS_attention_pairs']}；来源标定无效{cov['invalid_source_alignment_pairs']}、非有限预测相机/内参{cov['nonfinite_predicted_camera_intrinsics_pairs']}。原所有零覆盖与fidelity_03稀疏地点权重保持；无GT/零支持不能宣称准确或无用。覆盖／每阶段拒绝和每点残差仍有原case身份，[coverage](scoring/coverage.json)和[条件支持分层](scoring/support_strata.json)不替代全体端点。

{coverage_table}

该覆盖表只连接不变的原label清单，不重复旧像素指标计算。

![覆盖](figures/coverage.png)

{wt}
全部方向的view对差与最大反例未截断地保留于[清单](scoring/all_pairwise_counterexamples.json)。图包括固定首开发例及最大开发F/G−F/P、F/P−M/P、F/G−L反例；无图像排除或视觉调阈值。sensor投影、遮挡与场景边界不等于独立真实物理轨迹。

![坐标与反例](figures/tracking_cases.png)
![全部训练轨迹](figures/training.png)

图已实际查看，具体观察见[figure_review](figure_review.json)。新21120预测由独立NumPy计算、Torch FP64复核PCK/EPE/median/tail／失败／无标签，最大绝对差{s['independent_NumPy_Torch_max_difference']:.12g}，覆盖{s['independent_NumPy_Torch_records']}记录；旧12672训练参照与2816Native评分直接连接原独立审计结果，没有重算旧像素指标。

## 来源、完整query与摊销成本

{ct}
全局source-only PCA实际{cost['shared_global_PCA_seconds']:.6f}s，44场绑定合计{cost['binding_all44_seconds']:.6f}s，均已计本轮GPUworker。逐场绑定定时含原场和来源特征读取、真实4view alpha raster、投影、F/M两份写盘／重载及数值披露；保守将两表示共同形成时间各计一次适用部署。绑定从既有四来源DINO token起点；其历史编码依赖由旧prepared-source formation中的fresh frozen encoder/lifting范围保留，本轮没有为44场重复编码。若部署没有token cache，必须支付四来源编码；表中binding秒数不能代替原始RGB端到端形成。不是反复benchmark。原GS physical formation没有重跑，也不免费：其旧来源预测／拟合／绑定warm计时范围与未知采集、预处理、冷加载在[cost_inherited](cost_inherited.json)原样保留，不能用小warm数充作全端到端形成。

唯一成本案例fit00/a0/q2、首seed，从既有448 RGB张量重新DINO编码，真实source native前缀＋DPT＋camera/depth/sensor标定；query重新编码、初始完整native query＋camera、真实GS raster和字段、第二完整带记忆native query＋DPT＋TrackHead。2次DINO、1source frame、1初始化query、5适配query／5四迭代TrackHead、1raster；无新增Native2/4/P/G/L成本调用。共享阶段执行一次并向各适用pipeline计账；F/M状态各一次加载、PCA／binding／源prefix／query／模型与adapter冷加载单列。1/4/8摊销包括每场一次binding及载入、native anchor形成，训练共同PCA全局开销单列，原GS形成另列，绝不当作免费。因此表中摊销是已存在原场／来源token条件下的新形成加混合锚与完整query成本，不是完整raw RGB→状态→query的总延迟。

固定顺序F/P,F/S,F/G,M/P,M/G，首次TrackHead/kernel初始化与warm缓存不能单独辨识；只一个案例，不推断稳定态速度排序或独立cold-first-GS延迟。成本卡准入独占、其他卡现场见[cost_dispatch](cost_dispatch.json)；不是服务器隔离benchmark。原raw图像decode/resize、sensor获取不在448张量起点内；这里的重新编码不是fresh研究数据。旧Native2/4/P/G/L成本只作为带原范围的历史参考，不是与新五组严格同时的延迟比较。

单锚依赖逻辑KV {b['single_anchor_native_KV']/2**20:.3f}MiB、实际底层QKV backing {b['single_anchor_KV_unique_storage_including_strided_QKV_base']/2**20:.3f}MiB，DPT {b['single_anchor_Track_DPT']/2**20:.3f}MiB，source camera KV backing {b['source_camera_KV_unique_storage']/2**20:.6f}MiB与公制对齐常量{b['source_metric_alignment']}bytes也必须保留；F/M内容每份约{b['source_resolved_F_content']/2**20:.3f}MiB，全局PCA {b['shared_global_PCA']/2**20:.3f}MiB，原几何／源中心依赖{(b['original_GS_field']+b['original_source_GEOMETRY_fields'])/2**20:.3f}MiB。原完整均值f在新成本read中去除，不作模型输入；存量原场文件仍含该未使用均值，不能把其几何张量字节当作共享文件物理体积。以上计数是一锚形成后的运行态张量，不是新落盘的纯GS历史：native prefix/DPT实际由只读source-DINO cache重建，未写新的prefix文件；660依赖索引补全该源cache路径。[actual_cost](actual_cost.json)另列原source-DINO/RGB/sensor/场文件的实际路径和字节。已有重建输入的存储／形成不免费。全局adapter参数约{b['global_adapter_FP32_parameters']/2**20:.3f}MiB，完整Adam是训练恢复依赖而非部署必要项；全局native权重和query临时激活另列，不能由4MiB内容文件称整个混合历史已经压缩成功。

## 实现、恢复、实际资源

五个真实预检query/backward的source/位置输出／两LoRA梯度有限非零，native参数与源prefix只读；一次F/P完整参数＋Adam＋CPU/CUDA RNG＋1000sampler精确恢复。[preflight](preflight.json)记录所有实际梯度和有限精度差，未重复旧原生禁用记忆／空支持检查，引用其不变接口的[原验证](../native_track_memory_pe1/preflight.json)。最初CPU在受限文件系统写DATA失败exit1、0GPU/模型调用；随后使用已授权host环境正常构建，记录[startup001](engineering_revisions/startup001.json)。GPU4随后出现外部高利用率任务，F/P评价保存34marker、M/P训练保存完整step8后actualwait75。旧FIFO同卡补位逻辑被队首待恢复评价阻住；版本化修复为按每卡选择兼容pending项，以及同一训一评容量的双向补位。仅SIGTERM自有manager114384，旧owner22661 actualexit75、所有child实际wait；F/S已133评价、F/G204评价、M/Gstep139一并保存。新唯一manager89284/owner24252按完整状态只补缺失；恢复时GPU6也已有外部任务，未当空卡使用。M/P从9继续、F/P从第35个未提交case继续，不重做有效训练／已提交预测；所有加载、恢复与争用均进worker账。[head_of_line002](engineering_revisions/head_of_line002.json)保留代码前版本／身份和实际退出。GPU4两次短空隙又使恢复任务仅前进到M/G140、141就礼让，所有冷载计入预算。随后不再刷新正常6/7任务，增加CPU-only guard持续观测GPU4，外部活动后稳定空闲120秒才在未来worker导入torch／CUDA／大模型前放行；冷却内CPU快速exit75、0模型调用记录保留，pool墙钟仍保守计入预算，详见[admission_guard003](engineering_revisions/admission_guard003.json)。已知自有正常阶段切换不机械重置等待；新外部活动才重置该卡。guard没有作业派发权，fits完成自动终止。后来实际发现短外部PID可在5秒观察间隔内出现后消失，未来worker的pre-CUDA准入遂同时读取同卡external_PID_arrived的actualwait收据，以最近退出为120秒冷却下界，覆盖GPU7同类事件；未重启manager89284或中断正常worker，[foreign_receipt007](engineering_revisions/foreign_receipt007.json)保留代码版本。一次16.7秒只读利用率观察中，GPU6连续5–84%、均值26.25%，两worker各用约1.54／1.68个CPU核、物理磁盘读取0、总写入约200KiB；短时低util包含主机发射／同步与小算子突发，未用额外模型或benchmark细分其占比，也未发现足以安全改变live数值路径的明显优化，见[utilization006](engineering_revisions/utilization006_live_observation.json)。其余普通失败与礼让以[worker](workers)、[工程记录](engineering_revisions)实际发生为准，不把失败预检记成功。

唯一动态队列按实时CUDA/PID、显存／RAM／空间准入所有安全空卡，无默认双卡上限；foreign PID到达只礼让本轮进程，在当前有效更新／case原子提交后保存完整checkpoint并actualwait，永久final/已提交marker不重复。一个已final的约5.7GB evaluator可与独立待训约29.4GB fit在有余量的自有安全卡共存，两份训练缓存不共卡；不以余显存冒称他人高利用率卡空闲。所有外部工作和旧结果保持，旧GLIBC不动、无系统安装／下载。

总GPUworker **{r['GPU_worker_hours']:.6f}h /18h**，含加载、PCA／绑定、失败、预检、恢复、缓存重建、完整训练／评价／成本，所有{r['worker_count']}子worker实际wait返回。新DATA最终{r['DATA_final_bytes']/2**30:.4f}GiB、含并发原子暂存保守峰{r['DATA_peak_conservative_bytes']/2**30:.4f}GiB /8GiB；项目含源码最终{r['project_final_bytes']/2**30:.4f}GiB、保守峰{r['project_peak_conservative_bytes']/2**30:.4f}GiB /1GiB。完整Adam checkpoint最大{r['full_Adam_checkpoint_max_bytes']/2**20:.3f}MiB，15永久final保留，无自有RAM文件系统暂存或隐藏副本。失败worker未采全峰值明记unknown。[完整调用／资源账](runtime_budget_calls.json)含PID/卡/实际wait、checkpoint反传重计算、恢复和唯一更新；CPU评分／图／交付实际资源另列。最终终端退出确认见[actual_terminal_confirmation](actual_terminal_confirmation.json)。

## 大纲与交回边界

O2/O3公共粗GS、O4/O5真实多query读取和O6同信息强中心比较保持；仅C2逐来源表示实质改变，C1/C3/C4/C5/C6匹配复用。PCA255、2048×1024、静态四源、混合原生锚前缀、来源sensor/pose特权及旧有限标签是实施选择。round027旧完整均值家族关闭、历史区域内容K证据／R替换未决、depth/pose/writer/address等正负与反例不改；本轮不证明准确物理Σ、纯GS替代历史、一般PE有效／无效、fresh或总体成功。只有本合同事实交主→原Astra作集中科学取舍，不自选后继或追加训练。
'''
    (d.OUT/'report.md').write_text(report);living(s)
    d.write(d.OUT/'CPU_delivery_runtime.json',dict(complete=True,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_model_calls=0))
    handoff=dict(round=28,experiment='native_source_resolved_pe1',status='completed_ready_for_original_Astra_assessment',authoritative_science_plan=dict(path=str(d.SCIENCE),sha256=d.SCIENCE_SHA),report=dict(path=str(d.OUT/'report.md'),sha256=d.sha(d.OUT/'report.md')),artifacts=str(d.OUT/'artifact_manifest.json'),summary=str(d.OUT/'scoring/summary.json'),tables=str(d.OUT/'report_tables.md'),runtime_budget=str(d.OUT/'runtime_budget_calls.json'),actual_cost=str(d.OUT/'actual_cost.json'),figure_review=str(d.OUT/'figure_review.json'),actual=dict(fits=15,formal_updates=15000,new_predictions=21120,inherited_trained_references=12672,inherited_native_references=2816,GPU_worker_hours=r['GPU_worker_hours'],all_GPU_children_actual_waited=True),primary_FG_minus_FP=P['F_G-F_P'],representation_FP_minus_MP=P['F_P-M_P'],PE_interaction=I,development_flags=flags,unresolved_material_issues=[],limits=['All used development, no fresh confirmation','FP16/BF16 F/M initial rounding disclosed: fixed preflight coordinate max4.655px/RMS1.051px, no bitwise initial-output claim','Inherited old full mean P/G/L are stronger full-content references, not new PCA initialization-paired controls','Sensor/pose labels and source privilege, sparse/zero coverage retained','Mixed native source anchor plus GS, no pure GS history replacement claim','One fixed-order prepared448-RGB cost case; old reference cost timings inherited, no matched steady-state benchmark'],roles=dict(task='/root/research_implementation_sol_recovery',original_creation='gpt-6.1-sol/high',backend_effective_unknown=True,Fast_requested_or_verified=False,main_goal_state_events_modified=False),no_successor_started=True,scientific_disposition_reserved_for_original_Astra=True,completed_utc=d.now())
    d.write(d.OUT/'implementation_handoff.json',handoff);print(json.dumps(dict(report=handoff['report'],primary=handoff['primary_FG_minus_FP'],GPU_worker_hours=r['GPU_worker_hours'])),flush=True)
if __name__=='__main__':main()
