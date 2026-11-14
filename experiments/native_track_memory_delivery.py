"""Round026 complete fact report, finite call/resource ledger and one handoff."""
import json,collections,math,resource,time
from pathlib import Path
from experiments import native_track_memory_data as d

def fmt(x,n=3):return 'undefined' if x is None else f'{x:.{n}f}'
def estimate(x):
    if x.get('interval95') is None:return fmt(x.get('mean'))+' [undefined]'
    return f'{fmt(x["mean"])} [{fmt(x["interval95"][0])}, {fmt(x["interval95"][1])}]'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def update_living_docs(summary):
    p=summary['comparisons_pp']['dev12/original'];flags=summary['flags'];links='[完整报告](../results/gaussian_memory/native_track_memory_pe1/report.md) · [实现交接](../results/gaussian_memory/native_track_memory_pe1/implementation_handoff.json) · [唯一合同](../results/gaussian_memory/autonomous_research_loop/round_026/science_plan.md)'
    result=f'G−P {estimate(p["G-P"])} pp；G−L {estimate(p["G-L"])} pp、G−Native4 {estimate(p["G-Native4"])} pp。'
    updates={
      'research_current.md':f'**round026原生对应实施交付（2026-11-14）。** 12×1000及16896学习预测／2816固定原生参考完整完成。主开发{result} G开发筛标记={flags["G"]["development_candidate"]}、S次要筛={flags["S"]["development_candidate"]}，机械标记不替代尾部／成本取舍。{links}。原44场只读、同信息强中心、真实预测query相机与混合锚前缀范围保留；全部used，内容K条件阳性／R未决及旧关闭不改，科学取舍交原Astra，无fresh或总体成功主张。',
      'research_outline_alignment.md':f'**round026原生对应实施交付（2026-11-14）。** O2/O3原44场保持，O4/O5多query真实读取与O6同信息比较完成；主G−P {estimate(p["G-P"])} pp。{links}。C3只读接口／C5对应任务／C6有限学习已检验，C1/C2/C4共同信息和来源特权范围保持；尾部、零覆盖、Native／L与成本分列，科学取舍待原Astra。全部used，旧正负和关闭不改，无fresh或总体成功主张。',
      'gaussian_memory_evidence_scope.md':f'**round026原生跟踪结果交付（2026-11-14）。** {result} G开发筛标记={flags["G"]["development_candidate"]}、S次要筛={flags["S"]["development_candidate"]}；共同适配／内容、PE增量与Native净用途分列。{links}。该任务含来源sensor/pose特权、混合native锚前缀和有限sensor标签覆盖，全部used；K区域信号／R替换未决及历史反例、旧关闭继续有效。实现完整不等于独立确认或总体目标完成，科学取舍交原Astra。'}
    for name,line in updates.items():
        path=d.ROOT/'docs'/name;text=path.read_text();start=text.index('**round026');end=text.index('\n\n',start);text=text[:start]+line+text[end:]
        if name=='research_outline_alignment.md':
            text=text.replace('主G−P与原生净用途均待实施，非fresh。',f'已完整实测：G−P {estimate(p["G-P"])} pp，Native净用途和fresh分列；取舍待原Astra。')
            text=text.replace('新任务待实施，不续训旧家族，不自动加步数或模块。','12×1000及固定末步已完成，科学取舍待原Astra；不续训旧家族或自动扩展。')
            text=text.replace('已锁定待实施：实际预测query相机、原生TrackHead，同信息P/S/G及L／Native参考。',f'已完成12×1000及全量评价：[报告](../results/gaussian_memory/native_track_memory_pe1/report.md)；G−P {estimate(p["G-P"])} pp，L／Native、连续误差与尾部保留。')
            text=text.replace('结束跨view内容测量并只选择native_track_memory_pe1；','结束跨view内容测量并已完成[native_track_memory_pe1](../results/gaussian_memory/native_track_memory_pe1/report.md)；')
        path.write_text(text)
def ledger():
    workers=[d.read(p) for p in sorted((d.OUT/'workers').glob('*.json'))];assert all(w.get('actual_wait_returned') for w in workers);assert all(w.get('exit_code') is not None for w in workers)
    events=[]
    for p in sorted((d.OUT/'calls').glob('*.jsonl')):
        for line in p.read_text().splitlines():events.append(dict(worker=p.stem,**json.loads(line)))
    counts=collections.Counter(e['kind'] for e in events);updates=[e for e in events if e['kind']=='formal_update'];byfit=collections.Counter((e['arm'],e['seed']) for e in updates);assert len(byfit)==12 and all(v==1000 for v in byfit.values())
    keys=[(e['arm'],e['seed'],e['step']) for e in updates];assert len(set(keys))==12000
    zero_GT=[e for e in updates if e['GT_points']==0];zero_with_GT=[e for e in updates if e['GT_points']>0 and e['loss']==0]
    unsupervised=dict(zero_GT_batches=len(zero_GT),per_fit={f'{k}_{seed}':sum(e['arm']==k and e['seed']==seed for e in zero_GT) for k in d.KINDS for seed in d.SEEDS},all_zero_GT_loss_and_gradient_norm_zero=all(e['loss']==0 and e['gradient_norm']==0 for e in zero_GT),zero_loss_with_positive_GT=len(zero_with_GT),interpretation='No-valid-GT batches use graph-connected zero supervision, no resampling; existing AdamW step retains momentum/decay and counts in fixed1000 updates. Zero loss here is not zero track error.')
    d.write(d.OUT/'unsupervised_training_batches.json',dict(summary=unsupervised,all_zero_GT_records=zero_GT,all_zero_loss_positive_GT_records=zero_with_GT))
    finals=[d.read(p) for p in sorted((d.OUT/'fits').glob('*_complete.json'))];assert len(finals)==12
    paired_initials={}
    for seed in d.SEEDS:
        ii=[d.read(d.OUT/'fits'/f'{kind}_{seed}_initial.json') for kind in d.KINDS]
        assert len({x['initial_sha256'] for x in ii})==1 and len({x['schedule_sha256'] for x in ii})==1
        paired_initials[str(seed)]=dict(initial_sha256=ii[0]['initial_sha256'],sampler_sha256=ii[0]['schedule_sha256'],all_four_arms_exact=True)
    evals=[d.read(p) for p in (d.OUT/'evaluation').glob('*.json')];assert len(evals)==12 and all(e['cases']==1408 for e in evals)
    resources=[d.read(p) for p in (d.OUT/'worker_resources').glob('*.json')];data=sum(p.stat().st_size for p in d.CACHE.rglob('*') if p.is_file());project=sum(p.stat().st_size for p in d.OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in d.ROOT.glob('experiments/native_track_memory*.py'))+(d.ROOT/'models/native_track_memory.py').stat().st_size
    field_stat=[]
    for old in d.read(d.OUT/'input_identities.json')['scenes']:
        stat=Path(old['field_path']).stat();identity=old['field_published_identity'];ok=stat.st_size==identity['bytes'] and stat.st_mtime_ns==identity['mtime_ns'];assert ok
        field_stat.append(dict(id=old['id'],path=old['field_path'],bytes_and_mtime_unchanged=ok,old_large_hash_not_recomputed=True))
    d.write(d.OUT/'source_field_readonly_verification.json',dict(complete=True,fields=field_stat,all44unchanged=True,scope='Byte size and mtime plus read-only code path; inherited published hashes retained, no new large-weight hash scan'))
    # Normalize metadata after every evaluator has exited. L never consumed a GS;
    # early generic indices included an unused GS reference, which is not an input.
    for kind in d.KINDS:
        for seed in d.SEEDS:
            for meta in d.metas():
                p=d.OUT/'public_states'/f'{kind}_{seed}_{meta["id"]}.json';v=d.read(p);v['GS_field_consumed']=kind!='L'
                if kind=='L':v.update(field_reference=None,geometry_reference=None,representation='Persistent dependencies: source-only native anchor prefix and global adapter. Current-query DINO/unit-ray fields are ephemeral, no GS consumed.')
                d.write(p,v)
    full_ckpt=max(Path(r['final']).stat().st_size for r in finals);case=max(p.stat().st_size for p in (d.CACHE/'cases').rglob('*.pt'))
    cost=d.read(d.OUT/'actual_cost.json');before_cost={e['worker'] for e in events if e['kind']=='cost_stage'}
    # Each named query event is one genuine full query chain. Activation-checkpoint
    # recomputation during backward is additional block execution, stated separately.
    cost_stages=[e for e in events if e['kind']=='cost_stage'];cost_queries=sum(e['stage'] in ('initial_native_query','Native4_native_query') or e['stage'].endswith('_second_native_query_with_GS_read') for e in cost_stages)
    modelcalls=dict(preparation_initializer_native_queries=counts['query_native_initializer_and_Native2'],preparation_Native4_queries=counts['Native4_query_and_track'],formal_train_queries=counts['formal_update'],formal_evaluation_queries=counts['evaluation_query'],preflight_extra_queries=6,cost_queries=cost_queries,frozen_anchor_reconstructions_in_fits_or_preflight=counts['frozen_anchor_reconstruction'],preparation_source_anchors=counts['anchor_native_prefix'],preparation_extra_Native4_sources=counts['Native4_additional_source'],cost_native_source_frames=sum(e['stage']=='anchor_native_prefix' or (e['stage'].startswith('Native4_source') and e['stage'].endswith('_native_prefix')) for e in cost_stages),formal_backwards=12000,preflight_backwards=4,discarded_preflight_Adam_updates=1,formal_Adam_updates=12000,actual_GS_raster_completed=counts['actual_GS_raster']+3+sum(e['stage']=='fresh_GS_raster_and_fields' for e in cost_stages),failed_GS_raster_before_kernel=3,cost_fresh_DINO_encodings=sum('fresh_DINO' in e['stage'] or e['stage'] in ('fresh_anchor_DINO','fresh_query_DINO') for e in cost_stages),cost_stage_attempts=counts['cost_stage_attempt'],cost_completed_stages=len(cost_stages),training_and_preparation_raw_DINO_encodings=0,cached_DINO_is_disclosed=True,activation_checkpoint_native_pair_recomputations='24 native frame/global pairs per training backward are checkpointed; backward recomputation is not a free cached forward and is included in all worker wall time')
    modelcalls['full_native_query_chains']=sum(modelcalls[k] for k in ['preparation_initializer_native_queries','preparation_Native4_queries','formal_train_queries','formal_evaluation_queries','preflight_extra_queries','cost_queries'])
    modelcalls['official_TrackHead_calls_4iterations_each']=modelcalls['full_native_query_chains']
    hours=sum(w['elapsed_seconds'] for w in workers)/3600
    result=dict(complete=True,all_owned_GPU_children_actual_waited=True,workers=workers,worker_count=len(workers),GPU_worker_hours=hours,GPU_worker_hour_cap=12,worker_wall_includes_load_IO_failure_preflight_restore_training_evaluation_cost=True,failed_exit1_workers=sum(w['exit_code']==1 for w in workers),intentional_checkpoint_or_prepare_exit75_workers=sum(w['exit_code']==75 for w in workers),local_geometry_validation_conservative_reserved_seconds=30,DATA_final_bytes=data,DATA_peak_conservative_bytes=data+256*2**20,DATA_peak_bound_explanation='All immutable cases retained throughout; finals plus concurrent latest/atomic saves bounded by256MiB reserve (larger than three complete checkpoints and simultaneous case/prediction writes). No owned RAM filesystem checkpoint or prefix files.',project_final_bytes=project,project_peak_conservative_bytes=project+64*2**20,full_Adam_checkpoint_max_bytes=full_ckpt,prepared_case_max_bytes=case,GPU_peak_allocated_bytes=max([r.get('peak_GPU_allocated_bytes',0) for r in resources]+[cost['peak_GPU_allocated_bytes']]),host_peak_RSS_single_worker_bytes=max([r.get('peak_RSS_bytes',0) for r in resources]+[cost['peak_RSS_bytes']]),normal_worker_resource_records=resources,failed_worker_peak_RSS_GPU_unknown=True,new_downloads=0,new_sensor_or_RGB_data=0,new_or_reformed_GS_fields=0,call_counts=modelcalls,event_counts=dict(counts),paired_initial_and_sampler_identities=paired_initials,finals=finals,scientific_steps_repeated_or_discarded=0,formal_journal_unique_steps=12000)
    if (d.OUT/'cohost_observations.jsonl').exists():
        snapshots=[json.loads(l) for l in (d.OUT/'cohost_observations.jsonl').read_text().splitlines()];result['cohost_peak_sampled_whole_card_used_MiB']=max(g['used_MiB'] for x in snapshots for g in x['GPUs'] if sum(w['GPU']==g['index'] for w in x['active'])>=2);result['cohost_memory_samples_scope']='NVML whole-card samples; per-process exact allocated peaks recorded separately. Not an unobserved absolute resident-memory maximum.'
    assert hours<=12 and result['DATA_peak_conservative_bytes']<=12*2**30 and result['project_peak_conservative_bytes']<=2**30
    d.write(d.OUT/'runtime_budget_calls.json',result);return result

def main():
    start=time.monotonic();s=d.read(d.OUT/'scoring/summary.json');r=ledger();cost=d.read(d.OUT/'actual_cost.json');places=d.read(d.OUT/'scoring/place.json');counter=d.read(d.OUT/'scoring/all_pairwise_counterexamples.json');cov=s['coverage'];primary=s['comparisons_pp']['dev12/original'];view=d.read(d.OUT/'figure_review.json');assert view['actually_viewed']
    tables=[]
    for cohort in ('dev12','train32','train_original12','train_added20'):
        for qset in ('original','other','all8'):
            key=cohort+'/'+qset;tables.append(f'## {key}\n\n'+table(['Method','PCK@8 %, paired95%','Mean place EPE px','Mean pair-tail95 EPE px','Failed labelled points'],[[k,estimate(v['PCK8']),fmt(v['EPE_place_mean']),fmt(v['EPE95_place_mean']),v['failed_valid_points']] for k,v in s['absolute'][key].items()]))
            tables.append(table(['Contrast A−B','PCK pp [95%]','EPE px [95%]','EPE tail px [95%]'],[[k,estimate(v),estimate(s['comparisons_continuous_pixels'][key][k]['EPE']),estimate(s['comparisons_continuous_pixels'][key][k]['EPE95'])] for k,v in s['comparisons_pp'][key].items()]))
    tables.append('## Every place and seed, original / other queries\n\n'+table(['Place','Cohort','Origin','Method','Seed','Queries','PCK8 %','EPE px','EPE tail95 px','Defined query / pairs','Failed points'],[[v['id'],v['cohort'],v['training_origin'],v['kind'],v['seed'],v['query_set'],fmt(None if v['PCK8'] is None else v['PCK8']*100),fmt(v['EPE']),fmt(v['EPE95']),f'{v["defined_queries"]}/{v["defined_viewpairs"]}',v['failed_valid_points']] for v in places if v['query_set']!='all8']))
    (d.OUT/'report_tables.md').write_text('\n'.join(tables))
    contrasts=table(['Development original contrast','PCK difference pp [paired95%]'],[[k,estimate(primary[k])] for k in ('G-P','S-P','G-S','G-L','G-Native2','G-Native4','P-L','P-Native4','S-Native4','L-Native4')])
    absprimary=table(['Method','PCK8 % [paired95%]','EPE px','Tail95 px'],[[k,estimate(v['PCK8']),fmt(v['EPE_place_mean']),fmt(v['EPE95_place_mean'])] for k,v in s['absolute']['dev12/original'].items()])
    costtable=table(['Method','One-time source prefix/calibration s','Fresh full query s','Source amortized at8queries + fullquery s'],[[v['kind'],fmt(v['source_dependency_seconds'],4),fmt(v['fresh_full_query_seconds'],4),fmt(v['amortized'][-1]['seconds_per_query'],4)] for v in cost['records']])
    b=cost['state_tensor_bytes'];state_table=table(['One-anchor deployment dependency','Tensor / unique-storage MiB'],[['Native anchor KV, logical / backing storage',f'{b["single_anchor_native_KV"]/2**20:.3f} / {b["single_anchor_KV_unique_storage_including_strided_QKV_base"]/2**20:.3f}'],['Anchor Track DPT',fmt(b['single_anchor_Track_DPT']/2**20)],['Original shared GS + source geometry',fmt((b['original_GS_field']+b['original_source_GEOMETRY_fields'])/2**20)],['Mixed anchor KV backing + DPT + GS/geometry',fmt((b['single_anchor_KV_unique_storage_including_strided_QKV_base']+b['single_anchor_Track_DPT']+b['original_GS_field']+b['original_source_GEOMETRY_fields'])/2**20)],['Native4 KV backing + same anchor DPT',fmt((b['Native4_source_KV_unique_storage']+b['single_anchor_Track_DPT'])/2**20)]])
    worst=[x for x in counter if x['cohort']=='dev12' and x['query_set']=='original' and x['PCK_difference_pp']<0][:12]
    worsttable=table(['Contrast','Place','Seed','Anchor→query','ΔPCK pp','Labelled points'],[[v['A']+'−'+v['B'],v['id'],v['seed'],f'{v["anchor"]}→{v["query"]}',fmt(v['PCK_difference_pp']),v['valid_GT']] for v in worst])
    initials=[d.read(p) for p in (d.OUT/'fits').glob('*_initial.json')];parameters=initials[0]['parameters'];decisions=s['flags']
    report=f'''# native_track_memory_pe1：固定公共GS的原生点跟踪用途

2026-11-14，round026实现完整交付。唯一科学合同为[science_plan.md](../autonomous_research_loop/round_026/science_plan.md)，SHA256 `{d.SCIENCE_SHA}`。12 fits×1000更新、全部16896学习组预测及2816固定原生参考完成；固定末步，无挑选checkpoint、扩步或新增seed。所有数据是used development。下列是合同终点事实，后继科学取舍交回原Astra，不自行启动修补。

## 主要结果及界限

{contrasts}
{absprimary}
Native4以上的提升也出现在L（+17.625pp），而G−L区间跨零；这些对Native的共同收益不能归为GS内容或PE。P/S/G的位置差异、来源内容相对L与共同适配分别判断。

原规则机械标记：G PE增量={decisions['G']['PE_increment']}，G对L与Native4净用途={decisions['G']['net_over_L_and_Native4']}；S预定次要PE增量={decisions['S']['PE_increment']}，S净用途={decisions['S']['net_over_L_and_Native4']}。候选标记只表示本合同的开发筛选条件，不能替代尾部／成本判断、fresh确认或总体目标。均值≥1pp且95%下界>0适用于G−P；S相同门限仅属次要比较。共同内容／适配、PE增量和原生净用途分列，不由某一阳性替代另一项。

![主结果](figures/primary.png)

主量先对每view对有效点计算PCK@8，再按锚、query、地点等权。95%区间为5000次配对交叉重采样种子与地点（seed20261114），两组使用相同索引；点／view对不是独立地点。Native参考跨种子共享；只有三训练种子，有限地点与sensor标签范围限制泛化。连续EPE和每view对95%尾部同层级汇总；标签不存在记undefined，真实模型／初值失败记PCK0、EPE∞，绝不当成缺标签删除。

训练32、开发12，原四query与另四query、全部八query，P/S/G/L/Native2/Native4绝对值、全部配对区间、每地点／每种子原始结果见[完整数表](report_tables.md)与[scoring](scoring/summary.json)。[逐query](scoring/query.json)、[逐锚](scoring/source.json)、[逐view对及全部64误差](scoring/viewpairs.json)可直接核对；原训练12与新增20的身份保留在逐地点和逐view记录中，不把added或other当fresh。

## 真实接口、信息与学习范围

复用原44份alpha绑定2048×1024粗GS，不是task-writer C/J场。P/S/G共享原完整f、普通完整logΣ六通道、opacity、B、density、实际逐槽可见质量、公共25维几何和同一原生query初值；仅K位置四槽分别为强中心density、同迹球footprint、完整footprint，共同29→64→1024零输出MLP。完整Σ普通内容和中心集合结构始终给P，不称G独占信息。GS投影中心连续RoPE、原图像2DRoPE共同保留；位置只加GS K，V无位置项，背景K/V为零。

源锚保持完全冻结的因果原生前缀；仅query第一个已有global读取只读GS K/V，无GS query更新、额外层、第二reader或反馈。其后全部24对原生块、官方DPT特征和官方TrackHead四次迭代真实输出64点。训练共同1045→1024映射、上述位置MLP及query首frame/global的rank8 QKV/proj LoRA，每组{parameters:,}参数，其余原模型、camera/depth/TrackHead全部冻结。AdamW1e−4、wd.01、clip5，FP32参数/Adam、原生BF16、batch1；同seed完全相同初始化与1000配对sampler。四次query坐标输出等权SmoothL1 beta8，没有feature/pose/visibility目标。激活checkpoint的反传重计算与来源前缀cache miss均计成本。

L用query原1024 DINO patch各复制两份，2048槽、相同参数/P字段，几何来自预测相机的单位1深度射线平面，普通logΣ/opacity/visibility置零，无query真值或GS内容；它是计算／参数匹配的当前图像控制，不是同信息控制。Native2用锚→query、Native4用锚先再其余三source→query，均取同锚与query官方DPT特征交同一真实两帧tracker；并非返回输入点的伪基线。Native4共享四source RGB，但没有GS形成时来源sensor/pose特权，GS净用途严格限该特权范围。

每锚source-only已知来源相机、sensor/common_valid与原生来源预测深度作公制对齐。query相机/内参来自未适配相同原生锚→query，queryGT只构标签，不进入模型/初值/PE；没有人为噪声或真值回退。所有query由同一只读场与全局adapter起始；query坐标编码是临时值。原生anchor前缀仍是必要混合状态，本轮不是纯GS取代全部原生历史。输入索引和既有小清单身份在[input_identities.json](input_identities.json)，没有重哈希大旧权重或复制场。528份public-state依赖索引中，396份P/S/G引用原场，132份L只记录source-native前缀与全局adapter依赖；L的当前query字段是临时值，绝不冒称source-only内容。全部引用12个小最终adapter，不是528份独立场或逐scene可训练参数。

## 标签、覆盖、零支持与反例

原生0-index固定8×8网格27.5+56i；source原PIL最近邻448→210 sensor/common_valid，query原210 sensor与K_metric构造静态投影。双向深度容差max(.02m,.02depth)，两cycle≤1像素210；无feature/区域配对筛选。模型始终预测全部64点，conf/visibility不作后验筛选。标签只是sensor/pose条件下的可见对应，不能称独立物理追踪真值。

1408对共90112候选点，标签有效{cov['valid_GT']}；无标签view对{cov['zero_GT_pairs']}，GS实际渲染零支持{cov['zero_GS_raster_pairs']}，零可用attention键{cov['zero_GS_attention_pairs']}，来源标定或相机初值无效{cov['invalid_initializer_pairs']}（来源公制标定无效{cov['invalid_source_alignment_pairs']}，非有限query相机/内参{cov['nonfinite_predicted_camera_intrinsics_pairs']}）。每一阶段拒绝量、每点残差和无定义点保留在原case文件和[coverage](scoring/coverage.json)；[支持分层](scoring/support_strata.json)保留条件样本量、失败与同层级PCK，不能取代全体结论。旧家族的给定query相机覆盖结论不被本轮预测相机范围改写。

![覆盖](figures/coverage.png)

{worsttable}
全部成对反例与正差见[未截断清单](scoring/all_pairwise_counterexamples.json)。图例固定第一个开发例以及最大开发G−P／G−Native4 view对缺口；无图像排除或用视觉质量修改阈值。墙面／边界／遮挡及sensor投影歧义仍保留。

![真实坐标与反例](figures/tracking_cases.png)
![全部训练轨迹](figures/training.png)
实际图像查看与观察见[figure_review.json](figure_review.json)，不是仅导出文件。

## 有效实现检查及普通失败

[preflight](preflight.json)使用固定首个有标签训练pair：P/S/G/L真实损失反传，source/位置输出/两LoRA QKV梯度非零且有限；native/source前缀冻结。禁用记忆／零LoRA回放Native2四次坐标最大差0像素；空GS支持仍有背景且真实完整query/Track有限。一个P预检Adam更新丢弃，完整参数+Adam、RNG与1000sampler精确恢复；正式12000步另计，未重复或丢弃正式更新。独立NumPy/Torch终点复算最大绝对差{fmt(s['independent_NumPy_Torch_max_difference'],12)}，不是从已有score复制结论。

两次准备启动的3+3进程exit1：首先venv Ninja未在PATH，其后原CPU几何helper与CUDA默认eye冲突。前者保留已发生原生调用和3次gsplat kernel前失败；后者保留3次完成raster但未提交case。兼容venv/CUDA路径和原helper设备界面已修，无安装系统库／下载／GLIBC变更。随后为消除CPU几何瓶颈，在两个固定已完成训练case验证同公式CUDA端口：所有PE/中心/mask精确、普通字段最大8.88e−16；只中断自有准备进程并提交当前case后，恢复缺失包，未重做已提交科学预测。保守30worker秒包括该小检查。

首三fit用8-entry只读GPU anchor LRU，随机配对几乎每步重建来源前缀；后续九fit扩为128-entry覆盖32×4锚，源prefix数值与sampler不变，无中断首三fit。该缓存不写DATA／RAM文件；实际GPU峰值单列。所有历史版本及普通失败在[engineering_revisions](engineering_revisions)，失败、主动结束、加载、I/O和重建均计总worker墙钟；没有以超时重启或重复队列。

## 完整成本与预算

{costtable}
{state_table}
上述是固定单锚案例的依赖张量，不含全局native权重、全局adapter与query临时激活；KV底层QKV共享存储与逻辑视图分别记账，不将小GS文件大小冒称完整混合状态大小。

固定fit00/a0/q2、首种子一个实际成本例，无分布／硬件benchmark推论。固定执行顺序为Native2首次TrackHead、P/S/G/L、最后Native4；早晚调用的kernel/TrackHead首次初始化不独立可辨，后续链已承接前面初始化，不能由本例推断稳定态速度排序或独立cold-first-GS延迟。没有追加warmup或重复benchmark；所有首次调用真实worker时间完整入账。原已准备448 RGB张量重新经DINO编码、冻结锚前缀+DPT、camera/depth/source sensor标定；query已准备RGB重新编码、第一次原生query+camera初值、真实gsplat/全部字段、第二次带记忆原生query和官方DPT+Track全部计入。L同样计原生初值和单位ray字段；Native2/4只计对应所需真实source/query，不给其添加GS相机头。成本准入时本卡无其他CUDA进程；其他卡正式任务并行与现场见[cost_dispatch](cost_dispatch.json)，不将其解释为隔离服务器benchmark。共同阶段实测一次并明确计入各适用部署pipeline，不把共同缓存当免费；5次真实DINO、4source原生frame、6query/Track、1raster及所有cold load在[actual_cost.json](actual_cost.json)。表中source amortization只摊一次来源依赖，query部分含两次原生query，1/4/8范围均单列；模型/adapter/文件cold load另列。该例从既有448 RGB张量开始，raw图像解码/resize与sensor获取未重做、未计入该延迟；重新编码不意味着新数据或fresh科学确认。

GS原来源形成没有本轮重做。原24 warm prepared-source→field样例3.230/3.155秒不含raw解码、AnySplat来源预测或冷加载；新增20场来源预测／拟合／绑定及缓存worker合计10.554/36.986/47.976秒，范围不同且并非单scene端到端时间。sensor获取、旧预处理和冷predictor未知；[继承成本范围](cost_inherited.json)保留，不能称GS形成免费或用warm数替代全部来源形成。sourceprefix+GS混合状态的体积与查询开销不证明压缩原生历史已成功。

总GPU worker **{r['GPU_worker_hours']:.6f}h /12h**，包括全部失败、恢复、预检、加载、I/O、训练、评价与成本；新DATA最终{r['DATA_final_bytes']/2**30:.4f}GiB、原子并发保守峰上界{r['DATA_peak_conservative_bytes']/2**30:.4f}GiB /12GiB；项目含脚本最终{r['project_final_bytes']/2**30:.4f}GiB、保守峰{r['project_peak_conservative_bytes']/2**30:.4f}GiB /1GiB。所有GPU子进程实际wait返回，共{r['worker_count']}个worker，exit1 {r['failed_exit1_workers']}、主动exit75 {r['intentional_checkpoint_or_prepare_exit75_workers']}；完整Adam最大{r['full_Adam_checkpoint_max_bytes']/2**20:.2f}MiB，永久12final均保存。没有自有RAM文件系统checkpoint或隐藏临时副本；失败进程峰RSS/GPU未采全，明确unknown，不伪造零。

[runtime_budget_calls.json](runtime_budget_calls.json)列每worker时段/卡/PID/实际退出、精确逻辑fullquery与官方tracker调用、反传checkpoint重计算、峰RAM/GPU与保守磁盘暂存。常规CPU评分/作图及最终组织独立记录；未占GPU的管理／编辑时间不充作GPU训练时间。实时保护0–3/5等外部任务，仅安全空卡获准，三卡独立唯一fit；无默认两卡上限、无扩大预算换利用率。

## O/C与交回边界

O2/O3原公共粗GS只读，O4多真实query、O5实际投影raster，O6同信息PE比较与原生用途分别实测。C1/C2不重建；C3一个既有global只读接口；C4完整共同Σ普通内容+强中心、只比较位置分配；C5对应任务及未知query相机；C6有限共同学习。sensor/来源pose特权、静态四源、2048×1024、混合native前缀、旧数据／有限点标签均为本次实施选择，不是用户核心要求。历史内容区域K正证据、R替换未决及旧depth/pose/地址等关闭保持，不据本轮阴性推出公共GS或所有PE无用，也不据阳性宣称总体成功。所有后继判断集中交主→原Astra。
'''
    unsup=d.read(d.OUT/'unsupervised_training_batches.json')['summary'];report=report.replace('正式12000步另计，未重复或丢弃正式更新。',f'正式12000步另计，未重复或丢弃正式更新。其中{unsup["zero_GT_batches"]}个固定batch无有效GT，使用连接计算图的零监督项、梯度范数0，保持AdamW动量/weight decay与既定step，不重采样；不把它们解释为真实零跟踪误差。有效GT但loss0的batch有{unsup["zero_loss_with_positive_GT"]}个，完整记录见[无监督batch清单](unsupervised_training_batches.json)。')
    cont=s['comparisons_continuous_pixels']['dev12/original']['G-P'];report=report.replace('主量先对每view对有效点计算',f'主比较G−P连续EPE为{estimate(cont["EPE"])}像素，view对95%尾部差{estimate(cont["EPE95"])}像素，负值有利G；它们与PCK净用途分列。[点级合并尾部分布](scoring/pooled_point_tails.json)另保留各组/种子50/90/95/99百分位与最大误差，属于描述而不是主地点等权估计。\n\n主量先对每view对有效点计算')
    if (d.OUT/'engineering_revisions/external_yield005.json').exists():
        report=report.replace('首三fit用8-entry', 'GPU4在P第二种子评价期间出现外部训练；仅SIGTERM我方评价，当前case提交后actualwait75，695/1408已提交结果与1000步永久final/完整Adam保留。迁至7号后外部6/7加载任务到达，再于876对后礼让；外部退出后同管理器实时准入并从876继续，只恢复缺失评价。没有重做正式更新、重复已提交预测或操作外部PID，全部cold reload与礼让前时间计账。\n\n首三fit用8-entry')
    if (d.OUT/'engineering_revisions/external_yield008.json').exists():
        report=report.replace('首三fit用8-entry','最终G/L第三种子评价又遇外部6/7任务，16620对提交后仅我方actualwait75，余G76/L200。旧manager44052在04:32:25正常停止路径exit75，来源未知，调用方未发送停止信号；所有旧child均actualwait、资源未达限，不把未知原因改写为科学失败或主动刷新。随后同锁唯一恢复manager只从永久final1000补缺失评价。外部显存约9444MiB连续数分钟稳定，以准入free≥36GiB／卡util≤35%、我方allocator8GiB、运行free≥24GiB和新外部PID／持续高利用率礼让，保守共享少量评价；未来外部峰值未知，不冒称认证。真实source/adapter/采样/坐标计算不变，没有新fit、正式更新或benchmark；成本保持独占空卡。全部新增加载、共享争用和恢复worker墙钟入账。\n\n首三fit用8-entry')
    coverage_table=table(['Cohort','Queries','Defined / all view pairs','Labelled / candidate points','Zero-GT pairs'],[[x['cohort'],x['queries'],f'{x["defined_pairs"]}/{x["pairs"]}',f'{x["valid_GT_points"]}/{x["candidate_points"]}',x['zero_GT_pairs']] for x in d.read(d.OUT/'scoring/label_coverage_cohorts.json')]);report=report.replace('1408对共90112候选点',coverage_table+'\n开发fidelity_03的原四query仅1个锚/query对有标签（30点），另四也仅1对（41点）；仍保留其地点权重，不把点数当独立地点。56个来源公制标定无效对及38个raster零支持对均无GT点，因此该条件的准确性端点未定义，不由它们宣称零误差或失败用途。\n\n1408对共90112候选点')
    report=report.replace('完整Σ普通内容和中心集合结构始终给P，不称G独占信息。','完整Σ普通内容和中心集合结构始终给P，不称G独占信息。G/P是匹配学习链的总差异，最终adapter分别学习；原场固定，不能改称固定同一接收器的直接干预或准确物理Σ的证明。')
    if (d.OUT/'cohost_admissions.json').exists():
        admissions=d.read(d.OUT/'cohost_admissions.json');report=report.replace('首三fit用8-entry',f'外部任务占用其余卡后，根据实测29408MiB训练占用与约5704MiB评价占用，在安全卡实施每卡一训一评，共{len(admissions)}次既定pending fit的同卡准入。为加入动态准入接口，仅在G/L第二种子均已final1000后刷新唯一manager，旧会话实际exit75且全部子进程actualwait，所有评价marker保留。P第三种子正式首100步144.69秒（此前solo131–136秒），据此及剩余原工作投影控制窗口；没有额外benchmark、fit或步数。新外部PID到达只礼让我方；[准入](cohost_admissions.json)、[实测预算审查](engineering_revisions/cohost006_throughput_review.json)和[显存现场](cohost_observations.jsonl)保留，全部加载/恢复/争用时间都进GPUworker账。\n\n首三fit用8-entry')
    (d.OUT/'report.md').write_text(report)
    update_living_docs(s)
    d.write(d.OUT/'CPU_delivery_runtime.json',dict(wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_GPU_calls=0,completed=True))
    # Final handoff is refreshed after living docs and process-terminal inspection.
    manifest=[]
    for p in sorted(d.OUT.rglob('*')):
        if p.is_file() and p.name not in ('artifact_manifest.json','implementation_handoff.json','pool.lock'):manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha(p)))
    sources=[*d.ROOT.glob('experiments/native_track_memory*.py'),d.ROOT/'models/native_track_memory.py'];living=[d.ROOT/'docs'/n for n in ('research_current.md','research_outline_alignment.md','gaussian_memory_evidence_scope.md')]
    d.write(d.OUT/'artifact_manifest.json',dict(complete=True,project_artifacts=manifest,implementation_sources=[dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha(p)) for p in sources],living_document_updates=[dict(path=str(p),sha256=d.sha(p)) for p in living],new_final_weights=[dict(path=x['final'],sha256=x['sha256']) for x in r['finals']],predictions_index=str(d.OUT/'completed'),original_fields_untouched=True))
    handoff=dict(round=26,experiment='native_track_memory_pe1',status='completed_ready_for_scientific_assessment',authoritative_science_plan=dict(path=str(d.SCIENCE),sha256=d.SCIENCE_SHA),report=dict(path=str(d.OUT/'report.md'),sha256=d.sha(d.OUT/'report.md')),artifacts=str(d.OUT/'artifact_manifest.json'),summary=str(d.OUT/'scoring/summary.json'),complete_tables=str(d.OUT/'report_tables.md'),runtime_budget=str(d.OUT/'runtime_budget_calls.json'),actual_cost=str(d.OUT/'actual_cost.json'),figure_review=str(d.OUT/'figure_review.json'),actual=dict(fits=12,formal_updates=12000,learned_predictions=16896,native_references=2816,GPU_worker_hours=r['GPU_worker_hours'],all_GPU_children_actual_waited=True),primary_G_minus_P=primary['G-P'],secondary_S_minus_P=primary['S-P'],development_flags=decisions,unresolved_material_issues=[],limits=['All data used development; no fresh confirmation','Sensor/pose-derived point labels and limited coverage','Native4 lacks GS source sensor/pose privileges','Mixed native anchor prefix + GS is not pure GS history replacement','One fixed prepared-RGB cost example; fixed-order first-use overhead is not separately identified, no cold-first-GS or steady-state latency claim','Failed worker RSS/GPU peak unknown'],roles=dict(task='/root/research_implementation_sol_recovery',original_request='gpt-6.1-sol/high',backend_effective_unknown=True,Fast_requested_or_verified=False,main_goal_state_events_modified=False),no_successor_started=True,scientific_disposition_reserved_for_original_Astra=True,actual_CPU_and_GPU_terminal_confirmation='GPU children all actual waited; final CPU delivery process exit checked by caller; no task self-dispatch',completed_utc=d.now())
    d.write(d.OUT/'implementation_handoff.json',handoff);print(json.dumps(dict(report=handoff['report'],primary=primary['G-P'],GPU_worker_hours=r['GPU_worker_hours'])),flush=True)
if __name__=='__main__':main()
